//! One E-core coordinator polls GPU doorbells and dispatches the existing P-core pool.
use hayate_core::{Error, Result};

pub(super) const READY: usize = 0;
pub(super) const DONE: usize = 64;
/// Rung after the deferred experts of a device-decode layer (`--defer-below`).
pub(super) const DONE2: usize = 96;
pub(super) const IDS: usize = 128;
pub(super) const INPUT: usize = IDS + 4 * 8 * 4;
pub(super) const OUTPUT: usize = INPUT + 4 * 8192;
pub(super) const SPARSE_COUNTS: usize = OUTPUT + 4 * 9 * 8192;
pub(super) const BYTES: usize = SPARSE_COUNTS + 64;

fn wait_ready(
    flag: &std::sync::atomic::AtomicU32,
    sequence: u32,
    stop: &std::sync::atomic::AtomicBool,
) -> bool {
    use std::sync::atomic::Ordering;
    while flag.load(Ordering::Acquire) != sequence {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        std::hint::spin_loop();
    }
    true
}

/// Expert ids and the masks of assignments the GPU dropped (bit 31 of an id:
/// `--skip-cpu-below`, not computed) and deferred (bit 30: `--defer-below`,
/// computed after DONE and published with DONE2).
fn routes(raw: &[u8], rows: usize) -> Result<(Vec<u32>, u32, u32)> {
    if !(1..=4).contains(&rows) || raw.len() != rows * 8 * 4 {
        return Err(Error::InvalidShape("GLM mailbox routes".into()));
    }
    let (mut dropped, mut late) = (0, 0);
    let ids: Vec<_> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(a, b)| {
            let id = u32::from_le_bytes(*b);
            dropped |= (id >> 31) << a;
            late |= (id >> 30 & 1) << a;
            id & 0x3fff_ffff
        })
        .collect();
    super::prefill::groups(&ids)?;
    Ok((ids, dropped, late))
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::super::{
        cpu_experts::{CpuExperts, Miss},
        experts::TransferStats,
        transfer::HostImage,
    };
    use super::*;
    use hayate_kernels::runtime::{Device, MappedHost, Stream};
    use std::{
        ffi::c_void,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, AtomicU32, Ordering},
            mpsc,
        },
        thread::JoinHandle,
        time::Instant,
    };

    #[derive(Debug)]
    pub(in crate::glm53) struct Snapshot {
        pub host: Vec<Option<Arc<HostImage>>>,
        /// GPU image address per expert (0: not on the GPU). Slots change only at
        /// token boundaries, after every job of the token completed.
        pub gpu: Vec<u64>,
        /// [gate/up, down] CPU formats per expert of a layer with LOWBIT experts.
        pub formats: Option<Arc<Vec<[hayate_kernels::glm53::cpu::Format; 2]>>>,
    }
    #[derive(Debug)]
    pub(in crate::glm53) struct Mailbox {
        memory: MappedHost,
    }
    impl Mailbox {
        pub fn new(device: &Device) -> Result<Arc<Self>> {
            Ok(Arc::new(Self {
                memory: device.alloc_mapped(BYTES)?,
            }))
        }
        pub fn device(&self, at: usize) -> *mut c_void {
            unsafe { self.memory.device_ptr().cast::<u8>().add(at).cast() }
        }
        /// # Safety
        /// The sampling read must have retired the layer's sparse publication.
        pub unsafe fn sparse_counts(&self, rows: usize) -> (usize, usize) {
            let counts = unsafe { self.read(SPARSE_COUNTS, 20) };
            let counts: Vec<_> = counts
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b) as usize)
                .collect();
            (counts[1..1 + rows].iter().sum(), counts[0])
        }
        fn flag(&self, at: usize) -> &AtomicU32 {
            unsafe { &*self.memory.host_ptr().cast::<u8>().add(at).cast() }
        }
        // The ready acquire grants exclusive CPU access; done releases it to GPU.
        unsafe fn read(&self, at: usize, bytes: usize) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(self.memory.host_ptr().cast::<u8>().add(at), bytes)
            }
        }
    }
    #[derive(Debug)]
    pub(in crate::glm53) struct LayerAsync {
        pub mailbox: Arc<Mailbox>,
        pub table: MappedHost,
        pub snapshot: Option<Arc<Snapshot>>,
        generation: u32,
    }
    impl LayerAsync {
        pub fn new(device: &Device) -> Result<Self> {
            Ok(Self {
                mailbox: Mailbox::new(device)?,
                table: device.alloc_mapped(288 * 8)?,
                snapshot: None,
                generation: 0,
            })
        }
        pub fn refresh(&mut self, snapshot: Snapshot) {
            // Only called after the previous sampling read retired every layer.
            for (id, &address) in snapshot.gpu.iter().enumerate() {
                unsafe { self.table.host_ptr().cast::<u64>().add(id).write(address) };
            }
            self.snapshot = Some(Arc::new(snapshot));
        }
        pub fn submit(
            &mut self,
            worker: &Worker,
            layer: usize,
            rows: usize,
            format: (bool, bool),
            hadamard: u32,
        ) -> Result<u32> {
            let sequence = self.generation.wrapping_add(1);
            self.submit_with(worker, layer, rows, format, hadamard, sequence)
        }
        /// Job for a doorbell rung by the GPU with `sequence` (device decode:
        /// one sequence for every layer of a token, from the control block).
        pub fn submit_with(
            &mut self,
            worker: &Worker,
            layer: usize,
            rows: usize,
            format: (bool, bool),
            hadamard: u32,
            sequence: u32,
        ) -> Result<u32> {
            self.generation = sequence;
            worker
                .send
                .as_ref()
                .unwrap()
                .send(Job {
                    mailbox: self.mailbox.clone(),
                    snapshot: self
                        .snapshot
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidPlan("GLM async placement not prepared".into())
                        })?
                        .clone(),
                    layer,
                    rows,
                    sequence,
                    format,
                    hadamard,
                })
                .map_err(|_| Error::Weights("GLM doorbell worker stopped".into()))?;
            Ok(sequence)
        }
        /// Host-side READY for a job whose GPU doorbell never rang (failed
        /// launch), so the worker completes it instead of waiting forever.
        pub fn ring(&self, sequence: u32) {
            let ready = self.mailbox.flag(READY);
            if ready.load(Ordering::Acquire) != sequence {
                ready.store(sequence, Ordering::Release);
            }
        }
        pub fn signal(&self, stream: &Stream, sequence: u32) -> Result<()> {
            unsafe { stream.write_value32(self.mailbox.device(READY), sequence) }
        }
        pub fn wait(&self, stream: &Stream, sequence: u32) -> Result<()> {
            unsafe { stream.wait_value32(self.mailbox.device(DONE), sequence) }
        }
    }
    /// HYT files, per-expert entry lists (statistics order) and image bytes.
    pub(in crate::glm53) type Disk = (
        Arc<Vec<hayate_core::hyt::HytFile>>,
        Arc<Vec<Vec<usize>>>,
        usize,
    );
    #[derive(Debug)]
    struct Job {
        mailbox: Arc<Mailbox>,
        snapshot: Arc<Snapshot>,
        layer: usize,
        rows: usize,
        sequence: u32,
        format: (bool, bool),
        hadamard: u32,
    }
    #[derive(Debug)]
    pub(in crate::glm53) struct Completed {
        pub layer: usize,
        pub ids: Vec<u32>,
        pub stats: TransferStats,
        pub result: Result<()>,
    }
    #[derive(Debug)]
    pub(in crate::glm53) struct Worker {
        send: Option<mpsc::Sender<Job>>,
        pub done: mpsc::Receiver<Completed>,
        stop: Arc<AtomicBool>,
        thread: Option<JoinHandle<()>>,
    }
    impl Worker {
        /// `disk` reads misses that have neither a GPU slot nor a RAM copy
        /// (elastic shrink past the RAM limit) straight from the HYT files.
        pub fn new(cpu: Option<Arc<Mutex<CpuExperts>>>, disk: Option<Disk>) -> Result<Self> {
            let (send, receive) = mpsc::channel::<Job>();
            let (completed, done) = mpsc::channel();
            let (startup, started) = mpsc::sync_channel(1);
            let stop = Arc::new(AtomicBool::new(false));
            let cancel = stop.clone();
            let thread = std::thread::Builder::new()
                .name("glm-doorbell-22".into())
                .spawn(move || {
                    let affinity = hayate_kernels::cpu::pin_thread(22);
                    let ok = affinity.is_ok();
                    let _ = startup.send(affinity);
                    if !ok {
                        return;
                    }
                    while let Ok(job) = receive.recv() {
                        if !wait_ready(job.mailbox.flag(READY), job.sequence, &cancel) {
                            return;
                        }
                        let mut ids = Vec::new();
                        let mut stats = TransferStats::default();
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                            || -> Result<()> {
                                let (dropped, late);
                                (ids, dropped, late) = routes(
                                    unsafe { job.mailbox.read(IDS, job.rows * 8 * 4) },
                                    job.rows,
                                )?;
                                let groups = crate::glm53::prefill::groups(&ids)?;
                                let kept = |a: &Vec<u32>| a.iter().any(|&a| dropped >> a & 1 == 0);
                                let start = Instant::now();
                                let mut cpu = cpu.as_ref().map(|c| c.lock().unwrap());
                                let before = cpu.as_ref().map(|c| c.timing()).unwrap_or_default();
                                // Gather every miss of the layer for one batched CPU call.
                                let mut owned = Vec::new();
                                for (expert, assignments) in groups.iter().enumerate() {
                                    if !kept(assignments)
                                        || job.snapshot.gpu[expert] != 0
                                        || job.snapshot.host[expert].is_some()
                                    {
                                        continue;
                                    }
                                    let (files, entries, bytes) =
                                        disk.as_ref().ok_or_else(|| {
                                            Error::Weights("GLM async miss has no RAM image".into())
                                        })?;
                                    let mut buffer = vec![0u8; bytes + 256];
                                    let at = buffer.as_ptr().align_offset(256);
                                    super::super::experts::read_image(
                                        &files[job.layer],
                                        &entries[(job.layer - 3) * 288 + expert],
                                        &mut buffer[at..at + bytes],
                                    )?;
                                    owned.push((expert, buffer, at));
                                }
                                let mut work: Vec<(&[u8], &Vec<u32>, bool, usize)> = Vec::new();
                                for (expert, assignments) in groups.iter().enumerate() {
                                    if !kept(assignments) || job.snapshot.gpu[expert] != 0 {
                                        continue;
                                    }
                                    let image = match &job.snapshot.host[expert] {
                                        Some(image) => image.as_slice(),
                                        None => {
                                            let (_, buffer, at) =
                                                owned.iter().find(|o| o.0 == expert).unwrap();
                                            let bytes = disk.as_ref().unwrap().2;
                                            &buffer[*at..at + bytes]
                                        }
                                    };
                                    let deferred = assignments.iter().all(|&a| late >> a & 1 == 1);
                                    work.push((image, assignments, deferred, expert));
                                }
                                // This layer's experts first (DONE), then the deferred ones (DONE2).
                                for deferred in [false, true] {
                                    let work: Vec<_> = work
                                        .iter()
                                        .filter(|w| w.2 == deferred)
                                        .map(|w| (w.0, w.1, w.3))
                                        .collect();
                                    if deferred {
                                        job.mailbox
                                            .flag(DONE)
                                            .store(job.sequence, Ordering::Release);
                                    }
                                    if work.is_empty() {
                                        continue;
                                    }
                                    let shared = job.rows == 1;
                                    let inputs: Vec<Vec<u8>> = if shared {
                                        Vec::new()
                                    } else {
                                        work.iter()
                                            .map(|(_, assignments, _)| {
                                                let mut input =
                                                    Vec::with_capacity(assignments.len() * 8192);
                                                for &a in assignments.iter() {
                                                    input.extend_from_slice(unsafe {
                                                        job.mailbox.read(
                                                            INPUT + a as usize / 8 * 8192,
                                                            8192,
                                                        )
                                                    });
                                                }
                                                input
                                            })
                                            .collect()
                                    };
                                    let row0 = unsafe { job.mailbox.read(INPUT, 8192) };
                                    let mut outputs: Vec<Vec<u8>> = work
                                        .iter()
                                        .map(|(_, a, _)| vec![0; a.len() * 8192])
                                        .collect();
                                    let mut misses: Vec<_> = work
                                        .iter()
                                        .zip(&mut outputs)
                                        .enumerate()
                                        .map(|(i, ((image, _, expert), output))| Miss {
                                            image,
                                            input: if shared { row0 } else { &inputs[i] },
                                            output,
                                            expert: Some((job.layer, *expert)),
                                        })
                                        .collect();
                                    let cpu = cpu.as_mut().ok_or_else(|| {
                                        Error::InvalidPlan("GLM async CPU pool missing".into())
                                    })?;
                                    if let Some(formats) = &job.snapshot.formats {
                                        let formats: Vec<_> =
                                            work.iter().map(|w| formats[w.2]).collect();
                                        cpu.forward_layer_formats(
                                            &mut misses,
                                            &formats,
                                            job.hadamard,
                                        )?;
                                    } else {
                                        cpu.forward_layer(
                                            &mut misses,
                                            job.format.0,
                                            job.format.1,
                                            job.hadamard,
                                        )?;
                                    }
                                    for ((image, assignments, _), output) in
                                        work.iter().zip(&outputs)
                                    {
                                        for (i, &a) in assignments.iter().enumerate() {
                                            let offset = OUTPUT
                                                + (a as usize / 8 * 9 + a as usize % 8) * 8192;
                                            unsafe {
                                                std::ptr::copy_nonoverlapping(
                                                    output.as_ptr().add(i * 8192),
                                                    job.mailbox
                                                        .memory
                                                        .host_ptr()
                                                        .cast::<u8>()
                                                        .add(offset),
                                                    8192,
                                                );
                                            }
                                        }
                                        stats.cpu_experts += 1;
                                        stats.cpu_bytes += image.len() as u64;
                                    }
                                }
                                if stats.cpu_experts != 0 {
                                    let after = cpu.as_ref().unwrap().timing();
                                    stats.cpu_ms = start.elapsed().as_secs_f64() * 1000.;
                                    stats.cpu_worker_ms = after.0 - before.0;
                                    stats.cpu_dispatch_ms = after.1 - before.1;
                                }
                                Ok(())
                            },
                        ))
                        .unwrap_or_else(|_| {
                            Err(Error::Weights("GLM doorbell task panicked".into()))
                        });
                        // Release every GPU wait even on a recoverable CPU error. The
                        // sampling boundary observes and returns that error before use.
                        if result.is_err() {
                            unsafe {
                                job.mailbox
                                    .memory
                                    .host_ptr()
                                    .cast::<u8>()
                                    .add(OUTPUT)
                                    .write_bytes(0, BYTES - OUTPUT);
                            }
                        }
                        let sequence = job.sequence;
                        let mailbox = job.mailbox.clone();
                        let layer = job.layer;
                        drop(job); // No borrowed cache images survive completion publication.
                        let _ = completed.send(Completed {
                            layer,
                            ids,
                            stats,
                            result,
                        });
                        mailbox.flag(DONE).store(sequence, Ordering::Release);
                        mailbox.flag(DONE2).store(sequence, Ordering::Release);
                    }
                })?;
            let mut worker = Self {
                send: Some(send),
                done,
                stop,
                thread: Some(thread),
            };
            if let Err(e) = started
                .recv()
                .map_err(|_| Error::Weights("GLM doorbell startup".into()))?
            {
                worker.shutdown();
                return Err(e);
            }
            Ok(worker)
        }
        fn shutdown(&mut self) {
            self.stop.store(true, Ordering::Release);
            self.send.take();
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }
    impl Drop for Worker {
        fn drop(&mut self) {
            self.shutdown();
        }
    }
}
#[cfg(feature = "cuda")]
pub(super) use gpu::*;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn acquire_release_payload_reuse_and_cancel() {
        use std::{
            cell::UnsafeCell,
            sync::{
                Arc,
                atomic::{AtomicBool, AtomicU32, Ordering},
            },
        };
        struct Shared {
            ready: AtomicU32,
            done: AtomicU32,
            payload: UnsafeCell<[u64; 2]>,
        }
        // The two flags grant alternating exclusive ownership of the payload.
        unsafe impl Sync for Shared {}
        let state = Arc::new(Shared {
            ready: AtomicU32::new(0),
            done: AtomicU32::new(0),
            payload: UnsafeCell::new([0; 2]),
        });
        let shared = state.clone();
        let worker = std::thread::spawn(move || {
            let stop = AtomicBool::new(false);
            for i in 1..=1024 {
                assert!(wait_ready(&shared.ready, i, &stop));
                unsafe {
                    let data = &mut *shared.payload.get();
                    assert_eq!(data[0], i as u64);
                    data[1] = data[0] * 17;
                }
                shared.done.store(i, Ordering::Release);
            }
        });
        let stop = AtomicBool::new(false);
        for i in 1..=1024 {
            unsafe {
                (*state.payload.get())[0] = i as u64;
            }
            state.ready.store(i, Ordering::Release);
            assert!(wait_ready(&state.done, i, &stop));
            assert_eq!(unsafe { (*state.payload.get())[1] }, i as u64 * 17);
        }
        worker.join().unwrap();
        assert!(!wait_ready(&state.ready, 1025, &AtomicBool::new(true)));
    }
    #[test]
    fn mailbox_layout_and_routes() {
        assert_eq!(READY % 64, 0);
        assert_eq!(DONE % 64, 0);
        const { assert!(DONE < DONE2 && DONE2 + 4 <= IDS) };
        const {
            assert!(
                IDS + 4 * 8 * 4 <= INPUT
                    && INPUT + 4 * 8192 <= OUTPUT
                    && OUTPUT + 4 * 9 * 8192 <= SPARSE_COUNTS
                    && SPARSE_COUNTS + 20 <= BYTES
            );
        }
        let ids: Vec<_> = (0u32..32).flat_map(u32::to_le_bytes).collect();
        assert_eq!(
            routes(&ids, 4).unwrap(),
            ((0..32).collect::<Vec<_>>(), 0, 0)
        );
        let flagged: Vec<_> = (0u32..8)
            .flat_map(|i| (i | u32::from(i == 3) << 31 | u32::from(i == 5) << 30).to_le_bytes())
            .collect();
        assert_eq!(
            routes(&flagged, 1).unwrap(),
            ((0..8).collect::<Vec<_>>(), 1 << 3, 1 << 5)
        );
        assert!(routes(&ids, 0).is_err());
        let duplicate = 0u32.to_le_bytes().repeat(8);
        assert!(routes(&duplicate, 1).is_err());
        assert!(routes(&288u32.to_le_bytes().repeat(8), 1).is_err());
    }
}
