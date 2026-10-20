use hayate_core::{Error, QuantSpec, Result, hyt::HytFile};
use std::{fs::File, os::unix::fs::FileExt};

const ALIGN: usize = 4096;

// Read full aligned tensors (multi-MB), including the last short file block.
// The caller scatters payloads into the 256-byte-aligned runtime image.
fn read_aligned(file: &File, offset: u64, needed: usize, dst: &mut [u8]) -> Result<()> {
    if !offset.is_multiple_of(ALIGN as u64)
        || !dst.len().is_multiple_of(ALIGN)
        || dst.as_ptr().align_offset(ALIGN) != 0
        || needed > dst.len()
    {
        return Err(Error::InvalidShape("GLM direct read alignment".into()));
    }
    let mut filled = 0;
    while filled < needed {
        let n = match file.read_at(&mut dst[filled..], offset + filled as u64) {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        filled += n;
        if filled < needed && (n == 0 || !filled.is_multiple_of(ALIGN)) {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }
    }
    Ok(())
}

pub fn read_direct_image(
    file: &File,
    hyt: &HytFile,
    entries: &[usize],
    direct: &mut [u8],
    dst: &mut [u8],
) -> Result<u64> {
    read_direct_image_timed(file, hyt, entries, direct, dst, &mut ReadTimes::default())
}

#[derive(Debug, Default)]
struct ReadTimes {
    io_ms: f64,
    pack_ms: f64,
}

fn read_direct_image_timed(
    file: &File,
    hyt: &HytFile,
    entries: &[usize],
    direct: &mut [u8],
    dst: &mut [u8],
    times: &mut ReadTimes,
) -> Result<u64> {
    let start = direct.as_ptr().align_offset(ALIGN);
    let mut offset = 0;
    let mut bytes = 0;
    for &index in entries {
        let e = &hyt.entries()[index];
        let needed = e.bytes as usize;
        let size = needed.next_multiple_of(ALIGN);
        if start + size > direct.len() || offset + needed.next_multiple_of(256) > dst.len() {
            return Err(Error::InvalidShape("GLM direct staging length".into()));
        }
        let raw = &mut direct[start..start + size];
        let timer = std::time::Instant::now();
        read_aligned(file, hyt.header().data_offset + e.offset, needed, raw)?;
        times.io_ms += timer.elapsed().as_secs_f64() * 1000.;
        let timer = std::time::Instant::now();
        dst[offset..offset + needed].copy_from_slice(&raw[..needed]);
        if e.quant == QuantSpec::Nvfp4 {
            let dims = e.shape.dims();
            super::validate_nvfp4(
                &dst[offset..offset + needed],
                dims[0] as usize,
                dims[1] as usize,
            )?;
        } else if let QuantSpec::Lowbit { grid, group } = e.quant {
            let dims = e.shape.dims();
            super::validate_lowbit(
                &dst[offset..offset + needed],
                (dims[0] * dims[1]) as usize,
                grid,
                group,
            )?;
        } else if let QuantSpec::Exl3 { .. } = e.quant {
            let dims = e.shape.dims();
            super::validate_exl3(
                &dst[offset..offset + needed],
                dims[0] as usize,
                dims[1] as usize,
            )?;
        } else {
            super::validate_raw(&dst[offset..offset + needed], e.quant)?;
        }
        let end = (offset + needed).next_multiple_of(256);
        dst[offset + needed..end].fill(0);
        offset = end;
        bytes += needed as u64;
        times.pack_ms += timer.elapsed().as_secs_f64() * 1000.;
    }
    dst[offset..].fill(0);
    Ok(bytes)
}

#[cfg(any(test, feature = "cuda"))]
pub(super) fn worker_status(failure: &std::sync::atomic::AtomicI32) -> Result<()> {
    match failure.load(std::sync::atomic::Ordering::Acquire) {
        0 => Ok(()),
        -1 => Err(Error::Weights("GLM transfer worker failed".into())),
        code => Err(Error::CudaAsync {
            code,
            sync_point: "GLM transfer worker",
        }),
    }
}

#[cfg(any(test, feature = "cuda"))]
fn fail_workers(failure: &std::sync::atomic::AtomicI32, error: &Error) {
    use std::sync::atomic::Ordering;
    if let Error::CudaAsync { code, .. } = error {
        failure.store(*code, Ordering::Release);
    } else {
        let _ = failure.compare_exchange(0, -1, Ordering::AcqRel, Ordering::Acquire);
    }
}

#[cfg(feature = "cuda")]
mod workers {
    use super::*;
    use hayate_kernels::runtime::{Device, DeviceBuffer, PinnedHost};
    use std::sync::{Arc, mpsc};
    use std::thread::JoinHandle;

    pub(crate) const DEFAULT_READERS: usize = 4;
    pub(crate) const DIRECT_BYTES: usize = 2048 * 4096 + ALIGN;

    #[derive(Debug)]
    pub struct ReadJob {
        pub index: usize,
        pub scratch: usize,
        pub gpu: DeviceBuffer,
    }
    #[derive(Debug)]
    pub struct ReadDone {
        pub scratch: usize,
        pub gpu: DeviceBuffer,
        pub read_ms: f64,
        pub io_ms: f64,
        pub pack_ms: f64,
        pub upload_ms: f64,
        pub bytes: u64,
        pub result: Result<()>,
    }
    #[derive(Debug)]
    pub struct Readers {
        jobs: Vec<mpsc::Sender<ReadJob>>,
        pub done: mpsc::Receiver<ReadDone>,
        threads: Vec<JoinHandle<()>>,
        next: usize,
    }

    impl Readers {
        pub fn new(
            device: &Device,
            files: Arc<Vec<HytFile>>,
            direct: Arc<Vec<File>>,
            entries: Arc<Vec<Vec<usize>>>,
            image_bytes: usize,
            failure: Arc<std::sync::atomic::AtomicI32>,
            reader_count: usize,
        ) -> Result<Self> {
            let (completed, done) = mpsc::channel();
            let mut result = Self {
                jobs: Vec::new(),
                done,
                threads: Vec::new(),
                next: 0,
            };
            for worker in 0..reader_count {
                let mut image = device.alloc_pinned(image_bytes)?;
                let mut staging = device.alloc_pinned(DIRECT_BYTES + ALIGN - 1)?;
                let stream = device.stream()?;
                let files = files.clone();
                let direct = direct.clone();
                let entries = entries.clone();
                let completed = completed.clone();
                let failure = failure.clone();
                let (send, receive) = mpsc::channel::<ReadJob>();
                result.jobs.push(send);
                result.threads.push(std::thread::spawn(move || {
                    let affinity = hayate_kernels::cpu::pin_thread(
                        hayate_core::machine::MAIN.cpu.service[worker % 8],
                    );
                    while let Ok(mut job) = receive.recv() {
                        let timer = std::time::Instant::now();
                        let mut bytes = 0;
                        let mut times = ReadTimes::default();
                        let mut upload_ms = 0.;
                        let mut upload_start = None;
                        let result = worker_status(&failure).and_then(|_| {
                            let result = stream.independent_work(|| {
                                if affinity.is_err() {
                                    return Err(Error::InvalidPlan("GLM reader affinity".into()));
                                }
                                bytes = read_direct_image_timed(
                                    &direct[job.index / 288 + 3],
                                    &files[job.index / 288 + 3],
                                    &entries[job.index],
                                    staging.as_mut_slice(),
                                    image.as_mut_slice(),
                                    &mut times,
                                )?;
                                worker_status(&failure)?;
                                upload_start = Some(std::time::Instant::now());
                                unsafe { stream.copy_h2d_async(&mut job.gpu, &image) }
                            });
                            // Always retire the pinned source, even after a launch error.
                            let sync = stream.independent_work(|| stream.synchronize());
                            if let Some(start) = upload_start {
                                upload_ms = start.elapsed().as_secs_f64() * 1000.;
                            }
                            sync.and(result)
                        });
                        if let Err(error) = &result {
                            fail_workers(&failure, error);
                        }
                        if completed
                            .send(ReadDone {
                                scratch: job.scratch,
                                gpu: job.gpu,
                                read_ms: timer.elapsed().as_secs_f64() * 1000.,
                                bytes,
                                io_ms: times.io_ms,
                                pack_ms: times.pack_ms,
                                upload_ms,
                                result,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                }));
            }
            Ok(result)
        }
        pub fn submit(&mut self, job: ReadJob) -> Result<()> {
            self.jobs[self.next % self.jobs.len()]
                .send(job)
                .map_err(|_| Error::Weights("GLM disk reader stopped".into()))?;
            self.next += 1;
            Ok(())
        }
    }
    // Shared only for read-only DMA. Mutation requires unique ownership after
    // the worker is done and all token/transfer readers have retired.
    #[derive(Debug)]
    pub struct HostImage(pub PinnedHost);
    unsafe impl Sync for HostImage {}
    impl HostImage {
        pub fn as_slice(&self) -> &[u8] {
            self.0.as_slice()
        }
    }
    #[derive(Debug)]
    pub struct GpuImage(pub DeviceBuffer);
    unsafe impl Sync for GpuImage {}

    #[derive(Debug)]
    /// In-place swap: `gpu` is the slot's own buffer. With a RAM source the
    /// evicted expert (`evict_bytes`) is copied into `image` first and the
    /// loaded one (`load_bytes` of `source`) overwrites the slot; from disk
    /// `image` is the read buffer. Only run at a token boundary.
    pub struct SwapJob {
        pub replacement: crate::expert_cache::Replacement,
        pub load: usize,
        pub source: Option<Arc<HostImage>>,
        pub gpu: DeviceBuffer,
        pub image: PinnedHost,
        pub evict_bytes: usize,
        pub load_bytes: usize,
    }
    #[derive(Debug)]
    pub struct SwapDone {
        pub replacement: crate::expert_cache::Replacement,
        pub gpu: DeviceBuffer,
        pub image: PinnedHost,
        pub from_host: bool,
        pub disk_bytes: u64,
        pub worker_ms: f64,
        pub result: Result<()>,
    }
    #[derive(Debug)]
    pub struct Maintenance {
        jobs: Option<mpsc::Sender<SwapJob>>,
        pub done: mpsc::Receiver<SwapDone>,
        thread: Option<JoinHandle<()>>,
    }
    impl Maintenance {
        pub fn new(
            device: &Device,
            files: Arc<Vec<HytFile>>,
            direct: Arc<Vec<File>>,
            entries: Arc<Vec<Vec<usize>>>,
            failure: Arc<std::sync::atomic::AtomicI32>,
        ) -> Result<Self> {
            let mut read_buffer = device.alloc_pinned(DIRECT_BYTES + ALIGN - 1)?;
            let stream = device.stream()?;
            let (jobs, receive) = mpsc::channel::<SwapJob>();
            let (completed, done) = mpsc::channel();
            let thread = std::thread::spawn(move || {
                let affinity =
                    hayate_kernels::cpu::pin_thread(hayate_core::machine::MAIN.cpu.service[4]);
                while let Ok(mut job) = receive.recv() {
                    let mut disk_bytes = 0;
                    let start = std::time::Instant::now();
                    let from_host = job.source.is_some();
                    let result = worker_status(&failure).and_then(|_| {
                        let result = stream.independent_work(|| {
                            if affinity.is_err() {
                                return Err(Error::InvalidPlan("GLM maintenance affinity".into()));
                            }
                            if let Some(source) = &job.source {
                                worker_status(&failure)?;
                                unsafe {
                                    stream.copy_d2h_prefix_async(
                                        &mut job.image,
                                        &job.gpu,
                                        job.evict_bytes,
                                    )
                                }?;
                                unsafe {
                                    stream.copy_h2d_prefix_async(
                                        &mut job.gpu,
                                        &source.0,
                                        job.load_bytes,
                                    )
                                }?;
                            } else {
                                disk_bytes = read_direct_image(
                                    &direct[job.load / 288 + 3],
                                    &files[job.load / 288 + 3],
                                    &entries[job.load],
                                    read_buffer.as_mut_slice(),
                                    job.image.as_mut_slice(),
                                )?;
                                worker_status(&failure)?;
                                unsafe {
                                    stream.copy_h2d_prefix_async(
                                        &mut job.gpu,
                                        &job.image,
                                        job.load_bytes,
                                    )
                                }?;
                            }
                            Ok(())
                        });
                        let sync = stream.independent_work(|| stream.synchronize());
                        sync.and(result)
                    });
                    if let Err(error) = &result {
                        fail_workers(&failure, error);
                    }
                    drop(job.source);
                    if completed
                        .send(SwapDone {
                            replacement: job.replacement,
                            gpu: job.gpu,
                            image: job.image,
                            disk_bytes,
                            from_host,
                            worker_ms: start.elapsed().as_secs_f64() * 1000.,
                            result,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            });
            Ok(Self {
                jobs: Some(jobs),
                done,
                thread: Some(thread),
            })
        }
        pub fn submit(&self, job: SwapJob) -> Result<()> {
            self.jobs
                .as_ref()
                .unwrap()
                .send(job)
                .map_err(|_| Error::Weights("GLM maintenance stopped".into()))
        }
    }
    impl Drop for Maintenance {
        fn drop(&mut self) {
            self.jobs.take();
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }
    impl Drop for Readers {
        fn drop(&mut self) {
            self.jobs.clear();
            for thread in self.threads.drain(..) {
                let _ = thread.join();
            }
        }
    }
}
#[cfg(feature = "cuda")]
pub(super) use workers::*;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_workers_cancel_pending_jobs_and_keep_persistent_cuda_errors() {
        let failure = std::sync::atomic::AtomicI32::new(0);
        assert!(worker_status(&failure).is_ok());
        fail_workers(&failure, &Error::Weights("short read".into()));
        assert!(worker_status(&failure).is_err());
        fail_workers(
            &failure,
            &Error::CudaAsync {
                code: 700,
                sync_point: "copy",
            },
        );
        fail_workers(&failure, &Error::Weights("cancelled".into()));
        assert!(matches!(
            worker_status(&failure),
            Err(Error::CudaAsync { code: 700, .. })
        ));
    }
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires supported CUDA GPU; eight direct readers, no checkpoint needed"]
    fn glm53_parallel_disk_transfers() -> Result<()> {
        use hayate_core::{
            DType, MachineId, Shape,
            hyt::{FORMAT_VERSION, Header, TensorEntry, encode_table},
        };
        use hayate_kernels::runtime::Device;
        use std::{os::unix::fs::OpenOptionsExt, sync::Arc};
        let device = Device::select()?;
        let path = std::env::temp_dir().join(format!("glm-worker-{}", std::process::id()));
        let entries: Vec<_> = (0..8)
            .map(|i| TensorEntry {
                name: format!("expert{i}"),
                quant: QuantSpec::Plain(DType::Bf16),
                shape: Shape::new(&[1, 32]).unwrap(),
                offset: i * ALIGN as u64,
                bytes: 64,
            })
            .collect();
        let mut raw = Header {
            format_version: FORMAT_VERSION,
            machine: 0,
            tensor_count: 8,
            table_offset: 64,
            data_offset: ALIGN as u64,
            source_hash: [0; 16],
        }
        .encode()
        .to_vec();
        raw.extend(encode_table(&entries)?);
        raw.resize(8 * ALIGN + 64, 0);
        for i in 0..8 {
            raw[(i + 1) * ALIGN..(i + 1) * ALIGN + 64].fill(17 + i as u8);
        }
        std::fs::write(&path, &raw)?;
        let files = Arc::new(
            (0..8)
                .map(|_| HytFile::open(&path, MachineId::Main))
                .collect::<Result<Vec<_>>>()?,
        );
        let direct = Arc::new(
            (0..8)
                .map(|_| {
                    std::fs::OpenOptions::new()
                        .read(true)
                        .custom_flags(0o40000)
                        .open(&path)
                })
                .collect::<std::io::Result<Vec<_>>>()?,
        );
        std::fs::remove_file(path)?;
        let indices = Arc::new((0..8).map(|i| vec![i]).collect());
        let mut readers = Readers::new(
            &device,
            files,
            direct,
            indices,
            256,
            Arc::new(std::sync::atomic::AtomicI32::new(0)),
            8,
        )?;
        for i in 0..8 {
            readers.submit(ReadJob {
                index: i,
                scratch: i,
                gpu: device.alloc(256)?,
            })?;
        }
        let stream = device.stream()?;
        let mut seen = [false; 8];
        for _ in 0..8 {
            let done = readers
                .done
                .recv()
                .map_err(|_| Error::Weights("test reader stopped".into()))?;
            done.result?;
            let mut image = [0; 256];
            stream.copy_d2h(&mut image, &done.gpu)?;
            assert!(image[..64].iter().all(|&b| b == 17 + done.scratch as u8));
            assert!(image[64..].iter().all(|&b| b == 0));
            assert_eq!(done.bytes, 64);
            seen[done.scratch] = true;
        }
        assert!(seen.into_iter().all(|s| s));
        Ok(())
    }
    #[test]
    #[ignore = "read-only real NVMe benchmark; requires GLM53_HYT, no GPU"]
    fn glm53_direct_io_bandwidth() -> Result<()> {
        use std::{os::unix::fs::OpenOptionsExt, sync::Barrier};
        let root = std::env::var("GLM53_HYT")
            .map_err(|_| Error::InvalidPlan("set GLM53_HYT for IO benchmark".into()))?;
        let path = std::path::Path::new(&root).join("layer3/weights.hyt");
        let hyt = HytFile::open(&path, hayate_core::MachineId::Main)?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(0o40000)
            .open(path)?;
        let entries: Vec<Vec<usize>> = (0..128)
            .map(|expert| {
                ["gate", "up", "down"]
                    .into_iter()
                    .flat_map(|part| {
                        let prefix = format!("mlp.experts.{expert}.{part}_proj.");
                        hyt.entries()
                            .iter()
                            .enumerate()
                            .filter_map(move |(i, e)| e.name.starts_with(&prefix).then_some(i))
                    })
                    .collect()
            })
            .collect();
        if entries.iter().any(|e| e.is_empty()) {
            return Err(Error::Weights("GLM IO benchmark missing expert".into()));
        }
        let image_bytes = entries
            .iter()
            .map(|indices| {
                indices
                    .iter()
                    .map(|&i| (hyt.entries()[i].bytes as usize).next_multiple_of(256))
                    .sum::<usize>()
            })
            .max()
            .unwrap();
        let direct_bytes = hyt
            .entries()
            .iter()
            .map(|e| e.bytes as usize)
            .filter(|&n| n <= image_bytes)
            .max()
            .unwrap()
            .next_multiple_of(ALIGN)
            + ALIGN
            - 1;
        for readers in [4, 8, 16] {
            for run in 0..5 {
                let barrier = Barrier::new(readers + 1);
                std::thread::scope(|scope| -> Result<()> {
                    let jobs: Vec<_> = (0..readers)
                        .map(|worker| {
                            let (file, hyt, entries, barrier) = (&file, &hyt, &entries, &barrier);
                            scope.spawn(move || -> Result<(u64, ReadTimes)> {
                                let affinity = hayate_kernels::cpu::pin_thread(
                                    hayate_core::machine::MAIN.cpu.service[worker % 8],
                                );
                                let mut direct = vec![0; direct_bytes];
                                let mut image = vec![0; image_bytes];
                                barrier.wait();
                                affinity?;
                                let mut bytes = 0;
                                let mut times = ReadTimes::default();
                                for expert in (worker..128).step_by(readers) {
                                    bytes += read_direct_image_timed(
                                        file,
                                        hyt,
                                        &entries[expert],
                                        &mut direct,
                                        &mut image,
                                        &mut times,
                                    )?;
                                }
                                Ok((bytes, times))
                            })
                        })
                        .collect();
                    barrier.wait();
                    let start = std::time::Instant::now();
                    let mut bytes = 0;
                    let mut times = ReadTimes::default();
                    for job in jobs {
                        let (n, t) = job
                            .join()
                            .map_err(|_| Error::Weights("GLM IO worker panicked".into()))??;
                        bytes += n;
                        times.io_ms += t.io_ms;
                        times.pack_ms += t.pack_ms;
                    }
                    let seconds = start.elapsed().as_secs_f64();
                    if run >= 2 {
                        eprintln!(
                            "glm_io readers={readers} repeat={} experts=128 payload_bytes={bytes} wall_s={seconds:.6} payload_GBps={:.3} io_worker_ms={:.3} pack_worker_ms={:.3} direct=true",
                            run - 1,
                            bytes as f64 / seconds / 1e9,
                            times.io_ms,
                            times.pack_ms
                        );
                    }
                    Ok(())
                })?;
            }
        }
        Ok(())
    }

    #[test]
    fn direct_packing_keeps_scales_and_dirty_padding_out_of_images() -> Result<()> {
        use hayate_core::{
            DType, MachineId, Shape,
            hyt::{FORMAT_VERSION, Header, TensorEntry, encode_table},
        };
        use std::os::unix::fs::OpenOptionsExt;
        let path = std::env::temp_dir().join(format!("glm-direct-pack-{}", std::process::id()));
        let size = QuantSpec::Nvfp4.matrix_bytes(2, 16).unwrap();
        let mut entries: Vec<_> = (0..3)
            .map(|part| TensorEntry {
                name: format!("part{part}"),
                quant: QuantSpec::Nvfp4,
                shape: Shape::new(&[2, 16]).unwrap(),
                offset: part * ALIGN as u64,
                bytes: size,
            })
            .collect();
        entries[2].quant = QuantSpec::Plain(DType::F8E4M3);
        entries[2].bytes = 32;
        entries.push(TensorEntry {
            name: "scale".into(),
            quant: QuantSpec::Plain(DType::F32),
            shape: Shape::new(&[1])?,
            offset: 3 * ALIGN as u64,
            bytes: 4,
        });
        let mut raw = Header {
            format_version: FORMAT_VERSION,
            machine: 0,
            tensor_count: 4,
            table_offset: 64,
            data_offset: ALIGN as u64,
            source_hash: [0; 16],
        }
        .encode()
        .to_vec();
        raw.extend(encode_table(&entries)?);
        raw.resize(4 * ALIGN + 4, 0);
        for part in 0..2 {
            let at = (part + 1) * ALIGN;
            raw[at..at + size as usize - 4].fill(0x22 + part as u8);
            raw[at + size as usize - 4..at + size as usize].copy_from_slice(&0.25f32.to_le_bytes());
        }
        raw[3 * ALIGN..3 * ALIGN + 32].fill(0x38);
        raw[4 * ALIGN..].copy_from_slice(&0.5f32.to_le_bytes());
        std::fs::write(&path, &raw)?;
        let hyt = HytFile::open(&path, MachineId::Main)?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(0o40000)
            .open(&path)?;
        std::fs::remove_file(path)?;
        let mut direct = vec![0; 2 * ALIGN - 1];
        let mut image = vec![0xff; 2048];
        let bytes = read_direct_image(&file, &hyt, &[0, 1, 2, 3], &mut direct, &mut image)?;
        assert_eq!(bytes, 2 * size + 36);
        let mut at = 0;
        for e in &entries {
            let source = ALIGN + e.offset as usize;
            assert_eq!(
                &image[at..at + e.bytes as usize],
                &raw[source..source + e.bytes as usize]
            );
            let end = (at + e.bytes as usize).next_multiple_of(256);
            assert!(image[at + e.bytes as usize..end].iter().all(|&b| b == 0));
            at = end;
        }
        assert!(image[at..].iter().all(|&b| b == 0));
        assert!(
            read_direct_image(&file, &hyt, &[0, 1, 2, 3], &mut direct, &mut image[..100]).is_err()
        );
        Ok(())
    }
    #[test]
    fn aligned_read_accepts_short_tail_and_rejects_truncation() -> Result<()> {
        let path = std::env::temp_dir().join(format!("glm-direct-{}", std::process::id()));
        std::fs::write(&path, vec![19; ALIGN + 73])?;
        let file = File::open(&path)?;
        std::fs::remove_file(path)?;
        let mut storage = vec![0; 2 * ALIGN - 1];
        let start = storage.as_ptr().align_offset(ALIGN);
        let dst = &mut storage[start..start + ALIGN];
        read_aligned(&file, ALIGN as u64, 73, dst)?;
        assert!(dst[..73].iter().all(|&b| b == 19));
        assert!(read_aligned(&file, ALIGN as u64, 74, dst).is_err());
        assert!(read_aligned(&file, 1, 73, dst).is_err());
        assert!(read_aligned(&file, 0, ALIGN, &mut dst[..ALIGN - 1]).is_err());
        Ok(())
    }
}
