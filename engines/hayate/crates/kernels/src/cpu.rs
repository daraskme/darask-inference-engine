//! CPU kernels for the two hosts: AVX2 + AVX-VNNI on the main PC
//! (i9-12900KS), AVX-512 on the sub PC (Ryzen 7 8700F). Selected per
//! machine at build time via `target_feature`, never probed at runtime.
//!
//! Used for cold MoE experts (GLM-5.3-Flash) and n-gram table gathers
//! (Qwen3.8-Flash-Next). Threads are pinned to the machine's `compute`
//! cores only; on the main PC mixing in E-cores makes a barrier-synchronised
//! GEMV run at E-core speed.

/// Scalar reference for the `QuantSpec::Q4Bf16` GEMV that the SIMD paths
/// must match bit-for-bit in the dequantized values.
///
/// Layout (one row-major `[rows, cols]` matrix):
/// - `w`: two 4-bit values per byte, low nibble first, stored with a +8 offset;
/// - `scales`: one BF16 (raw `u16` bits) per `group` columns of each row.
pub fn gemv_q4_bf16_ref(w: &[u8], scales: &[u16], x: &[f32], y: &mut [f32], group: usize) {
    let cols = x.len();
    assert!(
        group > 0 && group.is_multiple_of(2),
        "group must be a positive even number"
    );
    assert!(
        cols.is_multiple_of(group),
        "cols must be a multiple of group"
    );
    assert_eq!(w.len(), y.len() * cols / 2);
    assert_eq!(scales.len(), y.len() * cols / group);

    for (r, out) in y.iter_mut().enumerate() {
        let row = &w[r * cols / 2..(r + 1) * cols / 2];
        let row_scales = &scales[r * cols / group..(r + 1) * cols / group];
        let mut acc = 0.0f32;
        for (c, &xv) in x.iter().enumerate() {
            let byte = row[c / 2];
            let q = if c % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            acc += (q as f32 - 8.0) * bf16_to_f32(row_scales[c / group]) * xv;
        }
        *out = acc;
    }
}

pub const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

use hayate_core::{Error, Machine, Result};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

/// Group-32 GEMV, selected by target-cpu at build time.
pub fn gemv_q4_bf16(w: &[u8], scales: &[u16], x: &[f32], y: &mut [f32]) -> Result<()> {
    validate(w, scales, x, y)?;
    gemv_rows(w, scales, x, y);
    Ok(())
}

fn validate(w: &[u8], scales: &[u16], x: &[f32], y: &[f32]) -> Result<()> {
    let n = y.len().checked_mul(x.len());
    if !x.len().is_multiple_of(32)
        || n.is_none()
        || n.map(|n| n / 2) != Some(w.len())
        || n.map(|n| n / 32) != Some(scales.len())
    {
        return Err(Error::InvalidShape(
            "Q4Bf16 requires group 32 and exact row-major storage".into(),
        ));
    }
    Ok(())
}

fn gemv_rows(w: &[u8], scales: &[u16], x: &[f32], y: &mut [f32]) {
    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    // SAFETY: the build enables the ISA; validate checks all slice lengths.
    unsafe {
        gemv_avx512(w, scales, x, y)
    }
    #[cfg(all(
        target_feature = "avx2",
        target_feature = "fma",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    // SAFETY: the build enables the ISA; validate checks all slice lengths.
    unsafe {
        gemv_avx2(w, scales, x, y)
    }
    #[cfg(not(any(
        all(target_feature = "avx512f", target_feature = "avx512bw"),
        all(target_feature = "avx2", target_feature = "fma")
    )))]
    gemv_q4_bf16_ref(w, scales, x, y, 32);
}

#[cfg(all(
    target_feature = "avx2",
    target_feature = "fma",
    not(all(target_feature = "avx512f", target_feature = "avx512bw"))
))]
unsafe fn gemv_avx2(w: &[u8], scales: &[u16], x: &[f32], y: &mut [f32]) {
    use std::arch::x86_64::*;
    // SAFETY: callers supply exact group-32 storage; loads stay within each group.
    unsafe {
        let mask = _mm_set1_epi8(15);
        let offset = _mm_set1_epi8(8);
        for (r, out) in y.iter_mut().enumerate() {
            let mut acc = [_mm256_setzero_ps(); 4];
            for g in 0..x.len() / 32 {
                let packed = _mm_loadu_si128(w.as_ptr().add(r * x.len() / 2 + g * 16).cast());
                let lo = _mm_and_si128(packed, mask);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
                let a = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), offset);
                let b = _mm_sub_epi8(_mm_unpackhi_epi8(lo, hi), offset);
                let q = [a, _mm_srli_si128::<8>(a), b, _mm_srli_si128::<8>(b)];
                let scale = _mm256_set1_ps(bf16_to_f32(scales[r * x.len() / 32 + g]));
                for i in 0..4 {
                    let weight =
                        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q[i])), scale);
                    acc[i] = _mm256_fmadd_ps(
                        weight,
                        _mm256_loadu_ps(x.as_ptr().add(g * 32 + i * 8)),
                        acc[i],
                    );
                }
            }
            let total = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
            let halves = _mm_add_ps(
                _mm256_castps256_ps128(total),
                _mm256_extractf128_ps::<1>(total),
            );
            let pairs = _mm_hadd_ps(halves, halves);
            *out = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
        }
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
// AVX-512 std::arch intrinsics require Rust 1.89 on the sub-PC build.
#[clippy::msrv = "1.89.0"]
unsafe fn gemv_avx512(w: &[u8], scales: &[u16], x: &[f32], y: &mut [f32]) {
    use std::arch::x86_64::*;
    // SAFETY: callers supply exact group-32 storage; loads stay within each group.
    unsafe {
        let mask = _mm_set1_epi8(15);
        let offset = _mm_set1_epi8(8);
        for (r, out) in y.iter_mut().enumerate() {
            let mut acc = [_mm512_setzero_ps(); 2];
            for g in 0..x.len() / 32 {
                let packed = _mm_loadu_si128(w.as_ptr().add(r * x.len() / 2 + g * 16).cast());
                let lo = _mm_and_si128(packed, mask);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
                let q = [
                    _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), offset),
                    _mm_sub_epi8(_mm_unpackhi_epi8(lo, hi), offset),
                ];
                let scale = _mm512_set1_ps(bf16_to_f32(scales[r * x.len() / 32 + g]));
                for i in 0..2 {
                    let weight =
                        _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q[i])), scale);
                    acc[i] = _mm512_fmadd_ps(
                        weight,
                        _mm512_loadu_ps(x.as_ptr().add(g * 32 + i * 16)),
                        acc[i],
                    );
                }
            }
            *out = _mm512_reduce_add_ps(_mm512_add_ps(acc[0], acc[1]));
        }
    }
}

unsafe extern "C" {
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
}

/// Pin the calling Linux thread; errors never silently change core placement.
pub fn pin_thread(core: u16) -> Result<()> {
    let mut mask = [0u64; 16]; // Linux cpu_set_t (1024 bits).
    if usize::from(core) >= mask.len() * 64 {
        return Err(Error::UnsupportedMachine(format!(
            "CPU {core} exceeds affinity mask"
        )));
    }
    mask[usize::from(core) / 64] = 1 << (core % 64);
    // SAFETY: pid 0 names this thread; mask points to cpusetsize readable bytes.
    if unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

#[derive(Debug)]
enum Job {
    Gemv {
        w: *const u8,
        scales: *const u16,
        x: *const f32,
        y: *mut f32,
        rows: usize,
        cols: usize,
    },
    Read {
        words: *const u64,
        len: usize,
    },
    Stop,
}

// SAFETY: jobs are private. ComputePool methods retain the input borrows and
// exclusive, disjoint output slices until every submitted job has completed.
unsafe impl Send for Job {}

#[derive(Debug)]
struct Worker {
    tx: Sender<Job>,
    done: Receiver<Result<u64>>,
    thread: Option<JoinHandle<()>>,
}

/// One persistent worker per Machine::cpu.compute core. Calls are synchronous;
/// row slices are borrowed without allocation or copying into worker storage.
#[derive(Debug)]
pub struct ComputePool {
    workers: Vec<Worker>,
}

impl ComputePool {
    pub fn new(machine: &Machine) -> Result<Self> {
        if machine.cpu.compute.is_empty() {
            return Err(Error::UnsupportedMachine("no compute cores".into()));
        }
        let mut pool = Self {
            workers: Vec::new(),
        };
        for &core in machine.cpu.compute {
            let (tx, rx) = mpsc::channel();
            let (complete, done) = mpsc::channel();
            let thread = thread::Builder::new()
                .name(format!("hayate-cpu-{core}"))
                .spawn(move || {
                    let ready = pin_thread(core).map(|()| 0);
                    let pinned = ready.is_ok();
                    if complete.send(ready).is_err() || !pinned {
                        return;
                    }
                    while let Ok(job) = rx.recv() {
                        if matches!(job, Job::Stop) {
                            break;
                        }
                        let result = std::panic::catch_unwind(|| {
                            // SAFETY: the submitting call waits for all completions before
                            // releasing its borrows, including on errors and worker panics.
                            unsafe {
                                match job {
                                    Job::Gemv {
                                        w,
                                        scales,
                                        x,
                                        y,
                                        rows,
                                        cols,
                                    } => {
                                        gemv_rows(
                                            std::slice::from_raw_parts(w, rows * cols / 2),
                                            std::slice::from_raw_parts(scales, rows * cols / 32),
                                            std::slice::from_raw_parts(x, cols),
                                            std::slice::from_raw_parts_mut(y, rows),
                                        );
                                        0
                                    }
                                    Job::Read { words, len } => {
                                        read_words(std::slice::from_raw_parts(words, len))
                                    }
                                    Job::Stop => unreachable!(),
                                }
                            }
                        })
                        .map_err(|_| Error::InvalidPlan("CPU worker panicked".into()));
                        if complete.send(result).is_err() {
                            break;
                        }
                    }
                })?;
            pool.workers.push(Worker {
                tx,
                done,
                thread: Some(thread),
            });
            pool.workers
                .last()
                .unwrap()
                .done
                .recv()
                .map_err(|_| worker_closed())??;
        }
        Ok(pool)
    }

    pub fn gemv_q4_bf16(
        &mut self,
        w: &[u8],
        scales: &[u16],
        x: &[f32],
        y: &mut [f32],
    ) -> Result<()> {
        validate(w, scales, x, y)?;
        let rows = y.len();
        let cols = x.len();
        let count = self.workers.len();
        let output = y.as_mut_ptr();
        self.dispatch(|i| {
            let (start, end) = partition(rows, i, count);
            Job::Gemv {
                w: w[start * cols / 2..].as_ptr(),
                scales: scales[start * cols / 32..].as_ptr(),
                x: x.as_ptr(),
                y: output.wrapping_add(start),
                rows: end - start,
                cols,
            }
        })
        .map(|_| ())
    }

    /// Read every word once on compute cores; return a checksum so the reads
    /// remain observable. Buffers used for RAM measurement must exceed LLC.
    pub fn read_ram(&mut self, words: &[u64]) -> Result<u64> {
        let count = self.workers.len();
        self.dispatch(|i| {
            let (start, end) = partition(words.len(), i, count);
            Job::Read {
                words: words[start..].as_ptr(),
                len: end - start,
            }
        })
    }

    fn dispatch(&mut self, mut job: impl FnMut(usize) -> Job) -> Result<u64> {
        let mut error = None;
        let mut submitted = 0;
        for (i, worker) in self.workers.iter().enumerate() {
            if worker.tx.send(job(i)).is_err() {
                error = Some(worker_closed());
                break;
            }
            submitted += 1;
        }
        let mut checksum = 0u64;
        for worker in &mut self.workers[..submitted] {
            match worker.done.recv() {
                Ok(Ok(value)) => checksum = checksum.wrapping_add(value),
                Ok(Err(e)) => error = Some(e),
                Err(_) => {
                    // Join before releasing borrowed memory if a thread died
                    // outside the protected job execution.
                    if let Some(thread) = worker.thread.take() {
                        let _ = thread.join();
                    }
                    error = Some(worker_closed());
                }
            }
        }
        match error {
            Some(e) => Err(e),
            None => Ok(checksum),
        }
    }
}

impl Drop for ComputePool {
    fn drop(&mut self) {
        for worker in &self.workers {
            let _ = worker.tx.send(Job::Stop);
        }
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

// Divide without len * worker-index overflow: constructing jobs must never
// unwind while earlier workers still hold pointers to the borrowed buffers.
fn partition(len: usize, i: usize, count: usize) -> (usize, usize) {
    let start = (len / count) * i + i.min(len % count);
    (start, start + len / count + usize::from(i < len % count))
}

fn worker_closed() -> Error {
    Error::InvalidPlan("CPU worker channel closed".into())
}

fn read_words(words: &[u64]) -> u64 {
    #[cfg(target_feature = "avx2")]
    {
        use std::arch::x86_64::*;
        // SAFETY: avx2 is enabled at build time; chunks cover only valid words.
        unsafe {
            let mut acc = [_mm256_setzero_si256(); 4];
            let mut chunks = words.chunks_exact(16);
            for chunk in &mut chunks {
                for (i, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_add_epi64(*a, _mm256_loadu_si256(chunk.as_ptr().add(i * 4).cast()));
                }
            }
            let total = _mm256_add_epi64(
                _mm256_add_epi64(acc[0], acc[1]),
                _mm256_add_epi64(acc[2], acc[3]),
            );
            let mut lanes = [0u64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast(), total);
            lanes
                .iter()
                .chain(chunks.remainder())
                .fold(0u64, |a, &b| a.wrapping_add(b))
        }
    }
    #[cfg(not(target_feature = "avx2"))]
    words.iter().fold(0u64, |a, &b| a.wrapping_add(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    const HALF: u16 = 0x3f00; // 0.5
    const TWO: u16 = 0x4000; // 2.0

    #[test]
    fn dequantizes_one_row() {
        // Values 9, 7, 8, 15 → +1, -1, 0, +7 after the offset.
        let w = [0x79, 0xf8];
        let mut y = [0.0];
        gemv_q4_bf16_ref(&w, &[HALF, TWO], &[1.0; 4], &mut y, 2);
        assert_eq!(y[0], 0.5 - 0.5 + 0.0 + 14.0);
    }

    #[test]
    fn uses_per_row_scales_and_non_uniform_x() {
        // Row 0: +1, +1 (scale 0.5). Row 1: -8, +7 (scale 2.0).
        let w = [0x99, 0xf0];
        let mut y = [0.0; 2];
        gemv_q4_bf16_ref(&w, &[HALF, TWO], &[3.0, -1.0], &mut y, 2);
        assert_eq!(y, [0.5 * 3.0 - 0.5, 2.0 * (-8.0 * 3.0 - 7.0)]);
    }

    #[test]
    #[should_panic(expected = "group")]
    fn rejects_zero_group() {
        gemv_q4_bf16_ref(&[], &[], &[], &mut [], 0);
    }

    fn fixture(rows: usize, cols: usize) -> (Vec<u8>, Vec<u16>, Vec<f32>) {
        let mut state = 123456789u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let w = (0..rows * cols / 2).map(|_| next() as u8).collect();
        let scales = (0..rows * cols / 32)
            .map(|_| 0x3b00 + (next() % 2048) as u16)
            .collect();
        let x = (0..cols)
            .map(|_| (next() as i32 as f32) / i32::MAX as f32)
            .collect();
        (w, scales, x)
    }

    fn assert_close(w: &[u8], scales: &[u16], x: &[f32], actual: &[f32]) {
        let mut expected = vec![0.0; actual.len()];
        gemv_q4_bf16_ref(w, scales, x, &mut expected, 32);
        for r in 0..actual.len() {
            // Bound by the absolute products, so cancellation does not turn
            // an otherwise valid SIMD/FMA result into a relative-error failure.
            let magnitude: f32 = x
                .iter()
                .enumerate()
                .map(|(c, &v)| {
                    let byte = w[r * x.len() / 2 + c / 2];
                    let q = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                    ((q as f32 - 8.0) * bf16_to_f32(scales[r * x.len() / 32 + c / 32]) * v).abs()
                })
                .sum();
            let tolerance = 2e-6 * magnitude + 1e-6;
            assert!(
                (actual[r] - expected[r]).abs() <= tolerance,
                "row {r}: {} vs {}, tolerance {tolerance}",
                actual[r],
                expected[r]
            );
        }
    }

    #[test]
    fn selected_gemv_matches_reference() {
        for (rows, cols) in [
            (0, 32),
            (3, 0),
            (1, 32),
            (9, 96),
            (4096, 2048),
            (2048, 4096),
        ] {
            let (w, scales, x) = fixture(rows, cols);
            let mut y = vec![f32::NAN; rows];
            gemv_q4_bf16(&w, &scales, &x, &mut y).unwrap();
            assert_close(&w, &scales, &x, &y);
        }
    }

    #[test]
    fn persistent_workers_match_reference_and_read_every_word() {
        let mut pool = ComputePool::new(&hayate_core::machine::MAIN).unwrap();
        for (rows, cols) in [(3, 32), (11, 96), (0, 32), (3, 0), (17, 2048)] {
            let (w, scales, x) = fixture(rows, cols);
            let mut y = vec![f32::NAN; rows];
            pool.gemv_q4_bf16(&w, &scales, &x, &mut y).unwrap();
            assert_close(&w, &scales, &x, &y);
            let words: Vec<u64> = (0..1003).map(|i| u64::MAX - i).collect();
            let expected = words.iter().fold(0u64, |a, &b| a.wrapping_add(b));
            assert_eq!(pool.read_ram(&words).unwrap(), expected);
            assert_eq!(pool.read_ram(&[]).unwrap(), 0);
        }
    }

    #[test]
    fn rejects_malformed_group32_storage() {
        assert!(gemv_q4_bf16(&[0; 16], &[HALF], &[0.0; 31], &mut [0.0]).is_err());
        assert!(gemv_q4_bf16(&[0; 15], &[HALF], &[0.0; 32], &mut [0.0]).is_err());
        assert!(gemv_q4_bf16(&[0; 16], &[], &[0.0; 32], &mut [0.0]).is_err());
    }
}
