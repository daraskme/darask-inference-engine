//! cargo run --release -p hayate-kernels --features cuda --example gpu_probe
//! Warmup/repetitions are fixed before measuring (ARCHITECTURE.md §10).

use hayate_core::{Error, Result};
use hayate_kernels::cuda::ffi;
use hayate_kernels::runtime::{Device, Stream, check};

const BYTES: usize = 256 << 20;
const WARMUP: usize = 5;
const SAMPLES: usize = 21;
const COPIES_PER_SAMPLE: usize = 8;

fn median_ms(stream: &Stream, mut submit: impl FnMut() -> Result<()>) -> Result<f64> {
    let start = stream.event()?;
    let end = stream.event()?;
    let mut samples = Vec::with_capacity(SAMPLES);
    for sample in 0..WARMUP + SAMPLES {
        let result: Result<()> = (|| {
            start.record(stream)?;
            for _ in 0..COPIES_PER_SAMPLE {
                submit()?;
            }
            end.record(stream)?;
            end.synchronize()?;
            Ok(())
        })();
        // Preserve allocation lifetimes even when a submission/event fails.
        stream.synchronize()?;
        result?;
        if sample >= WARMUP {
            samples.push(f64::from(start.elapsed_ms(&end)?));
        }
    }
    samples.sort_by(f64::total_cmp);
    let ms = samples[SAMPLES / 2];
    if !ms.is_finite() || ms <= 0.0 {
        return Err(Error::InvalidShape(
            "CUDA event returned invalid elapsed time".into(),
        ));
    }
    Ok(ms)
}

fn report(label: &str, traffic: usize, ms: f64) {
    let gb_s = traffic as f64 * COPIES_PER_SAMPLE as f64 / (ms * 1e6);
    println!("{label}: {gb_s:.2} GB/s (median {ms:.3} ms / {COPIES_PER_SAMPLE} copies)");
}

fn main() -> Result<()> {
    let device = Device::select()?;
    println!(
        "Startup validation: PASS ({:?}, {}, CC {}.{})",
        device.machine, device.name, device.sm.0, device.sm.1
    );
    let (free, total) = device.memory_info()?;
    println!(
        "VRAM after context initialization: {:.3} GiB free / {:.3} GiB total ({free} / {total} bytes)",
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64
    );
    println!(
        "256 MiB/buffer; pinned host; {WARMUP} warmup batches; {SAMPLES} measured batches; CUDA events; GB = 10^9 bytes"
    );
    println!("DRAM traffic counts reads + writes; PCIe reports one-way payload bandwidth.");

    println!("注意: 計測中の PCIe リンク世代は nvidia-smi で確認すること。");
    let mut blocks = 0;
    check(
        unsafe { ffi::hayate_bandwidth_blocks(&mut blocks) },
        "hayate_bandwidth_blocks",
    )?;
    println!("DRAM grid: {blocks} blocks × 256 threads (SM count × resident blocks)");

    let stream = device.stream()?;
    let mut host = device.alloc_pinned(BYTES)?;
    let mut src = device.alloc(BYTES)?;
    let mut dst = device.alloc(BYTES)?;
    host.as_mut_slice().fill(0x5a);
    stream.copy_h2d(&mut src, host.as_slice())?;
    let raw = stream.as_raw();
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let dram = median_ms(&stream, || {
        // Disjoint arenas; both outlive measurement and its error-path drain.
        check(
            unsafe { ffi::hayate_bandwidth_copy(dst_ptr, src_ptr, BYTES, blocks, raw) },
            "hayate_bandwidth_copy",
        )
    })?;
    report("GPU DRAM", 2 * BYTES, dram);
    stream.copy_d2h(host.as_mut_slice(), &dst)?;
    if host.as_slice().iter().any(|&v| v != 0x5a) {
        return Err(Error::InvalidShape("DRAM copy verification failed".into()));
    }

    let h2d = median_ms(&stream, || {
        // The allocations stay alive; no host access until median_ms drains.
        unsafe { stream.copy_h2d_async(&mut src, &host) }
    })?;
    report("PCIe H2D", BYTES, h2d);
    host.as_mut_slice().fill(0);
    let d2h = median_ms(&stream, || unsafe {
        stream.copy_d2h_async(&mut host, &src)
    })?;
    report("PCIe D2H", BYTES, d2h);
    if host.as_slice().iter().any(|&v| v != 0x5a) {
        return Err(Error::InvalidShape(
            "PCIe round-trip verification failed".into(),
        ));
    }
    println!("Copy verification: PASS (all 256 MiB checked)");
    Ok(())
}
