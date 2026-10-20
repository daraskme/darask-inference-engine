//! M0 micro-benchmarks (plan §14.2). Each prints a table and appends JSON lines to
//! `results/m0.jsonl` so the numbers that set the M1–M2 targets are kept with the code.
//!
//!   tsumuji-bench [info|bw|l2|pdl|gemv|zerocopy|mma|all] [--host-gib N]

mod gemv_shapes;
mod moe;
mod nvfp4;

use std::io::Write as _;
use tsumuji_cuda::sys::*;
use tsumuji_cuda::{
    Device, DeviceBuffer, Function, LaunchDims, MappedHostBuffer, Module, Result, Stream, args,
    launch, time_ms,
};

pub const GIB: usize = 1 << 30;

pub struct Bench {
    dev: Device,
    sms: u32,
    pub stream: Stream,
    sink: DeviceBuffer,
    log: std::fs::File,
}

impl Bench {
    pub fn record(&self, json: String) {
        let _ = writeln!(&self.log, "{json}");
    }

    pub fn fill_random(&self, buf: &DeviceBuffer, seed: u64) -> Result<()> {
        let m = Module::load(tsumuji_kernels::BENCH_BW)?;
        let f = m.function("fill_random")?;
        let n = (buf.bytes / 16) as u64;
        let mut a = args!(buf.ptr, n, seed);
        unsafe {
            launch(
                f,
                LaunchDims::linear(self.sms * 8, 256),
                &self.stream,
                false,
                &mut a,
            )?
        };
        self.stream.synchronize()
    }

    /// Bytes/s of one bw kernel over `buf`, best over a few grid sizes.
    pub fn read_bw(&self, f: Function, buf: &DeviceBuffer, iters: usize) -> Result<(f64, u32)> {
        let n = (buf.bytes / 16) as u64;
        let mut best = (0.0, 0);
        for per_sm in [2u32, 4, 8, 16] {
            let grid = self.sms * per_sm;
            let ms = time_ms(&self.stream, 3, iters, |_| {
                let mut a = args!(buf.ptr, n, self.sink.ptr);
                unsafe {
                    launch(
                        f,
                        LaunchDims::linear(grid, 256),
                        &self.stream,
                        false,
                        &mut a,
                    )
                }
            })?;
            let bps = buf.bytes as f64 / (ms * 1e-3);
            if bps > best.0 {
                best = (bps, grid);
            }
        }
        Ok(best)
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let which = argv.first().map_or("all", String::as_str).to_string();
    let host_gib = argv
        .iter()
        .position(|a| a == "--host-gib")
        .and_then(|i| argv.get(i + 1))
        .map_or(8, |v| v.parse().expect("--host-gib N"));

    let dev = Device::open(0)?;
    let sms = dev.attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
    std::fs::create_dir_all("results").ok();
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("results/m0.jsonl")
        .unwrap();
    let b = Bench {
        stream: Stream::new()?,
        sink: DeviceBuffer::zeroed(64)?,
        sms,
        dev,
        log,
    };

    let all = which == "all";
    if all || which == "info" {
        info(&b)?;
    }
    if all || which == "bw" {
        bw(&b)?;
    }
    if all || which == "l2" {
        l2(&b)?;
    }
    if all || which == "pdl" {
        pdl(&b)?;
    }
    if all || which == "gemv" {
        gemv(&b)?;
    }
    if all || which == "zerocopy" {
        zerocopy(&b, host_gib)?;
    }
    if all || which == "mma" {
        mma(&b)?;
    }
    if which == "gemv-shapes" {
        let n = argv
            .iter()
            .find_map(|a| a.strip_prefix("n=").and_then(|v| v.parse().ok()))
            .unwrap_or(1);
        gemv_shapes::run(&b, argv.iter().any(|a| a == "int8"), n)?;
    }
    if which == "moe" {
        moe::run(&b)?;
    }
    Ok(())
}

fn info(b: &Bench) -> Result<()> {
    let d = &b.dev;
    let (free, total) = d.mem_info()?;
    let mem_khz = d.attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)? as f64;
    let bus = d.attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)? as f64;
    // GDDR7 attribute reports the command clock; nominal peak is cross-checked against bw below.
    let nominal = mem_khz * 1e3 * 2.0 * bus / 8.0;
    let l2 = d.attr(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?;
    let persist = d.attr(CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE)?;
    let smem_blk = d.attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?;
    let smem_sm = d.attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?;
    let clk = d.attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE)?;
    let name = d.name()?;
    println!("== info: {name}");
    println!(
        "   SMs {}  clock {} MHz  VRAM {:.1} GiB ({:.1} free)",
        b.sms,
        clk / 1000,
        total as f64 / GIB as f64,
        free as f64 / GIB as f64
    );
    println!(
        "   L2 {} MiB (persisting max {} MiB)  smem/block {} KiB  smem/SM {} KiB",
        l2 >> 20,
        persist >> 20,
        smem_blk >> 10,
        smem_sm >> 10
    );
    println!(
        "   memory: bus {bus} bit, attr clock {} MHz -> attr-derived {:.0} GB/s",
        mem_khz / 1e3,
        nominal / 1e9
    );
    b.record(format!(
        r#"{{"bench":"info","name":"{name}","sms":{},"l2":{l2},"l2_persist":{persist},"smem_block":{smem_blk},"smem_sm":{smem_sm},"clock_khz":{clk},"vram":{total}}}"#,
        b.sms
    ));
    Ok(())
}

fn bw(b: &Bench) -> Result<()> {
    let m = Module::load(tsumuji_kernels::BENCH_BW)?;
    let buf = DeviceBuffer::new(4 * GIB)?;
    b.fill_random(&buf, 1)?;
    println!("== bw: streaming read of 4 GiB (DRAM)");
    for name in ["bw_read", "bw_read_cs"] {
        let (bps, grid) = b.read_bw(m.function(name)?, &buf, 10)?;
        println!("   {name:<11} {:7.1} GB/s  (grid {grid} x 256)", bps / 1e9);
        b.record(format!(
            r#"{{"bench":"bw","kernel":"{name}","gbps":{:.1},"grid":{grid}}}"#,
            bps / 1e9
        ));
    }
    Ok(())
}

fn l2(b: &Bench) -> Result<()> {
    let m = Module::load(tsumuji_kernels::BENCH_BW)?;
    let f = m.function("bw_read")?;
    println!("== l2: repeated reads of one working set (read bandwidth vs size)");
    for mib in [8usize, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 512] {
        let buf = DeviceBuffer::new(mib << 20)?;
        b.fill_random(&buf, 2)?;
        let (bps, _) = b.read_bw(f, &buf, 200)?;
        println!("   {mib:4} MiB  {:7.1} GB/s", bps / 1e9);
        b.record(format!(
            r#"{{"bench":"l2","mib":{mib},"gbps":{:.1}}}"#,
            bps / 1e9
        ));
    }
    Ok(())
}

fn pdl(b: &Bench) -> Result<()> {
    let m = Module::load(tsumuji_kernels::BENCH_PDL)?;
    let io = DeviceBuffer::zeroed(2 * 64)?;
    let (x0, x1) = (io.ptr, io.ptr + 64);
    let s = &b.stream;
    println!("== pdl: dependent kernel chains (1 block-wave each)");

    // Pure dependency/launch overhead.
    let empty = m.function("chain_empty")?;
    let steps = 1000;
    let enqueue_empty = |st: &Stream, pdl: bool| -> Result<()> {
        for i in 0..steps {
            let (src, dst) = if i % 2 == 0 { (x0, x1) } else { (x1, x0) };
            let mut a = args!(src, dst);
            unsafe { launch(empty, LaunchDims::linear(b.sms, 128), st, pdl, &mut a)? };
        }
        Ok(())
    };
    let plain = time_ms(s, 2, 10, |_| enqueue_empty(s, false))?;
    let g = s.capture(|st| enqueue_empty(st, false))?;
    let graph = time_ms(s, 2, 10, |_| g.launch(s))?;
    let gp = s.capture(|st| enqueue_empty(st, true))?;
    let graph_pdl = time_ms(s, 2, 10, |_| gp.launch(s))?;
    let us = |ms: f64| ms * 1e3 / steps as f64;
    println!(
        "   empty kernel: stream {:.2} us/kernel, graph {:.2}, graph+PDL {:.2}",
        us(plain),
        us(graph),
        us(graph_pdl)
    );
    b.record(format!(
        r#"{{"bench":"pdl_empty","stream_us":{:.3},"graph_us":{:.3},"graph_pdl_us":{:.3}}}"#,
        us(plain),
        us(graph),
        us(graph_pdl)
    ));

    // Weight-streaming steps. A 2 GiB pool of distinct weights keeps every step in DRAM.
    let pool = DeviceBuffer::new(2 * GIB)?;
    b.fill_random(&pool, 3)?;
    let (peak, _) = b.read_bw(
        Module::load(tsumuji_kernels::BENCH_BW)?.function("bw_read_cs")?,
        &pool,
        5,
    )?;
    let threads = b.sms as usize * 256;
    for vpt in [1usize, 2, 4, 8, 16] {
        let f = m.function(&format!("chain_step_{vpt}"))?;
        let step_bytes = threads * vpt * 16;
        let n_steps = (pool.bytes / step_bytes).min(400);
        let enqueue = |st: &Stream, pdl: bool| -> Result<()> {
            for i in 0..n_steps {
                let w = pool.ptr + (i * step_bytes) as u64;
                let (src, dst) = if i % 2 == 0 { (x0, x1) } else { (x1, x0) };
                let mut a = args!(w, src, dst);
                unsafe { launch(f, LaunchDims::linear(b.sms, 256), st, pdl, &mut a)? };
            }
            Ok(())
        };
        let g = s.capture(|st| enqueue(st, false))?;
        let t = time_ms(s, 2, 10, |_| g.launch(s))?;
        let gp = s.capture(|st| enqueue(st, true))?;
        let tp = time_ms(s, 2, 10, |_| gp.launch(s))?;
        let total = (n_steps * step_bytes) as f64;
        let ideal_us = step_bytes as f64 / peak * 1e6;
        let (u, up) = (t * 1e3 / n_steps as f64, tp * 1e3 / n_steps as f64);
        println!(
            "   step {:6.2} MiB: graph {:6.2} us ({:5.1}% of peak)  graph+PDL {:6.2} us ({:5.1}%)  ideal {:5.2} us",
            step_bytes as f64 / (1 << 20) as f64,
            u,
            100.0 * total / (t * 1e-3) / peak,
            up,
            100.0 * total / (tp * 1e-3) / peak,
            ideal_us
        );
        b.record(format!(
            r#"{{"bench":"pdl_step","step_bytes":{step_bytes},"graph_us":{u:.3},"graph_pdl_us":{up:.3},"ideal_us":{ideal_us:.3},"peak_gbps":{:.1}}}"#,
            peak / 1e9
        ));
    }
    Ok(())
}

fn gemv(b: &Bench) -> Result<()> {
    const N: usize = 1280;
    const K: usize = 2560;
    let m = Module::load(tsumuji_kernels::BENCH_GEMV)?;
    let f = m.function("gemv_nvfp4_k2560")?;
    let (w_bytes, sf_bytes) = (N * K / 2, N * K / 16);
    let mat = w_bytes + sf_bytes;
    let copies = (1200 << 20) / mat; // > 1 GiB in total: every launch reads DRAM
    let pool = DeviceBuffer::new(copies * mat)?;
    b.fill_random(&pool, 4)?;

    // Copy 0 gets a checked matrix: random codes, finite scales, random activations.
    let case = nvfp4::Case::random(N, K, 7);
    pool.upload(&case.packed_with_scales())?;
    let x = DeviceBuffer::from_slice(&case.x_bf16)?;
    let y = DeviceBuffer::zeroed(N * 4)?;
    let block = 128u32;
    let dims = LaunchDims::linear((N as u32).div_ceil(block / 32), block);
    let (w0, sf0, gs, n) = (
        pool.ptr,
        pool.ptr + w_bytes as u64,
        case.global_scale,
        N as i32,
    );
    let mut a = args!(w0, sf0, gs, x.ptr, y.ptr, n);
    unsafe { launch(f, dims, &b.stream, false, &mut a)? };
    b.stream.synchronize()?;
    let got: Vec<f32> = y.to_vec(N)?;
    let (max_rel, worst) = case.compare(&got);
    println!(
        "== gemv: NVFP4 W4A16, {N}x{K}, batch 1 (check vs f64 CPU: max rel err {max_rel:.2e} at row {worst})"
    );
    assert!(
        max_rel < 1e-4,
        "GEMV result does not match the CPU reference"
    );

    let s = &b.stream;
    let enqueue = |st: &Stream, pdl: bool| -> Result<()> {
        for i in 0..copies {
            let w = pool.ptr + (i * mat) as u64;
            let sf = w + w_bytes as u64;
            let mut a = args!(w, sf, gs, x.ptr, y.ptr, n);
            unsafe { launch(f, dims, st, pdl, &mut a)? };
        }
        Ok(())
    };
    let peak = b
        .read_bw(
            Module::load(tsumuji_kernels::BENCH_BW)?.function("bw_read_cs")?,
            &pool,
            5,
        )?
        .0;
    for pdl in [false, true] {
        let g = s.capture(|st| enqueue(st, pdl))?;
        let ms = time_ms(s, 2, 5, |_| g.launch(s))?;
        let us = ms * 1e3 / copies as f64;
        let bps = mat as f64 / (us * 1e-6);
        println!(
            "   graph{}: {us:6.2} us/GEMV  {:7.1} GB/s  ({:4.1}% of stream peak {:.0})",
            if pdl { "+PDL" } else { "    " },
            bps / 1e9,
            100.0 * bps / peak,
            peak / 1e9
        );
        b.record(format!(
            r#"{{"bench":"gemv_nvfp4","n":{N},"k":{K},"pdl":{pdl},"us":{us:.3},"gbps":{:.1},"max_rel":{max_rel:.3e}}}"#,
            bps / 1e9
        ));
    }
    Ok(())
}

fn zerocopy(b: &Bench, host_gib: usize) -> Result<()> {
    let m = Module::load(tsumuji_kernels::BENCH_ZEROCOPY)?;
    let f = m.function("ple_gather_probe")?;
    const ROW: usize = 160;
    let bytes = host_gib * GIB;
    let n_rows = (bytes / ROW) as u64;
    println!("== zerocopy: random 160 B rows, 16 per token, from a {host_gib} GiB table");
    let t0 = std::time::Instant::now();
    let mut host = MappedHostBuffer::new(bytes)?;
    fill_host(host.as_mut_slice());
    println!(
        "   pinned + filled host table in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    let dev_table = DeviceBuffer::new(bytes.min(8 * GIB))?;
    b.fill_random(&dev_table, 5)?;
    let dev_rows = (dev_table.bytes / ROW) as u64;
    let out = DeviceBuffer::new(16384 * 16 * ROW)?;
    let s = &b.stream;
    for t in [1u32, 16, 256, 4096, 16384] {
        let mut res = Vec::new();
        for (label, base, rows) in [
            ("host", host.dev, n_rows),
            ("device", dev_table.ptr, dev_rows),
        ] {
            // A graph of 20 dependent gathers: per-gather time without per-launch CPU cost.
            let g = s.capture(|st| {
                for i in 0..20u64 {
                    let seed = i * 0x1234_5678 + u64::from(t);
                    let mut a = args!(base, rows, seed, out.ptr);
                    unsafe { launch(f, LaunchDims::linear(t, 160), st, false, &mut a)? };
                }
                Ok(())
            })?;
            let us = time_ms(s, 3, 20, |_| g.launch(s))? * 1e3 / 20.0;
            let gbps = f64::from(t) * 16.0 * ROW as f64 / (us * 1e-6) / 1e9;
            res.push((label, us, gbps));
        }
        println!(
            "   T={t:5}: host {:8.2} us ({:6.2} GB/s)   device {:8.2} us ({:7.2} GB/s)",
            res[0].1, res[0].2, res[1].1, res[1].2
        );
        b.record(format!(
            r#"{{"bench":"zerocopy","tokens":{t},"host_gib":{host_gib},"host_us":{:.3},"host_gbps":{:.3},"dev_us":{:.3},"dev_gbps":{:.3}}}"#,
            res[0].1, res[0].2, res[1].1, res[1].2
        ));
    }
    Ok(())
}

/// Fills pinned memory with noise from all cores (a 48 GiB table takes seconds, not minutes).
fn fill_host(buf: &mut [u8]) {
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let chunk = buf.len().div_ceil(threads).next_multiple_of(8);
    std::thread::scope(|sc| {
        for (ci, part) in buf.chunks_mut(chunk).enumerate() {
            sc.spawn(move || {
                let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ ci as u64;
                for w in part.chunks_mut(8) {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    let bytes = x.to_le_bytes();
                    w.copy_from_slice(&bytes[..w.len()]);
                }
            });
        }
    });
}

fn mma(b: &Bench) -> Result<()> {
    let m = Module::load(tsumuji_kernels::BENCH_MMA)?;
    println!("== mma: sustained mma.sync throughput (registers only)");
    let iters: i32 = 20_000;
    let seed: u32 = 0;
    for (name, flop) in [
        ("mma_bf16", 4096.0),
        ("mma_fp8", 8192.0),
        ("mma_nvfp4", 16384.0),
    ] {
        let f = m.function(name)?;
        let mut best: f64 = 0.0;
        for per_sm in [4u32, 8, 16] {
            let grid = b.sms * per_sm;
            let ms = time_ms(&b.stream, 1, 3, |_| {
                let mut a = args!(iters, seed, b.sink.ptr);
                unsafe { launch(f, LaunchDims::linear(grid, 128), &b.stream, false, &mut a) }
            })?;
            let warps = f64::from(grid * 4);
            let tflops = warps * f64::from(iters) * 8.0 * flop / (ms * 1e-3) / 1e12;
            best = best.max(tflops);
        }
        println!("   {name:<10} {best:7.1} TFLOPS (dense)");
        b.record(format!(
            r#"{{"bench":"mma","kernel":"{name}","tflops":{best:.1}}}"#
        ));
    }
    Ok(())
}
