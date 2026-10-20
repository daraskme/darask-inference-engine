//! Decode GEMV shapes of the real model: bandwidth of each kernel variant (rows per warp R,
//! prefetched vectors P, K slices S) in a graph of dependent PDL launches over distinct matrices.

use crate::{Bench, GIB};
use tsumuji_cuda::{DeviceBuffer, LaunchDims, Module, Result, Stream, args, launch, time_ms};

const SHAPES: &[(&str, usize, usize)] = &[
    ("gdn_in", 16480, 2560),
    ("gdn_out", 2560, 6144),
    ("qsa_proj", 13952, 2560),
    ("qsa_o", 2560, 6144),
    ("gr_down", 324, 10240),
    ("gr_up", 10240, 320),
    ("router", 513, 2560),
    ("shared_gu", 1280, 2560),
    ("shared_dn", 2560, 640),
    ("ple_kv", 12800, 2560),
    ("lm_head", 248_320, 2560),
];

const VARIANTS: &[(usize, usize)] = &[(1, 4), (1, 8), (2, 8), (4, 2)];

/// Mirrors `Stage` in kernels/decode/gemv2.cu (plain input: only `x` is used).
#[repr(C)]
#[derive(Clone, Copy)]
struct Stage {
    x: u64,
    ss: u64,
    w: u64,
    n: i32,
    h: i32,
    hc: i32,
    eps: f32,
}

/// `n` tokens per pass (1, 2, 4 or 8: the compiled capacities).
pub fn run(b: &Bench, int8: bool, n: usize) -> Result<()> {
    let m1 = Module::load(tsumuji_kernels::DECODE_GEMV)?;
    // int8: MXINT8 weights (1 byte per element + 1/32 scales), kernels gemv_i8_*.
    let (prefix, esize) = if int8 {
        ("gemv_i8", 1usize)
    } else {
        ("gemv_bf16", 2usize)
    };
    let vw = 16 / esize;
    let m2 = Module::load(tsumuji_kernels::DECODE_GEMV2)?;
    let old = m1.function("gemv_bf16")?;
    let reduce = m2.function("gemv_reduce")?;
    let peak = b
        .read_bw(
            Module::load(tsumuji_kernels::BENCH_BW)?.function("bw_read_cs")?,
            &DeviceBuffer::new(2 * GIB)?,
            5,
        )?
        .0;
    let x = DeviceBuffer::zeroed(n * 16384 * 4)?;
    let s = &b.stream;
    println!(
        "== gemv-shapes: {prefix} decode GEMVs, {n} token(s), graph of PDL launches, % of stream peak {:.0} GB/s",
        peak / 1e9
    );
    println!(
        "   {:<10} {:>14} {:>9} {:>9}  best",
        "shape", "O x K", "M1 old", "best new"
    );
    for &(name, o, k) in SHAPES {
        let mat = o * k * esize + if int8 { o * k / 32 } else { 0 };
        let copies = ((1usize << 30) / mat).clamp(1, 512);
        let pool = DeviceBuffer::new(copies * mat)?;
        b.fill_random(&pool, 9)?;
        let part = DeviceBuffer::zeroed(16 * n * o * 4)?;
        let y = DeviceBuffer::zeroed(n * o * 4)?;
        let (oi, ki, ni, noi) = (o as i32, k as i32, n as i32, (n * o) as i32);
        let time = |f: &dyn Fn(&Stream, u64) -> Result<()>| -> Result<f64> {
            let g =
                s.capture(|st| (0..copies).try_for_each(|i| f(st, pool.ptr + (i * mat) as u64)))?;
            Ok(time_ms(s, 2, 5, |_| g.launch(s))? * 1e3 / copies as f64)
        };
        // The M1 kernel reads bf16; it has no int8 form.
        let us_old = if int8 || n > 1 {
            f64::NAN
        } else {
            time(&|st, w| {
                let mut a = args!(w, x.ptr, y.ptr, oi, ki);
                unsafe {
                    launch(
                        old,
                        LaunchDims {
                            grid: ((o as u32).div_ceil(8), 1, 1),
                            block: (256, 1, 1),
                            smem: (k * 4) as u32,
                        },
                        st,
                        false,
                        &mut a,
                    )
                }
            })?
        };
        let mut best = (f64::MAX, String::new());
        for &(r, p) in VARIANTS {
            let f = m2.function(&format!("{prefix}_r{r}_p{p}_n{n}"))?;
            f.set_max_dynamic_smem(96 * 1024)?;
            for sl in [1usize, 2, 4, 8, 16] {
                if sl > 1 && (k / vw) / sl < 8 {
                    continue; // slices too short to be useful
                }
                let rows_per_block = 8 * r;
                let per = (k / vw).div_ceil(sl);
                let dims = LaunchDims {
                    grid: ((o as u32).div_ceil(rows_per_block as u32), sl as u32, 1),
                    block: (256, 1, 1),
                    smem: (n * per * vw * 4) as u32,
                };
                let (si, out) = (sl as i32, if sl == 1 { y.ptr } else { part.ptr });
                let stage = Stage {
                    x: x.ptr,
                    ss: 0,
                    w: 0,
                    n: 0,
                    h: 0,
                    hc: 0,
                    eps: 0.0,
                };
                let us = time(&|st, w| {
                    let sc = if int8 { w + (o * k) as u64 } else { 0u64 };
                    let mut a = args!(w, sc, stage, out, oi, ki, ni);
                    unsafe { launch(f, dims, st, true, &mut a)? };
                    if sl > 1 {
                        let mut a = args!(part.ptr, y.ptr, noi, si);
                        unsafe {
                            launch(
                                reduce,
                                LaunchDims::linear(((n * o) as u32).div_ceil(256).min(1024), 256),
                                st,
                                true,
                                &mut a,
                            )?
                        };
                    }
                    Ok(())
                })?;
                if us < best.0 {
                    best = (us, format!("R{r} P{p} S{sl}"));
                }
            }
        }
        let pct = |us: f64| 100.0 * mat as f64 / (us * 1e-6) / peak;
        println!(
            "   {name:<10} {:>14} {:>8.1}% {:>8.1}%  {} ({:.2} us)",
            format!("{o}x{k}"),
            pct(us_old),
            pct(best.0),
            best.1,
            best.0
        );
        b.record(format!(r#"{{"bench":"gemv_shape","shape":"{name}","fmt":"{prefix}","tokens":{n},"old_pct":{:.1},"new_pct":{:.1},"best":"{}","us":{:.3}}}"#, pct(us_old), pct(best.0), best.1, best.0));
    }
    Ok(())
}
