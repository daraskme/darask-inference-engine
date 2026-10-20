//! MXFP8 / MXINT8 conversion (kernels/decode/quant.cu) and the decode GEMV against an f64 CPU
//! reference. GPU test: `scripts/gpu-run.sh -- target/release/deps/mxfp8-* --ignored --nocapture`.

use tsumuji_cuda::{Device, DeviceBuffer, LaunchDims, Module, Stream, args, launch};

/// Mirrors `Stage` in kernels/decode/gemv2.cu.
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

fn e4m3(code: u8) -> f64 {
    let (s, e, m) = (code >> 7, i32::from((code >> 3) & 15), f64::from(code & 7));
    let v = if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    if s != 0 { -v } else { v }
}

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[test]
#[ignore = "needs the GPU"]
fn mx8_quant_and_gemv() {
    for (name, quant_fn, gemv_prefix, top, err_bound) in [
        ("MXFP8", "quant_mxfp8", "gemv_fp8", 448.0, 0.0626),
        (
            "MXINT8",
            "quant_mxint8",
            "gemv_i8",
            127.0,
            0.5 / 63.5 + 1e-9,
        ),
    ] {
        check_format(name, quant_fn, gemv_prefix, top, err_bound);
    }
}

fn check_format(name: &str, quant_fn: &str, gemv_prefix: &str, top: f64, err_bound: f64) {
    let decode = |code: u8| {
        if top == 448.0 {
            e4m3(code)
        } else {
            f64::from(code as i8)
        }
    };
    let _dev = Device::open(0).unwrap();
    let stream = Stream::new().unwrap();
    let (o, k) = (1000usize, 2560usize);
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    // Rows with very different magnitudes, some blocks all zero.
    let w: Vec<u16> = (0..o * k)
        .map(|i| {
            let (row, col) = (i / k, i % k);
            if row % 97 == 0 && col < 64 {
                0
            } else {
                bf16((next() * 10f64.powi((row % 7) as i32 - 3)) as f32)
            }
        })
        .collect();

    let quant = Module::load(tsumuji_kernels::DECODE_QUANT).unwrap();
    let gemv = Module::load(tsumuji_kernels::DECODE_GEMV2).unwrap();
    let (wd, q, sc) = (
        DeviceBuffer::from_slice(&w).unwrap(),
        DeviceBuffer::new(o * k).unwrap(),
        DeviceBuffer::new(o * k / 32).unwrap(),
    );
    tsumuji_cuda::synchronize_context().unwrap(); // the upload above ran on the legacy stream
    let blocks = (o * k / 32) as u64;
    let mut a = args!(wd.ptr, q.ptr, sc.ptr, blocks);
    unsafe {
        launch(
            quant.function(quant_fn).unwrap(),
            LaunchDims::linear(blocks.div_ceil(256) as u32, 256),
            &stream,
            false,
            &mut a,
        )
        .unwrap()
    };
    stream.synchronize().unwrap();
    let (qh, sh): (Vec<u8>, Vec<u8>) = (q.to_vec(o * k).unwrap(), sc.to_vec(o * k / 32).unwrap());

    // Every block: the scale keeps max |w| in (224, 448] and each value is the nearest e4m3.
    let mut worst_rel = 0f64;
    for b in 0..o * k / 32 {
        let scale = 2f64.powi(i32::from(sh[b]) - 127);
        let vals: Vec<f64> = (0..32)
            .map(|i| f64::from(f32::from_bits(u32::from(w[b * 32 + i]) << 16)))
            .collect();
        let amax = vals.iter().fold(0f64, |a, v| a.max(v.abs()));
        if amax == 0.0 {
            continue;
        }
        assert!(
            amax / scale <= top && amax / scale > top / 2.0,
            "{name} block {b}: amax/scale {}",
            amax / scale
        );
        for i in 0..32 {
            let got = decode(qh[b * 32 + i]) * scale;
            let rel = (got - vals[i]).abs() / amax;
            worst_rel = worst_rel.max(rel);
        }
    }
    // e4m3: rounding error <= 2^-4 of a value (<= 2^-4 * 448 / 224 of amax); int8: <= scale / 2.
    assert!(
        worst_rel <= err_bound,
        "{name} quantization error {worst_rel}"
    );

    // GEMV (every compiled variant, with and without split-K, 1..8 tokens) vs f64 on the
    // dequantized weights; a token's partial sums must be bit-identical for every token count.
    let nmax = 8usize;
    let xs: Vec<f32> = (0..nmax * k).map(|_| next() as f32).collect();
    let deq = |r: usize, c: usize| {
        decode(qh[r * k + c]) * 2f64.powi(i32::from(sh[(r * k + c) / 32]) - 127)
    };
    let want: Vec<f64> = (0..nmax * o)
        .map(|i| {
            (0..k)
                .map(|c| deq(i % o, c) * f64::from(xs[i / o * k + c]))
                .sum()
        })
        .collect();
    let mag: Vec<f64> = (0..nmax * o)
        .map(|i| {
            (0..k)
                .map(|c| (deq(i % o, c) * f64::from(xs[i / o * k + c])).abs())
                .sum()
        })
        .collect();
    let xd = DeviceBuffer::from_slice(&xs).unwrap();
    let y = DeviceBuffer::zeroed(8 * nmax * o * 4).unwrap();
    let mut reference: Vec<Option<Vec<Vec<f32>>>> = vec![None; 3];
    for (r, p) in [(1usize, 4usize), (1, 8), (2, 8), (4, 2)] {
        for (si, sl) in [1usize, 2, 8].into_iter().enumerate() {
            let per = (k / 16).div_ceil(sl);
            let run = |nt: usize, n: usize, x0: usize| -> Vec<f32> {
                let f = gemv
                    .function(&format!("{gemv_prefix}_r{r}_p{p}_n{nt}"))
                    .unwrap();
                f.set_max_dynamic_smem(96 * 1024).unwrap();
                let stage = Stage {
                    x: xd.ptr + (x0 * k * 4) as u64,
                    ss: 0,
                    w: 0,
                    n: 0,
                    h: 0,
                    hc: 0,
                    eps: 0.0,
                };
                let (oi, ki, ni) = (o as i32, k as i32, n as i32);
                let mut a = args!(q.ptr, sc.ptr, stage, y.ptr, oi, ki, ni);
                let dims = LaunchDims {
                    grid: (o.div_ceil(8 * r) as u32, sl as u32, 1),
                    block: (256, 1, 1),
                    smem: (nt * per * 16 * 4) as u32,
                };
                unsafe { launch(f, dims, &stream, false, &mut a).unwrap() };
                stream.synchronize().unwrap();
                y.to_vec(sl * n * o).unwrap()
            };
            // Baseline: each token alone. parts [s][n][o].
            let single: Vec<Vec<f32>> = (0..nmax).map(|t| run(1, 1, t)).collect();
            if let Some(want) = &reference[si] {
                assert!(
                    single
                        .iter()
                        .flatten()
                        .zip(want.iter().flatten())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "R{r} P{p} S{sl}: changing R/P changed a token's bits"
                );
            } else {
                reference[si] = Some(single.clone());
            }
            for t in 0..nmax {
                for row in 0..o {
                    let got: f64 = (0..sl).map(|s| f64::from(single[t][s * o + row])).sum();
                    let i = t * o + row;
                    assert!(
                        (got - want[i]).abs() <= 1e-5 * mag[i] + 1e-30,
                        "R{r} P{p} S{sl} token {t} row {row}: {got} vs {}",
                        want[i]
                    );
                }
            }
            for (nt, n) in [(2usize, 2usize), (4, 3), (4, 4), (8, 5), (8, 8)] {
                let parts = run(nt, n, 0);
                for t in 0..n {
                    for s in 0..sl {
                        let (a, b) = (&parts[(s * n + t) * o..][..o], &single[t][s * o..][..o]);
                        assert!(
                            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                            "R{r} P{p} S{sl} NT{nt} N{n}: token {t} differs from a single-token pass"
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "{name}: worst block-relative quantization error {worst_rel:.4}; GEMV matches f64 for all variants, bit-identical for 1..8 tokens"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn gr_up_mix_token_capacities() {
    let _dev = Device::open(0).unwrap();
    let stream = Stream::new().unwrap();
    let quant = Module::load(tsumuji_kernels::DECODE_QUANT).unwrap();
    let gemv = Module::load(tsumuji_kernels::DECODE_GEMV2).unwrap();
    let (h, nm, slices, stats) = (64usize, 8usize, 3usize, 3usize);
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    // Covers a partially used second BF16 vector per lane and all 64 vectors, plus padding of
    // the scatter reduction when fewer than four streams or NT tokens are active.
    for rank in [320usize, 512usize] {
        for hc in [1usize, 2usize, 4usize] {
            let ld = rank + hc;
            let up: Vec<u16> = (0..hc * h * rank)
                .map(|_| bf16(next() as f32 * 0.1))
                .collect();
            let up = DeviceBuffer::from_slice(&up).unwrap();
            let di: Vec<f32> = (0..slices * nm * ld).map(|_| next() as f32).collect();
            let r: Vec<f32> = (0..nm * hc * h).map(|_| next() as f32).collect();
            let ss: Vec<f32> = (0..nm * stats * hc)
                .map(|_| next().abs() as f32 + 0.1)
                .collect();
            let w: Vec<f32> = (0..hc * h).map(|_| next() as f32 * 0.1).collect();
            let (r, ss, w) = (
                DeviceBuffer::from_slice(&r).unwrap(),
                DeviceBuffer::from_slice(&ss).unwrap(),
                DeviceBuffer::from_slice(&w).unwrap(),
            );
            let y = DeviceBuffer::new(nm * h * 4).unwrap();
            for (fmt, prefix) in [(0, "gr_up_mix"), (1, "gr_up_mix_fp8"), (2, "gr_up_mix_i8")] {
                let (q, sc) = (
                    DeviceBuffer::new(hc * h * rank).unwrap(),
                    DeviceBuffer::new(hc * h * rank / 32).unwrap(),
                );
                tsumuji_cuda::synchronize_context().unwrap();
                if fmt != 0 {
                    let blocks = (hc * h * rank / 32) as u64;
                    let mut a = args!(up.ptr, q.ptr, sc.ptr, blocks);
                    let quant_fn = quant
                        .function(if fmt == 1 {
                            "quant_mxfp8"
                        } else {
                            "quant_mxint8"
                        })
                        .unwrap();
                    unsafe {
                        launch(
                            quant_fn,
                            LaunchDims::linear(blocks.div_ceil(256) as u32, 256),
                            &stream,
                            false,
                            &mut a,
                        )
                        .unwrap()
                    };
                    stream.synchronize().unwrap();
                }
                let (weights, scales) = if fmt == 0 {
                    (up.ptr, 0u64)
                } else {
                    (q.ptr, sc.ptr)
                };
                let run = |nt: usize, n: usize, token: usize, parts: &[f32]| {
                    let parts = DeviceBuffer::from_slice(parts).unwrap();
                    tsumuji_cuda::synchronize_context().unwrap();
                    let (hi, ci, ri, si, li, pi, ni, eps) = (
                        h as i32,
                        hc as i32,
                        rank as i32,
                        slices as i32,
                        ld as i32,
                        stats as i32,
                        n as i32,
                        1e-6f32,
                    );
                    let (rp, sp) = (
                        r.ptr + (token * hc * h * 4) as u64,
                        ss.ptr + (token * stats * hc * 4) as u64,
                    );
                    let mut a = args!(
                        weights, scales, parts.ptr, si, li, rp, sp, pi, w.ptr, y.ptr, hi, ci, ri,
                        eps, ni
                    );
                    let dims = LaunchDims {
                        grid: (h.div_ceil(8) as u32, 1, 1),
                        block: (256, 1, 1),
                        smem: (nt * rank * 4) as u32,
                    };
                    unsafe {
                        launch(
                            gemv.function(&format!("{prefix}_n{nt}")).unwrap(),
                            dims,
                            &stream,
                            false,
                            &mut a,
                        )
                        .unwrap()
                    };
                    stream.synchronize().unwrap();
                    y.to_vec::<f32>(n * h).unwrap()
                };
                let single: Vec<Vec<f32>> = (0..nm)
                    .map(|t| {
                        let parts: Vec<f32> = (0..slices)
                            .flat_map(|s| di[(s * nm + t) * ld..][..ld].iter().copied())
                            .collect();
                        run(1, 1, t, &parts)
                    })
                    .collect();
                for n in 2..=nm {
                    let nt = n.next_power_of_two();
                    let parts: Vec<f32> = (0..slices)
                        .flat_map(|s| di[s * nm * ld..][..n * ld].iter().copied())
                        .collect();
                    let got = run(nt, n, 0, &parts);
                    assert!(
                        got.iter()
                            .zip(single.iter().take(n).flatten())
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{prefix} rank{rank} HC{hc} NT{nt} N{n}: GR mix differs from one-token passes"
                    );
                }
            }
        }
    }
}
