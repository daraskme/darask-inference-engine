//! Expert union vs the original per-token kernels, including intermediate gate/up results.
//! GPU test: scripts/gpu-run.sh -- target/release/deps/moe_union-* --ignored --nocapture
use tsumuji_cuda::{Device, DeviceBuffer, LaunchDims, Module, Stream, args, launch};

fn same(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    if let Some(i) = a
        .iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
    {
        panic!(
            "{label}: element {i}: {} ({:08x}) vs {} ({:08x})",
            a[i],
            a[i].to_bits(),
            b[i],
            b[i].to_bits()
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn expert_union_is_bit_identical() {
    let _dev = Device::open(0).unwrap();
    let stream = Stream::new().unwrap();
    let moe = Module::load(tsumuji_kernels::DECODE_GEMV).unwrap();
    let quant = Module::load(tsumuji_kernels::DECODE_QUANT).unwrap();
    let (e, topk, nmax) = (32usize, 4usize, 8usize);
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    // Real shapes, shorter rows, and a down row with a nonempty nvfp4_dot_regs<5> tail.
    for (h, f) in [(2560usize, 640usize), (256, 96), (512, 768)] {
        let fh = f * h;
        let stride = fh * 27 / 16;
        let mut packed = vec![0u8; e * stride];
        for expert in packed.chunks_mut(stride) {
            for b in &mut expert[..fh * 3 / 2] {
                *b = next() as u8;
            }
            for b in &mut expert[fh * 3 / 2..] {
                *b = 0x30 + (next() % 16) as u8; // finite, positive e4m3 scales
            }
        }
        let experts = DeviceBuffer::from_slice(&packed).unwrap();
        let gs: Vec<f32> = (0..e * 3)
            .map(|_| 0.1 + (next() % 1000) as f32 / 1000.0)
            .collect();
        let gs = DeviceBuffer::from_slice(&gs).unwrap();
        let x: Vec<f32> = (0..nmax * h)
            .map(|_| (next() % 65536) as f32 / 32768.0 - 1.0)
            .collect();
        let x = DeviceBuffer::from_slice(&x).unwrap();
        let sw: Vec<u16> = (0..3 * fh)
            .map(|_| {
                let v = (next() % 65536) as f32 / 32768.0 - 1.0;
                let b = v.to_bits();
                ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
            })
            .collect();
        let gu = DeviceBuffer::from_slice(&sw[..2 * fh]).unwrap();
        let dn = DeviceBuffer::from_slice(&sw[2 * fh..]).unwrap();
        for (fmt, suffix) in [(0, ""), (1, "_fp8"), (2, "_i8")] {
            let (guq, gus, dnq, dns) = if fmt == 0 {
                (None, None, None, None)
            } else {
                let (qg, sg, qd, sd) = (
                    DeviceBuffer::new(2 * fh).unwrap(),
                    DeviceBuffer::new(2 * fh / 32).unwrap(),
                    DeviceBuffer::new(fh).unwrap(),
                    DeviceBuffer::new(fh / 32).unwrap(),
                );
                let quant_fn = quant
                    .function(if fmt == 1 {
                        "quant_mxfp8"
                    } else {
                        "quant_mxint8"
                    })
                    .unwrap();
                tsumuji_cuda::synchronize_context().unwrap();
                for (w, q, s, len) in [(&gu, &qg, &sg, 2 * fh), (&dn, &qd, &sd, fh)] {
                    let blocks = (len / 32) as u64;
                    let mut a = args!(w.ptr, q.ptr, s.ptr, blocks);
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
                }
                stream.synchronize().unwrap();
                (Some(qg), Some(sg), Some(qd), Some(sd))
            };
            let (guw, dnw) = (
                guq.as_ref().unwrap_or(&gu).ptr,
                dnq.as_ref().unwrap_or(&dn).ptr,
            );
            let (gsc, dsc) = (
                gus.as_ref().map_or(0, |b| b.ptr),
                dns.as_ref().map_or(0, |b| b.ptr),
            );
            for pattern in 0..3 {
                // Identical sets in different slot order, partial overlap, and no overlap.
                let ids: Vec<i32> = (0..nmax * topk)
                    .map(|i| {
                        let (n, s) = (i / topk, i % topk);
                        (match pattern {
                            0 => (s + n) % topk,
                            1 => (2 * n + s) % e,
                            _ => n * topk + s,
                        }) as i32
                    })
                    .collect();
                let sel = DeviceBuffer::from_slice(&ids).unwrap();
                let (hb, yb, hu, yu) = (
                    DeviceBuffer::new(nmax * (topk + 1) * f * 4).unwrap(),
                    DeviceBuffer::new(nmax * (topk + 1) * h * 4).unwrap(),
                    DeviceBuffer::new(nmax * (topk + 1) * f * 4).unwrap(),
                    DeviceBuffer::new(nmax * (topk + 1) * h * 4).unwrap(),
                );
                tsumuji_cuda::synchronize_context().unwrap();
                for (nt, n) in [
                    (2usize, 2usize),
                    (4, 3),
                    (4, 4),
                    (8, 5),
                    (8, 6),
                    (8, 7),
                    (8, 8),
                ] {
                    let poison_h = vec![f32::NAN; n * (topk + 1) * f];
                    let poison_y = vec![f32::NAN; n * (topk + 1) * h];
                    for pdl in [false, true] {
                        for (union, oh, oy) in [(false, &hb, &yb), (true, &hu, &yu)] {
                            // NaN poison exposes slots not written by the union (even on reuse).
                            oh.upload_on(&poison_h, &stream).unwrap();
                            oy.upload_on(&poison_y, &stream).unwrap();
                            let tail = if union {
                                format!("_union_n{nt}")
                            } else {
                                String::new()
                            };
                            let kg = moe.function(&format!("moe_gate_up{suffix}{tail}")).unwrap();
                            let kd = moe.function(&format!("moe_down{suffix}{tail}")).unwrap();
                            if union {
                                kg.set_max_dynamic_smem(96 * 1024).unwrap();
                                kd.set_max_dynamic_smem(96 * 1024).unwrap();
                            }
                            let cols = if union { nt } else { 1 };
                            let (hi, fi, ti) = (h as i32, f as i32, topk as i32);
                            let mut a = args!(
                                experts.ptr,
                                gs.ptr,
                                sel.ptr,
                                guw,
                                gsc,
                                x.ptr,
                                oh.ptr,
                                hi,
                                fi,
                                ti
                            );
                            let dims = LaunchDims {
                                grid: (f.div_ceil(16) as u32, (topk + 1) as u32, n as u32),
                                block: (256, 1, 1),
                                smem: (cols * h / 32 * 33 * 4) as u32,
                            };
                            unsafe { launch(kg, dims, &stream, pdl, &mut a).unwrap() };
                            let mut a = args!(
                                experts.ptr,
                                gs.ptr,
                                sel.ptr,
                                dnw,
                                dsc,
                                oh.ptr,
                                oy.ptr,
                                hi,
                                fi,
                                ti
                            );
                            let dims = LaunchDims {
                                grid: (h.div_ceil(64) as u32, (topk + 1) as u32, n as u32),
                                block: (256, 1, 1),
                                smem: (cols * f / 32 * 33 * 4) as u32,
                            };
                            unsafe { launch(kd, dims, &stream, pdl, &mut a).unwrap() };
                        }
                        stream.synchronize().unwrap();
                        let label =
                            format!("H{h} F{f} fmt{fmt} pattern{pattern} NT{nt} N{n} PDL{pdl}");
                        same(
                            &hb.to_vec::<f32>(n * (topk + 1) * f).unwrap(),
                            &hu.to_vec(n * (topk + 1) * f).unwrap(),
                            &label,
                        );
                        same(
                            &yb.to_vec::<f32>(n * (topk + 1) * h).unwrap(),
                            &yu.to_vec(n * (topk + 1) * h).unwrap(),
                            &label,
                        );
                    }
                }
            }
        }
    }
}
