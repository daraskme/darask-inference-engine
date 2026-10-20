//! The GDN kernels against the original one-token-at-a-time gdn_fused_ref: the reworked gdn_fused
//! (docs/m3.md §14) and the column split (gdn_window, §11). Outputs, delayed state, conv history and
//! every record bit.
//! GPU test: scripts/gpu-run.sh -- target/release/deps/gdn_window-* --ignored --nocapture
use tsumuji_cuda::{Device, DeviceBuffer, LaunchDims, Module, Stream, args, launch};

/// The differing elements of two bit patterns, as a message (None when identical).
fn diff(a: &[f32], b: &[f32], label: &str) -> Option<String> {
    assert_eq!(a.len(), b.len());
    let bad: Vec<usize> = (0..a.len())
        .filter(|&i| a[i].to_bits() != b[i].to_bits())
        .collect();
    let show: Vec<String> = bad
        .iter()
        .take(16)
        .map(|&j| format!("{j}:{:08x}/{:08x}", a[j].to_bits(), b[j].to_bits()))
        .collect();
    (!bad.is_empty()).then(|| {
        format!(
            "{label}: {} of {} differ: {}",
            bad.len(),
            a.len(),
            show.join(" ")
        )
    })
}

#[test]
#[ignore = "needs the GPU"]
fn gdn_kernels_match_reference() {
    let _dev = Device::open(0).unwrap();
    let stream = Stream::new().unwrap();
    let module = Module::load(tsumuji_kernels::DECODE_GDN).unwrap();
    let reference = module.function("gdn_fused_ref").unwrap();
    let fused = module.function("gdn_fused").unwrap();
    let window = module.function("gdn_window").unwrap();
    let norm = module.function("gdn_window_norm").unwrap();
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % 65536) as f32 / 32768.0 - 1.0
    };
    // Tiny tests exercise D=32. D=64 and the real D=128 also check independent column CTAs.
    for d in [32usize, 64, 128] {
        let (qk, v) = (2usize, 6usize);
        let c = (2 * qk + v) * d;
        let ld = c + v * d + 2 * v;
        let rs = 8 * c + 8 * v * (2 * d + 1);
        let cw: Vec<f32> = (0..c * 4).map(|_| next() * 0.2).collect();
        let al: Vec<f32> = (0..v).map(|_| next()).collect();
        let db: Vec<f32> = (0..v).map(|_| next()).collect();
        let nw: Vec<f32> = (0..d).map(|_| 1.0 + next() * 0.1).collect();
        let cw = DeviceBuffer::from_slice(&cw).unwrap();
        let al = DeviceBuffer::from_slice(&al).unwrap();
        let db = DeviceBuffer::from_slice(&db).unwrap();
        let nw = DeviceBuffer::from_slice(&nw).unwrap();
        for pdl in [false, true] {
            let initial: Vec<f32> = (0..v * d * d).map(|_| next() * 0.1).collect();
            let initial_conv: Vec<f32> = (0..2 * 3 * c).map(|_| next() * 0.1).collect();
            let mut buffers = Vec::new();
            for _ in 0..3 {
                buffers.push((
                    DeviceBuffer::from_slice(&initial).unwrap(),
                    DeviceBuffer::from_slice(&initial_conv).unwrap(),
                    DeviceBuffer::zeroed(2 * rs * 4).unwrap(),
                    DeviceBuffer::new(8 * v * d * 4).unwrap(),
                ));
            }
            let mut previous_n = 0;
            // Every accepted prefix length 0..8, then changing N (including 1, 3 and 5).
            let windows = [
                (8usize, 0usize),
                (8, 0),
                (8, 1),
                (8, 2),
                (8, 3),
                (8, 4),
                (8, 5),
                (8, 6),
                (8, 7),
                (8, 8),
                (1, 8),
                (2, 1),
                (4, 2),
                (3, 4),
                (5, 3),
                (8, 5),
                (2, 8),
                (5, 0),
                (4, 2),
                (1, 4),
                (8, 1),
                (3, 6),
            ];
            for (step, (n, apply)) in windows.into_iter().enumerate() {
                assert!(apply <= previous_n);
                let slices = [1usize, 4, 8][step % 3];
                let proj: Vec<f32> = (0..slices * n * ld).map(|_| next() * 0.3).collect();
                let proj = DeviceBuffer::from_slice(&proj).unwrap();
                let par = DeviceBuffer::from_slice(&[(step % 2) as i32]).unwrap();
                let na = DeviceBuffer::from_slice(&[apply as i32]).unwrap();
                tsumuji_cuda::synchronize_context().unwrap();
                let (si, li, qi, vi, di, ni, eps) = (
                    slices as i32,
                    ld as i32,
                    qk as i32,
                    v as i32,
                    d as i32,
                    n as i32,
                    1e-6f32,
                );
                // Capture the three implementations with independent states; replay once per step.
                let graph = stream
                    .capture::<tsumuji_cuda::Error>(|stream| {
                        for (kind, (st, cv, rec, y)) in buffers.iter().enumerate() {
                            let mut a = args!(
                                proj.ptr, si, li, cv.ptr, par.ptr, na.ptr, rec.ptr, cw.ptr, al.ptr,
                                db.ptr, st.ptr, nw.ptr, y.ptr, qi, vi, di, eps, ni
                            );
                            let dims = if kind < 2 {
                                LaunchDims {
                                    grid: (v as u32, 1, 1),
                                    block: (4 * d as u32, 1, 1),
                                    smem: (((4 * n + 1) * d
                                        + (1 + kind) * (d / 32) * d
                                        + kind * 16 * d)
                                        * 4) as u32,
                                }
                            } else {
                                LaunchDims {
                                    grid: (v as u32, (d / 32) as u32, 1),
                                    block: (d as u32, 1, 1),
                                    smem: ((2 * n * d + n * 32 + 32 + 2 * d) * 4) as u32,
                                }
                            };
                            unsafe {
                                launch([reference, fused, window][kind], dims, stream, pdl, &mut a)?
                            };
                            if kind == 2 {
                                let mut a =
                                    args!(y.ptr, proj.ptr, si, li, nw.ptr, qi, vi, di, eps, ni);
                                let dims = LaunchDims {
                                    grid: (v as u32, n as u32, 1),
                                    block: (d as u32, 1, 1),
                                    smem: 0,
                                };
                                unsafe { launch(norm, dims, stream, pdl, &mut a)? };
                            }
                        }
                        Ok(())
                    })
                    .unwrap();
                graph.launch(&stream).unwrap();
                stream.synchronize().unwrap();
                let label = format!("D{d} N{n} S{slices} apply{apply} PDL{pdl} step{step}");
                let a = &buffers[0];
                let mut errs = Vec::new();
                for (b, name) in [(&buffers[1], "gdn_fused"), (&buffers[2], "gdn_window")] {
                    errs.extend(
                        [
                            (&a.0, &b.0, v * d * d, "state"),
                            (&a.1, &b.1, 6 * c, "conv"),
                            (&a.2, &b.2, 2 * rs, "records"),
                            (&a.3, &b.3, n * v * d, "output"),
                        ]
                        .into_iter()
                        .filter_map(|(x, y, len, kind)| {
                            diff(
                                &x.to_vec::<f32>(len).unwrap(),
                                &y.to_vec(len).unwrap(),
                                &format!("{name} {label} {kind}"),
                            )
                        }),
                    );
                }
                assert!(errs.is_empty(), "{}", errs.join("\n"));
                previous_n = n;
            }
        }
    }
}
