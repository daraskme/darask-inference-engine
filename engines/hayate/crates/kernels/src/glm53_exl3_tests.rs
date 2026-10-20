//! EXL3 device kernels (cu/glm53/exl3.cu) against the CPU path (`glm53::exl3`), checkpoint-free: any trellis bit
//! pattern is a valid encoding, so the matrices are random words with FP16 suh/svh around one.  The GPU rounds the
//! rotated input and each weight to FP16 (the CPU keeps F32), so the comparisons allow that rounding.
#[cfg(feature = "cuda")]
mod gpu {
    use crate::glm53::exl3::{Exl3, block, had128};
    use crate::glm53::{cpu::bf16, cpu::swiglu, exl3_ffi as x3, ffi as g};
    use crate::runtime::{Device, DeviceBuffer, Stream, check};
    use hayate_core::{QuantSpec, Result};
    use std::ffi::c_void;

    const H: usize = 4096;
    const I: usize = 2048;

    fn random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }
    /// One [n, k] matrix image: random trellis words, suh/svh of either sign in [1/2, 2).
    fn matrix(n: usize, k: usize, k2: u8, seed: &mut u64) -> Vec<u8> {
        let mut raw: Vec<u8> = (0..n * k * usize::from(k2) / 16)
            .map(|_| random(seed) as u8)
            .collect();
        for _ in 0..n + k {
            let h = (0x3800u16 + (random(seed) % 0x800) as u16)
                | if random(seed) & 1 == 1 { 0x8000 } else { 0 };
            raw.extend(h.to_le_bytes());
        }
        assert_eq!(
            QuantSpec::Exl3 { k2, cb: 2 }.matrix_bytes(n as u64, k as u64),
            Some(raw.len() as u64)
        );
        raw
    }
    fn expert(gk: u8, dk: u8, seed: &mut u64) -> Vec<u8> {
        let mut image = matrix(I, H, gk, seed);
        image.extend(matrix(I, H, gk, seed));
        image.extend(matrix(H, I, dk, seed));
        image
    }
    fn parts(image: &[u8], gk: u8, dk: u8) -> [Exl3<'_>; 3] {
        let gb = QuantSpec::Exl3 { k2: gk, cb: 2 }
            .matrix_bytes(I as u64, H as u64)
            .unwrap() as usize;
        [
            Exl3::new(&image[..gb], I, H, gk, 2).unwrap(),
            Exl3::new(&image[gb..2 * gb], I, H, gk, 2).unwrap(),
            Exl3::new(&image[2 * gb..], H, I, dk, 2).unwrap(),
        ]
    }
    /// The CPU path's y = H(H(x * suh) W') * svh for one row.
    fn cpu_mv(m: &Exl3<'_>, x: &[f32]) -> Vec<f32> {
        let mut xh = vec![0.; m.k];
        let sum = m.prepare(x, &mut xh);
        let mut y = vec![0.; m.n];
        for c0 in (0..m.n).step_by(128) {
            let mut out = [[0.; 128]; 1];
            unsafe { block::<1>(m, [xh.as_ptr()], [sum], c0, &mut out) };
            y[c0..c0 + 128].copy_from_slice(&out[0]);
        }
        y
    }
    fn bits(v: f32) -> u16 {
        (bf16(v).to_bits() >> 16) as u16
    }
    fn value(b: u16) -> f32 {
        f32::from_bits(u32::from(b) << 16)
    }
    fn upload(device: &Device, stream: &Stream, bytes: &[u8]) -> Result<DeviceBuffer> {
        let mut out = device.alloc(bytes.len())?;
        stream.copy_h2d(&mut out, bytes)?;
        Ok(out)
    }
    fn download(stream: &Stream, buffer: &DeviceBuffer) -> Result<Vec<u16>> {
        let mut out = vec![0; buffer.bytes()];
        stream.copy_d2h(&mut out, buffer)?;
        Ok(out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect())
    }
    fn to_bytes(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|b| b.to_le_bytes()).collect()
    }
    /// Rows agree to the FP16 rounding of the GPU's inputs and weights and the BF16 outputs.
    fn close(got: &[f32], want: &[f32], what: &str) {
        let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
        let (mut err, mut energy) = (0f64, 0f64);
        for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
            assert!(a.is_finite(), "{what}: {i} not finite");
            assert!(
                (a - b).abs() <= 0.03 * scale + 0.02 * b.abs(),
                "{what}: {i}: {a} vs {b} (row max {scale})"
            );
            err += f64::from(a - b).powi(2);
            energy += f64::from(b).powi(2);
        }
        assert!(
            (err / energy.max(1e-30)).sqrt() < 1e-2,
            "{what}: relative L2 {}",
            (err / energy).sqrt()
        );
    }

    #[test]
    #[ignore = "requires sm_120 GPU; device decode's prefetch copy: completion tags, and a stale copy (target layer routed) sets none"]
    fn glm53_dev_prefetch_tags() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let bytes = 1 << 20;
        let mut host = Vec::new();
        for e in 0..2u8 {
            let mut p = device.alloc_pinned(bytes)?;
            p.as_mut_slice()
                .iter_mut()
                .enumerate()
                .for_each(|(i, b)| *b = (i as u8).wrapping_mul(13) ^ (e + 1));
            host.push(p);
        }
        let slots: Vec<DeviceBuffer> = (0..2)
            .map(|_| upload(&device, &stream, &vec![0u8; bytes]))
            .collect::<Result<_>>()?;
        let table = |p: &[u64]| -> Vec<u8> {
            (0..8)
                .flat_map(|i| p.get(i).copied().unwrap_or(0).to_le_bytes())
                .collect()
        };
        let dst = upload(
            &device,
            &stream,
            &table(&slots.iter().map(|b| b.as_ptr() as u64).collect::<Vec<_>>()),
        )?;
        let src = upload(
            &device,
            &stream,
            &table(&host.iter().map(|h| h.as_ptr() as u64).collect::<Vec<_>>()),
        )?;
        let mut done = upload(&device, &stream, &[0u8; 32])?;
        let mut bad = upload(&device, &stream, &[0u8; 32])?;
        let mut tags = upload(&device, &stream, &[0u8; 64])?;
        let mut progress = upload(&device, &stream, &[0u8; 8])?;
        let mut ctl = upload(&device, &stream, &[0u8; 20])?;
        let (done_p, bad_p, tags_p) = (done.as_mut_ptr(), bad.as_mut_ptr(), tags.as_mut_ptr());
        let run = |ctl: &mut DeviceBuffer,
                   progress: &mut DeviceBuffer,
                   seq: u32,
                   routed: u64|
         -> Result<Vec<u64>> {
            let mut c = [0u8; 20];
            c[8..12].copy_from_slice(&seq.to_le_bytes());
            stream.copy_h2d(ctl, &c)?;
            stream.copy_h2d(progress, &routed.to_le_bytes())?;
            check(
                unsafe {
                    g::hayate_glm_dev_prefetch(
                        dst.as_ptr(),
                        src.as_ptr(),
                        bytes,
                        8,
                        done_p,
                        bad_p,
                        tags_p,
                        progress.as_ptr(),
                        ctl.as_ptr(),
                        10,
                        stream.as_raw(),
                    )
                },
                "prefetch",
            )?;
            let mut t = [0u8; 64];
            stream.copy_d2h(&mut t, &tags)?;
            Ok(t.as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b))
                .collect())
        };
        // Layer 10 of token 5, the router at layer 9: both slots copied and tagged, the unused six untouched.
        let t = run(&mut ctl, &mut progress, 5, 5 << 8 | 9)?;
        assert_eq!(t, [5 << 8 | 10, 5 << 8 | 10, 0, 0, 0, 0, 0, 0]);
        for (slot, h) in slots.iter().zip(&host) {
            let mut back = vec![0u8; bytes];
            stream.copy_d2h(&mut back, slot)?;
            assert!(back == h.as_slice(), "prefetched bytes");
        }
        // Token 6 with layer 10 already routed: stale, the tags keep token 5's, the counters are back at zero.
        let t = run(&mut ctl, &mut progress, 6, 6 << 8 | 10)?;
        assert_eq!(t[..2], [5 << 8 | 10, 5 << 8 | 10]);
        let (mut d, mut b) = ([0u8; 32], [0u8; 32]);
        stream.copy_d2h(&mut d, &done)?;
        stream.copy_d2h(&mut b, &bad)?;
        assert!(d == [0; 32] && b == [0; 32], "done / bad reset");
        // Token 6 before the router reaches layer 10: tagged again.
        let t = run(&mut ctl, &mut progress, 6, 6 << 8 | 9)?;
        assert_eq!(t[..2], [6 << 8 | 10, 6 << 8 | 10]);
        Ok(())
    }

    #[test]
    #[ignore = "requires sm_120 GPU; EXL3 restore (BF16 rows) against the F64 full weight, K 2..4"]
    fn glm53_exl3_gpu_restore() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut seed = 0x51ed_270b;
        for (k2, n, k) in [
            (5u8, 256, 384),
            (7, 128, 256),
            (4, 256, 128),
            (6, 128, 128),
            (8, 128, 256),
            (5, I, H),
        ] {
            let raw = matrix(n, k, k2, &mut seed);
            let m = Exl3::new(&raw, n, k, k2, 2).unwrap();
            let image = upload(&device, &stream, &raw)?;
            let mut out = device.alloc(n * k * 2)?;
            check(
                unsafe {
                    x3::hayate_glm_exl3_restore(
                        out.as_mut_ptr(),
                        image.as_ptr(),
                        n as u32,
                        k as u32,
                        u32::from(k2),
                        stream.as_raw(),
                    )
                },
                "EXL3 restore",
            )?;
            let got = download(&stream, &out)?;
            // F64 W = diag(suh) H W' H diag(svh) from the exactly decoded W' (rows: outputs)
            let inner = m.inner();
            let mut w = vec![0f32; k * n]; // [k][n]
            for c in 0..n {
                let mut col: Vec<f32> = (0..k).map(|r| inner[r * n + c]).collect();
                had128(&mut col);
                for r in 0..k {
                    w[r * n + c] = col[r];
                }
            }
            for r in 0..k {
                had128(&mut w[r * n..(r + 1) * n]);
            }
            for c in 0..n {
                let want: Vec<f32> = (0..k).map(|r| w[r * n + c] * m.suh(r) * m.svh(c)).collect();
                let row: Vec<f32> = got[c * k..(c + 1) * k].iter().map(|&b| value(b)).collect();
                close(&row, &want, &format!("restore k2 {k2} {n}x{k} row {c}"));
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires sm_120 GPU; EXL3 decode gate/up + down, eight experts (two CPU misses), shared expert, M 1..4"]
    fn glm53_exl3_gpu_moe() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut seed = 0x2545_f491;
        for (gk, dk) in [(5u8, 5u8), (7, 7), (5, 7), (4, 8)] {
            let images: Vec<Vec<u8>> = (0..8).map(|_| expert(gk, dk, &mut seed)).collect();
            let device_images: Vec<DeviceBuffer> = images
                .iter()
                .map(|i| upload(&device, &stream, i))
                .collect::<Result<_>>()?;
            // experts 2 and 5 are CPU misses (null pointers)
            let pointers: Vec<u64> = device_images
                .iter()
                .enumerate()
                .map(|(e, b)| {
                    if e == 2 || e == 5 {
                        0
                    } else {
                        b.as_ptr() as u64
                    }
                })
                .collect();
            let table = upload(
                &device,
                &stream,
                &pointers
                    .iter()
                    .flat_map(|p| p.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            // shared expert: BF16 dense weights
            let dense = |n: usize, k: usize, seed: &mut u64| -> Vec<u16> {
                (0..n * k)
                    .map(|_| bits(((random(seed) % 2001) as f32 - 1000.) * 2e-5))
                    .collect()
            };
            let (sg, su, sd) = (
                dense(I, H, &mut seed),
                dense(I, H, &mut seed),
                dense(H, I, &mut seed),
            );
            let (sgb, sub, sdb) = (
                upload(&device, &stream, &to_bytes(&sg))?,
                upload(&device, &stream, &to_bytes(&su))?,
                upload(&device, &stream, &to_bytes(&sd))?,
            );
            let weight = |b: &DeviceBuffer| g::DenseWeight {
                data: b.as_ptr(),
                scale: std::ptr::null(),
                format: 0,
            };
            for m in 1..=4usize {
                let x: Vec<u16> = (0..m * H)
                    .map(|_| bits(((random(&mut seed) % 2001) as f32 - 1000.) / 700.))
                    .collect();
                let xb = upload(&device, &stream, &to_bytes(&x))?;
                let mut product = device.alloc(9 * m * I * 2)?;
                check(
                    unsafe {
                        x3::hayate_glm_exl3_gate_up(
                            product.as_mut_ptr(),
                            xb.as_ptr(),
                            table.as_ptr(),
                            weight(&sgb),
                            weight(&sub),
                            m as u32,
                            u32::from(gk),
                            H as u32,
                            I as u32,
                            stream.as_raw(),
                        )
                    },
                    "EXL3 gate/up",
                )?;
                let prod = download(&stream, &product)?;
                for e in 0..8 {
                    let [gm, um, _] = parts(&images[e], gk, dk);
                    for t in 0..m {
                        let row = &prod[(e * m + t) * I..][..I];
                        if e == 2 || e == 5 {
                            assert!(row.iter().all(|&v| v == 0), "miss {e} writes zeros");
                            continue;
                        }
                        let xt: Vec<f32> = x[t * H..][..H].iter().map(|&b| value(b)).collect();
                        let (a, b) = (cpu_mv(&gm, &xt), cpu_mv(&um, &xt));
                        let want: Vec<f32> = a
                            .iter()
                            .zip(&b)
                            .map(|(&a, &b)| swiglu(bf16(a), bf16(b)))
                            .collect();
                        let got: Vec<f32> = row.iter().map(|&v| value(v)).collect();
                        close(&got, &want, &format!("gate/up k2 {gk} m {m} e {e} t {t}"));
                    }
                }
                // shared expert row 8: F64 BF16 dense
                for t in 0..m {
                    let xt: Vec<f64> = x[t * H..][..H]
                        .iter()
                        .map(|&b| f64::from(value(b)))
                        .collect();
                    let want: Vec<f32> = (0..I)
                        .map(|r| {
                            let a: f64 = (0..H)
                                .map(|c| f64::from(value(sg[r * H + c])) * xt[c])
                                .sum();
                            let b: f64 = (0..H)
                                .map(|c| f64::from(value(su[r * H + c])) * xt[c])
                                .sum();
                            swiglu(bf16(a as f32), bf16(b as f32))
                        })
                        .collect();
                    let got: Vec<f32> = prod[(8 * m + t) * I..][..I]
                        .iter()
                        .map(|&v| value(v))
                        .collect();
                    close(&got, &want, &format!("shared gate/up m {m} t {t}"));
                }
                // down over the GPU's own product; misses keep the sentinel
                let mut out = upload(&device, &stream, &vec![0x77; 9 * m * H * 2])?;
                check(
                    unsafe {
                        x3::hayate_glm_exl3_down(
                            out.as_mut_ptr(),
                            product.as_ptr(),
                            table.as_ptr(),
                            weight(&sdb),
                            m as u32,
                            u32::from(gk),
                            u32::from(dk),
                            H as u32,
                            I as u32,
                            stream.as_raw(),
                        )
                    },
                    "EXL3 down",
                )?;
                let down = download(&stream, &out)?;
                for e in 0..9 {
                    for t in 0..m {
                        let row = &down[(e * m + t) * H..][..H];
                        if e == 2 || e == 5 {
                            assert!(row.iter().all(|&v| v == 0x7777), "miss {e} left untouched");
                            continue;
                        }
                        let h: Vec<f32> = prod[(e * m + t) * I..][..I]
                            .iter()
                            .map(|&v| value(v))
                            .collect();
                        let want: Vec<f32> = if e == 8 {
                            (0..H)
                                .map(|r| {
                                    (0..I)
                                        .map(|c| f64::from(value(sd[r * I + c])) * f64::from(h[c]))
                                        .sum::<f64>() as f32
                                })
                                .collect()
                        } else {
                            cpu_mv(&parts(&images[e], gk, dk)[2], &h)
                        };
                        let got: Vec<f32> = row.iter().map(|&v| value(v)).collect();
                        close(&got, &want, &format!("down k2 {dk} m {m} e {e} t {t}"));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires sm_120 GPU; EXL3 grouped prefill GEMMs, ragged expert rows"]
    fn glm53_exl3_gpu_grouped() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut seed = 0x9e37_79b9;
        for (gk, dk) in [(5u8, 5u8), (7, 7)] {
            let rows_of = [1usize, 33, 0, 70, 5];
            let count = rows_of.len();
            let images: Vec<Vec<u8>> = (0..count).map(|_| expert(gk, dk, &mut seed)).collect();
            let device_images: Vec<DeviceBuffer> = images
                .iter()
                .map(|i| upload(&device, &stream, i))
                .collect::<Result<_>>()?;
            let table: Vec<u8> = device_images
                .iter()
                .flat_map(|b| (b.as_ptr() as u64).to_le_bytes())
                .collect();
            let table = upload(&device, &stream, &table)?;
            let mut ranges = Vec::new();
            let mut at = 0u32;
            for &r in &rows_of {
                ranges.extend([r as u32, at]);
                at += r as u32;
            }
            let total = at as usize;
            let ranges_b = upload(
                &device,
                &stream,
                &ranges
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let x: Vec<u16> = (0..total * H)
                .map(|_| bits(((random(&mut seed) % 2001) as f32 - 1000.) / 700.))
                .collect();
            let xb = upload(&device, &stream, &to_bytes(&x))?;
            let mut product = device.alloc(total * I * 2)?;
            let mut out = device.alloc(total * H * 2)?;
            check(
                unsafe {
                    x3::hayate_glm_grouped_exl3_gate_up(
                        product.as_mut_ptr(),
                        xb.as_ptr(),
                        table.as_ptr(),
                        ranges_b.as_ptr(),
                        count as u32,
                        u32::from(gk),
                        H as u32,
                        I as u32,
                        stream.as_raw(),
                    )
                },
                "EXL3 grouped gate/up",
            )?;
            check(
                unsafe {
                    x3::hayate_glm_grouped_exl3_down(
                        out.as_mut_ptr(),
                        product.as_ptr(),
                        table.as_ptr(),
                        ranges_b.as_ptr(),
                        count as u32,
                        u32::from(gk),
                        u32::from(dk),
                        H as u32,
                        I as u32,
                        stream.as_raw(),
                    )
                },
                "EXL3 grouped down",
            )?;
            let (prod, down) = (download(&stream, &product)?, download(&stream, &out)?);
            let mut row = 0;
            for (j, &r) in rows_of.iter().enumerate() {
                let [gm, um, dm] = parts(&images[j], gk, dk);
                for _ in 0..r {
                    let xt: Vec<f32> = x[row * H..][..H].iter().map(|&b| value(b)).collect();
                    let (a, b) = (cpu_mv(&gm, &xt), cpu_mv(&um, &xt));
                    let want: Vec<f32> = a
                        .iter()
                        .zip(&b)
                        .map(|(&a, &b)| swiglu(bf16(a), bf16(b)))
                        .collect();
                    let got: Vec<f32> = prod[row * I..][..I].iter().map(|&v| value(v)).collect();
                    close(
                        &got,
                        &want,
                        &format!("grouped gate/up k2 {gk} expert {j} row {row}"),
                    );
                    let want = cpu_mv(&dm, &got);
                    let got: Vec<f32> = down[row * H..][..H].iter().map(|&v| value(v)).collect();
                    close(
                        &got,
                        &want,
                        &format!("grouped down k2 {dk} expert {j} row {row}"),
                    );
                    row += 1;
                }
            }
        }
        Ok(())
    }

    /// The PCIe lane reads RAM-tier (pinned) images directly: the same results as device images, and the
    /// decode time with 0..8 of the eight experts in host memory.
    #[test]
    #[ignore = "requires sm_120 GPU; EXL3 decode with experts read over PCIe from pinned host memory"]
    fn glm53_exl3_gpu_zero_copy() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut seed = 11;
        let raw: Vec<Vec<u8>> = (0..8).map(|_| expert(5, 5, &mut seed)).collect();
        let on_device: Vec<DeviceBuffer> = raw
            .iter()
            .map(|i| upload(&device, &stream, i))
            .collect::<Result<_>>()?;
        let mut on_host = Vec::new();
        for image in &raw {
            let mut pinned = device.alloc_pinned(image.len())?;
            pinned.as_mut_slice().copy_from_slice(image);
            on_host.push(pinned);
        }
        let dense = upload(&device, &stream, &vec![0u8; I * H * 2])?;
        let weight = g::DenseWeight {
            data: dense.as_ptr(),
            scale: std::ptr::null(),
            format: 0,
        };
        let x = upload(
            &device,
            &stream,
            &to_bytes(
                &(0..H)
                    .map(|i| bits((i % 17) as f32 / 17. - 0.5))
                    .collect::<Vec<_>>(),
            ),
        )?;
        let mut product = device.alloc(9 * I * 2)?;
        let mut out = device.alloc(9 * H * 2)?;
        let (pp, op) = (product.as_mut_ptr(), out.as_mut_ptr());
        let mut reference = None;
        for host in [0usize, 1, 2, 4, 8] {
            let pointers: Vec<u8> = (0..8)
                .map(|e| {
                    if e < host {
                        on_host[e].as_ptr() as u64
                    } else {
                        on_device[e].as_ptr() as u64
                    }
                })
                .flat_map(|p| p.to_le_bytes())
                .collect();
            let table = upload(&device, &stream, &pointers)?;
            let run = || -> Result<()> {
                check(
                    unsafe {
                        x3::hayate_glm_exl3_gate_up(
                            pp,
                            x.as_ptr(),
                            table.as_ptr(),
                            weight,
                            weight,
                            1,
                            5,
                            H as u32,
                            I as u32,
                            stream.as_raw(),
                        )
                    },
                    "gate/up",
                )?;
                check(
                    unsafe {
                        x3::hayate_glm_exl3_down(
                            op,
                            pp,
                            table.as_ptr(),
                            weight,
                            1,
                            5,
                            5,
                            H as u32,
                            I as u32,
                            stream.as_raw(),
                        )
                    },
                    "down",
                )
            };
            run()?;
            let got = (download(&stream, &product)?, download(&stream, &out)?);
            match &reference {
                None => reference = Some(got),
                Some(r) => assert!(
                    *r == got,
                    "host-memory experts give other bits ({host} on the host)"
                ),
            }
            for _ in 0..3 {
                run()?;
            }
            let (a, b) = (stream.event()?, stream.event()?);
            a.record(&stream)?;
            for _ in 0..32 {
                run()?;
            }
            b.record(&stream)?;
            b.synchronize()?;
            let us = f64::from(a.elapsed_ms(&b)?) * 1000. / 32.;
            let bytes = host as f64 * raw[0].len() as f64;
            eprintln!(
                "glm53_exl3_zero_copy host_experts={host} decode_us={us:.1} host_mb={:.1} gb_s_if_pcie_bound={:.1}",
                bytes / 1e6,
                if host > 0 { bytes / us / 1e3 } else { 0. }
            );
        }
        Ok(())
    }

    /// Bulk copies of pinned host images into device slots: the in-graph fetch kernel (data-dependent pointers)
    /// against the DMA engine (cudaMemcpyAsync), 1..8 K2.5 experts.
    #[test]
    #[ignore = "requires sm_120 GPU; PCIe fetch kernel and DMA copy throughput"]
    fn glm53_exl3_gpu_fetch() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let bytes = 7_901_184usize;
        let mut host = Vec::new();
        for e in 0..8u8 {
            let mut p = device.alloc_pinned(bytes)?;
            p.as_mut_slice()
                .iter_mut()
                .enumerate()
                .for_each(|(i, b)| *b = (i as u8).wrapping_mul(7) ^ e);
            host.push(p);
        }
        let mut slots: Vec<DeviceBuffer> =
            (0..8).map(|_| device.alloc(bytes)).collect::<Result<_>>()?;
        let dst = upload(
            &device,
            &stream,
            &slots
                .iter()
                .flat_map(|b| (b.as_ptr() as u64).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            for _ in 0..3 {
                f()?;
            }
            let (a, b) = (stream.event()?, stream.event()?);
            a.record(&stream)?;
            for _ in 0..16 {
                f()?;
            }
            b.record(&stream)?;
            b.synchronize()?;
            Ok(f64::from(a.elapsed_ms(&b)?) * 1000. / 16.)
        };
        for count in [1usize, 2, 4, 8] {
            let src = upload(
                &device,
                &stream,
                &(0..8)
                    .map(|e| {
                        if e < count {
                            host[e].as_ptr() as u64
                        } else {
                            0
                        }
                    })
                    .flat_map(|p| p.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            for blocks in [4u32, 8, 16, 32, 64, 188] {
                let us = time(&mut || {
                    check(
                        unsafe {
                            x3::hayate_glm_exl3_fetch(
                                dst.as_ptr(),
                                src.as_ptr(),
                                bytes,
                                blocks,
                                stream.as_raw(),
                            )
                        },
                        "fetch",
                    )
                })?;
                eprintln!(
                    "glm53_exl3_fetch kernel experts={count} blocks={blocks} us={us:.1} gb_s={:.1}",
                    (count * bytes) as f64 / us / 1e3
                );
            }
            let us = time(&mut || {
                for (slot, h) in slots.iter_mut().zip(&host).take(count) {
                    unsafe { stream.copy_h2d_async(slot, h)? };
                }
                Ok(())
            })?;
            eprintln!(
                "glm53_exl3_fetch dma experts={count} us={us:.1} gb_s={:.1}",
                (count * bytes) as f64 / us / 1e3
            );
        }
        for (slot, h) in slots.iter().zip(&host) {
            let mut back = vec![0u8; bytes];
            stream.copy_d2h(&mut back, slot)?;
            assert!(back == h.as_slice(), "fetched bytes");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires sm_120 GPU; EXL3 decode (eight K2.5 experts, M 1) and grouped prefill timings"]
    fn glm53_exl3_gpu_bench() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut seed = 7;
        let images: Vec<DeviceBuffer> = (0..8)
            .map(|_| upload(&device, &stream, &expert(5, 5, &mut seed)))
            .collect::<Result<_>>()?;
        let table = upload(
            &device,
            &stream,
            &images
                .iter()
                .flat_map(|b| (b.as_ptr() as u64).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let dense = upload(&device, &stream, &vec![0u8; I * H * 2])?;
        let weight = g::DenseWeight {
            data: dense.as_ptr(),
            scale: std::ptr::null(),
            format: 0,
        };
        let x = upload(&device, &stream, &to_bytes(&vec![bits(0.25); 4 * H]))?;
        let mut product = device.alloc(9 * 4 * I * 2)?;
        let mut out = device.alloc(9 * 4 * H * 2)?;
        let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            for _ in 0..4 {
                f()?;
            }
            let (a, b) = (stream.event()?, stream.event()?);
            a.record(&stream)?;
            for _ in 0..64 {
                f()?;
            }
            b.record(&stream)?;
            b.synchronize()?;
            Ok(f64::from(a.elapsed_ms(&b)?) * 1000. / 64.)
        };
        for m in [1u32, 2, 4] {
            let gu = time(&mut || {
                check(
                    unsafe {
                        x3::hayate_glm_exl3_gate_up(
                            product.as_mut_ptr(),
                            x.as_ptr(),
                            table.as_ptr(),
                            weight,
                            weight,
                            m,
                            5,
                            H as u32,
                            I as u32,
                            stream.as_raw(),
                        )
                    },
                    "gate/up",
                )
            })?;
            let dn = time(&mut || {
                check(
                    unsafe {
                        x3::hayate_glm_exl3_down(
                            out.as_mut_ptr(),
                            product.as_ptr(),
                            table.as_ptr(),
                            weight,
                            m,
                            5,
                            5,
                            H as u32,
                            I as u32,
                            stream.as_raw(),
                        )
                    },
                    "down",
                )
            })?;
            eprintln!(
                "glm53_exl3_bench decode m={m} gate_up_us={gu:.1} down_us={dn:.1} (8 K2.5 experts + BF16 shared)"
            );
        }
        // prefill: 8 experts x 1024 rows
        let rows = 1024usize;
        let ranges: Vec<u8> = (0..8u32)
            .flat_map(|j| [rows as u32, j * rows as u32])
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let ranges = upload(&device, &stream, &ranges)?;
        let xin = upload(&device, &stream, &to_bytes(&vec![bits(0.25); 8 * rows * H]))?;
        let mut prod = device.alloc(8 * rows * I * 2)?;
        let mut down = device.alloc(8 * rows * H * 2)?;
        let gu = time(&mut || {
            check(
                unsafe {
                    x3::hayate_glm_grouped_exl3_gate_up(
                        prod.as_mut_ptr(),
                        xin.as_ptr(),
                        table.as_ptr(),
                        ranges.as_ptr(),
                        8,
                        5,
                        H as u32,
                        I as u32,
                        stream.as_raw(),
                    )
                },
                "grouped gate/up",
            )
        })?;
        let dn = time(&mut || {
            check(
                unsafe {
                    x3::hayate_glm_grouped_exl3_down(
                        down.as_mut_ptr(),
                        prod.as_ptr(),
                        table.as_ptr(),
                        ranges.as_ptr(),
                        8,
                        5,
                        5,
                        H as u32,
                        I as u32,
                        stream.as_raw(),
                    )
                },
                "grouped down",
            )
        })?;
        let flops = 2. * (8 * rows) as f64 * 3. * (H * I) as f64;
        eprintln!(
            "glm53_exl3_bench grouped rows={} gate_up_us={gu:.0} down_us={dn:.0} tflops={:.1}",
            8 * rows,
            flops / (gu + dn) / 1e6
        );
        let _ = std::ptr::null::<c_void>();
        Ok(())
    }
}
