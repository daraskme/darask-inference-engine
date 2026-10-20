use super::ffi as g;
use crate::runtime::{Device, DeviceBuffer, Stream, check};
use hayate_core::Result;

fn upload(device: &Device, stream: &Stream, raw: &[u8]) -> Result<DeviceBuffer> {
    let mut out = device.alloc(raw.len())?;
    stream.copy_h2d(&mut out, raw)?;
    Ok(out)
}
fn bf(value: f32) -> [u8; 2] {
    ((super::cpu::bf16(value).to_bits() >> 16) as u16).to_le_bytes()
}
fn bytes(stream: &Stream, buffer: &DeviceBuffer) -> Result<Vec<u8>> {
    let mut raw = vec![0; buffer.bytes()];
    stream.copy_d2h(&mut raw, buffer)?;
    Ok(raw)
}

#[test]
#[ignore = "requires main CUDA GPU; checkpoint-free conv/pool causal chunk boundaries"]
fn glm53_prefill_conv_pool_chunk_boundaries() -> Result<()> {
    let device = Device::select()?;
    let stream = device.stream()?;
    let st = stream.as_raw();
    let cols = 128usize;
    let weight = upload(
        &device,
        &stream,
        &(0..cols * 4)
            .flat_map(|i| bf((i % 11) as f32 / 13. - 0.3))
            .collect::<Vec<_>>(),
    )?;
    let ape = upload(
        &device,
        &stream,
        &(0..cols * 4)
            .flat_map(|i| bf((i % 7) as f32 / 11. - 0.2))
            .collect::<Vec<_>>(),
    )?;
    for start in 0..8usize {
        for rows in [1, 2, 3, 4, 5, 127, 128, 129] {
            let raw: Vec<_> = (0..rows * cols)
                .flat_map(|i| bf((i % 43) as f32 / 31. - 0.7))
                .collect();
            let gates: Vec<_> = (0..rows * cols)
                .flat_map(|i| bf((i % 19) as f32 / 23. - 0.4))
                .collect();
            let x = upload(&device, &stream, &raw)?;
            let gate = upload(&device, &stream, &gates)?;
            let history: Vec<_> = (0..3 * cols)
                .flat_map(|i| bf((i % 23) as f32 / 17. - 0.5))
                .collect();
            let hc = upload(&device, &stream, &history)?;
            let hs = upload(&device, &stream, &history)?;
            let chunk = device.alloc(rows * cols * 2)?;
            let serial = device.alloc(rows * cols * 2)?;
            check(
                unsafe {
                    g::hayate_glm_conv_chunk(
                        chunk.as_ptr().cast_mut(),
                        hc.as_ptr().cast_mut(),
                        x.as_ptr(),
                        weight.as_ptr(),
                        rows as u32,
                        cols as u32,
                        st,
                    )
                },
                "test chunk conv",
            )?;
            for row in 0..rows {
                check(
                    unsafe {
                        g::hayate_glm_conv_step(
                            serial
                                .as_ptr()
                                .cast::<u8>()
                                .add(row * cols * 2)
                                .cast_mut()
                                .cast(),
                            hs.as_ptr().cast_mut(),
                            x.as_ptr().cast::<u8>().add(row * cols * 2).cast(),
                            weight.as_ptr(),
                            cols as u32,
                            st,
                        )
                    },
                    "test serial conv",
                )?;
            }
            assert_eq!(
                bytes(&stream, &chunk)?,
                bytes(&stream, &serial)?,
                "conv start={start} rows={rows}"
            );
            assert_eq!(bytes(&stream, &hc)?, bytes(&stream, &hs)?, "conv history");
            let old: Vec<_> = (0..4 * cols)
                .flat_map(|i| bf((i % 29) as f32 / 37. - 0.6))
                .collect();
            let kc = upload(&device, &stream, &old)?;
            let gc = upload(&device, &stream, &old)?;
            let ks = upload(&device, &stream, &old)?;
            let gs = upload(&device, &stream, &old)?;
            let pools = (start + rows).div_ceil(4) * cols * 2;
            let pc = upload(&device, &stream, &vec![0; pools])?;
            let ps = upload(&device, &stream, &vec![0; pools])?;
            check(
                unsafe {
                    g::hayate_glm_pool_chunk(
                        pc.as_ptr().cast_mut(),
                        kc.as_ptr().cast_mut(),
                        gc.as_ptr().cast_mut(),
                        x.as_ptr(),
                        gate.as_ptr(),
                        ape.as_ptr(),
                        start as u32,
                        rows as u32,
                        st,
                    )
                },
                "test chunk pools",
            )?;
            for row in 0..rows {
                let pos = start + row;
                for (out, input) in [(&ks, &x), (&gs, &gate)] {
                    check(
                        unsafe {
                            g::hayate_glm_copy(
                                out.as_ptr()
                                    .cast::<u8>()
                                    .add(pos % 4 * cols * 2)
                                    .cast_mut()
                                    .cast(),
                                input.as_ptr().cast::<u8>().add(row * cols * 2).cast(),
                                cols * 2,
                                st,
                            )
                        },
                        "test pool append",
                    )?;
                }
                if pos % 4 == 3 {
                    check(
                        unsafe {
                            g::hayate_glm_pool(
                                ps.as_ptr()
                                    .cast::<u8>()
                                    .add(pos / 4 * cols * 2)
                                    .cast_mut()
                                    .cast(),
                                ks.as_ptr(),
                                gs.as_ptr(),
                                ape.as_ptr(),
                                1,
                                cols as u32,
                                st,
                            )
                        },
                        "test serial pool",
                    )?;
                }
            }
            assert_eq!(
                bytes(&stream, &pc)?,
                bytes(&stream, &ps)?,
                "pools start={start} rows={rows}"
            );
            let kc = bytes(&stream, &kc)?;
            let gc = bytes(&stream, &gc)?;
            let ks = bytes(&stream, &ks)?;
            let gs = bytes(&stream, &gs)?;
            let live = (start + rows) % 4 * cols * 2;
            assert_eq!(&kc[..live], &ks[..live], "live key tail");
            assert_eq!(&gc[..live], &gs[..live], "live gate tail");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires main CUDA GPU; indexed attention retains decode softmax and chronology"]
fn glm53_prefill_indexed_attention_bits() -> Result<()> {
    let device = Device::select()?;
    let stream = device.stream()?;
    let st = stream.as_raw();
    for count in [1usize, 3, 127, 128, 129, 2051] {
        let rows = 3usize;
        let union = count + 3;
        let raw: Vec<_> = (0..union * 32768)
            .flat_map(|i| bf(((i * 13 % 113) as f32 - 51.) / 101.))
            .collect();
        let packed = upload(&device, &stream, &raw)?;
        let q = upload(
            &device,
            &stream,
            &(0..rows * 16384)
                .flat_map(|i| bf((i % 17) as f32 / 23. - 0.3))
                .collect::<Vec<_>>(),
        )?;
        let mut indices = vec![0; rows * 2051 * 4];
        for row in 0..rows {
            for j in 0..count {
                let at = (row * 2051 + j) * 4;
                indices[at..at + 4].copy_from_slice(&((j + row) as u32).to_le_bytes());
            }
        }
        let map = upload(&device, &stream, &indices)?;
        let counts = upload(&device, &stream, &(count as u32).to_le_bytes().repeat(rows))?;
        let batched = device.alloc(rows * 16384 * 2)?;
        check(
            unsafe {
                g::hayate_glm_attention_indexed(
                    batched.as_ptr().cast_mut(),
                    q.as_ptr(),
                    packed.as_ptr(),
                    map.as_ptr(),
                    counts.as_ptr(),
                    rows as u32,
                    st,
                )
            },
            "test indexed attention",
        )?;
        let serial = device.alloc(rows * 16384 * 2)?;
        let keys = device.alloc(count * 16384 * 2)?;
        let vals = device.alloc(count * 16384 * 2)?;
        let probs = device.alloc(64 * count * 2)?;
        for row in 0..rows {
            check(
                unsafe {
                    g::hayate_glm_split_selected(
                        keys.as_ptr().cast_mut(),
                        vals.as_ptr().cast_mut(),
                        packed.as_ptr(),
                        map.as_ptr().cast::<u8>().add(row * 2051 * 4).cast(),
                        count as u32,
                        st,
                    )
                },
                "test indexed split",
            )?;
            check(
                unsafe {
                    g::hayate_glm_attention_step(
                        serial
                            .as_ptr()
                            .cast::<u8>()
                            .add(row * 16384 * 2)
                            .cast_mut()
                            .cast(),
                        probs.as_ptr().cast_mut(),
                        q.as_ptr().cast::<u8>().add(row * 16384 * 2).cast(),
                        keys.as_ptr(),
                        vals.as_ptr(),
                        count as u32,
                        st,
                    )
                },
                "test serial attention",
            )?;
        }
        assert_eq!(
            bytes(&stream, &batched)?,
            bytes(&stream, &serial)?,
            "count={count}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires main CUDA GPU; causal sparse selection across the 2048-token budget"]
fn glm53_prefill_selection_causal_bits() -> Result<()> {
    let device = Device::select()?;
    let stream = device.stream()?;
    let st = stream.as_raw();
    for start in [0usize, 1, 3, 2047, 2048, 2051] {
        for rows in [1usize, 3, 129] {
            let np = (start + rows) / 4;
            let pools = upload(
                &device,
                &stream,
                &(0..np.max(1) * 128)
                    .flat_map(|i| bf((i % 59) as f32 / 113. - 0.2))
                    .collect::<Vec<_>>(),
            )?;
            let q = upload(
                &device,
                &stream,
                &(0..rows * 4096)
                    .flat_map(|i| bf((i % 41) as f32 / 71. - 0.1))
                    .collect::<Vec<_>>(),
            )?;
            let w = upload(
                &device,
                &stream,
                &(0..rows * 32)
                    .flat_map(|i| bf((i % 19) as f32 / 23. - 0.3))
                    .collect::<Vec<_>>(),
            )?;
            let scores = device.alloc((rows * np * 4).max(4))?;
            let chunk = device.alloc(rows * 2051 * 4)?;
            let serial = device.alloc(rows * 2051 * 4)?;
            check(
                unsafe {
                    g::hayate_glm_select_chunk(
                        chunk.as_ptr().cast_mut(),
                        scores.as_ptr().cast_mut(),
                        q.as_ptr(),
                        pools.as_ptr(),
                        w.as_ptr(),
                        start as u32,
                        rows as u32,
                        st,
                    )
                },
                "test chunk selection",
            )?;
            for row in 0..rows {
                check(
                    unsafe {
                        g::hayate_glm_select_step(
                            serial
                                .as_ptr()
                                .cast::<u8>()
                                .add(row * 2051 * 4)
                                .cast_mut()
                                .cast(),
                            scores.as_ptr().cast_mut(),
                            q.as_ptr().cast::<u8>().add(row * 4096 * 2).cast(),
                            pools.as_ptr(),
                            w.as_ptr().cast::<u8>().add(row * 32 * 2).cast(),
                            (start + row + 1) as u32,
                            st,
                        )
                    },
                    "test causal serial selection",
                )?;
            }
            assert_eq!(
                bytes(&stream, &chunk)?,
                bytes(&stream, &serial)?,
                "start={start} rows={rows}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires supported CUDA GPU; parallel KDA against actual one-token recurrence"]
fn glm53_prefill_kda_chunk_relative_l2() -> Result<()> {
    let device = Device::select()?;
    let stream = device.stream()?;
    let st = stream.as_raw();
    for dim in [8usize, 128] {
        for rows in [1usize, 15, 16, 17, 33, 129, 257] {
            let heads = 2usize;
            let width = heads * dim;
            let input = |salt: usize| -> Vec<u8> {
                (0..rows * width)
                    .flat_map(
                        |i| bf(((i * (salt * 2 + 1) + i / dim * 17) % 137) as f32 / 71. - 0.9),
                    )
                    .collect()
            };
            let q = upload(&device, &stream, &input(3))?;
            let k = upload(&device, &stream, &input(11))?;
            let v = upload(&device, &stream, &input(7))?;
            let decay = upload(
                &device,
                &stream,
                &(0..rows * width)
                    .flat_map(|i| (-5. * (i % 31) as f32 / 30.).to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let beta = upload(
                &device,
                &stream,
                &(0..rows * heads)
                    .flat_map(|i| bf((i % 7) as f32 / 6.))
                    .collect::<Vec<_>>(),
            )?;
            let initial: Vec<_> = (0..heads * dim * dim)
                .flat_map(|i| (((i * 13 % 61) as f32 - 30.) / 79.).to_le_bytes())
                .collect();
            let sc = upload(&device, &stream, &initial)?;
            let ss = upload(&device, &stream, &initial)?;
            let chunk = device.alloc(rows * width * 2)?;
            let serial = device.alloc(rows * width * 2)?;
            let nq = device.alloc(rows * width * 4)?;
            let nk = device.alloc(rows * width * 4)?;
            // Also test non-aligned caller chunk boundaries and state carry-in.
            for start in (0..rows).step_by(33) {
                let count = 33.min(rows - start);
                let offset = |buffer: &DeviceBuffer, stride: usize| -> *const std::ffi::c_void {
                    unsafe { buffer.as_ptr().cast::<u8>().add(start * stride).cast() }
                };
                check(
                    unsafe {
                        g::hayate_glm_kda_chunk(
                            offset(&chunk, width * 2).cast_mut(),
                            sc.as_ptr().cast_mut(),
                            nq.as_ptr().cast_mut(),
                            nk.as_ptr().cast_mut(),
                            offset(&q, width * 2),
                            offset(&k, width * 2),
                            offset(&v, width * 2),
                            offset(&decay, width * 4),
                            offset(&beta, heads * 2),
                            count as u32,
                            heads as u32,
                            dim as u32,
                            st,
                        )
                    },
                    "test KDA chunk",
                )?;
            }
            for row in 0..rows {
                let offset = |buffer: &DeviceBuffer, stride: usize| -> *const std::ffi::c_void {
                    unsafe { buffer.as_ptr().cast::<u8>().add(row * stride).cast() }
                };
                check(
                    unsafe {
                        g::hayate_glm_kda(
                            offset(&serial, width * 2).cast_mut(),
                            ss.as_ptr().cast_mut(),
                            offset(&q, width * 2),
                            offset(&k, width * 2),
                            offset(&v, width * 2),
                            offset(&decay, width * 4),
                            offset(&beta, heads * 2),
                            1,
                            heads as u32,
                            dim as u32,
                            st,
                        )
                    },
                    "test tokenwise KDA",
                )?;
            }
            let decode = |raw: Vec<u8>, f32_values: bool| -> Vec<f64> {
                if f32_values {
                    raw.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b) as f64)
                        .collect()
                } else {
                    raw.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16) as f64)
                        .collect()
                }
            };
            for (a, b, is_state) in [(&chunk, &serial, false), (&sc, &ss, true)] {
                let actual = decode(bytes(&stream, a)?, is_state);
                let expected = decode(bytes(&stream, b)?, is_state);
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();
                let norm = expected.iter().map(|v| v * v).sum::<f64>();
                let l2 = (error / norm.max(1e-30)).sqrt();
                eprintln!("KDA dim={dim} rows={rows} state={is_state} relative_l2={l2:.9}");
                assert!(l2.is_finite() && l2 <= 1e-3, "relative L2 {l2}");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires supported CUDA GPU; small-row NVFP4 GEMM versus unrounded decode arithmetic"]
fn glm53_prefill_nvfp4_global_scale() -> Result<()> {
    let device = Device::select()?;
    let stream = device.stream()?;
    let (n, k) = (2048usize, 4096usize);
    let size = hayate_core::QuantSpec::Nvfp4
        .matrix_bytes(n as u64, k as u64)
        .unwrap() as usize;
    let mut raw = vec![0; size];
    for (i, value) in raw[..n * k / 2].iter_mut().enumerate() {
        *value = ((i * 13 % 16) | ((i * 7 % 16) << 4)) as u8;
    }
    for (i, value) in raw[n * k / 2..size - 4].iter_mut().enumerate() {
        *value = 0x30 + (i % 17) as u8;
    }
    raw[size - 4..].copy_from_slice(&0.071312346f32.to_le_bytes());
    let image = upload(&device, &stream, &raw)?;
    let weight = device.alloc(n * k * 2)?;
    let scale = device.alloc(4)?;
    check(
        unsafe {
            g::hayate_glm_nvfp4_restore_unscaled(
                weight.as_ptr().cast_mut(),
                scale.as_ptr().cast_mut(),
                image.as_ptr(),
                n as u32,
                k as u32,
                stream.as_raw(),
            )
        },
        "test NVFP4 unscaled restore",
    )?;
    let metadata = device.alloc(608)?;
    for rows in [1usize, 2, 3, 4] {
        let raw: Vec<_> = (0..rows * k)
            .flat_map(|i| bf((i * 31 % 113) as f32 / 53. - 1.))
            .collect();
        let input = upload(&device, &stream, &raw)?;
        let ranges = upload(
            &device,
            &stream,
            &[rows as u32, 0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let accumulator = device.alloc(rows * n * 4)?;
        let output = device.alloc(rows * n * 2)?;
        let reference = device.alloc(rows * n * 2)?;
        check(
            unsafe {
                g::hayate_glm_grouped_nvfp4(
                    output.as_ptr().cast_mut(),
                    accumulator.as_ptr().cast_mut(),
                    input.as_ptr(),
                    weight.as_ptr(),
                    ranges.as_ptr(),
                    1,
                    n as u32,
                    k as u32,
                    metadata.as_ptr().cast_mut(),
                    scale.as_ptr(),
                    rows as u32,
                    stream.as_raw(),
                )
            },
            "test NVFP4 grouped rescale",
        )?;
        check(
            unsafe {
                g::hayate_glm_dense_batch(
                    reference.as_ptr().cast_mut(),
                    input.as_ptr(),
                    g::DenseWeight {
                        data: image.as_ptr(),
                        scale: std::ptr::null(),
                        format: 3,
                    },
                    rows as u32,
                    n as u32,
                    k as u32,
                    0,
                    stream.as_raw(),
                )
            },
            "test NVFP4 decode unrounded reference",
        )?;
        let values = |raw: Vec<u8>| {
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16) as f64)
                .collect::<Vec<_>>()
        };
        let actual = values(bytes(&stream, &output)?);
        let expected = values(bytes(&stream, &reference)?);
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>();
        let energy = expected.iter().map(|v| v * v).sum::<f64>();
        let relative = (error / energy).sqrt();
        assert!(relative <= 1e-3, "NVFP4 rows={rows} L2={relative}");
        eprintln!("NVFP4 grouped/decode rows={rows} relative_l2={relative:.9}");
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA GPU; 1 GiB pinned DMA, two streams, 30 GB/s gate on main PC"]
fn glm53_prefill_pinned_bandwidth() -> Result<()> {
    let device = Device::select()?;
    let control = device.stream()?;
    let copies = [device.stream()?, device.stream()?];
    let mut host = [
        device.alloc_pinned(512 << 20)?,
        device.alloc_pinned(512 << 20)?,
    ];
    let mut gpu = [device.alloc(512 << 20)?, device.alloc(512 << 20)?];
    for (i, image) in host.iter_mut().enumerate() {
        image.as_mut_slice().fill(17 + i as u8);
    }
    let start = control.event()?;
    let end = control.event()?;
    let done = [copies[0].event()?, copies[1].event()?];
    let mut rates = Vec::new();
    for run in 0..7 {
        start.record(&control)?;
        let result = (|| -> Result<()> {
            for i in 0..2 {
                copies[i].wait_event(&start)?;
                unsafe {
                    copies[i].copy_h2d_async(&mut gpu[i], &host[i])?;
                }
                done[i].record(&copies[i])?;
                control.wait_event(&done[i])?;
            }
            end.record(&control)?;
            end.synchronize()
        })();
        let retired: Vec<_> = copies.iter().map(Stream::synchronize).collect();
        for result in retired {
            result?;
        }
        result?;
        if run >= 2 {
            rates.push((1u64 << 30) as f64 / start.elapsed_ms(&end)? as f64 / 1e6);
        }
    }
    for (i, image) in gpu.iter().enumerate() {
        let mut sample = [0; 4096];
        control.copy_d2h(&mut sample, image)?;
        assert!(sample.iter().all(|&v| v == 17 + i as u8));
    }
    rates.sort_by(f64::total_cmp);
    eprintln!(
        "glm_prefill_dma bytes={} streams=2 warmups=2 repeats=5 median_gb_s={:.3} gpu={:?}",
        1u64 << 30,
        rates[2],
        device.name
    );
    if device.machine == hayate_core::MachineId::Main {
        assert!(
            rates[2] >= 30.,
            "main PC pinned DMA below 30 GB/s: {}",
            rates[2]
        );
    }
    Ok(())
}
