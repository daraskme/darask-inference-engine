use hayate_core::Result;
use hayate_kernels::{
    glm53::ffi as g,
    runtime::{Device, check, host_sync_count},
};

#[test]
#[ignore = "requires supported CUDA GPU; mapped doorbell reuse, no checkpoint"]
fn glm53_mapped_doorbell_reuse_one_host_sync() -> Result<()> {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    let device = Device::select()?;
    let stream = device.stream()?;
    let mailbox = device.alloc_mapped(192)?;
    let mut input = device.alloc(32 * 4)?;
    let output = device.alloc(32 * 4)?;
    let values: Vec<_> = (1u32..=32).flat_map(u32::to_le_bytes).collect();
    stream.copy_h2d(&mut input, &values)?;
    let stop = AtomicBool::new(false);
    let mut actual = vec![0; values.len()];
    let syncs = host_sync_count();
    std::thread::scope(|scope| -> Result<()> {
        let worker = scope.spawn(|| {
            let ptr = mailbox.host_ptr().cast::<u32>();
            let ready = unsafe { &*ptr.cast::<AtomicU32>() };
            let done = unsafe { &*ptr.add(16).cast::<AtomicU32>() };
            for seq in 1..=32 {
                while ready.load(Ordering::Acquire) != seq {
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    std::hint::spin_loop();
                }
                unsafe {
                    assert_eq!(ptr.add(32).read(), seq);
                    ptr.add(33).write(seq * 17);
                }
                done.store(seq, Ordering::Release);
            }
        });
        let result = (|| -> Result<()> {
            for seq in 1..=32 {
                let dst = unsafe { mailbox.device_ptr().cast::<u32>().add(32).cast() };
                let src = unsafe { input.as_ptr().cast::<u32>().add(seq as usize - 1).cast() };
                check(
                    unsafe { g::hayate_glm_async_publish(dst, src, 4, stream.as_raw()) },
                    "test publish",
                )?;
                unsafe {
                    stream.write_value32(mailbox.device_ptr(), seq)?;
                    stream.wait_value32(mailbox.device_ptr().cast::<u32>().add(16).cast(), seq)?;
                }
                let dst = unsafe {
                    output
                        .as_ptr()
                        .cast::<u32>()
                        .add(seq as usize - 1)
                        .cast_mut()
                        .cast()
                };
                let src = unsafe { mailbox.device_ptr().cast::<u32>().add(33).cast() };
                check(
                    unsafe { g::hayate_glm_async_publish(dst, src, 4, stream.as_raw()) },
                    "test consume",
                )?;
            }
            stream.copy_d2h(&mut actual, &output)
        })();
        stop.store(true, Ordering::Release);
        worker.join().unwrap();
        result
    })?;
    assert_eq!(host_sync_count() - syncs, 1);
    assert_eq!(
        actual,
        (1u32..=32)
            .flat_map(|v| (v * 17).to_le_bytes())
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
#[ignore = "requires supported CUDA GPU; dynamic KV union GEMM bits, no checkpoint"]
fn glm53_mapped_sparse_union_gemm_bits() -> Result<()> {
    use super::speculation::{SPARSE_WIDTH, sparse_union};
    use hayate_kernels::text::ffi::hayate_text_linear;
    let device = Device::select()?;
    let stream = device.stream()?;
    let bf = |v: f32| ((hayate_kernels::glm53::cpu::bf16(v).to_bits() >> 16) as u16).to_le_bytes();
    let mut latent = device.alloc_pinned(512 * 512 * 2)?;
    let raw: Vec<_> = (0..512 * 512)
        .flat_map(|i| bf((i % 59) as f32 / 31. - 1.))
        .collect();
    latent.as_mut_slice().copy_from_slice(&raw);
    let weights: Vec<_> = (0..32768 * 512)
        .flat_map(|i| bf((i % 37) as f32 / 257. - 0.0625))
        .collect();
    let mut weight = device.alloc(weights.len())?;
    stream.copy_h2d(&mut weight, &weights)?;
    for m in 1..=4 {
        let queries: Vec<Vec<i32>> = (0..m)
            .map(|row| {
                (0..129 + row)
                    .map(|i| ((i * 7 + row * 31) % 499) as i32)
                    .collect()
            })
            .collect();
        let selected: Vec<_> = queries
            .iter()
            .flat_map(|q| {
                q.iter()
                    .copied()
                    .chain(std::iter::repeat(-1))
                    .take(SPARSE_WIDTH)
                    .flat_map(i32::to_le_bytes)
            })
            .collect();
        let (union, indices) = sparse_union(&selected, 511, m)?;
        let bound = 499;
        let mut selection = device.alloc(selected.len())?;
        stream.copy_h2d(&mut selection, &selected)?;
        let gathered = device.alloc(bound * 1024)?;
        let sorted = device.alloc(m * 4096 * 4)?;
        let ids = device.alloc(m * SPARSE_WIDTH * 4)?;
        let maps = device.alloc(m * SPARSE_WIDTH * 4)?;
        let counts = device.alloc(20)?;
        check(
            unsafe {
                g::hayate_glm_sparse_gather(
                    gathered.as_ptr().cast_mut(),
                    sorted.as_ptr().cast_mut(),
                    ids.as_ptr().cast_mut(),
                    maps.as_ptr().cast_mut(),
                    counts.as_ptr().cast_mut(),
                    selection.as_ptr(),
                    latent.as_ptr(),
                    m as u32,
                    bound as u32,
                    stream.as_raw(),
                )
            },
            "test union gather",
        )?;
        let actual = device.alloc(bound * 32768 * 2)?;
        check(
            unsafe {
                g::hayate_glm_sparse_project(
                    actual.as_ptr().cast_mut(),
                    gathered.as_ptr(),
                    weight.as_ptr(),
                    counts.as_ptr(),
                    bound as u32,
                    stream.as_raw(),
                )
            },
            "test dynamic union GEMM",
        )?;
        let host_gather: Vec<_> = union
            .iter()
            .flat_map(|&id| raw[id * 1024..(id + 1) * 1024].iter().copied())
            .collect();
        let mut reference_input = device.alloc(host_gather.len())?;
        stream.copy_h2d(&mut reference_input, &host_gather)?;
        let reference = device.alloc(union.len() * 32768 * 2)?;
        check(
            unsafe {
                hayate_text_linear(
                    reference.as_ptr().cast_mut(),
                    reference_input.as_ptr(),
                    weight.as_ptr(),
                    union.len() as u32,
                    32768,
                    512,
                    stream.as_raw(),
                )
            },
            "test ordinary union GEMM",
        )?;
        let mut expected = vec![0; reference.bytes()];
        let mut got = vec![0; reference.bytes()];
        stream.copy_d2h(&mut expected, &reference)?;
        stream.copy_d2h(&mut got, &actual)?;
        assert_eq!(got, expected, "m={m}");
        let mut mapped = vec![0; m * SPARSE_WIDTH * 4];
        stream.copy_d2h(&mut mapped, &maps)?;
        for (row, query) in indices.iter().enumerate() {
            let actual: Vec<_> = mapped
                [row * SPARSE_WIDTH * 4..(row * SPARSE_WIDTH + query.len()) * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            assert_eq!(&actual, query);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires supported CUDA GPU; grouped routed FFN bits, no checkpoint"]
fn glm53_async_grouped_expert_and_merge_bits() -> Result<()> {
    use super::experts::matrix_stride;
    use hayate_core::QuantSpec;
    let device = Device::select()?;
    let stream = device.stream()?;
    let bf = |v: f32| ((hayate_kernels::glm53::cpu::bf16(v).to_bits() >> 16) as u16).to_le_bytes();
    let m = 4;
    let input: Vec<_> = (0..m * 4096)
        .flat_map(|i| bf((i % 37) as f32 / 32. - 0.5))
        .collect();
    let ids: Vec<u32> = (0..m)
        .flat_map(|row| (0..8).map(move |j| ((j + row) % 11) as u32))
        .collect();
    let raw_ids: Vec<_> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    for (nv, down_fp8, rotate) in [
        (false, false, 0),
        (true, false, 0),
        (true, false, 32),
        (true, true, 32),
    ] {
        let stride = matrix_stride(nv);
        let mut image = vec![0; 2 * stride + matrix_stride(nv && !down_fp8)];
        for (part, (n, k)) in [(2048, 4096), (2048, 4096), (4096, 2048)]
            .into_iter()
            .enumerate()
        {
            let packed = nv && !(part == 2 && down_fp8);
            let bytes = if packed {
                QuantSpec::Nvfp4.matrix_bytes(n, k).unwrap() as usize
            } else {
                matrix_stride(false)
            };
            let payload = &mut image[part * stride..part * stride + bytes];
            if packed {
                payload[..(n * k / 2) as usize].fill(0x91);
                payload[(n * k / 2) as usize..bytes - 4].fill(0x38);
                payload[bytes - 4..].copy_from_slice(&(1f32 / 256.).to_le_bytes());
            } else {
                payload[..(n * k) as usize].fill(0x30);
                for scale in payload[(n * k) as usize..].as_chunks_mut::<4>().0 {
                    scale.copy_from_slice(&(1f32 / 256.).to_le_bytes());
                }
            }
        }
        let mut weight = device.alloc(image.len())?;
        stream.copy_h2d(&mut weight, &image)?;
        let dense = |part: usize| {
            let data = unsafe { weight.as_ptr().cast::<u8>().add(part * stride).cast() };
            let packed = nv && !(part == 2 && down_fp8);
            g::DenseWeight {
                data,
                scale: if packed {
                    std::ptr::null()
                } else {
                    unsafe { data.cast::<u8>().add(4096 * 2048).cast() }
                },
                format: if packed { 3 } else { 1 },
            }
        };
        let table = device.alloc_mapped(288 * 8)?;
        for e in 0..288 {
            unsafe {
                table.host_ptr().cast::<u64>().add(e).write(if e % 2 == 0 {
                    weight.as_ptr() as u64
                } else {
                    0
                });
            }
        }
        let mut x = device.alloc(input.len())?;
        stream.copy_h2d(&mut x, &input)?;
        let mut routes = device.alloc(raw_ids.len())?;
        stream.copy_h2d(&mut routes, &raw_ids)?;
        let groups = device.alloc(32 * 32)?;
        let count = device.alloc(4)?;
        let gathered = device.alloc(32 * 4 * 4096 * 2)?;
        let product = device.alloc(32 * 4 * 2048 * 2)?;
        let actual = device.alloc(m * 9 * 8192)?;
        check(
            unsafe {
                g::hayate_glm_async_groups(
                    groups.as_ptr().cast_mut(),
                    count.as_ptr().cast_mut(),
                    gathered.as_ptr().cast_mut(),
                    x.as_ptr(),
                    routes.as_ptr(),
                    table.device_ptr(),
                    m as u32,
                    stream.as_raw(),
                )
            },
            "test grouped routes",
        )?;
        check(
            unsafe {
                g::hayate_glm_async_gate(
                    product.as_ptr().cast_mut(),
                    gathered.as_ptr(),
                    groups.as_ptr(),
                    count.as_ptr(),
                    32,
                    nv as i32,
                    down_fp8 as i32,
                    stride,
                    stream.as_raw(),
                )
            },
            "test grouped gate",
        )?;
        if rotate != 0 && nv && !down_fp8 {
            check(
                unsafe {
                    g::hayate_glm_hadamard(
                        product.as_ptr().cast_mut(),
                        product.as_ptr(),
                        128,
                        2048,
                        rotate,
                        stream.as_raw(),
                    )
                },
                "test grouped rotation",
            )?;
        }
        check(
            unsafe {
                g::hayate_glm_async_down(
                    actual.as_ptr().cast_mut(),
                    product.as_ptr(),
                    groups.as_ptr(),
                    count.as_ptr(),
                    32,
                    nv as i32,
                    down_fp8 as i32,
                    stride,
                    stream.as_raw(),
                )
            },
            "test grouped down",
        )?;
        let host = device.alloc_mapped(m * 9 * 8192)?;
        unsafe {
            host.host_ptr().cast::<u8>().write_bytes(0x31, host.bytes());
        }
        check(
            unsafe {
                g::hayate_glm_async_merge(
                    actual.as_ptr().cast_mut(),
                    host.device_ptr(),
                    routes.as_ptr(),
                    table.device_ptr(),
                    m as u32,
                    stream.as_raw(),
                )
            },
            "test CPU merge",
        )?;
        let mut got = vec![0; actual.bytes()];
        stream.copy_d2h(&mut got, &actual)?;
        for row in 0..m {
            let images = device.alloc(64)?;
            let p = device.alloc(9 * 2048 * 2)?;
            let y = device.alloc(9 * 8192)?;
            let route = unsafe { routes.as_ptr().cast::<u32>().add(row * 8).cast() };
            let input = unsafe { x.as_ptr().cast::<u8>().add(row * 8192).cast() };
            check(
                unsafe {
                    g::hayate_glm_async_images(
                        images.as_ptr().cast_mut(),
                        route,
                        table.device_ptr(),
                        1,
                        stream.as_raw(),
                    )
                },
                "test row images",
            )?;
            check(
                unsafe {
                    g::hayate_glm_moe_gate_up(
                        p.as_ptr().cast_mut(),
                        input,
                        input,
                        images.as_ptr(),
                        dense(0),
                        dense(1),
                        nv as i32,
                        down_fp8 as i32,
                        stride,
                        stream.as_raw(),
                    )
                },
                "test ordinary gate",
            )?;
            if rotate != 0 && nv && !down_fp8 {
                check(
                    unsafe {
                        g::hayate_glm_hadamard(
                            p.as_ptr().cast_mut(),
                            p.as_ptr(),
                            8,
                            2048,
                            rotate,
                            stream.as_raw(),
                        )
                    },
                    "test row rotation",
                )?;
            }
            check(
                unsafe {
                    g::hayate_glm_moe_down(
                        y.as_ptr().cast_mut(),
                        p.as_ptr(),
                        images.as_ptr(),
                        dense(2),
                        nv as i32,
                        down_fp8 as i32,
                        stride,
                        stream.as_raw(),
                    )
                },
                "test ordinary down",
            )?;
            let mut reference = vec![0; y.bytes()];
            stream.copy_d2h(&mut reference, &y)?;
            for slot in 0..8 {
                let start = (row * 9 + slot) * 8192;
                if ids[row * 8 + slot].is_multiple_of(2) {
                    assert_eq!(
                        got[start..start + 8192],
                        reference[slot * 8192..(slot + 1) * 8192]
                    );
                } else {
                    assert!(got[start..start + 8192].iter().all(|&v| v == 0x31));
                }
            }
        }
    }
    Ok(())
}
