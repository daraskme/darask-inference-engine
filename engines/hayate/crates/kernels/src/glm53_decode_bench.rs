//! Decode kernel timings at real GLM-5.3 shapes (one token). Weights rotate
//! through copies larger than L2 so every call streams from DRAM.
use super::ffi as g;
use crate::runtime::{Device, DeviceBuffer, Stream, check};
use hayate_core::{QuantSpec, Result};

const REPS: usize = 64;

fn bf(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
fn filled(device: &Device, stream: &Stream, bytes: usize, byte: u8) -> Result<DeviceBuffer> {
    let mut buffer = device.alloc(bytes)?;
    stream.copy_h2d(&mut buffer, &vec![byte; bytes])?;
    Ok(buffer)
}
fn bf16s(device: &Device, stream: &Stream, count: usize, modulus: usize) -> Result<DeviceBuffer> {
    let mut buffer = device.alloc(count * 2)?;
    let bytes: Vec<u8> = (0..count)
        .flat_map(|i| bf((i % modulus) as f32 / modulus as f32 - 0.5).to_le_bytes())
        .collect();
    stream.copy_h2d(&mut buffer, &bytes)?;
    Ok(buffer)
}
/// Per-tensor FP8 [n, k] + scale, `copies` distinct buffers.
fn fp8(
    device: &Device,
    stream: &Stream,
    n: usize,
    k: usize,
    copies: usize,
) -> Result<Vec<DeviceBuffer>> {
    let mut raw = vec![0x30; n * k];
    raw.extend(0.01f32.to_le_bytes());
    (0..copies)
        .map(|_| {
            let mut b = device.alloc(raw.len())?;
            stream.copy_h2d(&mut b, &raw)?;
            Ok(b)
        })
        .collect()
}
fn copies(bytes: usize) -> usize {
    (512usize << 20).div_ceil(bytes).clamp(2, 64)
}
/// Mean microseconds per call over REPS back-to-back submissions.
fn time(stream: &Stream, mut submit: impl FnMut(usize) -> Result<()>) -> Result<f64> {
    for i in 0..4 {
        submit(i)?;
    }
    let start = stream.event()?;
    let end = stream.event()?;
    start.record(stream)?;
    for i in 0..REPS {
        submit(i)?;
    }
    end.record(stream)?;
    end.synchronize()?;
    Ok(start.elapsed_ms(&end)? as f64 * 1000. / REPS as f64)
}
fn report(name: &str, us: f64, bytes: usize) {
    eprintln!(
        "glm_kbench kernel={name} us={us:.2} bytes={bytes} gb_s={:.1}",
        bytes as f64 / us / 1e3
    );
}

#[test]
#[ignore = "requires sm_120 GPU; decode kernel timings at real shapes (no checkpoint)"]
fn glm53_decode_kernel_bench() -> Result<()> {
    let device = Device::select()?;
    let stream = device.stream()?;
    let st = stream.as_raw();
    let x = bf16s(&device, &stream, 16384, 61)?;
    let mut out = device.alloc(160_000 * 2)?;
    let o = out.as_mut_ptr();
    // Dense FP8 GEMVs of one token (KDA, sparse, head).
    for (n, k) in [
        (8192usize, 4096usize),
        (4096, 8192),
        (16384, 1536),
        (4096, 16384),
        (1536, 4096),
        (4096, 1536),
        (512, 4096),
        (128, 4096),
        (8192, 128),
        (64, 4096),
        (12288, 4096),
        (4096, 12288),
        (2048, 4096),
        (4096, 2048),
        (154_880, 4096),
    ] {
        let w = fp8(&device, &stream, n, k, copies(n * k))?;
        let us = time(&stream, |i| {
            check(
                unsafe {
                    g::hayate_glm_dense(
                        o,
                        x.as_ptr(),
                        g::DenseWeight {
                            data: w[i % w.len()].as_ptr(),
                            scale: std::ptr::null(),
                            format: 2,
                        },
                        n as u32,
                        k as u32,
                        st,
                    )
                },
                "bench dense",
            )
        })?;
        report(&format!("dense_fp8_{n}x{k}"), us, n * k);
    }
    // Routed experts: 8 NVFP4 images + FP8 shared, all on the GPU.
    let stride = 4_718_848usize;
    let bytes = QuantSpec::Nvfp4.matrix_bytes(2048, 4096).unwrap() as usize;
    let mut host = vec![0; 3 * stride];
    for part in 0..3 {
        host[part * stride..part * stride + 4096 * 2048 / 2].fill(0x22);
        host[part * stride + 4096 * 2048 / 2..part * stride + bytes - 4].fill(0x38);
        host[part * stride + bytes - 4..part * stride + bytes]
            .copy_from_slice(&(1f32 / 4096.).to_le_bytes());
    }
    let image_count = 8 * copies(8 * 3 * stride).max(4);
    let mut images = Vec::new();
    for _ in 0..image_count {
        let mut image = device.alloc(host.len())?;
        stream.copy_h2d(&mut image, &host)?;
        images.push(image);
    }
    let sets = image_count / 8;
    let mut tables = Vec::new();
    for set in 0..sets {
        for misses in [0usize, 2] {
            let mut table = device.alloc(64)?;
            let raw: Vec<u8> = (0..8)
                .flat_map(|j| {
                    let p = if j < misses {
                        0
                    } else {
                        images[set * 8 + j].as_ptr() as u64
                    };
                    p.to_le_bytes()
                })
                .collect();
            stream.copy_h2d(&mut table, &raw)?;
            tables.push(table);
        }
    }
    let shared = fp8(&device, &stream, 2048, 4096, 2 * sets)?;
    let shared_down = fp8(&device, &stream, 4096, 2048, sets)?;
    let mut product = device.alloc(9 * 2048 * 2)?;
    let mut expert = device.alloc(9 * 4096 * 2)?;
    let dense = |b: &DeviceBuffer| g::DenseWeight {
        data: b.as_ptr(),
        scale: std::ptr::null(),
        format: 2,
    };
    for misses in [0usize, 2] {
        let hit = 8 - misses;
        let pick = |i: usize| (i % sets) * 2 + misses / 2;
        let us = time(&stream, |i| {
            check(
                unsafe {
                    g::hayate_glm_moe_gate_up(
                        product.as_mut_ptr(),
                        x.as_ptr(),
                        x.as_ptr(),
                        tables[pick(i)].as_ptr(),
                        dense(&shared[2 * (i % sets)]),
                        dense(&shared[2 * (i % sets) + 1]),
                        1,
                        0,
                        stride,
                        st,
                    )
                },
                "bench gate_up",
            )
        })?;
        report(
            &format!("moe_gate_up_hits{hit}"),
            us,
            hit * 2 * stride + 2 * 2048 * 4096,
        );
        let us = time(&stream, |i| {
            check(
                unsafe {
                    g::hayate_glm_moe_down(
                        expert.as_mut_ptr(),
                        product.as_ptr(),
                        tables[pick(i)].as_ptr(),
                        dense(&shared_down[i % sets]),
                        1,
                        0,
                        stride,
                        st,
                    )
                },
                "bench down",
            )
        })?;
        report(
            &format!("moe_down_hits{hit}"),
            us,
            hit * stride + 4096 * 2048,
        );
    }
    let mut ids = device.alloc(8 * 4)?;
    stream.copy_h2d(
        &mut ids,
        &[3u32, 9, 17, 40, 77, 120, 200, 287]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let mut weights = device.alloc(8 * 4)?;
    stream.copy_h2d(&mut weights, &0.3f32.to_le_bytes().repeat(8))?;
    let us = time(&stream, |_| {
        check(
            unsafe {
                g::hayate_glm_moe_reduce(o, expert.as_ptr(), ids.as_ptr(), weights.as_ptr(), st)
            },
            "bench reduce",
        )
    })?;
    report("moe_reduce", us, 9 * 4096 * 2);
    // Router: BF16 [288, 4096] + top-8.
    let router: Vec<_> = (0..copies(288 * 4096 * 2))
        .map(|_| bf16s(&device, &stream, 288 * 4096, 97))
        .collect::<Result<_>>()?;
    let bias = filled(&device, &stream, 288 * 4, 0)?;
    let mut logits = device.alloc(288 * 4)?;
    let mut route_w = device.alloc(8 * 4)?;
    let mut route_ids = device.alloc(8 * 4)?;
    let us = time(&stream, |i| {
        check(
            unsafe {
                g::hayate_glm_route_decode(
                    logits.as_mut_ptr(),
                    route_w.as_mut_ptr(),
                    route_ids.as_mut_ptr(),
                    x.as_ptr(),
                    router[i % router.len()].as_ptr(),
                    bias.as_ptr(),
                    st,
                )
            },
            "bench router",
        )
    })?;
    report("route_decode", us, 288 * 4096 * 2);
    // mHC: [24, 16384] BF16 function.
    let function: Vec<_> = (0..copies(24 * 16384 * 2))
        .map(|_| bf16s(&device, &stream, 24 * 16384, 17))
        .collect::<Result<_>>()?;
    let mut base = device.alloc(24 * 4)?;
    stream.copy_h2d(&mut base, &0.1f32.to_le_bytes().repeat(24))?;
    let mut scale = device.alloc(3 * 4)?;
    stream.copy_h2d(&mut scale, &0.001f32.to_le_bytes().repeat(3))?;
    let norm_w = bf16s(&device, &stream, 4096, 23)?;
    let mut map = device.alloc(24 * 4)?;
    let mut collapsed = device.alloc(4096 * 2)?;
    let us = time(&stream, |i| {
        check(
            unsafe {
                g::hayate_glm_hc_norm(
                    map.as_mut_ptr(),
                    collapsed.as_mut_ptr(),
                    o,
                    x.as_ptr(),
                    function[i % function.len()].as_ptr(),
                    base.as_ptr(),
                    scale.as_ptr(),
                    norm_w.as_ptr(),
                    st,
                )
            },
            "bench hc_norm",
        )
    })?;
    report("hc_norm", us, 24 * 16384 * 2);
    let branch = bf16s(&device, &stream, 4096, 29)?;
    let mut expanded = device.alloc(16384 * 2)?;
    let us = time(&stream, |_| {
        check(
            unsafe {
                g::hayate_glm_expand(
                    expanded.as_mut_ptr(),
                    x.as_ptr(),
                    branch.as_ptr(),
                    map.as_ptr(),
                    1,
                    4096,
                    st,
                )
            },
            "bench expand",
        )
    })?;
    report("expand", us, 16384 * 4);
    // KDA one token.
    let mut state = filled(&device, &stream, 64 * 128 * 128 * 4, 0)?;
    let mut decay = device.alloc(8192 * 4)?;
    stream.copy_h2d(&mut decay, &(-0.03f32).to_le_bytes().repeat(8192))?;
    let small = bf16s(&device, &stream, 8192, 19)?;
    let mut core = device.alloc(8192 * 2)?;
    let mut gated = device.alloc(8192 * 2)?;
    let us = time(&stream, |_| {
        check(
            unsafe {
                g::hayate_glm_kda_gated(
                    core.as_mut_ptr(),
                    gated.as_mut_ptr(),
                    state.as_mut_ptr(),
                    small.as_ptr(),
                    small.as_ptr(),
                    small.as_ptr(),
                    decay.as_ptr(),
                    small.as_ptr(),
                    small.as_ptr(),
                    small.as_ptr(),
                    st,
                )
            },
            "bench kda",
        )
    })?;
    report("kda_gated", us, 2 * 64 * 128 * 128 * 4);
    let mut history = filled(&device, &stream, 3 * 8192 * 2, 0)?;
    let conv_w = bf16s(&device, &stream, 8192 * 4, 7)?;
    let us = time(&stream, |_| {
        check(
            unsafe {
                g::hayate_glm_conv_step(
                    core.as_mut_ptr(),
                    history.as_mut_ptr(),
                    small.as_ptr(),
                    conv_w.as_ptr(),
                    8192,
                    st,
                )
            },
            "bench conv",
        )
    })?;
    report("conv_step", us, 8192 * 2 * 6);
    let dt = filled(&device, &stream, 8192 * 4, 0)?;
    let alog = filled(&device, &stream, 64 * 4, 0)?;
    let us = time(&stream, |_| {
        check(
            unsafe {
                g::hayate_glm_decay(
                    decay.as_mut_ptr(),
                    small.as_ptr(),
                    dt.as_ptr(),
                    alog.as_ptr(),
                    1,
                    64,
                    128,
                    st,
                )
            },
            "bench decay",
        )
    })?;
    report("decay", us, 8192 * 6);
    let us = time(&stream, |_| {
        check(
            unsafe {
                g::hayate_glm_point(
                    gated.as_mut_ptr(),
                    small.as_ptr(),
                    std::ptr::null(),
                    64,
                    0,
                    st,
                )
            },
            "bench point",
        )
    })?;
    report("point64", us, 64 * 4);
    // Head.
    let us = time(&stream, |_| {
        check(
            unsafe { g::hayate_glm_head_mean(collapsed.as_mut_ptr(), x.as_ptr(), 4096, st) },
            "bench head mean",
        )
    })?;
    report("head_mean", us, 16384 * 2);
    let us = time(&stream, |_| {
        check(
            unsafe {
                g::hayate_glm_norm(
                    o,
                    collapsed.as_ptr(),
                    norm_w.as_ptr(),
                    std::ptr::null(),
                    1,
                    4096,
                    0,
                    st,
                )
            },
            "bench norm",
        )
    })?;
    report("norm4096", us, 4096 * 6);
    let mut next = device.alloc(4)?;
    let vocab_logits = bf16s(&device, &stream, 154_880, 997)?;
    let us = time(&stream, |_| {
        check(
            unsafe { g::hayate_glm_greedy(next.as_mut_ptr(), vocab_logits.as_ptr(), 154_880, st) },
            "bench greedy",
        )
    })?;
    report("greedy", us, 154_880 * 2);
    // Absorbed sparse attention at 4096 tokens of context.
    let context = 4096usize;
    let wkt: Vec<_> = (0..copies(64 * 512 * 256 * 2))
        .map(|_| bf16s(&device, &stream, 64 * 512 * 256, 37))
        .collect::<Result<_>>()?;
    let q = bf16s(&device, &stream, 16384, 41)?;
    let mut qabs = device.alloc(64 * 4 * 512 * 2)?;
    let us = time(&stream, |i| {
        check(
            unsafe {
                g::hayate_glm_mla_absorb_q(
                    qabs.as_mut_ptr(),
                    q.as_ptr(),
                    wkt[i % wkt.len()].as_ptr(),
                    1,
                    st,
                )
            },
            "bench absorb q",
        )
    })?;
    report("mla_absorb_q", us, 64 * 512 * 256 * 2);
    let latent = bf16s(&device, &stream, context * 512, 43)?;
    let pools = bf16s(&device, &stream, context / 4 * 128, 47)?;
    let iq = bf16s(&device, &stream, 4096, 53)?;
    let iw = bf16s(&device, &stream, 32, 59)?;
    let mut selected = device.alloc(2051 * 4)?;
    let mut scores = device.alloc(context / 4 * 4)?;
    let us = time(&stream, |_| {
        check(
            unsafe {
                g::hayate_glm_mla_select(
                    selected.as_mut_ptr(),
                    scores.as_mut_ptr(),
                    iq.as_ptr(),
                    pools.as_ptr(),
                    iw.as_ptr(),
                    (context - 1) as u32,
                    1,
                    context / 4,
                    st,
                )
            },
            "bench select",
        )
    })?;
    report("mla_select_4k", us, context / 4 * 256);
    let mut olat = device.alloc(64 * 4 * 512 * 2)?;
    let mut part = device.alloc(8 * 64 * 4 * 512 * 4)?;
    let mut stats = device.alloc(8 * 64 * 4 * 2 * 4)?;
    let us = time(&stream, |_| {
        check(
            unsafe {
                g::hayate_glm_mla_attention(
                    olat.as_mut_ptr(),
                    part.as_mut_ptr(),
                    stats.as_mut_ptr(),
                    qabs.as_ptr(),
                    latent.as_ptr(),
                    selected.as_ptr(),
                    1,
                    8,
                    st,
                )
            },
            "bench mla attention",
        )
    })?;
    report("mla_attention_2051", us, 2051 * 1024);
    let us = time(&stream, |i| {
        check(
            unsafe {
                g::hayate_glm_mla_absorb_out(o, olat.as_ptr(), wkt[i % wkt.len()].as_ptr(), 1, st)
            },
            "bench absorb out",
        )
    })?;
    report("mla_absorb_out", us, 64 * 512 * 256 * 2);
    let us = time(&stream, |_| {
        check(
            unsafe { g::hayate_glm_copy(collapsed.as_mut_ptr(), x.as_ptr(), 1024, st) },
            "bench copy",
        )
    })?;
    report("copy1k", us, 1024);
    Ok(())
}

#[test]
#[ignore = "requires sm_120 GPU; stage-19 device decode kernel timings (no checkpoint)"]
fn glm53_device_kernel_bench() -> Result<()> {
    let device = Device::select()?;
    let stream = device.stream()?;
    let st = stream.as_raw();
    let x = bf16s(&device, &stream, 16384, 61)?;
    let mut ctl = filled(&device, &stream, 32, 0)?;
    let mut counter = filled(&device, &stream, 16, 0)?;
    let mut scratch = device.alloc(1 << 20)?;
    let s = scratch.as_mut_ptr();
    let at = |offset: usize| unsafe { s.cast::<u8>().add(offset).cast::<std::ffi::c_void>() };
    // mHC (24 x 16384 BF16 function rows), 24 blocks + last-block collapse.
    let function: Vec<_> = (0..copies(24 * 16384 * 2))
        .map(|_| bf16s(&device, &stream, 24 * 16384, 17))
        .collect::<Result<_>>()?;
    let mut base = device.alloc(24 * 4)?;
    stream.copy_h2d(&mut base, &0.1f32.to_le_bytes().repeat(24))?;
    let mut scale = device.alloc(3 * 4)?;
    stream.copy_h2d(&mut scale, &0.001f32.to_le_bytes().repeat(3))?;
    let norm_w = bf16s(&device, &stream, 4096, 23)?;
    let us = time(&stream, |i| {
        check(
            unsafe {
                g::hayate_glm_dev_hc_norm(
                    at(0),
                    at(4096),
                    at(16384),
                    x.as_ptr(),
                    function[i % function.len()].as_ptr(),
                    base.as_ptr(),
                    scale.as_ptr(),
                    norm_w.as_ptr(),
                    at(32768),
                    counter.as_mut_ptr(),
                    st,
                )
            },
            "bench dev hc",
        )
    })?;
    report("dev_hc_norm", us, 24 * 16384 * 2);
    // KDA input projections as one launch (q, k, v, f_a, b, g_a).
    let shapes = [
        (8192usize, 4096usize),
        (8192, 4096),
        (8192, 4096),
        (128, 4096),
        (64, 4096),
        (128, 4096),
    ];
    let sets = copies(25 * 4096 * 1024);
    let weights: Vec<Vec<DeviceBuffer>> = shapes
        .iter()
        .map(|&(n, k)| fp8(&device, &stream, n, k, sets))
        .collect::<Result<_>>()?;
    let mut outs: Vec<DeviceBuffer> = shapes
        .iter()
        .map(|&(n, _)| device.alloc(n * 2))
        .collect::<Result<_>>()?;
    let segments = |set: usize, outs: &mut [DeviceBuffer]| -> Vec<g::Segment> {
        shapes
            .iter()
            .zip(&weights)
            .zip(outs.iter_mut())
            .map(|((&(n, _), w), out)| g::Segment {
                w: g::DenseWeight {
                    data: w[set].as_ptr(),
                    scale: std::ptr::null(),
                    format: 2,
                },
                out: out.as_mut_ptr(),
                rows: n as u32,
            })
            .collect()
    };
    let us = time(&stream, |i| {
        let segs = segments(i % sets, &mut outs);
        check(
            unsafe {
                g::hayate_glm_dev_gemv(segs.as_ptr(), segs.len() as u32, x.as_ptr(), 4096, st)
            },
            "bench dev gemv",
        )
    })?;
    report(
        "dev_gemv_kda_inputs",
        us,
        shapes.iter().map(|&(n, k)| n * k).sum::<usize>(),
    );
    // KDA step (conv + f_b/g_b + decay + beta + recurrence + gated norm).
    let mut state = filled(&device, &stream, 64 * 128 * 128 * 4, 0)?;
    let mut hist: Vec<DeviceBuffer> = (0..3)
        .map(|_| filled(&device, &stream, 3 * 8192 * 2, 0))
        .collect::<Result<_>>()?;
    let conv_w = bf16s(&device, &stream, 8192 * 4, 7)?;
    let small = bf16s(&device, &stream, 8192, 19)?;
    let fbw = fp8(&device, &stream, 8192, 128, 2)?;
    let dt = filled(&device, &stream, 8192 * 4, 0)?;
    let alog = filled(&device, &stream, 64 * 4, 0)?;
    let o_norm = bf16s(&device, &stream, 128, 13)?;
    let dense = |b: &DeviceBuffer| g::DenseWeight {
        data: b.as_ptr(),
        scale: std::ptr::null(),
        format: 2,
    };
    let (h0, rest) = hist.split_at_mut(1);
    let (h1, h2) = rest.split_at_mut(1);
    let us = time(&stream, |_| {
        check(
            unsafe {
                g::hayate_glm_dev_kda(
                    at(0),
                    at(65536),
                    state.as_mut_ptr(),
                    h0[0].as_mut_ptr(),
                    h1[0].as_mut_ptr(),
                    h2[0].as_mut_ptr(),
                    small.as_ptr(),
                    small.as_ptr(),
                    small.as_ptr(),
                    conv_w.as_ptr(),
                    conv_w.as_ptr(),
                    conv_w.as_ptr(),
                    small.as_ptr(),
                    dense(&fbw[0]),
                    small.as_ptr(),
                    dense(&fbw[1]),
                    dt.as_ptr(),
                    alog.as_ptr(),
                    small.as_ptr(),
                    o_norm.as_ptr(),
                    st,
                )
            },
            "bench dev kda",
        )
    })?;
    report("dev_kda_step", us, 2 * 64 * 128 * 128 * 4 + 2 * 8192 * 128);
    // Router + top-8 + table + mailbox doorbell.
    let router: Vec<_> = (0..copies(288 * 4096 * 2))
        .map(|_| bf16s(&device, &stream, 288 * 4096, 97))
        .collect::<Result<_>>()?;
    let bias = filled(&device, &stream, 288 * 4, 0)?;
    let table = filled(&device, &stream, 288 * 8, 0)?;
    let mailbox = device.alloc_mapped(1 << 20)?;
    let us = time(&stream, |i| {
        check(
            unsafe {
                g::hayate_glm_dev_route(
                    at(0),
                    at(4096),
                    at(8192),
                    at(12288),
                    at(16384),
                    x.as_ptr(),
                    router[i % router.len()].as_ptr(),
                    bias.as_ptr(),
                    table.as_ptr(),
                    mailbox.device_ptr(),
                    ctl.as_mut_ptr(),
                    counter.as_mut_ptr(),
                    0.,
                    0.,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null_mut(),
                    st,
                )
            },
            "bench dev router",
        )
    })?;
    report("dev_route_doorbell", us, 288 * 4096 * 2);
    let us = time(&stream, |_| {
        check(
            unsafe {
                g::hayate_glm_dev_moe_finish(
                    at(65536),
                    at(98304),
                    at(262144),
                    x.as_ptr(),
                    at(131072),
                    at(131072),
                    at(8192),
                    at(4096),
                    at(12288),
                    mailbox.device_ptr(),
                    at(16384),
                    g::Deferred {
                        mailbox: std::ptr::null(),
                        ids: std::ptr::null(),
                        weights: std::ptr::null(),
                        images: std::ptr::null(),
                        post: std::ptr::null(),
                        plan: std::ptr::null(),
                    },
                    st,
                )
            },
            "bench dev finish",
        )
    })?;
    report("dev_moe_finish_8miss", us, 8 * 8192);
    Ok(())
}

/// DRAM sharing between CPU expert reads and GPU zero-copy reads of pinned
/// host memory: GPU alone, CPU alone (8 P-core threads), and both over the
/// same window (CPU threads keep streaming while the GPU reads).
#[test]
#[ignore = "requires main GPU; 2 x 2 GiB pinned host buffers, P cores 0,2,..,14"]
fn glm53_zero_copy_bandwidth() -> Result<()> {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Instant;
    let device = Device::select()?;
    let stream = device.stream()?;
    let bytes = 2usize << 30;
    let gpu_buffer = device.alloc_mapped(bytes)?;
    let mut cpu_buffer = device.alloc_pinned(bytes)?;
    cpu_buffer.as_mut_slice().fill(1);
    let mut sink = device.alloc(4)?;
    let sink_ptr = sink.as_mut_ptr() as usize;
    let gpu_read = |reps: usize| -> Result<f64> {
        let start = Instant::now();
        for _ in 0..reps {
            check(
                unsafe {
                    g::hayate_glm_dev_read_probe(
                        gpu_buffer.device_ptr(),
                        bytes,
                        sink_ptr as *mut _,
                        376,
                        stream.as_raw(),
                    )
                },
                "zero-copy probe",
            )?;
        }
        stream.synchronize()?;
        Ok((reps * bytes) as f64 / start.elapsed().as_secs_f64() / 1e9)
    };
    let cpu_slice = cpu_buffer.as_slice();
    let done = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    // CPU readers stream 1 MiB chunks until `stop`, counting bytes in `done`.
    fn cpu_stream<'scope, 'env>(
        scope: &'scope std::thread::Scope<'scope, 'env>,
        slice: &'env [u8],
        done: &'env AtomicU64,
        stop: &'env AtomicBool,
    ) {
        let bytes = slice.len();
        for t in 0..8usize {
            let part = &slice[t * bytes / 8..(t + 1) * bytes / 8];
            scope.spawn(move || {
                let _ = crate::cpu::pin_thread((2 * t) as u16);
                let mut acc = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    for chunk in part.chunks(1 << 20) {
                        for word in chunk.as_chunks::<8>().0 {
                            acc ^= u64::from_le_bytes(*word);
                        }
                        done.fetch_add(chunk.len() as u64, Ordering::Relaxed);
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                    }
                }
                std::hint::black_box(acc);
            });
        }
    }
    gpu_read(1)?;
    let gpu = gpu_read(4)?;
    let cpu_window = |seconds: f64| -> f64 {
        done.store(0, Ordering::Relaxed);
        stop.store(false, Ordering::Relaxed);
        std::thread::scope(|scope| {
            cpu_stream(scope, cpu_slice, &done, &stop);
            std::thread::sleep(std::time::Duration::from_secs_f64(0.2));
            let start = (Instant::now(), done.load(Ordering::Relaxed));
            std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
            let rate = (done.load(Ordering::Relaxed) - start.1) as f64
                / start.0.elapsed().as_secs_f64()
                / 1e9;
            stop.store(true, Ordering::Relaxed);
            rate
        })
    };
    let cpu = cpu_window(1.);
    // Both: CPU streams continuously; measure CPU bytes over the GPU window.
    done.store(0, Ordering::Relaxed);
    stop.store(false, Ordering::Relaxed);
    let (both_gpu, both_cpu) = std::thread::scope(|scope| {
        cpu_stream(scope, cpu_slice, &done, &stop);
        std::thread::sleep(std::time::Duration::from_secs_f64(0.2));
        let start = (Instant::now(), done.load(Ordering::Relaxed));
        let g = gpu_read(8);
        let c =
            (done.load(Ordering::Relaxed) - start.1) as f64 / start.0.elapsed().as_secs_f64() / 1e9;
        stop.store(true, Ordering::Relaxed);
        (g, c)
    });
    let both_gpu = both_gpu?;
    eprintln!(
        "glm_zero_copy gpu_only_gb_s={gpu:.1} cpu_only_gb_s={cpu:.1} both_gpu_gb_s={both_gpu:.1} both_cpu_gb_s={both_cpu:.1} both_total_gb_s={:.1}",
        both_gpu + both_cpu
    );
    Ok(())
}
