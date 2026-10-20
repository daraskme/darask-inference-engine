//! Routed-expert decode kernels in isolation: real layout (512 experts x 27/16 F H bytes per
//! layer), 10 random distinct experts per layer, a graph of dependent gate/up + down launches over
//! several layers' buffers (more than L2), with PDL.

use crate::{Bench, GIB};
use tsumuji_cuda::{DeviceBuffer, LaunchDims, Module, Result, args, launch, time_ms};

pub fn run(b: &Bench) -> Result<()> {
    const H: usize = 2560;
    const F: usize = 640;
    const E: usize = 512;
    const TOPK: usize = 10;
    let m = Module::load(tsumuji_kernels::DECODE_GEMV)?;
    let (gate_up, down) = (m.function("moe_gate_up")?, m.function("moe_down")?);
    let stride = F * H * 27 / 16;
    let layer_bytes = E * stride;
    let layers = 4; // 5.7 GB: no L2 reuse across the chain
    let pool = DeviceBuffer::new(layers * layer_bytes)?;
    b.fill_random(&pool, 11)?;
    // Scales: random e4m3 bytes can be NaN; that does not change the timing.
    let gs = DeviceBuffer::from_slice(&vec![1.0f32; E * 3])?;
    let steps = 96;
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut sels = Vec::new();
    for _ in 0..steps {
        let mut s = Vec::new();
        while s.len() < TOPK {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let e = (rng % E as u64) as i32;
            if !s.contains(&e) {
                s.push(e);
            }
        }
        sels.extend(s);
    }
    let sel = DeviceBuffer::from_slice(&sels)?;
    let x = DeviceBuffer::zeroed(H * 4)?;
    let h = DeviceBuffer::zeroed((TOPK + 1) * F * 4)?;
    let y = DeviceBuffer::zeroed((TOPK + 1) * H * 4)?;
    let peak = b
        .read_bw(
            Module::load(tsumuji_kernels::BENCH_BW)?.function("bw_read_cs")?,
            &DeviceBuffer::new(2 * GIB)?,
            5,
        )?
        .0;
    let s = &b.stream;
    let (hi, fi, ki) = (H as i32, F as i32, TOPK as i32);
    let null = 0u64;
    println!(
        "== moe: routed experts, {TOPK} of {E}, per layer (stream peak {:.0} GB/s)",
        peak / 1e9
    );
    for (label, gu_on, dn_on, pdl) in [
        ("gate/up + down, PDL", true, true, true),
        ("gate/up + down, no PDL", true, true, false),
        ("gate/up only, PDL", true, false, true),
        ("down only, PDL", false, true, true),
    ] {
        let g = s.capture(|st| {
            for i in 0..steps {
                let ex = pool.ptr + ((i % layers) * layer_bytes) as u64;
                let sp = sel.ptr + (i * TOPK * 4) as u64;
                if gu_on {
                    let mut a = args!(ex, gs.ptr, sp, null, null, x.ptr, h.ptr, hi, fi, ki);
                    let dims = LaunchDims {
                        grid: (F.div_ceil(16) as u32, TOPK as u32, 1),
                        block: (256, 1, 1),
                        smem: (H / 32 * 33 * 4) as u32,
                    };
                    unsafe { launch(gate_up, dims, st, pdl, &mut a)? };
                }
                if dn_on {
                    let mut a = args!(ex, gs.ptr, sp, null, null, h.ptr, y.ptr, hi, fi, ki);
                    let dims = LaunchDims {
                        grid: (H.div_ceil(64) as u32, TOPK as u32, 1),
                        block: (256, 1, 1),
                        smem: (F / 32 * 33 * 4) as u32,
                    };
                    unsafe { launch(down, dims, st, pdl, &mut a)? };
                }
            }
            Ok::<(), tsumuji_cuda::Error>(())
        })?;
        let us = time_ms(s, 2, 5, |_| g.launch(s))? * 1e3 / steps as f64;
        let bytes = (TOPK * stride) as f64
            * match (gu_on, dn_on) {
                (true, true) => 1.0,
                (true, false) => 2.0 / 3.0,
                _ => 1.0 / 3.0,
            };
        println!(
            "   {label:<24} {us:7.2} us/layer  {:6.1} GB/s ({:4.1}%)",
            bytes / (us * 1e-6) / 1e9,
            100.0 * bytes / (us * 1e-6) / peak
        );
        b.record(format!(
            r#"{{"bench":"moe","case":"{label}","us_per_layer":{us:.3},"gbps":{:.1}}}"#,
            bytes / (us * 1e-6) / 1e9
        ));
    }
    Ok(())
}
