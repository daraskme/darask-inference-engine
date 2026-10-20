use super::{HC, HIDDEN, Layer, relative_l2};
use hayate_core::{Error, Result};
use hayate_kernels::runtime::Device;
use std::{collections::BTreeMap, path::PathBuf};

fn values(raw: &[u8], f32_values: bool) -> Vec<f64> {
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
}

#[test]
#[ignore = "requires supported CUDA GPU; real KV dimensions, no checkpoint"]
fn glm53_sparse_union_projection_and_attention_bits() -> Result<()> {
    use hayate_kernels::{glm53::ffi as g, runtime::check, text::ffi::hayate_text_linear};
    let device = Device::select()?;
    let stream = device.stream()?;
    let bf = |v: f32| ((hayate_kernels::glm53::cpu::bf16(v).to_bits() >> 16) as u16).to_le_bytes();
    let (n, k, rows) = (32768usize, 512usize, 271usize);
    let latent: Vec<_> = (0..rows * k)
        .flat_map(|i| bf((i % 113) as f32 / 53. - 1.))
        .collect();
    let mut input = device.alloc(latent.len())?;
    stream.copy_h2d(&mut input, &latent)?;
    // Non-unit FP8 scales and signed values, restored exactly as linear_ptrs.
    let payload = n * k;
    let mut raw: Vec<_> = (0..payload)
        .map(|i| ((i * 17 % 111) as u8) | if i % 3 == 0 { 128 } else { 0 })
        .collect();
    raw.extend(0.0713f32.to_le_bytes());
    let mut weight = device.alloc(raw.len())?;
    stream.copy_h2d(&mut weight, &raw)?;
    let restored = device.alloc(n * k * 2)?;
    check(
        unsafe {
            g::hayate_glm_dense_unpack(
                restored.as_ptr().cast_mut(),
                g::DenseWeight {
                    data: weight.as_ptr(),
                    scale: std::ptr::null(),
                    format: 2,
                },
                n as u32,
                k as u32,
                stream.as_raw(),
            )
        },
        "glm union weight restore",
    )?;
    let union = device.alloc(rows * n * 2)?;
    check(
        unsafe {
            hayate_text_linear(
                union.as_ptr().cast_mut(),
                input.as_ptr(),
                restored.as_ptr(),
                rows as u32,
                n as u32,
                k as u32,
                stream.as_raw(),
            )
        },
        "glm union projection",
    )?;
    for (offset, count, stride) in [
        (0usize, 1usize, 1usize),
        (7, 3, 31),
        (13, 127, 2),
        (129, 131, 1),
    ] {
        let indices: Vec<_> = (0..count).map(|i| offset + i * stride).collect();
        let mut selected = device.alloc(count * 4)?;
        stream.copy_h2d(
            &mut selected,
            &indices
                .iter()
                .flat_map(|&i| (i as u32).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut query_input = device.alloc(count * k * 2)?;
        let gathered: Vec<_> = indices
            .iter()
            .flat_map(|&i| latent[i * k * 2..(i + 1) * k * 2].iter().copied())
            .collect();
        stream.copy_h2d(&mut query_input, &gathered)?;
        let projected = device.alloc(count * n * 2)?;
        check(
            unsafe {
                hayate_text_linear(
                    projected.as_ptr().cast_mut(),
                    query_input.as_ptr(),
                    restored.as_ptr(),
                    count as u32,
                    n as u32,
                    k as u32,
                    stream.as_raw(),
                )
            },
            "glm per-query projection",
        )?;
        let mut q = device.alloc(16384 * 2)?;
        stream.copy_h2d(
            &mut q,
            &(0..16384)
                .flat_map(|i| bf((i % 31) as f32 / 113.))
                .collect::<Vec<_>>(),
        )?;
        let probs = device.alloc(64 * count * 2)?;
        let mut outputs = Vec::new();
        let mut split = Vec::new();
        for indexed in [false, true] {
            let keys = device.alloc(count * 16384 * 2)?;
            let vals = device.alloc(count * 16384 * 2)?;
            check(
                unsafe {
                    if indexed {
                        g::hayate_glm_split_selected(
                            keys.as_ptr().cast_mut(),
                            vals.as_ptr().cast_mut(),
                            union.as_ptr(),
                            selected.as_ptr(),
                            count as u32,
                            stream.as_raw(),
                        )
                    } else {
                        g::hayate_glm_split(
                            keys.as_ptr().cast_mut(),
                            vals.as_ptr().cast_mut(),
                            projected.as_ptr(),
                            count as u32,
                            64,
                            256,
                            stream.as_raw(),
                        )
                    }
                },
                "glm union/per-query split",
            )?;
            let output = device.alloc(16384 * 2)?;
            check(
                unsafe {
                    g::hayate_glm_attention_step(
                        output.as_ptr().cast_mut(),
                        probs.as_ptr().cast_mut(),
                        q.as_ptr(),
                        keys.as_ptr(),
                        vals.as_ptr(),
                        count as u32,
                        stream.as_raw(),
                    )
                },
                "glm union/per-query attention",
            )?;
            let mut key_bytes = vec![0; keys.bytes()];
            let mut val_bytes = vec![0; vals.bytes()];
            stream.copy_d2h(&mut key_bytes, &keys)?;
            stream.copy_d2h(&mut val_bytes, &vals)?;
            split.push((key_bytes, val_bytes));
            let mut bytes = vec![0; output.bytes()];
            stream.copy_d2h(&mut bytes, &output)?;
            outputs.push(bytes);
        }
        assert_eq!(
            split[0], split[1],
            "KV bits at offset={offset} count={count}"
        );
        assert_eq!(
            outputs[0], outputs[1],
            "attention bits at offset={offset} count={count}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires main GPU and glm53_reference + glm53_convert; see glm53/README.md"]
fn glm53_five_layer_golden() -> Result<()> {
    let root = PathBuf::from(std::env::var("GLM53_GOLDEN").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/golden/glm53".into()
    }));
    let device = Device::select()?;
    let mut chained = std::fs::read(root.join("layer0/input.bin"))?;
    if !chained.len().is_multiple_of(HC * HIDDEN * 2) {
        return Err(Error::Weights("GLM golden input layout".into()));
    }
    let tokens = chained.len() / (HC * HIDDEN * 2);
    let mut report = BTreeMap::new();
    for index in 0..5 {
        let folder = root.join(format!("layer{index}"));
        let mut layer = Layer::load(&device, &folder.join("weights.hyt"), index, tokens)?;
        let input = std::fs::read(folder.join("input.bin"))?;
        for _ in 0..2 {
            layer.prefill(&input)?;
        }
        let mut times = Vec::new();
        for _ in 0..5 {
            times.push(layer.prefill(&input)?.cuda_ms);
        }
        let actual = layer.prefill(&input)?;
        let mut stages = vec![
            ("attn_collapsed", "attn_hc_2", false),
            ("attn_norm", "input_layernorm", false),
            ("attention", "self_attn", false),
            ("ffn_collapsed", "ffn_hc_2", false),
            ("ffn_norm", "post_attention_layernorm", false),
            ("mlp", "mlp", false),
            ("output", "output", false),
            ("attention_residual", "attention_residual", false),
        ];
        if index % 4 == 3 {
            stages[2].1 = "self_attn_0";
            stages.extend([
                ("qr", "self_attn.q_a_layernorm", false),
                ("ikn", "self_attn.indexer.k_norm", false),
            ]);
        } else {
            stages.extend([
                ("decay", "self_attn.forget_gate", true),
                ("kda", "kda_0", false),
                ("state", "kda_1", true),
                ("gated", "self_attn.o_norm", false),
            ]);
        }
        if index >= 3 {
            stages.extend([("router_logits", "mlp.gate_0", true)]);
        }
        for (name, reference, f32_values) in stages {
            let raw = std::fs::read(folder.join(format!("{reference}.bin")))?;
            if raw.len() != actual.tensors[name].len() {
                return Err(Error::Weights(format!("{name}: golden byte count")));
            }
            let error = relative_l2(
                &values(&actual.tensors[name], f32_values),
                &values(&raw, f32_values),
            )?;
            eprintln!("layer{index}.{name}: relative L2={error:.9}");
            assert!(error < 0.01, "layer{index}.{name}: {error}");
            report.insert(format!("layer{index}.{name}"), error);
        }
        for (map, site) in [("attn_map", "attn_hc"), ("ffn_map", "ffn_hc")] {
            let full = values(&actual.tensors[map], true);
            for (suffix, range) in [(0, 4..8), (1, 8..24)] {
                let got: Vec<_> = full
                    .as_chunks::<24>()
                    .0
                    .iter()
                    .flat_map(|row| row[range.clone()].iter().copied())
                    .collect();
                let reference = values(
                    &std::fs::read(folder.join(format!("{site}_{suffix}.bin")))?,
                    true,
                );
                let error = relative_l2(&got, &reference)?;
                assert!(error < 0.01, "{site}_{suffix}: {error}");
                report.insert(format!("layer{index}.{site}_{suffix}"), error);
            }
        }
        // topk(sorted=False) order is unspecified. Compare ID->weight, not order.
        if index >= 3 {
            let ids = std::fs::read(folder.join("mlp.gate_2.bin"))?;
            let weights = std::fs::read(folder.join("mlp.gate_1.bin"))?;
            let ref_ids: Vec<_> = ids
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as i32)
                .collect();
            let got_ids: Vec<_> = actual.tensors["route_ids"]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect();
            let ref_w = values(&weights, true);
            let got_w = values(&actual.tensors["route_weights"], true);
            for t in 0..tokens {
                let reference: BTreeMap<_, _> = ref_ids[t * 8..t * 8 + 8]
                    .iter()
                    .copied()
                    .zip(ref_w[t * 8..t * 8 + 8].iter().copied())
                    .collect();
                let got: BTreeMap<_, _> = got_ids[t * 8..t * 8 + 8]
                    .iter()
                    .copied()
                    .zip(got_w[t * 8..t * 8 + 8].iter().copied())
                    .collect();
                assert_eq!(
                    got.keys().collect::<Vec<_>>(),
                    reference.keys().collect::<Vec<_>>()
                );
                assert!(
                    relative_l2(
                        &got.values().copied().collect::<Vec<_>>(),
                        &reference.values().copied().collect::<Vec<_>>()
                    )? < 0.01
                );
            }
        }
        if index % 4 == 3 {
            let ids = std::fs::read(folder.join("self_attn.indexer.bin"))?;
            let reference: Vec<_> = ids
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect();
            let actual_ids: Vec<_> = actual.tensors["selected"]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect();
            for t in 0..tokens {
                let visible = |row: &[i32]| {
                    let mut ids: Vec<_> = row.iter().copied().filter(|&id| id >= 0).collect();
                    ids.sort_unstable();
                    ids
                };
                let a = visible(&actual_ids[t * 2051..(t + 1) * 2051]);
                let r = visible(&reference[t * 2051..(t + 1) * 2051]);
                assert!(a.iter().all(|&id| id <= t as i32));
                assert_eq!(a, r, "DSA visible token set");
            }
        }
        let output = layer.prefill(&chained)?;
        chained = output.tensors["output"].clone();
        let error = relative_l2(
            &values(&chained, false),
            &values(&std::fs::read(folder.join("output.bin"))?, false),
        )?;
        assert!(error < 0.01, "layer{index}.chained {error}");
        report.insert(format!("layer{index}.chained"), error);
        eprintln!(
            "layer{index}: warmup=2 repeat=5 CUDA event ms={times:?}; excludes loading and input/output copies"
        );
    }
    eprintln!("GLM relative L2 report: {report:?}");
    if let Ok(path) = std::env::var("GLM53_REPORT") {
        let entries: Vec<_> = report
            .iter()
            .map(|(name, error)| format!("\"{name}\":{error}"))
            .collect();
        std::fs::write(path, format!("{{{}}}\n", entries.join(",")))?;
    }
    Ok(())
}

#[test]
#[ignore = "requires main GPU and GLM53_HYT; native BF16 teacher-forcing row alignment"]
fn glm53_eval_matches_forced_steps() -> Result<()> {
    use super::{Executor, Options};
    let hyt = PathBuf::from(std::env::var("GLM53_HYT").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53-fp8".into()
    }));
    let device = Device::select()?;
    let mut executor = Executor::load(
        &device,
        &hyt,
        Options {
            layers: 5,
            context: 4,
            host_bytes: Some(0),
            allow_disk: true,
            prefill_chunk: 1,
            ..Options::default()
        },
    )?;
    let ids = [17, 19, 23, 29];
    let out = std::env::temp_dir().join(format!("hayate-glm-eval-gpu-{}", std::process::id()));
    executor.evaluate(&ids, 1, &out)?;
    assert!(executor.evaluate(&ids, 1, &out).is_err());
    let raw = std::fs::read(out.join("eval_logits.bf16.bin"))?;
    executor.reset()?;
    executor.step(ids[0])?;
    let width = raw.len() / 3;
    for (row, &id) in ids[1..].iter().enumerate() {
        executor.step(id)?;
        let actual = executor.logits()?;
        let expected: Vec<_> = raw[row * width..(row + 1) * width]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16) as f64)
            .collect();
        assert_eq!(actual, expected);
    }
    std::fs::remove_dir_all(out)?;
    Ok(())
}

#[test]
#[ignore = "requires main GPU, all-layer HYT and glm53_reference --max-tokens; see README"]
fn glm53_generation_golden() -> Result<()> {
    use super::{Executor, Options};
    let root = PathBuf::from(std::env::var("GLM53_GOLDEN").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/golden/glm53-45".into()
    }));
    let hyt = PathBuf::from(std::env::var("GLM53_HYT").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53-fp8".into()
    }));
    let layers = std::env::var("GLM53_LAYERS")
        .unwrap_or_else(|_| "45".into())
        .parse()
        .map_err(|_| Error::InvalidPlan("GLM53_LAYERS".into()))?;
    let tolerance: f64 = std::env::var("GLM53_LOGITS_TOLERANCE")
        .unwrap_or_else(|_| "0.01".into())
        .parse()
        .map_err(|_| Error::InvalidPlan("GLM53_LOGITS_TOLERANCE".into()))?;
    if !tolerance.is_finite() || tolerance <= 0. {
        return Err(Error::InvalidPlan(
            "GLM53_LOGITS_TOLERANCE must be finite and positive".into(),
        ));
    }
    let ids: Vec<_> = std::fs::read(root.join("input_ids.bin"))?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b) as u32)
        .collect();
    let steps: usize = std::env::var("GLM53_STEPS")
        .unwrap_or_else(|_| "4".into())
        .parse()
        .map_err(|_| Error::InvalidPlan("GLM53_STEPS".into()))?;
    let device = Device::select()?;
    let mut executor = Executor::load(
        &device,
        &hyt,
        Options {
            layers,
            context: ids.len() + steps,
            allow_disk: true,
            ..Options::default()
        },
    )?;
    let mut next = executor.prefill(&ids)?;
    for step in 0..steps {
        let reference = values(&std::fs::read(root.join(format!("step{step}.bin")))?, true);
        let actual = executor.logits()?;
        let error = relative_l2(&actual, &reference)?;
        let greedy = reference
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0 as u32;
        eprintln!(
            "layers={layers} step={step} logits_L2={error:.9} greedy={} reference={greedy}",
            next
        );
        assert!(error < tolerance, "logits error {error}");
        assert_eq!(next, greedy, "greedy step {step}");
        if step + 1 < steps {
            next = executor.step(next)?;
        }
    }
    eprintln!(
        "memory={:?}, cache={:?}",
        executor.memory,
        executor.cache_counts()
    );
    Ok(())
}

#[test]
#[ignore = "requires main GPU and real GLM HYT; compares grouped vs sequential and reset"]
fn glm53_prefill_matches_step() -> Result<()> {
    use super::{Executor, Options};
    let hyt = PathBuf::from(std::env::var("GLM53_HYT").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53-nvfp4".into()
    }));
    let layers = std::env::var("GLM53_LAYERS")
        .unwrap_or_else(|_| "8".into())
        .parse()
        .map_err(|_| Error::InvalidPlan("GLM53_LAYERS".into()))?;
    let count: usize = std::env::var("GLM53_PREFILL_TOKENS")
        .unwrap_or_else(|_| "17".into())
        .parse()
        .map_err(|_| Error::InvalidPlan("GLM53_PREFILL_TOKENS".into()))?;
    let device = Device::select()?;
    let mut executor = Executor::load(
        &device,
        &hyt,
        Options {
            layers,
            context: count + 4,
            allow_disk: true,
            ..Options::default()
        },
    )?;
    let ids: Vec<_> = (0..count).map(|i| (17 + i % 11) as u32).collect();
    let next = executor.prefill(&ids)?;
    let prefill = executor.logits()?;
    let continued = executor.step(next)?;
    let decode = executor.logits()?;
    executor.reset()?;
    let mut sequential = 0;
    for &id in &ids {
        sequential = executor.step(id)?;
    }
    assert_eq!(next, sequential);
    let error = relative_l2(&executor.logits()?, &prefill)?;
    assert!(error < 0.01, "grouped/sequential prefill L2={error}");
    assert_eq!(continued, executor.step(sequential)?);
    let error = relative_l2(&executor.logits()?, &decode)?;
    assert!(error < 0.01, "grouped/sequential continuation L2={error}");
    executor.reset()?;
    assert_eq!(next, executor.prefill(&ids)?);
    assert_eq!(executor.logits()?, prefill);
    Ok(())
}

#[test]
#[ignore = "requires main GPU and GLM53_HYT; legacy/eager comparison and changed-route Graph replay"]
fn glm53_decode_stage6_graph() -> Result<()> {
    use super::{Executor, Options};
    let hyt = PathBuf::from(std::env::var("GLM53_HYT").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53-nvfp4-plain".into()
    }));
    let device = Device::select()?;
    let mut executor = Executor::load(
        &device,
        &hyt,
        Options {
            layers: 5,
            context: 12,
            host_bytes: Some(0),
            allow_disk: true,
            prefill_chunk: 1,
            expert_prefetch: true,
            pcie_frac: 1.,
            ..Options::default()
        },
    )?;
    let ids = [17, 19, 23, 29, 31, 37, 41, 43];
    let mut reference = Vec::new();
    executor.decode_reference(true);
    for id in ids {
        executor.step(id)?;
        reference.push(executor.logits()?);
    }
    executor.reset()?;
    executor.decode_reference(false);
    let mut graphs = Vec::new();
    let mut greedy = Vec::new();
    let mut worst = 0f64;
    for (i, id) in ids.into_iter().enumerate() {
        greedy.push(executor.step(id)?);
        let actual = executor.logits()?;
        worst = worst.max(relative_l2(&actual, &reference[i])?);
        graphs.push(actual);
    }
    // This is an arithmetic regression bound, not the 45-layer KLD gate.
    assert!(worst < 0.02, "legacy/Graph logits relative L2={worst}");
    executor.reset()?;
    executor.decode_reference(false);
    executor.disable_graphs();
    for (i, id) in ids.into_iter().enumerate() {
        assert_eq!(executor.step(id)?, greedy[i], "greedy step {i}");
        assert_eq!(executor.logits()?, graphs[i], "Graph replay step {i}");
    }
    eprintln!(
        "GLM stage6: 8 forced tokens, legacy/Graph max logits L2={worst}, eager/Graph bit equality"
    );
    Ok(())
}

#[test]
#[ignore = "requires main GPU, 45-layer GLM53_HYT and 100 GB host RAM; CPU misses with Graph replay"]
fn glm53_decode_cpu_graph_matches_eager() -> Result<()> {
    use super::{Executor, Options};
    let hyt = PathBuf::from(std::env::var("GLM53_HYT").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53-stage6-fp8-dense".into()
    }));
    let device = Device::select()?;
    let mut executor = Executor::load(
        &device,
        &hyt,
        Options {
            context: 12,
            maintenance_interval: 0,
            ..Options::default()
        },
    )?;
    let ids = [17, 19, 23, 29, 31, 37, 41, 43];
    let mut graphs = Vec::new();
    for id in ids {
        let greedy = executor.step(id)?;
        graphs.push((greedy, executor.logits()?));
    }
    let (hits, routes, disk) = executor.cache_counts();
    assert!(routes > hits, "test must exercise host CPU experts");
    assert_eq!(disk, 0);
    executor.reset()?;
    executor.disable_graphs();
    for (i, id) in ids.into_iter().enumerate() {
        assert_eq!(executor.step(id)?, graphs[i].0, "greedy step {i}");
        assert_eq!(executor.logits()?, graphs[i].1, "CPU Graph replay step {i}");
    }
    eprintln!(
        "GLM stage10: 45 layers, 8 forced tokens, {} CPU experts, eager/Graph bit equality",
        routes - hits
    );
    Ok(())
}

#[test]
#[ignore = "requires main GPU and 45-layer GLM53_HYT; previous KLD <= 0.01, all GEMM <= 0.02"]
fn glm53_prefill_matches_tokenwise_prefix_logits() -> Result<()> {
    use super::{Executor, Options};
    let hyt = PathBuf::from(std::env::var("GLM53_HYT").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53-stage6-fp8-dense".into()
    }));
    let device = Device::select()?;
    let mut executor = Executor::load(
        &device,
        &hyt,
        Options {
            context: 261,
            prefill_chunk: 128,
            maintenance_interval: 0,
            // Diagnostics: 1 computes every miss on the GPU in both paths.
            pcie_frac: std::env::var("GLM53_PMATCH_PCIE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.),
            ..Options::default()
        },
    )?;
    // GLM53_PMATCH_IDS: little-endian i32 token file (natural text) instead
    // of the periodic synthetic sequence.
    let ids: Vec<u32> = match std::env::var("GLM53_PMATCH_IDS") {
        Ok(path) => std::fs::read(path)?
            .as_chunks::<4>()
            .0
            .iter()
            .take(261)
            .map(|b| u32::from_le_bytes(*b))
            .collect(),
        Err(_) => (0..261).map(|i| 17 + i % 101).collect(),
    };
    assert_eq!(ids.len(), 261);
    let only_all = std::env::var("GLM53_PMATCH_ALL_ONLY").is_ok();
    let mut failures = Vec::new();
    let mut reference = Vec::new();
    for (row, &id) in ids.iter().enumerate() {
        executor.step(id)?;
        if row >= 256 {
            reference.push(executor.logits()?);
        }
    }
    let logp = |values: &[f64]| {
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z = values.iter().map(|v| (v - max).exp()).sum::<f64>().ln() + max;
        values.iter().map(|v| v - z).collect::<Vec<_>>()
    };
    let mut all_logits: Vec<Vec<f64>> = Vec::new();
    for (start, all) in [
        (0usize, false),
        (1, false),
        (3, false),
        (127, false),
        (0, true),
        (0, true),
    ] {
        if only_all && !all {
            continue;
        }
        let repeat = all && !all_logits.is_empty();
        executor.prefill_all_experts(all);
        executor.reset()?;
        for &id in &ids[..start] {
            executor.step(id)?;
        }
        executor.prefill(&ids[start..257])?;
        for row in 0..5 {
            if row > 0 {
                executor.step(ids[256 + row])?;
            }
            let actual = executor.logits()?;
            assert!(actual.iter().all(|v| v.is_finite()));
            if all {
                if repeat {
                    assert_eq!(actual, all_logits[row], "all prefill replay row={row}");
                } else {
                    all_logits.push(actual.clone());
                }
            }
            let p = logp(&reference[row]);
            let q = logp(&actual);
            let kld = p
                .iter()
                .zip(&q)
                .map(|(p, q)| p.exp() * (p - q))
                .sum::<f64>()
                .max(0.);
            // Tensor-core GEMM and GEMV have different FP32 reduction orders;
            // 45-layer W4A16 all-prefill measured KLD=0.01435 on 25f0cd1.
            let limit = if all { 0.02 } else { 0.01 };
            let argmax = |v: &[f64]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            eprintln!(
                "prefill_start={start} all={all} suffix_row={row} KLD={kld:.9} limit={limit} repeat={repeat} top1_match={}",
                argmax(&reference[row]) == argmax(&actual)
            );
            if kld > limit {
                failures.push(format!("start={start} all={all} row={row} KLD={kld:.6}"));
            }
        }
    }
    assert!(failures.is_empty(), "prefill KLD over limit: {failures:?}");
    Ok(())
}

#[test]
#[ignore = "requires main GPU and 45-layer GLM53_HYT; session export/import bits, prefill cancel"]
fn glm53_session_roundtrip_and_cancel() -> Result<()> {
    use super::{Executor, Options, PrefillOutcome};
    let hyt = PathBuf::from(std::env::var("GLM53_HYT").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53-stage6-fp8-dense".into()
    }));
    let device = Device::select()?;
    let mut executor = Executor::load(
        &device,
        &hyt,
        Options {
            context: 2048,
            prefill_chunk: 256,
            maintenance_interval: 0,
            ..Options::default()
        },
    )?;
    executor.prefill_all_experts(true);
    let ids: Vec<u32> = (0..1030).map(|i| 17 + (i * 7919 % 9973)).collect();
    let (prefix, suffix) = ids.split_at(1024);
    // Reference: continuous prefill then steps.
    executor.reset()?;
    executor.prefill(prefix)?;
    let mut reference = Vec::new();
    for &id in suffix {
        executor.step(id)?;
        reference.push(executor.logits()?);
    }
    // Same prefix, exported, reset, imported: continuation must be bit-identical.
    executor.reset()?;
    executor.prefill(prefix)?;
    let session = executor.export_session()?;
    assert_eq!(session.position, 1024);
    executor.reset()?;
    executor.import_session(&session)?;
    for (row, &id) in suffix.iter().enumerate() {
        executor.step(id)?;
        assert_eq!(executor.logits()?, reference[row], "session row {row}");
    }
    // Cancel mid-prefill, then resume from the reported position.
    executor.reset()?;
    let cancel = executor.cancel_handle();
    let start = std::time::Instant::now();
    let flag = cancel.clone();
    let timer = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(1500));
        flag.store(true, std::sync::atomic::Ordering::Release);
        std::time::Instant::now()
    });
    let outcome = executor.prefill_cancellable(prefix)?;
    let raised = timer.join().unwrap();
    let latency = raised.elapsed().as_secs_f64();
    eprintln!(
        "glm_cancel outcome={outcome:?} total_s={:.3} latency_after_flag_s={latency:.3}",
        start.elapsed().as_secs_f64()
    );
    if let PrefillOutcome::Cancelled { position } = outcome {
        assert!(position < 1024 && position.is_multiple_of(256));
        assert_eq!(executor.position(), position);
        executor.prefill(&prefix[position..])?;
    }
    for (row, &id) in suffix.iter().enumerate() {
        executor.step(id)?;
        let actual = executor.logits()?;
        let argmax = |v: &[f64]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        eprintln!(
            "glm_cancel_resume row={row} top1_match={}",
            argmax(&actual) == argmax(&reference[row])
        );
        if matches!(outcome, PrefillOutcome::Done(_)) {
            assert_eq!(actual, reference[row]);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires main GPU and 45-layer GLM53_HYT; device decode graph vs per-op sync bits"]
fn glm53_device_decode_matches_sync() -> Result<()> {
    use super::{Executor, Options};
    let hyt = PathBuf::from(std::env::var("GLM53_HYT").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53-stage6-fp8-dense".into()
    }));
    let device = Device::select()?;
    let mut executor = Executor::load(
        &device,
        &hyt,
        Options {
            context: 256,
            decode_device: true,
            maintenance_interval: 0,
            ..Options::default()
        },
    )?;
    // Prefill (batched), then forced steps across pool boundaries; the first
    // device token is eager, the rest replay the captured graph.
    let prompt: Vec<u32> = (0..37).map(|i| 17 + (i * 7919 % 9973)).collect();
    let forced: Vec<u32> = (0..14).map(|i| 101 + (i * 4099 % 7919)).collect();
    let run = |executor: &mut Executor<'_>| -> Result<Vec<(u32, Vec<f64>)>> {
        executor.reset()?;
        let mut out = vec![(executor.prefill(&prompt)?, executor.logits()?)];
        for &id in &forced {
            out.push((executor.step(id)?, executor.logits()?));
        }
        Ok(out)
    };
    let graph = run(&mut executor)?;
    let parked = executor.swap_device(None);
    let sync = run(&mut executor)?;
    executor.swap_device(parked);
    let replay = run(&mut executor)?;
    let (hits, routes, _) = executor.cache_counts();
    assert!(routes > hits, "test must exercise CPU experts");
    for (row, ((g, s), r)) in graph.iter().zip(&sync).zip(&replay).enumerate() {
        assert_eq!(g.0, s.0, "greedy row {row}");
        assert_eq!(g.1, s.1, "device vs sync logits row {row}");
        assert_eq!(r, g, "graph replay row {row}");
    }
    eprintln!(
        "GLM stage19: device decode = per-op sync, {} rows bit-identical, CPU experts {}",
        graph.len(),
        routes - hits
    );
    Ok(())
}

#[test]
#[ignore = "requires main GPU and a LOWBIT mix with split slots (GLM53_LOWBIT_HYT, default f100)"]
fn glm53_lowbit_shrink_returns_vram() -> Result<()> {
    // Split blocks (many 2-bit images per cudaMalloc) are freed only whole: a
    // shrink must pack the survivors so the released bytes really go back,
    // keep every expert's bytes, and grow again in blocks.
    use super::{Executor, Options};
    let hyt = PathBuf::from(std::env::var("GLM53_LOWBIT_HYT").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53q-f100-packed".into()
    }));
    let device = Device::select()?;
    let mut executor = Executor::load(
        &device,
        &hyt,
        Options {
            context: 1024,
            decode_device: true,
            maintenance_interval: 0,
            vram_budget: Some(84 << 30),
            ..Options::default()
        },
    )?;
    let prompt: Vec<u32> = (0..300).map(|i| 17 + (i * 7919 % 9973)).collect();
    let forced: Vec<u32> = (0..8).map(|i| 101 + (i * 4099 % 7919)).collect();
    executor.prefill(&prompt)?;
    let mut reference = Vec::new();
    for &id in &forced {
        reference.push(executor.step(id)?);
    }
    executor.reset()?;
    executor.prefill(&prompt)?;
    let shrink = executor.set_vram_budget(60 << 30)?;
    let mut top = 0;
    for (row, &id) in forced.iter().enumerate() {
        top += usize::from(executor.step(id)? == reference[row]);
        if row == 3 {
            let grow = executor.set_vram_budget(84 << 30)?;
            eprintln!("GLM lowbit grow: {grow:?}");
            assert!(grow.gpu_delta < 0 && grow.vram_freed < 0);
        }
    }
    eprintln!(
        "GLM lowbit shrink: {shrink:?}, greedy matches {top}/{}",
        forced.len()
    );
    assert!(shrink.gpu_delta > 0 && shrink.packed > 0, "{shrink:?}");
    // The CUDA free memory grew by (nearly) the released expert bytes.
    assert!(
        shrink.vram_freed as f64 >= 0.9 * shrink.gpu_delta as f64,
        "released {} of {} bytes",
        shrink.vram_freed,
        shrink.gpu_delta
    );
    assert!(
        top + 1 >= forced.len(),
        "the shrink moved greedy tokens: {top}"
    );
    Ok(())
}

#[test]
#[ignore = "requires main GPU and 45-layer GLM53_HYT; device decode with sessions, cancellable prefill and an elastic budget"]
fn glm53_device_decode_session_and_budget() -> Result<()> {
    use super::{Executor, Options, PrefillOutcome};
    let hyt = PathBuf::from(std::env::var("GLM53_HYT").unwrap_or_else(|_| {
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53-stage6-fp8-dense".into()
    }));
    let device = Device::select()?;
    let mut executor = Executor::load(
        &device,
        &hyt,
        Options {
            context: 1024,
            decode_device: true,
            maintenance_interval: 0,
            vram_budget: Some(80 << 30),
            allow_disk: true,
            ..Options::default()
        },
    )?;
    let prompt: Vec<u32> = (0..300).map(|i| 17 + (i * 7919 % 9973)).collect();
    let forced: Vec<u32> = (0..8).map(|i| 101 + (i * 4099 % 7919)).collect();
    // Reference: prefill then device steps.
    executor.prefill(&prompt)?;
    let mut reference = Vec::new();
    for &id in &forced {
        reference.push((executor.step(id)?, executor.logits()?));
    }
    // Export after the prefill, reset, import: the device continuation is bit-identical.
    executor.reset()?;
    assert!(matches!(
        executor.prefill_cancellable(&prompt)?,
        PrefillOutcome::Done(_)
    ));
    let session = executor.export_session()?;
    executor.reset()?;
    executor.import_session(&session)?;
    for (row, &id) in forced.iter().enumerate() {
        assert_eq!(
            executor.step(id)?,
            reference[row].0,
            "session greedy row {row}"
        );
        assert_eq!(
            executor.logits()?,
            reference[row].1,
            "session logits row {row}"
        );
    }
    // Shrink and grow the expert cache between device tokens: the table follows.
    executor.import_session(&session)?;
    let shrink = executor.set_vram_budget(60 << 30)?;
    let mut top = 0;
    for (row, &id) in forced.iter().enumerate() {
        top += usize::from(executor.step(id)? == reference[row].0);
        if row == 3 {
            executor.set_vram_budget(80 << 30)?;
        }
    }
    eprintln!(
        "GLM stage19 device session/budget: session rows bit-identical, shrink released {} bytes, greedy matches {top}/{}",
        shrink.gpu_delta,
        forced.len()
    );
    assert!(shrink.gpu_delta > 0);
    assert!(
        top + 1 >= forced.len(),
        "budget changes moved greedy tokens: {top}"
    );
    Ok(())
}
