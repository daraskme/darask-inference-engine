//! tsumuji-pack: checkpoint tooling.
//!
//!   tsumuji-pack inspect <model-dir>     tensor families, dtypes, sizes; shapes checked against the model
//!   tsumuji-pack check-ple <model-dir>   PLE hash tables in the checkpoint vs tsumuji-model's

use std::collections::BTreeMap;
use std::path::Path;
use tsumuji_model::{ple, shape::*};
use tsumuji_pack::safetensors::{TensorInfo, dtype_bytes, read_dir};

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let result = match (argv.first().map(String::as_str), argv.get(1)) {
        (Some("inspect"), Some(dir)) => inspect(Path::new(dir)),
        (Some("check-ple"), Some(dir)) => check_ple(Path::new(dir)),
        _ => Err("usage: tsumuji-pack inspect|check-ple <model-dir>".into()),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Replaces numeric path components (layer, expert, shard indices) with N.
fn family(name: &str) -> String {
    name.split('.')
        .map(|p| {
            if p.chars().all(|c| c.is_ascii_digit()) {
                "N".to_string()
            } else if let Some(n) = p
                .strip_prefix("shard_")
                .filter(|n| n.chars().all(|c| c.is_ascii_digit()))
            {
                let _ = n;
                "shard_N".to_string()
            } else {
                p.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// Expected (family suffix, dtype, shape) for the text model. A family matches the first
/// entry whose suffix it ends with.
fn expectations() -> Vec<(&'static str, &'static str, Vec<u64>)> {
    let (h, hcw, r) = (H as u64, HCW as u64, GR_RANK as u64);
    let vd = (GDN_V * GDN_D) as u64;
    let mut v = vec![
        ("lm_head.weight", "BF16", vec![VOCAB as u64, h]),
        (
            "language_model.embed_tokens.weight",
            "BF16",
            vec![VOCAB as u64, h],
        ),
        ("hc_norm.weight", "BF16", vec![hcw]),
        ("input_mix_weight_down.weight", "BF16", vec![r, hcw]),
        ("input_mix_weight_up.weight", "BF16", vec![hcw, r]),
        ("block_inject_weight.weight", "BF16", vec![HC as u64, hcw]),
        (
            "linear_attn.in_proj_qkv.weight",
            "BF16",
            vec![GDN_QKV as u64, h],
        ),
        ("linear_attn.in_proj_z.weight", "BF16", vec![vd, h]),
        (
            "linear_attn.in_proj_a.weight",
            "BF16",
            vec![GDN_V as u64, h],
        ),
        (
            "linear_attn.in_proj_b.weight",
            "BF16",
            vec![GDN_V as u64, h],
        ),
        (
            "linear_attn.conv1d.weight",
            "BF16",
            vec![GDN_QKV as u64, 1, GDN_CONV as u64],
        ),
        ("linear_attn.dt_bias", "BF16", vec![GDN_V as u64]),
        ("linear_attn.A_log", "BF16", vec![GDN_V as u64]),
        ("linear_attn.norm.weight", "BF16", vec![GDN_D as u64]),
        ("linear_attn.out_proj.weight", "BF16", vec![h, vd]),
        (
            "self_attn.q_proj.weight",
            "BF16",
            vec![(ATT_Q * ATT_D * 2) as u64, h],
        ),
        (
            "self_attn.k_proj.weight",
            "BF16",
            vec![(ATT_KV * ATT_D) as u64, h],
        ),
        (
            "self_attn.v_proj.weight",
            "BF16",
            vec![(ATT_KV * ATT_D) as u64, h],
        ),
        (
            "self_attn.o_proj.weight",
            "BF16",
            vec![h, (ATT_Q * ATT_D) as u64],
        ),
        ("self_attn.q_norm.weight", "BF16", vec![ATT_D as u64]),
        ("self_attn.k_norm.weight", "BF16", vec![ATT_D as u64]),
        (
            "indexer.index_qk_proj.weight",
            "BF16",
            vec![((IDX_H + 1) * IDX_D) as u64, h],
        ),
        ("indexer.q_layernorm.weight", "BF16", vec![IDX_D as u64]),
        ("indexer.k_layernorm.weight", "BF16", vec![IDX_D as u64]),
        ("mlp.gate.weight", "BF16", vec![E as u64, h]),
        ("shared_expert.gate_proj.weight", "BF16", vec![FF as u64, h]),
        ("shared_expert.up_proj.weight", "BF16", vec![FF as u64, h]),
        ("shared_expert.down_proj.weight", "BF16", vec![h, FF as u64]),
        ("shared_expert_gate.weight", "BF16", vec![1, h]),
        ("experts.N.gate_proj.weight", "U8", vec![FF as u64, h / 2]),
        ("experts.N.up_proj.weight", "U8", vec![FF as u64, h / 2]),
        ("experts.N.down_proj.weight", "U8", vec![h, FF as u64 / 2]),
        (
            "experts.N.gate_proj.weight_scale",
            "F8_E4M3",
            vec![FF as u64, h / 16],
        ),
        (
            "experts.N.up_proj.weight_scale",
            "F8_E4M3",
            vec![FF as u64, h / 16],
        ),
        (
            "experts.N.down_proj.weight_scale",
            "F8_E4M3",
            vec![h, FF as u64 / 16],
        ),
        ("weight_scale_2", "F32", vec![]),
        ("input_scale", "F32", vec![]),
        ("ple.key_proj.weight", "BF16", vec![hcw, h]),
        ("ple.value_proj.weight", "BF16", vec![h, h]),
        ("ple.norm_key.weight", "BF16", vec![hcw]),
        ("ple.norm_query.weight", "BF16", vec![hcw]),
        ("ple.norm_conv.weight", "BF16", vec![hcw]),
        ("ple.conv1d.weight", "BF16", vec![hcw, 1, PLE_CONV as u64]),
        ("ple_embedding.layer_multipliers", "I64", vec![3]),
        (
            "ple_embedding.ngram_heads_vocab_sizes",
            "I64",
            vec![PLE_HEADS as u64],
        ),
        (
            "ple_embedding.ngram_heads_offsets",
            "I64",
            vec![PLE_HEADS as u64],
        ),
        (
            "ngram_embedding.shard_N.weight",
            "F8_E4M3",
            vec![2_500_012, PLE_ROW as u64],
        ),
        ("ngram_embedding.weight_scale", "BF16", vec![1]),
        ("mtp.fc_embedding.weight", "BF16", vec![h, h]),
        ("mtp.fc_hidden.weight", "BF16", vec![h, h]),
        ("mtp.pre_fc_norm_embedding.weight", "BF16", vec![h]),
        ("mtp.pre_fc_norm_hidden.weight", "BF16", vec![hcw]),
        // The MTP layer keeps its experts fused and in BF16 (the NVFP4 recipe excluded `mtp.*`).
        (
            "mtp.layers.N.mlp.experts.gate_up_proj",
            "BF16",
            vec![E as u64, 2 * FF as u64, h],
        ),
        (
            "mtp.layers.N.mlp.experts.down_proj",
            "BF16",
            vec![E as u64, h, FF as u64],
        ),
    ];
    v.sort_by_key(|e| std::cmp::Reverse(e.0.len())); // longest suffix wins
    v
}

fn inspect(dir: &Path) -> Result<(), String> {
    let tensors = read_dir(dir)?;
    let mut fam: BTreeMap<String, Vec<&TensorInfo>> = BTreeMap::new();
    for t in &tensors {
        fam.entry(family(&t.name)).or_default().push(t);
    }
    let exp = expectations();
    let (mut bad, mut unchecked) = (0, Vec::new());
    let mut by_kind: BTreeMap<&str, f64> = BTreeMap::new();
    println!(
        "{:>7}  {:<9} {:<18} {:>9}  family",
        "count", "dtype", "shape", "GB"
    );
    for (name, ts) in &fam {
        let t0 = ts[0];
        let gb = ts.iter().map(|t| t.bytes as f64).sum::<f64>() / 1e9;
        let mixed = ts
            .iter()
            .any(|t| t.dtype != t0.dtype || t.shape != t0.shape);
        let mut status = String::new();
        if let Some((_, dt, sh)) = exp.iter().find(|(suf, _, _)| name.ends_with(suf)) {
            if mixed || t0.dtype != *dt || t0.shape != *sh {
                status = format!("  << expected {dt} {sh:?}");
                bad += 1;
            }
        } else if !name.starts_with("model.visual") {
            unchecked.push(name.clone());
        }
        for t in ts.iter() {
            let want = dtype_bytes(&t.dtype).map(|b| b * t.shape.iter().product::<u64>() as f64);
            if want.is_some_and(|w| w as u64 != t.bytes) {
                status += &format!("  << {} has {} bytes for its shape", t.name, t.bytes);
                bad += 1;
            }
        }
        let kind = if name.contains("visual") {
            "vision"
        } else if name.starts_with("mtp") {
            "mtp"
        } else if name.contains("ngram_embedding") {
            "ple table"
        } else if name.contains(".experts.") {
            "routed experts"
        } else {
            "dense"
        };
        *by_kind.entry(kind).or_default() += gb;
        if !name.starts_with("model.visual") {
            let shape = format!("{:?}{}", t0.shape, if mixed { "*" } else { "" });
            println!(
                "{:>7}  {:<9} {:<18} {:>9.3}  {name}{status}",
                ts.len(),
                t0.dtype,
                shape,
                gb
            );
        }
    }
    println!("\nby kind (GB):");
    for (k, gb) in &by_kind {
        println!("  {k:<15} {gb:8.2}");
    }
    println!("  {:<15} {:8.2}", "total", by_kind.values().sum::<f64>());
    // Per layer, each projection family should appear once per layer of its kind.
    let n_gdn = (0..L).filter(|&l| !is_qsa(l)).count();
    for (suffix, want) in [
        ("linear_attn.in_proj_qkv.weight", n_gdn),
        ("self_attn.q_proj.weight", L - n_gdn),
        ("mlp.gate.weight", L),
        ("experts.N.gate_proj.weight", L * E),
    ] {
        if let Some((_, ts)) = fam
            .iter()
            .find(|(n, _)| n.starts_with("model.language_model") && n.ends_with(suffix))
        {
            let ok = if ts.len() == want {
                "ok"
            } else {
                "incomplete (download in progress?)"
            };
            println!("count {suffix}: {} / {want} {ok}", ts.len());
        }
    }
    if !unchecked.is_empty() {
        println!("\nfamilies without an expectation (review by hand):");
        for u in &unchecked {
            println!("  {u}");
        }
    }
    if bad > 0 {
        return Err(format!("{bad} mismatches"));
    }
    println!("\nall checked families match the model shapes");
    Ok(())
}

fn check_ple(dir: &Path) -> Result<(), String> {
    let tensors = read_dir(dir)?;
    let find = |suffix: &str| {
        tensors
            .iter()
            .find(|t| t.name.ends_with(suffix))
            .ok_or(format!("no tensor *{suffix}"))
    };
    let io = |e: std::io::Error| e.to_string();
    let mult = find("ple_embedding.layer_multipliers")?
        .read_i64()
        .map_err(io)?;
    let sizes = find("ple_embedding.ngram_heads_vocab_sizes")?
        .read_i64()
        .map_err(io)?;
    let offsets = find("ple_embedding.ngram_heads_offsets")?
        .read_i64()
        .map_err(io)?;
    let t = ple::Tables::qwen38();
    let as_u = |v: &[i64]| v.iter().map(|&x| x as u64).collect::<Vec<_>>();
    println!(
        "multipliers  checkpoint {:?}\n             tsumuji    {:?}",
        mult, t.mult
    );
    println!(
        "sizes        checkpoint {:?}\n             tsumuji    {:?}",
        sizes, t.size
    );
    let mut ok = as_u(&mult) == t.mult && as_u(&sizes) == t.size && as_u(&offsets) == t.offset;
    let shards: Vec<_> = tensors
        .iter()
        .filter(|t| t.name.contains("ngram_embedding.shard_"))
        .collect();
    if !shards.is_empty() {
        let rows: u64 = shards.iter().map(|s| s.shape[0]).sum();
        let widths: Vec<_> = shards.iter().map(|s| s.shape[1]).collect();
        println!(
            "table        {} shards, {rows} rows (want {}), row width {:?}, dtype {}",
            shards.len(),
            t.padded_rows,
            widths.first(),
            shards[0].dtype
        );
        ok &= rows == t.padded_rows && widths.iter().all(|&w| w == PLE_ROW as u64);
    }
    if let Ok(s) = find("ngram_embedding.weight_scale") {
        println!("table scale  {} {:?}", s.dtype, s.shape);
    }
    if ok {
        println!("PLE tables match");
        Ok(())
    } else {
        Err("PLE tables differ".into())
    }
}
