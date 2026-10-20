//! Checks against the transformers golden fixtures (`python/golden_tiny.py`). Skips with a note
//! when `fixtures/tiny` has not been generated.

use std::path::{Path, PathBuf};
use tsumuji_model::ple::{Context, SEED, Tables};
use tsumuji_pack::json::Json;
use tsumuji_pack::safetensors::{TensorInfo, read_header};

fn fixtures() -> Option<PathBuf> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny");
    if d.join("meta.json").exists() {
        Some(d)
    } else {
        eprintln!(
            "skipped: {} missing (run python/golden_tiny.py)",
            d.display()
        );
        None
    }
}

fn tensor<'a>(all: &'a [TensorInfo], name: &str) -> &'a TensorInfo {
    all.iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} not in fixture"))
}

/// The PLE n-gram embedding: hash tables, EOS-reset context and row gather must reproduce
/// the reference output bit for bit.
#[test]
fn ple_ngram_embedding_matches_reference() {
    let Some(dir) = fixtures() else { return };
    let meta = Json::parse(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let tokens: Vec<u32> = meta
        .get("tokens")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect();
    let eos = meta.get("eos").unwrap().as_u64().unwrap() as u32;
    let cfg = Json::parse(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let vocab = cfg.get("vocab_size").unwrap().as_u64().unwrap();
    let base = cfg.get("ngram_vocab_size_base").unwrap().as_u64().unwrap();

    let weights = read_header(&dir.join("weights.safetensors")).unwrap();
    let tensors = read_header(&dir.join("tensors.safetensors")).unwrap();
    let prefix = "model.layers.1.ple.ple_embedding.";
    let t = Tables::new(vocab, base, 0, SEED);
    let as_u = |v: Vec<i64>| v.into_iter().map(|x| x as u64).collect::<Vec<_>>();
    assert_eq!(
        as_u(
            tensor(&weights, &format!("{prefix}layer_multipliers"))
                .read_i64()
                .unwrap()
        ),
        t.mult
    );
    assert_eq!(
        as_u(
            tensor(&weights, &format!("{prefix}ngram_heads_vocab_sizes"))
                .read_i64()
                .unwrap()
        ),
        t.size
    );
    assert_eq!(
        as_u(
            tensor(&weights, &format!("{prefix}ngram_heads_offsets"))
                .read_i64()
                .unwrap()
        ),
        t.offset
    );

    let table_info = tensor(&weights, &format!("{prefix}ngram_embedding.weight"));
    assert_eq!(table_info.shape[0], t.padded_rows);
    let width = table_info.shape[1] as usize;
    let table = table_info.read_f32().unwrap();
    let want = tensor(&tensors, "L1.ple.ngram.out").read_f32().unwrap();
    assert_eq!(want.len(), tokens.len() * 16 * width);

    let mut ctx = Context::new(eos);
    let mut mismatches = 0;
    for (p, &tok) in tokens.iter().enumerate() {
        let rows = t.rows(tok, ctx.t1, ctx.t2);
        for (h, &row) in rows.iter().enumerate() {
            let got = &table[row as usize * width..][..width];
            let exp = &want[(p * 16 + h) * width..][..width];
            if got.iter().zip(exp).any(|(a, b)| a.to_bits() != b.to_bits()) {
                mismatches += 1;
            }
        }
        ctx.push(tok);
    }
    assert_eq!(
        mismatches,
        0,
        "{mismatches} of {} rows differ",
        tokens.len() * 16
    );
}
