//! End to end on the shrunken golden model (`python/golden_tiny.py`): the engine, loading the
//! RadixArk-style tiny checkpoint, must reproduce transformers' logits at every position.
//! GPU test: `scripts/gpu-run.sh -- target/release/deps/tiny_e2e-* --ignored --nocapture`.
//!
//! The reference runs in fp32; the engine keeps the KV and indexer caches in bf16 like the real
//! model. That can flip the QSA top-k at near-ties, which changes that position's attention
//! legitimately. Such positions must be near-ties (score gap < 1 %) and are excluded from the
//! strict checks; every other position must match the greedy token and agree in KL.

use std::collections::BTreeSet;
use std::path::Path;
use tsumuji_pack::json::Json;
use tsumuji_pack::safetensors::read_header;

#[test]
#[ignore = "needs the GPU"]
fn tiny_model_matches_transformers() {
    // The reference has bf16 dense weights; keep them bf16 so the comparison stays exact.
    unsafe { std::env::set_var("TSJ_DENSE", "none") };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny");
    if !dir.join("ckpt/model.safetensors").exists() {
        eprintln!("skipped: run python/golden_tiny.py first");
        return;
    }
    let meta = Json::parse(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let tokens: Vec<u32> = meta
        .get("tokens")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect();
    let golden = read_header(&dir.join("tensors.safetensors")).unwrap();
    let want = golden
        .iter()
        .find(|t| t.name == "logits")
        .unwrap()
        .read_f32()
        .unwrap();
    let n = tokens.len();

    let mut eng = tsumuji_engine::open(&dir.join("ckpt"), 256, |_| {}).unwrap();
    eng.capture = Some(Vec::new());
    let v = eng.dims.vocab;
    let mut logits = Vec::new();
    for &t in &tokens {
        eng.step(Some(t)).unwrap();
        logits.push(eng.logits().unwrap());
    }

    // The captured graph (with PDL) must give bit-identical logits to the direct launches above.
    let capture = eng.capture.take();
    eng.reset().unwrap();
    for (p, &t) in tokens.iter().enumerate() {
        eng.step(Some(t)).unwrap();
        let g = eng.logits().unwrap();
        assert!(
            g.iter()
                .zip(&logits[p])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "graph replay differs at position {p}"
        );
    }

    // Steps of N tokens must give the same bits as N steps of one (M3: verification windows).
    for chunk in [2, 3, 4, 5, 8] {
        eng.reset().unwrap();
        for (c, part) in tokens.chunks(chunk).enumerate() {
            eng.step_tokens(part).unwrap();
            let all = eng.logits_all().unwrap();
            let greedy = eng.greedy_all().unwrap();
            for (i, got) in all.chunks(v).enumerate() {
                let p = c * chunk + i;
                assert!(
                    got.iter()
                        .zip(&logits[p])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "N = {chunk}: position {p} differs from single-token steps"
                );
                let am = logits[p]
                    .iter()
                    .enumerate()
                    .fold(0, |bi, (j, &y)| if y > logits[p][bi] { j } else { bi });
                assert_eq!(
                    greedy[i] as usize, am,
                    "N = {chunk}: greedy token at position {p}"
                );
            }
        }
    }
    eprintln!("multi-token steps (N = 2, 3, 4, 5, 8) bit-identical to single-token steps");
    eng.capture = capture;

    // Positions where some QSA layer selected other tokens than the reference.
    let qsa_layers: Vec<usize> = (0..eng.dims.layers).filter(|&l| eng.dims.qsa[l]).collect();
    let mut flipped = BTreeSet::new();
    for (qi, &l) in qsa_layers.iter().enumerate() {
        let mask = golden
            .iter()
            .find(|t| t.name == format!("L{l}.indexer.out"))
            .unwrap()
            .read_f32()
            .unwrap();
        for p in 0..n {
            let (toks, scores) = &eng.capture_qsa[p * qsa_layers.len() + qi];
            let got: BTreeSet<usize> = toks.iter().map(|&t| t as usize / 4).collect();
            let exp: BTreeSet<usize> = (0..n)
                .filter(|&j| mask[p * n + j] == 0.0)
                .map(|j| j / 4)
                .collect();
            if got != exp {
                let worst_in = got
                    .difference(&exp)
                    .map(|&b| scores[b])
                    .fold(f32::MAX, f32::min);
                let best_out = exp
                    .difference(&got)
                    .map(|&b| scores[b])
                    .fold(0f32, f32::max);
                let gap = (worst_in - best_out).abs() / worst_in.abs().max(1e-6);
                eprintln!("layer {l} position {p}: QSA selection differs, score gap {gap:.2e}");
                assert!(gap < 1e-2, "selection differs without a near-tie");
                flipped.insert(p);
            }
        }
    }

    let (mut worst, mut top1, mut kl_sum, mut kl_max, mut checked) = (0f32, 0, 0f64, 0f64, 0);
    for p in (0..n).filter(|p| !flipped.contains(p)) {
        let (got, exp) = (&logits[p], &want[p * v..(p + 1) * v]);
        worst = got
            .iter()
            .zip(exp)
            .fold(worst, |a, (x, y)| a.max((x - y).abs()));
        let am = |x: &[f32]| {
            x.iter()
                .enumerate()
                .fold(0, |bi, (i, &y)| if y > x[bi] { i } else { bi })
        };
        top1 += usize::from(am(got) == am(exp));
        let k = kl(exp, got);
        kl_sum += k;
        kl_max = kl_max.max(k);
        checked += 1;
    }
    eprintln!(
        "{checked} positions checked ({} near-tie flips excluded): max |diff| {worst:.3e}, top-1 {top1}/{checked}, KL mean {:.3e} max {kl_max:.3e}",
        flipped.len(),
        kl_sum / checked as f64
    );
    assert!(flipped.len() <= n / 20, "too many selection flips");
    assert_eq!(top1, checked, "greedy token differs");
    // fp32 caches leave only summation-order differences; bf16 caches add their rounding.
    let limit = if tsumuji_kernels::KV_BYTES == 4 {
        1e-10
    } else {
        1e-5
    };
    assert!(kl_max < limit, "KL too large");
}

/// KL(p_ref || p_got) of the softmaxes, in f64.
fn kl(reference: &[f32], got: &[f32]) -> f64 {
    let lse = |x: &[f32]| {
        let m = x
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(f64::from(b)));
        m + x
            .iter()
            .map(|&y| (f64::from(y) - m).exp())
            .sum::<f64>()
            .ln()
    };
    let (lr, lg) = (lse(reference), lse(got));
    reference
        .iter()
        .zip(got)
        .map(|(&r, &g)| {
            let (lpr, lpg) = (f64::from(r) - lr, f64::from(g) - lg);
            lpr.exp() * (lpr - lpg)
        })
        .sum()
}
