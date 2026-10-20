//! Debugging aid: per-layer residual of the engine vs transformers on the tiny golden model.
//! `scripts/gpu-run.sh -- target/release/deps/tiny_layers-* --ignored --nocapture`

use std::path::Path;
use tsumuji_pack::json::Json;
use tsumuji_pack::safetensors::read_header;

#[test]
#[ignore = "needs the GPU; prints a report"]
fn tiny_layer_residuals() {
    // The reference has bf16 dense weights; keep them bf16 so the comparison stays exact.
    unsafe { std::env::set_var("TSJ_DENSE", "none") };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny");
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
    let mut eng = tsumuji_engine::open(&dir.join("ckpt"), 256, |_| {}).unwrap();
    eng.capture = Some(Vec::new());
    for &t in &tokens {
        eng.step(Some(t)).unwrap();
    }
    let cap = eng.capture.take().unwrap();
    let (layers, hcw) = (eng.dims.layers, eng.dims.hcw());
    for l in 0..layers {
        let want = golden
            .iter()
            .find(|t| t.name == format!("L{l}.out"))
            .unwrap()
            .read_f32()
            .unwrap();
        let mut worst = (0f32, 0usize);
        let mut first_bad = None;
        for p in 0..tokens.len() {
            let got = &cap[p * layers + l];
            let exp = &want[p * hcw..(p + 1) * hcw];
            let scale = exp.iter().fold(0f32, |a, b| a.max(b.abs()));
            let err = got
                .iter()
                .zip(exp)
                .fold(0f32, |a, (x, y)| a.max((x - y).abs()))
                / scale;
            if err > worst.0 {
                worst = (err, p);
            }
            if l == layers - 1 && std::env::var_os("TSJ_PER_POS").is_some() {
                eprint!("{p}:{err:.1e} ");
            }
            if err > 1e-2 && first_bad.is_none() {
                first_bad = Some(p);
            }
        }
        eprintln!(
            "layer {l}: max rel err {:.3e} at position {}, first position > 1e-2: {first_bad:?}",
            worst.0, worst.1
        );
    }
}

#[test]
#[ignore = "needs the GPU; prints a report"]
fn tiny_qsa_selection() {
    // The reference has bf16 dense weights; keep them bf16 so the comparison stays exact.
    unsafe { std::env::set_var("TSJ_DENSE", "none") };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny");
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
    let mask = golden
        .iter()
        .find(|t| t.name == "L3.indexer.out")
        .unwrap()
        .read_f32()
        .unwrap();
    let n = tokens.len();
    let mut eng = tsumuji_engine::open(&dir.join("ckpt"), 256, |_| {}).unwrap();
    eng.capture = Some(Vec::new());
    for &t in &tokens {
        eng.step(Some(t)).unwrap();
    }
    let mut differ = 0;
    for (p, (toks, scores)) in eng.capture_qsa.iter().enumerate() {
        let mut got: Vec<usize> = toks.iter().map(|&t| t as usize).collect();
        got.sort_unstable();
        let want: Vec<usize> = (0..n).filter(|&j| mask[p * n + j] == 0.0).collect();
        if got != want {
            differ += 1;
            let only_got: Vec<_> = got.iter().filter(|j| !want.contains(j)).collect();
            let only_want: Vec<_> = want.iter().filter(|j| !got.contains(j)).collect();
            let blk = |v: &[&usize]| {
                v.iter()
                    .map(|&&j| j / 4)
                    .collect::<std::collections::BTreeSet<_>>()
            };
            let (bg, bw) = (blk(&only_got), blk(&only_want));
            let sc = |b: &std::collections::BTreeSet<usize>| {
                b.iter()
                    .map(|&b| format!("{b}:{:.6}", scores[b]))
                    .collect::<Vec<_>>()
            };
            eprintln!(
                "position {p}: engine-only blocks {:?}, reference-only blocks {:?}",
                sc(&bg),
                sc(&bw)
            );
        }
    }
    eprintln!(
        "{differ} of {} positions select different tokens",
        eng.capture_qsa.len()
    );
}
