//! M3 speculative verification on the shrunken golden model (`python/golden_tiny.py`): windows of
//! [next token, drafts...] with right and wrong drafts must commit exactly the right prefix, and
//! every committed position must give the same logits, bit for bit, as plain greedy decoding;
//! after a rejection the sequence must go on as if the rejected drafts had never been seen.
//! Also: `save_state` / `restore_state` replays a continuation bit for bit.
//! GPU test: `scripts/gpu-run.sh -- target/release/deps/tiny_verify-* --ignored --nocapture`.

use std::path::Path;
use tsumuji_pack::json::Json;

fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
#[ignore = "needs the GPU"]
fn verification_matches_greedy_decoding() {
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
    let prompt = &tokens[..13];
    let m = 120; // generated tokens; the tiny model's QSA turns sparse after 19 positions
    let mut eng = tsumuji_engine::open(&dir.join("ckpt"), 256, |_| {}).unwrap();
    let v = eng.dims.vocab as u32;

    // Reference: greedy decoding one token per step. g[i] is fed at position |prompt| + i and
    // gives logits lref[i].
    eng.prefill(prompt).unwrap();
    let mut g = vec![eng.next_token().unwrap()];
    let mut lref = Vec::new();
    for _ in 0..m {
        eng.step(None).unwrap();
        lref.push(eng.logits().unwrap());
        g.push(eng.next_token().unwrap());
    }

    // Verification windows of 1..=8 tokens, drafts taken from the reference with some wrong ones.
    eng.reset().unwrap();
    eng.prefill(prompt).unwrap();
    let mut i = 0; // index in g of the window's token 0
    let mut windows = 0;
    let mut rejected = 0;
    while i < m {
        let k = (windows * 3 % 8).min(m - i - 1); // drafts
        let mut window: Vec<u32> = g[i..=i + k].to_vec();
        let wrong = (windows % 3 == 1 && k > 0).then(|| 1 + windows % k); // one wrong draft
        if let Some(j) = wrong {
            window[j] = (window[j] + 1) % v;
        }
        assert_eq!(
            eng.next_token().unwrap(),
            window[0],
            "window {windows}: next token"
        );
        let c = eng.verify(&window).unwrap();
        let want = wrong.map_or(k + 1, |j| j);
        assert_eq!(
            c,
            want,
            "window {windows} at {i}: committed {c} of {}, wrong draft at {wrong:?}",
            k + 1
        );
        rejected += usize::from(c < k + 1);
        let all = eng.logits_all().unwrap();
        for t in 0..c {
            assert!(
                same(&all[t * v as usize..][..v as usize], &lref[i + t]),
                "window {windows}: position {} logits differ",
                i + t
            );
        }
        i += c;
        assert_eq!(eng.pos, prompt.len() + i);
        windows += 1;
    }
    assert_eq!(
        eng.next_token().unwrap(),
        g[i],
        "greedy token after the last window"
    );
    eprintln!(
        "{windows} windows ({rejected} with a rejected draft) reproduce {m} greedy positions bit for bit"
    );

    // Snapshot: continue 20 steps, restore, continue again: identical.
    eng.reset().unwrap();
    eng.prefill(prompt).unwrap();
    for _ in 0..30 {
        eng.step(None).unwrap();
    }
    let snap = eng.save_state().unwrap();
    let run = |eng: &mut tsumuji_engine::engine::Engine| -> Vec<Vec<f32>> {
        (0..20)
            .map(|s| {
                if s % 4 == 0 {
                    let t = eng.next_token().unwrap();
                    eng.step_tokens(&[t, g[0], g[1]]).unwrap(); // any tokens: committed as given
                } else {
                    eng.step(None).unwrap();
                }
                eng.logits_all().unwrap()
            })
            .collect()
    };
    let first = run(&mut eng);
    // Move the sequence somewhere else first (past the snapshot position: the caches below it
    // must stay as they were).
    eng.prefill(&tokens[20..40]).unwrap();
    eng.restore_state(&snap).unwrap();
    let second = run(&mut eng);
    assert!(
        first.iter().zip(&second).all(|(a, b)| same(a, b)),
        "restored state continues differently"
    );
    eprintln!(
        "snapshot at position {} replays 20 steps bit for bit",
        snap.pos
    );

    // Reused buffers: `save_state_into` overwrites a snapshot in place, and a dropped snapshot's
    // buffers serve the next `save_state` (the pool). Both restore the same continuation.
    eng.restore_state(&snap).unwrap();
    let mut into = tsumuji_engine::engine::Snapshot::default();
    eng.save_state_into(&mut into).unwrap();
    let bytes = snap.bytes();
    drop(snap);
    let pooled = eng.save_state().unwrap();
    assert_eq!((into.bytes(), pooled.bytes()), (bytes, bytes));
    for s in [&pooled, &into] {
        eng.prefill(&tokens[20..40]).unwrap();
        eng.restore_state(s).unwrap();
        let again = run(&mut eng);
        assert!(
            first.iter().zip(&again).all(|(a, b)| same(a, b)),
            "reused snapshot buffers continue differently"
        );
    }
    eprintln!("save_state_into and pooled save_state replay bit for bit");
}

#[test]
#[ignore = "needs the GPU"]
fn sampled_verification_matches_sampled_decoding() {
    use tsumuji_engine::engine::Sampling;
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
    let prompt = &tokens[..13];
    let m = 100;
    let mut eng = tsumuji_engine::open(&dir.join("ckpt"), 256, |_| {}).unwrap();
    let v = eng.dims.vocab;
    let decode =
        |eng: &mut tsumuji_engine::engine::Engine, s: Sampling| -> (Vec<u32>, Vec<Vec<f32>>) {
            eng.set_sampling(s).unwrap();
            eng.reset().unwrap();
            eng.prefill(prompt).unwrap();
            let mut g = vec![eng.next_token().unwrap()];
            let mut l = Vec::new();
            for _ in 0..m {
                eng.step(None).unwrap();
                l.push(eng.logits().unwrap());
                g.push(eng.next_token().unwrap());
            }
            (g, l)
        };
    let greedy = decode(&mut eng, Sampling::default()).0;
    let top1 = decode(
        &mut eng,
        Sampling {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: 7,
        },
    )
    .0;
    assert_eq!(top1, greedy, "top_k 1 must be greedy");

    let s = Sampling {
        temperature: 1.0,
        top_k: 20,
        top_p: 0.95,
        seed: 42,
    };
    let (g, lref) = decode(&mut eng, s);
    assert_ne!(
        g, greedy,
        "sampling at temperature 1 should leave the greedy path"
    );
    // Every sample is among the 20 largest logits of its position.
    for (i, l) in lref.iter().enumerate() {
        let x = l[g[i + 1] as usize];
        assert!(
            l.iter().filter(|&&y| y > x).count() < 20,
            "position {i}: sample outside the top 20"
        );
    }
    let other = decode(&mut eng, Sampling { seed: 43, ..s }).0;
    assert_ne!(other, g, "another seed should sample differently");

    // Windows with right and wrong drafts give the same samples.
    eng.set_sampling(s).unwrap();
    eng.reset().unwrap();
    eng.prefill(prompt).unwrap();
    let (mut i, mut windows) = (0, 0);
    while i < m {
        let k = (windows * 5 % 8).min(m - i - 1);
        let mut window: Vec<u32> = g[i..=i + k].to_vec();
        let wrong = (windows % 3 == 2 && k > 0).then(|| 1 + windows % k);
        if let Some(j) = wrong {
            window[j] = (window[j] + 1) % v as u32;
        }
        let c = eng.verify(&window).unwrap();
        assert_eq!(c, wrong.map_or(k + 1, |j| j), "window {windows} at {i}");
        let all = eng.logits_all().unwrap();
        for t in 0..c {
            assert!(
                same(&all[t * v..][..v], &lref[i + t]),
                "window {windows}: position {} logits differ",
                i + t
            );
        }
        i += c;
        assert_eq!(
            eng.next_token().unwrap(),
            g[i],
            "window {windows}: sampled next token"
        );
        windows += 1;
    }
    eprintln!(
        "sampling (T 1, top_k 20, top_p 0.95): {windows} verification windows reproduce {m} sampled tokens"
    );
}
