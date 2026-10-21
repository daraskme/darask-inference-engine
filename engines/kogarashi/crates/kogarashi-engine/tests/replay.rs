#![cfg(feature = "cuda")]

use darask_formats::gguf::Gguf;
use kogarashi_engine::{engine::Engine, tokenizer};
use std::path::PathBuf;

#[test]
#[ignore = "requires ORNITH_GGUF, the shared GPU slot, and 78 GiB free VRAM"]
fn requests_do_not_inherit_previous_sequence_state() {
    let model = PathBuf::from(std::env::var_os("ORNITH_GGUF").expect("set ORNITH_GGUF"));
    let tok = tokenizer::load(&Gguf::open(&model).unwrap()).unwrap();
    let prompts = [
        tok.encode("def fibonacci(n):\n    "),
        tok.encode("Return only JSON with keys name and count.\n{"),
    ];
    let mut engine = Engine::load(&model, 512).unwrap();
    let mut expected = Vec::new();
    for repeat in 0..3 {
        for (case, prompt) in prompts.iter().enumerate() {
            engine.reset().unwrap();
            let mut next = 0;
            for &id in prompt {
                next = engine.step(id).unwrap();
            }
            let mut ids = Vec::new();
            for _ in 0..64 {
                ids.push(next);
                next = engine.step(next).unwrap();
            }
            assert!(engine.logits().unwrap().iter().all(|v| v.is_finite()));
            if repeat == 0 {
                expected.push(ids);
            } else {
                assert_eq!(ids, expected[case], "request {case}, repeat {repeat}");
            }
        }
    }
}

#[test]
#[ignore = "requires ORNITH_GGUF, DARASK_ORNITH_VISION and the shared GPU slot"]
fn image_and_text_requests_replay_the_same_logits() {
    let model = PathBuf::from(std::env::var_os("ORNITH_GGUF").expect("set ORNITH_GGUF"));
    let vision =
        PathBuf::from(std::env::var_os("DARASK_ORNITH_VISION").expect("set DARASK_ORNITH_VISION"));
    let tok = tokenizer::load(&Gguf::open(&model).unwrap()).unwrap();
    let text = tok.encode("What is 2 + 2? Answer using only the digit.");
    let mut engine = Engine::load(&model, 512).unwrap();
    let feed = |engine: &mut Engine, prompt: &[u32]| {
        for &id in prompt {
            engine.step(id).unwrap();
        }
        engine.logits().unwrap()
    };
    let mut expected = None;
    for _ in 0..3 {
        engine.reset().unwrap();
        engine.set_images(&[], &vision).unwrap();
        let actual = feed(&mut engine, &text);
        assert!(actual.iter().all(|v| v.is_finite()));
        if let Some(ref expected) = expected {
            assert_eq!(&actual, expected);
        } else {
            expected = Some(actual);
        }
        let patches = (0..256)
            .flat_map(|_| [vec![1.0; 512], vec![-1.0; 512], vec![-1.0; 512]].concat())
            .collect();
        let spans = vec![darask_vision::Span {
            start: 1,
            image: darask_vision::Image {
                grid_h: 16,
                grid_w: 16,
                patches,
            },
        }];
        let prompt = [vec![248053], vec![248056; 64], vec![248054]].concat();
        engine.reset().unwrap();
        engine.set_images(&spans, &vision).unwrap();
        assert!(feed(&mut engine, &prompt).iter().all(|v| v.is_finite()));
    }
}
