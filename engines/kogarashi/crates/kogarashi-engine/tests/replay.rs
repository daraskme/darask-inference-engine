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
