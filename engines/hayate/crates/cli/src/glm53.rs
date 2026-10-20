use hayate_core::Result;

pub fn run(args: &[String]) -> Result<()> {
    if args.first().is_some_and(|a| a == "--help" || a == "-h") {
        println!(
            "usage: hayate run glm53 --prompt TEXT --max-tokens N\n  [--model-dir DIR --hyt-dir DIR --layers 45 --context 4096 --host-gb N]\n  [--allow-disk --frequencies FILE --save-frequencies FILE --warmup N]\n  [--prefill-chunk 0 (auto, max 8192) --prefill-legacy --prefill-experts previous|all --raw --dump-logits PREFIX]\n  previous: short prompts (<512) use decode CPU misses; all forces GPU batch prefill\n  [--profile --no-graphs --decode-reference --decode-host sync|async (default sync)]\n  async: pcie-frac=0, RAM misses, no expert-prefetch/cpu-spin; poller on E-core 22\n  [--spec-draft-n 0..3 --spec-no-adaptive --profile-spec --ignore-eos]\n  [--spec-reserve-mtp (n=0 only) --mtp-experts gpu|cpu (default gpu)]\n  --dump-logits adds synchronous diagnostic I/O; exclude from performance runs\n  [--output-ids FILE --spec-validate TOKENS] (all n=1..3, forced acceptance 0..n; requires --mtp conversion)\n  [--vram-gib N (total GLM VRAM budget; default: machine plan with 8 GiB spare) --resize-at TOKEN:GIB (elastic test)]\n  [--lora FILE.gguf[:STRENGTH] (repeatable; llama.cpp GGUF LoRA: attn_output, ffn_down_shexp/exps, dense ffn; side path, no MTP)]\n  [--maintenance-interval 1 --expert-readers 4 --pcie-frac 0 (0..1) --expert-prefetch]\n  [--pcie-take 0,0,1,1,2,2,3,3,4 | off (device decode: CPU share of a layer's n RAM misses; the rest are fetched over PCIe and computed on the GPU)]\n  [--pcie-prefetch 1 (0..4: next-layer predicted RAM misses fetched ahead on a side stream; 0 off)]\n  [--cpu-threads 8 --host-core 23 (none/0..23) --cpu-spin]\n  [--maintenance-window 64 --maintenance-swaps 1 --maintenance-hysteresis 4]\n  [--maintenance-async (comparison only; default copies finish at token boundary)]\n  --eval-ids FILE [--eval-prefix 64 --eval-mode prefill|step] --out DIR (teacher forcing, native BF16 logits)\nHost budget defaults to the required expert bytes within MemTotal - 24 GiB.\n45-layer timing: collect --save-frequencies FILE, then reload with --frequencies FILE --warmup 2.\nRotation and down FP8 are read from HYT weights; no runtime quantization flag."
        );
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    {
        hayate_models::glm53::run(args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(hayate_core::Error::InvalidPlan(
            "GLM5.3 requires --features cuda".into(),
        ))
    }
}
