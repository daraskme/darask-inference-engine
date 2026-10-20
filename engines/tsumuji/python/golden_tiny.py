"""Golden fixtures from transformers `qwen4_exp` on a shrunken random config (plan §10.1).

Builds a 4-layer text model (GDN, GDN+PLE, GDN, QSA) whose ratios match Qwen3.8-Flash-Next
(GDN v/k heads 3, GQA groups, partial RoPE 1/4, 4 hyper-connection streams, 16 PLE heads, QSA
budget smaller than the sequence so the indexer really selects), rounds every weight to BF16,
runs it in float32 on the CPU, and saves every op's inputs and outputs:

  fixtures/tiny/config.json        the config
  fixtures/tiny/weights.safetensors model weights (float32 holding BF16 values)
  fixtures/tiny/tensors.safetensors intermediates, named L<i>.<op>.<in|out>[.k]
  fixtures/tiny/meta.json           token ids, versions, incremental-decode check

The incremental check runs the same tokens as prefill + single-token steps with the cache and
compares the logits with the full forward: it pins down cache semantics (GDN/conv states,
PLE n-gram context with EOS resets, QSA compressed keys) that the decode kernels must match.

Run in Hayate's tools shell (transformers 5.17, torch 2.13), CPU only:
  nix develop /mnt/NVME4TB/darask-inference-engine/engines/hayate#tools -c python3 python/golden_tiny.py
"""

import json
import sys
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import DynamicCache
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "tiny"
EOS = 511
T = 96  # sequence length; > QSA dense limit (budget 16 + 3) so selection is exercised


def tiny_config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=6,
        num_key_value_heads=1,  # GQA 6 (real: 12)
        head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        hc_count=4,
        hc_lowrank=32,
        ple_layer_ids=[2],
        ple_embed_dim=256,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=1009,
        split_ngram_parts=4,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=16,
        indexer_compress_ratio=4,
        output_gate_type="sigmoid",
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [3, 3, 2],
        },
        rms_norm_eps=1e-6,
        eos_token_id=EOS,
        bos_token_id=EOS,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )


def randomize(model: torch.nn.Module) -> None:
    """Random weights with realistic scales, rounded to BF16. Norm weights are (1 + w) style,
    so small values around 0; A_log/dt_bias as in the reference init."""
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                v = torch.empty(p.shape).uniform_(0.01, 16, generator=g).log()
            elif name.endswith("dt_bias"):
                v = torch.ones(p.shape) + 0.1 * torch.randn(p.shape, generator=g)
            elif "norm" in name:
                v = 0.1 * torch.randn(p.shape, generator=g)
            elif "conv1d" in name:
                v = 0.3 * torch.randn(p.shape, generator=g)
            else:
                fan_in = p.shape[-1] if p.dim() > 1 else 1
                v = torch.randn(p.shape, generator=g) / fan_in**0.5
            p.copy_(v.bfloat16().float())


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """modelopt-style NVFP4 of a [rows, K] matrix: e2m1 codes (low nibble = even k), e4m3 scale
    per 16, fp32 global scale. Returns (codes u8 [rows, K/2], scales e4m3 [rows, K/16], gs, dequant)."""
    rows, k = w.shape
    gs = (w.abs().max() / (6.0 * 448.0)).float()
    blocks = w.float().view(rows, k // 16, 16)
    sf = (blocks.abs().amax(-1) / 6.0 / gs).clamp(min=2.0**-9).to(torch.float8_e4m3fn)
    step = (sf.float() * gs).unsqueeze(-1)
    scaled = blocks / step
    idx = (scaled.abs().clamp(max=6.0).unsqueeze(-1) - E2M1).abs().argmin(-1)
    neg = (scaled < 0) & (idx > 0)
    codes = (idx | (neg.long() << 3)).view(rows, k).to(torch.uint8)
    deq = (E2M1[idx] * torch.where(neg, -1.0, 1.0) * step).view(rows, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), sf.contiguous(), gs.reshape(()), deq


def quantize_like_radixark(model: Qwen4ExpForCausalLM) -> dict[str, torch.Tensor]:
    """Replaces routed experts by their NVFP4 values and the PLE table by FP8 x BF16 scale (what
    RadixArk/Qwen3.8-Flash-Next-NVFP4 ships), and returns that checkpoint's tensors under the real
    names, so the engine loads the tiny model through the same code as the real one."""
    out: dict[str, torch.Tensor] = {}
    text = "model.language_model."
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".mlp.experts." in name or "ngram_embedding" in name:
                continue
            key = "lm_head.weight" if name == "lm_head.weight" else text + name.removeprefix("model.")
            out[key] = p.detach().to(torch.bfloat16).contiguous()
        for name, b in model.named_buffers():
            if "ple_embedding" in name:
                out[text + name.removeprefix("model.")] = b.detach().contiguous()
        for li, layer in enumerate(model.model.layers):
            ex = layer.mlp.experts
            f = ex.intermediate_dim
            for e in range(ex.num_experts):
                pre = f"{text}layers.{li}.mlp.experts.{e}."
                mats = {
                    "gate_proj": ex.gate_up_proj[e, :f],
                    "up_proj": ex.gate_up_proj[e, f:],
                    "down_proj": ex.down_proj[e],
                }
                for proj, w in mats.items():
                    codes, sf, gs, deq = nvfp4(w)
                    w.copy_(deq)
                    out[pre + f"{proj}.weight"] = codes
                    out[pre + f"{proj}.weight_scale"] = sf
                    out[pre + f"{proj}.weight_scale_2"] = gs.clone()
                    out[pre + f"{proj}.input_scale"] = torch.ones(())
            if layer.ple is not None:
                emb = layer.ple.ple_embedding.ngram_embedding.weight
                scale = (emb.abs().max() / 448.0).to(torch.bfloat16)
                q = (emb / scale.float()).to(torch.float8_e4m3fn)
                emb.copy_(q.float() * scale.float())
                pre = f"{text}layers.{li}.ple.ple_embedding.ngram_embedding."
                parts = model.config.split_ngram_parts
                for i, shard in enumerate(q.chunk(parts, dim=0)):
                    out[pre + f"shard_{i}.weight"] = shard.contiguous()
                out[pre + "weight_scale"] = scale.reshape(1)
    return out


def tokens() -> list[int]:
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, EOS, (T,), generator=g).tolist()
    for p in (10, 11, 40, 41, 42, 70):  # EOS runs reset the PLE n-gram context
        ids[p] = EOS
    return ids


def capture(model: Qwen4ExpForCausalLM) -> tuple[dict, list]:
    out: dict[str, torch.Tensor] = {}

    def save(key, value):
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().float().contiguous().clone()
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                save(f"{key}.{i}", v)

    hooks = []

    def watch(module, key, inputs=True):
        def fwd(_m, args, kwargs, result):
            if inputs:
                save(f"{key}.in", list(args) + [v for v in kwargs.values() if isinstance(v, torch.Tensor)])
            save(f"{key}.out", result)

        hooks.append(module.register_forward_hook(fwd, with_kwargs=True))

    m = model.model
    watch(m.embed_tokens, "embed", inputs=False)
    for i, layer in enumerate(m.layers):
        watch(layer, f"L{i}")
        if layer.ple is not None:
            watch(layer.ple, f"L{i}.ple")
            watch(layer.ple.ple_embedding, f"L{i}.ple.ngram", inputs=False)
        watch(layer.attn_hyper_connection, f"L{i}.gr_attn")
        watch(layer.mlp_hyper_connection, f"L{i}.gr_mlp")
        if layer.layer_type == "linear_attention":
            watch(layer.linear_attn, f"L{i}.gdn")
        else:
            watch(layer.self_attn, f"L{i}.attn")
            watch(layer.self_attn.indexer, f"L{i}.indexer", inputs=False)
        watch(layer.mlp.gate, f"L{i}.router")
        watch(layer.mlp.experts, f"L{i}.experts")
        watch(layer.mlp.shared_expert, f"L{i}.shared")
        watch(layer.mlp, f"L{i}.moe")
    watch(m.hyper_connection_mixer, "mixer")
    watch(model.lm_head, "lm_head")
    return out, hooks


def main() -> int:
    torch.manual_seed(0)
    cfg = tiny_config()
    cfg._attn_implementation = "eager"
    model = Qwen4ExpForCausalLM(cfg).eval()
    randomize(model)
    checkpoint = quantize_like_radixark(model)
    ids = tokens()
    x = torch.tensor([ids])

    tensors, hooks = capture(model)
    with torch.no_grad():
        full = model(input_ids=x, use_cache=False).logits[0]
    for h in hooks:
        h.remove()

    # Incremental: prefill P tokens, then one token at a time with the cache.
    P = 48
    worst = 0.0
    with torch.no_grad():
        cache = DynamicCache(config=cfg)
        step = model(input_ids=x[:, :P], past_key_values=cache, use_cache=True).logits[0]
        worst = max(worst, (step - full[:P]).abs().max().item())
        for t in range(P, T):
            step = model(input_ids=x[:, t : t + 1], past_key_values=cache, use_cache=True).logits[0, -1]
            worst = max(worst, (step - full[t]).abs().max().item())
    scale = full.abs().max().item()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg.to_json_file(OUT / "config.json")
    weights = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    save_file(weights, OUT / "weights.safetensors")
    tensors["logits"] = full.contiguous()
    save_file(tensors, OUT / "tensors.safetensors")
    ckpt = OUT / "ckpt"
    ckpt.mkdir(exist_ok=True)
    save_file(checkpoint, ckpt / "model.safetensors")
    (ckpt / "config.json").write_text(json.dumps({"text_config": cfg.to_dict()}, indent=1) + "\n")
    meta = {
        "tokens": ids,
        "eos": EOS,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "incremental_prefill": P,
        "incremental_max_abs_diff": worst,
        "logits_max_abs": scale,
        "n_tensors": len(tensors),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(tensors)} tensors, {len(weights)} weights -> {OUT}")
    print(f"incremental decode vs full forward: max |diff| {worst:.3e} (logits max {scale:.2f})")
    return 0 if worst < 1e-3 * max(scale, 1.0) else 1


if __name__ == "__main__":
    sys.exit(main())
