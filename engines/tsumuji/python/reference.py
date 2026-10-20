"""Reference logits from transformers `qwen4_exp` for the KL gate (plan §10.2), on one GPU.

The full BF16 model (~250 GB without the PLE table) does not fit, so layers are streamed: each
layer's weights go to the GPU once, every eval sequence runs through it, then it is freed. The PLE
table (51 GB FP8 / 102 GB BF16) is never loaded; the rows a sequence needs are read from the
checkpoint files directly.

  python3 python/reference.py --model DIR [--experts nvfp4] --out DIR eval/a.ids eval/b.ids ...

--model         checkpoint for everything: the BF16 original (Qwen/Qwen3.8-Flash-Next) or the
                RadixArk NVFP4 one, whose experts are dequantized to BF16 and whose FP8 PLE rows are
                scaled to BF16 (then the reference has the same weights as the engine, so the KL
                measures the engine alone).
Writes <out>/<ids stem>.f32: logits [tokens][vocab] float32, the same layout as `tsumuji run --logits`.

Run in Hayate's tools shell (transformers 5.17) through scripts/gpu-run.sh.
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextRotaryEmbedding

DEV = "cuda"
DTYPE = torch.bfloat16
TEXT = "model.language_model."
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {}

    def has(self, name: str) -> bool:
        return name in self.map

    def get(self, name: str) -> torch.Tensor:
        f = self.map[name]
        if f not in self.handles:
            self.handles[f] = safe_open(self.path / f, framework="pt", device="cpu")
        return self.handles[f].get_tensor(name)

    def raw_location(self, name: str) -> tuple[str, int, str, list[int]]:
        """(file, absolute data offset, dtype, shape) of a tensor, for direct row reads."""
        f = self.path / self.map[name]
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        h = header[name]
        return str(f), 8 + n + h["data_offsets"][0], h["dtype"], h["shape"]


class PleRows(torch.nn.Module):
    """Stands in for the n-gram nn.Embedding: reads only the requested rows from the shards."""

    def __init__(self, ck: Checkpoint, layer: int):
        super().__init__()
        pre = f"{TEXT}layers.{layer}.ple.ple_embedding.ngram_embedding."
        self.shards = []
        i = 0
        while ck.has(f"{pre}shard_{i}.weight"):
            self.shards.append(ck.raw_location(f"{pre}shard_{i}.weight"))
            i += 1
        self.rows_per_shard = self.shards[0][3][0]
        self.width = self.shards[0][3][1]
        self.dtype = self.shards[0][2]
        self.scale = ck.get(pre + "weight_scale").float().item() if ck.has(pre + "weight_scale") else 1.0
        self.fds = {}
        self.weight = torch.empty(0, device=DEV)  # the caller reads .weight.device

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).tolist()
        uniq = sorted(set(flat))
        elem = 1 if self.dtype == "F8_E4M3" else 2
        row_bytes = self.width * elem
        buf = bytearray(len(uniq) * row_bytes)
        for j, r in enumerate(uniq):
            path, off, _, _ = self.shards[r // self.rows_per_shard]
            if path not in self.fds:
                self.fds[path] = os.open(path, os.O_RDONLY)
            local = r % self.rows_per_shard
            buf[j * row_bytes : (j + 1) * row_bytes] = os.pread(self.fds[path], row_bytes, off + local * row_bytes)
        if self.dtype == "F8_E4M3":
            table = torch.frombuffer(buf, dtype=torch.float8_e4m3fn).float() * self.scale
        else:
            table = torch.frombuffer(buf, dtype=torch.bfloat16).float()
        table = table.view(len(uniq), self.width).to(DEV)
        index = {r: j for j, r in enumerate(uniq)}
        sel = torch.tensor([index[r] for r in flat], device=DEV)
        return table[sel].view(*ids.shape, self.width).to(DTYPE)


def nvfp4_dequant(ck: Checkpoint, pre: str) -> torch.Tensor:
    codes = ck.get(pre + ".weight").to(DEV)
    sf = ck.get(pre + ".weight_scale").to(DEV).float()
    gs = ck.get(pre + ".weight_scale_2").float().item()
    lut = E2M1.to(DEV)
    vals = torch.stack([lut[(codes & 15).long()], lut[(codes >> 4).long()]], dim=-1).view(codes.shape[0], -1)
    return (vals.view(codes.shape[0], -1, 16) * sf.unsqueeze(-1) * gs).view(codes.shape[0], -1)


def load_layer(layer: torch.nn.Module, ck: Checkpoint, i: int) -> None:
    if layer.ple is not None:  # before to_empty: the real table would not fit anywhere
        layer.ple.ple_embedding.ngram_embedding = PleRows(ck, i)
    layer.to_empty(device=DEV)
    pre = f"{TEXT}layers.{i}."
    for name, t in list(layer.named_parameters()) + list(layer.named_buffers()):
        if name == "ple.ple_embedding.ngram_embedding.weight":
            continue
        key = pre + name
        if name.startswith("mlp.experts.") and not ck.has(key):
            e = layer.mlp.experts
            f = e.intermediate_dim
            for x in range(e.num_experts):
                ep = f"{pre}mlp.experts.{x}."
                if name.endswith("gate_up_proj"):
                    t[x, :f].copy_(nvfp4_dequant(ck, ep + "gate_proj"))
                    t[x, f:].copy_(nvfp4_dequant(ck, ep + "up_proj"))
                else:
                    t[x].copy_(nvfp4_dequant(ck, ep + "down_proj"))
            continue
        t.copy_(ck.get(key).to(DEV))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="compute dtype of the reference model")
    ap.add_argument("ids", nargs="+", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ck = Checkpoint(a.model)
    cfg = AutoConfig.from_pretrained(a.model).text_config
    cfg._attn_implementation = "sdpa"
    seqs = [[int(x) for x in p.read_text().replace(",", " ").split()] for p in a.ids]

    global DTYPE
    DTYPE = getattr(torch, a.dtype)
    torch.set_default_dtype(DTYPE)
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(cfg)
    torch.set_default_dtype(torch.float32)
    m = model.model
    rotary = Qwen4ExpTextRotaryEmbedding(cfg).to(DEV)
    embed = ck.get(TEXT + "embed_tokens.weight").to(DEV).to(DTYPE)

    states, extras = [], []
    for ids in seqs:
        x = torch.tensor([ids], device=DEV)
        emb = embed[x]
        t = len(ids)
        pos = torch.arange(t, device=DEV).view(1, 1, -1).expand(4, 1, -1)
        kw = dict(config=cfg, inputs_embeds=emb, attention_mask=None, past_key_values=None, position_ids=pos[0], allow_is_causal_skip=False)
        mask = create_causal_mask(**kw)
        conv_mask = create_recurrent_attention_mask(**kw)
        pe = rotary(emb, pos[1:])
        states.append(emb.repeat(1, 1, cfg.hc_count))
        extras.append((mask, conv_mask, pe, x))
    del embed

    t0 = time.time()
    for i, layer in enumerate(m.layers):
        load_layer(layer, ck, i)
        for s, (mask, conv_mask, pe, x) in enumerate(extras):
            states[s] = layer(states[s], position_embeddings=pe, attention_mask=mask, conv_mask=conv_mask, past_key_values=None, ple_input_ids=x)
        layer.to("meta")
        torch.cuda.empty_cache()
        print(f"layer {i} done ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)

    mixer = m.hyper_connection_mixer.to_empty(device=DEV)
    for name, t in mixer.named_parameters():
        t.copy_(ck.get(TEXT + "hyper_connection_mixer." + name).to(DEV))
    head = ck.get("lm_head.weight").to(DEV).float()
    for p, st in zip(a.ids, states):
        logits = (mixer(st).float() @ head.T)[0]  # fp32 head: bf16 logits would add noise to the KL
        (a.out / (p.stem + ".f32")).write_bytes(logits.cpu().numpy().tobytes())
        print(f"{p.stem}: {logits.shape[0]} positions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    with torch.inference_mode():
        sys.exit(main())
