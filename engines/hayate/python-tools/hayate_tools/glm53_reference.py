"""Layer-at-a-time transformers GLM oracle; BF16 activations, restored source FP8."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
from .glm53_source import Source, SOURCE, GOLDEN, PREFIX
from .golden import write_tensor


def text_config(raw, layers=5):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    raw = dict(raw)
    raw['num_hidden_layers'] = layers
    for name in ('layer_types', 'mlp_layer_types', 'indexer_types'):
        raw[name] = raw[name][:layers]
    config = Glm5NextTextConfig(**raw)
    config._attn_implementation = 'eager'
    config._experts_implementation = 'eager'
    return config


def checkpoint_name(name):
    for site, old in (('attn_hc.', 'hc_attn_'), ('ffn_hc.', 'hc_ffn_')):
        if name.startswith(site):
            return old + name[len(site):]
    return name.replace('self_attn.forget_gate.', 'self_attn.')


class StreamingExperts(torch.nn.Module):
    """Execute only selected experts, in expert-ID order and top-k-slot/token row order."""
    def __init__(self, source, prefix, device):
        super().__init__()
        self.source, self.prefix, self.device = source, prefix, device

    def forward(self, hidden, ids, weights):
        result = torch.zeros_like(hidden)
        for expert in sorted(set(ids.flatten().tolist())):
            slot, token = torch.nonzero(ids.T == expert, as_tuple=True)
            stem = self.prefix + f'{expert}.'
            gate = torch.nn.functional.linear(hidden[token], self.source.tensor(stem + 'gate_proj.weight', self.device)).clamp(max=10)
            up = torch.nn.functional.linear(hidden[token], self.source.tensor(stem + 'up_proj.weight', self.device)).clamp(-10, 10)
            product = torch.nn.functional.silu(gate) * up
            value = torch.nn.functional.linear(product, self.source.tensor(stem + 'down_proj.weight', self.device))
            result.index_add_(0, token, (value * weights[token, slot, None]).to(result.dtype))
        return result


def load_layer(source, config, index, device):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
    with torch.device('meta'):
        layer = Glm5NextTextDecoderLayer(config, index)
    prefix = PREFIX + f'layers.{index}.'
    if config.mlp_layer_types[index] == 'sparse':
        layer.mlp.experts = StreamingExperts(source, prefix + 'mlp.experts.', device)
    for name, _ in list(layer.named_parameters()):
        if name == 'self_attn.conv1d.weight':
            value = torch.cat([source.tensor(prefix + f'self_attn.{kind}_conv1d.weight', device) for kind in 'qkv'])
        else:
            value = source.tensor(prefix + checkpoint_name(name), device)
        parent, _, leaf = name.rpartition('.')
        setattr(layer.get_submodule(parent), leaf, torch.nn.Parameter(value, requires_grad=False))
    for name, _ in list(layer.named_buffers()):
        parent, _, leaf = name.rpartition('.')
        setattr(layer.get_submodule(parent), leaf, source.tensor(prefix + checkpoint_name(name), device))
    return layer.eval()


def trace_layer(layer, hidden, folder):
    from transformers.models.glm5_next import modeling_glm5_next as impl
    hooks = []
    def save(name, value):
        if isinstance(value, torch.Tensor):
            write_tensor(folder / name, value)
        elif isinstance(value, (tuple, list)):
            for i, part in enumerate(value):
                save(f'{name}_{i}', part)
    hooks.append(layer.ffn_hc.register_forward_pre_hook(lambda _, args: save('attention_residual', args[0])))
    for name, module in layer.named_modules():
        if name and not name.startswith('mlp.experts.'):
            hooks.append(module.register_forward_hook(lambda _, args, value, n=name: save(n, value)))
    # Functions (conv/KDA) are not modules, so record their returned tensors too.
    original_kda, original_conv = impl.chunk_kimi_delta_attention, impl.causal_conv1d_fn
    def kda(*args, **kwargs):
        kwargs['output_final_state'] = True
        output = original_kda(*args, **kwargs)
        save('kda', output)
        return output
    def conv(*args, **kwargs):
        output = original_conv(*args, **kwargs)
        save('conv', output.transpose(1, 2))
        return output
    impl.chunk_kimi_delta_attention, impl.causal_conv1d_fn = kda, conv
    write_tensor(folder / 'input', hidden)
    try:
        result, _ = layer(hidden, attention_mask=torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device))
        write_tensor(folder / 'output', result)
        return result
    finally:
        impl.chunk_kimi_delta_attention, impl.causal_conv1d_fn = original_kda, original_conv
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def generate(args):
    import transformers
    from transformers.models.glm5_next import modeling_glm5_next as impl
    if not 5 <= args.layers <= 45 or not 1 <= args.tokens <= 8192:
        raise ValueError('expected 5..45 layers, 1..8192 tokens')
    if not 1 <= getattr(args, 'max_tokens', 1) <= 8192:
        raise ValueError('expected max-tokens 1..8192')
    torch.backends.cuda.matmul.allow_tf32 = False
    source = Source(args.source)
    config = text_config(source.config['text_config'], args.layers)
    if getattr(args, 'prompt', None) is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
        ids = torch.tensor([tokenizer.encode(args.prompt) if args.raw else tokenizer.apply_chat_template([{'role':'user','content':args.prompt}], tokenize=True, add_generation_prompt=True, return_dict=False)])
    else:
        ids = ((torch.arange(args.tokens) * 7919 + 17) % config.vocab_size)[None]
    if ids.shape[1] + args.max_tokens - 1 > 8192:
        raise ValueError('reference prompt + generation exceeds 8192')
    hidden = source.rows(PREFIX + 'embed_tokens.weight', ids).to(args.device)
    hidden = hidden.unsqueeze(2).expand(-1, -1, 4, -1).contiguous()
    write_tensor(args.out / 'input_ids', ids)
    start = time.perf_counter()
    for index in range(args.layers):
        layer = load_layer(source, config, index, args.device)
        hidden = trace_layer(layer, hidden, args.out / f'layer{index}')
        del layer
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    # Average streams, weighted RMS and final projection are part of the oracle.
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm
    norm = Glm5NextTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(args.device, torch.bfloat16)
    norm.weight.data.copy_(source.tensor(PREFIX+'norm.weight', args.device))
    final = norm(hidden.mean(dim=2))
    head = source.tensor('lm_head.weight', args.device)
    logits = torch.nn.functional.linear(final, head).float()
    write_tensor(args.out/'logits', logits)
    write_tensor(args.out/'greedy', logits.argmax(-1))
    generated = [int(logits[0,-1].argmax())]
    write_tensor(args.out/'step0', logits[0,-1])
    if getattr(args, 'max_tokens', 1) > 1:
        # Layer streaming avoids keeping 45 layers or expanded experts resident.
        # Recompute the prefix for each oracle decode step: deliberately slow,
        # independent of Rust's persistent state implementation.
        del hidden, final, head, logits, norm
        for step in range(1, args.max_tokens):
            ids = torch.cat([ids, torch.tensor([[generated[-1]]])], dim=1)
            current = streaming_logits(source, config, ids, args.device)
            write_tensor(args.out/f'step{step}', current)
            generated.append(int(current.argmax()))
    prompt_length = ids.shape[1] - len(generated) + 1
    (args.out/'ids.json').write_text(json.dumps({'input': ids[0,:prompt_length].tolist(), 'generated': generated})+'\n')
    metadata = {'model': 'glm5_next_text', 'layers': args.layers, 'tokens': prompt_length, 'generated_tokens': len(generated),
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'implementation_sha256': hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest(),
                'device': args.device, 'seed': 53, 'source': str(args.source),
                'source_bytes': source.inventory(), 'wall_seconds': time.perf_counter() - start,
                'attention': 'eager', 'experts': 'expert ID ascending; slot then token',
                'fp8': 'source E4M3 times F32 block scales rounded to BF16, no requantization'}
    (args.out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


@torch.inference_mode()
def streaming_logits(source, config, ids, device):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm
    hidden = source.rows(PREFIX+'embed_tokens.weight', ids).to(device)
    hidden = hidden.unsqueeze(2).expand(-1,-1,config.hc_mult,-1).contiguous()
    for index in range(config.num_hidden_layers):
        layer = load_layer(source, config, index, device)
        hidden, _ = layer(hidden, attention_mask=torch.ones(ids.shape, dtype=torch.bool, device=device))
        del layer
        gc.collect()
    norm = Glm5NextTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device, torch.bfloat16)
    norm.weight.data.copy_(source.tensor(PREFIX+'norm.weight', device))
    final = norm(hidden[:, -1:].mean(dim=2))
    return torch.nn.functional.linear(final, source.tensor('lm_head.weight', device))[0,0].float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=GOLDEN)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--tokens', type=int, default=17)
    parser.add_argument('--prompt')
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    generate(parser.parse_args())


if __name__ == '__main__':
    main()
