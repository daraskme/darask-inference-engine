"""GLM header inventory and one-matrix block-FP8 restoration."""
import json
from pathlib import Path
import torch
from .sources import read_safetensors, TORCH_DTYPES
from .hyt import E4M3, F32

SOURCE = Path('/mnt/NVME4TB/darask-inference-engine/local/hayate/src/GLM-5.3-Flash-Uncensored-FP8')
GOLDEN = Path('/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/golden/glm53')
PREFIX = 'model.language_model.'


def dequantize(weight, scale):
    if weight.ndim != 2 or tuple(scale.shape) != tuple((d + 127) // 128 for d in weight.shape):
        raise ValueError('GLM requires 128x128 FP8 scales')
    return (weight.float() * scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
            [:weight.shape[0], :weight.shape[1]]).to(torch.bfloat16)


class Source:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root / 'config.json').read_text())
        if self.config['model_type'] != 'glm5_next' or self.config['quantization_config']['weight_block_size'] != [128, 128]:
            raise ValueError('unsupported GLM checkpoint')
        self.entries = {}
        for path in sorted(self.root.glob('*.safetensors')):
            for tensor in read_safetensors(path, {}):
                if tensor.name in self.entries:
                    raise ValueError('duplicate source tensor')
                self.entries[tensor.name] = tensor
        if not self.entries:
            raise ValueError('empty checkpoint')
        for name, tensor in self.entries.items():
            if tensor.dtype == E4M3:
                scale = self.entries.get(name.removesuffix('weight') + 'weight_scale_inv')
                if len(tensor.shape) != 2 or scale is None or scale.dtype != F32 or scale.shape != [(d + 127) // 128 for d in tensor.shape]:
                    raise ValueError(f'{name}: missing/invalid FP8 scale')

    def raw_tensor(self, name, device='cpu'):
        item = self.entries[name]
        with item.path.open('rb') as file:
            file.seek(item.start)
            raw = bytearray(file.read(item.size))
        if len(raw) != item.size:
            raise ValueError('short source tensor')
        value = torch.frombuffer(raw, dtype=TORCH_DTYPES[item.dtype]).reshape(item.shape).to(device)
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"{name}: nonfinite weight")
        return value

    def tensor(self, name, device='cpu'):
        item = self.entries[name]
        value = self.raw_tensor(name, device)
        if item.dtype == E4M3:
            value = dequantize(value, self.tensor(name.removesuffix('weight') + 'weight_scale_inv', device))
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}: nonfinite weight")
        return value

    def rows(self, name, ids):
        item = self.entries[name]
        if item.dtype != 1 or len(item.shape) != 2:
            raise ValueError('expected BF16 embedding')
        width = item.shape[1]
        raw = bytearray()
        with item.path.open('rb') as file:
            for index in ids.flatten().tolist():
                if not 0 <= index < item.shape[0]:
                    raise ValueError('embedding ID out of range')
                file.seek(item.start + index * width * 2)
                row = file.read(width * 2)
                if len(row) != width * 2:
                    raise ValueError('short embedding row')
                raw.extend(row)
        return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(*ids.shape, width)

    def inventory(self):
        totals = {}
        for name, item in self.entries.items():
            group = ('mtp' if '.layers.45.' in name else 'routed_experts' if '.mlp.experts.' in name
                     else 'vision' if '.visual.' in name else 'text_other')
            totals[group] = totals.get(group, 0) + item.size
        return totals
