"""Little-endian row-major tensor bytes plus a JSON sidecar for Rust oracles."""

import json
import math
from pathlib import Path
import sys

import torch

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
          "F64": torch.float64, "I64": torch.int64, "I32": torch.int32, "I8": torch.int8,
          "U8": torch.uint8}


def write_tensor(prefix, value, **metadata):
    if sys.byteorder != "little":
        raise ValueError("golden writer requires a little-endian host")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    dtype = next((name for name, kind in DTYPES.items() if kind == value.dtype), None)
    if dtype is None:
        raise ValueError(f"unsupported golden dtype: {value.dtype}")
    info = {"version": 1, "byte_order": "little", "dtype": dtype, "shape": list(value.shape)}
    if info.keys() & metadata.keys():
        raise ValueError("metadata overwrites tensor layout")
    info.update(metadata)
    # Copy into fresh storage: broadcast (stride-0) views keep stride 0 through clone().
    flat = value.detach().cpu().reshape(-1)
    torch.empty(flat.numel(), dtype=flat.dtype).copy_(flat).view(torch.uint8).numpy().tofile(str(prefix) + ".bin")
    Path(str(prefix) + ".json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")


def read_tensor(prefix, device="cpu"):
    prefix = str(prefix)
    info = json.loads(Path(prefix + ".json").read_text())
    shape, dtype = info.get("shape"), DTYPES.get(info.get("dtype"))
    if (info.get("version") != 1 or info.get("byte_order") != "little" or dtype is None
            or not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape)):
        raise ValueError("invalid golden layout")
    data = bytearray(Path(prefix + ".bin").read_bytes())
    count = math.prod(shape)
    if len(data) != count * torch.empty((), dtype=dtype).element_size():
        raise ValueError("golden byte count mismatch")
    value = (torch.frombuffer(data, dtype=dtype).reshape(shape) if count
             else torch.empty(shape, dtype=dtype))
    return value.to(device), info
