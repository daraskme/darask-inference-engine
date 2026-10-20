"""GLM-5.3-Flash EXL3 (exllamav3 1.6, e.g. turboderp's sc_2.60bpw) -> Hayate HYT.

Routed experts keep their trellis bytes: HYT tag 7 (`QuantSpec::Exl3`, crates/models/src/glm53/EXL3.md), one
matrix [outputs, inputs] = trellis tiles | FP16 suh [inputs] | FP16 svh [outputs].  Every other EXL3 matrix
(attention, shared experts, dense FFN, lm_head) is decoded here on the CPU and written in the tensor schema of an
existing Hayate GLM checkpoint (by default the f100 mix: per-tensor FP8 linears, BF16/F32 for the rest), so the
runtime's dense paths read it as they read f100.  The fused KDA q|k|v projection and conv are split in three.

The format (exllamav3, MIT, turboderp; decoded as Project Maya's tools/exl3_pack.py does, MIT): a k x n matrix
(y = x W) is [k/16][n/16] tiles of 256 steps; step p's 16-bit state is the window ending at bit
floor((p + 1) k2 / 2) of the tile's 4 k2 little-endian uint32 words read MSB-first, around the tile (k2 = 2 K; odd
k2 is exllamav3 1.6's half-integer K, mul1 only).  The mul1 codebook value of a state s is
fp16((1024 + bytesum(s * 0x83DCD12D)) * fp16(0x1eee) + fp16(0xc931)); step p sits at tile row
(p/8 % 4) * 2 + (p & 1) + 8 (p/2 & 1), column p/32 + 8 (p/4 & 1).  With H the 128-point Sylvester Hadamard over
sqrt(128) on each 128 consecutive values, W = diag(suh) H W' H diag(svh).

    python3 -m hayate_tools.glm53_exl3 fixtures python-tools/tests/data/exl3-exllamav3
    python3 -m hayate_tools.glm53_exl3 convert --model DIR --out DIR [--schema DIR] [--estimate]
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .hyt import (BF16, E4M3, EXL3, F32, FORMAT_VERSION, HEADER, PER_TENSOR, PLAIN, align, encode_entry,
                  exl3_valid, storage_bytes)

PREFIX = 'model.language_model.'
SCHEMA = Path('/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53q-f100-packed')
MUL1 = 0x83DCD12D
LAYERS, EXPERTS, KDA_INNER = 45, 288, 8192
_P = np.arange(256)
ROW = (_P >> 3 & 3) * 2 + (_P & 1) + 8 * (_P >> 1 & 1)
COL = (_P >> 5) + 8 * (_P >> 2 & 1)


def _sylvester(n):
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


HAD = _sylvester(128)


# ------------------------------------------------------------------ the decoder
def tile_states(words, k2):
    """(..., 4 k2) uint32 tile words -> (..., 256) uint32 16-bit states in step order."""
    w = 4 * k2
    b1 = ((_P + 1) * k2) // 2 + 128 * k2      # one ring on: a window never starts below bit 0
    i1, i0 = (b1 - 1) >> 5, (b1 - 16) >> 5
    shift = (((i1 + 1) << 5) - b1).astype(np.uint64)
    m = (words[..., i0 % w].astype(np.uint64) << np.uint64(32)) | words[..., i1 % w].astype(np.uint64)
    return ((m >> shift) & np.uint64(0xFFFF)).astype(np.uint32)


def codebook(states, cb):
    """States -> their codebook values as float32 (exactly the FP16 values the kernels decode)."""
    f16 = lambda bits: np.uint16(bits).view(np.float16).astype(np.float64)  # noqa: E731
    if cb == 2:
        x = (states.astype(np.uint64) * MUL1) & np.uint64(0xFFFFFFFF)
        total = (x & 0xFF) + (x >> 8 & 0xFF) + (x >> 16 & 0xFF) + (x >> 24)
        # One FP16 FMA on the device: the product and sum are exact in F64, rounded once.
        return ((1024. + total.astype(np.float64)) * f16(0x1EEE) + f16(0xC931)).astype(np.float16).astype(np.float32)
    x = states.astype(np.uint64) * (0xCBAC1FED if cb == 1 else 89226354)
    if cb == 0:
        x += np.uint64(64248484)
    x = ((x & np.uint64(0xFFFFFFFF)) & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
    lo = (x & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float64)
    hi = (x >> np.uint64(16)).astype(np.uint16).view(np.float16).astype(np.float64)
    return (lo + hi).astype(np.float16).astype(np.float32)


def k2_of(trellis_shape, cb=2):
    """k2 = 2 K from the int16 words a tile (16 K): 8 k2 words."""
    words = trellis_shape[-1]
    if words % 8 or not exl3_valid(words // 8, cb):
        raise ValueError(f'{words} int16 words a tile is no EXL3 bitrate')
    return words // 8


def inner(trellis, k2, cb=2, c0=0, nc=None):
    """W' [k][nc] (float32, the rotated basis) of output columns [c0, c0 + nc); trellis int16 [k/16][n/16][8 k2]."""
    kt, nt, _ = trellis.shape
    nc = nt * 16 - c0 if nc is None else nc
    if c0 % 16 or nc % 16 or c0 + nc > nt * 16:
        raise ValueError('EXL3 column range')
    words = np.ascontiguousarray(trellis[:, c0 // 16:(c0 + nc) // 16, :]).view(np.uint32)
    values = codebook(tile_states(words, k2), cb)
    tile = np.empty(values.shape, dtype=np.float32)
    tile[..., ROW * 16 + COL] = values
    return tile.reshape(kt, nc // 16, 16, 16).transpose(0, 2, 1, 3).reshape(kt * 16, nc)


def rows(trellis, suh, svh, k2, cb=2, c0=0, nc=None):
    """Rows [c0, c0 + nc) of the full weight in Hayate's [outputs, inputs] layout, float32 (F64 rotations)."""
    w = inner(trellis, k2, cb, c0, nc).astype(np.float64)
    k, nc = w.shape
    if k % 128 or nc % 128 or c0 % 128:
        raise ValueError('EXL3 rotations need 128-multiples')
    w = (w.reshape(k, nc // 128, 128) @ HAD).reshape(k, nc)
    w = (HAD @ w.reshape(k // 128, 128, nc)).reshape(k, nc)
    w *= suh.astype(np.float64)[:, None]
    w *= svh[c0:c0 + nc].astype(np.float64)[None, :]
    return np.ascontiguousarray(w.T, dtype=np.float32)


def read_fixture(path):
    """exllamav3's own decode of random tiles (Project Maya tools/exl3_exllamav3_fixture.py)."""
    raw = Path(path).read_bytes()
    if raw[:4] != b'EXH1':
        raise ValueError(f'{path}: not an EXH1 fixture')
    k2, ki, nj, cb, words = struct.unpack('<5i', raw[4:24])
    tiles = ki * nj
    if words != 8 * k2 or len(raw) != 24 + tiles * words * 2 + tiles * 512 * 2:
        raise ValueError(f'{path}: bad fixture')
    at = 24 + tiles * words * 2
    trellis = np.frombuffer(raw, np.int16, tiles * words, 24).reshape(ki, nj, words)
    states = np.frombuffer(raw, np.uint16, tiles * 256, at).reshape(ki, nj, 256)
    w_hat = np.frombuffer(raw, np.uint16, tiles * 256, at + tiles * 512).reshape(ki * 16, nj * 16)
    return k2, cb, trellis, states, w_hat


def check_fixtures(folder):
    results = []
    for path in sorted(Path(folder).glob('exl3_k2_*_cb*.bin')):
        k2, cb, trellis, states, w_hat = read_fixture(path)
        bad_states = int((tile_states(trellis.view(np.uint32), k2) != states).sum())
        got = inner(trellis, k2, cb).astype(np.float16).view(np.uint16)
        results.append({'file': path.name, 'k2': k2, 'cb': cb, 'bad_states': bad_states,
                        'bad_weights': int((got != w_hat).sum()), 'weights': int(w_hat.size)})
    if not results:
        raise ValueError(f'no fixtures in {folder}')
    return results


# ------------------------------------------------------------------ the checkpoint
class SafeTensors:
    """Every tensor of an EXL3 model directory, read in place (memory-mapped)."""
    DTYPES = {'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16, 'I16': np.int16, 'I32': np.int32}

    def __init__(self, root):
        self.root = Path(root)
        index = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        self.tensors, self.maps = {}, {}
        for name in sorted(set(index.values())):
            path = self.root / name
            with path.open('rb') as file:
                length = struct.unpack('<Q', file.read(8))[0]
                header = json.loads(file.read(length))
            self.maps[name] = np.memmap(path, dtype=np.uint8, mode='r')
            for key, info in header.items():
                if key == '__metadata__' or index.get(key) != name:
                    continue
                start, end = info['data_offsets']
                if info['dtype'] not in self.DTYPES:
                    raise ValueError(f'{key}: unsupported dtype {info["dtype"]}')
                self.tensors[key] = (name, info['dtype'], list(info['shape']), 8 + length + start, end - start)
        missing = set(index) - set(self.tensors)
        if missing:
            raise ValueError(f'{len(missing)} indexed tensors missing, e.g. {sorted(missing)[0]}')

    def has(self, name):
        return name in self.tensors

    def raw(self, name):
        file, _, _, offset, size = self.tensors[name]
        return self.maps[file][offset:offset + size]

    def array(self, name):
        _, dtype, shape, _, _ = self.tensors[name]
        return self.raw(name).view(self.DTYPES[dtype]).reshape(shape)

    def f32(self, name):
        dtype = self.tensors[name][1]
        value = self.array(name)
        if dtype == 'BF16':
            return (value.astype(np.uint32) << 16).view(np.float32)
        if dtype in ('F16', 'F32'):
            return value.astype(np.float32)
        raise ValueError(f'{name} is {dtype}, not a float tensor')

    def matrix(self, base):
        """(trellis int16, suh f16, svh f16, k2) of an EXL3 matrix; only the mul1 codebook is written by 1.6."""
        mul1 = self.array(base + '.mul1')
        if int(mul1.reshape(-1)[0]) & 0xFFFFFFFF != MUL1:
            raise ValueError(f'{base}: not the mul1 codebook')
        trellis = self.array(base + '.trellis')
        suh, svh = self.array(base + '.suh'), self.array(base + '.svh')
        k2 = k2_of(trellis.shape)
        if suh.shape != (trellis.shape[0] * 16,) or svh.shape != (trellis.shape[1] * 16,):
            raise ValueError(f'{base}: suh/svh shape')
        if not (np.isfinite(suh).all() and np.isfinite(svh).all()):
            raise ValueError(f'{base}: nonfinite suh/svh')
        return trellis, suh, svh, k2


def hyt_table(path):
    """[(name, quant, shape)] of a HYT file (the schema)."""
    with Path(path).open('rb') as file:
        _, _, _, count, table, _, _ = HEADER.unpack(file.read(HEADER.size))
        file.seek(table)
        entries = []
        for _ in range(count):
            length = struct.unpack('<I', file.read(4))[0]
            name = file.read(length).decode()
            tag, dtype, group, rank = struct.unpack('<4I', file.read(16))
            shape = list(struct.unpack(f'<{rank}Q', file.read(8 * rank)))
            file.read(16)
            entries.append((name, (tag, dtype, group), shape))
    return entries


# ------------------------------------------------------------------ one layer
def to_bf16(value):
    """float32 -> BF16 bytes (round to nearest even) and how many values were not exactly BF16."""
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = ((bits.astype(np.uint64) + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded.tobytes(), int(np.count_nonzero((rounded.astype(np.uint32) << 16) != bits))


def encode(value, quant):
    """(payload bytes, inexact BF16 count) of a float32 value in a schema format."""
    if not np.isfinite(value).all():
        raise ValueError('nonfinite weight')
    if quant == (PLAIN, BF16, 0):
        return to_bf16(value)
    if quant == (PLAIN, F32, 0):
        return np.ascontiguousarray(value, dtype=np.float32).tobytes(), 0
    if quant == (PER_TENSOR, E4M3, 0):
        import torch
        from .quant import fp8
        return fp8(torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))).numpy().tobytes(), 0
    raise ValueError(f'unsupported schema format {quant}')


def decoded(st, base, chunk=2048):
    """The full float32 weight [outputs, inputs] of an EXL3 matrix, decoded in column chunks."""
    trellis, suh, svh, k2 = st.matrix(base)
    n = trellis.shape[1] * 16
    return np.concatenate([rows(trellis, suh, svh, k2, 2, c0, min(chunk, n - c0)) for c0 in range(0, n, chunk)])


def dense_value(st, stem, name, shape):
    """A schema tensor's float32 value from the EXL3 checkpoint (stem: the layer's prefix, or '' for globals)."""
    parts = {'q': 0, 'k': 1, 'v': 2}
    for q, i in parts.items():
        if name == f'self_attn.{q}_proj.weight' and st.has(stem + 'self_attn.qkv_proj.trellis'):
            value = decoded(st, stem + 'self_attn.qkv_proj')[i * KDA_INNER:(i + 1) * KDA_INNER]
            break
        if name == f'self_attn.{q}_conv1d.weight' and st.has(stem + 'self_attn.conv1d.weight'):
            value = st.f32(stem + 'self_attn.conv1d.weight')
            value = value.reshape(3, value.shape[0] // 3, *value.shape[1:])[i]
            break
    else:
        base = 'lm_head' if name == 'lm_head.weight' else stem + name.removesuffix('.weight')
        if st.has(base + '.trellis'):
            value = decoded(st, base)
        elif st.has(stem + name):
            value = st.f32(stem + name)
        else:
            raise ValueError(f'{stem}{name}: not in the EXL3 checkpoint')
    if list(value.shape) != shape:
        raise ValueError(f'{stem}{name}: shape {list(value.shape)} != schema {shape}')
    return value


def expert_entries(st, layer):
    """[(name, shape, quant, payload parts)] of a layer's routed experts, bytes as stored."""
    out = []
    for e in range(EXPERTS):
        for part in ('gate', 'up', 'down'):
            trellis, suh, svh, k2 = st.matrix(f'{PREFIX}layers.{layer}.mlp.experts.{e}.{part}_proj')
            shape = [trellis.shape[1] * 16, trellis.shape[0] * 16]
            out.append((f'mlp.experts.{e}.{part}_proj.weight', shape, (EXL3, k2, 2), (trellis, suh, svh)))
    return out


def write_hyt(dest, entries, digest):
    """entries: [(name, shape, quant, payload)], payload bytes or a tuple of arrays written in order."""
    table, offset, placed = b'', 0, []
    for name, shape, quant, payload in entries:
        size = storage_bytes(shape, quant)
        table += encode_entry(name, shape, quant, offset, size)
        placed.append((offset, size, payload))
        offset = align(offset + size)
    base = align(64 + len(table))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix('.hyt.partial')
    if dest.exists() or partial.exists():
        raise FileExistsError(dest)
    try:
        with partial.open('xb') as out:
            out.write(HEADER.pack(b'HYT0', FORMAT_VERSION, 0, len(entries), 64, base, digest[:16]))
            out.write(table)
            for at, size, payload in placed:
                out.seek(base + at)
                parts = payload if isinstance(payload, tuple) else (payload,)
                written = 0
                for p in parts:
                    data = p.tobytes() if isinstance(p, np.ndarray) else p
                    out.write(data)
                    written += len(data)
                if written != size:
                    raise ValueError(f'{dest}: payload {written} != {size} bytes')
            out.flush()
            os.fsync(out.fileno())
        partial.rename(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return base + offset


def convert_layer(model, schema, out, layer, digest):
    start = time.perf_counter()
    st = SafeTensors(model)
    stem = f'{PREFIX}layers.{layer}.'
    entries, inexact, k2s = [], {}, {}
    for name, quant, shape in hyt_table(Path(schema) / f'layer{layer}' / 'weights.hyt'):
        if name.startswith('mlp.experts.'):
            continue
        payload, lost = encode(dense_value(st, stem, name, shape), quant)
        if lost:
            inexact[name] = lost
        entries.append((name, shape, quant, payload))
    if layer >= 3:
        for entry in expert_entries(st, layer):
            k2s[entry[2][1]] = k2s.get(entry[2][1], 0) + 1
            entries.append(entry)
    size = write_hyt(Path(out) / f'layer{layer}' / 'weights.hyt', entries, digest)
    return {'layer': layer, 'seconds': round(time.perf_counter() - start, 1), 'bytes': size,
            'expert_k2': k2s, 'bf16_inexact': inexact}


def convert_globals(model, schema, out, digest):
    st = SafeTensors(model)
    entries, inexact = [], {}
    for name, quant, shape in hyt_table(Path(schema) / 'globals.hyt'):
        payload, lost = encode(dense_value(st, PREFIX if name != 'lm_head.weight' else '', name, shape), quant)
        if lost:
            inexact[name] = lost
        entries.append((name, shape, quant, payload))
    return {'globals': write_hyt(Path(out) / 'globals.hyt', entries, digest), 'bf16_inexact': inexact}


def source_digest(model):
    """SHA-256 over the checkpoint's config, quantization config, index and every shard (in name order)."""
    digest = hashlib.sha256()
    root = Path(model)
    for name in ['config.json', 'quantization_config.json', 'model.safetensors.index.json'] + sorted(
            p.name for p in root.glob('*.safetensors')):
        digest.update(name.encode() + b'\0')
        with (root / name).open('rb') as file:
            while raw := file.read(16 << 20):
                digest.update(raw)
    return digest.digest()


def estimate(model, schema):
    st = SafeTensors(model)
    experts = {}
    for layer in range(3, LAYERS):
        for name, shape, quant, _ in expert_entries(st, layer):
            experts[quant[1]] = experts.get(quant[1], 0) + storage_bytes(shape, quant)
    dense = 0
    for path in [Path(schema) / 'globals.hyt'] + [Path(schema) / f'layer{i}' / 'weights.hyt' for i in range(LAYERS)]:
        dense += sum(storage_bytes(s, q) for n, q, s in hyt_table(path) if not n.startswith('mlp.experts.'))
    return {'expert_bytes_by_k2': experts, 'expert_bytes': sum(experts.values()), 'dense_bytes': dense,
            'total_bytes': dense + sum(experts.values())}


def convert(model, out, schema=SCHEMA, workers=6, layers=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    digest = source_digest(model)
    log = {'model': str(model), 'schema': str(schema), 'source_sha256': digest.hex(),
           'hash_seconds': round(time.perf_counter() - start, 1)}
    print(json.dumps(log), flush=True)
    indices = list(range(LAYERS)) if layers is None else layers
    results = []
    with ProcessPoolExecutor(workers) as pool:
        jobs = [pool.submit(convert_layer, model, schema, out, i, digest) for i in indices]
        jobs.append(pool.submit(convert_globals, model, schema, out, digest))
        for job in jobs:
            results.append(job.result())
            print(json.dumps(results[-1]), flush=True)
    if (Path(schema) / 'frequencies.bin').exists():
        shutil.copyfile(Path(schema) / 'frequencies.bin', out / 'frequencies.bin')
    log.update({'results': results, 'seconds': round(time.perf_counter() - start, 1)})
    (out / 'convert.json').write_text(json.dumps(log, indent=2) + '\n')
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    fx = sub.add_parser('fixtures', help="decode exllamav3's fixture tiles and compare bit for bit")
    fx.add_argument('folder')
    cv = sub.add_parser('convert')
    cv.add_argument('--model', required=True)
    cv.add_argument('--out')
    cv.add_argument('--schema', default=str(SCHEMA))
    cv.add_argument('--workers', type=int, default=6)
    cv.add_argument('--layers', type=int, nargs='*')
    cv.add_argument('--estimate', action='store_true')
    args = parser.parse_args()
    if args.command == 'fixtures':
        results = check_fixtures(args.folder)
        print(json.dumps(results, indent=2))
        raise SystemExit(int(any(r['bad_states'] or r['bad_weights'] for r in results)))
    if args.estimate:
        print(json.dumps(estimate(args.model, args.schema), indent=2))
        return
    if not args.out:
        parser.error('convert needs --out')
    convert(args.model, args.out, args.schema, args.workers, args.layers)


if __name__ == '__main__':
    main()
