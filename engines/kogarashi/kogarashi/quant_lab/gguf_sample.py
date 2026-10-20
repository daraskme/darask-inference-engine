"""Bounded GGUF header parsing and small float-weight samples, without mmap.

Partial files are accepted only with an explicit flag. Samples from them are
exploratory evidence, not verification of a complete or authentic model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import struct

import numpy as np


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    kind: int
    offset: int

    @property
    def elements(self):
        return math.prod(self.shape)


@dataclass(frozen=True)
class Header:
    path: Path
    file_bytes: int
    data_offset: int
    metadata: dict
    tensors: tuple[Tensor, ...]
    sha256: str


class _Reader:
    FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
               6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def __init__(self, file, size):
        self.file, self.size = file, size

    def read(self, count):
        if count < 0 or self.file.tell()+count > min(self.size, 64 << 20):
            raise ValueError("Truncated or oversized GGUF header")
        data = self.file.read(count)
        if len(data) != count:
            raise ValueError("Truncated GGUF header")
        return data

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, keep=True):
        size = self.unpack("Q")
        data = self.read(size)
        return data.decode("utf-8") if keep else None

    def value(self, kind, keep=False, depth=0):
        if depth > 1:
            raise ValueError("Nested GGUF arrays are unsupported")
        if kind in self.FORMATS:
            value = self.unpack(self.FORMATS[kind])
            return value if keep else None
        if kind == 8:
            return self.string(keep)
        if kind == 9:
            subtype, count = self.unpack("I"), self.unpack("Q")
            if count > 10_000_000 or subtype == 9:
                raise ValueError("Invalid GGUF array")
            if subtype in self.FORMATS:
                self.read(count * struct.calcsize(self.FORMATS[subtype]))
            else:
                for _ in range(count):
                    self.value(subtype, depth=depth+1)
            return None
        raise ValueError(f"Unknown GGUF metadata type {kind}")


def read_header(path: Path) -> Header:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as file:
        reader = _Reader(file, size)
        if reader.read(4) != b"GGUF" or reader.unpack("I") != 3:
            raise ValueError("Expected a little-endian GGUF v3 file")
        nt, nk = reader.unpack("Q"), reader.unpack("Q")
        if nt > 100_000 or nk > 100_000:
            raise ValueError("Excessive GGUF header counts")
        metadata = {}
        for _ in range(nk):
            name = reader.string()
            keep = name.startswith(("general.", "agention.", "prism."))
            value = reader.value(reader.unpack("I"), keep)
            if keep:
                metadata[name] = value
        tensors, names = [], set()
        for _ in range(nt):
            name, ndim = reader.string(), reader.unpack("I")
            if name in names or not 1 <= ndim <= 4:
                raise ValueError("Duplicate tensor or invalid dimensions")
            names.add(name)
            shape = tuple(reader.unpack("Q") for _ in range(ndim))
            kind, offset = reader.unpack("I"), reader.unpack("Q")
            if not all(shape):
                raise ValueError("Empty tensor")
            tensors.append(Tensor(name, shape, kind, offset))
        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or not 1 <= alignment <= 4096 or alignment & (alignment-1):
            raise ValueError("Invalid GGUF alignment")
        data_offset = (file.tell()+alignment-1) // alignment * alignment
        reader.read(data_offset-file.tell())
        file.seek(0)
        digest = hashlib.sha256(file.read(data_offset)).hexdigest()
    return Header(path, size, data_offset, metadata, tuple(tensors), digest)


def samples(header: Header, pattern: str, per_tensor=8, *, allow_partial=False):
    if per_tensor <= 0:
        raise ValueError("per_tensor must be positive")
    if not allow_partial:
        # Exact end-of-file validation is possible only for supported float types.
        sizes = {0: 4, 1: 2, 30: 2}
        if any(t.kind not in sizes for t in header.tensors):
            raise ValueError("Sampling requires an original F32/F16/BF16 GGUF, not a requantized model")
        expected_end = max((header.data_offset+t.offset+t.elements*sizes[t.kind]
                            for t in header.tensors), default=header.data_offset)
        if header.file_bytes < expected_end:
            raise ValueError("Incomplete model; use --allow-partial for exploratory samples")
    matched = [t for t in header.tensors if re.fullmatch(pattern, t.name)]
    if not matched:
        raise ValueError("No tensor names matched")
    with header.path.open("rb") as file:
        for tensor in matched:
            if tensor.kind not in (0, 1, 30) or tensor.shape[0] % 128:
                raise ValueError(f"Expected aligned original float weights: {tensor.name}")
            itemsize = 4 if tensor.kind == 0 else 2
            start = header.data_offset+tensor.offset
            available = min(tensor.elements, max(0, (header.file_bytes-start)//itemsize)) // 128
            if not available:
                if allow_partial:
                    continue
                raise ValueError(f"Unavailable tensor: {tensor.name}")
            indices = np.linspace(0, available-1, min(per_tensor, available), dtype=np.int64)
            for index in indices:
                byte_offset = start + int(index)*128*itemsize
                file.seek(byte_offset)
                data = file.read(128*itemsize)
                if len(data) != 128*itemsize:
                    raise ValueError("File changed or truncated during sampling")
                if tensor.kind == 30:
                    values = (np.frombuffer(data, dtype="<u2").astype("<u4") << 16).view("<f4")
                else:
                    values = np.frombuffer(data, dtype="<f4" if tensor.kind == 0 else "<f2").astype(np.float32)
                yield {"tensor": tensor.name, "shape": tensor.shape, "kind": tensor.kind,
                       "block_index": int(index), "byte_offset": byte_offset,
                       "source_sha256": hashlib.sha256(data).hexdigest()}, values.copy()
