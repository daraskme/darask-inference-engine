"""Availability of the separately built, pinned quantization reference backend."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_AVAILABLE = (
    (ROOT / "upstream/gyro-llama/ggml/src/ggml-common.h").is_file()
    and (ROOT / "build/backend/bin/libggml-base.so").is_file()
)
REFERENCE_REASON = "requires the pinned reference backend; see engines/kogarashi/REFERENCE.md"
