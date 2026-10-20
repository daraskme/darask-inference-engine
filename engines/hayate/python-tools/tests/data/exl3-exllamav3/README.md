# EXL3 decode fixtures

exllamav3's own decode of random trellis tiles (states and W' per bitrate and codebook), written by
Project Maya's `tools/exl3_exllamav3_fixture.py` against exllamav3 1.6.2 (copied from Project Maya PR #265,
`data/exl3-exllamav3`). Format: `"EXH1", int32 k2, ki, nj, cb, words`, then int16 trellis [ki][nj][words],
uint16 states [ki][nj][256], FP16 W' [16 ki][16 nj].

exllamav3: MIT License, Copyright (c) 2025 Turboderp. Project Maya: MIT License, Copyright (c) 2026 Peasant Smith
and the Project Maya contributors.
