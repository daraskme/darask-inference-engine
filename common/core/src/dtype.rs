//! Element and storage formats.
//!
//! `DType` names the packed element payload only. How payload and scales are
//! laid out in memory is a separate [`QuantSpec`], so the converter, loader,
//! kernels and capacity planner all derive sizes from one definition.

/// Element types Hayate stores in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    Bf16,
    F16,
    /// FP8 E4M3 with a per-tensor (or per-channel) scale — "FP8" in the docs.
    F8E4M3,
    F8E5M2,
    /// FP4 E2M1 payload of NVFP4 (block scales described by `QuantSpec`).
    F4E2M1,
    I8,
    I32,
    /// Signed 4-bit integers stored with a +8 offset (CPU expert path).
    I4,
}

impl DType {
    pub const fn bits(self) -> u64 {
        match self {
            DType::F32 | DType::I32 => 32,
            DType::Bf16 | DType::F16 => 16,
            DType::F8E4M3 | DType::F8E5M2 | DType::I8 => 8,
            DType::F4E2M1 | DType::I4 => 4,
        }
    }

    /// Bytes for `n` packed elements, rounded up. `None` on overflow.
    pub const fn payload_bytes(self, n: u64) -> Option<u64> {
        match n.checked_mul(self.bits()) {
            Some(bits) => Some(bits.div_ceil(8)),
            None => None,
        }
    }
}

/// How one tensor's payload and scales are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantSpec {
    /// Unquantized payload.
    Plain(DType),
    /// One FP32 scale for the whole tensor (FP8 E4M3 / INT8 per-tensor).
    PerTensor(DType),
    /// NVFP4: E2M1 payload, one E4M3 scale per 16 elements along K, plus a
    /// per-tensor FP32 scale. Scales use the SM120 128×4 tile layout.
    Nvfp4,
    /// MXFP8: E4M3 payload, one E8M0 scale per 32 elements along K, same
    /// 128×4 scale tile layout.
    Mxfp8,
    /// CPU expert format: I4 payload, one BF16 scale per `group` elements of
    /// a row, densely packed (no tile padding).
    Q4Bf16 { group: u32 },
    /// INT8 payload with one FP32 scale per row (output channel), stored
    /// after the payload. `convrot_group` > 0 means activations are rotated in
    /// groups of that many elements (regular Hadamard, ConvRot) before use;
    /// 0 means no rotation. Used as distributed by H3 INT8 checkpoints.
    Int8Row { convrot_group: u32 },
    /// GLM routed-expert LOWBIT (docs/GLM53_QUANT.md §3): 2- or 3-bit codes of
    /// one of five grids (`grid` 0..=4: u2, int2, int3, nf3, n2), one E4M3
    /// scale per `group` (16 or 32) elements, bit planes then scales then one
    /// FP32 tensor scale, no tile padding. `cols` must be a multiple of 256.
    Lowbit { grid: u8, group: u8 },
    /// GLM routed-expert EXL3 (exllamav3 trellis, crates/models/src/glm53/EXL3.md):
    /// a `[rows = outputs, cols = inputs]` matrix stored as its [cols/16][rows/16]
    /// tiles of 16·`k2` bytes (`k2` = twice the bits per weight), then FP16 suh
    /// [cols] and FP16 svh [rows]. `cb` is the codebook (0 3inst, 1 mcg, 2 mul1).
    /// Both dimensions are multiples of 128 (the 128-point rotation blocks).
    Exl3 { k2: u8, cb: u8 },
}

impl QuantSpec {
    /// exllamav3 1.6 bitrates: K 1..8 with any codebook, K 1.5/2.5/3.5 with mul1.
    pub const fn exl3_valid(k2: u8, cb: u8) -> bool {
        (matches!(k2, 2..=16) && k2.is_multiple_of(2) && cb <= 2)
            || (matches!(k2, 3 | 5 | 7) && cb == 2)
    }
}

/// SM120 block-scaled MMA reads scale factors in tiles of 128 rows × 4 scale
/// columns (CUTLASS "128x4" layout), so scale storage is padded to that tile.
const SCALE_TILE_ROWS: u64 = 128;
const SCALE_TILE_COLS: u64 = 4;

impl QuantSpec {
    /// Exact storage bytes for a row-major `[rows, cols]` matrix (K = cols),
    /// including scales and their tile padding. `None` on overflow or if
    /// `cols` is not a multiple of the block size.
    pub const fn matrix_bytes(self, rows: u64, cols: u64) -> Option<u64> {
        let Some(n) = rows.checked_mul(cols) else {
            return None;
        };
        match self {
            QuantSpec::Plain(d) => d.payload_bytes(n),
            QuantSpec::PerTensor(d) => match d.payload_bytes(n) {
                Some(b) => b.checked_add(4),
                None => None,
            },
            QuantSpec::Nvfp4 => block_scaled(n, rows, cols, DType::F4E2M1, 16, 1, 4, true),
            QuantSpec::Mxfp8 => block_scaled(n, rows, cols, DType::F8E4M3, 32, 1, 0, true),
            QuantSpec::Q4Bf16 { group } => {
                block_scaled(n, rows, cols, DType::I4, group as u64, 2, 0, false)
            }
            QuantSpec::Int8Row { .. } => match rows.checked_mul(4) {
                Some(s) => n.checked_add(s),
                None => None,
            },
            QuantSpec::Lowbit { grid, group } => {
                if group == 0 || !cols.is_multiple_of(256) {
                    return None;
                }
                let bits = if grid == 2 || grid == 3 { 3 } else { 2 };
                match (n / 8).checked_mul(bits) {
                    Some(p) => p.checked_add(n / group as u64 + 4),
                    None => None,
                }
            }
            QuantSpec::Exl3 { k2, cb } => {
                if !QuantSpec::exl3_valid(k2, cb)
                    || !rows.is_multiple_of(128)
                    || !cols.is_multiple_of(128)
                {
                    return None;
                }
                let Some(trellis) = (n / 16).checked_mul(k2 as u64) else {
                    return None;
                };
                match rows.checked_add(cols) {
                    Some(s) => trellis.checked_add(2 * s),
                    None => None,
                }
            }
        }
    }

    /// Bits per element including scales (ignoring tile padding), for
    /// capacity estimates in docs.
    pub fn effective_bits(self) -> f64 {
        match self {
            QuantSpec::Plain(d) => d.bits() as f64,
            QuantSpec::PerTensor(d) => d.bits() as f64,
            QuantSpec::Nvfp4 => 4.0 + 8.0 / 16.0,
            QuantSpec::Mxfp8 => 8.0 + 8.0 / 32.0,
            QuantSpec::Q4Bf16 { group } => 4.0 + 16.0 / group as f64,
            QuantSpec::Int8Row { .. } => 8.0,
            QuantSpec::Lowbit { grid, group } => {
                (if grid == 2 || grid == 3 { 3.0 } else { 2.0 }) + 8.0 / group as f64
            }
            QuantSpec::Exl3 { k2, .. } => k2 as f64 / 2.0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
const fn block_scaled(
    n: u64,
    rows: u64,
    cols: u64,
    payload: DType,
    block: u64,
    scale_bytes: u64,
    tensor_scale_bytes: u64,
    tile_padded: bool,
) -> Option<u64> {
    if block == 0 || !cols.is_multiple_of(block) {
        return None;
    }
    let Some(p) = payload.payload_bytes(n) else {
        return None;
    };
    let (scale_rows, scale_cols) = if tile_padded {
        match (
            rows.checked_next_multiple_of(SCALE_TILE_ROWS),
            (cols / block).checked_next_multiple_of(SCALE_TILE_COLS),
        ) {
            (Some(r), Some(c)) => (r, c),
            _ => return None,
        }
    } else {
        (rows, cols / block)
    };
    let Some(scales) = scale_rows.checked_mul(scale_cols) else {
        return None;
    };
    let Some(s) = scales.checked_mul(scale_bytes) else {
        return None;
    };
    match p.checked_add(s) {
        Some(t) => t.checked_add(tensor_scale_bytes),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_tile_aligned_shape() {
        // 4096 x 4096: 8 MiB payload + 1 MiB of E4M3 scales + FP32 tensor scale.
        let b = QuantSpec::Nvfp4.matrix_bytes(4096, 4096).unwrap();
        assert_eq!(b, 4096 * 4096 / 2 + 4096 * 4096 / 16 + 4);
    }

    #[test]
    fn nvfp4_scales_are_tile_padded() {
        // Values cross-checked in review round 2 against the CUTLASS 128x4 layout.
        assert_eq!(QuantSpec::Nvfp4.matrix_bytes(96, 4096), Some(229_380));
        assert_eq!(QuantSpec::Nvfp4.matrix_bytes(1, 4096), Some(34_820));
        // Gemma 4 26B-A4B expert shape.
        assert_eq!(QuantSpec::Nvfp4.matrix_bytes(704, 2816), Some(1_126_404));
    }

    #[test]
    fn q4_bf16_group32_is_dense() {
        let b = QuantSpec::Q4Bf16 { group: 32 }.matrix_bytes(2, 64).unwrap();
        assert_eq!(b, 64 + 2 * 2 * 2);
        assert_eq!(QuantSpec::Q4Bf16 { group: 32 }.effective_bits(), 4.5);
    }

    #[test]
    fn lowbit_sizes_match_the_spec_tables() {
        // docs/GLM53_QUANT.md §2: gate+up / down bytes with 256-byte alignment.
        let stride = |q: QuantSpec, r, c| q.matrix_bytes(r, c).unwrap().next_multiple_of(256);
        let nf3 = QuantSpec::Lowbit { grid: 3, group: 32 };
        let n2 = QuantSpec::Lowbit { grid: 4, group: 32 };
        assert_eq!(2 * stride(nf3, 2048, 4096), 6_816_256);
        assert_eq!(stride(nf3, 4096, 2048), 3_408_128);
        assert_eq!(2 * stride(n2, 2048, 4096), 4_719_104);
        assert_eq!(stride(n2, 4096, 2048), 2_359_552);
        assert_eq!(nf3.matrix_bytes(2048, 4000), None);
        assert_eq!(nf3.effective_bits(), 3.25);
        assert_eq!(n2.effective_bits(), 2.25);
    }

    #[test]
    fn exl3_sizes_match_the_converter() {
        // turboderp sc_2.60bpw: K 2.5 (k2 5) experts, layer 3 at K 3.5 (k2 7).
        let gate = |k2| {
            QuantSpec::Exl3 { k2, cb: 2 }
                .matrix_bytes(2048, 4096)
                .unwrap()
        };
        let down = |k2| {
            QuantSpec::Exl3 { k2, cb: 2 }
                .matrix_bytes(4096, 2048)
                .unwrap()
        };
        assert_eq!(2 * gate(5) + down(5), 7_901_184);
        assert_eq!(2 * gate(7) + down(7), 11_046_912);
        assert!([gate(5), down(5), gate(7)].iter().all(|b| b % 256 == 0));
        assert_eq!(
            QuantSpec::Exl3 { k2: 5, cb: 1 }.matrix_bytes(2048, 4096),
            None
        );
        assert_eq!(
            QuantSpec::Exl3 { k2: 6, cb: 0 }.matrix_bytes(2048, 4096),
            Some(2048 * 4096 * 6 / 16 + 2 * 6144)
        );
        assert_eq!(
            QuantSpec::Exl3 { k2: 5, cb: 2 }.matrix_bytes(2048, 4000),
            None
        );
        assert_eq!(
            QuantSpec::Exl3 { k2: 9, cb: 2 }.matrix_bytes(2048, 4096),
            None
        );
        assert_eq!(QuantSpec::Exl3 { k2: 5, cb: 2 }.effective_bits(), 2.5);
    }

    #[test]
    fn int8_row_has_one_f32_scale_per_row() {
        let q = QuantSpec::Int8Row { convrot_group: 256 };
        assert_eq!(q.matrix_bytes(21504, 5376), Some(21504 * 5376 + 21504 * 4));
    }

    #[test]
    fn overflow_is_rejected() {
        assert_eq!(DType::F32.payload_bytes(1 << 59), None);
        assert_eq!(QuantSpec::Nvfp4.matrix_bytes(u64::MAX, 16), None);
    }

    #[test]
    fn ragged_blocks_are_rejected() {
        assert_eq!(QuantSpec::Nvfp4.matrix_bytes(1, 17), None);
        assert_eq!(QuantSpec::Q4Bf16 { group: 0 }.matrix_bytes(1, 32), None);
    }
}
