//! Model shapes. Mirrors `kernels/include/q38.h`; the test below keeps the two in sync.
//! Source: Qwen/Qwen3.8-Flash-Next config.json @ de4b8e4 and transformers `qwen4_exp`.

pub const H: usize = 2560;
pub const HC: usize = 4;
pub const HCW: usize = H * HC;
pub const GR_RANK: usize = 320;
pub const L: usize = 48;
pub const VOCAB: usize = 248_320;

pub const E: usize = 512;
pub const TOPK: usize = 10;
pub const FF: usize = 640;

pub const GDN_QK: usize = 16;
pub const GDN_V: usize = 48;
pub const GDN_D: usize = 128;
pub const GDN_QKV: usize = 2 * GDN_QK * GDN_D + GDN_V * GDN_D;
pub const GDN_CONV: usize = 4;

pub const ATT_Q: usize = 24;
pub const ATT_KV: usize = 2;
pub const ATT_D: usize = 256;
pub const ROPE_D: usize = 64;
pub const ROPE_THETA: f64 = 1e7;
pub const IDX_H: usize = 4;
pub const IDX_D: usize = 128;
pub const IDX_R: usize = 4;
pub const IDX_TOPB: usize = 512;
pub const QSA_MAX_KEYS: usize = IDX_TOPB * IDX_R + IDX_R - 1;

pub const PLE_LAYER: usize = 1;
pub const PLE_HEADS: usize = 16;
pub const PLE_ROW: usize = 160;
pub const PLE_CONV: usize = 4;
pub const PLE_DILATION: usize = 3;

pub const EOS: u32 = 248_044;
pub const RMS_EPS: f32 = 1e-6;

/// Layers 3, 7, ..., 47 are QSA (indexed sparse attention); the rest are Gated DeltaNet.
pub const fn is_qsa(layer: usize) -> bool {
    (layer + 1).is_multiple_of(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `constexpr int NAME = <integer literal>;` in q38.h must equal the Rust constant.
    #[test]
    fn matches_cuda_header() {
        let header = include_str!("../../../kernels/include/q38.h");
        let rust: &[(&str, i64)] = &[
            ("H", H as i64),
            ("HC", HC as i64),
            ("GR_RANK", GR_RANK as i64),
            ("L", L as i64),
            ("VOCAB", VOCAB as i64),
            ("E", E as i64),
            ("TOPK", TOPK as i64),
            ("FF", FF as i64),
            ("GDN_QK", GDN_QK as i64),
            ("GDN_V", GDN_V as i64),
            ("GDN_D", GDN_D as i64),
            ("GDN_CONV", GDN_CONV as i64),
            ("ATT_Q", ATT_Q as i64),
            ("ATT_KV", ATT_KV as i64),
            ("ATT_D", ATT_D as i64),
            ("ROPE_D", ROPE_D as i64),
            ("IDX_H", IDX_H as i64),
            ("IDX_D", IDX_D as i64),
            ("IDX_R", IDX_R as i64),
            ("IDX_TOPB", IDX_TOPB as i64),
            ("PLE_LAYER", PLE_LAYER as i64),
            ("PLE_HEADS", PLE_HEADS as i64),
            ("PLE_ROW", PLE_ROW as i64),
            ("PLE_CONV", PLE_CONV as i64),
            ("PLE_DILATION", PLE_DILATION as i64),
            ("EOS", i64::from(EOS)),
        ];
        let mut seen = 0;
        for line in header.lines() {
            let Some(rest) = line.trim().strip_prefix("constexpr int ") else {
                continue;
            };
            let (name, value) = rest.split_once('=').unwrap();
            let value = value.split(';').next().unwrap().trim();
            let Ok(value) = value.parse::<i64>() else {
                continue;
            };
            let name = name.trim();
            let want = rust
                .iter()
                .find(|(n, _)| *n == name)
                .unwrap_or_else(|| panic!("{name} missing in shape.rs"))
                .1;
            assert_eq!(value, want, "{name}");
            seen += 1;
        }
        assert_eq!(seen, rust.len());
        assert_eq!(GDN_QKV, 10_240);
        assert_eq!(QSA_MAX_KEYS, 2051);
        assert_eq!(PLE_HEADS * PLE_ROW, H);
        assert!(!is_qsa(PLE_LAYER));
        assert_eq!((0..L).filter(|&l| is_qsa(l)).count(), 12);
    }
}
