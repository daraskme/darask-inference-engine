//! GLM-5.3-Flash layer oracle and stateful, block-FP8 greedy generation.
#[cfg(feature = "cuda")]
mod gpu;
#[cfg(feature = "cuda")]
mod profile;
#[cfg(any(test, feature = "cuda"))]
mod transfer;
#[cfg(feature = "cuda")]
pub use gpu::{Layer, LayerOutput, LayerSession};
#[cfg(all(test, feature = "cuda"))]
mod gpu_tests;

use hayate_core::{Error, Result};

pub const HIDDEN: usize = 4096;
pub const HC: usize = 4;
pub const EXPERTS: usize = 288;
pub const TOP_K: usize = 8;

/// Shared by the GPU oracle test and CPU tests. Zero references require exact
/// zeros; NaN/Inf and shape mismatches never pass the quality gate.
pub fn relative_l2(actual: &[f64], reference: &[f64]) -> Result<f64> {
    if actual.is_empty() || actual.len() != reference.len() {
        return Err(Error::InvalidShape("GLM comparison length mismatch".into()));
    }
    let (mut error, mut energy) = (0., 0.);
    for (&a, &r) in actual.iter().zip(reference) {
        if !a.is_finite() || !r.is_finite() {
            return Err(Error::Weights("GLM comparison nonfinite".into()));
        }
        error += (a - r).powi(2);
        energy += r * r;
    }
    Ok(if energy == 0. {
        if error == 0. { 0. } else { f64::INFINITY }
    } else {
        (error / energy).sqrt()
    })
}

/// LOWBIT matrix (HYT tag 6): finite E4M3 group scales and a positive finite
/// FP32 tensor scale. `n` = rows x cols.
#[cfg(any(test, feature = "cuda"))]
pub(crate) fn validate_lowbit(bytes: &[u8], n: usize, grid: u8, group: u8) -> Result<()> {
    let planes = n / 8 * if matches!(grid, 2 | 3) { 3 } else { 2 };
    let scales = planes + n / group as usize;
    let tail = bytes
        .len()
        .checked_sub(4)
        .filter(|&t| t >= scales)
        .ok_or_else(|| Error::Weights("GLM LOWBIT matrix size".into()))?;
    let scale = f32::from_le_bytes(bytes[tail..].try_into().unwrap());
    if bytes[planes..scales].iter().any(|b| b & 0x7f == 0x7f) || !(scale.is_finite() && scale > 0.)
    {
        return Err(Error::Weights("GLM LOWBIT scales".into()));
    }
    Ok(())
}

/// EXL3 matrix (HYT tag 7) `[rows, cols]`: finite FP16 suh [cols] and svh
/// [rows] after the trellis (any trellis bit pattern is a valid encoding).
#[cfg(any(test, feature = "cuda"))]
pub(crate) fn validate_exl3(bytes: &[u8], rows: usize, cols: usize) -> Result<()> {
    let scales = 2 * (rows + cols);
    let tail = bytes
        .len()
        .checked_sub(scales)
        .ok_or_else(|| Error::Weights("GLM EXL3 matrix size".into()))?;
    if bytes[tail..]
        .as_chunks::<2>()
        .0
        .iter()
        .any(|h| u16::from_le_bytes(*h) & 0x7c00 == 0x7c00)
    {
        return Err(Error::Weights("GLM EXL3 suh/svh not finite".into()));
    }
    Ok(())
}

#[cfg(any(test, feature = "cuda"))]
fn validate_raw(raw: &[u8], quant: hayate_core::QuantSpec) -> Result<()> {
    use hayate_core::{DType, QuantSpec};
    let finite = match quant {
        QuantSpec::Plain(DType::F32) => {
            raw.as_chunks::<4>().1.is_empty()
                && raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|b| f32::from_le_bytes(*b).is_finite())
        }
        QuantSpec::Plain(DType::Bf16) => {
            raw.as_chunks::<2>().1.is_empty()
                && raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|b| u16::from_le_bytes(*b) & 0x7f80 != 0x7f80)
        }
        QuantSpec::Plain(DType::F8E4M3) => raw.iter().all(|b| b & 0x7f != 0x7f),
        QuantSpec::PerTensor(DType::F8E4M3) => {
            raw.len() > 4 && raw[..raw.len() - 4].iter().all(|b| b & 0x7f != 0x7f) && {
                let scale = f32::from_le_bytes(raw[raw.len() - 4..].try_into().unwrap());
                scale.is_finite() && scale > 0.
            }
        }
        _ => false,
    };
    if finite {
        Ok(())
    } else {
        Err(Error::Weights(
            "GLM nonfinite or invalid weight bytes".into(),
        ))
    }
}

#[cfg(any(test, feature = "cuda"))]
fn validate_nvfp4(raw: &[u8], n: usize, k: usize) -> Result<()> {
    let expected = hayate_core::QuantSpec::Nvfp4
        .matrix_bytes(n as u64, k as u64)
        .ok_or_else(|| Error::Weights("GLM NVFP4 shape".into()))? as usize;
    if raw.len() != expected
        || raw[n * k / 2..raw.len() - 4]
            .iter()
            .any(|b| b & 0x7f == 0x7f)
        || !f32::from_le_bytes(raw[raw.len() - 4..].try_into().unwrap()).is_finite()
        || f32::from_le_bytes(raw[raw.len() - 4..].try_into().unwrap()) <= 0.
    {
        return Err(Error::Weights("GLM invalid NVFP4 scales/layout".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weight_bytes_reject_nan_inf_and_partial_elements() {
        use hayate_core::{DType, QuantSpec};
        assert!(validate_raw(&[0x7f], QuantSpec::Plain(DType::F8E4M3)).is_err());
        assert!(validate_raw(&[0x7e], QuantSpec::Plain(DType::F8E4M3)).is_ok());
        assert!(validate_raw(&f32::INFINITY.to_le_bytes(), QuantSpec::Plain(DType::F32)).is_err());
        assert!(validate_raw(&[0x80, 0x7f], QuantSpec::Plain(DType::Bf16)).is_err());
        assert!(validate_raw(&[0], QuantSpec::Plain(DType::Bf16)).is_err());
        let mut fp8 = vec![0x38];
        fp8.extend(0.75f32.to_le_bytes());
        assert!(validate_raw(&fp8, QuantSpec::PerTensor(DType::F8E4M3)).is_ok());
        fp8[0] = 0x7f;
        assert!(validate_raw(&fp8, QuantSpec::PerTensor(DType::F8E4M3)).is_err());
        fp8[0] = 0x38;
        for scale in [0., -1., f32::INFINITY] {
            fp8[1..].copy_from_slice(&scale.to_le_bytes());
            assert!(validate_raw(&fp8, QuantSpec::PerTensor(DType::F8E4M3)).is_err());
        }
    }
    #[test]
    fn nvfp4_scales_reject_nonfinite_and_wrong_lengths() {
        let mut raw =
            vec![0; hayate_core::QuantSpec::Nvfp4.matrix_bytes(128, 32).unwrap() as usize];
        let end = raw.len();
        raw[end - 4..].copy_from_slice(&1f32.to_le_bytes());
        assert!(validate_nvfp4(&raw, 128, 32).is_ok());
        raw[2048] = 0x7f;
        assert!(validate_nvfp4(&raw, 128, 32).is_err());
        raw[2048] = 0;
        raw[end - 4..].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(validate_nvfp4(&raw, 128, 32).is_err());
        assert!(validate_nvfp4(&raw[..end - 1], 128, 32).is_err());
    }
    #[test]
    fn comparison_rejects_nonfinite_and_zero_reference_drift() {
        assert_eq!(relative_l2(&[0.], &[0.]).unwrap(), 0.);
        assert!(relative_l2(&[1.], &[0.]).unwrap().is_infinite());
        assert!(relative_l2(&[f64::NAN], &[1.]).is_err());
        assert!(relative_l2(&[1.], &[f64::INFINITY]).is_err());
        assert!(relative_l2(&[], &[]).is_err());
        assert!(relative_l2(&[1., 2.], &[1.]).is_err());
        assert!((relative_l2(&[1.01, 2.02], &[1., 2.]).unwrap() - 0.01).abs() < 1.0e-14);
    }
}

#[cfg(feature = "cuda")]
mod executor;
#[cfg(any(feature = "cuda", test))]
mod experts;
#[cfg(any(feature = "cuda", test))]
mod load;
#[cfg(any(feature = "cuda", test))]
mod prefill;
#[cfg(feature = "cuda")]
pub use executor::{Executor, MemoryUsage, Options, PrefillOutcome, Session, SessionState, run};
#[cfg(any(feature = "cuda", test))]
mod eval;
#[cfg(any(feature = "cuda", test))]
mod json;
#[cfg(any(feature = "cuda", test))]
mod tokenizer;
#[cfg(feature = "cuda")]
pub use tokenizer::Tokenizer;

#[cfg(any(feature = "cuda", test))]
mod cpu_experts;
pub mod lora;

#[cfg(any(test, feature = "cuda"))]
mod speculation;

#[cfg(any(test, feature = "cuda"))]
mod async_decode;

#[cfg(all(test, feature = "cuda"))]
mod async_tests;
