//! Direct NVFP4 / LOWBIT / source block-FP8 GEMV for GLM host experts.
//!
//! Weight formats are pluggable through [`Format`]: each format provides a
//! row kernel that returns the F32 dot products of one weight row with up to
//! four activation rows. Workers claim contiguous chunks of the concatenated
//! matrix rows, so each core streams sequential memory and a whole layer of
//! misses costs two dispatches.
use hayate_core::{Error, QuantSpec, Result};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

/// Expert weight storage formats the CPU can read in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// E2M1 nibbles, E4M3 scale per 16 columns (128x4 swizzled tiles), F32 global.
    Nvfp4,
    /// E4M3 bytes with an F32 scale per 128x128 block (source checkpoint).
    Fp8Block,
    /// HYT tag 6: u2/int2/int3/nf3/n2 (grid 0..4), group 16 or 32.
    Lowbit { grid: u8, group: usize },
    /// HYT tag 7: EXL3 trellis (`super::exl3`), k2 = twice the bits per
    /// weight, codebook mul1. Segments carry each matrix's rotated input.
    Exl3 { k2: u8 },
}
impl Format {
    /// Unpadded matrix payload size. LOWBIT columns must be divisible by 256.
    pub fn matrix_bytes(self, n: usize, k: usize) -> Option<usize> {
        if n == 0 || k == 0 || !k.is_multiple_of(16) {
            return None;
        }
        let elements = n.checked_mul(k)?;
        match self {
            Self::Nvfp4 => QuantSpec::Nvfp4
                .matrix_bytes(n as u64, k as u64)
                .and_then(|b| usize::try_from(b).ok()),
            Self::Fp8Block => elements.checked_add(
                n.div_ceil(128)
                    .checked_mul(k.div_ceil(128))?
                    .checked_mul(4)?,
            ),
            Self::Lowbit { grid, group }
                if grid <= 4 && [16, 32].contains(&group) && k.is_multiple_of(256) =>
            {
                (elements / 8)
                    .checked_mul(if [2, 3].contains(&grid) { 3 } else { 2 })?
                    .checked_add(elements / group)?
                    .checked_add(4)
            }
            Self::Lowbit { .. } => None,
            Self::Exl3 { k2 } => QuantSpec::Exl3 { k2, cb: 2 }
                .matrix_bytes(n as u64, k as u64)
                .and_then(|b| usize::try_from(b).ok()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Matrix<'a> {
    raw: &'a [u8],
    n: usize,
    k: usize,
    format: Format,
    gpu_order: bool,
}
impl<'a> Matrix<'a> {
    pub fn new(raw: &'a [u8], n: usize, k: usize, nv: bool) -> Result<Self> {
        Self::with_format(raw, n, k, if nv { Format::Nvfp4 } else { Format::Fp8Block })
    }
    pub fn with_format(raw: &'a [u8], n: usize, k: usize, format: Format) -> Result<Self> {
        if format.matrix_bytes(n, k) != Some(raw.len()) {
            return Err(Error::InvalidShape("GLM CPU matrix layout".into()));
        }
        Ok(Self {
            raw,
            n,
            k,
            format,
            gpu_order: false,
        })
    }
    /// Reproduce decode dense.cuh's NVFP4 lane/FMA/reduction order.
    /// Other formats retain their usual kernel. Disabled by default.
    pub fn with_gpu_order(mut self, enabled: bool) -> Self {
        self.gpu_order = enabled && self.nv();
        self
    }
    pub fn rows(&self) -> usize {
        self.n
    }
    pub fn cols(&self) -> usize {
        self.k
    }
    fn nv(&self) -> bool {
        self.format == Format::Nvfp4
    }
    /// The fast NVFP4 kernel reads 64 columns (32 bytes, 4 scales) per step.
    fn fast(&self) -> bool {
        self.nv() && !self.gpu_order && self.k.is_multiple_of(64)
    }
    #[inline(always)]
    fn scale(self, row: usize, c: usize) -> f32 {
        if self.nv() {
            let cols = (self.k / 16).next_multiple_of(4);
            let col = c / 16;
            let at = ((row / 128) * (cols / 4) + col / 4) * 512
                + row % 32 * 16
                + (row % 128 / 32) * 4
                + col % 4;
            let base = self.n * self.k / 2;
            fp8(self.raw[base + at])
                * f32::from_le_bytes(self.raw[self.raw.len() - 4..].try_into().unwrap())
        } else {
            let at = self.n * self.k + ((row / 128) * self.k.div_ceil(128) + c / 128) * 4;
            f32::from_le_bytes(self.raw[at..at + 4].try_into().unwrap())
        }
    }
}
pub fn bf16(x: f32) -> f32 {
    f32::from_bits(x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) & 0xffff0000)
}
const fn fp8(b: u8) -> f32 {
    let (e, m) = ((b >> 3) & 15, b & 7);
    let v = if e == 0 {
        m as f32 / 512.
    } else {
        f32::from_bits(((e as u32 + 120) << 23) | ((m as u32) << 20))
    };
    if b & 128 != 0 { -v } else { v }
}
static FP8: [f32; 256] = {
    let mut table = [0.; 256];
    let mut b = 0;
    while b < 256 {
        table[b] = fp8(b as u8);
        b += 1;
    }
    table
};

/// GLM expert SwiGLU with the clamps and BF16 roundings of the GPU kernel.
pub fn swiglu(gate: f32, up: f32) -> f32 {
    let gate = gate.min(10.);
    bf16(bf16(gate / (1. + (-gate).exp())) * up.clamp(-10., 10.))
}
fn swiglu_gpu_order(gate: f32, up: f32) -> f32 {
    let gate = gate.min(10.);
    // CUDA uses gate*sigmoid(gate), rather than gate/(1+exp(-gate)).
    // Its fast-math exp/reciprocal still need the ignored GPU comparison.
    bf16(bf16(gate * (1. / (1. + (-gate).exp()))) * up.clamp(-10., 10.))
}

/// Column order of the fast NVFP4 kernel: per 16-column block, the eight even
/// columns then the eight odd ones, matching the low/high nibbles of a byte.
pub fn deinterleave(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    for (src, dst) in x
        .as_chunks::<16>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<16>().0)
    {
        for i in 0..8 {
            dst[i] = src[2 * i];
            dst[8 + i] = src[2 * i + 1];
        }
    }
}

/// Activations in CUDA lane order: eight lanes, each owning eight columns.
/// Transpose once per input, so exact-order GEMV needs no per-row gathers.
pub fn deinterleave_gpu(x: &[f32], out: &mut [f32], k: usize) {
    debug_assert_eq!(x.len(), out.len());
    debug_assert!(k.is_multiple_of(16));
    for (x, out) in x.chunks_exact(k).zip(out.chunks_exact_mut(k)) {
        for (src, dst) in x.chunks(64).zip(out.chunks_mut(64)) {
            let lanes = src.len() / 8;
            for j in 0..8 {
                for lane in 0..lanes {
                    dst[j * lanes + lane] = src[lane * 8 + j];
                }
            }
        }
    }
}

// Both supported hosts have AVX2/FMA. NVFP4 is nonlinear E2M1, so retain
// exact F32 activations rather than introduce an int8/VNNI quantization error.
// Block sums are scaled by the E4M3 block scale; the F32 global multiplier
// (and the 0.5 of the doubled E2M1 table) is applied once per row.
#[target_feature(enable = "avx2,fma")]
unsafe fn nvfp4_row<const M: usize>(w: Matrix<'_>, row: usize, xd: *const f32) -> [f32; M] {
    use std::arch::x86_64::*;
    unsafe {
        let k = w.k;
        let lut = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12, 0, 1, 2, 3, 4, 6, 8, 12, 0,
            -1, -2, -3, -4, -6, -8, -12,
        );
        let mask = _mm256_set1_epi8(15);
        let table = FP8.as_ptr();
        let weights = w.raw.as_ptr().add(row * k / 2);
        let cols = (k / 16).next_multiple_of(4);
        let scales = w.raw.as_ptr().add(
            w.n * k / 2 + (row / 128) * (cols / 4) * 512 + row % 32 * 16 + (row % 128 / 32) * 4,
        );
        let mut acc = [[_mm256_setzero_ps(); 2]; M];
        for j in 0..k / 64 {
            let at = weights.add(j * 32);
            _mm_prefetch::<_MM_HINT_T0>(at.wrapping_add(2048).cast());
            let packed = _mm256_loadu_si256(at.cast());
            let even = _mm256_shuffle_epi8(lut, _mm256_and_si256(packed, mask));
            let odd =
                _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(packed), mask));
            let even = [
                _mm256_castsi256_si128(even),
                _mm256_extracti128_si256::<1>(even),
            ];
            let odd = [
                _mm256_castsi256_si128(odd),
                _mm256_extracti128_si256::<1>(odd),
            ];
            let s = scales.add(j * 512);
            for b in 0..4 {
                let (e, o) = if b % 2 == 0 {
                    (even[b / 2], odd[b / 2])
                } else {
                    (
                        _mm_srli_si128::<8>(even[b / 2]),
                        _mm_srli_si128::<8>(odd[b / 2]),
                    )
                };
                let we = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(e));
                let wo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(o));
                let scale = _mm256_broadcast_ss(&*table.add(*s.add(b) as usize));
                for (m, sums) in acc.iter_mut().enumerate() {
                    let x = xd.add(m * k + j * 64 + b * 16);
                    let t = _mm256_mul_ps(we, _mm256_loadu_ps(x));
                    let t = _mm256_fmadd_ps(wo, _mm256_loadu_ps(x.add(8)), t);
                    sums[b % 2] = _mm256_fmadd_ps(scale, t, sums[b % 2]);
                }
            }
        }
        let global = f32::from_le_bytes(w.raw[w.raw.len() - 4..].try_into().unwrap()) * 0.5;
        let mut out = [0.; M];
        for (m, sums) in acc.iter().enumerate() {
            let v = _mm256_add_ps(sums[0], sums[1]);
            let h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
            let h = _mm_hadd_ps(h, h);
            out[m] = _mm_cvtss_f32(_mm_hadd_ps(h, h)) * global;
        }
        out
    }
}

/// Generic path: block-FP8 rows and NVFP4 rows whose width is not a multiple
/// of 64. Natural column order.
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot_batch<const M: usize>(w: Matrix<'_>, row: usize, x: *const f32) -> [f32; M] {
    use std::arch::x86_64::*;
    unsafe {
        let lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
        let mask = _mm_set1_epi8(15);
        let mut acc = [[_mm256_setzero_ps(); 2]; M];
        for c in (0..w.k).step_by(16) {
            let scale = w.scale(row, c);
            let values = if w.nv() {
                let packed = _mm_loadl_epi64(w.raw.as_ptr().add((row * w.k + c) / 2).cast());
                let lo = _mm_and_si128(packed, mask);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
                let q = _mm_shuffle_epi8(lut, _mm_unpacklo_epi8(lo, hi));
                [
                    _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)),
                        _mm256_set1_ps(scale * 0.5),
                    ),
                    _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(q))),
                        _mm256_set1_ps(scale * 0.5),
                    ),
                ]
            } else {
                let mut v = [0.; 16];
                for (j, out) in v.iter_mut().enumerate() {
                    *out = bf16(fp8(w.raw[row * w.k + c + j]) * scale);
                }
                [
                    _mm256_loadu_ps(v.as_ptr()),
                    _mm256_loadu_ps(v.as_ptr().add(8)),
                ]
            };
            for (t, sums) in acc.iter_mut().enumerate() {
                for i in 0..2 {
                    sums[i] = _mm256_fmadd_ps(
                        values[i],
                        _mm256_loadu_ps(x.add(t * w.k + c + i * 8)),
                        sums[i],
                    );
                }
            }
        }
        let mut out = [0.; M];
        for (t, sums) in acc.iter().enumerate() {
            let v = _mm256_add_ps(sums[0], sums[1]);
            let h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
            let h = _mm_hadd_ps(h, h);
            out[t] = _mm_cvtss_f32(_mm_hadd_ps(h, h));
        }
        out
    }
}

/// One row for M activation rows; `x` natural order, `xd` deinterleaved.
#[inline(always)]
unsafe fn row<const M: usize>(w: Matrix<'_>, r: usize, x: *const f32, xd: *const f32) -> [f32; M] {
    unsafe {
        if w.gpu_order {
            nvfp4_gpu_row::<M>(w, r, xd)
        } else if let Format::Lowbit { grid, group } = w.format {
            match (matches!(grid, 2 | 3), group) {
                (false, 16) => lowbit_row::<M, false, 16>(w, r, x, grid),
                (false, 32) => lowbit_row::<M, false, 32>(w, r, x, grid),
                (true, 16) => lowbit_row::<M, true, 16>(w, r, x, grid),
                (true, 32) => lowbit_row::<M, true, 32>(w, r, x, grid),
                _ => unreachable!(),
            }
        } else if w.fast() {
            nvfp4_row::<M>(w, r, xd)
        } else {
            row_dot_batch::<M>(w, r, x)
        }
    }
}

// Vector elements are CUDA lanes, not adjacent columns. Each lane performs
// eight FMAs, advances 256 columns, then takes part in the 16/8/4/2/1 tree.
// Scale*global and E2M1*multiplier round before the FMA, as in dense.cuh.
#[target_feature(enable = "avx2,fma")]
unsafe fn nvfp4_gpu_row<const M: usize>(w: Matrix<'_>, row: usize, xd: *const f32) -> [f32; M] {
    use std::arch::x86_64::*;
    unsafe {
        let lut = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3,
            4, 6, 8, 12,
        );
        let indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let global = f32::from_le_bytes(w.raw[w.raw.len() - 4..].try_into().unwrap());
        let cols = (w.k / 16).next_multiple_of(4);
        let base =
            w.n * w.k / 2 + row / 128 * (cols / 4) * 512 + row % 32 * 16 + (row % 128 / 32) * 4;
        let mut acc = [[_mm256_setzero_ps(); M]; 4];
        for c in (0..w.k).step_by(256) {
            for (g, sums) in acc.iter_mut().enumerate() {
                let at = c + g * 64;
                if at >= w.k {
                    break;
                }
                let mut packed = [0i32; 8];
                let mut scales = [0f32; 8];
                for lane in 0..8 {
                    let col = at + lane * 8;
                    if col < w.k {
                        packed[lane] = i32::from_le_bytes(
                            w.raw[(row * w.k + col) / 2..(row * w.k + col) / 2 + 4]
                                .try_into()
                                .unwrap(),
                        );
                        scales[lane] =
                            FP8[w.raw[base + col / 64 * 512 + col / 16 % 4] as usize] * global;
                    }
                }
                let packed = _mm256_loadu_si256(packed.as_ptr().cast());
                let multiplier = _mm256_loadu_ps(scales.as_ptr());
                let lanes = ((w.k - at) / 8).min(8);
                macro_rules! column {
                    ($j:expr) => {{
                        let code = _mm256_and_si256(
                            _mm256_srli_epi32::<{ 4 * $j }>(packed),
                            _mm256_set1_epi32(15),
                        );
                        let bytes = _mm256_shuffle_epi8(lut, code);
                        let value = _mm256_mul_ps(
                            _mm256_cvtepi32_ps(_mm256_and_si256(bytes, _mm256_set1_epi32(255))),
                            _mm256_set1_ps(0.5),
                        );
                        let sign = _mm256_and_si256(
                            _mm256_slli_epi32::<28>(code),
                            _mm256_set1_epi32(i32::MIN),
                        );
                        let weight = _mm256_mul_ps(
                            _mm256_xor_ps(value, _mm256_castsi256_ps(sign)),
                            multiplier,
                        );
                        for (m, sum) in sums.iter_mut().enumerate() {
                            let mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(lanes as i32), indices);
                            let a = _mm256_maskload_ps(xd.add(m * w.k + at + $j * lanes), mask);
                            *sum = _mm256_fmadd_ps(weight, a, *sum);
                        }
                    }};
                }
                column!(0);
                column!(1);
                column!(2);
                column!(3);
                column!(4);
                column!(5);
                column!(6);
                column!(7);
            }
        }
        std::array::from_fn(|m| {
            let v = _mm256_add_ps(
                _mm256_add_ps(acc[0][m], acc[2][m]),
                _mm256_add_ps(acc[1][m], acc[3][m]),
            );
            let h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
            let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps::<0x55>(h, h)))
        })
    }
}

const LOWBIT_LEVELS: [[i8; 8]; 5] = [
    [-3, -1, 1, 3, 0, 0, 0, 0],
    [-2, -1, 0, 1, 0, 0, 0, 0],
    [-4, -3, -2, -1, 0, 1, 2, 3],
    [-8, -6, -3, -1, 1, 3, 6, 8],
    [-4, -1, 1, 4, 0, 0, 0, 0],
];

// Each 32-byte load exposes 128 low-plane columns or 256 high-plane
// columns. Keep F32 activations: no Q8 rounding is introduced here.
#[target_feature(enable = "avx2,fma")]
unsafe fn lowbit_row<const M: usize, const THREE: bool, const GROUP: usize>(
    w: Matrix<'_>,
    row: usize,
    x: *const f32,
    grid: u8,
) -> [f32; M] {
    use std::arch::x86_64::*;
    unsafe {
        let levels = LOWBIT_LEVELS[grid as usize];
        let lut = _mm256_broadcastsi128_si256(_mm_loadl_epi64(levels.as_ptr().cast()));
        let low = w.raw.as_ptr().add(row * w.k / 4);
        let high = if THREE {
            w.raw.as_ptr().add(w.n * w.k / 4 + row * w.k / 8)
        } else {
            w.raw.as_ptr()
        };
        let scales = w
            .raw
            .as_ptr()
            .add(w.n * w.k / 8 * if THREE { 3 } else { 2 } + row * w.k / GROUP);
        let table = FP8.as_ptr();
        let mut acc = [_mm256_setzero_ps(); M];
        for c in (0..w.k).step_by(256) {
            let lo = [
                _mm256_loadu_si256(low.add(c / 4).cast()),
                _mm256_loadu_si256(low.add(c / 4 + 32).cast()),
            ];
            let hi = if THREE {
                _mm256_loadu_si256(high.add(c / 8).cast())
            } else {
                _mm256_setzero_si256()
            };
            // Constant plane shifts keep the two low-plane loads in registers.
            macro_rules! dot32 {
                ($j:expr) => {{
                    let j: usize = $j;
                    let mut codes = _mm256_and_si256(
                        _mm256_srl_epi16(lo[j / 4], _mm_cvtsi32_si128((j % 4 * 2) as i32)),
                        _mm256_set1_epi8(3),
                    );
                    if THREE {
                        codes = _mm256_or_si256(
                            codes,
                            _mm256_slli_epi16::<2>(_mm256_and_si256(
                                _mm256_srl_epi16(hi, _mm_cvtsi32_si128(j as i32)),
                                _mm256_set1_epi8(1),
                            )),
                        );
                    }
                    let q = _mm256_shuffle_epi8(lut, codes);
                    let halves = [_mm256_castsi256_si128(q), _mm256_extracti128_si256::<1>(q)];
                    let weights = [
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(halves[0])),
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(halves[0]))),
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(halves[1])),
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(halves[1]))),
                    ];
                    for g in 0..32 / GROUP {
                        let scale = _mm256_set1_ps(
                            *table.add(*scales.add((c + j * 32) / GROUP + g) as usize),
                        );
                        for (m, sum) in acc.iter_mut().enumerate() {
                            let mut partial = _mm256_setzero_ps();
                            for i in 0..GROUP / 8 {
                                let at = m * w.k + c + j * 32 + g * GROUP + i * 8;
                                partial = _mm256_fmadd_ps(
                                    weights[g * GROUP / 8 + i],
                                    _mm256_loadu_ps(x.add(at)),
                                    partial,
                                );
                            }
                            *sum = _mm256_fmadd_ps(scale, partial, *sum);
                        }
                    }
                }};
            }
            dot32!(0);
            dot32!(1);
            dot32!(2);
            dot32!(3);
            dot32!(4);
            dot32!(5);
            dot32!(6);
            dot32!(7);
        }
        let tensor = f32::from_le_bytes(w.raw[w.raw.len() - 4..].try_into().unwrap())
            * if grid == 3 { 0.5 } else { 1. };
        std::array::from_fn(|m| {
            let h = _mm_add_ps(
                _mm256_castps256_ps128(acc[m]),
                _mm256_extractf128_ps::<1>(acc[m]),
            );
            let h = _mm_hadd_ps(h, h);
            _mm_cvtss_f32(_mm_hadd_ps(h, h)) * tensor
        })
    }
}

#[cfg(test)]
unsafe fn row_dot(w: Matrix<'_>, r: usize, x: &[f32]) -> f32 {
    let mut xd = vec![0.; x.len()];
    if w.gpu_order {
        deinterleave_gpu(x, &mut xd, w.k);
    } else {
        deinterleave(x, &mut xd);
    }
    unsafe { row::<1>(w, r, x.as_ptr(), xd.as_ptr())[0] }
}

/// One matrix of a pool job. `Plain` writes `y[m * n + r]`; `GateUp` reads the
/// up projection from `up` and writes `swiglu(bf16(gate), bf16(up))` to `y`.
#[derive(Debug)]
pub struct Segment<'a> {
    matrix: Matrix<'a>,
    up: Option<Matrix<'a>>,
    /// rows x k activations in natural order.
    x: &'a [f32],
    /// Same activations through [`deinterleave`] (or [`deinterleave_gpu`] for
    /// a matrix with `with_gpu_order(true)`).
    xd: &'a [f32],
    y: *mut f32,
    rows: usize,
    /// EXL3: each matrix's rotated input H(x * suh) (rows x k) and its row
    /// sums, the gate's then the up's (`x`/`xd` hold the gate's).
    exl3: Option<[(&'a [f32], [f32; 4]); 2]>,
    _out: std::marker::PhantomData<&'a mut [f32]>,
}
impl<'a> Segment<'a> {
    /// EXL3 `y[m * n + r] = W[r] . x[m]` from the rotated input `xh` (see
    /// [`super::exl3::Exl3::prepare`]) and its row sums.
    pub fn exl3_plain(
        matrix: Matrix<'a>,
        xh: &'a [f32],
        sums: [f32; 4],
        y: &'a mut [f32],
        rows: usize,
    ) -> Result<Self> {
        let mut s = Self::new(matrix, None, xh, xh, y, rows)?;
        s.exl3_inputs([(xh, sums), (xh, sums)])?;
        Ok(s)
    }
    /// EXL3 gate/up with their own rotated inputs (their suh differ).
    #[allow(clippy::too_many_arguments)]
    pub fn exl3_gate_up(
        gate: Matrix<'a>,
        up: Matrix<'a>,
        xg: &'a [f32],
        gate_sums: [f32; 4],
        xu: &'a [f32],
        up_sums: [f32; 4],
        y: &'a mut [f32],
        rows: usize,
    ) -> Result<Self> {
        let mut s = Self::new(gate, Some(up), xg, xg, y, rows)?;
        if xu.len() != xg.len() {
            return Err(Error::InvalidShape("GLM CPU segment shape".into()));
        }
        s.exl3_inputs([(xg, gate_sums), (xu, up_sums)])?;
        Ok(s)
    }
    fn exl3_inputs(&mut self, inputs: [(&'a [f32], [f32; 4]); 2]) -> Result<()> {
        let exl3 = |m: Matrix<'_>| matches!(m.format, Format::Exl3 { .. });
        if !exl3(self.matrix)
            || self.up.is_some_and(|u| !exl3(u))
            || !self.matrix.n.is_multiple_of(128)
        {
            return Err(Error::InvalidShape("GLM CPU EXL3 segment".into()));
        }
        self.exl3 = Some(inputs);
        Ok(())
    }
    /// `y[m * n + r] = W[r] . x[m]` for `rows` (1..=4) activation rows.
    pub fn plain(
        matrix: Matrix<'a>,
        x: &'a [f32],
        xd: &'a [f32],
        y: &'a mut [f32],
        rows: usize,
    ) -> Result<Self> {
        Self::reject_exl3(matrix)?;
        Self::new(matrix, None, x, xd, y, rows)
    }
    /// `y[m * n + r] = swiglu(bf16(gate[r] . x[m]), bf16(up[r] . x[m]))`.
    pub fn gate_up(
        gate: Matrix<'a>,
        up: Matrix<'a>,
        x: &'a [f32],
        xd: &'a [f32],
        y: &'a mut [f32],
        rows: usize,
    ) -> Result<Self> {
        Self::reject_exl3(gate)?;
        Self::new(gate, Some(up), x, xd, y, rows)
    }
    /// EXL3 matrices need their rotated inputs (`exl3_plain` / `exl3_gate_up`).
    fn reject_exl3(matrix: Matrix<'_>) -> Result<()> {
        if matches!(matrix.format, Format::Exl3 { .. }) {
            return Err(Error::InvalidShape(
                "GLM CPU EXL3 needs rotated inputs".into(),
            ));
        }
        Ok(())
    }
    fn new(
        matrix: Matrix<'a>,
        up: Option<Matrix<'a>>,
        x: &'a [f32],
        xd: &'a [f32],
        y: &'a mut [f32],
        rows: usize,
    ) -> Result<Self> {
        let (n, k) = (matrix.n, matrix.k);
        if !(1..=4).contains(&rows)
            || x.len() != rows * k
            || xd.len() != rows * k
            || y.len() != rows * n
            || up.is_some_and(|u| u.n != n || u.k != k)
        {
            return Err(Error::InvalidShape("GLM CPU segment shape".into()));
        }
        Ok(Self {
            matrix,
            up,
            x,
            xd,
            y: y.as_mut_ptr(),
            rows,
            exl3: None,
            _out: std::marker::PhantomData,
        })
    }
}
// SAFETY: segments are only shared with pool workers while the submitter
// blocks; every worker writes a disjoint range of output rows.
unsafe impl Send for Segment<'_> {}
unsafe impl Sync for Segment<'_> {}

#[target_feature(enable = "avx2,fma")]
unsafe fn run_range(segments: &[Segment<'_>], begin: usize, end: usize) {
    let mut base = 0;
    for s in segments {
        let n = s.matrix.n;
        let (lo, hi) = (begin.max(base), end.min(base + n));
        if lo < hi {
            unsafe {
                match s.rows {
                    1 => run_rows::<1>(s, lo - base, hi - base),
                    2 => run_rows::<2>(s, lo - base, hi - base),
                    3 => run_rows::<3>(s, lo - base, hi - base),
                    4 => run_rows::<4>(s, lo - base, hi - base),
                    _ => unreachable!(),
                }
            }
        }
        base += n;
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn run_rows<'a, const M: usize>(s: &Segment<'a>, begin: usize, end: usize) {
    let n = s.matrix.n;
    if let Some(inputs) = s.exl3 {
        // 128 outputs (one rotation block) at a time; a range that does not
        // start or end on a block computes the whole block.
        fn view<'a>(m: Matrix<'a>) -> super::exl3::Exl3<'a> {
            let Format::Exl3 { k2 } = m.format else {
                unreachable!()
            };
            super::exl3::Exl3::new(m.raw, m.n, m.k, k2, 2).unwrap()
        }
        let gate = view(s.matrix);
        let up = s.up.map(view);
        let k = s.matrix.k;
        let ptrs = |x: &[f32]| std::array::from_fn::<_, M, _>(|m| unsafe { x.as_ptr().add(m * k) });
        let sums = |v: [f32; 4]| std::array::from_fn::<_, M, _>(|m| v[m]);
        let mut a = [[0f32; 128]; M];
        let mut b = [[0f32; 128]; M];
        let mut c0 = begin / 128 * 128;
        while c0 < end {
            unsafe {
                super::exl3::block::<M>(&gate, ptrs(inputs[0].0), sums(inputs[0].1), c0, &mut a);
                if let Some(up) = &up {
                    super::exl3::block::<M>(up, ptrs(inputs[1].0), sums(inputs[1].1), c0, &mut b);
                }
                for r in begin.max(c0)..end.min(c0 + 128) {
                    for m in 0..M {
                        *s.y.add(m * n + r) = if up.is_some() {
                            swiglu(bf16(a[m][r - c0]), bf16(b[m][r - c0]))
                        } else {
                            a[m][r - c0]
                        };
                    }
                }
            }
            c0 += 128;
        }
        return;
    }
    for r in begin..end {
        unsafe {
            let a = row::<M>(s.matrix, r, s.x.as_ptr(), s.xd.as_ptr());
            if let Some(up) = s.up {
                let b = row::<M>(up, r, s.x.as_ptr(), s.xd.as_ptr());
                for m in 0..M {
                    *s.y.add(m * n + r) = if s.matrix.gpu_order {
                        swiglu_gpu_order(bf16(a[m]), bf16(b[m]))
                    } else {
                        swiglu(bf16(a[m]), bf16(b[m]))
                    };
                }
            } else {
                for (m, v) in a.into_iter().enumerate() {
                    *s.y.add(m * n + r) = v;
                }
            }
        }
    }
}

/// Rows claimed per step. Small enough that a preempted core delays the layer
/// by at most one chunk; 128 rows also keep each swizzled scale tile on one
/// worker, avoiding four cores fetching the same cache lines.
const CHUNK: usize = 128;
/// Dispatch state shared by the submitter and every worker. `state` packs the
/// job epoch (16 bits), its row count (24 bits) and the next unclaimed row
/// (24 bits), so one load gives a consistent view and a claim (CAS) succeeds
/// only for the live job while rows remain: `segments` is alive for it.
#[derive(Debug)]
struct Shared {
    state: AtomicU64,
    done: AtomicU64,
    segments: AtomicPtr<Segment<'static>>,
    count: AtomicU64,
    submitter: std::sync::Mutex<Option<std::thread::Thread>>,
    stop: AtomicBool,
    work_ns: AtomicU64,
}
#[derive(Debug)]
struct Worker {
    thread: Option<JoinHandle<()>>,
}
/// Default busy-wait before parking: longer than the GPU work between two
/// layers, so workers stay awake (and clocked up) through a decode step and
/// park only when idle. Parking per layer measured 14-17 GB/s vs 28-31 GB/s.
/// Chunk claiming keeps a preempted spinning worker from stalling a job.
pub const DEFAULT_SPIN: Duration = Duration::from_millis(2);
#[derive(Debug)]
pub struct Pool {
    workers: Vec<Worker>,
    shared: Arc<Shared>,
    spin: Duration,
    epoch: u32,
    /// Sum over dispatches of the average worker's compute time.
    pub worker_ms: f64,
    /// Dispatch through completion, including host waiting.
    pub dispatch_ms: f64,
}
fn idle(spin: Duration, since: &mut Option<Instant>) {
    let start = *since.get_or_insert_with(Instant::now);
    if start.elapsed() < spin {
        for _ in 0..64 {
            std::hint::spin_loop();
        }
    } else {
        std::thread::park_timeout(Duration::from_millis(5));
    }
}
/// Claims and runs chunks of the current job; returns whether any row ran.
fn work(shared: &Shared) -> bool {
    let mut ran = false;
    loop {
        let state = shared.state.load(Ordering::Acquire);
        let (total, next) = ((state >> 24) & 0xff_ffff, state & 0xff_ffff);
        if next >= total {
            return ran;
        }
        let end = (next + CHUNK as u64).min(total);
        if shared
            .state
            .compare_exchange_weak(
                state,
                (state & !0xff_ffff) | end,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            continue;
        }
        let start = Instant::now();
        // SAFETY: the claim succeeded with rows remaining, so the submitter is
        // still waiting for `done` and keeps the segments and buffers alive.
        // Claimed row ranges are disjoint.
        unsafe {
            let segments = std::slice::from_raw_parts(
                shared.segments.load(Ordering::Acquire),
                shared.count.load(Ordering::Acquire) as usize,
            );
            run_range(segments, next as usize, end as usize);
        }
        shared
            .work_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        ran = true;
        if shared.done.fetch_add(end - next, Ordering::AcqRel) + (end - next) == total
            && let Some(t) = shared.submitter.lock().unwrap().as_ref()
        {
            t.unpark();
        }
    }
}
impl Pool {
    pub fn new(cores: &[u16]) -> Result<Self> {
        Self::with_spin(cores, DEFAULT_SPIN)
    }
    /// `spin = true` never parks (stage 8 comparison); otherwise [`DEFAULT_SPIN`].
    pub fn with_wait(cores: &[u16], spin: bool) -> Result<Self> {
        Self::with_spin(cores, if spin { Duration::MAX } else { DEFAULT_SPIN })
    }
    pub fn with_spin(cores: &[u16], spin: Duration) -> Result<Self> {
        if cores.is_empty() {
            return Err(Error::InvalidPlan("GLM CPU needs compute cores".into()));
        }
        let shared = Arc::new(Shared {
            state: AtomicU64::new(0),
            done: AtomicU64::new(0),
            segments: AtomicPtr::new(std::ptr::null_mut()),
            count: AtomicU64::new(0),
            submitter: std::sync::Mutex::new(None),
            stop: AtomicBool::new(false),
            work_ns: AtomicU64::new(0),
        });
        let mut result = Self {
            workers: Vec::new(),
            shared: shared.clone(),
            spin,
            epoch: 0,
            worker_ms: 0.,
            dispatch_ms: 0.,
        };
        for &core in cores {
            let shared = shared.clone();
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let thread = std::thread::Builder::new()
                .name(format!("glm-cpu-{core}"))
                .spawn(move || {
                    let ready = crate::cpu::pin_thread(core);
                    let ok = ready.is_ok();
                    if tx.send(ready).is_err() || !ok {
                        return;
                    }
                    let mut since = None;
                    while !shared.stop.load(Ordering::Acquire) {
                        if work(&shared) {
                            since = None;
                        } else {
                            idle(spin, &mut since);
                        }
                    }
                })?;
            result.workers.push(Worker {
                thread: Some(thread),
            });
            rx.recv()
                .map_err(|_| Error::InvalidPlan("GLM CPU worker startup".into()))??;
        }
        Ok(result)
    }
    pub fn threads(&self) -> usize {
        self.workers.len()
    }
    pub fn gemv(&mut self, matrix: Matrix<'_>, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.batch(matrix, x, y, 1)
    }
    /// At most four token rows; decode each weight once.
    pub fn batch(
        &mut self,
        matrix: Matrix<'_>,
        x: &[f32],
        y: &mut [f32],
        rows: usize,
    ) -> Result<()> {
        if !(1..=4).contains(&rows) || x.len() != rows * matrix.k || y.len() != rows * matrix.n {
            return Err(Error::InvalidShape("GLM CPU GEMV activation shape".into()));
        }
        let mut xd = vec![0.; x.len()];
        if matrix.gpu_order {
            deinterleave_gpu(x, &mut xd, matrix.k);
        } else {
            deinterleave(x, &mut xd);
        }
        self.run(&[Segment::plain(matrix, x, &xd, y, rows)?])
    }
    /// Runs every segment; workers claim chunks of the concatenated rows, so a
    /// preempted core does not hold the others back.
    pub fn run(&mut self, segments: &[Segment<'_>]) -> Result<()> {
        let total: usize = segments.iter().map(|s| s.matrix.n).sum();
        if total == 0 {
            return Ok(());
        }
        if total >= 1 << 24 {
            return Err(Error::InvalidShape("GLM CPU job rows".into()));
        }
        let start = Instant::now();
        let s = &self.shared;
        *s.submitter.lock().unwrap() = Some(std::thread::current());
        s.work_ns.store(0, Ordering::Relaxed);
        s.done.store(0, Ordering::Relaxed);
        s.segments
            .store(segments.as_ptr().cast_mut().cast(), Ordering::Relaxed);
        s.count.store(segments.len() as u64, Ordering::Relaxed);
        self.epoch = (self.epoch + 1) & 0xffff;
        // Publishes the fields above; rows start at 0 for the new epoch.
        s.state.store(
            (u64::from(self.epoch) << 48) | ((total as u64) << 24),
            Ordering::Release,
        );
        for w in &self.workers {
            w.thread.as_ref().unwrap().thread().unpark();
        }
        let mut since = None;
        while s.done.load(Ordering::Acquire) < total as u64 {
            idle(self.spin, &mut since);
        }
        self.dispatch_ms += start.elapsed().as_secs_f64() * 1000.;
        self.worker_ms +=
            s.work_ns.load(Ordering::Relaxed) as f64 / 1e6 / self.workers.len() as f64;
        Ok(())
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        for w in &self.workers {
            w.thread.as_ref().unwrap().thread().unpark();
        }
        for w in &mut self.workers {
            if let Some(t) = w.thread.take() {
                let _ = t.join();
            }
        }
    }
}

/// Independent F64 accumulation for SIMD tests.
#[cfg(test)]
fn gemv_reference(w: Matrix<'_>, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.k);
    assert_eq!(y.len(), w.n);
    let values = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
    for (r, out) in y.iter_mut().enumerate() {
        let mut sum = 0f64;
        for (c, &v) in x.iter().enumerate() {
            let a = if w.nv() {
                let q = (w.raw[(r * w.k + c) / 2] >> (4 * (c % 2))) & 15;
                values[(q & 7) as usize] * if q & 8 != 0 { -1. } else { 1. } * w.scale(r, c)
            } else {
                bf16(fp8(w.raw[r * w.k + c]) * w.scale(r, c))
            };
            sum += a as f64 * v as f64;
        }
        *out = sum as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(n: usize, k: usize, nv: bool) -> Vec<u8> {
        let bytes = if nv {
            QuantSpec::Nvfp4.matrix_bytes(n as u64, k as u64).unwrap() as usize
        } else {
            n * k + n.div_ceil(128) * k.div_ceil(128) * 4
        };
        let mut raw = vec![0; bytes];
        let mut s = 23u32;
        for b in &mut raw[..n * k / if nv { 2 } else { 1 }] {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *b = if nv { s as u8 } else { (s as u8) % 127 };
        }
        if nv {
            for (i, b) in raw[n * k / 2..bytes - 4].iter_mut().enumerate() {
                *b = 0x10 + (i % 96) as u8;
            }
            raw[bytes - 4..].copy_from_slice(&0.037f32.to_le_bytes());
        } else {
            for (i, b) in raw[n * k..].as_chunks_mut::<4>().0.iter_mut().enumerate() {
                b.copy_from_slice(&(0.0023f32 * (i + 1) as f32).to_le_bytes());
            }
        }
        raw
    }
    fn lowbit_fixture(n: usize, k: usize, grid: u8, group: usize) -> Vec<u8> {
        let format = Format::Lowbit { grid, group };
        let mut raw = vec![0; format.matrix_bytes(n, k).unwrap()];
        let mut seed = 0x12345678u32;
        let planes = n * k / 8 * if matches!(grid, 2 | 3) { 3 } else { 2 };
        for b in &mut raw[..planes] {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *b = seed as u8;
        }
        for (i, b) in raw[planes..].iter_mut().enumerate() {
            *b = (i * 31 % 120) as u8;
        }
        let end = raw.len();
        raw[end - 4..].copy_from_slice(&0.037f32.to_le_bytes());
        raw
    }
    // Decode individual columns independently of the SIMD plane loads.
    fn lowbit_value(w: Matrix<'_>, r: usize, c: usize) -> (f32, f32) {
        let Format::Lowbit { grid, group } = w.format else {
            unreachable!()
        };
        let low = w.raw[r * w.k / 4 + c / 128 * 32 + c % 32];
        let mut code = (low >> (2 * (c / 32 % 4))) & 3;
        if matches!(grid, 2 | 3) {
            let high = w.raw[w.n * w.k / 4 + r * w.k / 8 + c / 256 * 32 + c % 32];
            code |= ((high >> (c / 32 % 8)) & 1) << 2;
        }
        let planes = w.n * w.k / 8 * if matches!(grid, 2 | 3) { 3 } else { 2 };
        (
            // Literal specification grids, independent of the SIMD integer LUT.
            [
                [-3., -1., 1., 3., 0., 0., 0., 0.],
                [-2., -1., 0., 1., 0., 0., 0., 0.],
                [-4., -3., -2., -1., 0., 1., 2., 3.],
                [-4., -3., -1.5, -0.5, 0.5, 1.5, 3., 4.],
                [-4., -1., 1., 4., 0., 0., 0., 0.],
            ][grid as usize][code as usize]
                * if grid == 3 { 2. } else { 1. },
            fp8(w.raw[planes + r * w.k / group + c / group]),
        )
    }
    fn reduce_lanes(v: [f32; 8]) -> f32 {
        ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7]))
    }
    #[test]
    fn glm53_cpu_nvfp4_gpu_order_matches_scalar_bits() -> Result<()> {
        let mut pool = Pool::new(hayate_core::machine::MAIN.cpu.compute)?;
        for (n, k) in [(3, 16), (7, 80), (129, 256), (5, 2048), (3, 4096)] {
            let raw = fixture(n, k, true);
            let w = Matrix::new(&raw, n, k, true)?.with_gpu_order(true);
            let global = f32::from_le_bytes(raw[raw.len() - 4..].try_into().unwrap());
            for rows in 1..=4 {
                let x: Vec<_> = (0..rows * k)
                    .map(|i| bf16(((i * 83 % 251) as f32 - 125.) / 47.))
                    .collect();
                let mut actual = vec![0.; rows * n];
                pool.batch(w, &x, &mut actual, rows)?;
                let mut xd = vec![0.; x.len()];
                deinterleave_gpu(&x, &mut xd, k);
                let mut fused = vec![0.; rows * n];
                pool.run(&[Segment::gate_up(w, w, &x, &xd, &mut fused, rows)?])?;
                for (&a, &b) in actual.iter().zip(&fused) {
                    let gate = bf16(a).min(10.);
                    let sigmoid = 1. / (1. + (-gate).exp());
                    let expected = bf16(bf16(gate * sigmoid) * bf16(a).clamp(-10., 10.));
                    assert_eq!(b.to_bits(), expected.to_bits());
                }
                for m in 0..rows {
                    for r in 0..n {
                        let mut lanes = [0f32; 32];
                        for (lane, sum) in lanes.iter_mut().enumerate() {
                            for c in (lane * 8..k).step_by(256) {
                                let scol = c / 16;
                                let cols = (k / 16).next_multiple_of(4);
                                let at = ((r / 128) * (cols / 4) + scol / 4) * 512
                                    + r % 32 * 16
                                    + (r % 128 / 32) * 4
                                    + scol % 4;
                                let multiplier = fp8(raw[n * k / 2 + at]) * global;
                                for j in 0..8 {
                                    let code =
                                        (raw[(r * k + c + j) / 2] >> (4 * ((c + j) % 2))) & 15;
                                    let level = [0f32, 0.5, 1., 1.5, 2., 3., 4., 6.]
                                        [(code & 7) as usize]
                                        * if code & 8 != 0 { -1. } else { 1. };
                                    let weight = level * multiplier;
                                    *sum = weight.mul_add(x[m * k + c + j], *sum);
                                }
                            }
                        }
                        for offset in [16, 8, 4, 2, 1] {
                            for lane in 0..32 - offset {
                                lanes[lane] += lanes[lane + offset];
                            }
                        }
                        assert_eq!(
                            actual[m * n + r].to_bits(),
                            lanes[0].to_bits(),
                            "n={n} k={k} m={m} r={r}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn glm53_cpu_lowbit_all_layouts_match_scalar_bits_and_f64() -> Result<()> {
        let mut pool = Pool::new(hayate_core::machine::MAIN.cpu.compute)?;
        for grid in 0..5 {
            for group in [16, 32] {
                for (n, k) in [(1, 256), (7, 512), (129, 256), (3, 4096)] {
                    let raw = lowbit_fixture(n, k, grid, group);
                    let w = Matrix::with_format(&raw, n, k, Format::Lowbit { grid, group })?;
                    let tensor = f32::from_le_bytes(raw[raw.len() - 4..].try_into().unwrap())
                        * if grid == 3 { 0.5 } else { 1. };
                    for rows in 1..=4 {
                        let x: Vec<_> = (0..rows * k)
                            .map(|i| bf16(((i * 17 % 127) as f32 - 63.) / 13.))
                            .collect();
                        let mut actual = vec![0.; rows * n];
                        pool.batch(w, &x, &mut actual, rows)?;
                        let (mut error, mut energy) = (0f64, 0f64);
                        for m in 0..rows {
                            for r in 0..n {
                                let mut acc = [0f32; 8];
                                let mut reference = 0f64;
                                for c in (0..k).step_by(group) {
                                    for (lane, sum) in acc.iter_mut().enumerate() {
                                        let mut partial = 0f32;
                                        for j in (lane..group).step_by(8) {
                                            let (q, scale) = lowbit_value(w, r, c + j);
                                            let a = x[m * k + c + j];
                                            partial = q.mul_add(a, partial);
                                            reference += q as f64 * scale as f64 * a as f64;
                                        }
                                        *sum = lowbit_value(w, r, c).1.mul_add(partial, *sum);
                                    }
                                }
                                let expected = reduce_lanes(acc) * tensor;
                                assert_eq!(
                                    actual[m * n + r].to_bits(),
                                    expected.to_bits(),
                                    "grid={grid} group={group} n={n} k={k} m={m} r={r}"
                                );
                                reference *= tensor as f64;
                                error += (actual[m * n + r] as f64 - reference).powi(2);
                                energy += reference.powi(2);
                            }
                        }
                        assert!((error / energy).sqrt() < 1e-5);
                    }
                    assert!(Matrix::with_format(&raw[..raw.len() - 1], n, k, w.format).is_err());
                }
            }
        }
        for format in [
            Format::Lowbit { grid: 5, group: 16 },
            Format::Lowbit { grid: 0, group: 8 },
        ] {
            assert!(format.matrix_bytes(1, 256).is_none());
        }
        assert!(
            Format::Lowbit { grid: 0, group: 16 }
                .matrix_bytes(1, 128)
                .is_none()
        );
        assert!(
            Format::Lowbit { grid: 0, group: 16 }
                .matrix_bytes(usize::MAX, 256)
                .is_none()
        );
        Ok(())
    }
    #[test]
    fn glm53_cpu_nvfp4_preserves_original_accumulation_bits() -> Result<()> {
        let mut pool = Pool::new(hayate_core::machine::MAIN.cpu.compute)?;
        let levels = [0f32, 1., 2., 3., 4., 6., 8., 12.];
        for (n, k) in [(129, 256), (7, 4096)] {
            let raw = fixture(n, k, true);
            let w = Matrix::new(&raw, n, k, true)?;
            let global = f32::from_le_bytes(raw[raw.len() - 4..].try_into().unwrap()) * 0.5;
            for rows in 1..=4 {
                let x: Vec<_> = (0..rows * k)
                    .map(|i| bf16((i as f32 * 0.13).sin()))
                    .collect();
                let mut y = vec![0.; rows * n];
                pool.batch(w, &x, &mut y, rows)?;
                let mut up_raw = raw.clone();
                let end = up_raw.len();
                up_raw[end - 4..].copy_from_slice(&(-0.071f32).to_le_bytes());
                let up = Matrix::new(&up_raw, n, k, true)?;
                let mut up_y = vec![0.; rows * n];
                pool.batch(up, &x, &mut up_y, rows)?;
                let expected: Vec<_> = y
                    .iter()
                    .zip(&up_y)
                    .map(|(&a, &b)| swiglu(bf16(a), bf16(b)))
                    .collect();
                let mut fused = vec![0.; rows * n];
                let mut xd = vec![0.; x.len()];
                deinterleave(&x, &mut xd);
                pool.run(&[Segment::gate_up(w, up, &x, &xd, &mut fused, rows)?])?;
                assert_eq!(
                    fused.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                for m in 0..rows {
                    for r in 0..n {
                        let mut acc = [[0f32; 8]; 2];
                        for c in (0..k).step_by(16) {
                            let cols = (k / 16).next_multiple_of(4);
                            let col = c / 16;
                            let at = ((r / 128) * (cols / 4) + col / 4) * 512
                                + r % 32 * 16
                                + (r % 128 / 32) * 4
                                + col % 4;
                            let scale = fp8(raw[n * k / 2 + at]);
                            for lane in 0..8 {
                                let packed = raw[r * k / 2 + c / 2 + lane];
                                let q = |v: u8| {
                                    levels[(v & 7) as usize] * if v & 8 != 0 { -1. } else { 1. }
                                };
                                let t = q(packed >> 4).mul_add(
                                    x[m * k + c + lane * 2 + 1],
                                    q(packed & 15) * x[m * k + c + lane * 2],
                                );
                                acc[c / 16 % 2][lane] = scale.mul_add(t, acc[c / 16 % 2][lane]);
                            }
                        }
                        let sum = std::array::from_fn(|i| acc[0][i] + acc[1][i]);
                        assert_eq!(
                            y[m * n + r].to_bits(),
                            (reduce_lanes(sum) * global).to_bits()
                        );
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn glm53_cpu_simd_matches_f64_and_validates_layout() {
        for (n, k) in [(1, 16), (129, 80), (257, 256), (7, 4096)] {
            for nv in [false, true] {
                let raw = fixture(n, k, nv);
                let m = Matrix::new(&raw, n, k, nv).unwrap();
                let x: Vec<_> = (0..k).map(|c| bf16((c as f32 * 0.13).sin())).collect();
                let mut reference = vec![0.; n];
                gemv_reference(m, &x, &mut reference);
                let actual: Vec<_> = (0..n).map(|r| unsafe { row_dot(m, r, &x) }).collect();
                let err: f64 = actual
                    .iter()
                    .zip(&reference)
                    .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                    .sum();
                let energy: f64 = reference.iter().map(|&b| (b as f64).powi(2)).sum();
                assert!((err / energy).sqrt() < 1e-5);
                assert!(Matrix::new(&raw[..raw.len() - 1], n, k, nv).is_err());
            }
        }
        assert!(Matrix::new(&[], 0, 0, true).is_err());
    }
    #[test]
    fn glm53_cpu_batch_matches_single_row_bits() -> Result<()> {
        let mut pool = Pool::new(hayate_core::machine::MAIN.cpu.compute)?;
        for nv in [false, true] {
            for (n, k) in [(129, 80), (7, 4096)] {
                let raw = fixture(n, k, nv);
                let matrix = Matrix::new(&raw, n, k, nv)?;
                for rows in 1..=4 {
                    let x: Vec<_> = (0..rows * k)
                        .map(|i| bf16(((i * 17 % 127) as f32 - 63.) / 13.))
                        .collect();
                    let mut actual = vec![0.; rows * n];
                    pool.batch(matrix, &x, &mut actual, rows)?;
                    let mut expected = vec![0.; rows * n];
                    for row in 0..rows {
                        pool.gemv(
                            matrix,
                            &x[row * k..(row + 1) * k],
                            &mut expected[row * n..(row + 1) * n],
                        )?;
                    }
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    fn glm53_cpu_pool_rows_and_repeated_dispatch() {
        let mut pool = Pool::new(hayate_core::machine::MAIN.cpu.compute).unwrap();
        for (n, k) in [(3, 16), (129, 256)] {
            let raw = fixture(n, k, true);
            let m = Matrix::new(&raw, n, k, true).unwrap();
            let x = vec![0.5; k];
            let mut y = vec![0.; n];
            let mut expected = vec![0.; n];
            gemv_reference(m, &x, &mut expected);
            for _ in 0..20 {
                pool.gemv(m, &x, &mut y).unwrap();
                for (&a, &b) in y.iter().zip(&expected) {
                    assert!((a - b).abs() < 1e-5);
                }
            }
            assert!(pool.gemv(m, &[], &mut y).is_err());
        }
    }
    #[test]
    fn glm53_cpu_pool_wakes_after_idle_and_wait_modes_match() -> Result<()> {
        let raw = fixture(129, 256, true);
        let matrix = Matrix::new(&raw, 129, 256, true)?;
        let x = vec![0.5; 256];
        let expected: Vec<_> = (0..129)
            .map(|r| unsafe { row_dot(matrix, r, &x) })
            .collect();
        for spin in [false, true] {
            for count in [1, 4, 8] {
                let mut pool =
                    Pool::with_wait(&hayate_core::machine::MAIN.cpu.compute[..count], spin)?;
                let mut y = vec![0.; 129];
                for _ in 0..8 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    pool.gemv(matrix, &x, &mut y)?;
                    assert_eq!(y, expected);
                }
                assert!(pool.worker_ms > 0.);
                assert!(pool.dispatch_ms >= pool.worker_ms);
            }
        }
        assert!(Pool::new(&[]).is_err());
        Ok(())
    }
    #[test]
    fn glm53_cpu_scale_tile_boundaries() {
        let (n, k) = (129, 80);
        let bytes = QuantSpec::Nvfp4.matrix_bytes(n as u64, k as u64).unwrap() as usize;
        let mut raw = vec![0; bytes];
        raw[..n * k / 2].fill(0x11);
        raw[bytes - 4..].copy_from_slice(&2f32.to_le_bytes());
        for (at, scale) in [
            (0, 0x38),
            (499, 0x40),
            (4, 0x30),
            (1020, 0x48),
            (1536, 0x50),
        ] {
            raw[n * k / 2 + at] = scale;
        }
        let m = Matrix::new(&raw, n, k, true).unwrap();
        for (r, expected) in [(0, 16.), (31, 32.), (32, 8.), (127, 64.), (128, 128.)] {
            assert_eq!(unsafe { row_dot(m, r, &vec![1.; k]) }, expected);
        }
        let k = 256;
        let mut raw = vec![0x38; n * k + 16];
        for (i, scale) in [1f32, 2., 3., 4.].into_iter().enumerate() {
            raw[n * k + i * 4..n * k + (i + 1) * 4].copy_from_slice(&scale.to_le_bytes());
        }
        let m = Matrix::new(&raw, n, k, false).unwrap();
        assert_eq!(unsafe { row_dot(m, 0, &vec![1.; k]) }, 384.);
        assert_eq!(unsafe { row_dot(m, 128, &vec![1.; k]) }, 896.);
    }
    #[test]
    fn glm53_cpu_bf16_ties() {
        assert_eq!(bf16(f32::from_bits(0x3f808000)).to_bits(), 0x3f800000);
        assert_eq!(bf16(f32::from_bits(0x3f818000)).to_bits(), 0x3f820000);
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA GPU; exact decode GEMV order, rounded=0, M=1..4"]
    fn glm53_cpu_nvfp4_gpu_order_matches_gpu() -> Result<()> {
        use crate::{
            glm53::ffi as g,
            runtime::{Device, check},
        };
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut pool = Pool::new(hayate_core::machine::MAIN.cpu.compute)?;
        for (n, k) in [(129, 256), (2048, 4096), (4096, 2048)] {
            let raw = fixture(n, k, true);
            let mut weights = device.alloc(raw.len())?;
            stream.copy_h2d(&mut weights, &raw)?;
            for rows in 1..=4 {
                let x: Vec<_> = (0..rows * k)
                    .map(|c| bf16(((c * 83 % 251) as f32 - 125.) / 47.))
                    .collect();
                let mut cpu = vec![0.; rows * n];
                pool.batch(
                    Matrix::new(&raw, n, k, true)?.with_gpu_order(true),
                    &x,
                    &mut cpu,
                    rows,
                )?;
                let input: Vec<_> = x
                    .iter()
                    .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                    .collect();
                let mut activation = device.alloc(input.len())?;
                stream.copy_h2d(&mut activation, &input)?;
                let mut out = device.alloc(rows * n * 2)?;
                check(
                    unsafe {
                        g::hayate_glm_dense_batch(
                            out.as_mut_ptr(),
                            activation.as_ptr(),
                            g::DenseWeight {
                                data: weights.as_ptr(),
                                scale: std::ptr::null(),
                                format: 3,
                            },
                            rows as u32,
                            n as u32,
                            k as u32,
                            0,
                            stream.as_raw(),
                        )
                    },
                    "GPU-order NVFP4 GEMV",
                )?;
                let mut gpu = vec![0; rows * n * 2];
                stream.copy_d2h(&mut gpu, &out)?;
                let expected: Vec<_> = cpu
                    .iter()
                    .flat_map(|&v| ((bf16(v).to_bits() >> 16) as u16).to_le_bytes())
                    .collect();
                assert_eq!(expected, gpu, "{n}x{k} rows={rows}");
            }
        }
        Ok(())
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires supported CUDA GPU; CPU AVX2 and GPU NVFP4 GEMV same input"]
    fn glm53_cpu_nvfp4_gemv_matches_gpu() -> Result<()> {
        use crate::runtime::{Device, check};
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut pool = Pool::new(hayate_core::machine::MAIN.cpu.compute)?;
        for (n, k) in [(129, 256), (2048, 4096), (4096, 2048)] {
            let raw = fixture(n, k, true);
            let x: Vec<_> = (0..k).map(|c| bf16((c as f32 * 0.13).sin())).collect();
            let mut cpu = vec![0.; n];
            pool.gemv(Matrix::new(&raw, n, k, true)?, &x, &mut cpu)?;
            let input: Vec<_> = x
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let mut weights = device.alloc(raw.len())?;
            stream.copy_h2d(&mut weights, &raw)?;
            let mut activation = device.alloc(input.len())?;
            stream.copy_h2d(&mut activation, &input)?;
            let mut out = device.alloc(n * 2)?;
            check(
                unsafe {
                    crate::llm::ffi::hayate_llm_gemv(
                        out.as_mut_ptr(),
                        activation.as_ptr(),
                        weights.as_ptr(),
                        n as u32,
                        k as u32,
                        1,
                        0,
                        std::ptr::null_mut(),
                        4,
                        1,
                        0,
                        stream.as_raw(),
                    )
                },
                "CPU/GPU NVFP4 GEMV",
            )?;
            let mut gpu = vec![0; n * 2];
            stream.copy_d2h(&mut gpu, &out)?;
            let (mut error, mut energy) = (0f64, 0f64);
            for (&a, b) in cpu.iter().zip(gpu.as_chunks::<2>().0) {
                let b = f32::from_bits((u16::from_le_bytes(*b) as u32) << 16) as f64;
                error += (bf16(a) as f64 - b).powi(2);
                energy += b * b;
            }
            let relative = (error / energy).sqrt();
            println!("CPU/GPU NVFP4 {n}x{k} relative_l2={relative:.9}");
            assert!(relative < 1e-3);
        }
        Ok(())
    }
}
