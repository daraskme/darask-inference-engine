//! EXL3 (exllamav3 trellis) routed-expert matrices on the CPU (crates/models/src/glm53/EXL3.md).
//!
//! A `[n = outputs, k = inputs]` matrix image is its trellis ([k/16][n/16] tiles of 4·k2 little-endian u32 words,
//! k2 = twice the bits per weight), then FP16 suh [k] and svh [n]; y = H(H(x·suh)·W')·svh with H the 128-point
//! Sylvester Hadamard over √128. Step p of a tile is the 16-bit window ending at bit ⌊(p+1)·k2/2⌋ of the tile's
//! MSB-first word ring; mul1 decodes a state s to (1024 + bytesum(s·0x83DCD12D))·k_inv + c, so a column's sum is
//! k_inv·Σ x·bytesum + k_zero·Σ x and only the byte sums are accumulated.
//!
//! The AVX2 path for exllamav3 1.6's half-integer K (k2 3/5/7) is Project Maya's (src/kernels/cpu/exl3_cpu.cpp,
//! MIT) after exllamav3's CPU expert kernel (exllamav3_ext/cpu/moe_mul1.cpp, MIT, turboderp): a tile's words are
//! regrouped into k2 "pair" registers once, then each of 8 groups (4 tile rows × 8 columns) is one funnel shift of
//! two permuted words, and each row an immediate shift, a mask, vpmulld, vpmaddubsw and vpmaddwd. Other bitrates
//! take the scalar reference here (correct, slower). Activations stay F32; the rotated input is not rounded to FP16
//! (the GPU's is), so CPU and GPU agree to the FP16 rounding of the input and of each weight.
use hayate_core::{Error, QuantSpec, Result};

pub const MUL1: u32 = 0x83DC_D12D;

/// FP16 bits -> f32 (exact).
pub const fn f16(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 31) as u32;
    let man = (h & 1023) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: man · 2^-24
            let v = man as f32 * (1. / 16_777_216.);
            return if sign != 0 { -v } else { v };
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}
/// mul1: a state's weight is bytesum · K_INV + K_ZERO (before the GPU's FP16 rounding).
pub const K_INV: f32 = f16(0x1eee);
pub const K_ZERO: f32 = 1024. * f16(0x1eee) + f16(0xc931);
const HAD_SCALE: f32 = 0.088_388_35;

const fn win_end(p: usize, k2: usize) -> usize {
    (p + 1) * k2 / 2 + 128 * k2
}
/// (tile row, tile column) of step p (the tensor-core fragment order).
pub const fn step_cell(p: usize) -> (usize, usize) {
    let (lane, j) = (p >> 3, p & 7);
    (
        (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1),
        (lane >> 2) + 8 * (j >> 2),
    )
}

/// Step p's 16-bit state of the tile `words` (4·k2 words).
pub fn tile_state(words: &[u32], k2: usize, p: usize) -> u32 {
    let w = 4 * k2;
    let b1 = win_end(p, k2);
    let (i1, i0) = ((b1 - 1) >> 5, (b1 - 16) >> 5);
    let shift = ((i1 + 1) << 5) - b1;
    let m = (u64::from(words[i0 % w]) << 32) | u64::from(words[i1 % w]);
    ((m >> shift) & 0xffff) as u32
}
/// The codebook value of a state, exactly the FP16 value the GPU decodes (as f32).
pub fn value(state: u32, cb: u8) -> f32 {
    if cb == 2 {
        let x = state.wrapping_mul(MUL1);
        let total = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
        // The F32 FMA is exact here (22-bit product, bias of the same magnitude); one rounding to FP16.
        return round_f16(((1024 + total) as f32).mul_add(f16(0x1eee), f16(0xc931)));
    }
    let mut x = if cb == 1 {
        state.wrapping_mul(0xCBAC_1FED)
    } else {
        state.wrapping_mul(89_226_354).wrapping_add(64_248_484)
    };
    x = (x & 0x8fff_8fff) ^ 0x3b60_3b60;
    round_f16(f16(x as u16) + f16((x >> 16) as u16))
}
/// Round an f32 to the nearest FP16 value (ties to even), returned as f32 (no overflow handling: the codebooks'
/// values are far below 65504).
fn round_f16(v: f32) -> f32 {
    if v.abs() < 6.103_515_6e-5 {
        // below the smallest normal FP16: multiples of 2^-24
        return (v * 16_777_216.).round_ties_even() / 16_777_216.;
    }
    let bits = v.to_bits();
    let rounded = (bits + (1 << 12) - 1 + ((bits >> 13) & 1)) & !((1u32 << 13) - 1);
    f32::from_bits(rounded)
}

/// The blockwise 128-point Sylvester transform over √128, in place.
pub fn had128(v: &mut [f32]) {
    for x in v.as_chunks_mut::<128>().0 {
        let mut h = 1;
        while h < 128 {
            for i in (0..128).step_by(2 * h) {
                for j in i..i + h {
                    let (a, b) = (x[j], x[j + h]);
                    x[j] = a + b;
                    x[j + h] = a - b;
                }
            }
            h *= 2;
        }
        for v in x.iter_mut() {
            *v *= HAD_SCALE;
        }
    }
}

/// One EXL3 matrix image, read in place.
#[derive(Debug, Clone, Copy)]
pub struct Exl3<'a> {
    raw: &'a [u8],
    pub n: usize,
    pub k: usize,
    pub k2: usize,
    pub cb: u8,
}
impl<'a> Exl3<'a> {
    pub fn new(raw: &'a [u8], n: usize, k: usize, k2: u8, cb: u8) -> Result<Self> {
        let bytes = QuantSpec::Exl3 { k2, cb }.matrix_bytes(n as u64, k as u64);
        if bytes != Some(raw.len() as u64) {
            return Err(Error::InvalidShape("GLM EXL3 matrix layout".into()));
        }
        Ok(Self {
            raw,
            n,
            k,
            k2: k2 as usize,
            cb,
        })
    }
    fn trellis_bytes(&self) -> usize {
        self.n * self.k * self.k2 / 16
    }
    pub fn suh(&self, i: usize) -> f32 {
        let at = self.trellis_bytes() + 2 * i;
        f16(u16::from_le_bytes([self.raw[at], self.raw[at + 1]]))
    }
    pub fn svh(&self, i: usize) -> f32 {
        let at = self.trellis_bytes() + 2 * self.k + 2 * i;
        f16(u16::from_le_bytes([self.raw[at], self.raw[at + 1]]))
    }
    /// Words of tile (slice ks of 16 inputs, tile column tc of 16 outputs).
    fn tile(&self, ks: usize, tc: usize) -> *const u8 {
        let w = 4 * self.k2;
        unsafe { self.raw.as_ptr().add(((ks * (self.n / 16) + tc) * w) * 4) }
    }
    fn tile_words(&self, ks: usize, tc: usize) -> Vec<u32> {
        let p = self.tile(ks, tc);
        (0..4 * self.k2)
            .map(|i| unsafe { p.cast::<u32>().add(i).read_unaligned() })
            .collect()
    }
    /// W' [k][n] in the rotated basis, row-major (tests, references).
    pub fn inner(&self) -> Vec<f32> {
        let mut w = vec![0.; self.k * self.n];
        for ks in 0..self.k / 16 {
            for tc in 0..self.n / 16 {
                let words = self.tile_words(ks, tc);
                for p in 0..256 {
                    let (r, c) = step_cell(p);
                    w[(ks * 16 + r) * self.n + tc * 16 + c] =
                        value(tile_state(&words, self.k2, p), self.cb);
                }
            }
        }
        w
    }
    /// xh = H(x · suh) for one activation row, and Σ xh (the k_zero term).
    pub fn prepare(&self, x: &[f32], xh: &mut [f32]) -> f32 {
        debug_assert!(x.len() == self.k && xh.len() == self.k);
        for (i, (o, &v)) in xh.iter_mut().zip(x).enumerate() {
            *o = v * self.suh(i);
        }
        had128(xh);
        xh.iter().map(|&v| f64::from(v)).sum::<f64>() as f32
    }
}

/// Outputs [c0, c0 + 128) of M activations: `out[m][i]` from the rotated inputs `xh[m]` (k floats each) and their
/// sums. Uses the AVX2 path for mul1 with k2 5 or 7 (exllamav3 sc_2.60's), the scalar one otherwise.
///
/// # Safety
/// The CPU must have AVX2 and FMA; `xh[m]` must hold `w.k` floats.
pub unsafe fn block<const M: usize>(
    w: &Exl3<'_>,
    xh: [*const f32; M],
    xsum: [f32; M],
    c0: usize,
    out: &mut [[f32; 128]; M],
) {
    debug_assert!(c0.is_multiple_of(128) && c0 + 128 <= w.n);
    unsafe {
        match (w.cb, w.k2) {
            (2, 5) => band_half::<5, M>(w, xh, c0, out),
            (2, 7) => band_half::<7, M>(w, xh, c0, out),
            _ => {
                band_scalar::<M>(w, xh, c0, out);
                // band_scalar returns the final weights' sums (any codebook): no affine term
                for (m, o) in out.iter_mut().enumerate() {
                    finish(w, o, c0, None, xsum[m]);
                }
                return;
            }
        }
    }
    for (m, o) in out.iter_mut().enumerate() {
        finish(w, o, c0, Some(()), xsum[m]);
    }
}
/// The affine mul1 term (byte sums only), the output rotation and svh.
fn finish(w: &Exl3<'_>, o: &mut [f32; 128], c0: usize, affine: Option<()>, xsum: f32) {
    if affine.is_some() {
        let kz = K_ZERO * xsum;
        for v in o.iter_mut() {
            *v = v.mul_add(K_INV, kz);
        }
    }
    had128(o);
    for (i, v) in o.iter_mut().enumerate() {
        *v *= w.svh(c0 + i);
    }
}
/// Scalar reference: the decoded weights' sums (any bitrate and codebook).
unsafe fn band_scalar<const M: usize>(
    w: &Exl3<'_>,
    xh: [*const f32; M],
    c0: usize,
    out: &mut [[f32; 128]; M],
) {
    for o in out.iter_mut() {
        o.fill(0.);
    }
    for ks in 0..w.k / 16 {
        for t in 0..8 {
            let words = w.tile_words(ks, c0 / 16 + t);
            for p in 0..256 {
                let (r, c) = step_cell(p);
                let v = value(tile_state(&words, w.k2, p), w.cb);
                for m in 0..M {
                    let x = unsafe { *xh[m].add(ks * 16 + r) };
                    out[m][t * 16 + c] = v.mul_add(x, out[m][t * 16 + c]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- the half-integer K tables (k2 3, 5, 7)
/// Per group g = 2·l + h (tile rows 2l, 2l+1, 2l+8, 2l+9, columns 8h..8h+8): the pair registers and vpermd lanes of
/// the span's first (a) and last (b) words, and per column the last word's right shift (the first's left one is
/// 32 - sr). `dsh[jj]`: window jj's offset below the span's end.
#[derive(Clone, Copy)]
struct HalfTab {
    pa: [usize; 8],
    pb: [usize; 8],
    ia: [[i32; 8]; 8],
    ib: [[i32; 8]; 8],
    sr: [[i32; 8]; 8],
    sl: [[i32; 8]; 8],
    dsh: [i32; 4],
    ok: bool,
}
/// Lane of word `w` in pair register rho ([class rho | class rho + KA], word rho + k2·m at lane m), or -1.
const fn lane_of(k2: usize, rho: usize, w: usize) -> i32 {
    let (c, m, sig) = (w % k2, w / k2, (rho + k2 / 2) % k2);
    if c == rho {
        m as i32
    } else if c == sig {
        4 + m as i32
    } else {
        -1
    }
}
const fn make_half_tab(k2: usize) -> HalfTab {
    let w = 4 * k2;
    let mut t = HalfTab {
        pa: [0; 8],
        pb: [0; 8],
        ia: [[0; 8]; 8],
        ib: [[0; 8]; 8],
        sr: [[0; 8]; 8],
        sl: [[0; 8]; 8],
        dsh: [0; 4],
        ok: true,
    };
    let mut jj = 0;
    while jj < 4 {
        t.dsh[jj] = (win_end(3, k2) - win_end(jj, k2)) as i32;
        jj += 1;
    }
    let mut g = 0;
    while g < 8 {
        let (l, h) = (g >> 1, g & 1);
        let mut wa = [0usize; 8];
        let mut wb = [0usize; 8];
        let mut c = 0;
        while c < 8 {
            let q = 32 * c + 8 * l + 4 * h;
            let e = win_end(q + 3, k2);
            let last = (e - 1) >> 5;
            let sr = (last + 1) * 32 - e;
            wb[c] = last % w;
            wa[c] = (last - 1) % w;
            t.sr[g][c] = sr as i32;
            t.sl[g][c] = 32 - sr as i32;
            let mut jj = 0;
            while jj < 4 {
                if (e - win_end(q + jj, k2)) as i32 != t.dsh[jj] || t.dsh[0] + 16 > 32 {
                    t.ok = false;
                }
                jj += 1;
            }
            c += 1;
        }
        let mut which = 0;
        while which < 2 {
            let ws = if which == 0 { wa } else { wb };
            let mut found = usize::MAX;
            let mut rho = 0;
            while rho < k2 && found == usize::MAX {
                let mut all = true;
                let mut c = 0;
                while c < 8 {
                    if lane_of(k2, rho, ws[c]) < 0 {
                        all = false;
                    }
                    c += 1;
                }
                if all {
                    found = rho;
                }
                rho += 1;
            }
            if found == usize::MAX {
                t.ok = false;
            } else {
                let mut c = 0;
                while c < 8 {
                    if which == 0 {
                        t.ia[g][c] = lane_of(k2, found, ws[c]);
                    } else {
                        t.ib[g][c] = lane_of(k2, found, ws[c]);
                    }
                    c += 1;
                }
                if which == 0 {
                    t.pa[g] = found;
                } else {
                    t.pb[g] = found;
                }
            }
            which += 1;
        }
        g += 1;
    }
    t
}
const HALF5: HalfTab = make_half_tab(5);
const HALF7: HalfTab = make_half_tab(7);
const _: () = assert!(HALF5.ok && HALF7.ok);

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// A 4x4 transpose in each 128-bit half: o[q] = lanes q (and q + 4) of r0..r3.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn t4(r0: __m256i, r1: __m256i, r2: __m256i, r3: __m256i) -> [__m256i; 4] {
    let (t0, t1) = (_mm256_unpacklo_epi32(r0, r1), _mm256_unpackhi_epi32(r0, r1));
    let (t2, t3) = (_mm256_unpacklo_epi32(r2, r3), _mm256_unpackhi_epi32(r2, r3));
    [
        _mm256_unpacklo_epi64(t0, t2),
        _mm256_unpackhi_epi64(t0, t2),
        _mm256_unpacklo_epi64(t1, t3),
        _mm256_unpackhi_epi64(t1, t3),
    ]
}
/// Words lo..lo+4 in the lower half, hi..hi+4 in the upper.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn ld2(w: *const u8, lo: usize, hi: usize) -> __m256i {
    unsafe {
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(w.add(4 * lo).cast())),
            _mm_loadu_si128(w.add(4 * hi).cast()),
        )
    }
}
/// The 8 group spans X[g] of one tile (k2 5 or 7): the 32 bits ending at each column's window 3.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn spans<const K2: usize>(w: *const u8) -> [__m256i; 8] {
    let tab = if K2 == 5 { &HALF5 } else { &HALF7 };
    let mut p = [_mm256_setzero_si256(); 7];
    unsafe {
        if K2 == 7 {
            let o = t4(ld2(w, 0, 3), ld2(w, 7, 10), ld2(w, 14, 17), ld2(w, 21, 24));
            p[..4].copy_from_slice(&o);
            p[4] = _mm256_permute2x128_si256::<0x21>(o[1], o[0]); // [4 | 0]
            p[5] = _mm256_permute2x128_si256::<0x21>(o[2], o[1]); // [5 | 1]
            p[6] = _mm256_permute2x128_si256::<0x21>(o[3], o[2]); // [6 | 2]
        } else {
            let o = t4(ld2(w, 0, 1), ld2(w, 5, 6), ld2(w, 10, 11), ld2(w, 15, 16));
            p[0] = _mm256_permute2x128_si256::<0x30>(o[0], o[1]); // [0 | 2]
            p[1] = _mm256_permute2x128_si256::<0x30>(o[1], o[2]); // [1 | 3]
            p[2] = _mm256_permute2x128_si256::<0x30>(o[2], o[3]); // [2 | 4]
            p[3] = _mm256_permute2x128_si256::<0x20>(o[3], o[0]); // [3 | 0]
            p[4] = _mm256_permute2x128_si256::<0x31>(o[3], o[0]); // [4 | 1]
        }
        let mut x = [_mm256_setzero_si256(); 8];
        for g in 0..8 {
            let a = _mm256_permutevar8x32_epi32(
                p[tab.pa[g]],
                _mm256_loadu_si256(tab.ia[g].as_ptr().cast()),
            );
            let b = _mm256_permutevar8x32_epi32(
                p[tab.pb[g]],
                _mm256_loadu_si256(tab.ib[g].as_ptr().cast()),
            );
            x[g] = _mm256_or_si256(
                _mm256_srlv_epi32(b, _mm256_loadu_si256(tab.sr[g].as_ptr().cast())),
                _mm256_sllv_epi32(a, _mm256_loadu_si256(tab.sl[g].as_ptr().cast())),
            );
        }
        x
    }
}
/// Row r, half h of a tile: its 8 mul1 byte sums as F32.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn sums<const K2: usize>(x: &[__m256i; 8], r: usize, h: usize) -> __m256 {
    let tab = if K2 == 5 { &HALF5 } else { &HALF7 };
    let (g, jj) = (2 * ((r & 7) >> 1) + h, (r & 1) + 2 * (r >> 3));
    let st = _mm256_and_si256(
        _mm256_srl_epi32(x[g], _mm_cvtsi32_si128(tab.dsh[jj])),
        _mm256_set1_epi32(0xffff),
    );
    let bytes = _mm256_mullo_epi32(st, _mm256_set1_epi32(MUL1 as i32));
    _mm256_cvtepi32_ps(_mm256_madd_epi16(
        _mm256_maddubs_epi16(bytes, _mm256_set1_epi8(1)),
        _mm256_set1_epi16(1),
    ))
}
/// The block's byte sums Σ_r xh[r]·bytesum for 8 tiles x 16 columns, M activations.
#[allow(clippy::needless_range_loop)] // tile and activation indices also address the trellis and inputs
#[target_feature(enable = "avx2,fma")]
unsafe fn band_half<const K2: usize, const M: usize>(
    w: &Exl3<'_>,
    xh: [*const f32; M],
    c0: usize,
    out: &mut [[f32; 128]; M],
) {
    let mut acc = [[[_mm256_setzero_ps(); 2]; 8]; M];
    let tc0 = c0 / 16;
    unsafe {
        for ks in 0..w.k / 16 {
            let row = w.tile(ks, tc0);
            if ks + 4 < w.k / 16 {
                // four slices on (a slice is n/16 tiles of 4·K2 words)
                let ahead = row.add(4 * (w.n / 16) * 4 * K2 * 4);
                for l in (0..8 * 4 * K2 * 4).step_by(64) {
                    _mm_prefetch::<_MM_HINT_T0>(ahead.add(l).cast());
                }
            }
            for t in 0..8 {
                let x = spans::<K2>(row.add(t * 4 * K2 * 4));
                for r in 0..16 {
                    let lo = sums::<K2>(&x, r, 0);
                    let hi = sums::<K2>(&x, r, 1);
                    for m in 0..M {
                        let v = _mm256_broadcast_ss(&*xh[m].add(ks * 16 + r));
                        acc[m][t][0] = _mm256_fmadd_ps(lo, v, acc[m][t][0]);
                        acc[m][t][1] = _mm256_fmadd_ps(hi, v, acc[m][t][1]);
                    }
                }
            }
        }
        for m in 0..M {
            for t in 0..8 {
                _mm256_storeu_ps(out[m].as_mut_ptr().add(t * 16), acc[m][t][0]);
                _mm256_storeu_ps(out[m].as_mut_ptr().add(t * 16 + 8), acc[m][t][1]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> Vec<std::path::PathBuf> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../python-tools/tests/data/exl3-exllamav3");
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "bin"))
            .collect();
        files.sort();
        files
    }
    fn le_i32(b: &[u8], at: usize) -> usize {
        i32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize
    }

    /// exllamav3's own states and W' (Project Maya's fixtures) bit for bit.
    #[test]
    fn glm53_exl3_matches_exllamav3_fixtures() {
        let files = fixtures();
        assert_eq!(files.len(), 13);
        for path in files {
            let b = std::fs::read(&path).unwrap();
            assert_eq!(&b[..4], b"EXH1");
            let (k2, ki, nj, cb, words) = (
                le_i32(&b, 4),
                le_i32(&b, 8),
                le_i32(&b, 12),
                le_i32(&b, 16),
                le_i32(&b, 20),
            );
            let tiles = ki * nj;
            let at_states = 24 + tiles * words * 2;
            let at_w = at_states + tiles * 512;
            for t in 0..tiles {
                let tile: Vec<u32> = (0..4 * k2)
                    .map(|i| {
                        u32::from_le_bytes(b[24 + t * words * 2 + 4 * i..][..4].try_into().unwrap())
                    })
                    .collect();
                for p in 0..256 {
                    let want = u16::from_le_bytes(
                        b[at_states + (t * 256 + p) * 2..][..2].try_into().unwrap(),
                    );
                    assert_eq!(
                        tile_state(&tile, k2, p),
                        u32::from(want),
                        "{path:?} tile {t} step {p}"
                    );
                }
            }
            // W' row-major [16 ki][16 nj]: decode through an Exl3 view with unit suh/svh.
            let (k, n) = (16 * ki, 16 * nj);
            let mut raw = b[24..at_states].to_vec();
            raw.extend(std::iter::repeat_n([0x00, 0x3c], k + n).flatten());
            let rows = k.next_multiple_of(128);
            if rows == k && n.is_multiple_of(128) {
                let m = Exl3::new(&raw, n, k, k2 as u8, cb as u8).unwrap();
                let w = m.inner();
                for (i, v) in w.iter().enumerate() {
                    let want = u16::from_le_bytes(b[at_w + 2 * i..][..2].try_into().unwrap());
                    assert_eq!(v.to_bits(), f16(want).to_bits(), "{path:?} weight {i}");
                }
            } else {
                for t in 0..tiles {
                    let tile: Vec<u32> = (0..4 * k2)
                        .map(|i| {
                            u32::from_le_bytes(
                                b[24 + t * words * 2 + 4 * i..][..4].try_into().unwrap(),
                            )
                        })
                        .collect();
                    let (ks, tc) = (t / nj, t % nj);
                    for p in 0..256 {
                        let (r, c) = step_cell(p);
                        let i = (ks * 16 + r) * n + tc * 16 + c;
                        let want = u16::from_le_bytes(b[at_w + 2 * i..][..2].try_into().unwrap());
                        assert_eq!(
                            value(tile_state(&tile, k2, p), cb as u8).to_bits(),
                            f16(want).to_bits()
                        );
                    }
                }
            }
        }
    }

    fn random_matrix(n: usize, k: usize, k2: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut raw: Vec<u8> = (0..n * k * k2 / 16).map(|_| next() as u8).collect();
        // suh/svh: signed FP16 scales around 1
        for _ in 0..k + n {
            let v: u16 =
                (0x3800 + (next() % 0x800) as u16) | if next() & 1 == 1 { 0x8000 } else { 0 };
            raw.extend(v.to_le_bytes());
        }
        raw
    }

    /// The AVX2 half-integer path against the full F64 product with the exactly decoded weights.
    #[test]
    fn glm53_exl3_avx2_matches_reference() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for (k2, n, k) in [(5, 256, 384), (7, 128, 256), (5, 2048, 4096), (4, 128, 128)] {
            let raw = random_matrix(n, k, k2, 0x9e37_79b9 + k2 as u64);
            let m = Exl3::new(&raw, n, k, k2 as u8, 2).unwrap();
            let inner = m.inner();
            let xs: Vec<Vec<f32>> = (0..3)
                .map(|j| {
                    (0..k)
                        .map(|i| ((i * 7 + j * 13) % 29) as f32 / 29. - 0.5)
                        .collect()
                })
                .collect();
            let mut xh = vec![vec![0.; k]; 3];
            let mut sums = [0.; 3];
            for j in 0..3 {
                sums[j] = m.prepare(&xs[j], &mut xh[j]);
            }
            for c0 in (0..n).step_by(128) {
                let mut out = [[0.; 128]; 3];
                unsafe {
                    block::<3>(
                        &m,
                        [xh[0].as_ptr(), xh[1].as_ptr(), xh[2].as_ptr()],
                        sums,
                        c0,
                        &mut out,
                    )
                };
                for j in 0..3 {
                    // exact: y = H(xh · W') · svh in F64
                    let mut y = vec![0f64; 128];
                    for (r, &xv) in xh[j].iter().enumerate() {
                        for c in 0..128 {
                            y[c] += f64::from(xv) * f64::from(inner[r * n + c0 + c]);
                        }
                    }
                    let mut y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
                    had128(&mut y32);
                    let scale = y32.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    for c in 0..128 {
                        let want = y32[c] * m.svh(c0 + c);
                        let got = out[j][c];
                        // the CPU skips the FP16 rounding of each weight: tolerance well above it, far below a bug
                        assert!(
                            (got - want).abs() <= 2e-3 * scale * m.svh(c0 + c).abs() + 1e-6,
                            "k2 {k2} n {n} k {k} c {} got {got} want {want}",
                            c0 + c
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod speed {
    use super::*;

    /// One K2.5 expert (gate, up: 2048 x 4096; down: 4096 x 2048) on one core, M = 1.
    #[test]
    #[ignore = "timing"]
    fn glm53_exl3_cpu_expert_speed() {
        let raw = |n: usize, k: usize| -> Vec<u8> {
            let mut v: Vec<u8> = (0..n * k * 5 / 16)
                .map(|i| ((i * 2654435761usize) >> 7) as u8)
                .collect();
            v.extend(std::iter::repeat_n([0x00u8, 0x3c], n + k).flatten());
            v
        };
        let (g, d) = (raw(2048, 4096), raw(4096, 2048));
        let (gm, dm) = (
            Exl3::new(&g, 2048, 4096, 5, 2).unwrap(),
            Exl3::new(&d, 4096, 2048, 5, 2).unwrap(),
        );
        let x = vec![0.01f32; 4096];
        let mut xh = vec![0.; 4096];
        let s = gm.prepare(&x, &mut xh);
        let mut out = [[0.; 128]; 1];
        for _ in 0..2 {
            let t = std::time::Instant::now();
            for _ in 0..2 {
                for c0 in (0..2048).step_by(128) {
                    unsafe { block::<1>(&gm, [xh.as_ptr()], [s], c0, &mut out) };
                }
            }
            for c0 in (0..4096).step_by(128) {
                unsafe { block::<1>(&dm, [xh.as_ptr()], [s], c0, &mut out) };
            }
            println!(
                "glm53_exl3 one core: {:.3} ms an expert (K2.5, M 1)",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

#[cfg(test)]
mod pool {
    use super::*;
    use crate::glm53::cpu::{Format, Matrix, Pool, Segment, bf16, swiglu};

    fn raw(n: usize, k: usize, k2: usize, seed: usize) -> Vec<u8> {
        let mut v: Vec<u8> = (0..n * k * k2 / 16)
            .map(|i| ((i + seed).wrapping_mul(2_654_435_761) >> 11) as u8)
            .collect();
        for i in 0..n + k {
            let h: u16 = (0x3a00 + ((i * 37 + seed) % 0x400) as u16)
                | if (i + seed).is_multiple_of(3) {
                    0x8000
                } else {
                    0
                };
            v.extend(h.to_le_bytes());
        }
        v
    }

    /// Pool segments (EXL3 gate/up SwiGLU, then down) against direct `block` calls, 1..4 activations.
    #[test]
    fn glm53_exl3_pool_gate_up_and_down() -> Result<()> {
        let mut pool = Pool::new(hayate_core::machine::MAIN.cpu.compute)?;
        let (g, u, d) = (
            raw(2048, 4096, 5, 1),
            raw(2048, 4096, 5, 2),
            raw(4096, 2048, 7, 3),
        );
        let (gv, uv, dv) = (
            Exl3::new(&g, 2048, 4096, 5, 2)?,
            Exl3::new(&u, 2048, 4096, 5, 2)?,
            Exl3::new(&d, 4096, 2048, 7, 2)?,
        );
        for rows in 1..=4usize {
            let x: Vec<f32> = (0..rows * 4096)
                .map(|i| ((i * 31) % 97) as f32 / 97. - 0.5)
                .collect();
            let (mut xg, mut xu) = (vec![0.; rows * 4096], vec![0.; rows * 4096]);
            let (mut sg, mut su) = ([0.; 4], [0.; 4]);
            for m in 0..rows {
                sg[m] = gv.prepare(&x[m * 4096..][..4096], &mut xg[m * 4096..][..4096]);
                su[m] = uv.prepare(&x[m * 4096..][..4096], &mut xu[m * 4096..][..4096]);
            }
            let mut act = vec![0.; rows * 2048];
            let gate = Matrix::with_format(&g, 2048, 4096, Format::Exl3 { k2: 5 })?;
            let up = Matrix::with_format(&u, 2048, 4096, Format::Exl3 { k2: 5 })?;
            assert!(Segment::gate_up(gate, up, &xg, &xg, &mut act, rows).is_err());
            pool.run(&[Segment::exl3_gate_up(
                gate, up, &xg, sg, &xu, su, &mut act, rows,
            )?])?;
            for m in 0..rows {
                for c0 in (0..2048).step_by(128) {
                    let (mut a, mut b) = ([[0.; 128]; 1], [[0.; 128]; 1]);
                    unsafe {
                        block::<1>(&gv, [xg[m * 4096..].as_ptr()], [sg[m]], c0, &mut a);
                        block::<1>(&uv, [xu[m * 4096..].as_ptr()], [su[m]], c0, &mut b);
                    }
                    for i in 0..128 {
                        let want = swiglu(bf16(a[0][i]), bf16(b[0][i]));
                        assert_eq!(
                            act[m * 2048 + c0 + i].to_bits(),
                            want.to_bits(),
                            "rows {rows} m {m} {}",
                            c0 + i
                        );
                    }
                }
            }
            let mut ad = vec![0.; rows * 2048];
            let mut sd = [0.; 4];
            for m in 0..rows {
                sd[m] = dv.prepare(&act[m * 2048..][..2048], &mut ad[m * 2048..][..2048]);
            }
            let mut y = vec![0.; rows * 4096];
            let down = Matrix::with_format(&d, 4096, 2048, Format::Exl3 { k2: 7 })?;
            pool.run(&[Segment::exl3_plain(down, &ad, sd, &mut y, rows)?])?;
            for m in 0..rows {
                for c0 in (0..4096).step_by(128) {
                    let mut a = [[0.; 128]; 1];
                    unsafe { block::<1>(&dv, [ad[m * 2048..].as_ptr()], [sd[m]], c0, &mut a) };
                    for i in 0..128 {
                        // M > 1 sums each activation's FMAs in the same order: identical bits
                        assert_eq!(y[m * 4096 + c0 + i].to_bits(), a[0][i].to_bits());
                    }
                }
            }
        }
        Ok(())
    }

    /// One K2.5 expert per pool dispatch pair (the decode lane's shape), M = 1.
    #[test]
    #[ignore = "timing"]
    fn glm53_exl3_pool_expert_speed() -> Result<()> {
        let mut pool = Pool::new(hayate_core::machine::MAIN.cpu.compute)?;
        let (g, u, d) = (
            raw(2048, 4096, 5, 1),
            raw(2048, 4096, 5, 2),
            raw(4096, 2048, 5, 3),
        );
        let gate = Matrix::with_format(&g, 2048, 4096, Format::Exl3 { k2: 5 })?;
        let up = Matrix::with_format(&u, 2048, 4096, Format::Exl3 { k2: 5 })?;
        let down = Matrix::with_format(&d, 4096, 2048, Format::Exl3 { k2: 5 })?;
        let (xg, xu, ad) = (vec![0.01; 4096], vec![0.01; 4096], vec![0.01; 2048]);
        let (mut act, mut y) = (vec![0.; 2048], vec![0.; 4096]);
        for round in 0..3 {
            let t = std::time::Instant::now();
            let reps = 50;
            for _ in 0..reps {
                pool.run(&[Segment::exl3_gate_up(
                    gate, up, &xg, [1.; 4], &xu, [1.; 4], &mut act, 1,
                )?])?;
                pool.run(&[Segment::exl3_plain(down, &ad, [1.; 4], &mut y, 1)?])?;
            }
            println!(
                "glm53_exl3 pool ({} threads) round {round}: {:.3} ms an expert (K2.5, M 1)",
                pool.threads(),
                t.elapsed().as_secs_f64() * 1e3 / reps as f64
            );
        }
        Ok(())
    }
}
