//! Host-side NVFP4 test data and the f64 reference for the GEMV bench.

pub fn e2m1(code: u8) -> f64 {
    const MAG: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[usize::from(code & 7)];
    if code & 8 != 0 { -v } else { v }
}

pub fn e4m3(code: u8) -> f64 {
    let (s, e, m) = (code >> 7, i32::from((code >> 3) & 15), f64::from(code & 7));
    assert!(!(e == 15 && m == 7.0), "e4m3 NaN");
    let v = if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    if s != 0 { -v } else { v }
}

pub fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let round = 0x7fff + ((b >> 16) & 1);
    ((b + round) >> 16) as u16
}

pub fn bf16_value(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// One random NVFP4 matrix [n][k] with finite scales, and a bf16 activation vector.
pub struct Case {
    n: usize,
    k: usize,
    codes: Vec<u8>,
    scales: Vec<u8>,
    pub global_scale: f32,
    pub x_bf16: Vec<u16>,
}

impl Case {
    pub fn random(n: usize, k: usize, seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let codes = (0..n * k / 2).map(|_| r.next() as u8).collect();
        // Exponents 4..=9 keep scales in [2^-3, 2^2], mantissas anything.
        let scales = (0..n * k / 16)
            .map(|_| (((4 + r.next() % 6) << 3) | (r.next() % 8)) as u8)
            .collect();
        let x_bf16 = (0..k)
            .map(|_| bf16_bits((r.next() % 20001) as f32 / 10000.0 - 1.0))
            .collect();
        Self {
            n,
            k,
            codes,
            scales,
            global_scale: 0.0123,
            x_bf16,
        }
    }

    /// Device layout: codes [n][k/2] followed by scales [n][k/16].
    pub fn packed_with_scales(&self) -> Vec<u8> {
        [self.codes.as_slice(), self.scales.as_slice()].concat()
    }

    /// (max relative error, row) where the error is scaled by sum |w x| of the row, so
    /// cancellation in the exact sum cannot inflate it.
    pub fn compare(&self, got: &[f32]) -> (f64, usize) {
        let mut worst = (0.0, 0);
        for (row, &g) in got.iter().enumerate().take(self.n) {
            let (mut sum, mut mag) = (0.0, 0.0);
            for kk in 0..self.k {
                let byte = self.codes[row * self.k / 2 + kk / 2];
                let code = if kk % 2 == 0 { byte & 15 } else { byte >> 4 };
                let w = e2m1(code) * e4m3(self.scales[row * self.k / 16 + kk / 16]);
                let t = w * bf16_value(self.x_bf16[kk]);
                sum += t;
                mag += t.abs();
            }
            let gs = f64::from(self.global_scale);
            let rel = (f64::from(g) - sum * gs).abs() / (mag * gs + 1e-30);
            if rel > worst.0 {
                worst = (rel, row);
            }
        }
        worst
    }
}
