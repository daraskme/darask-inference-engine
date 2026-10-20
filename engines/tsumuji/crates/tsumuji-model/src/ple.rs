//! PLE (per-layer n-gram embedding) hashing, bit-exact with transformers `qwen4_exp`
//! (`Qwen4ExpTextNGramEmbedding`). One PLE module, on layer 1 (0-based), so
//! `ple_layer_index` is 0.
//!
//! For each token, head h in 0..8 hashes the 2-gram (t0, t1) and head h in 8..16 the 3-gram
//! (t0, t1, t2), where t0 is the current token and t1, t2 the previous ones:
//!   row = ((t0*m0) ^ (t1*m1) [^ (t2*m2)]) % size[h] + offset[h]
//! The products never overflow: t < vocab and every multiplier < (2^63 - 1) / vocab.

use crate::shape::{EOS, PLE_HEADS, VOCAB};

pub const SEED: u64 = 1234;
pub const NGRAM: usize = 3;
pub const HEADS_PER_NGRAM: usize = 8;
pub const VOCAB_BASE: u64 = 20_000_000;
pub const VOCAB_DIVISOR: u64 = 128;

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const PRIME_1: u64 = 10_007;

pub fn splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

/// `_build_layer_multipliers(vocab, ngram_size, ple_layer_index, seed)`.
pub fn layer_multipliers(vocab: u64, ple_layer_index: u64, seed: u64) -> [u64; NGRAM] {
    let half = (((1u64 << 63) - 1) / vocab.max(1) / 2).max(1);
    let base = seed.wrapping_add(PRIME_1.wrapping_mul(ple_layer_index));
    std::array::from_fn(|i| {
        2 * (splitmix64(base.wrapping_add(GAMMA.wrapping_mul(i as u64 + 1))) % half) + 1
    })
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v.is_multiple_of(2) {
        return v == 2;
    }
    let mut d = 3;
    while d * d <= v {
        if v.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// The hash tables of one PLE module: multipliers, per-head prime sizes and row offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tables {
    pub mult: [u64; NGRAM],
    pub size: [u64; PLE_HEADS],
    pub offset: [u64; PLE_HEADS],
    /// Rows of the embedding table, padded to a multiple of 128 (the checkpoint's row count).
    pub padded_rows: u64,
}

impl Tables {
    /// The tables of Qwen3.8-Flash-Next (one PLE module).
    pub fn qwen38() -> Self {
        Self::new(VOCAB as u64, VOCAB_BASE, 0, SEED)
    }

    /// `vocab` is the token vocabulary, `base` the config's `ngram_vocab_size_base`.
    pub fn new(vocab: u64, base: u64, ple_layer_index: u64, seed: u64) -> Self {
        // Head g (global index ple_layer_index * 16 + h) uses the (g + 1)-th prime >= base.
        let mut primes = Vec::new();
        let mut p = base - 1;
        while primes.len() < ((ple_layer_index + 1) as usize) * PLE_HEADS {
            p += 1;
            if is_prime(p) {
                primes.push(p);
            }
        }
        let first = ple_layer_index as usize * PLE_HEADS;
        let size: [u64; PLE_HEADS] = std::array::from_fn(|h| primes[first + h]);
        let mut offset = [0; PLE_HEADS];
        let mut total = 0;
        for h in 0..PLE_HEADS {
            offset[h] = total;
            total += size[h];
        }
        Self {
            mult: layer_multipliers(vocab, ple_layer_index, seed),
            size,
            offset,
            padded_rows: total.div_ceil(VOCAB_DIVISOR) * VOCAB_DIVISOR,
        }
    }

    /// Embedding rows for one token given its n-gram context (see `Context`).
    pub fn rows(&self, t0: u32, t1: u32, t2: u32) -> [u64; PLE_HEADS] {
        let bi = (u64::from(t0) * self.mult[0]) ^ (u64::from(t1) * self.mult[1]);
        let tri = bi ^ (u64::from(t2) * self.mult[2]);
        std::array::from_fn(|h| {
            let m = if h < HEADS_PER_NGRAM { bi } else { tri };
            m % self.size[h] + self.offset[h]
        })
    }
}

/// The two previous tokens seen by the n-gram hash. Starts as (EOS, EOS); after an EOS the
/// tokens before it are replaced by EOS (`_shift_right_ignore_eos`), so a new document never
/// hashes across the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Context {
    pub t1: u32,
    pub t2: u32,
    pub eos: u32,
}

impl Default for Context {
    fn default() -> Self {
        Self::new(EOS)
    }
}

impl Context {
    /// A fresh sequence for a model whose EOS id is `eos`.
    pub fn new(eos: u32) -> Self {
        Self {
            t1: eos,
            t2: eos,
            eos,
        }
    }

    /// Advances past token `t` (the one just hashed).
    pub fn push(&mut self, t: u32) {
        self.t2 = if t == self.eos { self.eos } else { self.t1 };
        self.t1 = t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight port of the reference's full-sequence path (`_shift_right_ignore_eos` over
    /// [EOS, EOS] + ids), to check the incremental `Context` against.
    fn reference_context(ids: &[u32], p: usize, shift: usize) -> u32 {
        let hist: Vec<u32> = [EOS, EOS].iter().chain(ids).copied().collect();
        let q = p + 2;
        let prev_eos = (0..q)
            .rev()
            .find(|&i| hist[i] == EOS)
            .map_or(-1, |i| i as i64);
        let pos_in_segment = q as i64 - (prev_eos + 1);
        if pos_in_segment >= shift as i64 && q >= shift {
            hist[q - shift]
        } else {
            EOS
        }
    }

    #[test]
    fn context_matches_reference_shift() {
        let ids = [5, 9, EOS, 7, 8, EOS, EOS, 3, 4, 6, EOS, 11];
        let mut ctx = Context::default();
        for (p, &t) in ids.iter().enumerate() {
            assert_eq!(ctx.t1, reference_context(&ids, p, 1), "t1 at {p}");
            assert_eq!(ctx.t2, reference_context(&ids, p, 2), "t2 at {p}");
            ctx.push(t);
        }
    }

    #[test]
    fn tables_shape() {
        let t = Tables::qwen38();
        assert!(t.size.windows(2).all(|w| w[0] < w[1]));
        assert!(t.size[0] >= VOCAB_BASE);
        assert!(
            t.mult
                .iter()
                .all(|m| m % 2 == 1 && *m < (1u64 << 63) / VOCAB as u64)
        );
        // 16 heads of ~20M rows, 160 values each: the 51.2B-parameter table.
        assert!((320_000_000..321_000_000).contains(&t.padded_rows));
        let r = t.rows(VOCAB as u32 - 1, VOCAB as u32 - 1, VOCAB as u32 - 1);
        assert!(
            r.iter()
                .zip(&t.offset)
                .zip(&t.size)
                .all(|((r, o), s)| r >= o && *r < o + s)
        );
    }
}
