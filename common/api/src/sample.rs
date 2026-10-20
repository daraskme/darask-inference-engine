//! Host-side sampling from the last token's logits: penalties, temperature, top-k, top-p,
//! min-p, with a seeded RNG. (Device-side sampling comes with M3.)

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// 0 = greedy.
    pub temperature: f32,
    /// 1 = off.
    pub top_p: f32,
    /// 0 = off.
    pub top_k: usize,
    /// 0 = off.
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// 1 = off (vLLM's definition: prompt and output tokens).
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

impl Params {
    pub fn penalties(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || self.repetition_penalty != 1.0
    }

    /// Plain argmax: the device's greedy token can be used without copying the logits.
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0 && !self.penalties()
    }
}

/// SplitMix64.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Debug)]
pub struct Sampler {
    p: Params,
    rng: Rng,
    counts: HashMap<u32, u32>,
    seen: HashSet<u32>,
    cand: Vec<(f32, u32)>,
    probs: Vec<f64>,
}

/// Logits more than this many temperature units below the maximum carry < e^-30 of the mass each.
const CUTOFF: f32 = 30.0;

/// top-k up to this size is kept in one pass over the logits (no candidate list of the vocab).
const SMALL_K: usize = 256;

/// The k largest logits, sorted by (descending logit, ascending id), in one pass. Chunks whose
/// maximum cannot enter the list are skipped without per-element branches.
fn top_k_into(logits: &[f32], k: usize, out: &mut Vec<(f32, u32)>) {
    let mut thr = f32::NEG_INFINITY;
    for (c, chunk) in logits.chunks(64).enumerate() {
        // An OR of comparisons vectorizes (a max over floats does not, for NaN's sake).
        if out.len() == k && !chunk.iter().fold(false, |a, &l| a | (l > thr)) {
            continue;
        }
        for (j, &l) in chunk.iter().enumerate() {
            // Equal to the threshold: the earlier id stays (ids arrive in ascending order).
            if out.len() == k && l <= thr {
                continue;
            }
            let pos = out.partition_point(|e| e.0 >= l);
            out.insert(pos, (l, (c * 64 + j) as u32));
            out.truncate(k);
            if out.len() == k {
                thr = out[k - 1].0;
            }
        }
    }
}

impl Sampler {
    pub fn new(p: &Params, prompt: &[u32]) -> Self {
        let seen = if p.repetition_penalty != 1.0 {
            prompt.iter().copied().collect()
        } else {
            HashSet::new()
        };
        Self {
            p: p.clone(),
            rng: Rng::new(p.seed),
            counts: HashMap::new(),
            seen,
            cand: Vec::new(),
            probs: Vec::new(),
        }
    }

    /// Records a generated token (for the penalties).
    pub fn accept(&mut self, tok: u32) {
        if self.p.penalties() {
            *self.counts.entry(tok).or_insert(0) += 1;
            if self.p.repetition_penalty != 1.0 {
                self.seen.insert(tok);
            }
        }
    }

    fn penalize(&self, logits: &mut [f32]) {
        let r = self.p.repetition_penalty;
        if r != 1.0 {
            for &t in &self.seen {
                if let Some(l) = logits.get_mut(t as usize) {
                    *l = if *l > 0.0 { *l / r } else { *l * r };
                }
            }
        }
        if self.p.presence_penalty != 0.0 || self.p.frequency_penalty != 0.0 {
            for (&t, &c) in &self.counts {
                if let Some(l) = logits.get_mut(t as usize) {
                    *l -= self.p.presence_penalty + self.p.frequency_penalty * c as f32;
                }
            }
        }
    }

    /// The next token from `logits` (modified in place by the penalties).
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        self.penalize(logits);
        if self.p.temperature <= 0.0 {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            return logits.iter().position(|&l| l == max).unwrap_or(0) as u32;
        }
        let t = self.p.temperature;
        let cand = &mut self.cand;
        cand.clear();
        // Candidates in descending logit, then ascending id (a total order: deterministic).
        let k = self.p.top_k;
        // Mass of the whole candidate set when only its head gets sorted.
        let mut mass = None;
        if k > 0 && k <= SMALL_K {
            top_k_into(logits, k, cand);
        } else {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let cut = max - CUTOFF * t;
            cand.extend(
                logits
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| **l >= cut)
                    .map(|(i, &l)| (l, i as u32)),
            );
            let order = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
            if k > 0 && cand.len() > k {
                cand.select_nth_unstable_by(k - 1, order);
                cand.truncate(k);
            }
            let z: f64 = cand
                .iter()
                .map(|&(l, _)| f64::from(((l - max) / t).exp()))
                .sum();
            if self.p.top_p >= 1.0 && self.p.min_p <= 0.0 {
                // The whole distribution: no order needed, draw in id order.
                let mut r = self.rng.f64() * z;
                for &(l, id) in cand.iter() {
                    r -= f64::from(((l - max) / t).exp());
                    if r < 0.0 {
                        return id;
                    }
                }
                return cand.last().map_or(0, |c| c.1);
            }
            // Sort only a head that covers top_p of the mass (grown when it does not).
            let need = f64::from(self.p.top_p) * z;
            let mut take = 256.min(cand.len());
            loop {
                if take < cand.len() {
                    cand.select_nth_unstable_by(take - 1, order);
                }
                cand[..take].sort_unstable_by(order);
                let head: f64 = cand[..take]
                    .iter()
                    .map(|&(l, _)| f64::from(((l - max) / t).exp()))
                    .sum();
                if head >= need || take == cand.len() {
                    cand.truncate(take);
                    break;
                }
                take = (take * 8).min(cand.len());
            }
            mass = Some(z);
        }
        let max = cand[0].0;
        let probs = &mut self.probs;
        probs.clear();
        probs.extend(cand.iter().map(|&(l, _)| f64::from(((l - max) / t).exp())));
        let sum: f64 = mass.unwrap_or_else(|| probs.iter().sum());
        let mut keep = cand.len();
        if self.p.top_p < 1.0 {
            let mut acc = 0.0;
            for (i, p) in probs.iter().enumerate() {
                acc += p / sum;
                if acc >= f64::from(self.p.top_p) {
                    keep = i + 1;
                    break;
                }
            }
        }
        if self.p.min_p > 0.0 {
            let floor = f64::from(self.p.min_p) * probs[0];
            keep = keep.min(probs.iter().take_while(|&&p| p >= floor).count().max(1));
        }
        let total: f64 = probs[..keep].iter().sum();
        let mut r = self.rng.f64() * total;
        for i in 0..keep {
            r -= probs[i];
            if r < 0.0 {
                return cand[i].1;
            }
        }
        cand[keep - 1].1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution() {
        // logits ln(1), ln(2), ln(3), ln(4) at temperature 1: probabilities 0.1 .. 0.4.
        let base: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|v| v.ln()).collect();
        let p = Params {
            seed: 7,
            ..Default::default()
        };
        let mut s = Sampler::new(&p, &[]);
        let mut hist = [0usize; 4];
        for _ in 0..40_000 {
            hist[s.sample(&mut base.clone()) as usize] += 1;
        }
        for (i, h) in hist.iter().enumerate() {
            let want = (i + 1) as f64 / 10.0;
            assert!((*h as f64 / 40_000.0 - want).abs() < 0.01, "{hist:?}");
        }
        // top_k 2 keeps ids 3 and 2 (4:3), top_p 0.35 keeps id 3 only.
        let mut s = Sampler::new(
            &Params {
                top_k: 2,
                seed: 1,
                ..Default::default()
            },
            &[],
        );
        assert!((0..1000).all(|_| s.sample(&mut base.clone()) >= 2));
        let mut s = Sampler::new(
            &Params {
                top_p: 0.35,
                seed: 1,
                ..Default::default()
            },
            &[],
        );
        assert!((0..1000).all(|_| s.sample(&mut base.clone()) == 3));
        let mut s = Sampler::new(
            &Params {
                temperature: 0.0,
                ..Default::default()
            },
            &[],
        );
        assert_eq!(s.sample(&mut base.clone()), 3);
    }

    #[test]
    fn top_k_matches_sort() {
        let mut r = Rng::new(3);
        // Coarse values so that ties occur.
        let logits: Vec<f32> = (0..248_320)
            .map(|_| (r.f64() * 400.0).floor() as f32 / 10.0)
            .collect();
        let mut all: Vec<(f32, u32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, i as u32))
            .collect();
        all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for k in [1, 20, 256] {
            let mut got = Vec::new();
            let t = std::time::Instant::now();
            top_k_into(&logits, k, &mut got);
            eprintln!(
                "top-{k} of {}: {:.0} µs",
                logits.len(),
                t.elapsed().as_secs_f64() * 1e6
            );
            assert_eq!(got, all[..k]);
        }
        // Whole-vocabulary paths (prints the time per sample).
        for (top_k, top_p) in [(20, 0.95), (0, 0.95), (0, 1.0)] {
            let mut s = Sampler::new(
                &Params {
                    top_k,
                    top_p,
                    temperature: 0.8,
                    seed: 5,
                    ..Default::default()
                },
                &[],
            );
            let t = std::time::Instant::now();
            for _ in 0..10 {
                s.sample(&mut logits.clone());
            }
            eprintln!(
                "sample top_k {top_k} top_p {top_p}: {:.0} µs",
                t.elapsed().as_secs_f64() * 1e5
            );
        }
    }
}
