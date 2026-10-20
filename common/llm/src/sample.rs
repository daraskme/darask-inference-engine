//! Host sampler over BF16 logits, for engines that download the head output.
//! Order follows vLLM: repetition penalty, temperature, min-p, top-k, top-p.
use crate::SamplingParams;
use hayate_core::{Error, Result};

/// Tokens more than this many nats (after temperature) below the best one are
/// never candidates; their total mass is below 155K * e^-25 ~ 2e-6 of the best.
const CUT: f32 = 25.;

#[derive(Debug, Default)]
pub struct Sampler {
    state: u64,
    values: Vec<f32>,
    candidates: Vec<(f32, u32)>,
    scratch: Vec<(f32, u32)>,
    #[cfg(test)]
    comparison_sort: bool,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        let mut s = Self::default();
        s.reseed(seed);
        s
    }
    pub fn reseed(&mut self, seed: u64) {
        // splitmix64 so nearby seeds give unrelated streams; never zero.
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        self.state = (z ^ (z >> 31)).max(1);
    }
    /// Uniform in [0, 1) from xorshift64*.
    fn uniform(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        (x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 * (1. / (1u64 << 53) as f64)
    }
    /// `logits` are little-endian BF16 bit patterns. `seen` holds every distinct
    /// token of the prompt and the reply so far; `penalty` 1 disables it.
    pub fn sample(
        &mut self,
        logits: &[u16],
        params: &SamplingParams,
        penalty: f32,
        seen: &[u32],
    ) -> Result<u32> {
        if logits.is_empty()
            || !params.temperature.is_finite()
            || params.temperature < 0.
            || !(params.top_p > 0. && params.top_p <= 1.)
            || !(0. ..1.).contains(&params.min_p)
            || !(penalty.is_finite() && penalty > 0.)
        {
            return Err(Error::InvalidPlan("sampling parameters".into()));
        }
        self.values.clear();
        self.values
            .extend(logits.iter().map(|&b| f32::from_bits(u32::from(b) << 16)));
        if penalty != 1. {
            for &id in seen {
                if let Some(v) = self.values.get_mut(id as usize) {
                    *v = if *v > 0. { *v / penalty } else { *v * penalty };
                }
            }
        }
        // Lowest ID wins ties, like the GPU greedy kernels.
        let (best, max) =
            self.values
                .iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                });
        if !max.is_finite() {
            return Err(Error::InvalidShape("nonfinite logits".into()));
        }
        if params.temperature == 0. {
            return Ok(best as u32);
        }
        let scale = 1. / params.temperature;
        let cut = if params.min_p > 0. {
            params.min_p.ln().max(-CUT)
        } else {
            -CUT
        };
        self.candidates.clear();
        for (id, &v) in self.values.iter().enumerate() {
            let z = (v - max) * scale;
            if z >= cut {
                self.candidates.push((z.exp(), id as u32));
            }
        }
        let k = params.top_k as usize;
        if k > 0 && k < self.candidates.len() {
            self.candidates
                .select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            self.candidates.truncate(k);
        }
        if params.top_p < 1. {
            // DeepInfer uses radix processing for vocabulary filtering. Here
            // a stable CPU radix order retains Hayate's exact old CDF, including
            // ascending token IDs on ties and therefore fixed-seed outputs.
            // select_nth above scrambles ties: use the comparator on that small
            // top-k subset. Unpartitioned candidates are already in ID order.
            let radix = self.candidates.len() >= 2048 && (k == 0 || k >= self.values.len());
            #[cfg(test)]
            let radix = radix && !self.comparison_sort;
            if radix {
                radix_order(&mut self.candidates, &mut self.scratch);
            } else {
                self.candidates
                    .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            }
            let total: f64 = self.candidates.iter().map(|c| f64::from(c.0)).sum();
            let mut mass = 0.;
            let mut keep = self.candidates.len();
            for (i, c) in self.candidates.iter().enumerate() {
                mass += f64::from(c.0);
                if mass >= total * f64::from(params.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            self.candidates.truncate(keep);
        }
        let total: f64 = self.candidates.iter().map(|c| f64::from(c.0)).sum();
        let draw = self.uniform() * total;
        let mut mass = 0.;
        for c in &self.candidates {
            mass += f64::from(c.0);
            if draw < mass {
                return Ok(c.1);
            }
        }
        Ok(self.candidates.last().map_or(best as u32, |c| c.1))
    }
}

/// Positive finite f32 probabilities have the same unsigned bit order as
/// their numeric order. Stable byte passes preserve ascending IDs on ties.
/// Four linear scans replace O(vocabulary * log(vocabulary)) comparisons.
fn radix_order(values: &mut Vec<(f32, u32)>, scratch: &mut Vec<(f32, u32)>) {
    scratch.resize(values.len(), (0., 0));
    for shift in [0, 8, 16, 24] {
        let mut counts = [0usize; 256];
        for &(p, _) in values.iter() {
            counts[((!p.to_bits() >> shift) & 255) as usize] += 1;
        }
        if counts.contains(&values.len()) {
            continue;
        }
        let mut offset = 0;
        for count in &mut counts {
            let n = *count;
            *count = offset;
            offset += n;
        }
        for &item in values.iter() {
            let bucket = ((!item.0.to_bits() >> shift) & 255) as usize;
            scratch[counts[bucket]] = item;
            counts[bucket] += 1;
        }
        std::mem::swap(values, scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bf16(v: &[f32]) -> Vec<u16> {
        v.iter().map(|x| (x.to_bits() >> 16) as u16).collect()
    }
    fn params(temperature: f32, top_p: f32, top_k: u32, min_p: f32) -> SamplingParams {
        SamplingParams {
            temperature,
            top_p,
            top_k,
            min_p,
            max_tokens: 1,
        }
    }
    fn histogram(logits: &[u16], p: &SamplingParams, n: usize) -> Vec<usize> {
        let mut s = Sampler::new(7);
        let mut counts = vec![0; logits.len()];
        for _ in 0..n {
            counts[s.sample(logits, p, 1., &[]).unwrap() as usize] += 1;
        }
        counts
    }
    #[test]
    fn greedy_ties_and_penalty() {
        let logits = bf16(&[1., 3., 3., -2.]);
        let mut s = Sampler::new(1);
        assert_eq!(
            s.sample(&logits, &params(0., 1., 0, 0.), 1., &[]).unwrap(),
            1
        );
        // 3 / 2 = 1.5 < 3: the penalty moves the argmax to token 2.
        assert_eq!(
            s.sample(&logits, &params(0., 1., 0, 0.), 2., &[1]).unwrap(),
            2
        );
        // Negative logits are multiplied: -2 * 2 = -4 stays last.
        assert!(s.sample(&logits, &params(0., 1., 0, 0.), 1., &[]).is_ok());
        assert!(s.sample(&logits, &params(-1., 1., 0, 0.), 1., &[]).is_err());
        assert!(s.sample(&logits, &params(1., 0., 0, 0.), 1., &[]).is_err());
        assert!(
            s.sample(&bf16(&[f32::NAN; 2]), &params(1., 1., 0, 0.), 1., &[])
                .is_err()
        );
    }
    #[test]
    fn filters_follow_min_p_top_k_top_p() {
        // Probabilities at T=1: e^0, e^-1, e^-2, e^-3, e^-30 (normalized).
        let logits = bf16(&[0., -1., -2., -3., -30.]);
        let all = histogram(&logits, &params(1., 1., 0, 0.), 20000);
        assert!(all[..4].iter().all(|&c| c > 0) && all[4] == 0);
        let ratio = all[0] as f64 / all[1] as f64;
        assert!((ratio - std::f64::consts::E).abs() < 0.25, "{ratio}");
        // min_p 0.3 keeps p >= 0.3 * p_max: e^-1 = 0.37 stays, e^-2 = 0.14 goes.
        let min_p = histogram(&logits, &params(1., 1., 0, 0.3), 5000);
        assert!(min_p[0] > 0 && min_p[1] > 0 && min_p[2] == 0 && min_p[3] == 0);
        let top_k = histogram(&logits, &params(1., 1., 1, 0.), 100);
        assert_eq!(top_k[0], 100);
        // Top-p 0.7: p0 = 0.644, p0 + p1 = 0.881 >= 0.7, so two tokens survive.
        let top_p = histogram(&logits, &params(1., 0.7, 0, 0.), 5000);
        assert!(top_p[0] > 0 && top_p[1] > 0 && top_p[2] == 0);
    }
    #[test]
    fn seeds_are_reproducible() {
        let logits = bf16(&(0..1000).map(|i| (i % 17) as f32 * 0.1).collect::<Vec<_>>());
        let p = params(1., 0.95, 0, 0.);
        let run = |seed| {
            let mut s = Sampler::new(seed);
            (0..32)
                .map(|_| s.sample(&logits, &p, 1., &[]).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    #[test]
    fn radix_matches_comparison_cdf_and_seeded_tokens() {
        // Include tied probabilities, sparse tails, and penalties that turn
        // originally tied BF16 logits into different f32 values.
        for n in [2047, 2048, 8192, 154880] {
            let logits = bf16(
                &(0..n)
                    .map(|i| ((i * 7919 % 4093) as f32 - 2046.) / 128.)
                    .collect::<Vec<_>>(),
            );
            for (temperature, top_p, top_k, min_p) in [
                (1., 0.95, 0, 0.),
                (0.2, 0.8, 0, 0.),
                (2., 0.99, 0, 0.01),
                (1., 0.95, 64, 0.),
                (1., 0.95, 154880, 0.),
                (1., 1., 0, 0.),
            ] {
                let p = params(temperature, top_p, top_k, min_p);
                let mut fast = Sampler::new(42);
                let mut reference = Sampler::new(42);
                reference.comparison_sort = true;
                for _ in 0..8 {
                    assert_eq!(
                        fast.sample(&logits, &p, 1.2, &[0, 17, 99]).unwrap(),
                        reference.sample(&logits, &p, 1.2, &[0, 17, 99]).unwrap()
                    );
                    assert_eq!(fast.candidates, reference.candidates);
                }
            }
        }
        let mut values = (0u32..8192)
            .map(|id| (if id % 2 == 0 { 0.25f32 } else { 0.5f32 }, id))
            .collect::<Vec<_>>();
        let mut expected = values.clone();
        expected.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        radix_order(&mut values, &mut Vec::new());
        assert_eq!(values, expected);
    }

    #[test]
    #[ignore = "CPU sampler benchmark; run with --release --ignored --nocapture"]
    fn radix_sample_benchmark() {
        use std::{hint::black_box, time::Instant};
        let logits = if let Ok(path) = std::env::var("HAYATE_BENCH_LOGITS") {
            let raw = std::fs::read(path).unwrap();
            raw.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        } else {
            bf16(
                &(0..154880)
                    .map(|i| ((i * 7919 % 4093) as f32 - 2046.) / 128.)
                    .collect::<Vec<_>>(),
            )
        };
        let p = params(1., 0.95, 0, 0.);
        let mut fast = Sampler::new(42);
        let mut reference = Sampler::new(42);
        reference.comparison_sort = true;
        for _ in 0..32 {
            assert_eq!(
                fast.sample(&logits, &p, 1., &[]).unwrap(),
                reference.sample(&logits, &p, 1., &[]).unwrap()
            );
            assert_eq!(fast.candidates, reference.candidates);
        }
        for comparison_sort in [true, false] {
            let mut sampler = Sampler::new(42);
            sampler.comparison_sort = comparison_sort;
            for _ in 0..8 {
                black_box(sampler.sample(&logits, &p, 1., &[]).unwrap());
            }
            let mut times = Vec::new();
            for _ in 0..7 {
                let start = Instant::now();
                for _ in 0..64 {
                    black_box(sampler.sample(&logits, &p, 1., &[]).unwrap());
                }
                times.push(start.elapsed().as_secs_f64() * 1000. / 64.);
            }
            times.sort_by(f64::total_cmp);
            eprintln!(
                "sampler comparison_sort={comparison_sort} vocab={} median_ms={:.6}",
                logits.len(),
                times[3]
            );
        }
    }
}
