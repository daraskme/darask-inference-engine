//! Wall-clock LLM timings; model loading and graph capture are reported separately.
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct GenerationMetrics {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub prefill: Duration,
    pub capture: Duration,
    /// One duration for each graph replay plus its four-byte token readback.
    pub decode: Vec<Duration>,
    pub peak_vram_bytes: usize,
}
impl GenerationMetrics {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        self.input_tokens as f64 / self.prefill.as_secs_f64()
    }
    pub fn decode_tokens_per_second(&self) -> f64 {
        if self.decode.is_empty() {
            0.
        } else {
            self.decode.len() as f64 / self.decode.iter().map(Duration::as_secs_f64).sum::<f64>()
        }
    }
    pub fn percentile_ms(&self, p: f64) -> f64 {
        if self.decode.is_empty() {
            return 0.;
        }
        let mut values: Vec<_> = self
            .decode
            .iter()
            .map(|t| t.as_secs_f64() * 1000.)
            .collect();
        values.sort_by(f64::total_cmp);
        values[((p.clamp(0., 1.) * values.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1)]
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metrics_exclude_first_token_from_decode_rate() {
        let m = GenerationMetrics {
            input_tokens: 512,
            output_tokens: 4,
            prefill: Duration::from_millis(100),
            capture: Duration::from_millis(2),
            decode: [10, 30, 20].map(Duration::from_millis).into(),
            peak_vram_bytes: 1024,
        };
        assert_eq!(m.prefill_tokens_per_second(), 5120.);
        assert_eq!(m.decode_tokens_per_second(), 50.);
        assert_eq!(m.percentile_ms(0.5), 20.);
        assert_eq!(m.percentile_ms(0.95), 30.);
    }
}
