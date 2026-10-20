//! LLM runtime: continuous batching, hybrid cache, MTP speculative decoding.

pub mod kv;
pub mod metrics;
pub mod sample;
pub mod session;

/// Per-layer sequence state. The three model families mix these within a
/// single model, so the cache manager is per layer, not per model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerCache {
    /// Paged KV for full/global attention (Gemma 4 global layers).
    PagedKv,
    /// Ring buffer of `window` tokens (Gemma 4 local layers: 512 / 1024).
    SlidingKv { window: u32 },
    /// Fixed-size recurrent state plus the short-conv history
    /// (Qwen GDN, GLM linear attention).
    Recurrent,
    /// Paged KV plus a compressed index cache for block selection
    /// (Qwen QSA: r = 4 compression, 2048-token budget; GLM sparse layers).
    SparseIndexedKv { compress: u32, budget: u32 },
}

impl LayerCache {
    /// How a rejected speculative suffix is undone (ARCHITECTURE.md §9.1).
    /// Paged and sparse KV just truncate; the others overwrite state in place, so the
    /// verifier keeps one snapshot per draft position and restores the one
    /// matching the accepted prefix.
    pub const fn rollback_needs_snapshot(self) -> bool {
        !matches!(
            self,
            LayerCache::PagedKv | LayerCache::SparseIndexedKv { .. }
        )
    }
}

/// Speculative decoding source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Draft {
    None,
    /// The model's MTP module run autoregressively for `steps` tokens
    /// (Qwen3.8: one MTP layer, 4 steps).
    Mtp {
        steps: u32,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    /// Drop tokens below `min_p` times the most likely token's probability
    /// (0 = off). Engines without a min-p sampler ignore it.
    pub min_p: f32,
    pub max_tokens: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 64,
            min_p: 0.,
            max_tokens: 4096,
        }
    }
}
