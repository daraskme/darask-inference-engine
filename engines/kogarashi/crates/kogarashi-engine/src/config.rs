use darask_formats::gguf::Gguf;

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub h: usize,
    pub layers: usize,
    pub vocab: usize,
    pub experts: usize,
    pub topk: usize,
    pub ff: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub state_dim: usize,
    pub rope: usize,
    pub theta: f32,
    pub eps: f32,
    pub max_ctx: usize,
}
impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        match g.meta.get("general.architecture") {
            Some(darask_formats::gguf::Value::Str(s)) if s == "qwen35moe" => (),
            _ => return Err("expected a qwen35moe GGUF".into()),
        }
        let u = |k: &str| {
            g.f64(&format!("qwen35moe.{k}"))
                .filter(|v| *v > 0.0 && v.fract() == 0.0)
                .map(|v| v as usize)
                .ok_or_else(|| format!("missing/invalid {k}"))
        };
        let c = Self {
            h: u("embedding_length")?,
            layers: u("block_count")?
                - g.f64("qwen35moe.nextn_predict_layers").unwrap_or(0.0) as usize,
            vocab: g
                .tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .ok_or("no embeddings")?
                .ne[1] as usize,
            experts: u("expert_count")?,
            topk: u("expert_used_count")?,
            ff: u("expert_feed_forward_length")?,
            q_heads: u("attention.head_count")?,
            kv_heads: u("attention.head_count_kv")?,
            head: u("attention.key_length")?,
            key_heads: u("ssm.group_count")?,
            value_heads: u("ssm.time_step_rank")?,
            state_dim: u("ssm.state_size")?,
            rope: u("rope.dimension_count")?,
            theta: g.f64("qwen35moe.rope.freq_base").ok_or("no theta")? as f32,
            eps: g
                .f64("qwen35moe.attention.layer_norm_rms_epsilon")
                .ok_or("no epsilon")? as f32,
            max_ctx: u("context_length")?,
        };
        // Native kernels currently implement this precise, validated shape only.
        if (
            c.h,
            c.layers,
            c.experts,
            c.topk,
            c.ff,
            c.q_heads,
            c.kv_heads,
            c.head,
            c.key_heads,
            c.value_heads,
            c.state_dim,
            c.rope,
        ) != (3072, 48, 256, 8, 1024, 32, 2, 256, 16, 64, 128, 64)
        {
            return Err(format!("unsupported Ornith shape: {c:?}"));
        }
        Ok(c)
    }
    pub fn recurrent(&self, layer: usize) -> bool {
        (layer + 1) % 4 != 0
    }
}

/// Bytes in a contiguous GGML tensor. Quantized rows must contain complete blocks.
pub fn tensor_bytes(ty: u32, ne: &[u64]) -> Result<usize, String> {
    let (block, bytes) = match ty {
        0 => (1, 4),
        1 | 30 => (1, 2),
        12 => (256, 144),
        14 => (256, 210),
        _ => return Err(format!("unsupported GGML type {ty}")),
    };
    if ne.is_empty() || ne[0] == 0 || ne[0] % block != 0 || ne.contains(&0) {
        return Err("invalid tensor shape/block alignment".into());
    }
    let n = ne
        .iter()
        .try_fold(1u64, |a, b| a.checked_mul(*b))
        .ok_or("tensor shape overflow")?;
    usize::try_from(n / block)
        .ok()
        .and_then(|n| n.checked_mul(bytes))
        .ok_or_else(|| "tensor size overflow".into())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ggml_blocks_and_overflow() {
        assert_eq!(tensor_bytes(12, &[3072, 1024, 256]).unwrap(), 452984832);
        assert_eq!(tensor_bytes(14, &[1024, 3072, 256]).unwrap(), 660602880);
        for ne in [&[255, 2][..], &[256, 0], &[u64::MAX, 2]] {
            assert!(tensor_bytes(12, ne).is_err());
        }
        assert!(tensor_bytes(2, &[256]).is_err());
    }
}
