//! Model dimensions read from `config.json` (the text config), so the same engine runs the real
//! model and the shrunken golden model. The real values are checked against `tsumuji_model::shape`.

use std::path::Path;
use tsumuji_pack::json::Json;

#[derive(Debug, Clone, PartialEq)]
pub struct Dims {
    pub h: usize,
    pub hc: usize,
    pub rank: usize,
    pub vocab: usize,
    pub layers: usize,
    pub experts: usize,
    pub topk: usize,
    pub ff: usize,
    pub ff_shared: usize,
    pub gdn_qk: usize,
    pub gdn_v: usize,
    pub gdn_d: usize,
    pub gdn_conv: usize,
    pub att_q: usize,
    pub att_kv: usize,
    pub att_d: usize,
    pub rope: usize,
    pub theta: f32,
    pub idx_h: usize,
    pub idx_d: usize,
    pub idx_r: usize,
    pub idx_topb: usize,
    /// 0-based layer with PLE (config's `ple_layer_ids` is 1-based).
    pub ple_layer: Option<usize>,
    pub ple_heads: usize,
    pub ple_row: usize,
    pub ngram_base: u64,
    pub ple_seed: u64,
    pub eos: u32,
    pub eps: f32,
    /// true where the layer is QSA (indexed attention), false for Gated DeltaNet.
    pub qsa: Vec<bool>,
    /// Longest context the model supports (max_position_embeddings; usize::MAX when absent).
    pub max_pos: usize,
}

impl Dims {
    pub fn hcw(&self) -> usize {
        self.h * self.hc
    }

    /// Channels of the GDN causal conv (q, k, v).
    pub fn gdn_c(&self) -> usize {
        (2 * self.gdn_qk + self.gdn_v) * self.gdn_d
    }

    /// Rows of the fused GDN input projection [qkv | z | b | a].
    pub fn gdn_in_rows(&self) -> usize {
        self.gdn_c() + self.gdn_v * self.gdn_d + 2 * self.gdn_v
    }

    /// Rows of the fused QSA projection [q+gate | k | v | indexer q + key].
    pub fn qsa_rows(&self) -> usize {
        self.att_q * 2 * self.att_d + 2 * self.att_kv * self.att_d + (self.idx_h + 1) * self.idx_d
    }

    /// Most tokens one query attends to: TOPB blocks plus the incomplete tail.
    pub fn qsa_max_tokens(&self) -> usize {
        self.idx_topb * self.idx_r + self.idx_r - 1
    }

    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(model_dir.join("config.json")).map_err(|e| e.to_string())?;
        let root = Json::parse(&text)?;
        let c = root.get("text_config").unwrap_or(&root);
        let u = |k: &str| {
            c.get(k)
                .and_then(Json::as_u64)
                .map(|v| v as usize)
                .ok_or(format!("config: no {k}"))
        };
        let f = |j: Option<&Json>| match j {
            Some(Json::Num(n)) => Some(*n),
            _ => None,
        };
        let rp = c.get("rope_parameters");
        let partial = f(rp.and_then(|r| r.get("partial_rotary_factor")))
            .or(f(c.get("partial_rotary_factor")))
            .unwrap_or(1.0);
        let theta = f(rp.and_then(|r| r.get("rope_theta"))).ok_or("config: no rope_theta")?;
        let layers = u("num_hidden_layers")?;
        let qsa = match c.get("layer_types").and_then(Json::as_arr) {
            Some(t) => t
                .iter()
                .map(|x| x.as_str() != Some("linear_attention"))
                .collect(),
            None => (0..layers)
                .map(|l| (l + 1) % u("full_attention_interval").unwrap_or(4) == 0)
                .collect(),
        };
        let ple_ids: Vec<usize> = c
            .get("ple_layer_ids")
            .and_then(Json::as_arr)
            .map(|a| {
                a.iter()
                    .filter_map(Json::as_u64)
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default();
        if ple_ids.len() > 1 {
            return Err("more than one PLE layer is not supported".into());
        }
        let eos = match c.get("eos_token_id") {
            Some(Json::Arr(a)) => a[0].as_u64(),
            Some(j) => j.as_u64(),
            None => None,
        }
        .ok_or("config: no eos_token_id")? as u32;
        let ngram = u("ngram_size").unwrap_or(3);
        let ple_heads = (ngram - 1) * u("heads_per_ngram").unwrap_or(8);
        let h = u("hidden_size")?;
        let head_dim = u("head_dim")?;
        let d = Self {
            h,
            hc: u("hc_count")?,
            rank: u("hc_lowrank")?,
            vocab: u("vocab_size")?,
            layers,
            experts: u("num_experts")?,
            topk: u("num_experts_per_tok")?,
            ff: u("moe_intermediate_size")?,
            ff_shared: u("shared_expert_intermediate_size")?,
            gdn_qk: u("linear_num_key_heads")?,
            gdn_v: u("linear_num_value_heads")?,
            gdn_d: u("linear_key_head_dim")?,
            gdn_conv: u("linear_conv_kernel_dim")?,
            att_q: u("num_attention_heads")?,
            att_kv: u("num_key_value_heads")?,
            att_d: head_dim,
            rope: (head_dim as f64 * partial) as usize,
            theta: theta as f32,
            idx_h: u("indexer_n_heads")?,
            idx_d: u("indexer_head_dim")?,
            idx_r: u("indexer_compress_ratio")?,
            idx_topb: u("indexer_budget")? / u("indexer_compress_ratio")?,
            ple_layer: ple_ids.first().map(|l| l - 1),
            ple_heads,
            ple_row: c
                .get("ple_embed_dim")
                .and_then(Json::as_u64)
                .map_or(h, |v| v as usize)
                / ple_heads,
            ngram_base: u("ngram_vocab_size_base").unwrap_or(20_000_000) as u64,
            ple_seed: u("seed").unwrap_or(1234) as u64,
            eos,
            eps: f(c.get("rms_norm_eps")).unwrap_or(1e-6) as f32,
            qsa,
            max_pos: u("max_position_embeddings").unwrap_or(usize::MAX),
        };
        d.validate()?;
        Ok(d)
    }

    /// Invariants the kernels rely on.
    fn validate(&self) -> Result<(), String> {
        let checks = [
            (
                self.gdn_d.is_multiple_of(32) && self.gdn_d <= 128,
                "GDN head dim must be 32, 64, 96 or 128",
            ),
            (
                self.gdn_v.is_multiple_of(self.gdn_qk),
                "GDN value heads must be a multiple of key heads",
            ),
            (
                self.att_q.is_multiple_of(self.att_kv),
                "query heads must be a multiple of KV heads",
            ),
            (self.att_d <= 256 && self.idx_d <= self.att_d, "head dims"),
            (
                self.ff == self.ff_shared,
                "shared and routed intermediate sizes differ",
            ),
            (
                self.h.is_multiple_of(32) && self.ff.is_multiple_of(32),
                "NVFP4 dims must be multiples of 32",
            ),
            (self.ple_heads == 16, "PLE kernels assume 16 heads"),
            (
                self.ple_row.is_multiple_of(16),
                "PLE row must be a multiple of 16 bytes",
            ),
            (self.gdn_conv == 4, "GDN conv kernel must be 4"),
            (self.experts <= 1024 && self.topk <= 32, "router limits"),
            (self.hc <= 4, "GR kernels assume at most 4 streams"),
            (
                self.rank <= 512 && self.rank.is_multiple_of(32),
                "GR rank must be a multiple of 32, at most 512",
            ),
            (self.qsa.len() == self.layers, "layer_types length"),
        ];
        for (ok, msg) in checks {
            if !ok {
                return Err(format!("unsupported config: {msg}"));
            }
        }
        Ok(())
    }

    /// True for the real Qwen3.8-Flash-Next shapes.
    pub fn is_qwen38(&self) -> bool {
        use tsumuji_model::shape as s;
        self.h == s::H
            && self.hc == s::HC
            && self.rank == s::GR_RANK
            && self.vocab == s::VOCAB
            && self.layers == s::L
            && self.experts == s::E
            && self.topk == s::TOPK
            && self.ff == s::FF
            && self.gdn_qk == s::GDN_QK
            && self.gdn_v == s::GDN_V
            && self.gdn_d == s::GDN_D
            && self.att_q == s::ATT_Q
            && self.att_kv == s::ATT_KV
            && self.att_d == s::ATT_D
            && self.rope == s::ROPE_D
            && self.idx_h == s::IDX_H
            && self.idx_d == s::IDX_D
            && self.idx_r == s::IDX_R
            && self.idx_topb == s::IDX_TOPB
            && self.ple_layer == Some(s::PLE_LAYER)
            && self.ple_row == s::PLE_ROW
            && self.eos == s::EOS
            && (0..s::L).all(|l| self.qsa[l] == s::is_qsa(l))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_config_matches_shapes() {
        let dir = Path::new(
            "/mnt/NVME4TB/darask-inference-engine/local/hayate/src/Qwen3.8-Flash-Next-NVFP4",
        );
        if !dir.join("config.json").exists() {
            eprintln!("skipped: no checkpoint");
            return;
        }
        let d = Dims::load(dir).unwrap();
        assert!(d.is_qwen38(), "{d:?}");
        assert_eq!(d.gdn_in_rows(), 16_480);
        assert_eq!(d.qsa_rows(), 13_952);
        assert_eq!(d.qsa_max_tokens(), 2051);
        assert!((d.theta - 1e7).abs() < 1.0);
    }
}
