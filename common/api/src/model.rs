//! The text side of the model: tokenizer, generation defaults and the context size. Everything a
//! front end needs besides the engine (this HTTP server, or Hayate's serve linking the library).

use crate::json::Json;
use crate::sample::Params;
use crate::tokenizer::Tokenizer;
use std::path::Path;

/// What the model directory says about generation.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    /// Generation stops at these (generation_config.json `eos_token_id`).
    pub eos: Vec<u32>,
    /// Sampling defaults from generation_config.json.
    pub defaults: Params,
    /// max_position_embeddings.
    pub max_ctx: usize,
}

fn read_json(p: &Path) -> Result<Json, String> {
    let t = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    Json::parse(&t).map_err(|e| format!("{}: {e}", p.display()))
}

impl ModelInfo {
    pub fn load(dir: &Path, name: Option<String>) -> Result<Self, String> {
        let g = read_json(&dir.join("generation_config.json"))?;
        let c = read_json(&dir.join("config.json"))?;
        let ids = |v: Option<&Json>| -> Vec<u32> {
            match v {
                Some(Json::Arr(a)) => a
                    .iter()
                    .filter_map(Json::as_u64)
                    .map(|x| x as u32)
                    .collect(),
                Some(x) => x.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
                None => Vec::new(),
            }
        };
        let mut eos = ids(g.get("eos_token_id"));
        if eos.is_empty() {
            eos = ids(c
                .get("text_config")
                .and_then(|t| t.get("eos_token_id"))
                .or(c.get("eos_token_id")));
        }
        let f = |k: &str, d: f32| g.get(k).and_then(Json::as_f64).map_or(d, |v| v as f32);
        let sample = g.get("do_sample").and_then(Json::as_bool).unwrap_or(true);
        let defaults = Params {
            temperature: if sample { f("temperature", 1.0) } else { 0.0 },
            top_p: f("top_p", 1.0),
            top_k: g.get("top_k").and_then(Json::as_u64).unwrap_or(0) as usize,
            ..Params::default()
        };
        let max_ctx = c
            .get("text_config")
            .and_then(|t| t.get("max_position_embeddings"))
            .or(c.get("max_position_embeddings"))
            .and_then(Json::as_u64)
            .unwrap_or(32768) as usize;
        let id = name.unwrap_or_else(|| "qwen38".into());
        Ok(Self {
            id,
            eos,
            defaults,
            max_ctx,
        })
    }
}

/// Tokenizer + generation defaults + context: what request parsing and output need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplate {
    #[default]
    Qwen38,
    Ornith,
}

#[derive(Debug)]
pub struct TextModel {
    pub template: ChatTemplate,
    pub tok: Tokenizer,
    pub info: ModelInfo,
    /// Most tokens (prompt + generated) a sequence may hold; the engine's context.
    pub ctx: usize,
    /// `<|im_start|>`: where a chat turn begins (the prefix cache snapshots there).
    pub turn_start: Option<u32>,
}

impl TextModel {
    /// The model directory's tokenizer and configs, with the context at the model's maximum
    /// (set `ctx` to the engine's).
    pub fn load(dir: &Path, name: Option<String>) -> Result<Self, String> {
        let info = ModelInfo::load(dir, name)?;
        let tok = Tokenizer::load(dir)?;
        let turn_start = tok.token_id("<|im_start|>");
        Ok(Self {
            template: ChatTemplate::Qwen38,
            ctx: info.max_ctx,
            tok,
            info,
            turn_start,
        })
    }
}
