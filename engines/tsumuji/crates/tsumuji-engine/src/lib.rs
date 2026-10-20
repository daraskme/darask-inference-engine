//! Tsumuji engine for Qwen3.8-Flash-Next: config, weight loading and the decode step.

pub mod adapter;
pub mod config;
pub mod engine;
pub mod profile;
pub mod weights;

use std::path::Path;
use std::rc::Rc;
use tsumuji_cuda::Device;

/// Opens device 0 and loads the model in `dir` with room for `ctx` tokens.
pub fn open(dir: &Path, ctx: usize, progress: impl FnMut(&str)) -> Result<engine::Engine, String> {
    let dev = Rc::new(Device::open(0).map_err(|e| e.to_string())?);
    let dims = config::Dims::load(dir)?;
    let weights = weights::load(dir, &dims, progress)?;
    engine::Engine::new(dev, dims, weights, ctx)
}

/// Token ids from a file: a JSON array or integers separated by whitespace or commas.
pub fn read_ids(path: &Path) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']')
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u32>()
                .map_err(|e| format!("{}: '{s}': {e}", path.display()))
        })
        .collect()
}
