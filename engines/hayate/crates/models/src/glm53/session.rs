//! Thin serve session API: append tokens, read logits, save/restore the state
//! at a token boundary and copy KV blocks between conversations. The serve
//! engine (crates/cli/src/serve/glm.rs) owns token history and prefix reuse.
use super::super::gpu::session_state::LayerState;
use super::*;

/// Every layer's recurrent state and pool tail at one position (~136 MiB,
/// pinned). Reuse a dropped state with `session_save(Some(old))`.
#[derive(Debug)]
pub struct SessionState {
    position: usize,
    layers: Vec<LayerState>,
    /// Routes of the last context tokens: a restore starts the same context
    /// placement probe as the end of a prefill to this position. The GPU set
    /// itself is not restored (it would cost seconds of swaps per restore),
    /// so a restored continuation matches the cold run up to the CPU/GPU
    /// expert rounding (relative 1e-4) where their placements differ.
    tail: Vec<Vec<u16>>,
}
impl SessionState {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(LayerState::bytes).sum()
    }
}

impl Executor<'_> {
    fn session_layers(&self) -> impl Iterator<Item = &Layer<'_>> {
        self.layers.iter().chain(self.mtp.as_ref())
    }
    pub fn session_position(&self) -> usize {
        self.position
    }
    pub fn session_context(&self) -> usize {
        self.context
    }
    pub fn session_vocab(&self) -> usize {
        self.vocab
    }
    /// Prefill several tokens or decode one; returns the greedy next token.
    pub fn session_append(&mut self, ids: &[u32]) -> Result<u32> {
        match ids {
            [] => Err(Error::InvalidShape("GLM empty append".into())),
            [id] => self.step(*id),
            _ => self.prefill(ids),
        }
    }
    /// BF16 logits of the last appended token.
    pub fn session_logits(&self, out: &mut Vec<u16>) -> Result<()> {
        let raw = self.logits_raw()?;
        out.clear();
        out.extend(
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b)),
        );
        Ok(())
    }
    pub fn session_save(&self, reuse: Option<SessionState>) -> Result<SessionState> {
        let mut state = match reuse {
            Some(s) if s.layers.len() == self.session_layers().count() => s,
            _ => SessionState {
                position: 0,
                layers: self
                    .session_layers()
                    .map(Layer::session_state)
                    .collect::<Result<_>>()?,
                tail: Vec::new(),
            },
        };
        state.position = self.position;
        state.tail = self.experts.tail_routes();
        for (layer, out) in self.session_layers().zip(&mut state.layers) {
            layer.session_save(out)?;
        }
        for layer in self.session_layers() {
            layer.session_synchronize()?;
        }
        Ok(state)
    }
    /// Recurrent state and position only; the caller makes sure the KV of
    /// tokens before `state.position()` belongs to the same prefix.
    pub fn session_restore(&mut self, state: &SessionState) -> Result<()> {
        self.device_ready = false;
        if state.position > self.context || state.layers.len() != self.session_layers().count() {
            return Err(Error::InvalidPlan("GLM checkpoint/model mismatch".into()));
        }
        self.stream.synchronize()?;
        for (layer, saved) in self
            .layers
            .iter_mut()
            .chain(self.mtp.as_mut())
            .zip(&state.layers)
        {
            layer.session_restore(saved)?;
        }
        for layer in self.session_layers() {
            layer.session_synchronize()?;
        }
        self.position = state.position;
        self.experts.set_tail(&state.tail);
        self.place_for_context()
    }
    /// KV of tokens `start..end` for every attention layer.
    pub fn session_kv_read(&self, start: usize, end: usize) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for layer in self.session_layers() {
            layer.session_kv_read(start, end, &mut out)?;
        }
        Ok(out)
    }
    pub fn session_kv_write(&mut self, start: usize, end: usize, data: &[u8]) -> Result<()> {
        self.device_ready = false;
        let mut at = 0;
        for layer in self.layers.iter_mut().chain(self.mtp.as_mut()) {
            at += layer.session_kv_write(start, end, &data[at..])?;
        }
        if at != data.len() {
            return Err(Error::InvalidPlan("GLM KV block size".into()));
        }
        Ok(())
    }
    /// (experts on the GPU, all routed experts).
    pub fn session_experts(&self) -> (usize, usize) {
        let gpu = self.experts.slot_count();
        let total = (self.layers.len().saturating_sub(3) + usize::from(self.mtp.is_some()))
            * super::super::EXPERTS;
        (gpu, total)
    }
}
impl Executor<'_> {
    pub fn session_synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
}
