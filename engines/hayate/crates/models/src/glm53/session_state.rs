//! Per-layer serve session state: recurrent checkpoints and KV block copies.
//! KDA layers keep a recurrent state and short-conv history; attention layers
//! keep the latent KV (device `latent_cache`, one row per token), the indexer
//! pools (device, one row per 4 tokens) and the 4-token pool tail.
use super::*;

const RECURRENT: [&str; 4] = ["state", "hq", "hk", "hv"];
const TAIL: [&str; 2] = ["pool_keys", "pool_gates"];
const LATENT_ROW: usize = 512 * 2;
const POOL_ROW: usize = 128 * 2;

/// One layer's state at a token boundary, in pinned buffers reused across saves.
#[derive(Debug)]
pub(crate) struct LayerState {
    pub(crate) position: usize,
    buffers: Vec<PinnedHost>,
    tail_keys: Vec<u8>,
    tail_gates: Vec<u8>,
}
impl LayerState {
    pub(crate) fn bytes(&self) -> usize {
        self.buffers
            .iter()
            .map(|b| b.as_slice().len())
            .sum::<usize>()
            + self.tail_keys.len()
            + self.tail_gates.len()
    }
}

impl Layer<'_> {
    fn attention(&self) -> bool {
        self.index % 4 == 3 || self.index == 45
    }
    fn state_names(&self) -> &'static [&'static str] {
        if self.attention() { &TAIL } else { &RECURRENT }
    }
    pub(crate) fn session_state(&self) -> Result<LayerState> {
        Ok(LayerState {
            position: 0,
            buffers: self
                .state_names()
                .iter()
                .map(|n| self.device.alloc_pinned(self.buffers[*n].bytes()))
                .collect::<Result<_>>()?,
            tail_keys: vec![0; self.tail_keys.len()],
            tail_gates: vec![0; self.tail_gates.len()],
        })
    }
    /// Enqueue the copies on this layer's stream; synchronize before use.
    pub(crate) fn session_save(&self, out: &mut LayerState) -> Result<()> {
        out.position = self.position;
        for (name, host) in self.state_names().iter().zip(&mut out.buffers) {
            unsafe { self.stream.copy_d2h_async(host, &self.buffers[*name])? };
        }
        out.tail_keys.copy_from_slice(&self.tail_keys);
        out.tail_gates.copy_from_slice(&self.tail_gates);
        Ok(())
    }
    pub(crate) fn session_restore(&mut self, state: &LayerState) -> Result<()> {
        if state.position > self.context || state.buffers.len() != self.state_names().len() {
            return Err(Error::InvalidPlan("GLM layer checkpoint".into()));
        }
        for (name, host) in self.state_names().iter().zip(&state.buffers) {
            let buffer = self.buffers.get_mut(*name).unwrap();
            unsafe { self.stream.copy_h2d_async(buffer, host)? };
        }
        self.tail_keys.copy_from_slice(&state.tail_keys);
        self.tail_gates.copy_from_slice(&state.tail_gates);
        self.position = state.position;
        Ok(())
    }
    pub(crate) fn session_synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
    /// KV of tokens `start..end` (`start` a multiple of 4): latent rows, then
    /// the pool rows of the complete 4-token groups. Empty for KDA layers.
    pub(crate) fn session_kv_read(
        &self,
        start: usize,
        end: usize,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        if !self.attention() {
            return Ok(());
        }
        if !start.is_multiple_of(4) || start > end || end > self.context {
            return Err(Error::InvalidPlan("GLM KV range".into()));
        }
        let rows = end / 4 - start / 4;
        let at = out.len();
        out.resize(at + (end - start) * LATENT_ROW + rows * POOL_ROW, 0);
        if end > start {
            self.stream.copy_d2h_at(
                &mut out[at..at + (end - start) * LATENT_ROW],
                &self.buffers["latent_cache"],
                start * LATENT_ROW,
            )?;
        }
        let at = at + (end - start) * LATENT_ROW;
        if rows > 0 {
            self.stream.copy_d2h_at(
                &mut out[at..],
                &self.buffers["pools"],
                start / 4 * POOL_ROW,
            )?;
        }
        Ok(())
    }
    /// Inverse of `session_kv_read`; returns the bytes consumed from `data`.
    pub(crate) fn session_kv_write(
        &mut self,
        start: usize,
        end: usize,
        data: &[u8],
    ) -> Result<usize> {
        if !self.attention() {
            return Ok(0);
        }
        let latent_bytes = (end - start) * LATENT_ROW;
        let rows = end / 4 - start / 4;
        if !start.is_multiple_of(4)
            || end > self.context
            || data.len() < latent_bytes + rows * POOL_ROW
        {
            return Err(Error::InvalidPlan("GLM KV block".into()));
        }
        if latent_bytes > 0 {
            self.stream.copy_h2d_at(
                self.buffers.get_mut("latent_cache").unwrap(),
                start * LATENT_ROW,
                &data[..latent_bytes],
            )?;
        }
        // Pinned mirror, present only for MTP verification.
        self.stream.synchronize()?;
        if let Some(host) = self.latent.as_mut() {
            host.as_mut_slice()[start * LATENT_ROW..end * LATENT_ROW]
                .copy_from_slice(&data[..latent_bytes]);
        }
        if rows > 0 {
            let pools = self.buffers.get_mut("pools").unwrap();
            self.stream.copy_h2d_at(
                pools,
                start / 4 * POOL_ROW,
                &data[latent_bytes..latent_bytes + rows * POOL_ROW],
            )?;
        }
        Ok(latent_bytes + rows * POOL_ROW)
    }
}
