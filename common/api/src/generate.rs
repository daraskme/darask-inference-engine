//! The boundary between the server and the decode engine. Front ends only see `Generator`: "run
//! this prompt and hand me tokens until I say stop". M3 (device-side sampling, speculative
//! decoding) replaces `EngineGen` behind the same trait; `emit` may then be called several times
//! per engine step.
//!
//! The prompt goes in through `PrefixCache`: agents resend the whole conversation every turn, so
//! the state after a shared prefix is restored from a snapshot and only the rest is prefilled.
//! Snapshots are taken where the next request most likely continues: at the start of the
//! prompt's last turn (the history before it comes back unchanged even when the client drops the
//! reasoning it was sent), at the end of the prompt, and at the end of generation. All live on
//! the engine's one KV region, so they are prefixes of one token history; writing past a shared
//! prefix drops the snapshots beyond it.

use crate::sample::Params;
use std::time::{Duration, Instant};
#[cfg(feature = "engine")]
use {
    crate::sample::Sampler,
    tsumuji_engine::engine::{Engine, N_MAX, Sampling, Snapshot},
};

type Res<T> = Result<T, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// Prompt in (the part not cached), first token known.
    pub prefill_s: f64,
    /// First token known to the last token known.
    pub decode_s: f64,
    pub generated: usize,
    /// Prompt tokens restored from the prefix cache.
    pub cached: usize,
    /// Sampled on the host (the logits round trip), not on the device.
    pub host_sampling: bool,
    /// How tokens were picked: "greedy", "device" (sampled on the GPU) or "host".
    pub sampling: &'static str,
    /// Speculative steps after the first (which only commits the prompt's next token) and the
    /// tokens they committed: `mtp_accept` = tokens / steps.
    pub spec_steps: usize,
    pub spec_tokens: usize,
    /// Host round trip per sampled token: logits copy after the step finished, and sampling.
    pub copy_s: f64,
    pub sample_s: f64,
}

impl Stats {
    /// Prefilled (not cached) prompt tokens per second.
    pub fn prefill_tps(&self, prompt: usize) -> f64 {
        prompt.saturating_sub(self.cached) as f64 / self.prefill_s.max(1e-9)
    }

    /// Tokens per second after the first (the first comes out of the prefill).
    pub fn decode_tps(&self) -> f64 {
        if self.generated < 2 {
            0.0
        } else {
            (self.generated - 1) as f64 / self.decode_s.max(1e-9)
        }
    }

    /// Mean tokens committed per speculative step (None without speculation).
    pub fn mtp_accept(&self) -> Option<f64> {
        (self.spec_steps > 0).then(|| self.spec_tokens as f64 / self.spec_steps as f64)
    }
}

/// One generation.
#[derive(Debug, Clone, Copy)]
pub struct GenRequest<'a> {
    pub prompt: &'a [u32],
    pub images: &'a [darask_vision::Span],
    pub params: &'a Params,
    pub max_new: usize,
    /// Prompt positions to snapshot at (see the module docs).
    pub marks: &'a [usize],
    /// Which form of the loaded model runs it (0 = as loaded; e.g. a runtime switch such as
    /// abliteration). States from different variants are never shared by the prefix cache.
    pub variant: u32,
}

pub trait Generator {
    /// Most tokens (prompt + generated) a sequence may hold.
    fn ctx(&self) -> usize;
    /// Runs the prompt and generates up to `max_new` tokens, calling `emit` with each one as soon
    /// as it is known; stops early when `emit` returns `Flow::Stop`, or during the prompt when
    /// `cancelled()` turns true (then no token is emitted).
    fn generate(
        &mut self,
        req: GenRequest<'_>,
        emit: &mut dyn FnMut(u32) -> Flow,
        cancelled: &dyn Fn() -> bool,
    ) -> Res<Stats>;
}

/// What the prefix cache needs from an engine.
pub trait StateEngine {
    type Snapshot;
    /// A fresh sequence at position 0.
    fn reset(&mut self) -> Res<()>;
    /// Feeds tokens at the next positions (may run asynchronously).
    fn prefill(&mut self, tokens: &[u32]) -> Res<()>;
    /// Waits for the queued work.
    fn synchronize(&mut self) -> Res<()>;
    /// The sequence state now (recurrent states, position; not the KV), or None when the engine
    /// cannot take snapshots (then every request starts from `reset`).
    fn save(&mut self) -> Res<Option<Self::Snapshot>>;
    fn restore(&mut self, s: &Self::Snapshot) -> Res<()>;
}

/// The token history of the KV region and the snapshots on it.
#[derive(Debug)]
pub struct PrefixCache<S> {
    /// Tokens the KV region holds, by position (what the current state has seen).
    kv: Vec<u32>,
    /// (length, state after kv[..length]), oldest first.
    snaps: Vec<(usize, S)>,
    /// Most snapshots kept (each holds the recurrent states: about 120 MiB on the GPU).
    pub max: usize,
    /// The model variant the KV region and the snapshots were computed with.
    variant: u32,
}

impl<S> PrefixCache<S> {
    pub fn new(max: usize) -> Self {
        Self {
            kv: Vec::new(),
            snaps: Vec::new(),
            max,
            variant: 0,
        }
    }

    pub fn tokens(&self) -> &[u32] {
        &self.kv
    }

    /// Snapshot lengths, oldest first.
    pub fn lengths(&self) -> Vec<usize> {
        self.snaps.iter().map(|(l, _)| *l).collect()
    }

    /// Puts `eng` in the state after the longest cached prefix of `prompt` computed with the same
    /// `variant` (at least one prompt token is left to run, for the logits) and returns its
    /// length; 0 after a reset.
    pub fn begin<E: StateEngine<Snapshot = S>>(
        &mut self,
        eng: &mut E,
        prompt: &[u32],
        variant: u32,
    ) -> Res<usize> {
        let common = if variant == self.variant {
            self.kv
                .iter()
                .zip(prompt)
                .take_while(|(a, b)| a == b)
                .count()
        } else {
            0
        };
        self.variant = variant;
        // Writing prompt[best..] keeps what matches; snapshots past the shared prefix go.
        self.snaps.retain(|(l, _)| *l <= common);
        let best = self
            .snaps
            .iter()
            .enumerate()
            .filter(|(_, (l, _))| *l < prompt.len())
            .max_by_key(|(_, (l, _))| *l)
            .map(|(i, _)| i);
        match best {
            Some(i) => {
                let l = self.snaps[i].0;
                if let Err(e) = eng.restore(&self.snaps[i].1) {
                    self.snaps.clear();
                    self.kv.clear();
                    eng.reset()?;
                    eprintln!("prefix cache: restore failed ({e}); starting over");
                    return Ok(0);
                }
                self.kv.truncate(l);
                Ok(l)
            }
            None => {
                eng.reset()?;
                self.kv.clear();
                self.snaps.clear();
                Ok(0)
            }
        }
    }

    /// Records tokens fed to the engine after `begin`.
    pub fn fed(&mut self, toks: &[u32]) {
        self.kv.extend_from_slice(toks);
    }

    /// Snapshots the state after the tokens fed so far (if the engine can, and if new).
    pub fn snapshot<E: StateEngine<Snapshot = S>>(&mut self, eng: &mut E) -> Res<()> {
        let len = self.kv.len();
        if len == 0 || self.snaps.iter().any(|(l, _)| *l == len) {
            return Ok(());
        }
        if let Some(s) = eng.save()? {
            self.snaps.push((len, s));
            if self.snaps.len() > self.max {
                self.snaps.remove(0);
            }
        }
        Ok(())
    }
}

/// Prompt tokens between checks for a cancelled request (about a second at 250-300 tok/s).
const PREFILL_PIECE: usize = 256;

/// Runs `prompt` through the prefix cache: restore, then prefill the rest in pieces (checking
/// `cancelled` between them) with snapshots at `marks` and at the end. Some(cached tokens), or
/// None when cancelled.
pub fn feed_prompt<E: StateEngine>(
    eng: &mut E,
    cache: &mut PrefixCache<E::Snapshot>,
    prompt: &[u32],
    marks: &[usize],
    variant: u32,
    cancelled: &dyn Fn() -> bool,
) -> Res<Option<usize>> {
    let cached = cache.begin(eng, prompt, variant)?;
    let mut at = cached;
    let mut cuts: Vec<usize> = marks
        .iter()
        .copied()
        .filter(|&m| m > at && m < prompt.len())
        .collect();
    cuts.sort_unstable();
    cuts.dedup();
    cuts.push(prompt.len());
    let mut first = true;
    for cut in cuts {
        for part in prompt[at..cut].chunks(PREFILL_PIECE) {
            if !first {
                eng.synchronize()?;
                if cancelled() {
                    return Ok(None);
                }
            }
            first = false;
            eng.prefill(part)?;
            cache.fed(part);
        }
        cache.snapshot(eng)?;
        at = cut;
    }
    Ok(Some(cached))
}

/// Snapshots kept by default.
pub const SNAPSHOTS: usize = 8;

#[cfg(feature = "engine")]
impl StateEngine for Engine {
    type Snapshot = Snapshot;

    fn reset(&mut self) -> Res<()> {
        Engine::reset(self)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Res<()> {
        Engine::prefill(self, tokens)
    }

    fn synchronize(&mut self) -> Res<()> {
        Engine::synchronize(self)
    }

    fn save(&mut self) -> Res<Option<Snapshot>> {
        self.save_state().map(Some)
    }

    fn restore(&mut self, s: &Snapshot) -> Res<()> {
        self.restore_state(s)
    }
}

/// Today's engine: the prompt through `prefill` (8 tokens a step), then speculative steps with the
/// MTP layer (`spec_auto`: verify the next token and the drafts, commit the accepted ones, draft
/// again, as many drafts as the recent acceptance pays for), picking tokens on the device: greedy, or exact speculative sampling with temperature,
/// top-k, top-p and the seed (`set_sampling`). Penalties and min-p are not on the device: those
/// requests copy the logits to the host and sample there, one token per step.
#[cfg(feature = "engine")]
pub struct EngineGen {
    pub eng: Engine,
    pub cache: PrefixCache<<Engine as StateEngine>::Snapshot>,
    /// MTP drafts per step: None = chosen per step by the engine (`spec_auto`), 0 = one token per step.
    pub drafts: Option<usize>,
    /// Sample every request on the host (debugging, or if the device sampler misbehaves).
    pub host_sampling: bool,
}

#[cfg(feature = "engine")]
impl EngineGen {
    pub fn new(eng: Engine, drafts: Option<usize>, host_sampling: bool) -> Self {
        Self {
            eng,
            cache: PrefixCache::new(SNAPSHOTS),
            drafts,
            host_sampling,
        }
    }
}

#[cfg(feature = "engine")]
impl Generator for EngineGen {
    fn ctx(&self) -> usize {
        self.eng.ctx
    }

    fn generate(
        &mut self,
        req: GenRequest<'_>,
        emit: &mut dyn FnMut(u32) -> Flow,
        cancelled: &dyn Fn() -> bool,
    ) -> Res<Stats> {
        let GenRequest {
            prompt,
            images,
            params,
            max_new,
            marks,
            variant,
        } = req;
        let e = &mut self.eng;
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if variant != 0 {
            return Err(format!(
                "model variant {variant} is not available in this engine"
            ));
        }
        if prompt.len() + max_new > e.ctx {
            return Err(format!(
                "{} prompt + {max_new} new tokens exceed the context of {}",
                prompt.len(),
                e.ctx
            ));
        }
        let t0 = Instant::now();
        // The device picks the tokens unless the request needs what only the host sampler has.
        let device = !self.host_sampling && !params.penalties() && params.min_p <= 0.0;
        let pick = if device {
            Sampling {
                temperature: params.temperature.max(0.0),
                top_k: params.top_k.min(u32::MAX as usize) as u32,
                top_p: params.top_p,
                seed: params.seed,
            }
        } else {
            Sampling {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: 0,
            }
        };
        e.set_sampling(pick)?;
        let sampling = match (device, params.temperature <= 0.0) {
            (false, _) => "host",
            (true, true) => "greedy",
            (true, false) => "device",
        };
        let mut st = Stats {
            host_sampling: !device,
            sampling,
            ..Default::default()
        };
        // Image states must never alias a text prefix or a different image with identical placeholders.
        let has_images = !images.is_empty();
        let cached = if has_images {
            self.cache = PrefixCache::new(SNAPSHOTS);
            e.reset()?;
            e.set_images(images)?;
            for piece in prompt.chunks(PREFILL_PIECE) {
                if cancelled() {
                    return Ok(st);
                }
                e.prefill(piece)?;
            }
            Some(0)
        } else {
            e.set_images(&[])?;
            feed_prompt(e, &mut self.cache, prompt, marks, variant, cancelled)?
        };
        let Some(cached) = cached else {
            st.prefill_s = t0.elapsed().as_secs_f64();
            return Ok(st);
        };
        st.cached = cached;
        if max_new == 0 {
            e.synchronize()?;
            st.prefill_s = t0.elapsed().as_secs_f64();
            return Ok(st);
        }
        if device {
            // The first step commits the token after the prompt and drafts; each later one
            // commits 1 + the accepted drafts. A window (the next token + the last step's drafts)
            // and the new drafts need room: near the end of the context, steps stop drafting.
            let mut t1 = None;
            'steps: loop {
                let drafts = if has_images { Some(0) } else { self.drafts };
                let toks = match drafts {
                    _ if e.pos + 3 * N_MAX >= e.ctx => e.spec_step(0)?,
                    None => e.spec_auto()?,
                    Some(k) => e.spec_step(k)?,
                };
                match t1 {
                    None => {
                        st.prefill_s = t0.elapsed().as_secs_f64();
                        t1 = Some(Instant::now());
                    }
                    Some(_) => {
                        st.spec_steps += 1;
                        st.spec_tokens += toks.len();
                    }
                }
                // The state holds every committed token, also past a stop.
                self.cache.fed(&toks);
                for t in toks {
                    st.generated += 1;
                    if emit(t) == Flow::Stop || st.generated >= max_new {
                        break 'steps;
                    }
                }
            }
            st.decode_s = t1.map_or(0.0, |t| t.elapsed().as_secs_f64());
            if self.drafts == Some(0) {
                st.spec_steps = 0;
            }
        } else {
            let mut sampler = Sampler::new(params, prompt);
            let mut logits = e.logits()?;
            st.prefill_s = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            loop {
                let ts = Instant::now();
                let tok = sampler.sample(&mut logits);
                sampler.accept(tok);
                st.sample_s += ts.elapsed().as_secs_f64();
                st.generated += 1;
                if emit(tok) == Flow::Stop || st.generated >= max_new {
                    break;
                }
                e.step(Some(tok))?;
                self.cache.fed(&[tok]);
                e.synchronize()?;
                let tc = Instant::now();
                logits = e.logits()?;
                st.copy_s += tc.elapsed().as_secs_f64();
            }
            st.decode_s = t1.elapsed().as_secs_f64();
        }
        if has_images {
            self.cache = PrefixCache::new(SNAPSHOTS);
        } else {
            self.cache.snapshot(e)?;
        }
        Ok(st)
    }
}

/// A sequence state that is just its tokens (snapshots are copies): the dry run's engine, and
/// the prefix cache's tests.
#[derive(Debug, Default)]
pub struct TokenState {
    pub seq: Vec<u32>,
}

impl StateEngine for TokenState {
    type Snapshot = Vec<u32>;

    fn reset(&mut self) -> Res<()> {
        self.seq.clear();
        Ok(())
    }

    fn prefill(&mut self, tokens: &[u32]) -> Res<()> {
        self.seq.extend_from_slice(tokens);
        Ok(())
    }

    fn synchronize(&mut self) -> Res<()> {
        Ok(())
    }

    fn save(&mut self) -> Res<Option<Vec<u32>>> {
        Ok(Some(self.seq.clone()))
    }

    fn restore(&mut self, s: &Vec<u32>) -> Res<()> {
        self.seq.clone_from(s);
        Ok(())
    }
}

/// A stand-in for the engine (`tsumuji-serve --dry-run`): "generates" a fixed token sequence at
/// a fixed pace (prefill 1 ms per 8 uncached tokens), through a real prefix cache, so the HTTP
/// side and the cache bookkeeping can be tested without a GPU.
pub struct ScriptGen {
    pub ctx: usize,
    pub script: Vec<u32>,
    pub delay: Duration,
    pub state: TokenState,
    pub cache: PrefixCache<Vec<u32>>,
}

impl ScriptGen {
    pub fn new(ctx: usize, script: Vec<u32>, delay: Duration) -> Self {
        Self {
            ctx,
            script,
            delay,
            state: TokenState::default(),
            cache: PrefixCache::new(SNAPSHOTS),
        }
    }
}

impl Generator for ScriptGen {
    fn ctx(&self) -> usize {
        self.ctx
    }

    fn generate(
        &mut self,
        req: GenRequest<'_>,
        emit: &mut dyn FnMut(u32) -> Flow,
        cancelled: &dyn Fn() -> bool,
    ) -> Res<Stats> {
        if !req.images.is_empty() {
            return Err("dry-run does not interpret images".into());
        }
        let t0 = Instant::now();
        let mut st = Stats {
            sampling: "script",
            ..Default::default()
        };
        let Some(cached) = feed_prompt(
            &mut self.state,
            &mut self.cache,
            req.prompt,
            req.marks,
            req.variant,
            cancelled,
        )?
        else {
            return Ok(st);
        };
        std::thread::sleep(Duration::from_micros(
            125 * (req.prompt.len() - cached) as u64,
        ));
        st.cached = cached;
        st.prefill_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for (i, &t) in self.script.iter().cycle().take(req.max_new).enumerate() {
            std::thread::sleep(self.delay);
            st.generated += 1;
            let stop = emit(t) == Flow::Stop;
            if stop || i + 1 == req.max_new {
                break;
            }
            self.state.seq.push(t);
            self.cache.fed(&[t]);
        }
        self.cache.snapshot(&mut self.state)?;
        st.decode_s = t1.elapsed().as_secs_f64();
        Ok(st)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs one turn: the prompt through the cache, then the `generated` tokens fed (the last one
    /// too, as the greedy loop does). Returns the cached length.
    fn turn(
        st: &mut TokenState,
        c: &mut PrefixCache<Vec<u32>>,
        prompt: &[u32],
        marks: &[usize],
        generated: &[u32],
    ) -> usize {
        turn_as(st, c, prompt, marks, generated, 0)
    }

    fn turn_as(
        st: &mut TokenState,
        c: &mut PrefixCache<Vec<u32>>,
        prompt: &[u32],
        marks: &[usize],
        generated: &[u32],
        variant: u32,
    ) -> usize {
        let cached = feed_prompt(st, c, prompt, marks, variant, &|| false)
            .unwrap()
            .unwrap();
        assert_eq!(st.seq, prompt, "the state must be exactly the prompt");
        st.seq.extend_from_slice(generated);
        c.fed(generated);
        c.snapshot(st).unwrap();
        assert_eq!(st.seq, c.tokens());
        cached
    }

    #[test]
    fn conversation() {
        let (mut st, mut c) = (TokenState::default(), PrefixCache::new(8));
        // 1 = turn start; the prompt ends with a generation prompt [1, 9, 8] that the next request
        // re-renders differently (as when the client drops the reasoning).
        let p1 = [1, 2, 3, 4, 1, 9, 8];
        assert_eq!(turn(&mut st, &mut c, &p1, &[4], &[50, 51, 52]), 0);
        // Turn 2: history + the assistant turn rendered without its reasoning + a new user turn.
        let p2 = [1, 2, 3, 4, 1, 9, 7, 51, 52, 1, 5, 6, 1, 9, 8];
        assert_eq!(
            turn(&mut st, &mut c, &p2, &[12], &[60]),
            4,
            "restored at the last-turn mark of turn 1"
        );
        // The same request again (a retry): back to the last-turn mark (the end-of-prompt
        // snapshot leaves no token to run for the logits).
        assert_eq!(turn(&mut st, &mut c, &p2, &[12], &[61]), 12);
        // A request that continues the generated text exactly: the generation-end snapshot.
        let mut p3 = p2.to_vec();
        p3.extend([61, 1, 5]);
        assert_eq!(turn(&mut st, &mut c, &p3, &[], &[]), 16);
        // A different conversation: nothing shared, start over.
        assert_eq!(turn(&mut st, &mut c, &[7, 7, 7], &[], &[]), 0);
        assert!(c.lengths().iter().all(|&l| l <= 3));
        // An edit in the middle of the history: back to the last snapshot before it.
        let (mut st, mut c) = (TokenState::default(), PrefixCache::new(8));
        turn(&mut st, &mut c, &[1, 2, 1, 3, 1, 4], &[2, 4], &[9]);
        assert_eq!(
            turn(&mut st, &mut c, &[1, 2, 1, 3, 1, 5, 1, 6], &[6], &[]),
            4
        );
    }

    #[test]
    fn variants_do_not_share() {
        let (mut st, mut c) = (TokenState::default(), PrefixCache::new(8));
        turn(&mut st, &mut c, &[1, 2, 3, 1, 4], &[3], &[5]);
        assert_eq!(
            turn_as(&mut st, &mut c, &[1, 2, 3, 1, 4, 5, 1, 6], &[6], &[], 1),
            0,
            "another variant starts over"
        );
        assert_eq!(
            turn_as(&mut st, &mut c, &[1, 2, 3, 1, 4, 5, 1, 7], &[6], &[], 1),
            6
        );
        assert_eq!(
            turn(&mut st, &mut c, &[1, 2, 3, 1, 4, 5, 1, 7], &[], &[]),
            0,
            "and back"
        );
    }

    #[test]
    fn eviction() {
        let (mut st, mut c) = (TokenState::default(), PrefixCache::new(2));
        turn(&mut st, &mut c, &[1, 2, 3, 4], &[1, 2, 3], &[]);
        assert_eq!(c.lengths(), [3, 4]);
    }
}
