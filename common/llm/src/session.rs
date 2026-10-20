//! Four recent conversations; immutable checkpoints precede any reused writes.
use std::{collections::VecDeque, rc::Rc};

pub const CHECKPOINT_TOKENS: usize = 1024;

pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

#[derive(Debug)]
pub struct Checkpoint<S> {
    pub tokens: usize,
    pub state: Rc<S>,
}

#[derive(Debug)]
pub struct Conversation<S, K> {
    pub tokens: Vec<u32>,
    pub checkpoints: Vec<Checkpoint<S>>,
    pub kv: K,
}

#[derive(Debug)]
pub struct PrefixCache<S, K> {
    entries: VecDeque<Conversation<S, K>>,
}

impl<S, K> Default for PrefixCache<S, K> {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }
}

impl<S, K> PrefixCache<S, K> {
    /// Select the furthest saved state inside the common prefix, never a state
    /// after it: truncating QSA alone cannot undo GDN or convolution history.
    pub fn select(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        self.entries
            .iter()
            .enumerate()
            .flat_map(|(entry, c)| {
                let common = common_prefix(&c.tokens, prompt);
                c.checkpoints
                    .iter()
                    .enumerate()
                    .filter(move |(_, p)| p.tokens <= common)
                    .map(move |(point, p)| ((entry, point), p.tokens))
            })
            .max_by_key(|(_, tokens)| *tokens)
            .map(|(indices, _)| indices)
    }

    pub fn entry(&self, index: usize) -> &Conversation<S, K> {
        &self.entries[index]
    }

    pub fn insert(&mut self, conversation: Conversation<S, K>) {
        // A later turn supersedes its ancestor, leaving room for four distinct
        // conversations rather than four requests from the same conversation.
        self.entries
            .retain(|entry| !conversation.tokens.starts_with(&entry.tokens));
        if self.entries.len() == 4 {
            self.entries.pop_front();
        }
        self.entries.push_back(conversation);
    }

    pub fn entries(&self) -> impl Iterator<Item = &Conversation<S, K>> {
        self.entries.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn branching_prefix_and_four_conversations() {
        let mut cache = PrefixCache::default();
        for last in 1..=4 {
            cache.insert(Conversation {
                tokens: vec![1, 2, last],
                kv: (),
                checkpoints: vec![
                    Checkpoint {
                        tokens: 2,
                        state: Rc::new(2),
                    },
                    Checkpoint {
                        tokens: 3,
                        state: Rc::new(last),
                    },
                ],
            });
        }
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(common_prefix(&[], &[1]), 0);
        let (e, p) = cache.select(&[1, 2, 3, 9]).unwrap();
        assert_eq!(cache.entry(e).checkpoints[p].tokens, 3);
        assert_eq!(*cache.entry(e).checkpoints[p].state, 3);
        let (e, p) = cache.select(&[1, 2, 8]).unwrap();
        assert_eq!(cache.entry(e).checkpoints[p].tokens, 2);
        assert!(cache.select(&[9]).is_none());
        cache.insert(Conversation {
            tokens: vec![7],
            checkpoints: vec![],
            kv: (),
        });
        assert_eq!(cache.entries().count(), 4);
        assert_eq!(cache.entry(0).tokens, [1, 2, 2]);
    }
}

use crate::SamplingParams;
use hayate_core::Result;
use std::time::Instant;

/// The runtime orchestrates reuse; model-specific copies and kernels stay in models.
pub trait Engine {
    type State;
    type Kv;
    fn tokens(&self) -> &[u32];
    fn capacity(&self) -> usize;
    fn chunk(&self) -> usize;
    fn reset(&mut self) -> Result<()>;
    fn append(&mut self, ids: &[u32], mtp: bool) -> Result<u32>;
    fn restore(&mut self, state: &Self::State, kv: &Self::Kv) -> Result<u32>;
    fn save(&self, next: u32, mtp: bool) -> Result<Self::State>;
    fn save_resident(&self, next: u32, mtp: bool) -> Result<Self::State> {
        self.save(next, mtp)
    }
    fn save_kv(&self) -> Result<Self::Kv>;
    fn mtp_valid(state: &Self::State) -> bool;
    fn state_bytes(state: &Self::State) -> usize;
    fn kv_bytes(kv: &Self::Kv) -> usize;
    fn idle_checkpoint_bytes(&self) -> usize {
        0
    }
    fn sampling(&mut self, params: SamplingParams, seed: u64) -> Result<()>;
    fn sample(&mut self, greedy: u32) -> Result<u32>;
    fn speculate(&mut self, anchor: u32, n: usize) -> Result<(Vec<u32>, usize)>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn synchronize(&self) -> Result<()>;
    fn eos(&self, token: u32) -> bool {
        matches!(token, 248044 | 248046)
    }
    fn body_start(&self, token: u32) -> bool {
        token == 248069
    }
    fn draft_max(&self) -> usize {
        3
    }
    fn plain_seconds(&self, elapsed: f64) -> f64 {
        elapsed
    }
    fn adaptive(&self) -> bool {
        true
    }
    /// Save a checkpoint every CHECKPOINT_TOKENS. Engines with large recurrent
    /// state (GLM: 136 MiB) keep only prompt points, prompt end, body start and end.
    fn interval_checkpoints(&self) -> bool {
        true
    }
    /// The last error was a cancelled prefill: state is valid up to `tokens()`.
    fn interrupted(&self) -> bool {
        false
    }
}
#[derive(Debug, Default)]
pub struct Timings {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_ms: f64,
    pub render_ms: f64,
    pub tokenize_ms: f64,
    pub sampling_ms: f64,
    pub checkpoint_wait_ms: f64,
    pub prefill_sync_ms: f64,
    pub restore_ms: f64,
    pub forward_ms: f64,
    pub prefill_checkpoint_ms: f64,
    pub decode_checkpoint_ms: f64,
    pub mtp_disabled: bool,
    pub decode_ms: f64,
    pub mtp_proposed: usize,
    pub mtp_accepted: usize,
    pub checkpoint_ms: f64,
    pub cache_host_bytes: usize,
    pub length: bool,
}
impl Timings {
    pub fn decode_tok_s(&self) -> f64 {
        if self.decode_ms > 0. {
            self.completion_tokens.saturating_sub(1) as f64 * 1000. / self.decode_ms
        } else {
            0.
        }
    }
}
#[derive(Debug)]
pub struct Session<E: Engine> {
    pub engine: E,
    cache: PrefixCache<E::State, E::Kv>,
    live: Option<(usize, u32, bool)>,
}
impl<E: Engine> Session<E> {
    pub fn new(engine: E) -> Self {
        Self {
            engine,
            cache: PrefixCache::default(),
            live: None,
        }
    }
    pub fn generate(
        &mut self,
        prompt: &[u32],
        params: SamplingParams,
        seed: u64,
        emit: &mut dyn FnMut(&str) -> Result<bool>,
    ) -> Result<Timings> {
        self.generate_at(prompt, &[], params, seed, emit)
    }
    /// A failed `emit` (client gone) keeps the conversation's saved prompt
    /// checkpoints and KV, so a retried or continued request still reuses them.
    pub fn generate_at(
        &mut self,
        prompt: &[u32],
        prompt_points: &[usize],
        params: SamplingParams,
        seed: u64,
        emit: &mut dyn FnMut(&str) -> Result<bool>,
    ) -> Result<Timings> {
        let mut points = Vec::new();
        let mut failed = false;
        let result = self.run(prompt, prompt_points, params, seed, &mut points, &mut |s| {
            let r = emit(s);
            failed |= r.is_err();
            r
        });
        if result.is_err() && (failed || self.engine.interrupted()) {
            let position = self.engine.tokens().len();
            points.retain(|p| p.tokens <= position);
            if !points.is_empty()
                && let Ok(kv) = self.engine.save_kv()
            {
                self.cache.insert(Conversation {
                    tokens: self.engine.tokens().to_vec(),
                    checkpoints: points,
                    kv,
                });
            }
        }
        result
    }
    fn run(
        &mut self,
        prompt: &[u32],
        prompt_points: &[usize],
        params: SamplingParams,
        seed: u64,
        points: &mut Vec<Checkpoint<E::State>>,
        emit: &mut dyn FnMut(&str) -> Result<bool>,
    ) -> Result<Timings> {
        use hayate_core::Error;
        if prompt.is_empty()
            || params.max_tokens == 0
            || prompt
                .len()
                .checked_add(params.max_tokens as usize)
                .is_none_or(|n| n > self.engine.capacity())
        {
            return Err(Error::InvalidPlan(
                "prompt + max_tokens exceeds server context".into(),
            ));
        }
        let mut stats = Timings {
            prompt_tokens: prompt.len(),
            ..Timings::default()
        };
        let start = Instant::now();
        let wait = Instant::now();
        self.engine.synchronize()?;
        stats.checkpoint_wait_ms = wait.elapsed().as_secs_f64() * 1000.;
        let sampling = Instant::now();
        self.engine.sampling(params, seed)?;
        stats.sampling_ms = sampling.elapsed().as_secs_f64() * 1000.;
        let restore = Instant::now();
        let interval = self.engine.interval_checkpoints();
        let mut next = 0;
        let mut mtp = params.temperature == 0. && self.engine.draft_max() > 0;
        let live = self.live.take().filter(|(n, _, _)| {
            *n == self.engine.tokens().len() && prompt.starts_with(self.engine.tokens())
        });
        if let Some((entry, point)) = self.cache.select(prompt) {
            let entry = self.cache.entry(entry);
            let state = &entry.checkpoints[point];
            if let Some((n, token, valid)) = live.filter(|(n, _, _)| *n >= state.tokens) {
                next = token;
                stats.cached_tokens = n;
                mtp &= valid;
            } else {
                next = self.engine.restore(&state.state, &entry.kv)?;
                stats.cached_tokens = state.tokens;
                mtp &= E::mtp_valid(&state.state);
            }
            points.extend(
                entry
                    .checkpoints
                    .iter()
                    .filter(|p| p.tokens <= stats.cached_tokens)
                    .map(|p| Checkpoint {
                        tokens: p.tokens,
                        state: p.state.clone(),
                    }),
            );
        } else {
            self.engine.reset()?;
        }
        stats.restore_ms = restore.elapsed().as_secs_f64() * 1000.;
        let mut position = stats.cached_tokens;
        while position < prompt.len() {
            let forward = Instant::now();
            let mut end = prompt.len().min(position + self.engine.chunk());
            if let Some(&point) = prompt_points.iter().filter(|&&p| p > position).min() {
                end = end.min(point);
            }
            next = self.engine.append(&prompt[position..end], mtp)?;
            stats.forward_ms += forward.elapsed().as_secs_f64() * 1000.;
            position = end;
            if interval && position.is_multiple_of(CHECKPOINT_TOKENS)
                || position == prompt.len()
                || prompt_points.contains(&position)
            {
                let checkpoint = Instant::now();
                points.push(Checkpoint {
                    tokens: position,
                    state: Rc::new(
                        if position == prompt.len() || prompt_points.contains(&position) {
                            self.engine.save_resident(next, mtp)?
                        } else {
                            self.engine.save(next, mtp)?
                        },
                    ),
                });
                stats.prefill_checkpoint_ms += checkpoint.elapsed().as_secs_f64() * 1000.;
            }
            // An empty chunk between prefill chunks lets a disconnected client
            // stop a long prefill (the caller's emit sees the closed stream).
            if position < prompt.len() && emit("")? {
                return Err(Error::InvalidPlan("stopped during prefill".into()));
            }
        }
        let wait = Instant::now();
        self.engine.synchronize()?;
        stats.prefill_sync_ms = wait.elapsed().as_secs_f64() * 1000.;
        stats.prefill_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let mut generated = Vec::new();
        let mut body_points = Vec::new();
        let mut plain_seconds = 0.;
        let mut plain_steps = 0;
        let mut spec_seconds = 0.;
        let mut spec_steps = 0;
        let mut spec_emitted = 0;
        let mut pending_text = Vec::new();
        let mut batch = vec![self.engine.sample(next)?];
        'decode: loop {
            for token in batch {
                generated.push(token);
                if self.engine.eos(token) {
                    break 'decode;
                }
                pending_text.push(token);
                let decoded = self.engine.decode(&pending_text)?;
                // Retain only the unfinished UTF-8 tail's tokens. Re-decoding
                // the complete response each step is quadratic in output length.
                if !decoded.ends_with('\u{fffd}') {
                    pending_text.clear();
                    if !decoded.is_empty() && emit(&decoded)? {
                        break 'decode;
                    }
                }
                if generated.len() == params.max_tokens as usize {
                    stats.length = true;
                    break 'decode;
                }
            }
            let remaining = params.max_tokens as usize - generated.len();
            let boundary = CHECKPOINT_TOKENS - self.engine.tokens().len() % CHECKPOINT_TOKENS;
            let anchor = *generated.last().unwrap();
            let body_start = self.engine.body_start(anchor);
            let n = self
                .engine
                .draft_max()
                .min(remaining.saturating_sub(1))
                .min(boundary.saturating_sub(1));
            let calibrating = self.engine.adaptive() && plain_steps < 8;
            if mtp && !calibrating && !body_start && n > 0 {
                let step = Instant::now();
                let (tokens, accepted) = self.engine.speculate(*generated.last().unwrap(), n)?;
                spec_seconds += step.elapsed().as_secs_f64();
                spec_steps += 1;
                spec_emitted += tokens.len();
                stats.mtp_proposed += n;
                stats.mtp_accepted += accepted;
                next = *tokens.last().unwrap();
                batch = tokens;
                // Same eight-step baseline/window as the run path. Compare cost
                // per emitted token, including draft, verification and repair.
                if self.engine.adaptive()
                    && spec_steps >= 8
                    && plain_steps > 0
                    && spec_seconds / spec_emitted as f64 >= plain_seconds / plain_steps as f64
                {
                    mtp = false;
                    stats.mtp_disabled = true;
                }
            } else {
                let step = Instant::now();
                next = self.engine.append(&[anchor], mtp)?;
                plain_seconds += self.engine.plain_seconds(step.elapsed().as_secs_f64());
                plain_steps += 1;
                batch = vec![self.engine.sample(next)?];
            }
            let position = self.engine.tokens().len();
            if interval && position.is_multiple_of(CHECKPOINT_TOKENS) || body_start {
                let checkpoint = Instant::now();
                if body_start {
                    body_points.push(position);
                }
                points.push(Checkpoint {
                    tokens: position,
                    state: Rc::new(self.engine.save(next, mtp)?),
                });
                stats.decode_checkpoint_ms += checkpoint.elapsed().as_secs_f64() * 1000.;
            }
        }
        let decoded = self.engine.decode(&pending_text)?;
        if !decoded.is_empty() {
            emit(&decoded)?;
        }
        self.engine.synchronize()?;
        stats.decode_ms = start.elapsed().as_secs_f64() * 1000.;
        stats.completion_tokens = generated.len();
        let start = Instant::now();
        let position = self.engine.tokens().len();
        if points.last().is_none_or(|p| p.tokens != position) {
            points.push(Checkpoint {
                tokens: position,
                state: Rc::new(self.engine.save(next, mtp)?),
            });
        }
        // Bound recurrent storage by context length, even for many short turns.
        // Keep interval checkpoints and this request's prompt/end states.
        points.retain(|p| {
            interval && p.tokens.is_multiple_of(CHECKPOINT_TOKENS)
                || p.tokens == prompt.len()
                || p.tokens == position
                || prompt_points.contains(&p.tokens)
                || body_points.contains(&p.tokens)
        });
        let conversation = Conversation {
            tokens: self.engine.tokens().to_vec(),
            kv: self.engine.save_kv()?,
            checkpoints: std::mem::take(points),
        };
        self.cache.insert(conversation);
        self.live = Some((position, next, mtp));
        stats.checkpoint_ms = start.elapsed().as_secs_f64() * 1000.;
        // Shared recurrent checkpoints are charged once.
        let mut seen = std::collections::HashSet::new();
        stats.cache_host_bytes = self.engine.idle_checkpoint_bytes();
        for entry in self.cache.entries() {
            stats.cache_host_bytes += E::kv_bytes(&entry.kv);
            for p in &entry.checkpoints {
                if seen.insert(Rc::as_ptr(&p.state)) {
                    stats.cache_host_bytes += E::state_bytes(&p.state);
                }
            }
        }
        Ok(stats)
    }
}

#[cfg(test)]
mod generation_tests {
    use super::*;
    #[derive(Clone, Debug, Default)]
    struct State {
        history: Vec<u32>,
        recurrent: u64,
        conv: Vec<u32>,
        next: u32,
        mtp: bool,
    }
    #[derive(Debug, Default)]
    struct Cpu {
        state: State,
        appended: usize,
        script: Vec<u32>,
        slow_speculation: bool,
        restores: usize,
        chunk: usize,
        batches: Vec<usize>,
        no_interval: bool,
    }
    impl Engine for Cpu {
        fn interval_checkpoints(&self) -> bool {
            !self.no_interval
        }
        type State = State;
        type Kv = Vec<u32>;
        fn tokens(&self) -> &[u32] {
            &self.state.history
        }
        fn capacity(&self) -> usize {
            10000
        }
        fn chunk(&self) -> usize {
            if self.chunk == 0 { 1024 } else { self.chunk }
        }
        fn reset(&mut self) -> Result<()> {
            self.state = State::default();
            Ok(())
        }
        fn append(&mut self, ids: &[u32], _: bool) -> Result<u32> {
            self.batches.push(ids.len());
            for &id in ids {
                self.state.recurrent = self
                    .state
                    .recurrent
                    .wrapping_mul(31)
                    .wrapping_add(u64::from(id));
                self.state.conv.push(id);
                if self.state.conv.len() > 9 {
                    self.state.conv.remove(0);
                }
                self.state.history.push(id);
                self.appended += 1;
            }
            self.state.next = self
                .script
                .get(self.state.history.len() - 1)
                .copied()
                .unwrap_or(33 + (self.state.recurrent % 80) as u32);
            Ok(self.state.next)
        }
        fn restore(&mut self, state: &State, kv: &Vec<u32>) -> Result<u32> {
            self.restores += 1;
            assert_eq!(state.history, kv[..state.history.len()]);
            self.state = state.clone();
            Ok(state.next)
        }
        fn save(&self, next: u32, mtp: bool) -> Result<State> {
            assert_eq!(next, self.state.next);
            let mut state = self.state.clone();
            state.mtp = mtp;
            Ok(state)
        }
        fn save_kv(&self) -> Result<Vec<u32>> {
            Ok(self.state.history.clone())
        }
        fn mtp_valid(state: &State) -> bool {
            state.mtp
        }
        fn state_bytes(_: &State) -> usize {
            100
        }
        fn kv_bytes(kv: &Vec<u32>) -> usize {
            kv.len() * 4
        }
        fn sampling(&mut self, _: SamplingParams, _: u64) -> Result<()> {
            Ok(())
        }
        fn sample(&mut self, greedy: u32) -> Result<u32> {
            Ok(greedy)
        }
        fn speculate(&mut self, anchor: u32, n: usize) -> Result<(Vec<u32>, usize)> {
            if self.slow_speculation {
                std::thread::sleep(std::time::Duration::from_millis(1));
                return Ok((vec![self.append(&[anchor], true)?], 0));
            }
            let mut result = Vec::new();
            let mut token = anchor;
            for _ in 0..=n {
                token = self.append(&[token], true)?;
                result.push(token);
            }
            Ok((result, n))
        }
        fn decode(&self, tokens: &[u32]) -> Result<String> {
            if !self.script.is_empty() {
                return Ok(String::from_utf8_lossy(
                    &tokens.iter().map(|&v| v as u8).collect::<Vec<_>>(),
                )
                .into_owned());
            }
            Ok(tokens.iter().map(|&v| char::from_u32(v).unwrap()).collect())
        }
        fn synchronize(&self) -> Result<()> {
            Ok(())
        }
    }
    fn run(session: &mut Session<Cpu>, prompt: &[u32], temperature: f32) -> (String, Timings) {
        let mut text = String::new();
        let stats = session
            .generate(
                prompt,
                SamplingParams {
                    temperature,
                    max_tokens: 12,
                    ..SamplingParams::default()
                },
                42,
                &mut |s| {
                    text.push_str(s);
                    Ok(false)
                },
            )
            .unwrap();
        (text, stats)
    }
    #[test]
    fn resumed_recurrence_conv_and_kv_equal_fresh_with_suffix_only_prefill() {
        let prompt = (1..=1100).collect::<Vec<_>>();
        let mut session = Session::new(Cpu::default());
        let (first, _) = run(&mut session, &prompt, 0.);
        let mut second = prompt.clone();
        second.extend(first.chars().map(|c| c as u32));
        second.extend([5, 6, 7]);
        let (warm, stats) = run(&mut session, &second, 0.);
        let (cold, _) = run(&mut Session::new(Cpu::default()), &second, 0.);
        assert_eq!(warm, cold);
        assert_eq!(stats.cached_tokens, prompt.len() + first.len() - 1);
        assert_eq!(stats.prompt_tokens - stats.cached_tokens, 4);
        assert_eq!(session.engine.restores, 0);
        assert!(stats.mtp_proposed > 0);
        let mut branched = prompt[..1050].to_vec();
        branched.extend([9, 10]);
        let (warm, stats) = run(&mut session, &branched, 0.);
        let (cold, _) = run(&mut Session::new(Cpu::default()), &branched, 0.);
        assert_eq!(warm, cold);
        assert_eq!(stats.cached_tokens, 1024);
        assert_eq!(session.engine.restores, 1);
    }
    #[test]
    fn switching_conversations_and_failed_stream_cannot_reuse_live_state() {
        let a = vec![1; 40];
        let b = vec![2; 40];
        let mut session = Session::new(Cpu::default());
        run(&mut session, &a, 0.6);
        run(&mut session, &b, 0.6);
        let (actual, _) = run(&mut session, &a, 0.6);
        assert_eq!(actual, run(&mut Session::new(Cpu::default()), &a, 0.6).0);
        assert_eq!(session.engine.restores, 1);
        let continued = session.engine.tokens().to_vec();
        let result = session.generate(&continued, SamplingParams::default(), 42, &mut |_| {
            Err(hayate_core::Error::InvalidPlan("disconnected".into()))
        });
        assert!(result.is_err());
        assert!(session.live.is_none());
        let (actual, _) = run(&mut session, &a, 0.6);
        assert_eq!(actual, run(&mut Session::new(Cpu::default()), &a, 0.6).0);
        assert_eq!(session.engine.restores, 2);
    }
    #[test]
    fn long_suffix_prefills_in_one_batch_before_thinking_header() {
        let mut session = Session::new(Cpu {
            chunk: 4096,
            ..Cpu::default()
        });
        let prompt = vec![100; 2104];
        let stats = session
            .generate_at(
                &prompt,
                &[2100],
                SamplingParams {
                    temperature: 0.6,
                    max_tokens: 1,
                    ..SamplingParams::default()
                },
                42,
                &mut |_| Ok(false),
            )
            .unwrap();
        assert_eq!(session.engine.batches, [2100, 4]);
        assert_eq!(stats.prompt_tokens, 2104);
        assert_eq!(stats.cached_tokens, 0);
        let mut next = prompt[..2100].to_vec();
        next.extend([200; 2104]);
        session.engine.batches.clear();
        let stats = session
            .generate_at(
                &next,
                &[4200],
                SamplingParams {
                    temperature: 0.6,
                    max_tokens: 1,
                    ..SamplingParams::default()
                },
                42,
                &mut |_| Ok(false),
            )
            .unwrap();
        assert_eq!(stats.cached_tokens, 2100);
        assert_eq!(session.engine.batches, [2100, 4]);
    }
    #[test]
    fn exact_replay_and_stochastic_state_disable_mtp() {
        let prompt = (1..=40).collect::<Vec<_>>();
        let mut session = Session::new(Cpu::default());
        let (first, _) = run(&mut session, &prompt, 0.6);
        let (second, stats) = run(&mut session, &prompt, 0.);
        assert_eq!(first, second);
        assert_eq!(stats.cached_tokens, 40);
        assert_eq!(stats.mtp_proposed, 0);
    }
    #[test]
    fn low_acceptance_stops_mtp_and_matches_plain_greedy() {
        let prompt = [1, 2, 3, 4];
        let mut session = Session::new(Cpu {
            slow_speculation: true,
            ..Cpu::default()
        });
        let params = SamplingParams {
            temperature: 0.,
            max_tokens: 64,
            ..SamplingParams::default()
        };
        let mut text = String::new();
        let stats = session
            .generate(&prompt, params, 42, &mut |s| {
                text.push_str(s);
                Ok(false)
            })
            .unwrap();
        assert!(stats.mtp_disabled);
        assert_eq!(stats.mtp_proposed, 8 * 3);
        assert_eq!(stats.mtp_accepted, 0);
        let mut plain = Session::new(Cpu::default());
        let mut expected = String::new();
        plain
            .generate(
                &prompt,
                SamplingParams {
                    temperature: 0.6,
                    ..params
                },
                42,
                &mut |s| {
                    expected.push_str(s);
                    Ok(false)
                },
            )
            .unwrap();
        assert_eq!(text, expected);
    }
    #[test]
    fn ten_turns_without_reasoning_reuse_each_previous_prompt_header() {
        let mut session = Session::new(Cpu::default());
        let mut prompt = vec![1, 2, 3, 248068];
        let mut previous_header = 0;
        for _ in 0..10 {
            let header = prompt.len() - 1;
            let mut script = vec![248046; prompt.len() + 8];
            script[prompt.len() - 1] = b'R' as u32;
            script[prompt.len()] = 248069;
            script[prompt.len() + 1] = b'A' as u32;
            session.engine.script = script;
            let stats = session
                .generate_at(
                    &prompt,
                    &[header],
                    SamplingParams {
                        temperature: 0.6,
                        max_tokens: 8,
                        ..SamplingParams::default()
                    },
                    42,
                    &mut |_| Ok(false),
                )
                .unwrap();
            assert_eq!(stats.cached_tokens, previous_header);
            let (_, body) = session
                .cache
                .select(&[prompt.clone(), vec![b'R' as u32, 248069, 9]].concat())
                .unwrap();
            let (entry, _) = session.cache.select(&prompt).unwrap();
            assert_eq!(
                session.cache.entry(entry).checkpoints[body].tokens,
                prompt.len() + 2
            );
            previous_header = header;
            // Studio omits reasoning and the old thinking header on re-render.
            prompt.pop();
            prompt.extend([b'A' as u32, 20, 21, 22, 248068]);
        }
    }
    #[test]
    fn disconnect_keeps_prompt_checkpoints_and_prefill_observes_it() {
        let params = SamplingParams {
            temperature: 0.6,
            max_tokens: 8,
            ..SamplingParams::default()
        };
        let prompt = (0..2500).map(|i| 40 + i % 50).collect::<Vec<_>>();
        let mut session = Session::new(Cpu {
            chunk: 100,
            no_interval: true,
            ..Cpu::default()
        });
        // The client goes away at the first generated text.
        let mut keepalives = 0;
        let result = session.generate_at(&prompt, &[2000], params, 1, &mut |s| {
            if s.is_empty() {
                keepalives += 1;
                Ok(false)
            } else {
                Err(hayate_core::Error::InvalidPlan("disconnected".into()))
            }
        });
        assert!(result.is_err());
        assert_eq!(keepalives, 24);
        let (entry, _) = session.cache.select(&prompt).unwrap();
        let kept = session
            .cache
            .entry(entry)
            .checkpoints
            .iter()
            .map(|p| p.tokens)
            .collect::<Vec<_>>();
        assert_eq!(kept, [2000, 2500]);
        let mut next = prompt.clone();
        next.extend([7, 8, 9]);
        session.engine.batches.clear();
        let stats = session
            .generate_at(&next, &[], params, 1, &mut |_| Ok(false))
            .unwrap();
        assert_eq!(stats.cached_tokens, 2500);
        assert_eq!(session.engine.batches[0], 3);
        // A disconnect seen by an empty prefill chunk stops the prefill early
        // and still keeps the checkpoint it passed.
        let mut session = Session::new(Cpu {
            chunk: 100,
            no_interval: true,
            ..Cpu::default()
        });
        let mut seen = 0;
        let result = session.generate_at(&prompt, &[300], params, 1, &mut |_| {
            seen += 1;
            if seen == 5 {
                Err(hayate_core::Error::InvalidPlan("disconnected".into()))
            } else {
                Ok(false)
            }
        });
        assert!(result.is_err());
        assert_eq!(session.engine.tokens().len(), 500);
        let (entry, point) = session.cache.select(&prompt).unwrap();
        assert_eq!(session.cache.entry(entry).checkpoints[point].tokens, 300);
    }
    #[test]
    fn byte_bpe_stream_holds_incomplete_utf8_and_flushes_at_length() {
        for (script, max_tokens, expected) in [
            (vec![0xe6, 0x97, 0xa5, 248046], 12, "日"),
            (vec![0xef, 0xbf, 0xbd, 248046], 12, "\u{fffd}"),
            (vec![0xe6, 0x97], 2, "\u{fffd}"),
        ] {
            let mut session = Session::new(Cpu {
                script,
                ..Cpu::default()
            });
            let mut chunks = Vec::new();
            session
                .generate(
                    &[1],
                    SamplingParams {
                        temperature: 0.6,
                        max_tokens,
                        ..SamplingParams::default()
                    },
                    42,
                    &mut |s| {
                        chunks.push(s.to_owned());
                        Ok(false)
                    },
                )
                .unwrap();
            assert_eq!(chunks, [expected]);
        }
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    #[test]
    fn prompt_header_and_body_boundaries_survive_reasoning_removal() {
        let mut cache = PrefixCache::default();
        // The next rendering can remove <think>, its contents, or the entire
        // generated suffix. Each checkpoint is valid only for identical IDs.
        let tokens = vec![1, 2, 3, 248068, 4, 5, 248069, 6, 7];
        cache.insert(Conversation {
            tokens: tokens.clone(),
            checkpoints: [3, 4, 7, 9]
                .into_iter()
                .map(|tokens| Checkpoint {
                    tokens,
                    state: Rc::new(tokens),
                })
                .collect(),
            kv: (),
        });
        for (prompt, expected) in [
            (vec![1, 2, 3, 9], 3),
            (vec![1, 2, 3, 248068, 248069, 6], 4),
            (vec![1, 2, 3, 248068, 4, 5, 248069, 9], 7),
            ([tokens, vec![8]].concat(), 9),
        ] {
            let (entry, point) = cache.select(&prompt).unwrap();
            assert_eq!(cache.entry(entry).checkpoints[point].tokens, expected);
        }
    }
}
