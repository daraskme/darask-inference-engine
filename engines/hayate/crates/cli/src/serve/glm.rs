//! GLM-5.3-Flash chat engine for serve: host sampling (temperature, top-p/k,
//! min-p, repetition penalty), a reasoning budget, and prefix reuse with
//! recurrent checkpoints plus KV blocks shared between saved conversations.
use super::*;
use hayate_kernels::runtime::Device;
use hayate_llm::{SamplingParams, sample::Sampler, session::Engine};
use hayate_models::glm53::{Executor, Options, PrefillOutcome, SessionState, Tokenizer};
use std::{cell::RefCell, rc::Rc};

pub const EOS: [u32; 3] = [154820, 154827, 154829];
pub const THINK_END: u32 = 154842;
/// Turn-opening added tokens; the prompt is tokenized per turn between them.
pub const MARKERS: [&str; 4] = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"];
const USER: u32 = 154827;
const OBSERVATION: u32 = 154829;
/// Non-special added tokens decode to their text (HF skip_special_tokens).
const TEXT_TOKENS: [(u32, &str); 18] = [
    (154838, "<|code_prefix|>"),
    (154839, "<|code_middle|>"),
    (154840, "<|code_suffix|>"),
    (154841, "<think>"),
    (154842, "</think>"),
    (154843, "<tool_call>"),
    (154844, "</tool_call>"),
    (154845, "<tool_response>"),
    (154846, "</tool_response>"),
    (154847, "<arg_key>"),
    (154848, "</arg_key>"),
    (154849, "<arg_value>"),
    (154850, "</arg_value>"),
    (154851, "/nothink"),
    (154852, "<|begin_of_box|>"),
    (154853, "<|end_of_box|>"),
    (154854, "<|image|>"),
    (154855, "<|video|>"),
];
/// Automatic load budget: free VRAM minus this margin ...
/// Prefill sizes its workspace to the VRAM that is free when it runs (smaller
/// chunks, then a cache loan), so a tight margin does not risk an
/// out-of-memory error: with an 84 / 90 / 92 GiB budget a 34K-token prefill
/// peaked at the same 92.5 GiB of the 95.6 GiB card (stage 21 measurements).
pub const GLM_MARGIN: u64 = 2 << 30;
/// ... and at most this much. Each GiB buys ~86 resident experts (LOWBIT mix).
/// Agent loop (34K prompt, 2.2K per turn, 128-token replies, 150 GB LOWBIT
/// mix; big prompts borrow cache slots, so the cap no longer slows the cold
/// prefill much): 84 GiB 17.0-20.1 tok/s, cold prefill 32.6 s; 90 GiB 22.2-22.7
/// tok/s, 34.1 s; 92 GiB 21.2-25.1 tok/s, 39.1 s. Peak VRAM at 90 GiB leaves
/// 2.7 GiB of the card free.
pub const GLM_AUTO_MAX: u64 = 90 << 30;
/// KV is copied and shared between saved conversations in blocks of this many tokens.
const BLOCK: usize = 1024;
/// Pooled 136 MiB pinned checkpoints kept for reuse after they are dropped.
const POOL: usize = 4;

/// KV of `tokens`, in BLOCK-token pieces shared with other snapshots.
#[derive(Debug)]
pub struct Kv {
    tokens: Vec<u32>,
    blocks: Vec<Rc<Vec<u8>>>,
}
#[derive(Debug)]
pub struct State {
    inner: Option<SessionState>,
    pool: Rc<RefCell<Vec<SessionState>>>,
    logits: Vec<u16>,
    next: u32,
}
impl Drop for State {
    fn drop(&mut self) {
        let mut pool = self.pool.borrow_mut();
        if pool.len() < POOL
            && let Some(state) = self.inner.take()
        {
            pool.push(state);
        }
    }
}

pub struct Glm<'d> {
    pub model: Executor<'d>,
    pub tokenizer: Tokenizer,
    pub prompt_tokens: PromptTokens,
    history: Vec<u32>,
    logits: Vec<u16>,
    sampler: Sampler,
    params: SamplingParams,
    pool: Rc<RefCell<Vec<SessionState>>>,
    /// Blocks whose tokens match the live history are already in the live KV.
    /// The last KV saved or restored: blocks the live KV still matches are
    /// shared instead of read back.
    base: RefCell<Rc<Kv>>,
    chunk: usize,
    // Per request.
    pub reasoning_budget: Option<usize>,
    pub repetition_penalty: f32,
    pub thinking: bool,
    reasoning: usize,
    seen: Vec<bool>,
    seen_ids: Vec<u32>,
    seen_len: usize,
    interrupted: bool,
}

impl<'d> Glm<'d> {
    pub fn load(
        device: &'d Device,
        config: &Config,
        vram_budget: u64,
        loras: &[hayate_models::glm53::lora::Spec],
    ) -> Result<Self> {
        let tokenizer = Tokenizer::open(config.glm_source.join("tokenizer.json"))?;
        for (id, text) in TEXT_TOKENS {
            if tokenizer.encode(text)? != [id] {
                return Err(Error::Weights(format!("GLM tokenizer: {text} is not {id}")));
            }
        }
        let frequencies = match &config.glm_frequencies {
            Some(path) if path.is_file() => {
                let bytes = std::fs::read(path)?;
                Some(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect(),
                )
            }
            _ => None,
        };
        let options = Options {
            context: config.glm_context,
            vram_budget: Some(vram_budget),
            host_bytes: (config.glm_host_gb > 0).then(|| config.glm_host_gb * 1_000_000_000),
            allow_disk: true,
            frequencies,
            decode_device: true,
            loras: loras.to_vec(),
            ..Options::default()
        };
        let model = Executor::load(device, &config.glm_hyt, options)?;
        let vocab = model.session_vocab();
        Ok(Self {
            model,
            tokenizer,
            prompt_tokens: PromptTokens::default(),
            history: Vec::new(),
            logits: Vec::with_capacity(vocab),
            sampler: Sampler::new(0),
            params: SamplingParams::default(),
            pool: Rc::default(),
            base: RefCell::new(Rc::new(Kv {
                tokens: vec![],
                blocks: vec![],
            })),
            chunk: config.glm_chunk,
            reasoning_budget: None,
            repetition_penalty: 1.,
            thinking: false,
            reasoning: 0,
            seen: vec![false; vocab],
            seen_ids: Vec::new(),
            seen_len: 0,
            interrupted: false,
        })
    }
    /// Checkpoints inside the prompt: before the newest user/tool turn (a
    /// trailing note Studio does not keep, a cancelled turn) and, for long
    /// system/tool prefixes, before the first user turn (history compaction).
    pub fn prompt_points(prompt: &[u32]) -> Vec<usize> {
        let mut points = Vec::new();
        if let Some(first) = prompt.iter().position(|&t| t == USER)
            && first >= 2048
        {
            points.push(first);
        }
        if let Some(last) = prompt
            .iter()
            .rposition(|&t| t == USER || t == OBSERVATION)
            .filter(|&p| p > 0)
        {
            points.push(last);
        }
        points.dedup();
        points
    }
    fn fold_seen(&mut self) {
        if self.history.len() < self.seen_len {
            for &id in &self.seen_ids {
                self.seen[id as usize] = false;
            }
            self.seen_ids.clear();
            self.seen_len = 0;
        }
        for &id in &self.history[self.seen_len..] {
            if let Some(seen) = self.seen.get_mut(id as usize)
                && !*seen
            {
                *seen = true;
                self.seen_ids.push(id);
            }
        }
        self.seen_len = self.history.len();
    }
}

fn blocks(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n.div_ceil(BLOCK)).map(move |i| (i * BLOCK, ((i + 1) * BLOCK).min(n)))
}

impl Engine for Glm<'_> {
    type State = State;
    type Kv = Rc<Kv>;
    fn tokens(&self) -> &[u32] {
        &self.history
    }
    fn capacity(&self) -> usize {
        self.model.session_context()
    }
    fn chunk(&self) -> usize {
        self.chunk
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset()?;
        self.history.clear();
        Ok(())
    }
    fn append(&mut self, ids: &[u32], _: bool) -> Result<u32> {
        let next = if ids.len() > 1 {
            match self.model.prefill_cancellable(ids)? {
                PrefillOutcome::Done(next) => next,
                PrefillOutcome::Cancelled { position } => {
                    // The client went away (cancel_handle, set from the
                    // connection thread): keep what was prefilled.
                    let done = position.saturating_sub(self.history.len()).min(ids.len());
                    self.history.extend_from_slice(&ids[..done]);
                    self.interrupted = true;
                    return Err(Error::InvalidPlan(
                        "client disconnected during prefill".into(),
                    ));
                }
            }
        } else {
            self.model.session_append(ids)?
        };
        self.history.extend_from_slice(ids);
        self.model.session_logits(&mut self.logits)?;
        Ok(next)
    }
    fn restore(&mut self, state: &State, kv: &Rc<Kv>) -> Result<u32> {
        let inner = state.inner.as_ref().unwrap();
        let n = inner.position();
        if n > kv.tokens.len() {
            return Err(Error::InvalidPlan("GLM checkpoint beyond its KV".into()));
        }
        let common = hayate_llm::session::common_prefix(&self.history, &kv.tokens);
        if common < n {
            // Another conversation overwrote the live KV: copy back only the
            // blocks from the first one that differs.
            for (i, (start, end)) in blocks(kv.tokens.len()).enumerate().skip(common / BLOCK) {
                if start >= n {
                    break;
                }
                self.model.session_kv_write(start, end, &kv.blocks[i])?;
            }
            *self.base.borrow_mut() = kv.clone();
        }
        self.model.session_restore(inner)?;
        self.history.clear();
        self.history.extend_from_slice(&kv.tokens[..n]);
        self.logits.clone_from(&state.logits);
        Ok(state.next)
    }
    fn save(&self, next: u32, _: bool) -> Result<State> {
        let reuse = self.pool.borrow_mut().pop();
        Ok(State {
            inner: Some(self.model.session_save(reuse)?),
            pool: self.pool.clone(),
            logits: self.logits.clone(),
            next,
        })
    }
    fn save_kv(&self) -> Result<Rc<Kv>> {
        let n = self.history.len();
        let saved = self.base.borrow().clone();
        let shared = hayate_llm::session::common_prefix(&saved.tokens, &self.history);
        let base = blocks(saved.tokens.len()).collect::<Vec<_>>();
        let mut out = Vec::new();
        for (i, (start, end)) in blocks(n).enumerate() {
            out.push(match base.get(i) {
                Some(&(_, e)) if e == end && end <= shared => saved.blocks[i].clone(),
                _ => Rc::new(self.model.session_kv_read(start, end)?),
            });
        }
        let kv = Rc::new(Kv {
            tokens: self.history.clone(),
            blocks: out,
        });
        *self.base.borrow_mut() = kv.clone();
        Ok(kv)
    }
    fn mtp_valid(_: &State) -> bool {
        false
    }
    fn state_bytes(state: &State) -> usize {
        state.inner.as_ref().map_or(0, SessionState::bytes) + state.logits.len() * 2
    }
    fn kv_bytes(kv: &Rc<Kv>) -> usize {
        kv.blocks.iter().map(|b| b.len()).sum()
    }
    fn idle_checkpoint_bytes(&self) -> usize {
        self.pool.borrow().iter().map(SessionState::bytes).sum()
    }
    fn sampling(&mut self, params: SamplingParams, seed: u64) -> Result<()> {
        self.params = params;
        self.sampler.reseed(seed);
        self.reasoning = 0;
        self.interrupted = false;
        Ok(())
    }
    fn sample(&mut self, greedy: u32) -> Result<u32> {
        let penalty = self.repetition_penalty;
        let mut token = if self.params.temperature == 0. && penalty == 1. {
            greedy
        } else {
            if penalty != 1. {
                self.fold_seen();
            }
            self.sampler
                .sample(&self.logits, &self.params, penalty, &self.seen_ids)?
        };
        if self.thinking {
            if token == THINK_END {
                self.thinking = false;
            } else if self
                .reasoning_budget
                .is_some_and(|budget| self.reasoning >= budget)
                && !EOS.contains(&token)
            {
                // Budget spent: close the thought and continue with the answer.
                token = THINK_END;
                self.thinking = false;
            } else {
                self.reasoning += 1;
            }
        }
        Ok(token)
    }
    fn speculate(&mut self, _: u32, _: usize) -> Result<(Vec<u32>, usize)> {
        Err(Error::InvalidPlan("GLM serve runs without MTP".into()))
    }
    fn draft_max(&self) -> usize {
        0
    }
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut out = String::new();
        let mut begin = 0;
        for (i, token) in tokens.iter().enumerate() {
            if *token < EOS[0] {
                continue;
            }
            out.push_str(&self.tokenizer.decode(&tokens[begin..i])?);
            if let Some((_, text)) = TEXT_TOKENS.iter().find(|(id, _)| id == token) {
                out.push_str(text);
            }
            begin = i + 1;
        }
        out.push_str(&self.tokenizer.decode(&tokens[begin..])?);
        Ok(out)
    }
    fn synchronize(&self) -> Result<()> {
        self.model.session_synchronize()
    }
    fn eos(&self, token: u32) -> bool {
        EOS.contains(&token)
    }
    fn body_start(&self, token: u32) -> bool {
        token == THINK_END
    }
    fn interval_checkpoints(&self) -> bool {
        false
    }
    fn interrupted(&self) -> bool {
        self.interrupted
    }
}
