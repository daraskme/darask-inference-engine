use hayate_core::{Error, Result};
use hayate_server::{Backend, json::Value};
use std::path::PathBuf;
#[cfg(feature = "cuda")]
mod glm;

#[cfg(any(test, feature = "cuda"))]
#[derive(Default)]
struct PromptTokens(Vec<(String, Vec<u32>)>);
#[cfg(any(test, feature = "cuda"))]
impl PromptTokens {
    /// Re-encode only the turns that changed. `markers` are added tokens that
    /// open a turn (`<|im_start|>`; GLM's role tokens).
    fn encode(
        &mut self,
        prompt: &str,
        markers: &[&str],
        mut encode: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<Vec<u32>> {
        // Added tokens separate BPE and NFC runs. Splitting before one is
        // exact even when the preceding message ends in combining marks.
        let mut parts = Vec::new();
        let mut ids = Vec::new();
        let mut begins = vec![0];
        for marker in markers {
            begins.extend(
                prompt
                    .match_indices(marker)
                    .map(|(i, _)| i)
                    .filter(|&i| i > 0),
            );
        }
        begins.sort_unstable();
        begins.dedup();
        begins.push(prompt.len());
        for (i, range) in begins.windows(2).enumerate() {
            let text = &prompt[range[0]..range[1]];
            let tokens = if let Some((_, tokens)) = self.0.get(i).filter(|(old, _)| old == text) {
                tokens.clone()
            } else {
                encode(text)?
            };
            ids.extend_from_slice(&tokens);
            parts.push((text.to_owned(), tokens));
        }
        self.0 = parts;
        Ok(ids)
    }
}

#[derive(Debug)]
struct Config {
    port: u16,
    /// `flock` file shared with batch GPU jobs; None disables it.
    gpu_lock: Option<PathBuf>,
    /// Adjust the GLM expert budget after this many idle seconds.
    yield_idle: u64,
    glm_hyt: PathBuf,
    glm_source: PathBuf,
    glm_frequencies: Option<PathBuf>,
    glm_context: usize,
    /// GLM's VRAM at load (weights, state, expert cache and scratch); 0 =
    /// everything free but GLM_MARGIN, up to GLM_AUTO_MAX.
    glm_vram_gib: f64,
    /// Pinned RAM for experts not on the GPU; 0 = what they need, up to the
    /// runtime's safe limit (MemTotal - 24 GiB), before any spill to disk.
    glm_host_gb: u64,
    glm_chunk: usize,
    /// GLM LoRA adapters when a load request names none (`--glm-lora`, repeatable).
    glm_loras: Vec<hayate_models::glm53::lora::Spec>,
}
impl Config {
    fn parse(args: &[String]) -> Result<Self> {
        let mut config = Self {
            gpu_lock: Some("/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/gpu.lock".into()),
            yield_idle: 2,
            glm_hyt: "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53x3-sc260".into(),
            glm_source: "/mnt/NVME4TB/darask-inference-engine/local/hayate/src/GLM-5.3-Flash-dealignai-FP8".into(),
            glm_frequencies: Some("/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/glm53-stage6/frequencies.bin".into()),
            glm_context: 131072,
            glm_vram_gib: 0.,
            glm_host_gb: 0,
            glm_chunk: 8192,
            glm_loras: Vec::new(),
            port: 7870,
        };
        let mut args = args.iter();
        while let Some(key) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| Error::InvalidPlan(format!("missing {key}")))?;
            match key.as_str() {
                "--port" => {
                    config.port = value
                        .parse()
                        .map_err(|_| Error::InvalidPlan("port".into()))?
                }
                "--gpu-lock" => {
                    config.gpu_lock = (value != "none").then(|| value.into());
                }
                "--yield-idle" => {
                    config.yield_idle = value
                        .parse()
                        .map_err(|_| Error::InvalidPlan("yield idle seconds".into()))?
                }
                "--glm-hyt" => config.glm_hyt = value.into(),
                "--glm-source" => config.glm_source = value.into(),
                "--glm-frequencies" => {
                    config.glm_frequencies = (value != "none").then(|| value.into());
                }
                "--glm-context" => {
                    config.glm_context = value
                        .parse()
                        .map_err(|_| Error::InvalidPlan("glm context".into()))?
                }
                "--glm-vram-gib" => {
                    config.glm_vram_gib = value
                        .parse()
                        .map_err(|_| Error::InvalidPlan("glm VRAM GiB".into()))?
                }
                "--glm-host-gb" => {
                    config.glm_host_gb = value
                        .parse()
                        .map_err(|_| Error::InvalidPlan("glm host GB".into()))?
                }
                "--glm-prefill-chunk" => {
                    config.glm_chunk = value
                        .parse()
                        .map_err(|_| Error::InvalidPlan("glm prefill chunk".into()))?
                }
                "--glm-lora" => config
                    .glm_loras
                    .push(hayate_models::glm53::lora::Spec::parse(value)?),
                _ => return Err(Error::InvalidPlan(format!("unknown serve option {key}"))),
            }
        }
        if config.port == 0 {
            return Err(Error::InvalidPlan("serve port/context".into()));
        }
        if !(1024..=131072).contains(&config.glm_context)
            || !(config.glm_vram_gib == 0.
                || config.glm_vram_gib.is_finite() && config.glm_vram_gib >= 16.)
            || config.glm_chunk == 0
        {
            return Err(Error::InvalidPlan(
                "glm context 1024..131072, VRAM >= 16 GiB, prefill chunk > 0".into(),
            ));
        }
        Ok(config)
    }
}
pub fn run(args: &[String]) -> Result<()> {
    if args == ["--render-template"] {
        use std::io::BufRead;
        for line in std::io::stdin().lock().lines() {
            let request = hayate_server::json::parse(&line?)?;
            println!(
                "{}",
                hayate_server::json::quote(&hayate_server::template::render(&request)?)
            );
        }
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "--help" || s == "-h") {
        println!(
            "usage: hayate serve [--port 7870] [--gpu-lock FILE|none] [--yield-idle SECONDS(2, 0=never)] [--glm-hyt DIR] [--glm-source DIR] [--glm-frequencies FILE|none] [--glm-context 131072] [--glm-vram-gib 0(auto)] [--glm-host-gb 0(auto)] [--glm-prefill-chunk 8192] [--glm-lora FILE.gguf[:STRENGTH]]..."
        );
        return Ok(());
    }
    let config = Config::parse(args)?;
    #[cfg(feature = "cuda")]
    hayate_kernels::cpu::pin_thread(hayate_core::machine::MAIN.cpu.service[0])?;
    hayate_server::serve(
        config.port,
        hayate_server::capabilities::json(cfg!(feature = "cuda")),
        move |receiver| {
            #[cfg(feature = "cuda")]
            let result = gpu::worker(receiver, config);
            #[cfg(not(feature = "cuda"))]
            let result = hayate_server::worker_loop(receiver, &mut Unavailable);
            if let Err(e) = result {
                eprintln!("serve GPU worker: {e}");
                std::process::exit(2);
            }
        },
    )
}

#[cfg(not(feature = "cuda"))]
struct Unavailable;
#[cfg(not(feature = "cuda"))]
impl Backend for Unavailable {
    fn models(&self) -> Result<String> {
        Ok("{\"resident\":[],\"available\":[\"glm53\"]}".into())
    }
    fn load(&mut self, _: &Value, _: &mut dyn FnMut(String) -> Result<()>) -> Result<()> {
        Err(Error::InvalidPlan("GLM serve requires cuda feature".into()))
    }
    fn unload(&mut self, _: &Value) -> Result<()> {
        Ok(())
    }
    fn chat(&mut self, _: &Value, _: &mut dyn FnMut(String) -> Result<()>) -> Result<String> {
        Err(Error::InvalidPlan(
            "glm53 is not loaded; build with --features cuda".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use hayate_kernels::runtime::Device;
    use hayate_llm::session::{Engine, Session};
    use hayate_server::chat::Chat;
    use hayate_server::queue::Receiver;
    use std::time::Instant;

    struct Service<'d> {
        device: &'d Device,
        config: Config,
        glm: Option<Session<glm::Glm<'d>>>,
        /// The LoRA adapters the loaded GLM runs with.
        glm_loras: Vec<hayate_models::glm53::lora::Spec>,
        /// GLM's device memory now (measured), and what it was loaded with.
        glm_bytes: usize,
        glm_default: usize,
        glm_tok_s: f64,
        /// Set when the current chat's client disconnects.
        gone: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    }
    pub(super) fn worker(receiver: Receiver, config: Config) -> Result<()> {
        let device = Device::select()?;
        hayate_kernels::cpu::pin_thread(hayate_core::machine::MAIN.cpu.compute[0])?;
        let mut gpu = hayate_server::GpuLock::new(
            config.gpu_lock.clone(),
            (config.yield_idle > 0).then(|| std::time::Duration::from_secs(config.yield_idle)),
        );
        let mut service = Service {
            device: &device,
            config,
            glm: None,
            glm_loras: Vec::new(),
            glm_bytes: 0,
            glm_default: 0,
            glm_tok_s: 0.,
            gone: None,
        };
        hayate_server::worker_loop_locked(receiver, &mut service, &mut gpu)
    }
    impl Backend for Service<'_> {
        fn models(&self) -> Result<String> {
            let mut rows = Vec::new();
            if let Some(session) = &self.glm {
                let loras: Vec<_> = self
                    .glm_loras
                    .iter()
                    .map(|l| {
                        format!(
                            r#"{{"path":{},"strength":{}}}"#,
                            hayate_server::json::quote(&l.path.to_string_lossy()),
                            l.strength
                        )
                    })
                    .collect();
                rows.push(format!(
                    r#"{{"id":"glm53","variant":"fp8","vram_bytes":{},"context_length":{},"budget":{},"loras":[{}]}}"#,
                    self.glm_bytes,
                    session.engine.capacity(),
                    self.glm_budget(session).json(),
                    loras.join(",")
                ));
            }
            let resident = rows.join(",");
            Ok(format!(
                "{{\"resident\":[{resident}],\"available\":[\"glm53\"],\"context_length\":{{\"glm53\":{}}}}}",
                self.config.glm_context
            ))
        }
        fn load(
            &mut self,
            request: &Value,
            progress: &mut dyn FnMut(String) -> Result<()>,
        ) -> Result<()> {
            check_glm(request)?;
            self.load_glm(request, progress)
        }
        fn unload(&mut self, request: &Value) -> Result<()> {
            check_glm(request)?;
            self.glm.take();
            self.glm_bytes = 0;
            self.glm_default = 0;
            self.glm_tok_s = 0.;
            self.glm_loras.clear();
            self.device.synchronize()
        }
        fn ledger(&self) -> Vec<(&'static str, usize)> {
            let mut rows = Vec::new();
            if self.glm.is_some() {
                rows.push(("glm53", self.glm_bytes));
            }
            rows
        }
        fn client_gone(&mut self, flag: std::sync::Arc<std::sync::atomic::AtomicBool>) {
            self.gone = Some(flag);
        }
        fn budgets(&self) -> Vec<hayate_server::gpu::Budget> {
            self.glm.iter().map(|s| self.glm_budget(s)).collect()
        }
        fn shrink_llm(&mut self, bytes: usize) -> Result<usize> {
            let target = self.glm_bytes.saturating_sub(bytes);
            self.resize_glm(target)
        }
        fn grow_llm(&mut self, free: usize) -> Result<usize> {
            if self.glm.is_none() || self.glm_bytes >= self.glm_default {
                return Ok(0);
            }
            let target = self.glm_default.min(self.glm_bytes + free);
            self.resize_glm(target)
        }
        fn memory(&self) -> Option<(usize, usize)> {
            self.device.memory_info().ok()
        }
        fn chat(
            &mut self,
            request: &Value,
            emit: &mut dyn FnMut(String) -> Result<()>,
        ) -> Result<String> {
            let start = Instant::now();
            let mut chat = Chat::parse(request)?;
            let render_ms = start.elapsed().as_secs_f64() * 1000.;

            let session = self
                .glm
                .as_mut()
                .ok_or_else(|| Error::InvalidPlan("glm53 is not loaded".into()))?;
            let start = Instant::now();
            let engine = &mut session.engine;
            let prompt = engine
                .prompt_tokens
                .encode(&chat.prompt, &glm::MARKERS, |s| engine.tokenizer.encode(s))?;
            let tokenize_ms = start.elapsed().as_secs_f64() * 1000.;
            let capacity = engine.capacity();
            if chat.max_tokens_default {
                chat.params.max_tokens =
                    capacity.saturating_sub(prompt.len()).clamp(1, 32768) as u32;
            }
            engine.thinking = chat.thinking;
            engine.reasoning_budget = chat.reasoning_budget;
            engine.repetition_penalty = chat.repetition_penalty;
            let points = glm::Glm::prompt_points(&prompt);
            // A disconnect stops a long prefill between layers: a watcher
            // forwards the connection's flag to the executor's cancel flag.
            let cancel = engine.model.cancel_handle();
            cancel.store(false, std::sync::atomic::Ordering::Release);
            let finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let watcher = self.gone.take().map(|gone| {
                let (cancel, finished) = (cancel.clone(), finished.clone());
                std::thread::spawn(move || {
                    use std::sync::atomic::Ordering::Acquire;
                    while !finished.load(Acquire) {
                        if gone.load(Acquire) {
                            cancel.store(true, std::sync::atomic::Ordering::Release);
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    }
                })
            });
            let stats = respond(
                session,
                &prompt,
                &points,
                chat,
                render_ms,
                tokenize_ms,
                emit,
            );
            finished.store(true, std::sync::atomic::Ordering::Release);
            if let Some(w) = watcher {
                let _ = w.join();
            }
            cancel.store(false, std::sync::atomic::Ordering::Release);
            if let Some(tok_s) = stats.as_ref().ok().and_then(|(_, t)| *t) {
                self.glm_tok_s = tok_s;
            }
            stats.map(|(body, _)| body)
        }
    }
    impl<'d> Service<'d> {
        fn glm_budget(&self, session: &Session<glm::Glm<'d>>) -> hayate_server::gpu::Budget {
            let (experts_gpu, experts_total) = session.engine.model.session_experts();
            hayate_server::gpu::Budget {
                id: "glm53",
                budget_bytes: self.glm_bytes,
                default_bytes: self.glm_default,
                experts_gpu,
                experts_total,
                decode_tok_s: self.glm_tok_s,
            }
        }
        /// Ask GLM to hold `target` bytes of VRAM (Executor::set_vram_budget
        /// moves experts between VRAM and RAM between requests); returns the
        /// bytes released or taken, as the driver measures them.
        fn resize_glm(&mut self, target: usize) -> Result<usize> {
            let Some(session) = self.glm.as_mut() else {
                return Ok(0);
            };
            let (before, _) = self.device.memory_info()?;
            let resize = match session.engine.model.set_vram_budget(target as u64) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("serve glm53 budget {target} refused: {e}");
                    return Ok(0);
                }
            };
            self.device.synchronize()?;
            let (after, _) = self.device.memory_info()?;
            eprintln!(
                "serve glm53 budget={target} slots {}->{} gpu_delta={} free_delta={} seconds={:.3}",
                resize.slots_before,
                resize.slots_after,
                resize.gpu_delta,
                after as i64 - before as i64,
                resize.seconds
            );
            Ok(if after >= before {
                self.glm_bytes = self.glm_bytes.saturating_sub(after - before);
                after - before
            } else {
                self.glm_bytes += before - after;
                before - after
            })
        }
        fn load_glm(
            &mut self,
            request: &Value,
            progress: &mut dyn FnMut(String) -> Result<()>,
        ) -> Result<()> {
            if request
                .optional("variant")
                .is_some_and(|v| v.string().ok() != Some("fp8"))
            {
                return Err(Error::InvalidPlan("GLM variant: fp8".into()));
            }
            // `loras` names the exact adapter set (a different one reloads GLM); without it a
            // loaded GLM stays as it is and a new load takes --glm-lora.
            let loras = request.optional("loras").map(glm_loras).transpose()?;
            if self.glm.is_some() {
                if loras.as_ref().is_none_or(|l| *l == self.glm_loras) {
                    return Ok(());
                }
                self.glm.take();
                self.glm_bytes = 0;
                self.device.synchronize()?;
            }
            let loras = loras.unwrap_or_else(|| self.config.glm_loras.clone());
            progress(r#"{"stage":"weights","unit":"stages","done":0,"total":1}"#.into())?;
            let start = Instant::now();
            let (free, _) = self.device.memory_info()?;
            // Speed first: take what is free (up to GLM_AUTO_MAX) and give it
            // back only when a slot waiter declares a need (balance_llm).
            let budget = if self.config.glm_vram_gib > 0. {
                (self.config.glm_vram_gib * (1u64 << 30) as f64) as u64
            } else {
                (free as u64)
                    .saturating_sub(glm::GLM_MARGIN)
                    .min(glm::GLM_AUTO_MAX)
            };
            let mut engine = glm::Glm::load(self.device, &self.config, budget, &loras)?;
            progress(r#"{"stage":"warmup","done":0,"total":1}"#.into())?;
            let warmup = engine
                .tokenizer
                .encode("[gMASK]<sop><|user|>hi<|assistant|><think></think>")?;
            engine.append(&warmup, false)?;
            let next = engine.append(&[warmup[0]], false)?;
            engine.append(&[next], false)?;
            engine.reset()?;
            progress(r#"{"stage":"warmup","done":1,"total":1}"#.into())?;
            let (after, _) = self.device.memory_info()?;
            self.glm_bytes = free.saturating_sub(after);
            self.glm_default = self.glm_bytes;
            eprintln!(
                "serve glm53 load_ms={:.3} vram_bytes={} context={} memory={:?}",
                start.elapsed().as_secs_f64() * 1000.,
                self.glm_bytes,
                self.config.glm_context,
                engine.model.memory
            );
            self.glm = Some(Session::new(engine));
            self.glm_loras = loras;
            progress(r#"{"stage":"weights","unit":"stages","done":1,"total":1}"#.into())?;
            Ok(())
        }
    }
    /// `[{"path": FILE.gguf, "strength": 1}]`, as the image requests' `loras`.
    fn glm_loras(value: &Value) -> Result<Vec<hayate_models::glm53::lora::Spec>> {
        value
            .array()?
            .iter()
            .map(|lora| {
                for (key, _) in lora.object()? {
                    if !["path", "strength"].contains(&key.as_str()) {
                        return Err(Error::InvalidPlan(format!("unknown LoRA field {key}")));
                    }
                }
                let path = lora.get("path")?.string()?;
                let strength = lora
                    .optional("strength")
                    .map(Value::number)
                    .transpose()?
                    .unwrap_or(1.) as f32;
                if path.is_empty() || !strength.is_finite() {
                    return Err(Error::InvalidPlan(
                        "LoRA requires a path and finite f32 strength".into(),
                    ));
                }
                Ok(hayate_models::glm53::lora::Spec {
                    path: path.into(),
                    strength,
                })
            })
            .collect()
    }
    fn respond<E: Engine>(
        session: &mut Session<E>,
        prompt: &[u32],
        prompt_points: &[usize],
        mut chat: Chat,
        render_ms: f64,
        tokenize_ms: f64,
        emit: &mut dyn FnMut(String) -> Result<()>,
    ) -> Result<(String, Option<f64>)> {
        if prompt
            .len()
            .checked_add(chat.params.max_tokens as usize)
            .is_none_or(|n| n > session.engine.capacity())
        {
            return Err(Error::InvalidPlan(format!(
                "prompt ({} tokens) + max_tokens ({}) exceeds server context ({} tokens)",
                prompt.len(),
                chat.params.max_tokens,
                session.engine.capacity()
            )));
        }
        emit(chat.start())?;
        let mut stats =
            session.generate_at(prompt, prompt_points, chat.params, chat.seed, &mut |s| {
                let mut sent = false;
                let stop = chat.push(s, &mut |chunk| {
                    sent = true;
                    emit(chunk)
                })?;
                // Hidden reasoning and incomplete tool calls still observe disconnects.
                if !sent {
                    emit(String::new())?;
                }
                Ok(stop)
            })?;
        stats.render_ms = render_ms;
        stats.tokenize_ms = tokenize_ms;
        eprintln!(
            "serve model={} prompt_tokens={} cached_tokens={} prefilled_tokens={} completion_tokens={} prefill_ms={:.3} decode_ms={:.3} decode_tok_s={:.3} mtp_accepted={} mtp_proposed={} checkpoint_ms={:.3} cache_pinned_bytes={}",
            chat.model,
            stats.prompt_tokens,
            stats.cached_tokens,
            stats.prompt_tokens - stats.cached_tokens,
            stats.completion_tokens,
            stats.prefill_ms,
            stats.decode_ms,
            stats.decode_tok_s(),
            stats.mtp_accepted,
            stats.mtp_proposed,
            stats.checkpoint_ms,
            stats.cache_host_bytes
        );
        let tok_s = (stats.completion_tokens > 1).then(|| stats.decode_tok_s());
        Ok((chat.finish(&stats, emit)?, tok_s))
    }
}

#[cfg(feature = "cuda")]
fn check_glm(request: &Value) -> Result<()> {
    if request.get("id")?.string()? != "glm53" {
        return Err(Error::InvalidPlan("only glm53 is available".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn glm_service_options_reject_removed_models() {
        let c = Config::parse(&["--glm-context".into(), "4096".into()]).unwrap();
        assert_eq!(c.glm_context, 4096);
        for key in ["--hyt", "--gemma-hyt", "--media-root", "--spec-draft-n-max"] {
            assert!(Config::parse(&[key.into(), "1".into()]).is_err());
        }
        assert!(Config::parse(&["--port".into(), "0".into()]).is_err());
        assert!(Config::parse(&["--glm-vram-gib".into(), "NaN".into()]).is_err());
    }
    #[test]
    fn prompt_tokens_reuse_only_unchanged_glm_turns() -> Result<()> {
        let mut cache = PromptTokens::default();
        for (text, calls) in [
            ("<|user|>hi<|assistant|>", 2),
            ("<|user|>hi<|assistant|>hello<|user|>next", 2),
        ] {
            let mut encoded = 0;
            let ids = cache.encode(text, &["<|user|>", "<|assistant|>"], |s| {
                encoded += 1;
                Ok(s.chars().map(u32::from).collect())
            })?;
            assert_eq!(ids, text.chars().map(u32::from).collect::<Vec<_>>());
            assert_eq!(encoded, calls);
        }
        Ok(())
    }
}
