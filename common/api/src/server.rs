//! Server plumbing: the shared state, the connection threads and the engine thread. Requests are
//! parsed and tokenized on their connection thread (`request`) and queued; one engine thread
//! runs them in order (one sequence at a time) through `output` and sends the pieces back. A
//! connection thread that finds its client gone sets the job's cancel flag, which stops the
//! prompt at the next piece and generation at the next token.

use crate::api;
use crate::generate::{Flow, GenRequest, Generator, ScriptGen, Stats};
use crate::http::Conn;
use crate::model::TextModel;
use crate::output::{Output, Piece};
use crate::request::Prepared;
use crate::wire::{Usage, error_json};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const LOADING: u8 = 0;
pub const READY: u8 = 1;

pub struct Shared {
    pub model: TextModel,
    /// Unix time the server started (`created` of the model).
    pub started: u64,
    pub state: AtomicU8,
    pub queued: AtomicUsize,
    pub busy: AtomicBool,
    pub jobs: Mutex<Sender<Job>>,
    counter: AtomicU64,
}

impl Shared {
    /// A fresh random number (ids, default seeds).
    pub fn next_id(&self) -> u64 {
        let n = self.counter.fetch_add(1, Relaxed);
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let mut r = crate::sample::Rng::new(t ^ n.rotate_left(32));
        r.next_u64()
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// UTC wall clock for the log.
fn clock() -> String {
    let s = unix_now();
    format!("{:02}:{:02}:{:02}Z", s / 3600 % 24, s / 60 % 60, s % 60)
}

/// One generation request.
pub struct Job {
    pub kind: &'static str,
    pub prep: Prepared,
    /// Tool call ids derive from it.
    pub id_base: u64,
    pub cancel: Arc<AtomicBool>,
    pub events: Sender<Event>,
    pub queued_at: Instant,
}

pub struct Done {
    pub finish: &'static str,
    pub usage: Usage,
    pub stats: Stats,
    pub queue_s: f64,
    /// Every generated id (with the EOS that ended it).
    pub token_ids: Vec<u32>,
}

pub enum Event {
    Piece(Piece),
    Done(Done),
    Error(String),
}

fn run_job(shared: &Shared, g: &mut dyn Generator, job: Job) {
    shared.queued.fetch_sub(1, Relaxed);
    let queue_s = job.queued_at.elapsed().as_secs_f64();
    let p = &job.prep;
    let cancelled = || job.cancel.load(Relaxed);
    if cancelled() {
        eprintln!(
            "[{}] {}: client left while queued ({:.1} s)",
            clock(),
            job.kind,
            queue_s
        );
        return;
    }
    shared.busy.store(true, Relaxed);
    let tok = &shared.model.tok;
    let send = |piece: Piece| {
        if !piece.is_empty() {
            let _ = job.events.send(Event::Piece(piece));
        }
    };
    let mut out = Output::new(p, job.id_base);
    let req = GenRequest {
        prompt: &p.prompt,
        images: &p.images,
        params: &p.params,
        max_new: p.max_new,
        marks: &p.marks,
        variant: p.variant,
    };
    let r = g.generate(
        req,
        &mut |t| {
            if cancelled() {
                return Flow::Stop;
            }
            let (flow, piece) = out.token(tok, t);
            send(piece);
            flow
        },
        &cancelled,
    );
    send(out.finish());
    let finish = if cancelled() {
        "cancelled"
    } else {
        out.finish_reason()
    };
    shared.busy.store(false, Relaxed);
    match r {
        Ok(stats) => {
            let n = stats.generated;
            let usage = Usage {
                prompt: p.prompt.len(),
                completion: n,
                cached: stats.cached,
            };
            let mut line = if finish == "cancelled" && n == 0 {
                format!(
                    "[{}] {}: prompt {} tok ({} cached), cancelled while the prompt ran ({:.2} s), queue {:.1} s",
                    clock(),
                    job.kind,
                    p.prompt.len(),
                    stats.cached,
                    stats.prefill_s,
                    queue_s
                )
            } else {
                format!(
                    "[{}] {}: prompt {} tok ({} cached), generated {n} tok ({finish}), prefill {:.0} tok/s ({:.2} s), decode {:.1} tok/s, queue {:.1} s",
                    clock(),
                    job.kind,
                    p.prompt.len(),
                    stats.cached,
                    stats.prefill_tps(p.prompt.len()),
                    stats.prefill_s,
                    stats.decode_tps(),
                    queue_s
                )
            };
            if let Some(a) = stats.mtp_accept() {
                line += &format!(", {a:.2} tokens per step");
            }
            if stats.host_sampling && n > 1 {
                let per = |s: f64| s * 1e6 / (n - 1).max(1) as f64;
                line += &format!(
                    ", host sampling: logits copy {:.0} µs + sample {:.0} µs per token",
                    per(stats.copy_s),
                    per(stats.sample_s)
                );
            }
            eprintln!("{line}");
            let _ = job.events.send(Event::Done(Done {
                finish,
                usage,
                stats,
                queue_s,
                token_ids: std::mem::take(&mut out.ids),
            }));
        }
        Err(e) => {
            eprintln!("[{}] {}: error: {e}", clock(), job.kind);
            let _ = job.events.send(Event::Error(e));
        }
    }
}

#[cfg(not(feature = "engine"))]
fn engine_thread(_: Arc<Shared>, _: Receiver<Job>, _: PathBuf, _: Option<usize>, _: bool) {
    eprintln!("error: built without the engine (feature \"engine\"); only --dry-run works");
    std::process::exit(1);
}

#[cfg(feature = "engine")]
fn engine_thread(
    shared: Arc<Shared>,
    rx: Receiver<Job>,
    model: PathBuf,
    drafts: Option<usize>,
    host_sampling: bool,
) {
    let t0 = Instant::now();
    let eng = match tsumuji_engine::open(&model, shared.model.ctx, |m| {
        eprintln!("[{:6.1} s] {m}", t0.elapsed().as_secs_f64())
    }) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: loading the model: {e}");
            std::process::exit(1);
        }
    };
    let vram = tsumuji_cuda::Device::open(0)
        .and_then(|d| d.mem_info())
        .map(|(free, total)| {
            format!(
                ", VRAM {:.1} GiB used of {:.1} GiB",
                gib(total - free),
                gib(total)
            )
        })
        .unwrap_or_default();
    eprintln!(
        "[{:6.1} s] ready: context {} tokens{vram}",
        t0.elapsed().as_secs_f64(),
        shared.model.ctx
    );
    shared.state.store(READY, Relaxed);
    let mut g = crate::generate::EngineGen::new(eng, drafts, host_sampling);
    for job in rx {
        run_job(&shared, &mut g, job);
    }
}

fn handle(shared: Arc<Shared>, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(900)));
    let mut c = Conn::new(stream);
    loop {
        let req = match c.read_request() {
            Ok(Some(r)) => r,
            Ok(None) => return,
            Err((status, msg)) => {
                let _ = c.respond(
                    status,
                    "application/json",
                    error_json(&msg, "invalid_request_error")
                        .to_string()
                        .as_bytes(),
                    false,
                );
                return;
            }
        };
        let keep = req.keep_alive;
        if !api::route(&shared, &mut c, &req) || !keep {
            return;
        }
    }
}

#[cfg(feature = "engine")]
fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// The default context: the model's maximum, or as many tokens as the free VRAM holds next to
/// the weights. The FP32 KV and indexer caches take 56,832 B per token (12 QSA layers x 4,736 B);
/// everything else was measured on the real model (2026-10-21, with the MTP layer loaded):
/// 80,716 MiB at 131,072 tokens. Only an estimate: the engine knows its footprint, the server does
/// not (`--ctx` overrides it).
#[cfg(feature = "engine")]
fn auto_ctx(max_ctx: usize) -> Result<usize, String> {
    const REST: usize = 73_612 << 20;
    const PER_TOKEN: usize = 56_832;
    const SPARE: usize = 4 << 30; // ViT weights and tiled attention workspace.
    let (free, _) = tsumuji_cuda::Device::open(0)
        .and_then(|d| d.mem_info())
        .map_err(|e| e.to_string())?;
    let fit = free.saturating_sub(REST + SPARE) / PER_TOKEN / 4096 * 4096;
    if fit < 4096 {
        return Err(format!(
            "only {:.1} GiB of VRAM is free; the model needs about {:.1} GiB plus 0.2 GiB per 4,096 tokens of context",
            gib(free),
            gib(REST + SPARE)
        ));
    }
    if fit < max_ctx {
        eprintln!(
            "{:.1} GiB of VRAM free: context {fit} tokens (the model allows {max_ctx}; --ctx sets it)",
            gib(free)
        );
    }
    Ok(fit.min(max_ctx))
}

#[cfg(not(feature = "engine"))]
fn auto_ctx(max_ctx: usize) -> Result<usize, String> {
    Ok(max_ctx)
}

pub struct Config {
    pub model: PathBuf,
    pub host: String,
    pub port: u16,
    pub ctx: Option<usize>,
    pub name: Option<String>,
    /// transformers 5.x's pre-tokenizer regex instead of tokenizer.json's.
    pub qwen2_regex: bool,
    /// No GPU: answer every request with a fixed script (tests the HTTP side).
    pub dry_run: bool,
    /// MTP drafts per speculative step: None = chosen by the engine per step, 0 = one token per step.
    pub drafts: Option<usize>,
    /// Sample on the host instead of on the device.
    pub host_sampling: bool,
}

/// `--dry-run`: the engine thread without the engine.
fn script_thread(shared: Arc<Shared>, rx: Receiver<Job>) {
    let m = &shared.model;
    let text = "The user greets me in two languages; answer briefly.\n</think>\n\nこんにちは！ Hello 👋🏽 — this is Tsumuji's dry run.<|im_end|>";
    let call = "I should look at the file first.\n</think>\n\nLet me read it.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call><|im_end|>";
    let delay = Duration::from_millis(20);
    let mut plain = ScriptGen::new(m.ctx, m.tok.encode(text), delay);
    let mut tools = ScriptGen::new(m.ctx, m.tok.encode(call), delay);
    shared.state.store(READY, Relaxed);
    eprintln!(
        "ready (dry run: no model; a fixed answer, or a fixed tool call when the request has tools)"
    );
    for job in rx {
        let g = if job.prep.tools.is_some() {
            &mut tools
        } else {
            &mut plain
        };
        run_job(&shared, g, job);
    }
}

pub fn serve(cfg: Config) -> Result<(), String> {
    let mut model = TextModel::load(&cfg.model, cfg.name.clone())?;
    if cfg.qwen2_regex {
        model.tok.regex = crate::tokenizer::Regex::Qwen2;
    }
    model.ctx = match cfg.ctx {
        Some(c) => c,
        None if cfg.dry_run => model.info.max_ctx,
        None => auto_ctx(model.info.max_ctx)?,
    };
    let listener = TcpListener::bind((cfg.host.as_str(), cfg.port))
        .map_err(|e| format!("bind {}:{}: {e}", cfg.host, cfg.port))?;
    eprintln!(
        "listening on http://{}:{} (model {}, context {}, eos {:?}); loading the model…",
        cfg.host, cfg.port, model.info.id, model.ctx, model.info.eos
    );
    let (tx, rx) = channel::<Job>();
    let shared = Arc::new(Shared {
        model,
        started: unix_now(),
        state: AtomicU8::new(LOADING),
        queued: AtomicUsize::new(0),
        busy: AtomicBool::new(false),
        jobs: Mutex::new(tx),
        counter: AtomicU64::new(0),
    });
    let s = shared.clone();
    let model = cfg.model.clone();
    let (dry, drafts, host) = (cfg.dry_run, cfg.drafts, cfg.host_sampling);
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            if dry {
                script_thread(s, rx)
            } else {
                engine_thread(s, rx, model, drafts, host)
            }
        })
        .map_err(|e| e.to_string())?;
    for stream in listener.incoming() {
        match stream {
            Ok(st) => {
                let s = shared.clone();
                let _ = std::thread::Builder::new()
                    .name("conn".into())
                    .spawn(move || handle(s, st));
            }
            Err(e) => eprintln!("accept: {e}"),
        }
    }
    Ok(())
}

/// Shared frontend for another native engine. CUDA is created on its owning worker thread.
pub fn serve_with<G, F>(model: TextModel, host: &str, port: u16, open: F) -> Result<(), String>
where
    G: Generator + 'static,
    F: FnOnce() -> Result<G, String> + Send + 'static,
{
    let listener = TcpListener::bind((host, port)).map_err(|e| e.to_string())?;
    let (tx, rx) = channel::<Job>();
    let shared = Arc::new(Shared {
        model,
        started: unix_now(),
        state: AtomicU8::new(LOADING),
        queued: AtomicUsize::new(0),
        busy: AtomicBool::new(false),
        jobs: Mutex::new(tx),
        counter: AtomicU64::new(0),
    });
    let s = shared.clone();
    std::thread::Builder::new()
        .name("native-engine".into())
        .spawn(move || {
            let mut g = match open() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("native model load failed: {e}");
                    std::process::exit(1);
                }
            };
            s.state.store(READY, Relaxed);
            for job in rx {
                run_job(&s, &mut g, job);
            }
        })
        .map_err(|e| e.to_string())?;
    eprintln!("listening on http://{host}:{port} (native engine)");
    for stream in listener.incoming() {
        match stream {
            Ok(st) => {
                let s = shared.clone();
                let _ = std::thread::Builder::new()
                    .name("conn".into())
                    .spawn(move || handle(s, st));
            }
            Err(e) => eprintln!("accept: {e}"),
        }
    }
    Ok(())
}
