//! GLM-5.3-Flash local chat API and GPU budget coordination.
pub mod capabilities;
pub mod chat;
pub mod gpu;
pub mod http;
pub mod json;
pub mod output;
pub mod queue;
pub mod template;
#[cfg(test)]
mod tests;

use hayate_core::{Error, Result};
use json::{Value, quote};
use std::{
    io::Write,
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        Arc,
        mpsc::{self, SyncSender},
    },
    thread,
    time::{Duration, Instant},
};

/// Construct and use the CUDA backend inside its single owning worker thread.
/// HTTP threads exchange only owned requests and response bytes with it.
pub trait Backend {
    fn models(&self) -> Result<String>;
    fn load(
        &mut self,
        request: &Value,
        progress: &mut dyn FnMut(String) -> Result<()>,
    ) -> Result<()>;
    fn unload(&mut self, request: &Value) -> Result<()>;
    /// VRAM ledger: each resident model and the bytes it added when loaded.
    fn ledger(&self) -> Vec<(&'static str, usize)> {
        Vec::new()
    }
    /// Device memory (free, total) from cudaMemGetInfo.
    fn memory(&self) -> Option<(usize, usize)> {
        None
    }
    /// Give back up to `bytes` of resident-LLM VRAM without unloading it, for a
    /// declared need the slot models alone do not cover; returns the bytes freed.
    /// GLM-5.3 does this with fewer experts on the GPU.
    fn shrink_llm(&mut self, _bytes: usize) -> Result<usize> {
        Ok(0)
    }
    /// The slot is free and nobody waits: take back up to `free` bytes towards
    /// the default budget; returns the bytes taken.
    fn grow_llm(&mut self, _free: usize) -> Result<usize> {
        Ok(0)
    }
    /// The flag the connection thread sets when this chat's client goes away;
    /// a backend can stop a long prefill with it.
    fn client_gone(&mut self, _: Arc<std::sync::atomic::AtomicBool>) {}
    /// Elastic LLM budgets for `/v1/gpu`.
    fn budgets(&self) -> Vec<gpu::Budget> {
        Vec::new()
    }
    fn chat(
        &mut self,
        request: &Value,
        emit: &mut dyn FnMut(String) -> Result<()>,
    ) -> Result<String>;
}
#[derive(Debug)]
pub struct Job {
    work: Work,
}
#[derive(Debug)]
enum Work {
    Http(http::Request, SyncSender<Event>),
}
#[derive(Debug)]
enum Event {
    Headers(u16, bool, usize),
    Data(String),
    End,
    Keepalive,
}

pub fn serve(
    port: u16,
    capabilities: String,
    worker: impl FnOnce(queue::Receiver) + Send + 'static,
) -> Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let (sender, receiver) = queue::channel(32);
    thread::Builder::new()
        .name("hayate-gpu".into())
        .spawn(move || worker(receiver))?;
    eprintln!("hayate serve listening on http://127.0.0.1:{port}");
    for stream in listener.incoming() {
        let stream = stream?;
        let sender = sender.clone();
        let capabilities = capabilities.clone();
        thread::Builder::new()
            .name("hayate-http".into())
            .spawn(move || {
                if let Err(e) = connection(stream, sender, &capabilities)
                    && !client_closed(&e)
                {
                    eprintln!("serve connection: {e}");
                }
            })?;
    }
    Ok(())
}
fn error_body(e: &impl std::fmt::Display) -> String {
    format!("{{\"error\":{{\"message\":{}}}}}", quote(&e.to_string()))
}
fn connection(mut stream: TcpStream, sender: queue::Sender, capabilities: &str) -> Result<()> {
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let request = match http::read(&mut stream) {
        Ok(r) => r,
        Err(e) => {
            let body = error_body(&e);
            http::headers(&mut stream, 400, false, body.len())?;
            stream.write_all(body.as_bytes())?;
            return Ok(());
        }
    };
    if request.method == "GET" && request.path == "/v1/capabilities" {
        return response(&mut stream, 200, capabilities);
    }
    if request.method == "GET" && request.path == "/v1/gpu" {
        return match sender.gpu() {
            Some(snapshot) => response(&mut stream, 200, &gpu::status(&snapshot)),
            None => response(&mut stream, 503, &error_body(&"GPU worker starting")),
        };
    }
    if request.method == "GET" && request.path == "/v1/models" {
        return match sender.models() {
            Some(models) => response(&mut stream, 200, &models),
            None => response(&mut stream, 503, &error_body(&"GPU worker starting")),
        };
    }
    if request.method == "POST"
        && request.path == "/v1/models/unload"
        && let Err(e) = json::parse(&request.body).and_then(|body| {
            if body.get("id")?.string()? != "glm53" {
                return Err(Error::InvalidPlan("unknown model id".into()));
            }
            Ok(body)
        })
    {
        return response(&mut stream, 400, &error_body(&e));
    }
    let (reply, events) = mpsc::sync_channel(8);
    let unload = request.method == "POST" && request.path == "/v1/models/unload";
    let gone = request.gone.clone();
    let job = Job {
        work: Work::Http(request, reply),
    };
    if unload {
        if sender.try_send(job).is_err() {
            return response(&mut stream, 409, &error_body(&"GPU queue unavailable"));
        }
    } else {
        sender
            .send(job)
            .map_err(|_| Error::InvalidPlan("GPU worker stopped".into()))?;
    }
    let mut sse = false;
    loop {
        // A client that closes the connection (Studio aborts the fetch) drops
        // `events`; the worker's next send fails and generation stops.
        let event = match events.recv_timeout(Duration::from_millis(200)) {
            Ok(event) => event,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if peer_closed(&stream) {
                    gone.store(true, std::sync::atomic::Ordering::Release);
                    return Ok(());
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        match event {
            Event::Headers(status, event_stream, len) => {
                sse = event_stream;
                http::headers(&mut stream, status, sse, len)?;
            }
            Event::Data(data) => {
                if sse {
                    http::sse(&mut stream, &data)?;
                } else {
                    stream.write_all(data.as_bytes())?;
                }
            }
            // Liveness probes of a non-streaming reply are not written.
            Event::Keepalive if sse => stream.write_all(b": keepalive\n\n")?,
            Event::Keepalive => {}
            Event::End => break,
        }
    }
    Ok(())
}
/// The peer closed its side (EOF or reset) after sending the request.
fn peer_closed(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let closed = match stream.peek(&mut [0; 1]) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => e.kind() != std::io::ErrorKind::WouldBlock,
    };
    let _ = stream.set_nonblocking(false);
    closed
}
/// Observe the GPU slot used by external jobs. GLM remains resident and
/// changes its expert-cache budget between requests when a job declares a need.
#[derive(Debug, Default)]
pub struct GpuLock {
    path: Option<PathBuf>,
    yield_idle: Option<Duration>,
}
impl GpuLock {
    pub fn new(path: Option<PathBuf>, yield_idle: Option<Duration>) -> Self {
        Self { path, yield_idle }
    }
    fn poll(&self) -> Option<Duration> {
        self.path.as_ref()?;
        self.yield_idle.map(|d| d.min(Duration::from_secs(1)))
    }
}

pub fn worker_loop(receiver: queue::Receiver, backend: &mut impl Backend) -> Result<()> {
    worker_loop_locked(receiver, backend, &mut GpuLock::default())
}
fn publish(
    receiver: &queue::Receiver,
    backend: &impl Backend,
    lock: &GpuLock,
    job: Option<&str>,
) -> Result<()> {
    receiver.publish_models(backend.models()?);
    receiver.publish_gpu(gpu::Snapshot {
        lock: lock.path.clone(),
        models: backend.ledger(),
        memory: backend.memory(),
        job: job.map(Into::into),
        budgets: backend.budgets(),
    });
    Ok(())
}
fn free(backend: &impl Backend) -> Option<u64> {
    backend.memory().map(|(free, _)| free as u64)
}
fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}
/// Elastic LLM VRAM while no slot model is resident: shrink for the first
/// declared waiter once it holds the slot (or the slot is free) and free VRAM
/// is short of its need; grow back while the slot is free and nobody waits.
fn balance_llm(backend: &mut impl Backend, lock: &GpuLock) -> Result<()> {
    let Some(path) = &lock.path else {
        return Ok(());
    };
    let me = std::process::id();
    let holder = gpu::lock_rows(path)
        .into_iter()
        .find(|r| !r.1)
        .map(|r| r.0)
        .filter(|&p| p != me);
    let Some(free) = free(backend) else {
        return Ok(());
    };
    if let Some(head) = gpu::declarations(path).first() {
        let need = head.need_bytes();
        if holder.is_none_or(|p| p == head.pid) && free < need {
            let freed = backend.shrink_llm((need - free) as usize)?;
            eprintln!(
                "serve GPU slot: {} needs {:.1} GiB free, {:.1} GiB free; LLM gave back {:.1} GiB",
                head.who,
                gib(need),
                gib(free),
                gib(freed as u64)
            );
        }
        return Ok(());
    }
    if holder.is_none() {
        let taken = backend.grow_llm(free.saturating_sub(gpu::MARGIN_BYTES) as usize)?;
        if taken > 0 {
            eprintln!(
                "serve GPU slot free: LLM took back {:.1} GiB",
                gib(taken as u64)
            );
        }
    }
    Ok(())
}
pub fn worker_loop_locked(
    mut receiver: queue::Receiver,
    backend: &mut impl Backend,
    gpu: &mut GpuLock,
) -> Result<()> {
    publish(&receiver, backend, gpu, None)?;
    let mut last_request = Instant::now();
    loop {
        let work = match receiver.wait(gpu.poll()) {
            queue::Next::Job(job) => job.work,
            queue::Next::Closed => return Ok(()),
            queue::Next::Idle => {
                if gpu
                    .yield_idle
                    .is_some_and(|wait| last_request.elapsed() >= wait)
                {
                    balance_llm(backend, gpu)?;
                }
                publish(&receiver, backend, gpu, None)?;
                continue;
            }
        };
        let Work::Http(request, reply) = work;
        let mut started = false;
        let mut streaming = false;
        let mut send = |data: String| -> Result<()> {
            if !started {
                reply
                    .send(Event::Headers(200, true, 0))
                    .map_err(disconnected)?;
                started = true;
            }
            reply
                .send(if data.is_empty() {
                    Event::Keepalive
                } else {
                    Event::Data(data)
                })
                .map_err(disconnected)
        };
        let result = (|| -> Result<String> {
            match (request.method.as_str(), request.path.as_str()) {
                ("GET", "/v1/models") => backend.models(),
                ("POST", "/v1/models/load") => {
                    let body = json::parse(&request.body)?;
                    streaming = request.sse;
                    check_glm(&body)?;
                    backend.load(&body, &mut |s| if streaming { send(s) } else { Ok(()) })?;
                    publish(&receiver, backend, gpu, None)?;
                    if streaming {
                        send("{\"stage\":\"ready\"}".into())?;
                    }
                    Ok("{}".into())
                }
                ("POST", "/v1/models/unload") => {
                    let body = json::parse(&request.body)?;
                    check_glm(&body)?;
                    backend.unload(&body)?;
                    Ok("{}".into())
                }
                ("POST", "/v1/chat/completions") => {
                    let body = json::parse(&request.body)?;
                    if body.get("model")?.string()? != "glm53" {
                        return Err(Error::InvalidPlan("chat model: glm53".into()));
                    }
                    backend.client_gone(request.gone.clone());
                    streaming = body
                        .optional("stream")
                        .map(Value::boolean)
                        .transpose()?
                        .unwrap_or(false);
                    // A non-streaming reply probes the connection with
                    // keepalives that are not written, so a closed client
                    // stops it too.
                    let response = backend.chat(&body, &mut |s| {
                        if streaming {
                            send(s)
                        } else {
                            reply.send(Event::Keepalive).map_err(disconnected)
                        }
                    })?;
                    if streaming {
                        send("[DONE]".into())?;
                    }
                    Ok(response)
                }
                _ => Err(Error::InvalidPlan("route not found".into())),
            }
        })();
        publish(&receiver, backend, gpu, None)?;
        match result {
            Ok(body) if !streaming => {
                let _ = reply.send(Event::Headers(200, false, body.len()));
                let _ = reply.send(Event::Data(body));
            }
            Ok(_) => {}
            Err(e) => {
                let fatal = matches!(e, Error::CudaAsync { .. });
                let status = if request.path != "/v1/models"
                    && ![
                        "/v1/models/load",
                        "/v1/models/unload",
                        "/v1/chat/completions",
                    ]
                    .contains(&request.path.as_str())
                {
                    404
                } else if matches!(e, Error::OutOfMemory { .. } | Error::Cuda { code: 2, .. }) {
                    409
                } else if matches!(e, Error::InvalidPlan(_) | Error::InvalidShape(_)) {
                    400
                } else {
                    500
                };
                let body = error_body(&e);
                if !started {
                    let _ = reply.send(Event::Headers(status, false, body.len()));
                }
                let _ = reply.send(Event::Data(body));
                if fatal {
                    return Err(e);
                }
            }
        }
        let _ = reply.send(Event::End);
        last_request = Instant::now();
    }
}
fn disconnected<T>(_: mpsc::SendError<T>) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::BrokenPipe,
        "client disconnected",
    ))
}

fn response(stream: &mut TcpStream, status: u16, body: &str) -> Result<()> {
    http::headers(&mut *stream, status, false, body.len())?;
    stream.write_all(body.as_bytes())?;
    Ok(())
}

fn client_closed(error: &Error) -> bool {
    matches!(error, Error::Io(e) if matches!(e.kind(),
        std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
        | std::io::ErrorKind::ConnectionAborted | std::io::ErrorKind::UnexpectedEof))
}

#[cfg(test)]
mod disconnect_tests {
    use super::*;
    #[test]
    fn suppress_only_peer_disconnects() {
        for kind in [
            std::io::ErrorKind::BrokenPipe,
            std::io::ErrorKind::ConnectionReset,
            std::io::ErrorKind::ConnectionAborted,
            std::io::ErrorKind::UnexpectedEof,
        ] {
            assert!(client_closed(&std::io::Error::from(kind).into()));
        }
        assert!(!client_closed(
            &std::io::Error::from(std::io::ErrorKind::PermissionDenied).into()
        ));
        assert!(!client_closed(&Error::InvalidPlan("bad request".into())));
    }
}

fn check_glm(body: &Value) -> Result<()> {
    if body.get("id")?.string()? == "glm53" {
        Ok(())
    } else {
        Err(Error::InvalidPlan("only glm53 is available".into()))
    }
}
