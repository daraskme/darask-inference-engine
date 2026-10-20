//! HTTP routes: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions.
//! The request → prompt → output → JSON work is in the library (`request`, `output`, `wire`);
//! this file only queues the job and writes the response or the SSE stream.

use crate::http::{Conn, Request};
use crate::json::Json;
use crate::output::Piece;
use crate::request::{ApiError, Prepared, parse_chat, parse_completion};
use crate::server::{Event, Job, READY, Shared, unix_now};
use crate::wire::{Meta, error_json, timings};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{RecvTimeoutError, channel};
use std::time::{Duration, Instant};

fn send_json(c: &mut Conn, status: u16, v: &Json, keep: bool) -> bool {
    c.respond(status, "application/json", v.to_string().as_bytes(), keep)
        .is_ok()
        && keep
}

fn refuse(c: &mut Conn, e: &ApiError, keep: bool) -> bool {
    send_json(
        c,
        e.status,
        &error_json(&e.message, "invalid_request_error"),
        keep,
    )
}

/// Handles one request; false when the connection should be closed.
pub fn route(shared: &Arc<Shared>, c: &mut Conn, req: &Request) -> bool {
    let keep = req.keep_alive;
    let path = req.path.trim_end_matches('/');
    let m = &shared.model;
    match (req.method.as_str(), path) {
        ("OPTIONS", _) => c.respond(204, "text/plain", b"", keep).is_ok() && keep,
        ("GET", "/health" | "/v1/health") => {
            let ready = shared.state.load(Relaxed) == READY;
            let v = Json::obj([
                ("status", Json::str(if ready { "ok" } else { "loading" })),
                ("model", Json::str(&m.info.id)),
                ("context", Json::num(m.ctx as f64)),
                ("busy", Json::Bool(shared.busy.load(Relaxed))),
                ("queued", Json::num(shared.queued.load(Relaxed) as f64)),
            ]);
            send_json(c, if ready { 200 } else { 503 }, &v, keep)
        }
        ("GET", "/v1/models") => {
            let v = Json::obj([
                ("object", Json::str("list")),
                ("data", Json::Arr(vec![model_json(shared)])),
            ]);
            send_json(c, 200, &v, keep)
        }
        ("GET", p) if p.strip_prefix("/v1/models/") == Some(m.info.id.as_str()) => {
            send_json(c, 200, &model_json(shared), keep)
        }
        ("POST", "/v1/chat/completions" | "/v1/completions") => {
            let chat = path.ends_with("chat/completions");
            let body = match std::str::from_utf8(&req.body)
                .map_err(|e| e.to_string())
                .and_then(Json::parse)
            {
                Ok(b @ Json::Obj(_)) => b,
                Ok(_) => return refuse(c, &"the body must be a JSON object".into(), keep),
                Err(e) => return refuse(c, &format!("invalid JSON body: {e}").into(), keep),
            };
            let seed = shared.next_id();
            match if chat {
                parse_chat(m, &body, seed)
            } else {
                parse_completion(m, &body, seed)
            } {
                Ok(p) => run(shared, c, p, keep),
                Err(e) => refuse(c, &e, keep),
            }
        }
        (
            _,
            "/health" | "/v1/health" | "/v1/models" | "/v1/chat/completions" | "/v1/completions",
        ) => send_json(
            c,
            405,
            &error_json("method not allowed", "invalid_request_error"),
            keep,
        ),
        _ => send_json(
            c,
            404,
            &error_json(
                &format!("no route for {} {}", req.method, req.path),
                "invalid_request_error",
            ),
            keep,
        ),
    }
}

fn model_json(shared: &Shared) -> Json {
    Json::obj([
        ("id", Json::str(&shared.model.info.id)),
        ("object", Json::str("model")),
        ("created", Json::num(shared.started as f64)),
        ("owned_by", Json::str("tsumuji")),
        ("max_model_len", Json::num(shared.model.ctx as f64)),
    ])
}

fn sse(c: &mut Conn, v: &Json) -> bool {
    c.chunk(format!("data: {v}\n\n").as_bytes()).is_ok()
}

/// Queues the request and answers it (streaming or not); false when the connection is done.
fn run(shared: &Arc<Shared>, c: &mut Conn, p: Prepared, keep: bool) -> bool {
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let id = shared.next_id();
    let meta = Meta {
        id: format!("{}-{id:016x}", if p.chat { "chatcmpl" } else { "cmpl" }),
        created: unix_now(),
        model: shared.model.info.id.clone(),
        chat: p.chat,
    };
    let (stream, reasoning, echo, return_ids, prompt_len) = (
        p.stream,
        p.reasoning,
        p.echo.clone(),
        p.return_token_ids,
        p.prompt.len(),
    );
    let prompt_ids = if return_ids {
        p.prompt.clone()
    } else {
        Vec::new()
    };
    let job = Job {
        kind: if p.chat { "chat" } else { "completion" },
        prep: p,
        id_base: id,
        cancel: cancel.clone(),
        events: tx,
        queued_at: Instant::now(),
    };
    shared.queued.fetch_add(1, Relaxed);
    if shared.jobs.lock().unwrap().send(job).is_err() {
        shared.queued.fetch_sub(1, Relaxed);
        return send_json(
            c,
            503,
            &error_json("the engine has stopped", "server_error"),
            false,
        );
    }
    let gone = |cancel: &AtomicBool| {
        cancel.store(true, Relaxed);
        false
    };
    if stream
        && (c.start_stream("text/event-stream").is_err()
            || !sse(c, &meta.first_chunk(echo.as_deref())))
    {
        return gone(&cancel);
    }
    let mut all = Piece::default();
    // Every 100 ms, whether or not text arrives (a write to a closed socket can still succeed).
    let mut checked = Instant::now();
    loop {
        let r = rx.recv_timeout(Duration::from_millis(100));
        if checked.elapsed() >= Duration::from_millis(100) {
            checked = Instant::now();
            if c.peer_closed() {
                return gone(&cancel);
            }
        }
        let ev = match r {
            Ok(ev) => ev,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => Event::Error("the engine stopped".into()),
        };
        match ev {
            Event::Piece(piece) => {
                if stream {
                    if !sse(c, &meta.delta_chunk(&piece, reasoning)) {
                        return gone(&cancel);
                    }
                } else {
                    all.append(piece);
                }
            }
            Event::Done(d) => {
                let t = timings(&d.stats, prompt_len, d.queue_s);
                if stream {
                    let ok = sse(c, &meta.final_chunk(d.finish, d.usage, t))
                        && c.chunk(b"data: [DONE]\n\n").is_ok()
                        && c.end_stream().is_ok();
                    return ok && keep;
                }
                let ids = return_ids.then_some((prompt_ids.as_slice(), d.token_ids.as_slice()));
                return send_json(
                    c,
                    200,
                    &meta.response(&all, reasoning, echo.as_deref(), d.finish, d.usage, t, ids),
                    keep,
                );
            }
            Event::Error(e) => {
                let err = error_json(&e, "server_error");
                if stream {
                    // Hayate's serve: the error is the last event, no [DONE], and the connection closes.
                    let _ = sse(c, &err) && c.end_stream().is_ok();
                    return false;
                }
                return send_json(c, 500, &err, keep);
            }
        }
    }
}
