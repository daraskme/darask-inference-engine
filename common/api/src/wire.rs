//! The JSON on the wire: OpenAI chat completion / completion objects and stream chunks, with
//! Hayate serve's additions (hayate-studio docs/hayate-serve.md §2): `usage` with
//! `prompt_tokens_details.cached_tokens` on the last chunk, and a top-level `timings`
//! `{prefill_ms, decode_ms, decode_tok_s, mtp_accept}` (plus a few more fields).

use crate::generate::Stats;
use crate::json::Json;
use crate::output::{Piece, ToolCallOut};

pub fn error_json(msg: &str, kind: &str) -> Json {
    Json::obj([(
        "error",
        Json::obj([
            ("message", Json::str(msg)),
            ("type", Json::str(kind)),
            ("param", Json::Null),
            ("code", Json::Null),
        ]),
    )])
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub prompt: usize,
    pub completion: usize,
    /// Prompt tokens whose state came from the prefix cache (not prefilled again).
    pub cached: usize,
}

impl Usage {
    pub fn json(&self) -> Json {
        Json::obj([
            ("prompt_tokens", Json::num(self.prompt as f64)),
            ("completion_tokens", Json::num(self.completion as f64)),
            (
                "total_tokens",
                Json::num((self.prompt + self.completion) as f64),
            ),
            (
                "prompt_tokens_details",
                Json::obj([("cached_tokens", Json::num(self.cached as f64))]),
            ),
        ])
    }
}

fn round1(x: f64) -> Json {
    Json::num((x * 10.0).round() / 10.0)
}

/// Where the time went. `prefill_ms` covers only the prompt part that was prefilled (not the
/// cached prefix); `mtp_accept` is the mean tokens committed per speculative step (null without).
pub fn timings(s: &Stats, prompt: usize, queue_s: f64) -> Json {
    Json::obj([
        ("prefill_ms", round1(s.prefill_s * 1e3)),
        ("decode_ms", round1(s.decode_s * 1e3)),
        ("decode_tok_s", round1(s.decode_tps())),
        (
            "mtp_accept",
            s.mtp_accept()
                .map_or(Json::Null, |a| Json::num((a * 100.0).round() / 100.0)),
        ),
        ("queue_ms", round1(queue_s * 1e3)),
        (
            "prefill_tok_s",
            round1((prompt - s.cached.min(prompt)) as f64 / s.prefill_s.max(1e-9)),
        ),
        ("sampling", Json::str(s.sampling)),
    ])
}

/// What every response and chunk of one request shares.
#[derive(Debug, Clone)]
pub struct Meta {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub chat: bool,
}

fn tool_calls_json(calls: &[ToolCallOut], index: bool) -> Json {
    Json::Arr(
        calls
            .iter()
            .map(|c| {
                let mut v = Json::Obj(Vec::new());
                if index {
                    v.push("index", Json::num(c.index as f64));
                }
                v.push("id", Json::str(&c.id));
                v.push("type", Json::str("function"));
                v.push(
                    "function",
                    Json::obj([
                        ("name", Json::str(&c.name)),
                        ("arguments", Json::str(&c.arguments)),
                    ]),
                );
                v
            })
            .collect(),
    )
}

impl Meta {
    fn base(&self, chunk: bool, choices: Vec<Json>) -> Json {
        let object = match (self.chat, chunk) {
            (true, true) => "chat.completion.chunk",
            (true, false) => "chat.completion",
            (false, _) => "text_completion",
        };
        Json::obj([
            ("id", Json::str(&self.id)),
            ("object", Json::str(object)),
            ("created", Json::num(self.created as f64)),
            ("model", Json::str(&self.model)),
            ("choices", Json::Arr(choices)),
        ])
    }

    fn choice(&self, body: Json, finish: Option<&str>) -> Json {
        let key = if self.chat { "delta" } else { "text" };
        Json::obj([
            ("index", Json::num(0)),
            (key, body),
            ("logprobs", Json::Null),
            ("finish_reason", Json::opt_str(finish)),
        ])
    }

    /// The first chunk: the role (chat) or the echoed prompt (completions).
    pub fn first_chunk(&self, echo: Option<&str>) -> Json {
        let body = if self.chat {
            Json::obj([("role", Json::str("assistant")), ("content", Json::str(""))])
        } else {
            Json::str(echo.unwrap_or(""))
        };
        self.base(true, vec![self.choice(body, None)])
    }

    /// A chunk with new output; `reasoning`: send `reasoning_content` (else it is dropped).
    pub fn delta_chunk(&self, p: &Piece, reasoning: bool) -> Json {
        let body = if self.chat {
            let mut d = Json::Obj(Vec::new());
            if reasoning && !p.reasoning.is_empty() {
                d.push("reasoning_content", Json::str(&p.reasoning));
            }
            if !p.content.is_empty() {
                d.push("content", Json::str(&p.content));
            }
            if !p.tool_calls.is_empty() {
                d.push("tool_calls", tool_calls_json(&p.tool_calls, true));
            }
            d
        } else {
            Json::str(&p.content)
        };
        self.base(true, vec![self.choice(body, None)])
    }

    /// The last chunk: finish_reason, usage and timings (always, as Hayate's serve sends them).
    pub fn final_chunk(&self, finish: &str, usage: Usage, timings: Json) -> Json {
        let body = if self.chat {
            Json::Obj(Vec::new())
        } else {
            Json::str("")
        };
        let mut v = self.base(true, vec![self.choice(body, Some(finish))]);
        v.push("usage", usage.json());
        v.push("timings", timings);
        v
    }

    /// The whole non-streaming response. `ids`: (prompt ids, output ids) for `return_token_ids`.
    #[allow(clippy::too_many_arguments)]
    pub fn response(
        &self,
        all: &Piece,
        reasoning: bool,
        echo: Option<&str>,
        finish: &str,
        usage: Usage,
        timings: Json,
        ids: Option<(&[u32], &[u32])>,
    ) -> Json {
        let arr = |x: &[u32]| Json::Arr(x.iter().map(|&t| Json::num(t)).collect());
        let mut ch = if self.chat {
            let content = if all.content.is_empty() && !all.tool_calls.is_empty() {
                Json::Null
            } else {
                Json::str(&all.content)
            };
            let mut msg = Json::obj([("role", Json::str("assistant")), ("content", content)]);
            if reasoning {
                msg.push(
                    "reasoning_content",
                    if all.reasoning.is_empty() {
                        Json::Null
                    } else {
                        Json::str(&all.reasoning)
                    },
                );
            }
            if !all.tool_calls.is_empty() {
                msg.push("tool_calls", tool_calls_json(&all.tool_calls, false));
            }
            Json::obj([
                ("index", Json::num(0)),
                ("message", msg),
                ("logprobs", Json::Null),
                ("finish_reason", Json::str(finish)),
            ])
        } else {
            let text = echo.unwrap_or("").to_string() + &all.content;
            Json::obj([
                ("index", Json::num(0)),
                ("text", Json::str(text)),
                ("logprobs", Json::Null),
                ("finish_reason", Json::str(finish)),
            ])
        };
        if let Some((_, out)) = ids {
            ch.push("token_ids", arr(out));
        }
        let mut v = self.base(false, vec![ch]);
        v.push("usage", usage.json());
        v.push("timings", timings);
        if let Some((prompt, _)) = ids {
            v.push("prompt_token_ids", arr(prompt));
        }
        v
    }
}
