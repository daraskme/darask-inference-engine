//! Tsumuji's OpenAI-compatible server (a subset of plan M5), and the library under it, which a
//! front end such as Hayate's serve can link without this HTTP server (std only):
//!
//! - `model::TextModel`: tokenizer (`tokenizer`, `unicode`), generation defaults, context
//! - `request`: request JSON → `Prepared` (chat template in `chat`, tools in `tools`)
//! - `generate`: the `Generator` boundary to the engine, with the prefix cache
//! - `output::Output`: generated tokens → text, `<think>` split (`text`), tool calls (`tools`)
//! - `wire`: OpenAI / Hayate serve JSON (chunks, usage with cached tokens, timings)
//! - `api`, `http`, `server`: the standalone HTTP server

pub mod api;
pub mod chat;
pub mod check;
pub mod generate;
pub mod http;
pub mod json;
pub mod model;
pub mod output;
pub mod request;
pub mod sample;
pub mod server;
pub mod text;
pub mod tokenizer;
pub mod tools;
pub mod unicode;

pub mod wire;
