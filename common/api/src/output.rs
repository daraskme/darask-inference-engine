//! A request's output, token by token: detokenize (never splitting UTF-8) → stop strings →
//! `<think>` split → tool calls. Each token gives a `Piece` (what to send now) and whether to go
//! on; the same pieces make the streaming chunks and the non-streaming message.

use crate::generate::Flow;
use crate::request::Prepared;
use crate::text::{StopStrings, ThinkSplit};
use crate::tokenizer::{DecodeStream, Tokenizer};
use crate::tools::ToolSplit;

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallOut {
    pub index: usize,
    pub id: String,
    pub name: String,
    /// JSON text of the arguments object.
    pub arguments: String,
}

/// Output to send: reasoning text, answer text, finished tool calls.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Piece {
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<ToolCallOut>,
}

impl Piece {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty() && self.tool_calls.is_empty()
    }

    pub fn append(&mut self, p: Piece) {
        self.reasoning += &p.reasoning;
        self.content += &p.content;
        self.tool_calls.extend(p.tool_calls);
    }
}

#[derive(Debug)]
pub struct Output {
    dec: DecodeStream,
    stops: StopStrings,
    think: Option<ThinkSplit>,
    tools: Option<ToolSplit>,
    stop_ids: Vec<u32>,
    /// "length" until something else ends it.
    finish: &'static str,
    /// A stop string matched: the rest of the text is dropped.
    cut: bool,
    /// Every generated id (with the EOS that ended it).
    pub ids: Vec<u32>,
    /// Tool call ids are `call_<id_base hex>_<n>`.
    id_base: u64,
    calls: usize,
}

impl Output {
    pub fn new(p: &Prepared, id_base: u64) -> Self {
        Self {
            dec: DecodeStream::new(p.skip_special),
            stops: StopStrings::new(p.stop.clone()),
            think: p.reasoning.then(|| ThinkSplit::new(true)),
            tools: p.tools.clone().map(ToolSplit::new),
            stop_ids: p.stop_ids.clone(),
            finish: "length",
            cut: false,
            ids: Vec::new(),
            id_base,
            calls: 0,
        }
    }

    /// stop / length / tool_calls.
    pub fn finish_reason(&self) -> &'static str {
        if self.finish == "stop" && self.calls > 0 {
            "tool_calls"
        } else {
            self.finish
        }
    }

    fn route(&mut self, text: &str, out: &mut Piece) {
        if text.is_empty() {
            return;
        }
        let d = match self.think.as_mut() {
            Some(t) => t.push(text),
            None => crate::text::Delta {
                reasoning: String::new(),
                content: text.to_string(),
            },
        };
        out.reasoning += &d.reasoning;
        self.answer(&d.content, out);
    }

    fn answer(&mut self, text: &str, out: &mut Piece) {
        let Some(t) = self.tools.as_mut() else {
            out.content += text;
            return;
        };
        let (s, calls) = t.push(text);
        out.content += &s;
        for c in calls {
            out.tool_calls.push(ToolCallOut {
                index: self.calls,
                id: format!(
                    "call_{:012x}{:02}",
                    self.id_base & 0xffff_ffff_ffff,
                    self.calls
                ),
                name: c.name,
                arguments: c.arguments,
            });
            self.calls += 1;
        }
    }

    /// One generated token: what to send, and whether generation goes on.
    pub fn token(&mut self, tok: &Tokenizer, t: u32) -> (Flow, Piece) {
        let mut out = Piece::default();
        self.ids.push(t);
        if self.stop_ids.contains(&t) {
            self.finish = "stop";
            return (Flow::Stop, out);
        }
        let s = self.dec.push(tok, t);
        let (text, hit) = self.stops.push(&s);
        self.route(&text, &mut out);
        if hit {
            self.finish = "stop";
            self.cut = true;
            return (Flow::Stop, out);
        }
        (Flow::Continue, out)
    }

    /// Everything still held back, at the end of generation.
    pub fn finish(&mut self) -> Piece {
        let mut out = Piece::default();
        if !self.cut {
            let s = self.dec.finish();
            let (text, hit) = self.stops.push(&s);
            self.route(&text, &mut out);
            if !hit {
                let rest = self.stops.finish();
                self.route(&rest, &mut out);
            }
        }
        if let Some(t) = self.think.as_mut() {
            let d = t.finish();
            out.reasoning += &d.reasoning;
            self.answer(&d.content, &mut out);
        }
        if let Some(t) = self.tools.as_mut() {
            out.content += &t.finish();
        }
        out
    }
}
