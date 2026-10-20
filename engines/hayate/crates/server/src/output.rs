//! Incremental tag and stop parsing; a partial delimiter is never published.
use crate::json::{self, Value, quote};
use hayate_core::Result;

#[derive(Clone, Debug, PartialEq)]
pub enum Delta {
    Content(String),
    Reasoning(String),
    Tool(Value),
}
#[derive(Debug)]
pub struct Splitter {
    pending: String,
    reasoning: bool,
    tool: bool,
    stops: Vec<String>,
    pub stopped: bool,
    tools: Vec<Value>,
    /// GLM XML calls (`name<arg_key>k</arg_key><arg_value>v</arg_value>`);
    /// whitespace between and after calls is not content.
    called: bool,
}
impl Splitter {
    pub fn new(thinking: bool, stops: Vec<String>, tools: Vec<Value>) -> Self {
        Self {
            pending: String::new(),
            reasoning: thinking,
            tool: false,
            stops,
            stopped: false,
            tools,
            called: false,
        }
    }
    pub fn push(&mut self, text: &str) -> Result<Vec<Delta>> {
        if self.stopped {
            return Ok(vec![]);
        }
        self.pending.push_str(text);
        self.drain(false)
    }
    pub fn finish(&mut self) -> Result<Vec<Delta>> {
        self.drain(true)
    }
    pub fn finish_truncated(&mut self) -> Result<Vec<Delta>> {
        if self.tool {
            self.pending.clear();
            self.tool = false;
        }
        self.drain(true)
    }
    fn drain(&mut self, final_chunk: bool) -> Result<Vec<Delta>> {
        let mut out = Vec::new();
        loop {
            if self.stopped {
                self.pending.clear();
                break;
            }
            let tags: &[&str] = if self.tool {
                &["</tool_call>"]
            } else if self.reasoning {
                &["<think>", "</think>"]
            } else {
                &["<think>", "</think>", "<tool_call>"]
            };
            let mut found: Option<(usize, &str, bool)> = None;
            for (s, stop) in tags
                .iter()
                .map(|s| (*s, false))
                .chain(self.stops.iter().map(|s| (s.as_str(), true)))
            {
                if let Some(at) = self.pending.find(s)
                    && found.is_none_or(|(best, _, old_stop)| {
                        at < best || (at == best && stop && !old_stop)
                    })
                {
                    found = Some((at, s, stop));
                }
            }
            if let Some((at, tag, stop)) = found {
                let tag = tag.to_owned();
                if self.tool {
                    if stop {
                        self.stopped = true;
                        continue;
                    }
                    let call = parse_glm_call(&self.pending[..at], &self.tools);
                    out.push(Delta::Tool(call));
                    self.called = true;
                } else if at > 0 {
                    let s = self.pending[..at].to_owned();
                    out.extend(self.text(s));
                }
                self.pending.drain(..at + tag.len());
                if stop {
                    self.stopped = true;
                } else {
                    match tag.as_str() {
                        "<think>" => self.reasoning = true,
                        "</think>" => self.reasoning = false,
                        "<tool_call>" => self.tool = true,
                        "</tool_call>" => self.tool = false,
                        _ => unreachable!(),
                    }
                }
                continue;
            }
            if self.tool {
                if final_chunk {
                    // The reply ended inside a call: keep it, never drop it.
                    out.push(Delta::Tool(parse_glm_call(&self.pending, &self.tools)));
                    self.pending.clear();
                    self.tool = false;
                    self.called = true;
                    break;
                }
                break;
            }
            let keep = if final_chunk {
                0
            } else {
                tags.iter()
                    .copied()
                    .chain(self.stops.iter().map(String::as_str))
                    .flat_map(|tag| {
                        (1..tag.len())
                            .filter(|&n| tag.is_char_boundary(n))
                            .map(move |n| &tag[..n])
                    })
                    .filter(|prefix| self.pending.ends_with(prefix))
                    .map(str::len)
                    .max()
                    .unwrap_or(0)
            };
            let n = self.pending.len() - keep;
            if n > 0 {
                let s: String = self.pending.drain(..n).collect();
                out.extend(self.text(s));
            }
            break;
        }
        Ok(out)
    }
    fn text(&self, s: String) -> Option<Delta> {
        if self.reasoning {
            Some(Delta::Reasoning(s))
        } else if self.called && s.trim().is_empty() {
            None
        } else {
            Some(Delta::Content(s))
        }
    }
}
/// GLM `name<arg_key>k</arg_key><arg_value>v</arg_value>...`. A value is a
/// string when the tool schema says so, else JSON when it parses, else the raw
/// text. Malformed markup still yields a call: Studio reports the error back
/// to the model instead of losing the call.
pub fn parse_glm_call(text: &str, tools: &[Value]) -> Value {
    let (name, mut rest) = match text.find("<arg_key>") {
        Some(at) => (text[..at].trim(), &text[at..]),
        None => (text.trim(), ""),
    };
    let schema = tools
        .iter()
        .map(|t| t.optional("function").unwrap_or(t))
        .find(|f| f.optional("name").and_then(|n| n.string().ok()) == Some(name))
        .and_then(|f| f.optional("parameters"))
        .and_then(|p| p.optional("properties"));
    let mut args: Vec<(String, Value)> = Vec::new();
    let mut broken = String::new();
    while !rest.is_empty() {
        let pair = rest
            .strip_prefix("<arg_key>")
            .and_then(|r| r.split_once("</arg_key>"))
            .and_then(|(key, r)| {
                let r = r.trim_start().strip_prefix("<arg_value>")?;
                let (value, r) = match r.split_once("</arg_value>") {
                    Some((value, r)) => (value, r),
                    // A call cut inside the last value keeps what was written.
                    None => (r, ""),
                };
                Some((key.trim(), value, r))
            });
        let Some((key, value, tail)) = pair else {
            broken.push_str(rest);
            break;
        };
        let kind = schema
            .and_then(|s| s.optional(key))
            .and_then(|s| s.optional("type"))
            .and_then(|t| t.string().ok());
        let value = if kind == Some("string") {
            Value::String(value.into())
        } else {
            json::parse(value.trim()).unwrap_or_else(|_| Value::String(value.into()))
        };
        args.retain(|(k, _)| k != key);
        args.push((key.to_owned(), value));
        rest = tail.trim_start();
    }
    if !broken.trim().is_empty() {
        args.push(("_raw".into(), Value::String(broken)));
    }
    Value::Object(vec![
        ("type".into(), Value::String("function".into())),
        (
            "function".into(),
            Value::Object(vec![
                ("name".into(), Value::String(name.into())),
                (
                    "arguments".into(),
                    Value::String(compact(&Value::Object(args))),
                ),
            ]),
        ),
    ])
}
/// Compact JSON (no spaces) for OpenAI `arguments` strings.
pub fn compact(v: &Value) -> String {
    match v {
        Value::Array(a) => format!("[{}]", a.iter().map(compact).collect::<Vec<_>>().join(",")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}:{}", quote(k), compact(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        other => other.dump(),
    }
}
#[cfg(test)]
mod glm_tests {
    use super::*;
    fn tools() -> Vec<Value> {
        vec![
            json::parse(r#"{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"count":{"type":"integer"},"flags":{"type":"array"}}}}}"#).unwrap(),
        ]
    }
    fn run(text: &str, thinking: bool, split: usize) -> (String, String, Vec<Value>) {
        let mut p = Splitter::new(thinking, vec![], tools());
        let mut deltas = p.push(&text[..split]).unwrap();
        deltas.extend(p.push(&text[split..]).unwrap());
        deltas.extend(p.finish().unwrap());
        let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), vec![]);
        for d in deltas {
            match d {
                Delta::Content(s) => content.push_str(&s),
                Delta::Reasoning(s) => reasoning.push_str(&s),
                Delta::Tool(v) => calls.push(v),
            }
        }
        (content, reasoning, calls)
    }
    fn call(v: &Value) -> (String, Value) {
        let f = v.get("function").unwrap();
        (
            f.get("name").unwrap().string().unwrap().into(),
            json::parse(f.get("arguments").unwrap().string().unwrap()).unwrap(),
        )
    }
    #[test]
    fn parallel_calls_typed_by_schema_at_every_boundary() {
        let text = "考える</think>直します。\n<tool_call>edit<arg_key>path</arg_key><arg_value>src/日本.rs</arg_value><arg_key>old_string</arg_key><arg_value>  let x = 1;\n</arg_value><arg_key>count</arg_key><arg_value>2</arg_value><arg_key>flags</arg_key><arg_value>[true, null]</arg_value></tool_call>\n<tool_call>read<arg_key>path</arg_key><arg_value>42</arg_value></tool_call>";
        for split in (0..=text.len()).filter(|&n| text.is_char_boundary(n)) {
            let (content, reasoning, calls) = run(text, true, split);
            assert_eq!(reasoning, "考える");
            assert_eq!(content, "直します。\n", "split {split}");
            assert_eq!(calls.len(), 2);
            let (name, args) = call(&calls[0]);
            assert_eq!(name, "edit");
            assert_eq!(args.get("path").unwrap().string().unwrap(), "src/日本.rs");
            // Strings keep their exact whitespace (edit needs old_string verbatim).
            assert_eq!(
                args.get("old_string").unwrap().string().unwrap(),
                "  let x = 1;\n"
            );
            assert_eq!(args.get("count").unwrap().integer().unwrap(), 2);
            assert_eq!(args.get("flags").unwrap().array().unwrap().len(), 2);
            // No schema for `read`: JSON-looking values become JSON.
            let (name, args) = call(&calls[1]);
            assert_eq!(name, "read");
            assert_eq!(args.get("path").unwrap().integer().unwrap(), 42);
        }
    }
    #[test]
    fn broken_and_unfinished_calls_are_kept() {
        // A non-JSON value for a typed parameter stays a string.
        let (_, _, calls) = run(
            "<tool_call>edit<arg_key>count</arg_key><arg_value>{two</arg_value></tool_call>",
            false,
            0,
        );
        assert_eq!(
            call(&calls[0]).1.get("count").unwrap().string().unwrap(),
            "{two"
        );
        // Missing </arg_value>, trailing junk, no closing tag at EOS.
        let (_, _, calls) = run(
            "<tool_call>edit<arg_key>path</arg_key><arg_value>a.rs",
            false,
            5,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(
            call(&calls[0]).1.get("path").unwrap().string().unwrap(),
            "a.rs"
        );
        let (_, _, calls) = run("<tool_call>edit garbage</tool_call>", false, 0);
        assert_eq!(call(&calls[0]).0, "edit garbage");
        let (_, _, calls) = run(
            "<tool_call>edit<arg_key>path</arg_key><arg_value>a</arg_value>junk</tool_call>",
            false,
            0,
        );
        let args = call(&calls[0]).1;
        assert_eq!(args.get("_raw").unwrap().string().unwrap(), "junk");
        // Cut by max_tokens: no partial call reaches the client.
        let mut p = Splitter::new(false, vec![], tools());
        assert!(p.push("<tool_call>edit<arg_key>pa").unwrap().is_empty());
        assert!(p.finish_truncated().unwrap().is_empty());
    }
    #[test]
    fn thinking_off_starts_in_content_and_compact_arguments() {
        let (content, reasoning, calls) = run(
            "答え<tool_call>edit<arg_key>count</arg_key><arg_value>1</arg_value></tool_call>",
            false,
            3,
        );
        assert_eq!((content.as_str(), reasoning.as_str()), ("答え", ""));
        assert_eq!(
            calls[0]
                .get("function")
                .unwrap()
                .get("arguments")
                .unwrap()
                .string()
                .unwrap(),
            "{\"count\":1}"
        );
    }
}
