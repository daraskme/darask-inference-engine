//! Tool calls in the model's output. The template teaches the model this format:
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! `ToolSplit` takes the answer text as it streams and separates the plain text from finished
//! calls, which become OpenAI `tool_calls` with the arguments as a JSON object. Values are typed
//! by the tool's parameter schema the way vLLM's qwen3_coder parser does it (string stays text,
//! integer / number / boolean parsed, object / array as JSON, "null" is null).

use crate::json::{Json, to_python};

const START: &str = "<tool_call>";
const END: &str = "</tool_call>";

#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub name: String,
    /// The arguments object as JSON text.
    pub arguments: String,
}

/// Parameter types by function name, from the request's `tools`.
#[derive(Debug, Clone, Default)]
pub struct Schema {
    funcs: Vec<(String, Vec<(String, String)>)>,
}

impl Schema {
    pub fn from_tools(tools: &[Json]) -> Self {
        let mut funcs = Vec::new();
        for t in tools {
            let f = t.get("function").unwrap_or(t);
            let Some(name) = f.get("name").and_then(Json::as_str) else {
                continue;
            };
            let props = f
                .get("parameters")
                .and_then(|p| p.get("properties"))
                .and_then(Json::as_obj)
                .unwrap_or(&[]);
            let params = props
                .iter()
                .map(|(k, v)| {
                    let ty = match v.get("type") {
                        Some(Json::Str(s)) => s.clone(),
                        // ["string", "null"]: the first that is not null
                        Some(Json::Arr(a)) => a
                            .iter()
                            .filter_map(Json::as_str)
                            .find(|s| *s != "null")
                            .unwrap_or("string")
                            .to_string(),
                        _ => "string".into(),
                    };
                    (k.clone(), ty.trim().to_ascii_lowercase())
                })
                .collect();
            funcs.push((name.to_string(), params));
        }
        Self { funcs }
    }

    fn param_type(&self, func: &str, param: &str) -> Option<&str> {
        let (_, ps) = self.funcs.iter().find(|(n, _)| n == func)?;
        ps.iter().find(|(p, _)| p == param).map(|(_, t)| t.as_str())
    }
}

fn convert(value: &str, ty: Option<&str>) -> Json {
    if value.eq_ignore_ascii_case("null") {
        return Json::Null;
    }
    let Some(ty) = ty else {
        return Json::str(value);
    };
    let s = || Json::str(value);
    match ty {
        "string" | "str" | "text" | "varchar" | "char" | "enum" => s(),
        t if ["int", "uint", "long", "short", "unsigned"]
            .iter()
            .any(|p| t.starts_with(p)) =>
        {
            value.trim().parse::<i64>().map_or_else(|_| s(), Json::Int)
        }
        t if t.starts_with("num") || t.starts_with("float") => match value.trim().parse::<f64>() {
            Ok(f) if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.0e15 => Json::Int(f as i64),
            Ok(f) => Json::Num(f),
            Err(_) => s(),
        },
        "boolean" | "bool" | "binary" => Json::Bool(value.trim().eq_ignore_ascii_case("true")),
        _ => Json::parse(value.trim()).unwrap_or_else(|_| s()),
    }
}

/// One `<tool_call>…</tool_call>` body; None when it has no `<function=…>`.
pub fn parse_call(body: &str, schema: &Schema) -> Option<Call> {
    let f = body.find("<function=")? + "<function=".len();
    let rest = &body[f..];
    let gt = rest.find('>')?;
    let name = rest[..gt].trim().to_string();
    let mut inner = &rest[gt + 1..];
    if let Some(e) = inner.find("</function>") {
        inner = &inner[..e];
    }
    let mut args = Vec::new();
    let mut at = inner;
    while let Some(p) = at.find("<parameter=") {
        let after = &at[p + "<parameter=".len()..];
        let Some(gt) = after.find('>') else { break };
        let key = after[..gt].trim().to_string();
        let body = &after[gt + 1..];
        // The value ends at </parameter>, or leniently at the next parameter or the end.
        let end = ["</parameter>", "<parameter="]
            .iter()
            .filter_map(|t| body.find(t))
            .min()
            .unwrap_or(body.len());
        let mut v = &body[..end];
        v = v.strip_prefix('\n').unwrap_or(v);
        v = v.strip_suffix('\n').unwrap_or(v);
        let ty = schema.param_type(&name, &key);
        // Without a schema entry the value stays a string (vLLM does the same).
        args.push((
            key,
            if schema.funcs.iter().any(|(n, _)| *n == name) {
                convert(v, ty)
            } else {
                Json::str(v)
            },
        ));
        at = &body[end..];
    }
    Some(Call {
        name,
        arguments: to_python(&Json::Obj(args)),
    })
}

/// Splits streamed answer text into plain text and finished tool calls. Holds back what could
/// still turn out to be a call: a partial `<tool_call>` and the blanks before it.
#[derive(Debug, Default)]
pub struct ToolSplit {
    schema: Schema,
    held: String,
    /// Inside `<tool_call>`: the body so far.
    call: Option<String>,
    pub calls: usize,
}

impl ToolSplit {
    pub fn new(schema: Schema) -> Self {
        Self {
            schema,
            ..Default::default()
        }
    }

    /// (text to show, finished calls).
    pub fn push(&mut self, s: &str) -> (String, Vec<Call>) {
        let mut text = String::new();
        let mut calls = Vec::new();
        self.held.push_str(s);
        loop {
            if let Some(body) = self.call.as_mut() {
                body.push_str(&self.held);
                self.held.clear();
                let Some(e) = body.find(END) else { break };
                let rest = body[e + END.len()..].to_string();
                let b = body[..e].to_string();
                self.call = None;
                match parse_call(&b, &self.schema) {
                    Some(c) => {
                        self.calls += 1;
                        calls.push(c);
                    }
                    None => text.push_str(&format!("{START}{b}{END}")),
                }
                self.held = rest;
                continue;
            }
            if let Some(p) = self.held.find(START) {
                // Text before a call: its trailing blanks belong to the call's layout.
                text.push_str(self.held[..p].trim_end());
                self.call = Some(String::new());
                self.held.drain(..p + START.len());
                continue;
            }
            let partial = self
                .held
                .char_indices()
                .map(|(i, _)| i)
                .find(|&i| START.starts_with(&self.held[i..]));
            let keep = self.held[..partial.unwrap_or(self.held.len())]
                .trim_end()
                .len();
            text.push_str(&self.held[..keep]);
            self.held.drain(..keep);
            break;
        }
        (text, calls)
    }

    /// The rest at the end: held blanks, or an unfinished call as plain text.
    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        if let Some(body) = self.call.take() {
            out.push_str(START);
            out.push_str(&body);
        }
        out.push_str(&std::mem::take(&mut self.held));
        if self.calls > 0 {
            out.trim_end().to_string()
        } else {
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Schema {
        let t = Json::parse(r#"[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"},"ratio":{"type":"number"},"all":{"type":"boolean"},"tags":{"type":"array"},"opt":{"type":["string","null"]}}}}}]"#).unwrap();
        Schema::from_tools(t.as_arr().unwrap())
    }

    #[test]
    fn parse() {
        let body = "\n<function=read>\n<parameter=path>\n/tmp/a b.txt\n</parameter>\n<parameter=limit>\n20\n</parameter>\n<parameter=ratio>\n0.5\n</parameter>\n<parameter=all>\nTrue\n</parameter>\n<parameter=tags>\n[\"x\", 1]\n</parameter>\n<parameter=opt>\nnull\n</parameter>\n<parameter=extra>\n7\n</parameter>\n</function>\n";
        let c = parse_call(body, &schema()).unwrap();
        assert_eq!(c.name, "read");
        assert_eq!(
            c.arguments,
            r#"{"path": "/tmp/a b.txt", "limit": 20, "ratio": 0.5, "all": true, "tags": ["x", 1], "opt": null, "extra": "7"}"#
        );
    }

    #[test]
    fn split_streamed() {
        let text = "Let me look.\n\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/main.rs\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=read>\n<parameter=limit>\n3\n</parameter>\n</function>\n</tool_call>";
        for step in [1, 3, 7, text.len()] {
            let mut t = ToolSplit::new(schema());
            let (mut shown, mut calls) = (String::new(), Vec::new());
            let chars: Vec<char> = text.chars().collect();
            for piece in chars.chunks(step) {
                let (s, c) = t.push(&piece.iter().collect::<String>());
                shown += &s;
                calls.extend(c);
            }
            shown += &t.finish();
            assert_eq!(shown, "Let me look.", "step {step}");
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[1].arguments, r#"{"limit": 3}"#);
        }
        let mut t = ToolSplit::new(schema());
        let (s, c) = t.push("a < b and <tool");
        assert_eq!((s.as_str(), c.len()), ("a < b and", 0));
        assert_eq!(t.finish(), " <tool");
    }
}
