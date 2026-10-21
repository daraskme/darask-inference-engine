//! GLM-5.3-Flash chat template (checkpoint `chat_template.jinja`).
//! The request processor expands images into their visual marker strings before rendering.
//! Byte-identical to transformers `apply_chat_template` for text turns, plus
//! three serve conventions the template lacks:
//! - `enable_thinking: false` appends `</think>`, so the turn starts as an
//!   empty thought, exactly how earlier assistant turns are rendered;
//! - `content: null` renders as empty text (Jinja would print `None`);
//! - `developer` messages render as `system`.
use crate::json::{self, Value};
use hayate_core::{Error, Result};

fn bad(s: &str) -> Error {
    Error::InvalidPlan(format!("GLM chat template: {s}"))
}

/// Template variable from `chat_template_kwargs`, else the top level.
pub fn kwarg<'a>(root: &'a Value, key: &str) -> Option<&'a Value> {
    root.optional("chat_template_kwargs")
        .and_then(|k| k.optional(key))
        .or_else(|| root.optional(key))
        .filter(|v| !matches!(v, Value::Null))
}
fn flag(root: &Value, key: &str, default: bool) -> Result<bool> {
    kwarg(root, key).map_or(Ok(default), Value::boolean)
}
/// Whether the reply starts inside `<think>`. `reasoning_effort: "none"` is the
/// OpenAI spelling of thinking off.
pub fn thinking(root: &Value) -> Result<bool> {
    if kwarg(root, "reasoning_effort").and_then(|v| v.string().ok()) == Some("none") {
        return Ok(false);
    }
    flag(root, "enable_thinking", true)
}

/// Python `str.isspace` (Rust's `char::is_whitespace` lacks U+001C..U+001F).
fn space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
fn strip(s: &str) -> &str {
    s.trim_matches(space)
}

/// `visible_text`: strings and text parts; media parts are not served.
fn text(content: Option<&Value>) -> Result<String> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(items)) => {
            let mut out = String::new();
            for item in items {
                match item {
                    Value::String(s) => out.push_str(s),
                    Value::Object(_) => match item.optional("type").and_then(|t| t.string().ok()) {
                        Some("text") => out.push_str(item.get("text")?.string()?),
                        Some(
                            "image" | "image_url" | "video" | "video_url" | "audio" | "audio_url"
                            | "input_audio",
                        ) => return Err(bad("vision/audio content is not supported")),
                        _ => {}
                    },
                    _ => {}
                }
            }
            Ok(out)
        }
        Some(other) => Ok(json_text(other)),
    }
}
/// Jinja's `{{ value }}` for a JSON number/bool (Python str()).
fn json_text(v: &Value) -> String {
    match v {
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => python_number(n),
        Value::Null => String::new(),
        other => python_json(other),
    }
}

/// Python `repr` of a JSON number as `json.loads` would produce it.
fn python_number(n: &str) -> String {
    if !n.contains(['.', 'e', 'E']) {
        return n.trim_start_matches('+').to_owned();
    }
    let Ok(x) = n.parse::<f64>() else {
        return n.to_owned();
    };
    if !x.is_finite() {
        return if x > 0. { "Infinity" } else { "-Infinity" }.into();
    }
    if x == 0. {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Shortest round-trip digits; Python switches to exponent form below 1e-4
    // and from 1e16 on.
    let sci = format!("{x:e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if negative { "-" } else { "" };
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        let body = if point <= 0 {
            format!("0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let tail = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        format!(
            "{sign}{}{tail}e{}{:02}",
            &digits[..1],
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        )
    }
}
/// transformers `tojson(ensure_ascii=False)`: json.dumps separators, insertion
/// order, no HTML escaping.
pub fn python_json(v: &Value) -> String {
    match v {
        Value::Number(n) => python_number(n),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(python_json).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", json::quote(k), python_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.dump(),
    }
}

/// Every tool object without `defer_loading` / `strict`.
fn tool_json(tool: &Value) -> Result<String> {
    let fields = tool.object()?;
    Ok(format!(
        "{{{}}}",
        fields
            .iter()
            .filter(|(k, _)| k != "defer_loading" && k != "strict")
            .map(|(k, v)| format!("\"{k}\": {}", python_json(v)))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}
fn function(tool: &Value) -> &Value {
    tool.optional("function").unwrap_or(tool)
}

fn id_of(v: &Value) -> Option<String> {
    let truthy = |key| {
        v.optional(key).and_then(|x| match x {
            Value::String(s) if !s.is_empty() => Some(s.clone()),
            Value::Number(n) => Some(n.clone()),
            _ => None,
        })
    };
    truthy("tool_call_id").or_else(|| truthy("id"))
}

/// `<arg_key>k</arg_key><arg_value>v</arg_value>` from an OpenAI arguments
/// string or mapping. Strings are written raw, everything else as JSON. Text
/// that is not a JSON object is kept verbatim so a broken call stays visible.
pub fn arguments(args: Option<&Value>) -> Result<String> {
    let parsed;
    let args = match args {
        None | Some(Value::Null) => return Ok(String::new()),
        Some(Value::String(s)) if s.trim().is_empty() => return Ok(String::new()),
        Some(Value::String(s)) => match json::parse(s) {
            Ok(v @ Value::Object(_)) => {
                parsed = v;
                &parsed
            }
            _ => return Ok(s.clone()),
        },
        Some(v) => v,
    };
    let mut out = String::new();
    for (k, v) in args.object()? {
        out.push_str("<arg_key>");
        out.push_str(k);
        out.push_str("</arg_key><arg_value>");
        match v {
            Value::String(s) => out.push_str(s),
            other => out.push_str(&python_json(other)),
        }
        out.push_str("</arg_value>");
    }
    Ok(out)
}

pub fn render_glm(root: &Value) -> Result<String> {
    let messages = root.get("messages")?.array()?;
    if messages.is_empty() {
        return Err(bad("no messages"));
    }
    let roles = messages
        .iter()
        .map(|m| {
            Ok(match m.get("role")?.string()? {
                "developer" => "system",
                r @ ("system" | "user" | "assistant" | "tool") => r,
                _ => return Err(bad("roles: system/developer/user/assistant/tool")),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let effort = match kwarg(root, "reasoning_effort").and_then(|v| v.string().ok()) {
        Some("low") => "Low",
        Some("high") => "High",
        _ => "Max",
    };
    let clear = flag(root, "clear_thinking", true)?;
    let tools = root
        .optional("tools")
        .filter(|v| !matches!(v, Value::Null))
        .map(Value::array)
        .transpose()?
        .unwrap_or(&[]);
    let mut out = String::from("[gMASK]<sop><|system|>Reasoning Effort: ");
    out.push_str(effort);
    if !tools.is_empty() {
        out.push_str("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
        for tool in tools {
            let f = function(tool);
            if f.optional("defer_loading")
                .is_some_and(|v| !matches!(v, Value::Null | Value::Bool(false)))
            {
                continue;
            }
            out.push_str(&tool_json(f)?);
            out.push('\n');
        }
        out.push_str("</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>");
    }
    let last_user = roles.iter().rposition(|r| *r == "user");
    for (i, (m, role)) in messages.iter().zip(&roles).enumerate() {
        match *role {
            "system" => {
                out.push_str("<|system|>");
                out.push_str(&text(m.optional("content"))?);
            }
            "user" => {
                out.push_str("<|user|>");
                out.push_str(&text(m.optional("content"))?);
            }
            "assistant" => {
                out.push_str("<|assistant|>");
                let mut content = text(m.optional("content"))?;
                let mut reasoning = None;
                if let Some(Value::String(r)) = m.optional("reasoning_content") {
                    reasoning = Some(r.clone());
                } else if content.contains("</think>") {
                    let before = content.split("</think>").next().unwrap();
                    reasoning = Some(before.rsplit("<think>").next().unwrap().to_owned());
                    content = content.rsplit("</think>").next().unwrap().to_owned();
                }
                match reasoning {
                    Some(r) if !clear || last_user.is_none_or(|u| i > u) => {
                        out.push_str("<think>");
                        out.push_str(&r);
                        out.push_str("</think>");
                    }
                    _ => out.push_str("<think></think>"),
                }
                out.push_str(strip(&content));
                let calls = m
                    .optional("tool_calls")
                    .filter(|v| !matches!(v, Value::Null))
                    .map(Value::array)
                    .transpose()?
                    .unwrap_or(&[]);
                for call in calls {
                    let f = call.optional("function").unwrap_or(call);
                    out.push_str("<tool_call>");
                    out.push_str(f.get("name")?.string()?);
                    out.push_str(&arguments(f.optional("arguments"))?);
                    out.push_str("</tool_call>");
                }
            }
            _ => {
                if i > 0 && roles[i - 1] == "tool" {
                    continue;
                }
                out.push_str("<|observation|>");
                let end = (i..messages.len())
                    .take_while(|&j| roles[j] == "tool")
                    .last()
                    .unwrap();
                let block = &messages[i..=end];
                let calls = (i > 0 && roles[i - 1] == "assistant")
                    .then(|| messages[i - 1].optional("tool_calls"))
                    .flatten()
                    .and_then(|v| v.array().ok())
                    .filter(|c| !c.is_empty());
                let ids = block.iter().map(id_of).collect::<Vec<_>>();
                let sortable = calls.is_some_and(|calls| {
                    let call_ids = calls.iter().map(id_of).collect::<Vec<_>>();
                    ids.iter().all(|id| {
                        id.as_ref().is_some_and(|id| {
                            ids.iter().filter(|o| o.as_ref() == Some(id)).count() == 1
                                && call_ids.iter().any(|c| c.as_ref() == Some(id))
                        })
                    }) && call_ids
                        .iter()
                        .enumerate()
                        .all(|(a, id)| id.is_some() && !call_ids[a + 1..].contains(id))
                });
                let response = |m: &Value| -> Result<String> {
                    Ok(format!(
                        "<tool_response>{}</tool_response>",
                        text(m.optional("content"))?
                    ))
                };
                if sortable {
                    for call in calls.unwrap() {
                        let id = id_of(call);
                        for (m, mid) in block.iter().zip(&ids) {
                            if *mid == id {
                                out.push_str(&response(m)?);
                            }
                        }
                    }
                } else {
                    for m in block {
                        out.push_str(&response(m)?);
                    }
                }
            }
        }
    }
    if root
        .optional("add_generation_prompt")
        .map_or(Ok(true), Value::boolean)?
    {
        out.push_str("<|assistant|><think>");
        if !thinking(root)? {
            out.push_str("</think>");
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// transformers 5.17 `apply_chat_template` output for the dealignai
    /// checkpoint (python-tools/hayate_tools/serve_template_check.py glm_expected;
    /// the full 868-case sweep runs with `--model glm53`).
    #[test]
    fn golden_strings_match_hf_template() {
        let golden = include_str!("glm_golden.jsonl");
        let mut n = 0;
        for line in golden.lines() {
            let case = json::parse(line).unwrap();
            assert_eq!(
                render_glm(case.get("request").unwrap()).unwrap(),
                case.get("rendered").unwrap().string().unwrap(),
                "{line}"
            );
            n += 1;
        }
        assert_eq!(n, 3);
        for bad in [
            r#"{"messages":[]}"#,
            r#"{"messages":[{"role":"bot","content":"x"}]}"#,
            r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]}"#,
        ] {
            assert!(render_glm(&json::parse(bad).unwrap()).is_err(), "{bad}");
        }
    }
    #[test]
    fn python_numbers() {
        for (json, python) in [
            ("2", "2"),
            ("-3", "-3"),
            ("1.0", "1.0"),
            ("2.50", "2.5"),
            ("0.1", "0.1"),
            ("1e-5", "1e-05"),
            ("0.0001", "0.0001"),
            ("1E5", "100000.0"),
            ("1e16", "1e+16"),
            ("123456789012345.6", "123456789012345.6"),
            ("-0.0", "-0.0"),
            ("1.5e-7", "1.5e-07"),
        ] {
            assert_eq!(python_number(json), python, "{json}");
        }
    }
    #[test]
    fn broken_arguments_stay_visible() {
        assert_eq!(
            arguments(Some(&Value::String("{\"path\": ".into()))).unwrap(),
            "{\"path\": "
        );
        assert_eq!(
            arguments(Some(
                &json::parse(r#"{"a":"日\n","n":1.0,"o":{"x":[true,null]}}"#).unwrap()
            ))
            .unwrap(),
            "<arg_key>a</arg_key><arg_value>日\n</arg_value><arg_key>n</arg_key><arg_value>1.0</arg_value><arg_key>o</arg_key><arg_value>{\"x\": [true, null]}</arg_value>"
        );
    }
}
