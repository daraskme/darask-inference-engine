//! The model's chat_template.jinja, written out by hand (no Jinja engine): system, user,
//! assistant (with `reasoning_content` and `tool_calls`) and tool turns, tool definitions, the
//! thinking switch (`enable_thinking`), `reasoning_effort` and `preserve_thinking`. `tojson` is
//! Python's `json.dumps`, as transformers defines it. `python/check_text.py` compares the output
//! with `apply_chat_template` byte for byte.

use crate::json::{Json, to_python};
use crate::unicode::py_trim;

/// An assistant's earlier tool call: the function name and its arguments (a JSON object).
#[derive(Debug, Clone, Default)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Vec<(String, Json)>,
}

#[derive(Debug, Clone, Default)]
pub struct Message {
    pub role: String,
    /// The content as the template's `render_content` gives it (text parts joined).
    pub content: String,
    /// `reasoning_content` when it is a string.
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    /// None = undefined (thinking on).
    pub enable_thinking: Option<bool>,
    /// xhigh (default), medium or low.
    pub reasoning_effort: Option<String>,
    /// None = undefined (keep every assistant turn's thinking block).
    pub preserve_thinking: Option<bool>,
    pub add_generation_prompt: bool,
}

impl Options {
    pub fn thinking(&self) -> bool {
        self.enable_thinking != Some(false)
    }
}

const XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
const TOOLS_HEAD: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_TAIL: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// The prompt text; errors carry the template's `raise_exception` messages. `tools` are the
/// request's tool definitions (OpenAI's `{"type": "function", "function": {...}}`), as given.
pub fn render(msgs: &[Message], tools: &[Json], o: &Options) -> Result<String, String> {
    if msgs.is_empty() {
        return Err("No messages provided.".into());
    }
    let mut instructions = "";
    if o.thinking() {
        instructions = match o.reasoning_effort.as_deref().unwrap_or("xhigh") {
            "xhigh" => XHIGH,
            "medium" => "",
            "low" => LOW,
            e => {
                return Err(format!(
                    "Unexpected reasoning effort {e}. Supported types are xhigh (default), medium, and low."
                ));
            }
        };
    }
    let mut out = String::new();
    let system = |out: &mut String, body: &str| {
        out.push_str("<|im_start|>system\n");
        out.push_str(body);
        out.push_str("<|im_end|>\n");
    };
    let first = if msgs[0].role == "system" {
        py_trim(&msgs[0].content)
    } else {
        ""
    };
    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        if !instructions.is_empty() {
            out.push_str(instructions);
            out.push_str("\n\n");
        }
        out.push_str(TOOLS_HEAD);
        for t in tools {
            out.push('\n');
            out.push_str(&to_python(t));
        }
        out.push_str(TOOLS_TAIL);
        if !first.is_empty() {
            out.push_str("\n\n");
            out.push_str(first);
        }
        out.push_str("<|im_end|>\n");
    } else if !first.is_empty() {
        let body = if instructions.is_empty() {
            first.to_string()
        } else {
            format!("{instructions}\n\n{first}")
        };
        system(&mut out, &body);
    } else if !instructions.is_empty() {
        system(&mut out, instructions);
    }
    // The last user turn that is not a wrapped tool response.
    let last_query = msgs.iter().enumerate().rev().find_map(|(i, m)| {
        let c = py_trim(&m.content);
        (m.role == "user" && !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>")))
            .then_some(i)
    });
    let Some(last_query) = last_query else {
        return Err("No user query found in messages.".into());
    };
    for (i, m) in msgs.iter().enumerate() {
        let content = py_trim(&m.content);
        match m.role.as_str() {
            "system" => {
                if i != 0 {
                    return Err("System message must be at the beginning.".into());
                }
            }
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                if o.preserve_thinking != Some(false) || i > last_query {
                    out.push_str("<think>\n");
                    out.push_str(py_trim(m.reasoning_content.as_deref().unwrap_or("")));
                    out.push_str("\n</think>\n\n");
                }
                out.push_str(content);
                for (j, tc) in m.tool_calls.iter().enumerate() {
                    out.push_str(match (j, content.is_empty()) {
                        (0, true) => "<tool_call>\n<function=",
                        (0, false) => "\n\n<tool_call>\n<function=",
                        _ => "\n<tool_call>\n<function=",
                    });
                    out.push_str(&tc.name);
                    out.push_str(">\n");
                    for (k, v) in &tc.arguments {
                        out.push_str("<parameter=");
                        out.push_str(k);
                        out.push_str(">\n");
                        match v {
                            Json::Str(s) => out.push_str(s),
                            v => out.push_str(&to_python(v)),
                        }
                        out.push_str("\n</parameter>\n");
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                if i > 0 && msgs[i - 1].role != "tool" {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
                if i + 1 == msgs.len() || msgs[i + 1].role != "tool" {
                    out.push_str("<|im_end|>\n");
                }
            }
            _ => return Err("Unexpected message role.".into()),
        }
    }
    if o.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        out.push_str(if o.enable_thinking == Some(false) {
            "<think>\n\n</think>\n\n"
        } else {
            "<think>\n"
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(role: &str, content: &str) -> Message {
        Message {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    #[test]
    fn basic() {
        let o = Options {
            enable_thinking: Some(false),
            add_generation_prompt: true,
            ..Default::default()
        };
        let p = render(&[m("system", " sys "), m("user", "hi")], &[], &o).unwrap();
        assert_eq!(
            p,
            "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let o = Options {
            add_generation_prompt: true,
            reasoning_effort: Some("medium".into()),
            ..Default::default()
        };
        let p = render(
            &[m("user", "hi"), m("assistant", "yo"), m("user", "again")],
            &[],
            &o,
        )
        .unwrap();
        assert_eq!(
            p,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nyo<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }
}
