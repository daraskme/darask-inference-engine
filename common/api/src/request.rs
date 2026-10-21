//! Request bodies (OpenAI chat completions and completions, plus the extensions Hayate's serve
//! and vLLM / SGLang clients send) into a `Prepared` request: prompt ids, sampling, limits, stop
//! conditions and how to treat the output. No HTTP here.

use crate::chat::{self, Message, ToolCall};
use crate::json::Json;
use crate::model::TextModel;
use crate::sample::Params;
use crate::tools::Schema;

/// A request the server refuses before generating: HTTP status and message.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
}

impl From<String> for ApiError {
    fn from(message: String) -> Self {
        Self {
            status: 400,
            message,
        }
    }
}

impl From<&str> for ApiError {
    fn from(message: &str) -> Self {
        Self {
            status: 400,
            message: message.into(),
        }
    }
}

/// A validated request, ready for the generator.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub chat: bool,
    pub prompt: Vec<u32>,
    pub images: Vec<darask_vision::Span>,
    /// Completions with `echo`: the prompt text to put before the output.
    pub echo: Option<String>,
    pub params: Params,
    pub max_new: usize,
    pub stop: Vec<String>,
    /// Token ids that end generation (EOS unless `ignore_eos`).
    pub stop_ids: Vec<u32>,
    /// Split `<think>…</think>` off into `reasoning_content`.
    pub reasoning: bool,
    /// Parse `<tool_call>` blocks in the answer (the request had tools).
    pub tools: Option<Schema>,
    pub skip_special: bool,
    pub stream: bool,
    /// vLLM's `return_token_ids`: prompt and output ids in the response.
    pub return_token_ids: bool,
    /// Prompt positions worth a state snapshot for the prefix cache: where the last turn starts,
    /// the part of the prompt the next request of the same conversation shares for sure.
    pub marks: Vec<usize>,
    /// Hayate's `reasoning_budget` (accepted; not applied yet).
    pub reasoning_budget: Option<u64>,
    /// Which form of the loaded model runs it (0 = as loaded; see `GenRequest::variant`).
    pub variant: u32,
}

fn num(body: &Json, key: &str) -> Result<Option<f64>, String> {
    match body.field(key) {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("'{key}' must be a number")),
    }
}

fn boolean(body: &Json, key: &str) -> Result<Option<bool>, String> {
    match body.field(key) {
        None => Ok(None),
        Some(v) => v
            .as_bool()
            .map(Some)
            .ok_or_else(|| format!("'{key}' must be a boolean")),
    }
}

/// Fields both endpoints share: sampling, limits, stop, stream. `seed` is used when the request
/// gives none.
fn common(
    m: &TextModel,
    body: &Json,
    chat: bool,
    prompt: Vec<u32>,
    seed: u64,
) -> Result<Prepared, ApiError> {
    if let Some(n) = body.field("n").map(Json::as_u64)
        && n != Some(1)
    {
        return Err("only n = 1 is supported".into());
    }
    if body
        .field("best_of")
        .and_then(Json::as_u64)
        .is_some_and(|v| v > 1)
    {
        return Err("'best_of' > 1 is not supported".into());
    }
    let logprobs = match body.field("logprobs") {
        None | Some(Json::Bool(false)) => false,
        Some(v @ (Json::Int(_) | Json::Num(_))) => v.as_f64().unwrap_or(0.0) > 0.0 || chat,
        Some(_) => true,
    };
    if logprobs
        || body
            .field("top_logprobs")
            .and_then(Json::as_u64)
            .is_some_and(|v| v > 0)
    {
        return Err("logprobs are not supported yet".into());
    }
    let mut p = m.info.defaults.clone();
    if let Some(t) = num(body, "temperature")? {
        if !(0.0..=100.0).contains(&t) {
            return Err("'temperature' must be >= 0".into());
        }
        p.temperature = t as f32;
    }
    if let Some(v) = num(body, "top_p")? {
        if !(v > 0.0 && v <= 1.0) {
            return Err("'top_p' must be in (0, 1]".into());
        }
        p.top_p = v as f32;
    }
    if let Some(v) = num(body, "top_k")? {
        if v.fract() != 0.0 || v < -1.0 {
            return Err("'top_k' must be an integer (0 or -1 = off)".into());
        }
        p.top_k = v.max(0.0) as usize;
    }
    if let Some(v) = num(body, "min_p")? {
        if !(0.0..=1.0).contains(&v) {
            return Err("'min_p' must be in [0, 1]".into());
        }
        p.min_p = v as f32;
    }
    for (k, slot) in [
        ("presence_penalty", &mut p.presence_penalty),
        ("frequency_penalty", &mut p.frequency_penalty),
    ] {
        if let Some(v) = num(body, k)? {
            if !(-2.0..=2.0).contains(&v) {
                return Err(format!("'{k}' must be in [-2, 2]").into());
            }
            *slot = v as f32;
        }
    }
    if let Some(v) = num(body, "repetition_penalty")? {
        if !(v > 0.0 && v <= 10.0) {
            return Err("'repetition_penalty' must be in (0, 10]".into());
        }
        p.repetition_penalty = v as f32;
    }
    p.seed = match body.field("seed") {
        None => seed,
        Some(v) => v.as_i64().ok_or("'seed' must be an integer")? as u64,
    };
    let stop = match body.field("stop") {
        None => Vec::new(),
        Some(Json::Str(s)) => vec![s.clone()],
        Some(Json::Arr(a)) => a
            .iter()
            .map(|s| {
                s.as_str()
                    .map(str::to_string)
                    .ok_or("'stop' must be a string or a list of strings")
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err("'stop' must be a string or a list of strings".into()),
    };
    let mut stop_ids = if boolean(body, "ignore_eos")?.unwrap_or(false) {
        Vec::new()
    } else {
        m.info.eos.clone()
    };
    if let Some(Json::Arr(a)) = body.field("stop_token_ids") {
        stop_ids.extend(
            a.iter()
                .map(|v| {
                    v.as_u64()
                        .map(|x| x as u32)
                        .ok_or("'stop_token_ids' must be integers")
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    if prompt.is_empty() {
        return Err("the prompt is empty".into());
    }
    let key = if body.field("max_completion_tokens").is_some() {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    let asked = match num(body, key)? {
        Some(v) if v.fract() != 0.0 || v < 1.0 => {
            return Err(format!("'{key}' must be a positive integer").into());
        }
        Some(v) => Some(v as usize),
        None => None,
    };
    // Hayate's serve contract: a request that does not fit is refused before the response, in
    // this wording (Studio shrinks max_tokens from it, or summarizes when the prompt alone is too long).
    let ctx = m.ctx;
    if prompt.len() + asked.unwrap_or(1) > ctx {
        return Err(format!(
            "prompt ({} tokens) + max_tokens ({}) exceeds server context ({ctx} tokens)",
            prompt.len(),
            asked.unwrap_or(0)
        )
        .into());
    }
    let max_new = match asked {
        Some(v) => v,
        None if chat => ctx - prompt.len(),
        None => 16.min(ctx - prompt.len()), // OpenAI's default for completions
    };
    Ok(Prepared {
        images: Vec::new(),
        chat,
        prompt,
        echo: None,
        params: p,
        max_new,
        stop,
        stop_ids,
        reasoning: false,
        tools: None,
        skip_special: boolean(body, "skip_special_tokens")?.unwrap_or(true),
        stream: boolean(body, "stream")?.unwrap_or(false),
        return_token_ids: boolean(body, "return_token_ids")?.unwrap_or(false),
        marks: Vec::new(),
        reasoning_budget: body.field("reasoning_budget").and_then(Json::as_u64),
        variant: 0,
    })
}

/// A tool call's `arguments`: a JSON object, or a string holding one (OpenAI's form).
fn tool_arguments(v: Option<&Json>) -> Result<Vec<(String, Json)>, String> {
    let parsed;
    let v = match v {
        None | Some(Json::Null) => return Ok(Vec::new()),
        Some(Json::Str(s)) if s.trim().is_empty() => return Ok(Vec::new()),
        Some(Json::Str(s)) => {
            parsed =
                Json::parse(s).map_err(|e| format!("tool call arguments are not JSON: {e}"))?;
            &parsed
        }
        Some(v) => v,
    };
    v.as_obj()
        .map(<[_]>::to_vec)
        .ok_or_else(|| "tool call arguments must be a JSON object".into())
}

/// OpenAI `messages` as the template's messages (text content only).
pub fn parse_messages(v: Option<&Json>) -> Result<Vec<Message>, String> {
    let arr = v
        .and_then(Json::as_arr)
        .ok_or("'messages' must be a list")?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, m) in arr.iter().enumerate() {
        let mut role = m
            .field("role")
            .and_then(Json::as_str)
            .ok_or_else(|| format!("messages[{i}]: 'role' must be a string"))?;
        if role == "developer" {
            role = "system";
        }
        let content = match m.field("content") {
            None => String::new(),
            Some(Json::Str(s)) => s.clone(),
            Some(Json::Arr(parts)) => {
                let mut s = String::new();
                for p in parts {
                    match (p.field("type").and_then(Json::as_str), p.field("text")) {
                        (Some("text") | None, Some(Json::Str(t))) => s.push_str(t),
                        (ty, _) => {
                            return Err(format!(
                                "messages[{i}]: only text content is supported (got part type {})",
                                ty.unwrap_or("?")
                            ));
                        }
                    }
                }
                s
            }
            Some(_) => {
                return Err(format!(
                    "messages[{i}]: 'content' must be a string or a list of parts"
                ));
            }
        };
        let mut tool_calls = Vec::new();
        for (j, tc) in m
            .field("tool_calls")
            .and_then(Json::as_arr)
            .unwrap_or(&[])
            .iter()
            .enumerate()
        {
            let f = tc.field("function").unwrap_or(tc);
            let name = f
                .field("name")
                .and_then(Json::as_str)
                .ok_or_else(|| format!("messages[{i}].tool_calls[{j}]: no function name"))?;
            let arguments = tool_arguments(f.get("arguments"))
                .map_err(|e| format!("messages[{i}].tool_calls[{j}]: {e}"))?;
            tool_calls.push(ToolCall {
                name: name.to_string(),
                arguments,
            });
        }
        if m.field("function_call").is_some() {
            return Err(format!(
                "messages[{i}]: 'function_call' is not supported (use tool_calls)"
            ));
        }
        let reasoning_content = m
            .field("reasoning_content")
            .or(m.field("reasoning"))
            .and_then(Json::as_str)
            .map(str::to_string);
        out.push(Message {
            role: role.to_string(),
            content,
            reasoning_content,
            tool_calls,
        });
    }
    Ok(out)
}

/// Template options from the top-level switches (Hayate's serve: `enable_thinking`,
/// `reasoning_effort`, `preserve_thinking`) and `chat_template_kwargs` (vLLM / SGLang), which
/// wins when both are given.
pub fn chat_options(body: &Json) -> Result<chat::Options, String> {
    let mut o = chat::Options {
        add_generation_prompt: true,
        ..Default::default()
    };
    let effort = |o: &mut chat::Options, e: &str| -> Result<(), String> {
        match e {
            "none" | "minimal" => o.enable_thinking = Some(false),
            "low" | "medium" => o.reasoning_effort = Some(e.into()),
            "high" | "xhigh" | "max" => o.reasoning_effort = Some("xhigh".into()),
            e => {
                return Err(format!(
                    "unknown reasoning_effort '{e}' (none, low, medium, high, xhigh, max)"
                ));
            }
        }
        Ok(())
    };
    if let Some(v) = body.field("reasoning_effort") {
        effort(
            &mut o,
            v.as_str().ok_or("'reasoning_effort' must be a string")?,
        )?;
    }
    if let Some(v) = boolean(body, "enable_thinking")? {
        o.enable_thinking = Some(v);
    }
    if let Some(v) = boolean(body, "preserve_thinking")? {
        o.preserve_thinking = Some(v);
    }
    if let Some(k) = body.field("chat_template_kwargs") {
        if k.as_obj().is_none() {
            return Err("'chat_template_kwargs' must be an object".into());
        }
        if let Some(v) = boolean(k, "enable_thinking")? {
            o.enable_thinking = Some(v);
        }
        if let Some(v) = boolean(k, "preserve_thinking")? {
            o.preserve_thinking = Some(v);
        }
        if let Some(v) = k.field("reasoning_effort") {
            effort(
                &mut o,
                v.as_str().ok_or("'reasoning_effort' must be a string")?,
            )?;
        }
    }
    if let Some(v) = boolean(body, "add_generation_prompt")? {
        o.add_generation_prompt = v;
    }
    Ok(o)
}

/// `tools` as the template wants them (each must name a function), or none for `tool_choice: none`.
fn parse_tools(body: &Json) -> Result<Vec<Json>, String> {
    if body.field("functions").is_some() {
        return Err("'functions' is not supported (use tools)".into());
    }
    let tools = body.field("tools").and_then(Json::as_arr).unwrap_or(&[]);
    for (i, t) in tools.iter().enumerate() {
        if t.get("function")
            .and_then(|f| f.get("name"))
            .and_then(Json::as_str)
            .is_none()
        {
            return Err(format!(
                "tools[{i}]: expected {{\"type\": \"function\", \"function\": {{\"name\": …}}}}"
            ));
        }
    }
    match body.field("tool_choice") {
        None => Ok(tools.to_vec()),
        Some(Json::Str(s)) if s == "auto" => Ok(tools.to_vec()),
        Some(Json::Str(s)) if s == "none" => Ok(Vec::new()),
        Some(_) => Err("only tool_choice auto and none are supported".into()),
    }
}

pub fn parse_chat(m: &TextModel, body: &Json, seed: u64) -> Result<Prepared, ApiError> {
    if let Some(f) = body.field("response_format")
        && f.field("type").and_then(Json::as_str) != Some("text")
    {
        return Err("response_format other than text is not supported yet".into());
    }
    let raw = darask_json::parse(&body.to_string()).map_err(|e| e.to_string())?;
    let (raw, images) = darask_vision::chat(&raw, darask_vision::Kind::Qwen, 1024)?;
    let body = &Json::parse(&raw.dump())?;
    let tools = parse_tools(body)?;
    let msgs = parse_messages(body.get("messages"))?;
    let mut o = chat_options(body)?;
    if m.template == crate::model::ChatTemplate::Ornith && o.reasoning_effort.is_none() {
        // Ornith's Qwen3.5 template has no Qwen3.8 xhigh system instruction.
        o.reasoning_effort = Some("medium".into());
    }
    let text = chat::render(&msgs, &tools, &o)?;
    let prompt = m.tok.encode(&text);
    let mut p = common(m, body, true, prompt, seed)?;
    if !images.is_empty() {
        p.images = darask_vision::locate(&p.prompt, images, darask_vision::Kind::Qwen)?;
    }
    p.reasoning = o.thinking()
        && o.add_generation_prompt
        && boolean(body, "separate_reasoning")?.unwrap_or(true);
    p.tools = (!tools.is_empty()).then(|| Schema::from_tools(&tools));
    if let Some(ts) = m.turn_start {
        p.marks = p
            .prompt
            .iter()
            .rposition(|&t| t == ts)
            .into_iter()
            .filter(|&i| i > 0)
            .collect();
    }
    Ok(p)
}

pub fn parse_completion(m: &TextModel, body: &Json, seed: u64) -> Result<Prepared, ApiError> {
    if body
        .field("suffix")
        .and_then(Json::as_str)
        .is_some_and(|s| !s.is_empty())
    {
        return Err("'suffix' is not supported".into());
    }
    let ids = |a: &[Json]| -> Option<Vec<u32>> {
        a.iter().map(|v| v.as_u64().map(|x| x as u32)).collect()
    };
    let (prompt, text) = match body.field("prompt") {
        Some(Json::Str(s)) => (m.tok.encode(s), s.clone()),
        Some(Json::Arr(a)) if a.len() == 1 && a[0].as_str().is_some() => {
            let s = a[0].as_str().unwrap();
            (m.tok.encode(s), s.to_string())
        }
        Some(Json::Arr(a)) if a.len() == 1 && a[0].as_arr().is_some() => {
            let v = ids(a[0].as_arr().unwrap()).ok_or("'prompt' token ids must be integers")?;
            let t = m.tok.decode(&v, false);
            (v, t)
        }
        Some(Json::Arr(a)) if a.iter().all(|v| v.as_f64().is_some()) => {
            let v = ids(a).ok_or("'prompt' token ids must be integers")?;
            let t = m.tok.decode(&v, false);
            (v, t)
        }
        Some(Json::Arr(_)) => return Err("only one prompt per request is supported".into()),
        _ => return Err("'prompt' must be a string or a list of token ids".into()),
    };
    if let Some(&t) = prompt.iter().find(|&&t| t as usize >= m.tok.len()) {
        return Err(format!("token id {t} is out of range").into());
    }
    let mut p = common(m, body, false, prompt, seed)?;
    if boolean(body, "echo")?.unwrap_or(false) {
        p.echo = Some(text);
    }
    Ok(p)
}
