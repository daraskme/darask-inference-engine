//! OpenAI chat envelopes; decoding and sampling belong to the LLM runtime.
use crate::{
    json::{self, Value, quote},
    output::{Delta, Splitter},
    template,
};
use hayate_core::{Error, Result};
use hayate_llm::{SamplingParams, session::Timings};
use std::sync::atomic::{AtomicU64, Ordering};
static IDS: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct Chat {
    pub model: String,
    pub prompt: String,
    pub images: Vec<darask_vision::Image>,
    pub params: SamplingParams,
    pub seed: u64,
    /// No `max_tokens` in the request: the server picks it from the context.
    pub max_tokens_default: bool,
    /// GLM: reply starts inside `<think>` (thinking on).
    pub thinking: bool,
    /// GLM: close the thought after this many reasoning tokens.
    pub reasoning_budget: Option<usize>,
    /// GLM: HF/vLLM repetition penalty over prompt and reply (1 = off).
    pub repetition_penalty: f32,
    splitter: Splitter,
    id: String,
    created: u64,
    content: String,
    reasoning: String,
    calls: Vec<Value>,
}
impl Chat {
    pub fn parse(root: &Value) -> Result<Self> {
        let (prepared, images) = darask_vision::chat(root, darask_vision::Kind::Glm, 1024)
            .map_err(Error::InvalidPlan)?;
        let root = &prepared;
        let model = root.get("model")?.string()?;
        if model != "glm53" {
            return Err(Error::InvalidPlan("chat model: glm53".into()));
        }
        if root
            .optional("n")
            .map(Value::integer)
            .transpose()?
            .unwrap_or(1)
            != 1
        {
            return Err(Error::InvalidPlan("n must be 1".into()));
        }
        for key in [
            "response_format",
            "logprobs",
            "logit_bias",
            "presence_penalty",
            "frequency_penalty",
            "tool_choice",
        ] {
            let auto = key == "tool_choice"
                && matches!(root.optional(key), Some(Value::String(s)) if s == "auto");
            if !auto
                && root
                    .optional(key)
                    .is_some_and(|v| !matches!(v, Value::Null))
            {
                return Err(Error::InvalidPlan(format!("{key} is not supported")));
            }
        }
        let number = |key: &str, default: f32| -> Result<f32> {
            Ok(root
                .optional(key)
                .map(Value::number)
                .transpose()?
                .map_or(default, |v| v as f32))
        };
        let given = |key: &str| root.optional(key).filter(|v| !matches!(v, Value::Null));
        let max_tokens = given("max_tokens").or_else(|| given("max_completion_tokens"));
        // GLM card: temperature 1.0, top_p 0.95, no top-k.
        let params = SamplingParams {
            temperature: number("temperature", 1.)?,
            top_p: number("top_p", 0.95)?,
            top_k: given("top_k")
                .map(Value::integer)
                .transpose()?
                .unwrap_or(0)
                .try_into()
                .map_err(|_| Error::InvalidPlan("top_k".into()))?,
            min_p: number("min_p", 0.)?,
            max_tokens: max_tokens
                .map(Value::integer)
                .transpose()?
                .unwrap_or(4096)
                .try_into()
                .map_err(|_| Error::InvalidPlan("max_tokens".into()))?,
        };
        let repetition_penalty = number("repetition_penalty", 1.)?;
        if !params.temperature.is_finite()
            || params.temperature < 0.
            || !params.top_p.is_finite()
            || params.top_p <= 0.
            || params.top_p > 1.
            || !(0. ..1.).contains(&params.min_p)
            || !(repetition_penalty.is_finite() && repetition_penalty > 0.)
            || params.top_k > 154880
            || params.max_tokens == 0
        {
            return Err(Error::InvalidPlan("invalid sampling parameters".into()));
        }
        let reasoning_budget = given("reasoning_budget")
            .or_else(|| given("thinking_budget"))
            .map(Value::integer)
            .transpose()?;
        let stops = match root.optional("stop") {
            None | Some(Value::Null) => vec![],
            Some(Value::String(s)) => vec![s.clone()],
            Some(v) => v
                .array()?
                .iter()
                .map(|s| Ok(s.string()?.to_owned()))
                .collect::<Result<Vec<_>>>()?,
        };
        if stops.iter().any(String::is_empty) {
            return Err(Error::InvalidPlan("stop must not be empty".into()));
        }
        let tools = root
            .optional("tools")
            .filter(|v| !matches!(v, Value::Null))
            .map(Value::array)
            .transpose()?
            .unwrap_or(&[])
            .to_vec();
        let thinking = template::glm::thinking(root)?;
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| Error::InvalidPlan(e.to_string()))?
            .as_secs();
        let id = IDS.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            images,
            model: model.into(),
            prompt: template::render(root)?,
            params,
            seed: root
                .optional("seed")
                .map(Value::integer)
                .transpose()?
                .map_or(created ^ id, |n| n as u64),
            max_tokens_default: max_tokens.is_none(),
            thinking,
            reasoning_budget,
            repetition_penalty,
            splitter: Splitter::new(thinking, stops, tools),
            id: format!("chatcmpl-hayate-{created}-{id}"),
            created,
            content: String::new(),
            reasoning: String::new(),
            calls: vec![],
        })
    }
    fn envelope(&self, object: &str, choices: &str) -> String {
        format!(
            "{{\"id\":{},\"object\":{},\"created\":{},\"model\":{},\"choices\":{choices}",
            quote(&self.id),
            quote(object),
            self.created,
            quote(&self.model)
        )
    }
    pub fn start(&self) -> String {
        format!(
            "{}}}",
            self.envelope(
                "chat.completion.chunk",
                "[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]"
            )
        )
    }
    fn deltas(
        &mut self,
        deltas: Vec<Delta>,
        emit: &mut dyn FnMut(String) -> Result<()>,
    ) -> Result<()> {
        for delta in deltas {
            let delta = match delta {
                Delta::Content(s) => {
                    self.content.push_str(&s);
                    format!("{{\"content\":{}}}", quote(&s))
                }
                Delta::Reasoning(s) => {
                    self.reasoning.push_str(&s);
                    format!("{{\"reasoning_content\":{}}}", quote(&s))
                }
                Delta::Tool(call) => {
                    let index = self.calls.len();
                    let id = format!("call_{}_{}", self.id, index);
                    let function = call.get("function")?.dump();
                    let call = json::parse(&format!(
                        "{{\"id\":{},\"type\":\"function\",\"function\":{function}}}",
                        quote(&id)
                    ))?;
                    self.calls.push(call);
                    format!(
                        "{{\"tool_calls\":[{{\"index\":{index},\"id\":{},\"type\":\"function\",\"function\":{function}}}]}}",
                        quote(&id)
                    )
                }
            };
            emit(format!(
                "{}}}",
                self.envelope(
                    "chat.completion.chunk",
                    &format!("[{{\"index\":0,\"delta\":{delta},\"finish_reason\":null}}]")
                )
            ))?;
        }
        Ok(())
    }
    pub fn push(&mut self, text: &str, emit: &mut dyn FnMut(String) -> Result<()>) -> Result<bool> {
        let deltas = self.splitter.push(text)?;
        self.deltas(deltas, emit)?;
        Ok(self.splitter.stopped)
    }
    pub fn finish(
        &mut self,
        stats: &Timings,
        emit: &mut dyn FnMut(String) -> Result<()>,
    ) -> Result<String> {
        let deltas = if stats.length {
            self.splitter.finish_truncated()?
        } else {
            self.splitter.finish()?
        };
        self.deltas(deltas, emit)?;
        let finish = if self.splitter.stopped {
            "stop"
        } else if stats.length {
            "length"
        } else if !self.calls.is_empty() {
            "tool_calls"
        } else {
            "stop"
        };
        let usage = format!(
            "{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":{}}}}}",
            stats.prompt_tokens,
            stats.completion_tokens,
            stats.prompt_tokens + stats.completion_tokens,
            stats.cached_tokens
        );
        let accept = if stats.mtp_proposed > 0 {
            stats.mtp_accepted as f64 / stats.mtp_proposed as f64
        } else {
            0.
        };
        let timings = format!(
            "{{\"prefill_ms\":{},\"decode_ms\":{},\"decode_tok_s\":{},\"mtp_accept\":{},\"checkpoint_ms\":{},\"cache_host_bytes\":{},\"render_ms\":{},\"tokenize_ms\":{},\"sampling_ms\":{},\"restore_ms\":{},\"forward_ms\":{},\"prefill_checkpoint_ms\":{},\"decode_checkpoint_ms\":{},\"mtp_disabled\":{},\"checkpoint_wait_ms\":{},\"prefill_sync_ms\":{}}}",
            stats.prefill_ms,
            stats.decode_ms,
            stats.decode_tok_s(),
            accept,
            stats.checkpoint_ms,
            stats.cache_host_bytes,
            stats.render_ms,
            stats.tokenize_ms,
            stats.sampling_ms,
            stats.restore_ms,
            stats.forward_ms,
            stats.prefill_checkpoint_ms,
            stats.decode_checkpoint_ms,
            stats.mtp_disabled,
            stats.checkpoint_wait_ms,
            stats.prefill_sync_ms
        );
        emit(format!(
            "{},\"usage\":{usage},\"timings\":{timings}}}",
            self.envelope(
                "chat.completion.chunk",
                &format!(
                    "[{{\"index\":0,\"delta\":{{}},\"finish_reason\":{}}}]",
                    quote(finish)
                )
            )
        ))?;
        let message = format!(
            "{{\"role\":\"assistant\",\"content\":{},\"reasoning_content\":{}{} }}",
            quote(&self.content),
            quote(&self.reasoning),
            if self.calls.is_empty() {
                String::new()
            } else {
                format!(
                    ",\"tool_calls\":{}",
                    Value::Array(self.calls.clone()).dump()
                )
            }
        );
        Ok(format!(
            "{},\"usage\":{usage},\"timings\":{timings}}}",
            self.envelope(
                "chat.completion",
                &format!(
                    "[{{\"index\":0,\"message\":{message},\"finish_reason\":{}}}]",
                    quote(finish)
                )
            )
        ))
    }
}
#[cfg(test)]
mod glm_tests {
    use super::*;
    #[test]
    fn glm_stream_reasoning_parallel_calls_usage_and_defaults() {
        let request = json::parse(r#"{"model":"glm53","messages":[{"role":"user","content":"直して"}],"stream":true,"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],"tool_choice":"auto","chat_template_kwargs":{"enable_thinking":true},"reasoning_budget":64,"min_p":0.05}"#).unwrap();
        let mut chat = Chat::parse(&request).unwrap();
        assert!(chat.prompt.ends_with("<|assistant|><think>"));
        assert!(chat.thinking && chat.max_tokens_default);
        assert_eq!(chat.reasoning_budget, Some(64));
        assert_eq!((chat.params.temperature, chat.params.top_k), (1., 0));
        assert_eq!(chat.params.min_p, 0.05);
        let mut events = vec![chat.start()];
        for piece in [
            "見る",
            "</think>",
            "<tool_call>read<arg_key>path</arg_key><arg_value>a.rs</arg_value></tool_call>",
            "<tool_call>read<arg_key>path</arg_key><arg_value>b.rs</arg_value></tool_call>",
        ] {
            chat.push(piece, &mut |s| {
                events.push(s);
                Ok(())
            })
            .unwrap();
        }
        let body = chat
            .finish(
                &Timings {
                    prompt_tokens: 30,
                    completion_tokens: 9,
                    cached_tokens: 20,
                    ..Timings::default()
                },
                &mut |s| {
                    events.push(s);
                    Ok(())
                },
            )
            .unwrap();
        let events = events
            .iter()
            .map(|e| json::parse(e).unwrap())
            .collect::<Vec<_>>();
        let delta = |i: usize| {
            events[i].get("choices").unwrap().array().unwrap()[0]
                .get("delta")
                .unwrap()
                .clone()
        };
        assert_eq!(
            delta(1).get("reasoning_content").unwrap().string().unwrap(),
            "見る"
        );
        let ids = (2..4)
            .map(|i| {
                let d = delta(i);
                let call = &d.get("tool_calls").unwrap().array().unwrap()[0];
                assert_eq!(call.get("index").unwrap().integer().unwrap(), i - 2);
                call.get("id").unwrap().string().unwrap().to_owned()
            })
            .collect::<Vec<_>>();
        assert_ne!(ids[0], ids[1]);
        let last = events.last().unwrap();
        assert_eq!(
            last.get("choices").unwrap().array().unwrap()[0]
                .get("finish_reason")
                .unwrap()
                .string()
                .unwrap(),
            "tool_calls"
        );
        let usage = last.get("usage").unwrap();
        assert_eq!(usage.get("prompt_tokens").unwrap().integer().unwrap(), 30);
        assert_eq!(
            usage.get("completion_tokens").unwrap().integer().unwrap(),
            9
        );
        assert_eq!(
            usage
                .get("prompt_tokens_details")
                .unwrap()
                .get("cached_tokens")
                .unwrap()
                .integer()
                .unwrap(),
            20
        );
        let body = json::parse(&body).unwrap();
        let message = body.get("choices").unwrap().array().unwrap()[0]
            .get("message")
            .unwrap()
            .clone();
        assert_eq!(message.get("tool_calls").unwrap().array().unwrap().len(), 2);
        assert_eq!(message.get("content").unwrap().string().unwrap(), "");
        // Thinking off: top-level enable_thinking and reasoning_effort "none".
        for off in [r#""enable_thinking":false"#, r#""reasoning_effort":"none""#] {
            let chat = Chat::parse(
                &json::parse(&format!(
                    r#"{{"model":"glm53","messages":[{{"role":"user","content":"x"}}],{off}}}"#
                ))
                .unwrap(),
            )
            .unwrap();
            assert!(!chat.thinking && chat.prompt.ends_with("<think></think>"));
        }
        for bad in [
            r#"{"model":"glm53","messages":[{"role":"user","content":"x"}],"tool_choice":"required"}"#,
            r#"{"model":"glm53","messages":[{"role":"user","content":"x"}],"min_p":1.5}"#,
            r#"{"model":"gemma4","messages":[{"role":"user","content":"x"}],"reasoning_budget":5}"#,
        ] {
            assert!(Chat::parse(&json::parse(bad).unwrap()).is_err(), "{bad}");
        }
    }
}
