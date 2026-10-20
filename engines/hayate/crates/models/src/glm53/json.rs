//! The tokenizer's JSON subset, without an external dependency.
use hayate_core::{Error, Result};
use std::collections::HashMap;

#[derive(Debug)]
pub(super) enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}
impl Value {
    pub(super) fn get(&self, key: &str) -> Result<&Self> {
        match self {
            Self::Object(o) => o.get(key).ok_or_else(bad),
            _ => None.ok_or_else(bad),
        }
    }
    pub(super) fn string(&self) -> Result<&str> {
        match self {
            Self::String(s) => Ok(s),
            _ => Err(bad()),
        }
    }
    pub(super) fn array(&self) -> Result<&[Self]> {
        match self {
            Self::Array(a) => Ok(a),
            _ => Err(bad()),
        }
    }
    pub(super) fn number(&self) -> Result<u32> {
        match self {
            Self::Number(n) if *n >= 0. && *n <= u32::MAX as f64 && n.fract() == 0. => {
                Ok(*n as u32)
            }
            _ => Err(bad()),
        }
    }
    pub(super) fn boolean(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            _ => Err(bad()),
        }
    }
}
fn bad() -> Error {
    Error::Weights("invalid tokenizer JSON".into())
}

pub(super) fn parse(text: &str) -> Result<Value> {
    darask_json::parse(text).map(convert).map_err(|_| bad())
}
fn convert(value: darask_json::Value) -> Value {
    use darask_json::Value as Shared;
    match value {
        Shared::Null => Value::Null,
        Shared::Bool(v) => Value::Bool(v),
        Shared::Number(v) => Value::Number(v.parse().expect("validated JSON number")),
        Shared::String(v) => Value::String(v),
        Shared::Array(v) => Value::Array(v.into_iter().map(convert).collect()),
        Shared::Object(v) => Value::Object(v.into_iter().map(|(k, v)| (k, convert(v))).collect()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn escaped_unicode_and_malformed_json() {
        assert_eq!(
            parse(r#""\uD83D\uDC08\n\u65e5""#)
                .unwrap()
                .string()
                .unwrap(),
            "🐈\n日"
        );
        assert_eq!(
            parse(r#"{"temperature":1.0,"top_p":0.95,"eos_token_id":[154820]}"#)
                .unwrap()
                .get("eos_token_id")
                .unwrap()
                .array()
                .unwrap()[0]
                .number()
                .unwrap(),
            154820
        );
        for s in [
            "1.",
            "1e",
            "-01",
            "1e999",
            "{\"a\":1,\"a\":2}",
            "[1,]",
            "01",
            "\"\\uD800\"",
            "true false",
        ] {
            assert!(parse(s).is_err(), "{s}");
        }
    }
}
