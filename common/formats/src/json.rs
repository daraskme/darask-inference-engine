//! Minimal JSON reader for safetensors headers and config files (no external crates).

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json, String> {
        darask_json::parse(text)
            .map(convert)
            .map_err(|e| e.to_string())
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        if let Json::Num(n) = self {
            (n.fract() == 0.0 && *n >= 0.0).then_some(*n as u64)
        } else {
            None
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }

    pub fn as_obj(&self) -> Option<&BTreeMap<String, Json>> {
        if let Json::Obj(m) = self {
            Some(m)
        } else {
            None
        }
    }
}

fn convert(value: darask_json::Value) -> Json {
    use darask_json::Value as Shared;
    match value {
        Shared::Null => Json::Null,
        Shared::Bool(v) => Json::Bool(v),
        Shared::Number(v) => Json::Num(v.parse().expect("validated JSON number")),
        Shared::String(v) => Json::Str(v),
        Shared::Array(v) => Json::Arr(v.into_iter().map(convert).collect()),
        Shared::Object(v) => Json::Obj(v.into_iter().map(|(k, v)| (k, convert(v))).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_safetensors_header() {
        let h = r#"{"__metadata__":{"format":"pt"},"a.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"s":{"dtype":"F32","shape":[],"data_offsets":[12,16]},"u":"é😀"}"#;
        let j = Json::parse(h).unwrap();
        let a = j.get("a.weight").unwrap();
        assert_eq!(a.get("dtype").unwrap().as_str(), Some("BF16"));
        assert_eq!(a.get("shape").unwrap().as_arr().unwrap().len(), 2);
        assert_eq!(
            j.get("s")
                .unwrap()
                .get("shape")
                .unwrap()
                .as_arr()
                .unwrap()
                .len(),
            0
        );
        assert_eq!(j.get("u").unwrap().as_str(), Some("é😀"));
    }
}
