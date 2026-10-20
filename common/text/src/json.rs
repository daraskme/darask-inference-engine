//! JSON values: a reader for requests and tokenizer.json, a compact writer for responses, and
//! Python's `json.dumps` format for the chat template's `tojson` (no external crates). Objects keep
//! their key order; integers stay integers (Python tells 1 from 1.0).

use std::fmt::{self, Write as _};

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// An integer literal that fits i64.
    Int(i64),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json, String> {
        darask_json::parse(text)
            .map(convert)
            .map_err(|e| e.to_string())
    }

    /// An object from (key, value) pairs.
    pub fn obj<const N: usize>(pairs: [(&str, Json); N]) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn str(s: impl Into<String>) -> Json {
        Json::Str(s.into())
    }

    pub fn num(v: impl Into<f64>) -> Json {
        Json::Num(v.into())
    }

    /// A string, or null when `s` is None.
    pub fn opt_str(s: Option<impl Into<String>>) -> Json {
        s.map_or(Json::Null, |s| Json::Str(s.into()))
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The value at `key` unless it is missing or null.
    pub fn field(&self, key: &str) -> Option<&Json> {
        self.get(key).filter(|v| !matches!(v, Json::Null))
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            Json::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    /// A non-negative integer.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Int(v) => u64::try_from(*v).ok(),
            _ => self
                .as_f64()
                .filter(|v| *v >= 0.0 && v.fract() == 0.0 && *v < 1.8e19)
                .map(|v| v as u64),
        }
    }

    /// An integer (possibly negative).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Int(v) => Some(*v),
            _ => self
                .as_f64()
                .filter(|v| v.fract() == 0.0 && v.abs() < 9.2e18)
                .map(|v| v as i64),
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }

    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        if let Json::Obj(m) = self {
            Some(m)
        } else {
            None
        }
    }

    pub fn push(&mut self, key: &str, v: Json) {
        if let Json::Obj(m) = self {
            m.push((key.to_string(), v));
        }
    }
}

/// Appends `s` as a quoted JSON string.
pub fn write_str(out: &mut String, s: &str) {
    out.push('"');
    let mut start = 0;
    for (i, b) in s.bytes().enumerate() {
        let esc = match b {
            b'"' => "\\\"",
            b'\\' => "\\\\",
            b'\n' => "\\n",
            b'\r' => "\\r",
            b'\t' => "\\t",
            0..0x20 | 0x7f => "",
            _ => continue,
        };
        out.push_str(&s[start..i]);
        if esc.is_empty() {
            let _ = write!(out, "\\u{b:04x}");
        } else {
            out.push_str(esc);
        }
        start = i + 1;
    }
    out.push_str(&s[start..]);
    out.push('"');
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        write_json(&mut s, self);
        f.write_str(&s)
    }
}

fn write_json(out: &mut String, v: &Json) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(n) => {
            let _ = write!(out, "{n}");
        }
        Json::Num(n) => {
            if !n.is_finite() {
                out.push_str("null");
            } else if n.fract() == 0.0 && n.abs() < 9.0e15 {
                let _ = write!(out, "{}", *n as i64);
            } else {
                let _ = write!(out, "{n}");
            }
        }
        Json::Str(s) => write_str(out, s),
        Json::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(out, x);
            }
            out.push(']');
        }
        Json::Obj(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_str(out, k);
                out.push(':');
                write_json(out, x);
            }
            out.push('}');
        }
    }
}

/// Python's `json.dumps(v, ensure_ascii=False)`: ", " and ": " separators, floats as `repr`.
pub fn to_python(v: &Json) -> String {
    let mut out = String::new();
    write_py(&mut out, v);
    out
}

fn write_py(out: &mut String, v: &Json) {
    match v {
        Json::Num(n) => out.push_str(&py_float(*n)),
        Json::Str(s) => write_py_str(out, s),
        Json::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py(out, x);
            }
            out.push(']');
        }
        Json::Obj(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_str(out, k);
                out.push_str(": ");
                write_py(out, x);
            }
            out.push('}');
        }
        _ => write_json(out, v),
    }
}

/// json.dumps' string escaping with ensure_ascii=False (DEL stays as it is).
fn write_py_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest digits that round-trip, positional for 1e-4 <= |x| < 1e16
/// (with ".0" when integral), otherwise d.ddde+XX.
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let e = format!("{x:e}"); // e.g. "-1.2345e-5", shortest round-trip digits
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (sign, mant) = mant.strip_prefix('-').map_or(("", mant), |m| ("-", m));
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        let s = if point <= 0 {
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
        format!("{sign}{s}")
    } else {
        let m = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        format!(
            "{sign}{m}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    }
}

fn convert(value: darask_json::Value) -> Json {
    use darask_json::Value as Shared;
    match value {
        Shared::Null => Json::Null,
        Shared::Bool(v) => Json::Bool(v),
        Shared::Number(v) => {
            if !v.contains(['.', 'e', 'E']) {
                v.parse::<i64>()
                    .map(Json::Int)
                    .unwrap_or_else(|_| Json::Num(v.parse().unwrap()))
            } else {
                Json::Num(v.parse().unwrap())
            }
        }
        Shared::String(v) => Json::Str(v),
        Shared::Array(v) => Json::Arr(v.into_iter().map(convert).collect()),
        Shared::Object(v) => Json::Obj(v.into_iter().map(|(k, v)| (k, convert(v))).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let t = r#"{"a":[1,2.5,-3e2,true,null],"b":"x\"\né😀","c":{}}"#;
        let v = Json::parse(t).unwrap();
        assert_eq!(v.get("b").unwrap().as_str(), Some("x\"\né😀"));
        assert_eq!(
            v.to_string(),
            r#"{"a":[1,2.5,-300,true,null],"b":"x\"\né😀","c":{}}"#
        );
        assert_eq!(Json::str("\u{1}\u{7f}").to_string(), r#""\u0001\u007f""#);
    }

    #[test]
    fn python_dumps() {
        let v = Json::parse(r#"{"a": [1, 1.0, 1e16, 1.5e-5, 0.0001, 123.456, -2.5e20], "b": "é\n\"\u0001", "c": {"d": null, "e": true}}"#).unwrap();
        assert_eq!(
            to_python(&v),
            r#"{"a": [1, 1.0, 1e+16, 1.5e-05, 0.0001, 123.456, -2.5e+20], "b": "é\n\"\u0001", "c": {"d": null, "e": true}}"#
        );
        assert_eq!(py_float(1e15), "1000000000000000.0");
        assert_eq!(py_float(0.1 + 0.2), "0.30000000000000004");
    }
}
