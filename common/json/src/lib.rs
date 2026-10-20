//! Request JSON and ordered objects, without an external dependency.
use hayate_core::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}
impl Value {
    pub fn get(&self, key: &str) -> Result<&Self> {
        match self {
            Self::Object(o) => o
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .ok_or_else(bad),
            _ => None.ok_or_else(bad),
        }
    }
    pub fn string(&self) -> Result<&str> {
        match self {
            Self::String(s) => Ok(s),
            _ => Err(bad()),
        }
    }
    pub fn array(&self) -> Result<&[Self]> {
        match self {
            Self::Array(a) => Ok(a),
            _ => Err(bad()),
        }
    }
    pub fn number(&self) -> Result<f64> {
        match self {
            Self::Number(n) => n.parse().map_err(|_| bad()),
            _ => Err(bad()),
        }
    }
    pub fn boolean(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            _ => Err(bad()),
        }
    }
    pub fn optional(&self, key: &str) -> Option<&Self> {
        self.get(key).ok()
    }
    pub fn object(&self) -> Result<&[(String, Value)]> {
        match self {
            Self::Object(o) => Ok(o),
            _ => Err(bad()),
        }
    }
    pub fn integer(&self) -> Result<usize> {
        match self {
            Self::Number(n) => n.parse().map_err(|_| bad()),
            _ => Err(bad()),
        }
    }
    /// HF tojson uses the Python separators, Unicode, and insertion order.
    pub fn dump(&self) -> String {
        match self {
            Self::Null => "null".into(),
            Self::Bool(b) => b.to_string(),
            Self::Number(n) => n.clone(),
            Self::String(s) => quote(s),
            Self::Array(a) => format!(
                "[{}]",
                a.iter().map(Self::dump).collect::<Vec<_>>().join(", ")
            ),
            Self::Object(o) => format!(
                "{{{}}}",
                o.iter()
                    .map(|(k, v)| format!("{}: {}", quote(k), v.dump()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}
pub fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn bad() -> Error {
    Error::InvalidPlan("invalid request JSON".into())
}

pub fn parse(text: &str) -> Result<Value> {
    let mut p = Parser { rest: text };
    let value = p.value(0)?;
    if !p.rest.trim().is_empty() {
        return Err(bad());
    }
    Ok(value)
}
struct Parser<'a> {
    rest: &'a str,
}
impl Parser<'_> {
    fn take(&mut self, s: &str) -> bool {
        self.rest = self.rest.trim_start_matches([' ', '\t', '\n', '\r']);
        if let Some(rest) = self.rest.strip_prefix(s) {
            self.rest = rest;
            true
        } else {
            false
        }
    }
    fn hex(&mut self) -> Result<u32> {
        let word = self.rest.get(..4).ok_or_else(bad)?;
        let n = u32::from_str_radix(word, 16).map_err(|_| bad())?;
        self.rest = &self.rest[4..];
        Ok(n)
    }
    fn string(&mut self) -> Result<String> {
        if !self.take("\"") {
            return Err(bad());
        }
        let mut out = String::new();
        loop {
            let c = self.rest.chars().next().ok_or_else(bad)?;
            self.rest = &self.rest[c.len_utf8()..];
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let c = self.rest.chars().next().ok_or_else(bad)?;
                    self.rest = &self.rest[c.len_utf8()..];
                    out.push(match c {
                        '"' | '\\' | '/' => c,
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        'b' => '\u{8}',
                        'f' => '\u{c}',
                        'u' => {
                            let mut n = self.hex()?;
                            if (0xd800..=0xdbff).contains(&n) {
                                self.rest = self.rest.strip_prefix("\\u").ok_or_else(bad)?;
                                let low = self.hex()?;
                                if !(0xdc00..=0xdfff).contains(&low) {
                                    return Err(bad());
                                }
                                n = 0x10000 + ((n - 0xd800) << 10) + low - 0xdc00;
                            }
                            char::from_u32(n).ok_or_else(bad)?
                        }
                        _ => return Err(bad()),
                    });
                }
                c if c < ' ' => return Err(bad()),
                c => out.push(c),
            }
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > 32 {
            return Err(bad());
        }
        if self.take("null") {
            return Ok(Value::Null);
        }
        if self.take("true") {
            return Ok(Value::Bool(true));
        }
        if self.take("false") {
            return Ok(Value::Bool(false));
        }
        if self.rest.starts_with('"') {
            return Ok(Value::String(self.string()?));
        }
        if self.take("[") {
            let mut a = Vec::new();
            if !self.take("]") {
                loop {
                    a.push(self.value(depth + 1)?);
                    if self.take("]") {
                        break;
                    }
                    if !self.take(",") {
                        return Err(bad());
                    }
                }
            }
            return Ok(Value::Array(a));
        }
        if self.take("{") {
            let mut o = Vec::new();
            let mut keys = std::collections::HashSet::new();
            if !self.take("}") {
                loop {
                    let key = self.string()?;
                    if !self.take(":") {
                        return Err(bad());
                    }
                    if !keys.insert(key.clone()) {
                        return Err(bad());
                    }
                    o.push((key, self.value(depth + 1)?));
                    if self.take("}") {
                        break;
                    }
                    if !self.take(",") {
                        return Err(bad());
                    }
                }
            }
            return Ok(Value::Object(o));
        }
        let start = self.rest;
        if self.rest.starts_with('-') {
            self.rest = &self.rest[1..];
        }
        let n = self.rest.bytes().take_while(u8::is_ascii_digit).count();
        if n == 0 || (n > 1 && self.rest.starts_with('0')) {
            return Err(bad());
        }
        self.rest = &self.rest[n..];
        if self.rest.starts_with('.') {
            self.rest = &self.rest[1..];
            let n = self.rest.bytes().take_while(u8::is_ascii_digit).count();
            if n == 0 {
                return Err(bad());
            }
            self.rest = &self.rest[n..];
        }
        if self.rest.starts_with(['e', 'E']) {
            self.rest = &self.rest[1..];
            if self.rest.starts_with(['+', '-']) {
                self.rest = &self.rest[1..];
            }
            let n = self.rest.bytes().take_while(u8::is_ascii_digit).count();
            if n == 0 {
                return Err(bad());
            }
            self.rest = &self.rest[n..];
        }
        let number = &start[..start.len() - self.rest.len()];
        if !number.parse::<f64>().map_err(|_| bad())?.is_finite() {
            return Err(bad());
        }
        Ok(Value::Number(number.into()))
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
        for s in [
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

#[cfg(test)]
mod number_tests {
    use super::*;
    #[test]
    fn numbers_order_and_escaping_roundtrip() {
        let text = r#"{"z": -2.3e-4, "a": [true, null, "日\n\"\\"]}"#;
        let value = parse(text).unwrap();
        assert_eq!(parse(&value.dump()).unwrap(), value);
        assert_eq!(value.dump(), text);
        for invalid in [
            "1e",
            "1.",
            "-.1",
            "-01",
            "1e999",
            "[NaN]",
            "{\"a\":1,\"a\":2}",
        ] {
            assert!(parse(invalid).is_err(), "{invalid}");
        }
    }
}
