//! GGUF BPE metadata uses the same byte-level tokenizer implementation as Tsumuji.
use darask_formats::gguf::{Gguf, Value};
use darask_text::{json::Json, tokenizer::Tokenizer};

pub fn load(g: &Gguf) -> Result<Tokenizer, String> {
    let arr = |k: &str| match g.meta.get(k) {
        Some(Value::Array(a)) => Ok(a),
        _ => Err(format!("missing {k}")),
    };
    let tokens = arr("tokenizer.ggml.tokens")?;
    let types = arr("tokenizer.ggml.token_type")?;
    if tokens.len() != types.len() {
        return Err("tokenizer metadata length mismatch".into());
    }
    let mut vocab = Vec::new();
    let mut added = Vec::new();
    for (id, (token, ty)) in tokens.iter().zip(types).enumerate() {
        let Value::Str(s) = token else {
            return Err("non-string tokenizer token".into());
        };
        if matches!(ty, Value::Int(3 | 4 | 5)) {
            added.push(Json::obj([
                ("id", Json::Int(id as i64)),
                ("content", Json::str(s)),
                ("special", Json::Bool(true)),
            ]));
        } else {
            vocab.push((s.clone(), Json::Int(id as i64)));
        }
    }
    let merges = arr("tokenizer.ggml.merges")?
        .iter()
        .map(|v| match v {
            Value::Str(s) => Ok(Json::str(s)),
            _ => Err("bad BPE merge".to_string()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    // GGUF tokenizer pre=qwen2 is the Qwen2 regexp, including one numeral per piece.
    let regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let json = Json::obj([
        (
            "model",
            Json::obj([
                ("type", Json::str("BPE")),
                ("vocab", Json::Obj(vocab)),
                ("merges", Json::Arr(merges)),
            ]),
        ),
        ("normalizer", Json::obj([("type", Json::str("NFC"))])),
        ("pre_tokenizer", Json::str(regex)),
        ("added_tokens", Json::Arr(added)),
    ]);
    Tokenizer::from_json(&json.to_string())
}
