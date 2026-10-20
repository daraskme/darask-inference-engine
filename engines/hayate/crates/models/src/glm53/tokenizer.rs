//! GLM checkpoint byte BPE; no normalization, three-digit number groups.
use super::json::{self, Value};
use hayate_core::{Error, Result};
use std::{collections::HashMap, path::Path};

const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
#[derive(Debug)]
pub struct Tokenizer {
    vocab: HashMap<Vec<u8>, u32>,
    merges: HashMap<(u32, u32), (usize, u32)>,
    added: Vec<(String, u32)>,
    decoded: HashMap<u32, Vec<u8>>,
}
fn bad(msg: &str) -> Error {
    Error::Weights(format!("GLM tokenizer: {msg}"))
}
fn byte_chars() -> [char; 256] {
    let mut extra = 0;
    std::array::from_fn(|b| {
        let code =
            if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
                b
            } else {
                let n = 256 + extra;
                extra += 1;
                n
            };
        char::from_u32(code as u32).unwrap()
    })
}
impl Tokenizer {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let tokenizer = Self::from_json(&std::fs::read_to_string(path)?)?;
        Ok(tokenizer)
    }
    fn from_json(text: &str) -> Result<Self> {
        let root = json::parse(text)?;
        if !matches!(root.get("normalizer")?, Value::Null) {
            return Err(bad("normalizer must be null"));
        }
        if !root.get("model")?.get("ignore_merges")?.boolean()? {
            return Err(bad("expected ignore_merges"));
        }
        let pre = root.get("pre_tokenizer")?;
        if pre.get("type")?.string()? != "Sequence" {
            return Err(bad("pretokenizer must be Sequence"));
        }
        let steps = pre.get("pretokenizers")?.array()?;
        if steps.len() != 2
            || steps[0].get("type")?.string()? != "Split"
            || steps[0]
                .get("pattern")?
                .get("Regex")?
                .string()?
                .replace("\\\\", "\\")
                != PATTERN
            || steps[0].get("behavior")?.string()? != "Isolated"
            || steps[0].get("invert")?.boolean()?
            || steps[1].get("type")?.string()? != "ByteLevel"
            || steps[1].get("add_prefix_space")?.boolean()?
            || steps[1].get("use_regex")?.boolean()?
        {
            return Err(bad("unsupported pretokenizer"));
        }
        let model = root.get("model")?;
        if model.get("type")?.string()? != "BPE" {
            return Err(bad("model must be BPE"));
        }
        for key in ["dropout", "unk_token"] {
            if !matches!(model.get(key)?, Value::Null) {
                return Err(bad("BPE dropout/unk unsupported"));
            }
        }
        if model.get("byte_fallback")?.boolean()? || model.get("fuse_unk")?.boolean()? {
            return Err(bad("BPE fallback unsupported"));
        }
        let inverse: HashMap<_, _> = byte_chars()
            .into_iter()
            .enumerate()
            .map(|(b, c)| (c, b as u8))
            .collect();
        let decode = |s: &str| {
            s.chars()
                .map(|c| {
                    inverse
                        .get(&c)
                        .copied()
                        .ok_or_else(|| bad("non-byte vocabulary entry"))
                })
                .collect::<Result<Vec<_>>>()
        };
        let Value::Object(object) = model.get("vocab")? else {
            return Err(bad("vocab must be object"));
        };
        let mut vocab = HashMap::new();
        let mut encoded_ids = HashMap::new();
        for (s, id) in object {
            let id = id.number()?;
            encoded_ids.insert(s.clone(), id);
            vocab.insert(decode(s)?, id);
        }
        if (0..256u16).any(|b| !vocab.contains_key(&vec![b as u8])) {
            return Err(bad("missing byte token"));
        }
        let mut merges = HashMap::new();
        for (rank, pair) in model.get("merges")?.array()?.iter().enumerate() {
            let (a, b) = match pair {
                Value::String(s) => s.split_once(' ').ok_or_else(|| bad("bad merge pair"))?,
                Value::Array(a) if a.len() == 2 => (a[0].string()?, a[1].string()?),
                _ => return Err(bad("bad merge pair")),
            };
            let lookup = |s: &str| {
                encoded_ids
                    .get(s)
                    .copied()
                    .ok_or_else(|| bad("missing merge token"))
            };
            if merges
                .insert(
                    (lookup(a)?, lookup(b)?),
                    (rank, lookup(&format!("{a}{b}"))?),
                )
                .is_some()
            {
                return Err(bad("duplicate merge"));
            }
        }
        let mut added: Vec<(String, u32)> = Vec::new();
        for token in root.get("added_tokens")?.array()? {
            if token.get("single_word")?.boolean()?
                || token.get("lstrip")?.boolean()?
                || token.get("rstrip")?.boolean()?
                || token.get("normalized")?.boolean()?
            {
                return Err(bad("unsupported added token flags"));
            }
            let content = token.get("content")?.string()?;
            if content.is_empty() {
                return Err(bad("empty added token"));
            }
            added.push((content.into(), token.get("id")?.number()?));
        }
        added.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let decoded = vocab
            .iter()
            .map(|(bytes, id)| (*id, bytes.clone()))
            .collect();
        Ok(Self {
            decoded,
            vocab,
            merges,
            added,
        })
    }
    /// Plain text conversation, with thinking disabled as in the supplied template.
    pub fn chat(&self, prompt: &str, template: &Path) -> Result<Vec<u32>> {
        let hash = std::fs::read(template)?
            .iter()
            .fold(0xcbf29ce484222325u64, |h, b| {
                (h ^ *b as u64).wrapping_mul(0x100000001b3)
            });
        if hash != 0x4c5235279d861d7d {
            return Err(bad(
                "unsupported chat_template.jinja; expected checkpoint template",
            ));
        }
        self.encode(&format!(
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{prompt}<|assistant|><think>"
        ))
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for id in ids {
            if self.added.iter().any(|(_, special)| special == id) {
                continue;
            }
            bytes.extend(
                self.decoded
                    .get(id)
                    .ok_or_else(|| bad("unknown token ID"))?,
            );
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = prompt;
        while !rest.is_empty() {
            let next = self
                .added
                .iter()
                .filter_map(|(s, id)| rest.find(s).map(|at| (at, s, *id)))
                .min_by_key(|(at, s, _)| (*at, std::cmp::Reverse(s.len())));
            let end = next.as_ref().map_or(rest.len(), |(at, _, _)| *at);
            let mut plain = &rest[..end];
            while !plain.is_empty() {
                let size = split_len(plain);
                let piece = &plain.as_bytes()[..size];
                if let Some(&id) = self.vocab.get(piece) {
                    out.push(id);
                    plain = &plain[size..];
                    continue;
                }
                let mut ids = piece
                    .iter()
                    .map(|b| self.vocab[&vec![*b]])
                    .collect::<Vec<_>>();
                loop {
                    let best = ids
                        .windows(2)
                        .enumerate()
                        .filter_map(|(i, p)| {
                            self.merges.get(&(p[0], p[1])).map(|&(r, id)| (r, i, id))
                        })
                        .min();
                    let Some((_, i, id)) = best else {
                        break;
                    };
                    ids[i] = id;
                    ids.remove(i + 1);
                }
                out.extend(ids);
                plain = &plain[size..];
            }
            if let Some((at, s, id)) = next {
                out.push(id);
                rest = &rest[at + s.len()..];
            } else {
                break;
            }
        }
        if out.is_empty() {
            return Err(Error::InvalidShape(
                "prompt must contain at least one token".into(),
            ));
        }
        Ok(out)
    }
}
fn letter(c: char) -> bool {
    props(c) & 1 != 0
}
fn number(c: char) -> bool {
    props(c) & 2 != 0
}
fn space(c: char) -> bool {
    props(c) & 4 != 0
}
fn newline(c: char) -> bool {
    matches!(c, '\r' | '\n')
}
// The checkpoint uses the serialized GPT-4 style split pattern.
fn split_len(s: &str) -> usize {
    let chars: Vec<_> = s.char_indices().collect();
    let c = chars[0].1;
    if c == '\'' {
        for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
            let mut tail = s[1..].chars();
            let mut bytes = 1;
            let matched = suffix.chars().all(|expected| {
                if let Some(c) = tail.next() {
                    bytes += c.len_utf8();
                    c.to_ascii_lowercase() == expected || (expected == 's' && c == 'ſ')
                } else {
                    false
                }
            });
            if matched {
                return bytes;
            }
        }
    }
    let start = usize::from(!letter(c) && !number(c) && !newline(c));
    if chars.get(start).is_some_and(|(_, c)| letter(*c)) {
        let count = start
            + chars[start..]
                .iter()
                .take_while(|(_, c)| letter(*c))
                .count();
        return chars.get(count).map_or(s.len(), |(at, _)| *at);
    }
    if number(c) {
        let count = chars.iter().take(3).take_while(|(_, c)| number(*c)).count();
        return chars.get(count).map_or(s.len(), |(at, _)| *at);
    }
    let start = usize::from(c == ' ');
    if chars
        .get(start)
        .is_some_and(|(_, c)| !space(*c) && !letter(*c) && !number(*c))
    {
        let mut count = start
            + chars[start..]
                .iter()
                .take_while(|(_, c)| !space(*c) && !letter(*c) && !number(*c))
                .count();
        count += chars[count..]
            .iter()
            .take_while(|(_, c)| newline(*c))
            .count();
        return chars.get(count).map_or(s.len(), |(at, _)| *at);
    }
    let count = chars.iter().take_while(|(_, c)| space(*c)).count();
    if let Some(last) = chars[..count].iter().rposition(|(_, c)| newline(*c)) {
        return chars.get(last + 1).map_or(s.len(), |(at, _)| *at);
    }
    // Greedy whitespace with (?!\S) backtracks one scalar before nonspace.
    let count = if count < chars.len() && count > 1 {
        count - 1
    } else {
        count
    };
    chars.get(count).map_or(s.len(), |(at, _)| *at)
}

const DATA: &[u8] = include_bytes!("unicode.bin");
fn word(offset: usize) -> u32 {
    u32::from_le_bytes(DATA[offset..offset + 4].try_into().unwrap())
}
fn props(c: char) -> u32 {
    let n = word(0) as usize;
    let code = c as u32;
    let mut lo = 0;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if word(12 + mid * 12 + 4) < code {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo < n && word(12 + lo * 12) <= code {
        word(12 + lo * 12 + 8)
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "local checkpoint tokenizer; CPU only"]
    fn glm53_eval_corpus_tokenizer() -> Result<()> {
        let source = std::env::var("GLM53_SOURCE").unwrap_or_else(|_| {
            "/mnt/NVME4TB/darask-inference-engine/local/hayate/src/GLM-5.3-Flash-Uncensored-FP8"
                .into()
        });
        let tokenizer = Tokenizer::open(Path::new(&source).join("tokenizer.json"))?;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python-tools/hayate_tools");
        let ids = tokenizer.encode(&std::fs::read_to_string(
            root.join("glm53_eval_corpus.txt"),
        )?)?;
        assert!(ids.len() >= 512, "corpus has {} tokens", ids.len());
        if let Ok(out) = std::env::var("GLM53_EVAL_IDS_OUT") {
            std::fs::write(out, format!("{ids:?}\n"))?;
        } else {
            let frozen = json::parse(&std::fs::read_to_string(
                root.join("glm53_eval_corpus.ids.json"),
            )?)?;
            let expected = frozen
                .get("ids")?
                .array()?
                .iter()
                .map(Value::number)
                .collect::<Result<Vec<_>>>()?;
            assert_eq!(ids, expected);
        }
        eprintln!(
            "GLM fixed corpus: {} exact Rust BPE tokens (use first 512)",
            ids.len()
        );
        Ok(())
    }
    #[test]
    fn glm_split_digits_and_combining_marks() {
        let mut rest = "日本語 e\u{301} 1234567!!\n";
        let mut parts = Vec::new();
        while !rest.is_empty() {
            let n = split_len(rest);
            parts.push(&rest[..n]);
            rest = &rest[n..];
        }
        assert_eq!(
            parts,
            ["日本語", " e", "\u{301}", " ", "123", "456", "7", "!!\n"]
        );
    }
    #[test]
    #[ignore = "local checkpoint tokenizer and glm53_tokenizer_corpus; CPU only"]
    fn glm53_tokenizer_corpus() -> Result<()> {
        let root = std::env::var("GLM53_SOURCE").unwrap_or_else(|_| {
            "/mnt/NVME4TB/darask-inference-engine/local/hayate/src/GLM-5.3-Flash-Uncensored-FP8"
                .into()
        });
        let root = Path::new(&root);
        let tokenizer = Tokenizer::open(root.join("tokenizer.json"))?;
        let corpus = std::env::var("GLM53_TOKENIZER_CORPUS")
            .unwrap_or_else(|_| "/tmp/glm53-tokenizer.jsonl".into());
        let mut count = 0;
        for line in std::fs::read_to_string(corpus)?.lines() {
            let row = json::parse(line)?;
            let ids = |name| {
                row.get(name)?
                    .array()?
                    .iter()
                    .map(Value::number)
                    .collect::<Result<Vec<_>>>()
            };
            let prompt = row.get("prompt")?.string()?;
            assert_eq!(tokenizer.encode(prompt)?, ids("ids")?, "encode {prompt:?}");
            assert_eq!(
                tokenizer.decode(&ids("ids")?)?,
                row.get("decoded")?.string()?,
                "decode {prompt:?}"
            );
            assert_eq!(
                tokenizer.chat(prompt, &root.join("chat_template.jinja"))?,
                ids("chat")?,
                "chat {prompt:?}"
            );
            count += 1;
        }
        assert!(count > 0);
        eprintln!("GLM tokenizer: {count} exact encode/chat/decode cases");
        Ok(())
    }
}
