//! The model's tokenizer.json (Qwen2-style byte-level BPE), matching Hugging Face `tokenizers`:
//!
//! 1. Added tokens are split off first, leftmost-longest, on the raw text (all are
//!    `normalized: false` here).
//! 2. The rest is NFC-normalized and split by the pre-tokenizer regex
//!    `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!    by a hand-written matcher with Oniguruma's leftmost-first semantics (`pretokenize`).
//!    transformers 5.x's `Qwen2Tokenizer` replaces that regex with Qwen2's, which lacks the
//!    `\p{M}` (combining marks then count as punctuation); `Regex::Qwen2` reproduces it.
//! 3. Each piece's UTF-8 bytes go through BPE with the merge ranks (same priority queue and
//!    tie-breaking as `tokenizers`' `Word::merge_all`).
//!
//! Decoding maps tokens back to bytes; `DecodeStream` emits text incrementally and never splits
//! a UTF-8 sequence (invalid bytes become U+FFFD exactly as `String::from_utf8_lossy` does).

use crate::json::Json;
use crate::unicode::{self, L, M, N};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

#[derive(Debug)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
}

/// Which pre-tokenizer regex to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regex {
    /// tokenizer.json's: letters and combining marks form words.
    Model,
    /// transformers 5.x's hard-coded Qwen2 regex: `\p{L}+` words, marks with the punctuation.
    Qwen2,
}

#[derive(Debug)]
pub struct Tokenizer {
    pub regex: Regex,
    /// Token id of each single byte.
    byte_id: [u32; 256],
    /// (left, right) -> (rank, merged id).
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Bytes of every id (added tokens: their content); empty for unused ids.
    bytes: Vec<Vec<u8>>,
    special: Vec<bool>,
    added: Vec<AddedToken>,
    /// Added tokens by first byte, longest first.
    added_by_first: Vec<Vec<usize>>,
    vocab: HashMap<Vec<u8>, u32>,
    added_lookup: HashMap<String, u32>,
}

/// GPT-2's byte -> printable char map used by byte-level BPE.
fn byte_chars() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut n = 0;
    for b in 0..256u32 {
        let printable =
            (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        out[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    out
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let p = model_dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        Self::from_json(&text).map_err(|e| format!("{}: {e}", p.display()))
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let j = Json::parse(text)?;
        let model = j.get("model").ok_or("no model")?;
        if model.get("type").and_then(Json::as_str) != Some("BPE") {
            return Err("model.type is not BPE".into());
        }
        let pre = j
            .get("pre_tokenizer")
            .map(Json::to_string)
            .unwrap_or_default();
        let regex = if pre.contains(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
            Regex::Model
        } else if pre.contains(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
            Regex::Qwen2
        } else {
            return Err(format!("unexpected pre_tokenizer (this tokenizer implements Qwen's regexes): {pre}"));
        };
        let norm = j
            .get("normalizer")
            .and_then(|n| n.get("type"))
            .and_then(Json::as_str);
        if norm != Some("NFC") {
            return Err(format!("unexpected normalizer {norm:?}"));
        }
        let chars = byte_chars();
        let mut char_byte = HashMap::new();
        for (b, c) in chars.iter().enumerate() {
            char_byte.insert(*c, b as u8);
        }
        let to_bytes = |s: &str| -> Result<Vec<u8>, String> {
            s.chars()
                .map(|c| {
                    char_byte
                        .get(&c)
                        .copied()
                        .ok_or_else(|| format!("token '{s}' is not byte-level"))
                })
                .collect()
        };

        let mut vocab = HashMap::new();
        let mut bytes: Vec<Vec<u8>> = Vec::new();
        for (k, v) in model
            .get("vocab")
            .and_then(Json::as_obj)
            .ok_or("no vocab")?
        {
            let id = v.as_u64().ok_or("bad vocab id")? as usize;
            let b = to_bytes(k)?;
            if bytes.len() <= id {
                bytes.resize(id + 1, Vec::new());
            }
            bytes[id] = b.clone();
            vocab.insert(b, id as u32);
        }
        let mut byte_id = [0u32; 256];
        for (b, id) in byte_id.iter_mut().enumerate() {
            *id = *vocab
                .get(&vec![b as u8])
                .ok_or_else(|| format!("byte {b} not in vocab"))?;
        }
        let mut merges = HashMap::new();
        for (rank, m) in model
            .get("merges")
            .and_then(Json::as_arr)
            .ok_or("no merges")?
            .iter()
            .enumerate()
        {
            let (a, b) = match m {
                Json::Str(s) => s.split_once(' ').ok_or("bad merge")?,
                Json::Arr(p) if p.len() == 2 => (
                    p[0].as_str().ok_or("bad merge")?,
                    p[1].as_str().ok_or("bad merge")?,
                ),
                _ => return Err("bad merge".into()),
            };
            let (ba, bb) = (to_bytes(a)?, to_bytes(b)?);
            let joined = [ba.as_slice(), bb.as_slice()].concat();
            let ida = *vocab.get(&ba).ok_or("merge part not in vocab")?;
            let idb = *vocab.get(&bb).ok_or("merge part not in vocab")?;
            let idm = *vocab.get(&joined).ok_or("merge result not in vocab")?;
            // A repeated pair keeps its last rank, as tokenizers' merge map does.
            merges.insert((ida, idb), (rank as u32, idm));
        }
        let mut added = Vec::new();
        for t in j.get("added_tokens").and_then(Json::as_arr).unwrap_or(&[]) {
            let id = t
                .get("id")
                .and_then(Json::as_u64)
                .ok_or("bad added token")? as u32;
            let content = t
                .get("content")
                .and_then(Json::as_str)
                .ok_or("bad added token")?
                .to_string();
            for flag in ["lstrip", "rstrip", "single_word", "normalized"] {
                if t.get(flag).and_then(Json::as_bool) == Some(true) {
                    return Err(format!("added token {content}: {flag} is not supported"));
                }
            }
            let special = t.get("special").and_then(Json::as_bool).unwrap_or(false);
            added.push(AddedToken {
                id,
                content,
                special,
            });
        }
        let max_id = added
            .iter()
            .map(|a| a.id as usize + 1)
            .max()
            .unwrap_or(0)
            .max(bytes.len());
        bytes.resize(max_id, Vec::new());
        let mut special = vec![false; max_id];
        let mut added_by_first = vec![Vec::new(); 256];
        let mut added_lookup = HashMap::new();
        for (i, a) in added.iter().enumerate() {
            bytes[a.id as usize] = a.content.as_bytes().to_vec();
            special[a.id as usize] = a.special;
            added_lookup.insert(a.content.clone(), a.id);
            if let Some(&f) = a.content.as_bytes().first() {
                added_by_first[f as usize].push(i);
            }
        }
        for v in &mut added_by_first {
            v.sort_by_key(|&i| Reverse(added[i].content.len()));
        }
        Ok(Self {
            regex,
            byte_id,
            merges,
            bytes,
            special,
            added,
            added_by_first,
            vocab,
            added_lookup,
        })
    }

    /// Number of ids with a token (vocab + added tokens).
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn added_tokens(&self) -> &[AddedToken] {
        &self.added
    }

    /// Id of an added token or a vocab token given as its text.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.added_lookup
            .get(text)
            .copied()
            .or_else(|| self.vocab.get(text.as_bytes()).copied())
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// Token ids of `text` (no special tokens added; added tokens in the text are recognized).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        let b = text.as_bytes();
        let (mut i, mut rest) = (0, 0);
        while i < b.len() {
            let hit = self.added_by_first[b[i] as usize]
                .iter()
                .map(|&k| &self.added[k])
                .find(|a| b[i..].starts_with(a.content.as_bytes()));
            if let Some(a) = hit {
                self.encode_plain(&text[rest..i], &mut out);
                out.push(a.id);
                i += a.content.len();
                rest = i;
            } else {
                i += 1;
            }
        }
        self.encode_plain(&text[rest..], &mut out);
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let text = unicode::nfc(text);
        pretokenize(&text, self.regex, |piece| self.bpe(piece.as_bytes(), out));
    }

    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) {
        if piece.len() == 1 {
            out.push(self.byte_id[piece[0] as usize]);
            return;
        }
        #[derive(Clone, Copy)]
        struct Sym {
            id: u32,
            prev: i32,
            next: i32,
            alive: bool,
        }
        let n = piece.len();
        let mut s: Vec<Sym> = piece
            .iter()
            .enumerate()
            .map(|(i, &b)| Sym {
                id: self.byte_id[b as usize],
                prev: i as i32 - 1,
                next: if i + 1 < n { i as i32 + 1 } else { -1 },
                alive: true,
            })
            .collect();
        // Min-heap on (rank, position), like tokenizers' `Merge` ordering.
        let mut q: BinaryHeap<Reverse<(u32, usize, u32)>> = BinaryHeap::with_capacity(n);
        for i in 0..n - 1 {
            if let Some(&(rank, id)) = self.merges.get(&(s[i].id, s[i + 1].id)) {
                q.push(Reverse((rank, i, id)));
            }
        }
        while let Some(Reverse((_, pos, new_id))) = q.pop() {
            if !s[pos].alive || s[pos].next < 0 {
                continue;
            }
            let nx = s[pos].next as usize;
            // Stale entry: the pair at `pos` changed since it was queued.
            if self
                .merges
                .get(&(s[pos].id, s[nx].id))
                .is_none_or(|&(_, id)| id != new_id)
            {
                continue;
            }
            s[pos].id = new_id;
            s[pos].next = s[nx].next;
            s[nx].alive = false;
            if s[nx].next >= 0 {
                let nn = s[nx].next as usize;
                s[nn].prev = pos as i32;
            }
            if s[pos].prev >= 0 {
                let pv = s[pos].prev as usize;
                if let Some(&(rank, id)) = self.merges.get(&(s[pv].id, s[pos].id)) {
                    q.push(Reverse((rank, pv, id)));
                }
            }
            if s[pos].next >= 0 {
                let nn = s[pos].next as usize;
                if let Some(&(rank, id)) = self.merges.get(&(s[pos].id, s[nn].id)) {
                    q.push(Reverse((rank, pos, id)));
                }
            }
        }
        out.extend(s.iter().filter(|x| x.alive).map(|x| x.id));
    }

    /// Text of `ids` (like `tokenizer.decode`).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut d = DecodeStream::new(skip_special);
        let mut s: String = ids.iter().map(|&t| d.push(self, t)).collect();
        s.push_str(&d.finish());
        s
    }
}

/// Character classes for the pre-tokenizer.
const WS: u8 = 8;
const NL: u8 = 16;

fn class(c: char) -> u8 {
    let mut k = unicode::category(c);
    if unicode::is_space_re(c) {
        k |= WS;
    }
    if c == '\r' || c == '\n' {
        k |= NL;
    }
    k
}

/// Splits `s` into the pre-tokenizer's pieces (every character belongs to exactly one piece).
pub fn pretokenize(s: &str, regex: Regex, mut f: impl FnMut(&str)) {
    // Word characters: L (and M with the model's regex).
    let word = if regex == Regex::Model { L | M } else { L };
    let cs: Vec<(usize, char)> = s.char_indices().collect();
    let k: Vec<u8> = cs.iter().map(|&(_, c)| class(c)).collect();
    let n = cs.len();
    let at = |i: usize| if i < n { cs[i].0 } else { s.len() };
    let mut i = 0;
    while i < n {
        let j = match_at(&cs, &k, word, i);
        debug_assert!(j > i);
        f(&s[at(i)..at(j)]);
        i = j;
    }
}

/// End (exclusive, in chars) of the regex match starting at `i`; alternatives are tried in order.
fn match_at(cs: &[(usize, char)], k: &[u8], word: u8, i: usize) -> usize {
    let n = cs.len();
    let c = cs[i].1;
    let fold = |j: usize| -> char {
        match cs.get(j).map(|x| x.1) {
            Some('\u{17f}') => 's', // LATIN SMALL LETTER LONG S folds to 's' under (?i)
            Some(x) => x.to_ascii_lowercase(),
            None => '\0',
        }
    };
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if c == '\'' {
        match (fold(i + 1), fold(i + 2)) {
            ('s' | 't' | 'm' | 'd', _) => return i + 2,
            ('r' | 'v', 'e') | ('l', 'l') => return i + 3,
            _ => {}
        }
    }
    let lm = |j: usize| j < n && k[j] & word != 0;
    let run = |mut j: usize, p: &dyn Fn(usize) -> bool| {
        while p(j) {
            j += 1;
        }
        j
    };
    // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if k[i] & (NL | L | N) == 0 && lm(i + 1) {
        return run(i + 1, &lm);
    }
    if lm(i) {
        return run(i, &lm);
    }
    // \p{N}
    if k[i] & N != 0 {
        return i + 1;
    }
    // ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
    let punct = |j: usize| j < n && k[j] & (WS | word | N) == 0;
    let start = if c == ' ' && punct(i + 1) {
        Some(i + 1)
    } else if punct(i) {
        Some(i)
    } else {
        None
    };
    if let Some(st) = start {
        let j = run(st, &punct);
        return run(j, &|j| j < n && k[j] & NL != 0);
    }
    // Whitespace (always matches here: every other character was taken above).
    let j = run(i, &|j| j < n && k[j] & WS != 0);
    debug_assert!(j > i);
    // \s*[\r\n]+ : up to the last newline of the run.
    if let Some(q) = (i..j).rev().find(|&q| k[q] & NL != 0) {
        return q + 1;
    }
    // \s+(?!\S) : the run at the end of text, else all but its last character.
    if j == n || j - i == 1 { j } else { j - 1 }
}

/// Incremental detokenization: `push` returns the text that is complete so far.
#[derive(Debug, Default)]
pub struct DecodeStream {
    pending: Vec<u8>,
    skip_special: bool,
}

impl DecodeStream {
    pub fn new(skip_special: bool) -> Self {
        Self {
            pending: Vec::new(),
            skip_special,
        }
    }

    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        if self.skip_special && tok.is_special(id) {
            return String::new();
        }
        self.pending.extend_from_slice(tok.token_bytes(id));
        self.take(false)
    }

    /// The rest, with an unfinished UTF-8 sequence as U+FFFD.
    pub fn finish(&mut self) -> String {
        self.take(true)
    }

    fn take(&mut self, all: bool) -> String {
        let mut out = String::new();
        let mut at = 0;
        while at < self.pending.len() {
            match std::str::from_utf8(&self.pending[at..]) {
                Ok(s) => {
                    out.push_str(s);
                    at = self.pending.len();
                }
                Err(e) => {
                    let v = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[at..at + v]).unwrap());
                    at += v;
                    match e.error_len() {
                        Some(l) => {
                            out.push('\u{FFFD}');
                            at += l;
                        }
                        None if all => {
                            out.push('\u{FFFD}');
                            at = self.pending.len();
                        }
                        None => break,
                    }
                }
            }
        }
        self.pending.drain(..at);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(s: &str) -> Vec<String> {
        let mut v = Vec::new();
        pretokenize(s, Regex::Model, |p| v.push(p.to_string()));
        v
    }

    #[test]
    fn pretokenizer() {
        assert_eq!(
            pieces("Hello world's 123"),
            ["Hello", " world", "'s", " ", "1", "2", "3"]
        );
        assert_eq!(
            pieces("a  b\n\n  c "),
            ["a", " ", " b", "\n\n", " ", " c", " "]
        );
        assert_eq!(pieces("x = (y);\n"), ["x", " =", " (", "y", ");\n"]);
        assert_eq!(pieces("日本語の文章です。"), ["日本語の文章です", "。"]);
        assert_eq!(pieces("I'LL x'ſ"), ["I", "'LL", " x", "'ſ"]);
    }
}
