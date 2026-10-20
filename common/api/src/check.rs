//! `tsumuji-serve check`: compares the tokenizer and the chat template with the cases
//! `python/check_text.py` wrote: encode with tokenizer.json's regex against `tokenizers`, and
//! with Qwen2's regex against transformers' AutoTokenizer; decode (full and streaming); chat
//! prompts, their ids, and the template's errors.

use crate::chat::{self, Options};
use crate::json::Json;
use crate::request::parse_messages;
use crate::tokenizer::{DecodeStream, Regex, Tokenizer};
use std::path::Path;
use std::time::Instant;

fn ids(v: Option<&Json>) -> Vec<u32> {
    v.and_then(Json::as_arr)
        .unwrap_or(&[])
        .iter()
        .filter_map(Json::as_u64)
        .map(|x| x as u32)
        .collect()
}

fn first_diff(a: &[u32], b: &[u32]) -> String {
    let i = a
        .iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .unwrap_or(a.len().min(b.len()));
    let w = |v: &[u32]| {
        v[i.saturating_sub(3).min(v.len())..(i + 4).min(v.len())]
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!(
        "lengths {} vs {}, first difference at {i}: got [{}] want [{}]",
        a.len(),
        b.len(),
        w(a),
        w(b)
    )
}

fn short(s: &str) -> String {
    let t: String = s.chars().take(60).collect();
    format!("{t:?}{}", if s.chars().count() > 60 { "…" } else { "" })
}

/// Options from apply_chat_template kwargs (absent = undefined).
fn options(k: Option<&Json>) -> Options {
    let b = |n: &str| k.and_then(|k| k.get(n)).and_then(Json::as_bool);
    Options {
        enable_thinking: b("enable_thinking"),
        reasoning_effort: k
            .and_then(|k| k.get("reasoning_effort"))
            .and_then(Json::as_str)
            .map(str::to_string),
        preserve_thinking: b("preserve_thinking"),
        add_generation_prompt: b("add_generation_prompt").unwrap_or(false),
    }
}

fn streamed(tok: &Tokenizer, v: &[u32], skip: bool) -> String {
    let mut ds = DecodeStream::new(skip);
    let mut s: String = v.iter().map(|&t| ds.push(tok, t)).collect();
    s += &ds.finish();
    s
}

pub fn run(model: &Path, cases: &Path) -> Result<bool, String> {
    let t = Instant::now();
    let mut tok = Tokenizer::load(model)?;
    eprintln!(
        "tokenizer loaded in {:.2} s ({} ids, regex {:?})",
        t.elapsed().as_secs_f64(),
        tok.len(),
        tok.regex
    );
    let text = std::fs::read_to_string(cases).map_err(|e| format!("{}: {e}", cases.display()))?;
    let j = Json::parse(&text)?;
    let arr = |k: &str| j.get(k).and_then(Json::as_arr).unwrap_or(&[]);
    let mut bad = 0usize;
    let mut report = |name: &str, n: usize, fails: usize| {
        println!(
            "{name:<28} {:>5} / {n:<5} {}",
            n - fails,
            if fails == 0 { "ok" } else { "FAIL" }
        );
        bad += fails;
    };

    for (regex, key, label) in [
        (Regex::Model, "ids", "tokenizer.json"),
        (Regex::Qwen2, "ids_hf", "AutoTokenizer"),
    ] {
        tok.regex = regex;
        let (mut fails, mut chars, mut secs) = (0, 0usize, 0.0);
        for c in arr("encode") {
            let s = c.get("text").and_then(Json::as_str).unwrap_or("");
            let want = ids(c.get(key));
            let t = Instant::now();
            let got = tok.encode(s);
            secs += t.elapsed().as_secs_f64();
            chars += s.chars().count();
            if got != want {
                fails += 1;
                println!("encode ({label}) {}: {}", short(s), first_diff(&got, &want));
            }
        }
        report(&format!("encode vs {label}"), arr("encode").len(), fails);
        println!(
            "{:<28} {:.1} M chars/s ({chars} chars)",
            "  encode speed",
            chars as f64 / secs.max(1e-9) / 1e6
        );

        let mut fails = 0;
        for c in arr("decode")
            .iter()
            .filter(|c| c.get("reencoded").is_some())
        {
            let name = c.get("name").and_then(Json::as_str).unwrap_or("?");
            let text = c.get("text").and_then(Json::as_str).unwrap_or("");
            let (got, want) = (
                tok.encode(text),
                ids(c.get(if regex == Regex::Model {
                    "reencoded"
                } else {
                    "reencoded_hf"
                })),
            );
            if got != want {
                fails += 1;
                println!("re-encode ({label}) {name}: {}", first_diff(&got, &want));
            }
            if regex == Regex::Qwen2 && c.get("roundtrip").and_then(Json::as_bool) == Some(false) {
                println!(
                    "  note: eval/{name}.ids: transformers' own encode(decode(ids)) != ids ({})",
                    first_diff(&want, &ids(c.get("ids")))
                );
            }
        }
        report(
            &format!("eval re-encode vs {label}"),
            arr("decode")
                .iter()
                .filter(|c| c.get("reencoded").is_some())
                .count(),
            fails,
        );

        let mut fails = 0;
        for c in arr("chat") {
            let name = c.get("name").and_then(Json::as_str).unwrap_or("?");
            let o = options(c.get("kwargs"));
            let tools = c.get("tools").and_then(Json::as_arr).unwrap_or(&[]);
            let r = parse_messages(c.get("messages")).and_then(|m| chat::render(&m, tools, &o));
            match (r, c.get("error").and_then(Json::as_str)) {
                (Ok(got), None) => {
                    let want = c.get("text").and_then(Json::as_str).unwrap_or("");
                    if got != want {
                        fails += 1;
                        println!("chat {name}: prompt differs\n  got  {got:?}\n  want {want:?}");
                        continue;
                    }
                    let (g, w) = (tok.encode(&got), ids(c.get(key)));
                    if g != w {
                        fails += 1;
                        println!("chat ({label}) {name}: ids differ: {}", first_diff(&g, &w));
                    }
                }
                (Err(e), Some(want)) => {
                    if e != want && regex == Regex::Model {
                        println!(
                            "  note: chat {name}: both refuse, messages differ: got {e:?}, transformers {want:?}"
                        );
                    }
                }
                (Ok(_), Some(want)) => {
                    fails += 1;
                    println!("chat {name}: transformers raises {want:?}, we render a prompt");
                }
                (Err(e), None) => {
                    fails += 1;
                    println!("chat {name}: we fail ({e}), transformers renders a prompt");
                }
            }
        }
        report(
            &format!("chat template + ids ({label})"),
            arr("chat").len(),
            fails,
        );
    }

    // Decode: the encode cases' ids, the eval files, random ids that split UTF-8 sequences.
    let mut fails = 0;
    let mut n = 0;
    for c in arr("encode") {
        let v = ids(c.get("ids"));
        for (key, skip) in [("decoded", false), ("decoded_skip", true)] {
            n += 1;
            let want = c.get(key).and_then(Json::as_str).unwrap_or("");
            let (full, st) = (tok.decode(&v, skip), streamed(&tok, &v, skip));
            if full != want || st != want {
                fails += 1;
                println!(
                    "decode (skip {skip}) {}: got {} streamed {} want {}",
                    short(want),
                    short(&full),
                    short(&st),
                    short(want)
                );
            }
        }
    }
    for c in arr("decode") {
        let v = ids(c.get("ids"));
        let name = c.get("name").and_then(Json::as_str).unwrap_or("?");
        for (key, skip) in [("text", false), ("text_skip", true)] {
            n += 1;
            let want = c.get(key).and_then(Json::as_str).unwrap_or("");
            let (full, st) = (tok.decode(&v, skip), streamed(&tok, &v, skip));
            if full != want || st != want {
                fails += 1;
                println!(
                    "decode {name} (skip {skip}): got {} streamed {} want {}",
                    short(&full),
                    short(&st),
                    short(want)
                );
            }
        }
    }
    report("decode (full + streaming)", n, fails);
    Ok(bad == 0)
}
