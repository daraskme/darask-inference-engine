//! tsumuji-serve --model DIR [--host 127.0.0.1] [--port 8090] [--ctx N] [--name ID]
//!               [--drafts auto|K] [--host-sampling] [--qwen2-regex] [--dry-run]
//! tsumuji-serve encode --model DIR [--chat [--think]] < text   (token ids, like python/tok.py)
//! tsumuji-serve decode --model DIR < ids
//! tsumuji-serve check  --model DIR CASES.json                  (see python/check_text.py)

use std::io::Read as _;
use std::path::PathBuf;
use tsumuji_server::{chat, check, server, tokenizer::Tokenizer};

const USAGE: &str = "usage: tsumuji-serve --model DIR [--host 127.0.0.1] [--port 8090] [--ctx N] [--name ID] [--drafts auto|K] [--host-sampling] [--qwen2-regex] [--dry-run]\n       tsumuji-serve encode --model DIR [--chat [--think]] < text\n       tsumuji-serve decode --model DIR < ids\n       tsumuji-serve check --model DIR CASES.json";

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}

fn stdin() -> Result<String, String> {
    let mut s = String::new();
    std::io::stdin()
        .read_to_string(&mut s)
        .map_err(|e| e.to_string())?;
    Ok(s)
}

fn run() -> Result<bool, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| {
        argv.iter()
            .position(|a| a == k)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    let flag = |k: &str| argv.iter().any(|a| a == k);
    if flag("--help") || flag("-h") {
        println!("{USAGE}");
        return Ok(true);
    }
    let model = PathBuf::from(opt("--model").ok_or(USAGE)?);
    match argv.first().map(String::as_str) {
        Some("encode") => {
            let mut tok = Tokenizer::load(&model)?;
            if flag("--qwen2-regex") {
                tok.regex = tsumuji_server::tokenizer::Regex::Qwen2;
            }
            let mut text = stdin()?;
            if flag("--chat") {
                let o = chat::Options {
                    enable_thinking: Some(flag("--think")),
                    add_generation_prompt: true,
                    ..Default::default()
                };
                text = chat::render(
                    &[chat::Message {
                        role: "user".into(),
                        content: text,
                        ..Default::default()
                    }],
                    &[],
                    &o,
                )?;
            }
            let ids = tok.encode(&text);
            println!(
                "{}",
                ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
            );
            eprintln!("{} tokens", ids.len());
            Ok(true)
        }
        Some("decode") => {
            let tok = Tokenizer::load(&model)?;
            let ids: Vec<u32> = stdin()?
                .split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']')
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().map_err(|e| format!("'{s}': {e}")))
                .collect::<Result<_, _>>()?;
            println!("{}", tok.decode(&ids, false));
            Ok(true)
        }
        Some("check") => {
            let cases = argv
                .iter()
                .skip(1)
                .enumerate()
                .find(|(i, a)| !a.starts_with("--") && !argv[*i].starts_with("--"))
                .map(|(_, a)| PathBuf::from(a))
                .ok_or(USAGE)?;
            check::run(&model, &cases)
        }
        _ => {
            let port = opt("--port")
                .map_or(Ok(8090), |v| v.parse())
                .map_err(|e| format!("--port: {e}"))?;
            let ctx = opt("--ctx")
                .map(|v| v.parse())
                .transpose()
                .map_err(|e| format!("--ctx: {e}"))?;
            let drafts = match opt("--drafts").as_deref() {
                None | Some("auto") => None,
                Some(v) => Some(v.parse().map_err(|e| format!("--drafts: {e}"))?),
            };
            server::serve(server::Config {
                model,
                host: opt("--host").unwrap_or_else(|| "127.0.0.1".into()),
                port,
                ctx,
                name: opt("--name"),
                qwen2_regex: flag("--qwen2-regex"),
                dry_run: flag("--dry-run"),
                drafts,
                host_sampling: flag("--host-sampling"),
            })?;
            Ok(true)
        }
    }
}
