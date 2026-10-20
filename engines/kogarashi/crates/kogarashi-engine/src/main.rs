use darask_formats::gguf::Gguf;
use kogarashi_engine::{config::Config, tokenizer};
use std::path::PathBuf;

fn main() {
    if let Err(e) = run() {
        eprintln!("kogarashi-native: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let action = args.first().map(String::as_str).unwrap_or("help");
    if action == "help" || args.iter().any(|a| a == "--help") {
        println!(
            "kogarashi-native inspect|encode|run|eval|serve --model FILE.gguf [--prompt TEXT | --prompt-file FILE | --ids FILE.json] [--tokens 128] [--context 8192] [--out FILE.json] [--port 8080]"
        );
        return Ok(());
    }
    let opt = |key: &str| args.windows(2).find(|a| a[0] == key).map(|a| a[1].clone());
    let model = PathBuf::from(opt("--model").ok_or("--model FILE.gguf required")?);
    let g = Gguf::open(&model)?;
    let config = Config::from_gguf(&g)?;
    if action == "inspect" {
        println!("{config:#?}\ntensors: {}", g.tensors.len());
        return Ok(());
    }
    let tok = tokenizer::load(&g)?;
    let prompt = if let Some(path) = opt("--prompt-file") {
        std::fs::read_to_string(path).map_err(|e| e.to_string())?
    } else {
        opt("--prompt").unwrap_or_else(|| "Hello".into())
    };
    let ids = if let Some(path) = opt("--ids") {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let j = darask_json::parse(&text).map_err(|e| e.to_string())?;
        j.array()
            .map_err(|e| e.to_string())?
            .iter()
            .map(|v| v.integer().map(|n| n as u32).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        tok.encode(&prompt)
    };
    if action == "encode" {
        println!("{ids:?}");
        return Ok(());
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (config, tok, ids);
        Err("build with --features cuda (nix develop; ./darask build kogarashi)".into())
    }
    #[cfg(feature = "cuda")]
    {
        let ctx = opt("--context")
            .unwrap_or_else(|| "8192".into())
            .parse::<usize>()
            .map_err(|e| e.to_string())?;
        if action == "serve" {
            let port = opt("--port")
                .unwrap_or_else(|| "8080".into())
                .parse::<u16>()
                .map_err(|e| e.to_string())?;
            let host = opt("--host").unwrap_or_else(|| "127.0.0.1".into());
            let info = darask_api::model::ModelInfo {
                id: "kogarashi-ornith".into(),
                eos: ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"]
                    .iter()
                    .filter_map(|k| g.f64(k).map(|n| n as u32))
                    .collect(),
                defaults: darask_api::sample::Params {
                    temperature: 0.6,
                    top_p: 0.95,
                    top_k: 20,
                    ..Default::default()
                },
                max_ctx: config.max_ctx,
            };
            let turn_start = tok.token_id("<|im_start|>");
            let text = darask_api::model::TextModel {
                template: darask_api::model::ChatTemplate::Ornith,
                tok,
                info,
                ctx,
                turn_start,
            };
            return darask_api::server::serve_with(text, &host, port, move || {
                kogarashi_engine::generator::NativeGen::open(&model, ctx)
            });
        }
        if action == "eval" {
            if ids.is_empty() || ids.len() > ctx {
                return Err("evaluation IDs exceed context".into());
            }
            let path = opt("--logits").ok_or("eval needs --logits FILE.f32")?;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|e| e.to_string())?;
            let mut eng = kogarashi_engine::engine::Engine::load(&model, ctx)?;
            use std::io::Write;
            for &id in &ids {
                eng.step(id)?;
                let row = eng.logits()?;
                let raw: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
                file.write_all(&raw).map_err(|e| e.to_string())?;
            }
            println!(
                "evaluation positions={}, vocab={}, dtype=f32le",
                ids.len(),
                config.vocab
            );
            return Ok(());
        }
        if action != "run" {
            return Err(format!("unknown action {action}"));
        }
        let n = opt("--tokens")
            .unwrap_or_else(|| "128".into())
            .parse::<usize>()
            .map_err(|e| e.to_string())?;
        if ids.is_empty() || ids.len().checked_add(n).is_none_or(|v| v > ctx) {
            return Err("empty prompt or prompt + tokens exceeds context".into());
        }
        let mut eng = kogarashi_engine::engine::Engine::load(&model, ctx)?;
        let start = std::time::Instant::now();
        let mut next = 0;
        for &id in &ids {
            next = eng.step(id)?;
        }
        let prefill_s = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let mut generated = Vec::new();
        for i in 0..n {
            generated.push(next);
            if i + 1 < n {
                next = eng.step(next)?;
            }
        }
        let decode_s = start.elapsed().as_secs_f64();
        let result = darask_text::json::Json::obj([
            (
                "prompt_ids",
                darask_text::json::Json::Arr(
                    ids.iter()
                        .map(|v| darask_text::json::Json::Int(*v as i64))
                        .collect(),
                ),
            ),
            (
                "generated_ids",
                darask_text::json::Json::Arr(
                    generated
                        .iter()
                        .map(|v| darask_text::json::Json::Int(*v as i64))
                        .collect(),
                ),
            ),
            (
                "text",
                darask_text::json::Json::str(tok.decode(&generated, false)),
            ),
            ("prefill_s", darask_text::json::Json::Num(prefill_s)),
            ("decode_s", darask_text::json::Json::Num(decode_s)),
            (
                "decode_tps",
                darask_text::json::Json::Num(n.saturating_sub(1) as f64 / decode_s.max(1e-9)),
            ),
        ])
        .to_string();
        if let Some(path) = opt("--out") {
            std::fs::write(path, &result).map_err(|e| e.to_string())?;
        }
        if let Some(path) = opt("--logits") {
            let floats = eng.logits()?;
            let bytes: Vec<u8> = floats.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(path, bytes).map_err(|e| e.to_string())?;
        }
        println!("{result}");
        Ok(())
    }
}
