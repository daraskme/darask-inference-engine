mod batch;
use std::{
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

const ENGINES: [(&str, &str); 3] = [
    ("hayate", "GLM-5.3-Flash"),
    ("kogarashi", "Ornith-1.5-122B"),
    ("tsumuji", "Qwen3.8-Flash-Next"),
];
fn root() -> PathBuf {
    std::env::var_os("DARASK_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
        .canonicalize()
        .expect("DARASK_ROOT")
}
fn main() -> ExitCode {
    match run() {
        Ok(n) => ExitCode::from(n as u8),
        Err(e) => {
            eprintln!("darask: {e}");
            ExitCode::FAILURE
        }
    }
}
fn run() -> Result<i32, String> {
    let root = root();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let action = args
        .first()
        .map(String::as_str)
        .unwrap_or("help")
        .to_owned();
    if action == "help" || action == "--help" {
        println!(
            "darask batch <engine> --input DIR --out captions.jsonl [--jobs 2]\ndarask list | doctor | check | build|serve|run <hayate|kogarashi|tsumuji> [--dry-run] [-- native-options]\ndarask reference kogarashi [--dry-run] [-- native llama-server options]"
        );
        return Ok(0);
    }
    if action == "batch" {
        if args.len() < 2 || args.iter().any(|a| a == "--help") {
            println!(
                "darask batch <hayate|kogarashi|tsumuji> --input FILE_OR_DIR --out captions.jsonl [--jobs 2] [--prompt TEXT] [--endpoint http://HOST:PORT/v1] [--model ID] [--max-tokens 256] [--detail auto|low|high] [--retry 2]"
            );
            return Ok(0);
        }
        return batch::run(&args[1], &args[2..]);
    }
    if action == "list" {
        for (name, model) in ENGINES {
            println!("{name:10} {model}");
        }
        return Ok(0);
    }
    if action == "doctor" {
        for (name, model) in ENGINES {
            let bin = match name {
                "hayate" => "hayate",
                "kogarashi" => "kogarashi-native",
                _ => "tsumuji-serve",
            };
            println!(
                "{name:10} {model:24} binary={} data={}",
                root.join("target/release").join(bin).is_file(),
                match name {
                    "kogarashi" => root
                        .join("engines/kogarashi/models/Ornith-1.5-122B-Q4_K_M.gguf")
                        .is_file(),
                    _ => root.join("local/hayate/src").is_dir(),
                }
            );
        }
        return Ok(0);
    }
    if action == "check" {
        return status(
            Command::new("bash")
                .arg(root.join("scripts/check.sh"))
                .current_dir(&root),
        );
    }
    if args.len() < 2 {
        return Err("engine required; run ./darask help".into());
    }
    let engine = args[1].clone();
    if !ENGINES.iter().any(|(n, _)| *n == engine) {
        return Err(format!("unknown engine {engine}"));
    }
    args.drain(..2);
    let dry = args.first().is_some_and(|a| a == "--dry-run");
    if dry {
        args.remove(0);
    }
    if args.first().is_some_and(|a| a == "--") {
        args.remove(0);
    }
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target"));
    let mut cmd = match action.as_str() {
        "build" => {
            let mut c = Command::new("cargo");
            c.args(["build", "--release"]);
            match engine.as_str() {
                "hayate" => {
                    c.args(["-p", "hayate-cli", "--features", "cuda"]);
                    c.env(
                        "RUSTFLAGS",
                        format!(
                            "-C target-cpu={} {}",
                            std::env::var("DARASK_CPU").unwrap_or_else(|_| "alderlake".into()),
                            std::env::var("RUSTFLAGS").unwrap_or_default()
                        ),
                    );
                }
                "kogarashi" => {
                    c.args(["-p", "kogarashi-engine", "--features", "cuda"]);
                }
                _ => {
                    c.args([
                        "-p",
                        "tsumuji-server",
                        "-p",
                        "tsumuji-engine",
                        "-p",
                        "tsumuji-pack",
                        "-p",
                        "tsumuji-bench",
                    ]);
                }
            }
            c
        }
        "serve" | "run" => match engine.as_str() {
            "hayate" => {
                let mut c = Command::new(target.join("release/hayate"));
                c.arg(&action);
                if action == "serve" {
                    c.args([
                        "--glm-hyt",
                        root.join("local/hayate/hyt/glm53x3-sc260")
                            .to_str()
                            .unwrap(),
                        "--glm-source",
                        root.join("local/hayate/src/GLM-5.3-Flash-dealignai-FP8")
                            .to_str()
                            .unwrap(),
                    ]);
                }
                c
            }
            "kogarashi" => {
                let mut c = Command::new(target.join("release/kogarashi-native"));
                c.args([
                    &action,
                    "--model",
                    root.join("engines/kogarashi/models/Ornith-1.5-122B-Q4_K_M.gguf")
                        .to_str()
                        .unwrap(),
                ]);
                c
            }
            _ => {
                let mut c = Command::new(target.join("release").join(if action == "serve" {
                    "tsumuji-serve"
                } else {
                    "tsumuji"
                }));
                if action == "run" {
                    c.arg("run");
                }
                c.args([
                    "--model",
                    root.join("local/hayate/src/Qwen3.8-Flash-Next-NVFP4")
                        .to_str()
                        .unwrap(),
                ]);
                c
            }
        },
        "reference" if engine == "kogarashi" => {
            let mut c = Command::new(root.join("engines/kogarashi/build/backend/bin/llama-server"));
            c.args([
                "--model",
                root.join("engines/kogarashi/models/Ornith-1.5-122B-Q4_K_M.gguf")
                    .to_str()
                    .unwrap(),
                "--alias",
                "kogarashi-reference",
                "--host",
                "127.0.0.1",
                "--port",
                "8081",
                "--ctx-size",
                "8192",
                "--gpu-layers",
                "99",
                "--flash-attn",
                "on",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--parallel",
                "1",
                "--threads",
                "8",
                "--load-mode",
                "dio",
                "--spec-type",
                "none",
            ]);
            c
        }
        _ => return Err(format!("unknown action {action}")),
    };
    if action == "reference" {
        cmd.env(
            "LD_LIBRARY_PATH",
            format!(
                "{}:/run/opengl-driver/lib:{}",
                root.join("engines/kogarashi/build/backend/bin").display(),
                std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
            ),
        );
    }
    cmd.args(args).current_dir(&root);
    if dry {
        println!("{cmd:?}");
        return Ok(0);
    }
    status(&mut cmd)
}
fn status(cmd: &mut Command) -> Result<i32, String> {
    cmd.status()
        .map(|s| s.code().unwrap_or(1))
        .map_err(|e| format!("{e}; enter nix develop and run ./darask build <engine>"))
}
