//! Hayate: GLM-5.3-Flash inference for the local RTX PRO 6000.
use std::process::ExitCode;
mod glm53;
mod host;
mod serve;

const USAGE: &str = "usage: hayate <command> [args]

commands:
  run glm53 [options]     GLM-5.3-Flash generation and evaluation
  serve [options]         local GLM chat server (127.0.0.1:7870)
  capabilities --json     machine-readable API capabilities
  bench host [options]    CPU, RAM and NVMe measurements

Use run glm53 --help or serve --help for options.
";
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("capabilities") if args[1..] == ["--json"] => {
            println!(
                "{}",
                hayate_server::capabilities::json(cfg!(feature = "cuda"))
            );
            Ok(())
        }
        Some("run") if args.get(1).map(String::as_str) == Some("glm53") => glm53::run(&args[2..]),
        Some("serve") => serve::run(&args[1..]),
        Some("bench") if args.get(1).map(String::as_str) == Some("host") => host::run(&args[2..]),
        Some("-h" | "--help") | None => {
            print!("{USAGE}");
            Ok(())
        }
        Some(other) => Err(hayate_core::Error::InvalidPlan(format!(
            "unknown command '{other}'\n\n{USAGE}"
        ))),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hayate: {e}");
            ExitCode::from(2)
        }
    }
}
