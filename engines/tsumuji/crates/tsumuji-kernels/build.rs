//! Compiles every `kernels/<dir>/*.cu` to an sm_120a cubin (one per source) and exposes each
//! as `pub const <DIR>_<FILE>: &[u8]` in `cubins.rs`. sm_120a only: no PTX, no other SMs.

use std::env;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../kernels");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    // Extra -D flags, e.g. TSJ_NVCC_DEFINES=-DTSJ_KV_BF16.
    println!("cargo:rerun-if-env-changed=TSJ_NVCC_DEFINES");
    let defines: Vec<String> = env::var("TSJ_NVCC_DEFINES")
        .unwrap_or_default()
        .split_whitespace()
        .map(String::from)
        .collect();
    let cuda_home = env::var("CUDA_HOME").expect("CUDA_HOME is unset: run inside `nix develop`");
    let nvcc = Path::new(&cuda_home).join("bin/nvcc");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    let mut sources = Vec::new();
    for dir in std::fs::read_dir(&root).unwrap() {
        let dir = dir.unwrap().path();
        if !dir.is_dir() || dir.file_name().unwrap() == "include" {
            continue;
        }
        for f in std::fs::read_dir(&dir).unwrap() {
            let f = f.unwrap().path();
            if f.extension().is_some_and(|e| e == "cu") {
                sources.push(f);
            }
        }
    }
    sources.sort();

    let mut jobs = Vec::new();
    for src in &sources {
        let group = src.parent().unwrap().file_name().unwrap().to_str().unwrap();
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let name = format!("{group}_{stem}").to_uppercase();
        let cubin = out.join(format!("{group}_{stem}.cubin"));
        let child = Command::new(&nvcc)
            .args(["-cubin", "-O3", "-std=c++20", "-arch=sm_120a", "-lineinfo"])
            .args(&defines)
            .arg(format!("-I{}", root.join("include").display()))
            .arg("-o")
            .arg(&cubin)
            .arg(src)
            .spawn()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
        jobs.push((src, name, cubin, child));
    }
    let mut rs = String::new();
    for (src, name, cubin, mut child) in jobs {
        assert!(
            child.wait().unwrap().success(),
            "nvcc failed on {}",
            src.display()
        );
        writeln!(
            rs,
            "pub const {name}: &[u8] = include_bytes!({:?});",
            cubin.display().to_string()
        )
        .unwrap();
    }
    let kv_bytes = if defines.iter().any(|d| d == "-DTSJ_KV_BF16") {
        2
    } else {
        4
    };
    writeln!(rs, "/// Bytes per KV / indexer cache element (4: fp32, 2: bf16 with -DTSJ_KV_BF16).\npub const KV_BYTES: usize = {kv_bytes};").unwrap();
    std::fs::write(out.join("cubins.rs"), rs).unwrap();
}
