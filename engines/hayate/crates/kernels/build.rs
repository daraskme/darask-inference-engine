//! Compiles `cu/**/*.cu` for `sm_120a` only — the single architecture both
//! Hayate machines share. No PTX fallback, no other SMs.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = PathBuf::from(&cuda_home).join("bin/nvcc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // FlashInfer and the CUTLASS it pins, fetched by scripts/fetch-third-party.sh.
    let third_party =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../third_party");
    let flashinfer = third_party.join("flashinfer/include");
    let cutlass = third_party.join("flashinfer/3rdparty/cutlass");
    assert!(
        flashinfer.is_dir() && cutlass.join("include").is_dir(),
        "missing {}: run scripts/fetch-third-party.sh",
        third_party.display()
    );
    let includes = [
        format!("-I{cuda_home}/include"),
        "-Icu/include".to_string(),
        format!("-I{}", flashinfer.display()),
        format!("-I{}", cutlass.join("include").display()),
        format!("-I{}", cutlass.join("tools/util/include").display()),
        // FlashInfer's SM120 FP8 attention (TRT-LLM fmha_v2) headers.
        format!(
            "-I{}",
            third_party.join("flashinfer/csrc/fmha_v2").display()
        ),
    ];

    let mut sources = Vec::new();
    collect_cu(PathBuf::from("cu"), &mut sources);
    assert!(!sources.is_empty(), "no .cu sources under cu/");
    sources.sort();

    // Bound template-heavy nvcc processes so a rebuild leaves RAM/CPU for the
    // hybrid GLM executor and other work on this workstation.
    let mut objects = Vec::new();
    let mut jobs = Vec::new();
    for src in &sources {
        let obj = out_dir.join(src.with_extension("o").to_string_lossy().replace('/', "_"));
        let child = Command::new(&nvcc)
            .args([
                "-c",
                "-O3",
                "-std=c++20",
                "--use_fast_math",
                "-gencode=arch=compute_120a,code=sm_120a",
                "-Xcompiler=-fPIC",
            ])
            .args(&includes)
            .arg("-o")
            .arg(&obj)
            .arg(src)
            .spawn()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
        jobs.push((src, child));
        objects.push(obj);
        if jobs.len() == 4 {
            for (src, mut child) in jobs.drain(..) {
                let status = child.wait().expect("nvcc did not run");
                assert!(status.success(), "nvcc failed on {}", src.display());
            }
        }
    }
    for (src, mut child) in jobs {
        let status = child.wait().expect("nvcc did not run");
        assert!(status.success(), "nvcc failed on {}", src.display());
    }

    // Recreate the archive: `ar r` never drops members of deleted sources.
    let lib = out_dir.join("libhayate_cuda.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar")
        .arg("crs")
        .arg(&lib)
        .args(&objects)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=hayate_cuda");
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-search=native={cuda_home}/lib");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn collect_cu(dir: PathBuf, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .path();
        if path.is_dir() {
            collect_cu(path, out);
        } else if path.extension().is_some_and(|e| e == "cu") {
            out.push(path);
        }
    }
}
