fn main() {
    println!("cargo:rerun-if-changed=../../kernels/native.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let cuda = std::env::var("CUDA_HOME").expect("CUDA_HOME is required; run nix develop");
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("native.cubin");
    let status = std::process::Command::new(format!("{cuda}/bin/nvcc"))
        .args([
            "-cubin",
            "-O3",
            "-std=c++20",
            "-arch=sm_120a",
            "-lineinfo",
            "../../kernels/native.cu",
            "-o",
        ])
        .arg(out)
        .status()
        .expect("nvcc");
    assert!(status.success(), "Kogarashi CUDA compilation failed");
}
