fn main() {
    println!("cargo:rerun-if-changed=kernels/vision.cu");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let cuda = std::env::var("CUDA_HOME").expect("CUDA_HOME");
    println!("cargo:rustc-link-search=native={cuda}/lib");
    println!("cargo:rustc-link-lib=cublas");
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    assert!(
        std::process::Command::new(format!("{cuda}/bin/nvcc"))
            .args([
                "-cubin",
                "-O3",
                "-std=c++20",
                "-arch=sm_120a",
                "kernels/vision.cu",
                "-o"
            ])
            .arg(out.join("vision.cubin"))
            .status()
            .unwrap()
            .success()
    );
}
