//! libcuda.so comes from the NixOS driver.
fn main() {
    println!("cargo:rustc-link-search=native=/run/opengl-driver/lib");
}
