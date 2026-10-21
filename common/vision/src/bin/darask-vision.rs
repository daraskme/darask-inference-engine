#[cfg(feature = "cuda")]
fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    let get = |k: &str| {
        args.windows(2)
            .find(|v| v[0] == k)
            .map(|v| v[1].clone())
            .ok_or_else(|| format!("required: {k}"))
    };
    let kind = match get("--kind")?.as_str() {
        "glm" => darask_vision::Kind::Glm,
        "qwen" => darask_vision::Kind::Qwen,
        _ => return Err("--kind glm|qwen".into()),
    };
    let max_tokens = get("--image-tokens")
        .ok()
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(1024);
    let image = darask_vision::decode(
        &std::fs::read(get("--image")?).map_err(|e| e.to_string())?,
        kind,
        max_tokens,
    )?;
    let raw = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    if let Ok(path) = get("--patches") {
        std::fs::write(path, raw(&image.patches)).map_err(|e| e.to_string())?;
    }
    let start = std::time::Instant::now();
    let mut encoder =
        darask_vision::encoder::Encoder::load(std::path::Path::new(&get("--model")?), kind)?;
    let load = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let output = encoder.encode(&image)?;
    std::fs::write(get("--out")?, raw(&output)).map_err(|e| e.to_string())?;
    println!(
        "grid={}x{} tokens={} hidden={} load_s={load:.4} encode_s={:.4}",
        image.grid_h,
        image.grid_w,
        image.tokens(),
        encoder.hidden(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
    std::process::exit(1);
}
