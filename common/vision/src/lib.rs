//! Image decoding, model-specific patchification, native CUDA vision transformers.
use darask_json::Value;
use image::{
    RgbImage,
    imageops::{self, FilterType},
};
#[cfg(feature = "cuda")]
pub mod encoder;
pub type Result<T> = std::result::Result<T, String>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Glm,
    Qwen,
}
impl Kind {
    pub fn image_id(self) -> u32 {
        if self == Self::Glm { 154854 } else { 248056 }
    }
    pub fn marker(self, count: usize) -> String {
        if self == Self::Glm {
            format!(
                "<|begin_of_image|>{}<|end_of_image|>",
                "<|image|>".repeat(count)
            )
        } else {
            format!(
                "<|vision_start|>{}<|vision_end|>",
                "<|image_pad|>".repeat(count)
            )
        }
    }
}
#[derive(Debug, Clone)]
pub struct Image {
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
}
impl Image {
    pub fn tokens(&self) -> usize {
        self.grid_h * self.grid_w / 4
    }
}
#[derive(Debug, Clone)]
pub struct Span {
    pub start: usize,
    pub image: Image,
}
/// Bounds expanded tokens before running the quadratic vision attention. `max_tokens` is per image.
pub fn decode(bytes: &[u8], kind: Kind, max_tokens: usize) -> Result<Image> {
    if !(16..=4096).contains(&max_tokens) {
        return Err("image token budget must be 16..4096".into());
    }
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(256 << 20);
    reader.limits(limits);
    let mut decoded = reader.decode().map_err(|e| format!("image decode: {e}"))?;
    // Camera JPEG orientation is part of the image, including rotations of non-square images.
    let mut metadata = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .into_decoder()
        .map_err(|e| e.to_string())?;
    use image::ImageDecoder;
    if let Ok(o) = metadata.orientation() {
        decoded.apply_orientation(o);
    }
    prepare(&decoded.to_rgb8(), kind, max_tokens)
}
pub fn prepare(rgb: &RgbImage, kind: Kind, max_tokens: usize) -> Result<Image> {
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    if w == 0 || h == 0 || !(16..=4096).contains(&max_tokens) {
        return Err("invalid image shape/budget".into());
    }
    let patch = if kind == Kind::Glm { 14 } else { 16 };
    let factor = patch * 2;
    let (th, tw, ch, cw) = if kind == Kind::Glm {
        let mut th = h.div_ceil(factor) * factor;
        let mut tw = w.div_ceil(factor) * factor;
        if th * tw < 16 * factor * factor {
            let s = (16.0 * (factor * factor) as f64 / (h * w) as f64).sqrt();
            th = ((h as f64 * s).ceil() as usize).div_ceil(factor) * factor;
            tw = ((w as f64 * s).ceil() as usize).div_ceil(factor) * factor;
        }
        if th * tw > max_tokens * factor * factor {
            let (mut lo, mut hi) = (1, h);
            (th, tw) = (factor, factor);
            while lo <= hi {
                let m = (lo + hi) / 2;
                let x = (w * m / h).max(1);
                let a = m.div_ceil(factor) * factor;
                let b = x.div_ceil(factor) * factor;
                if a * b <= max_tokens * factor * factor {
                    (th, tw) = (a, b);
                    lo = m + 1;
                } else {
                    hi = m - 1;
                }
            }
        }
        let mut s = (th as f64 / h as f64).min(tw as f64 / w as f64);
        if h * w >= 16 * factor * factor {
            s = s.min(1.0);
        }
        (
            th,
            tw,
            ((h as f64 * s).floor() as usize).clamp(1, th),
            ((w as f64 * s).floor() as usize).clamp(1, tw),
        )
    } else {
        if h.max(w) as f64 / h.min(w) as f64 > 200.0 {
            return Err("image aspect ratio exceeds 200".into());
        }
        let mut th = ((h as f64 / factor as f64).round_ties_even() as usize).max(1) * factor;
        let mut tw = ((w as f64 / factor as f64).round_ties_even() as usize).max(1) * factor;
        let min_pixels = 65536usize.min(max_tokens * factor * factor);
        if th * tw > max_tokens * factor * factor {
            let b = ((h * w) as f64 / (max_tokens * factor * factor) as f64).sqrt();
            th = ((h as f64 / b / factor as f64).floor() as usize).max(1) * factor;
            tw = ((w as f64 / b / factor as f64).floor() as usize).max(1) * factor;
        } else if th * tw < min_pixels {
            let b = (min_pixels as f64 / (h * w) as f64).sqrt();
            th = ((h as f64 * b / factor as f64).ceil() as usize) * factor;
            tw = ((w as f64 * b / factor as f64).ceil() as usize) * factor;
        }
        (th, tw, th, tw)
    };
    if th * tw / factor / factor > max_tokens {
        return Err("image resize exceeds token budget".into());
    }
    let scaled = imageops::resize(rgb, cw as u32, ch as u32, FilterType::CatmullRom);
    let (gh, gw) = (th / patch, tw / patch);
    let mut patches = Vec::with_capacity(gh * gw * 6 * patch * patch);
    let mean = if kind == Kind::Glm {
        [0.48145466, 0.4578275, 0.40821073]
    } else {
        [0.5; 3]
    };
    #[allow(clippy::excessive_precision)] // Keep the published processor constants readable.
    let std = if kind == Kind::Glm {
        [0.26862954, 0.26130258, 0.27577711]
    } else {
        [0.5; 3]
    };
    for by in 0..gh / 2 {
        for bx in 0..gw / 2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    for c in 0..3 {
                        for _t in 0..2 {
                            for y in 0..patch {
                                for x in 0..patch {
                                    let (iy, ix) =
                                        ((by * 2 + dy) * patch + y, (bx * 2 + dx) * patch + x);
                                    let v = if iy < ch && ix < cw {
                                        scaled.get_pixel(ix as u32, iy as u32)[c] as f32 / 255.0
                                    } else {
                                        0.0
                                    };
                                    patches.push((v - mean[c]) / std[c]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Image {
        patches,
        grid_h: gh,
        grid_w: gw,
    })
}
use darask_text::base64::decode as base64;
/// Resolve image content parts into the exact placeholders consumed by each model's processor.
/// Data URLs keep the inference server independent of network access and client file paths.
pub fn chat(root: &Value, kind: Kind, max_tokens: usize) -> Result<(Value, Vec<Image>)> {
    let mut root = root.clone();
    let mut images = Vec::new();
    let Value::Object(ref mut fields) = root else {
        return Err("request must be an object".into());
    };
    let Some((_, Value::Array(messages))) = fields.iter_mut().find(|(k, _)| k == "messages") else {
        return Ok((root, images));
    };
    for m in messages {
        let role = m
            .optional("role")
            .and_then(|r| r.string().ok())
            .unwrap_or("")
            .to_string();
        let Value::Object(fields) = m else { continue };
        let Some((_, Value::Array(parts))) = fields.iter_mut().find(|(k, _)| k == "content") else {
            continue;
        };
        for p in parts {
            let ty = p
                .optional("type")
                .and_then(|t| t.string().ok())
                .unwrap_or("");
            if ty != "image_url" && ty != "image" {
                continue;
            }
            if role != "user" {
                return Err("images are supported in user messages".into());
            }
            if images.len() >= 8 {
                return Err("at most 8 images per request".into());
            }
            let url = if ty == "image_url" {
                p.optional("image_url")
                    .and_then(|v| v.optional("url").or(Some(v)))
            } else {
                p.optional("url").or_else(|| p.optional("image"))
            }
            .and_then(|v| v.string().ok())
            .ok_or("image_url.url must be a data URL")?;
            let (header, data) = url
                .split_once(',')
                .ok_or("use a base64 data:image/...;base64 URL")?;
            if !header.starts_with("data:image/") || !header.ends_with(";base64") {
                return Err("use a base64 data:image/...;base64 URL".into());
            }
            let detail = p
                .optional("image_url")
                .and_then(|v| v.optional("detail"))
                .or_else(|| p.optional("detail"))
                .and_then(|v| v.string().ok())
                .unwrap_or("auto");
            let budget = match detail {
                "low" => max_tokens.min(256),
                "auto" | "high" => max_tokens,
                _ => return Err("image detail must be auto, low or high".into()),
            };
            let im = decode(&base64(data)?, kind, budget)?;
            *p = Value::Object(vec![
                ("type".into(), Value::String("text".into())),
                ("text".into(), Value::String(kind.marker(im.tokens()))),
            ]);
            images.push(im);
        }
    }
    Ok((root, images))
}
pub fn locate(prompt: &[u32], images: Vec<Image>, kind: Kind) -> Result<Vec<Span>> {
    let mut cursor = 0;
    let mut out = Vec::new();
    for image in images {
        let start = prompt[cursor..]
            .iter()
            .position(|&t| t == kind.image_id())
            .map(|n| cursor + n)
            .ok_or("image placeholder missing from tokenized prompt")?;
        let end = start + image.tokens();
        if end > prompt.len() || prompt[start..end].iter().any(|&t| t != kind.image_id()) {
            return Err("image placeholder/token count mismatch".into());
        }
        cursor = end;
        out.push(Span { start, image });
    }
    if prompt[cursor..].contains(&kind.image_id()) {
        return Err("image placeholder has no image".into());
    }
    Ok(out)
}
/// Three rotary coordinates for every sequence position, including generated text after images.
pub fn positions(spans: &[Span], context: usize) -> Vec<i32> {
    let mut result = Vec::with_capacity(context * 3);
    let (mut at, mut offset) = (0usize, 0i32);
    for s in spans {
        while at < s.start {
            result.extend([at as i32 + offset; 3]);
            at += 1;
        }
        let base = at as i32 + offset;
        let (h, w) = (s.image.grid_h / 2, s.image.grid_w / 2);
        for y in 0..h {
            for x in 0..w {
                result.extend([base, base + y as i32, base + x as i32]);
                at += 1;
            }
        }
        offset = base + h.max(w) as i32 - at as i32;
    }
    while at < context {
        result.extend([at as i32 + offset; 3]);
        at += 1;
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn patch_order_and_temporal_copy() {
        let mut rgb = RgbImage::new(256, 256);
        for (y, row) in rgb.rows_mut().enumerate() {
            for (x, p) in row.enumerate() {
                *p = image::Rgb([x as u8, y as u8, 128]);
            }
        }
        let i = prepare(&rgb, Kind::Qwen, 256).unwrap();
        assert_eq!((i.grid_h, i.grid_w, i.tokens()), (16, 16, 64));
        assert_eq!(&i.patches[..256], &i.patches[256..512]);
        assert_eq!(i.patches[1536], 16.0 / 255.0 * 2.0 - 1.0); // next patch is right neighbour
        assert_eq!(i.patches[2 * 1536 + 512], 16.0 / 255.0 * 2.0 - 1.0); // then below
    }
    #[test]
    fn mrope_after_two_images() {
        let im = Image {
            patches: vec![],
            grid_h: 4,
            grid_w: 6,
        };
        let spans = vec![
            Span {
                start: 2,
                image: im.clone(),
            },
            Span {
                start: 10,
                image: im,
            },
        ];
        let p = positions(&spans, 18);
        assert_eq!(&p[6..12], &[2, 2, 2, 2, 2, 3]);
        assert_eq!(&p[24..27], &[5, 5, 5]);
        assert_eq!(&p[48..51], &[10, 10, 10]);
    }
    #[test]
    fn rejects_bad_base64() {
        for s in ["====", "a===", "abcd=", "YWJj$===", "Zg=A", "Zh=="] {
            assert!(base64(s).is_err(), "{s}");
        }
        assert_eq!(base64("Zg==").unwrap(), b"f");
    }
    #[test]
    fn image_parts_preserve_order_and_count() {
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(RgbImage::from_pixel(112, 112, image::Rgb([255, 0, 0])))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let url = format!(
            "data:image/png;base64,{}",
            darask_text::base64::encode(png.get_ref())
        );
        let root = darask_json::parse(&format!(r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"before"}},{{"type":"image_url","image_url":{{"url":{}}}}},{{"type":"text","text":"after"}}]}}]}}"#, darask_json::quote(&url))).unwrap();
        for kind in [Kind::Glm, Kind::Qwen] {
            let (body, images) = chat(&root, kind, 256).unwrap();
            assert_eq!(images.len(), 1);
            let parts = body.get("messages").unwrap().array().unwrap()[0]
                .get("content")
                .unwrap()
                .array()
                .unwrap();
            assert_eq!(parts[0].get("text").unwrap().string().unwrap(), "before");
            assert_eq!(parts[2].get("text").unwrap().string().unwrap(), "after");
            assert_eq!(
                parts[1].get("text").unwrap().string().unwrap(),
                kind.marker(images[0].tokens())
            );
            let ids = vec![kind.image_id(); images[0].tokens()];
            assert_eq!(locate(&ids, images, kind).unwrap()[0].start, 0);
        }
    }
}
