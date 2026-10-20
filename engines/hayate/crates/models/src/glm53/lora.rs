//! LoRA adapters for GLM-5.3 as unmerged side paths y += s·B(A x) next to the converted
//! (quantized) weights, so the base is never requantized and small deltas are not rounded
//! away. Reads llama.cpp GGUF LoRA adapters (`general.type = adapter`, `adapter.type = lora`,
//! architecture `glm5-next`), e.g. heretic abliteration adapters. Targets: attention output
//! (`attn_output`), the shared expert's and every routed expert's down projection
//! (`ffn_down_shexp`, `ffn_down_exps`) and the dense FFN of layers 0-2 (`ffn_gate/up/down`).
//! Linears that the decode kernels fuse (KDA f_b/g_b, MLA kv_b, expert gate/up) are refused.
//! The scale is llama.cpp's: strength x alpha / rank (strength alone when alpha is 0).
use hayate_core::{Error, Result};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
};

use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub path: PathBuf,
    pub strength: f32,
}
impl Spec {
    pub fn parse(value: &str) -> Result<Self> {
        let (path, strength) = match value.rsplit_once(':') {
            Some((p, s)) => (p, s.parse::<f32>().map_err(|_| bad("invalid strength"))?),
            None => (value, 1.),
        };
        if path.is_empty() || !strength.is_finite() {
            return Err(bad("empty path or nonfinite strength"));
        }
        Ok(Self {
            path: path.into(),
            strength,
        })
    }
}

/// The routed experts' down projections of a layer (one A/B per expert, count 288).
pub const ROUTED: &str = "mlp.experts.down_proj.weight";
const LAYERS: usize = 45;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
}
impl Dtype {
    /// The kernel's code (cu/glm53/lora.cu).
    pub fn code(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Bf16 => 2,
        }
    }
    pub fn bytes(self) -> usize {
        if self == Self::F32 { 4 } else { 2 }
    }
    fn value(self, b: &[u8]) -> f32 {
        match self {
            Self::F32 => f32::from_le_bytes(b.try_into().unwrap()),
            Self::F16 => f16(u16::from_le_bytes(b.try_into().unwrap())),
            Self::Bf16 => f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16),
        }
    }
}
fn f16(h: u16) -> f32 {
    let (exp, mant) = (((h >> 10) & 31) as i32, (h & 1023) as f32);
    let v = match exp {
        0 => mant * 2f32.powi(-24),
        31 => {
            if mant == 0. {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1. + mant / 1024.) * 2f32.powi(exp - 15),
    };
    if h & 0x8000 != 0 { -v } else { v }
}

/// One adapter on one target: A [count][rank][k] and B [count][n][rank] as stored.
#[derive(Clone, Debug)]
pub struct Pair {
    pub rank: usize,
    pub k: usize,
    pub n: usize,
    pub count: usize,
    pub dtype: Dtype,
    pub a: Vec<u8>,
    pub b: Vec<u8>,
    pub scale: f32,
}
impl Pair {
    pub fn a_f32(&self) -> Vec<f32> {
        self.a
            .chunks_exact(self.dtype.bytes())
            .map(|b| self.dtype.value(b))
            .collect()
    }
    pub fn b_f32(&self) -> Vec<f32> {
        self.b
            .chunks_exact(self.dtype.bytes())
            .map(|b| self.dtype.value(b))
            .collect()
    }
}

/// Adapters of one layer by target weight name (`ROUTED` for the routed experts).
pub type Layer = BTreeMap<String, Vec<Pair>>;

/// Every adapter of `specs` for the first `layers` layers; tensors of later layers are
/// skipped (partial-model runs) and reported.
pub fn load(specs: &[Spec], layers: usize) -> Result<Vec<Layer>> {
    let mut out = vec![Layer::new(); layers];
    for spec in specs {
        let mut file = BufReader::new(File::open(&spec.path)?);
        let len = file.get_ref().metadata()?.len();
        let gguf = Gguf::read(&mut file, len)?;
        let (mut used, mut skipped) = (0, 0);
        for (layer, name, pair) in gguf.pairs(&mut file, spec.strength)? {
            if layer >= layers {
                skipped += 1;
                continue;
            }
            out[layer].entry(name).or_default().push(pair);
            used += 1;
        }
        eprintln!(
            "GLM LoRA {}: {used} targets, strength {}{}",
            spec.path.display(),
            spec.strength,
            if skipped > 0 {
                format!(", {skipped} skipped (layers >= {layers})")
            } else {
                String::new()
            }
        );
    }
    Ok(out)
}

/// Routed-expert adapters in F32 for the CPU experts.
#[derive(Debug, Default)]
pub struct HostExperts {
    layers: BTreeMap<usize, Vec<HostPair>>,
}
#[derive(Debug)]
struct HostPair {
    rank: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    scale: f32,
}
impl HostExperts {
    pub fn new(layers: &[Layer]) -> Self {
        let layers = layers
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                let pairs: Vec<_> = l
                    .get(ROUTED)?
                    .iter()
                    .map(|p| HostPair {
                        rank: p.rank,
                        a: p.a_f32(),
                        b: p.b_f32(),
                        scale: p.scale,
                    })
                    .collect();
                Some((i, pairs))
            })
            .collect();
        Self { layers }
    }
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
    /// `out[r] += scale * B_e (A_e act[r])` for every row (act [rows][2048], out [rows][4096]),
    /// F32 sums before the caller's BF16 rounding.
    pub fn apply(&self, layer: usize, expert: usize, act: &[f32], out: &mut [f32]) {
        let Some(pairs) = self.layers.get(&layer) else {
            return;
        };
        for p in pairs {
            let a = &p.a[expert * p.rank * 2048..][..p.rank * 2048];
            let b = &p.b[expert * 4096 * p.rank..][..4096 * p.rank];
            for (x, y) in act
                .as_chunks::<2048>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<4096>().0)
            {
                let t: Vec<f32> = a
                    .as_chunks::<2048>()
                    .0
                    .iter()
                    .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>())
                    .collect();
                for (y, row) in y.iter_mut().zip(b.chunks_exact(p.rank)) {
                    *y += p.scale * row.iter().zip(&t).map(|(w, v)| w * v).sum::<f32>();
                }
            }
        }
    }
}

/// (HF weight name, n, k, count) of a llama.cpp tensor stem in `layer`.
fn target(layer: usize, stem: &str) -> Result<(&'static str, usize, usize, usize)> {
    let dense = layer < 3;
    Ok(match stem {
        "attn_output" => (
            "self_attn.o_proj.weight",
            4096,
            if layer % 4 == 3 { 16384 } else { 8192 },
            1,
        ),
        "ffn_down_shexp" if !dense => ("mlp.shared_experts.down_proj.weight", 4096, 2048, 1),
        "ffn_down_exps" if !dense => (ROUTED, 4096, 2048, 288),
        "ffn_down" if dense => ("mlp.down_proj.weight", 4096, 12288, 1),
        "ffn_gate" if dense => ("mlp.gate_proj.weight", 12288, 4096, 1),
        "ffn_up" if dense => ("mlp.up_proj.weight", 12288, 4096, 1),
        _ => {
            return Err(Error::Weights(format!(
                "GLM LoRA: unsupported target blk.{layer}.{stem} (supported: attn_output, ffn_down_shexp, ffn_down_exps, dense ffn_gate/up/down)"
            )));
        }
    })
}

/// Metadata values this reader looks at (bools and arrays are skipped).
#[derive(Debug)]
enum Value {
    Int(i128),
    Float(f64),
    Str(String),
    Other,
}
#[derive(Debug)]
struct Info {
    name: String,
    dims: Vec<u64>,
    dtype: Dtype,
    offset: u64,
}
#[derive(Debug)]
struct Gguf {
    kv: BTreeMap<String, Value>,
    tensors: Vec<Info>,
    data: u64,
    len: u64,
}
fn bad(message: impl std::fmt::Display) -> Error {
    Error::Weights(format!("GLM LoRA GGUF: {message}"))
}
fn u32_of(r: &mut impl Read) -> Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn u64_of(r: &mut impl Read) -> Result<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn string(r: &mut impl Read) -> Result<String> {
    let n = u64_of(r)?;
    if n > 1 << 20 {
        return Err(bad("string too long"));
    }
    let mut b = vec![0; n as usize];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|_| bad("string is not UTF-8"))
}
fn value(r: &mut impl Read, ty: u32, depth: usize) -> Result<Value> {
    let mut fixed = |n: usize| -> Result<[u8; 8]> {
        let mut b = [0; 8];
        r.read_exact(&mut b[..n])?;
        Ok(b)
    };
    Ok(match ty {
        0 => Value::Int(fixed(1)?[0] as i128),
        1 => Value::Int(fixed(1)?[0] as i8 as i128),
        2 => Value::Int(u16::from_le_bytes(fixed(2)?[..2].try_into().unwrap()) as i128),
        3 => Value::Int(i16::from_le_bytes(fixed(2)?[..2].try_into().unwrap()) as i128),
        4 => Value::Int(u32::from_le_bytes(fixed(4)?[..4].try_into().unwrap()) as i128),
        5 => Value::Int(i32::from_le_bytes(fixed(4)?[..4].try_into().unwrap()) as i128),
        6 => Value::Float(f32::from_le_bytes(fixed(4)?[..4].try_into().unwrap()) as f64),
        7 => {
            fixed(1)?;
            Value::Other
        }
        8 => Value::Str(string(r)?),
        9 => {
            let (inner, n) = (u32_of(r)?, u64_of(r)?);
            if depth > 2 || n > 1 << 24 {
                return Err(bad("array too deep or long"));
            }
            for _ in 0..n {
                value(r, inner, depth + 1)?;
            }
            Value::Other
        }
        10 => Value::Int(u64::from_le_bytes(fixed(8)?) as i128),
        11 => Value::Int(i64::from_le_bytes(fixed(8)?) as i128),
        12 => Value::Float(f64::from_le_bytes(fixed(8)?)),
        _ => return Err(bad(format!("value type {ty}"))),
    })
}
impl Gguf {
    fn read<R: Read + Seek>(r: &mut R, len: u64) -> Result<Self> {
        let mut magic = [0; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(bad("not a GGUF file"));
        }
        let version = u32_of(r)?;
        if !(2..=3).contains(&version) {
            return Err(bad(format!("version {version}")));
        }
        let (tensors, kvs) = (u64_of(r)?, u64_of(r)?);
        if tensors > 1 << 20 || kvs > 1 << 16 {
            return Err(bad("too many tensors or keys"));
        }
        let mut kv = BTreeMap::new();
        for _ in 0..kvs {
            let key = string(r)?;
            let ty = u32_of(r)?;
            kv.insert(key, value(r, ty, 0)?);
        }
        let mut infos = Vec::with_capacity(tensors as usize);
        for _ in 0..tensors {
            let name = string(r)?;
            let rank = u32_of(r)?;
            if !(1..=4).contains(&rank) {
                return Err(bad(format!("{name}: rank {rank}")));
            }
            let dims = (0..rank).map(|_| u64_of(r)).collect::<Result<Vec<_>>>()?;
            let dtype = match u32_of(r)? {
                0 => Dtype::F32,
                1 => Dtype::F16,
                30 => Dtype::Bf16,
                t => {
                    return Err(bad(format!(
                        "{name}: tensor type {t} (F32, F16 or BF16 only)"
                    )));
                }
            };
            infos.push(Info {
                name,
                dims,
                dtype,
                offset: u64_of(r)?,
            });
        }
        let align = match kv.get("general.alignment") {
            Some(Value::Int(a)) if *a > 0 && (*a as u64).is_power_of_two() => *a as u64,
            None => 32,
            _ => return Err(bad("general.alignment")),
        };
        let data = r.stream_position()?.next_multiple_of(align);
        let text = |key: &str| match kv.get(key) {
            Some(Value::Str(s)) => s.as_str(),
            _ => "",
        };
        if text("general.type") != "adapter" || text("adapter.type") != "lora" {
            return Err(bad("not a LoRA adapter (general.type / adapter.type)"));
        }
        if text("general.architecture") != "glm5-next" {
            return Err(bad(format!(
                "architecture {:?}, expected glm5-next (GLM-5.3)",
                text("general.architecture")
            )));
        }
        Ok(Self {
            kv,
            tensors: infos,
            data,
            len,
        })
    }
    fn alpha(&self) -> Result<f64> {
        match self.kv.get("adapter.lora.alpha") {
            None => Ok(0.),
            Some(Value::Float(a)) if a.is_finite() => Ok(*a),
            Some(Value::Int(a)) => Ok(*a as f64),
            _ => Err(bad("adapter.lora.alpha")),
        }
    }
    fn tensor(&self, r: &mut (impl Read + Seek), info: &Info) -> Result<Vec<u8>> {
        let bytes = info
            .dims
            .iter()
            .try_fold(info.dtype.bytes() as u64, |n, &d| n.checked_mul(d))
            .ok_or_else(|| bad(format!("{}: size", info.name)))?;
        let start = self.data.checked_add(info.offset);
        if bytes > 1 << 34 || start.is_none_or(|s| s.saturating_add(bytes) > self.len) {
            return Err(bad(format!("{}: data outside the file", info.name)));
        }
        r.seek(SeekFrom::Start(start.unwrap()))?;
        let mut out = vec![0; bytes as usize];
        r.read_exact(&mut out)?;
        if out
            .chunks_exact(info.dtype.bytes())
            .any(|b| !info.dtype.value(b).is_finite())
        {
            return Err(bad(format!("{}: NaN or Inf", info.name)));
        }
        Ok(out)
    }
    /// (layer, HF target, pair) per lora_a/lora_b tensor pair.
    fn pairs(
        &self,
        r: &mut (impl Read + Seek),
        strength: f32,
    ) -> Result<Vec<(usize, String, Pair)>> {
        let alpha = self.alpha()?;
        let mut halves: BTreeMap<&str, [Option<&Info>; 2]> = BTreeMap::new();
        for info in &self.tensors {
            let (base, half) = if let Some(base) = info.name.strip_suffix(".lora_a") {
                (base, 0)
            } else if let Some(base) = info.name.strip_suffix(".lora_b") {
                (base, 1)
            } else {
                return Err(bad(format!("{}: not a lora_a/lora_b tensor", info.name)));
            };
            let slot = &mut halves.entry(base).or_default()[half];
            if slot.replace(info).is_some() {
                return Err(bad(format!("{}: duplicate", info.name)));
            }
        }
        let mut out = Vec::new();
        for (base, [a, b]) in halves {
            let (Some(a), Some(b)) = (a, b) else {
                return Err(bad(format!("{base}: needs both lora_a and lora_b")));
            };
            let parsed = base
                .strip_prefix("blk.")
                .and_then(|s| s.strip_suffix(".weight"))
                .and_then(|s| s.split_once('.'))
                .and_then(|(l, stem)| Some((l.parse::<usize>().ok()?, stem)));
            let Some((layer, stem)) = parsed else {
                return Err(bad(format!("{base}: expected blk.N.<tensor>.weight")));
            };
            if layer >= LAYERS {
                return Err(bad(format!(
                    "{base}: layer {layer} (the MTP layer is not supported)"
                )));
            }
            let (name, n, k, count) = target(layer, stem)?;
            let rank = a.dims.get(1).copied().unwrap_or(0) as usize;
            let shape = |dims: &[u64], first: usize, second: usize| {
                let mut want = vec![first as u64, second as u64];
                if count > 1 {
                    want.push(count as u64);
                }
                dims == want.as_slice()
            };
            if rank == 0 || rank > 4096 || !shape(&a.dims, k, rank) || !shape(&b.dims, rank, n) {
                return Err(bad(format!(
                    "{base}: shapes {:?} / {:?} do not fit {name} ({n} x {k}{})",
                    a.dims,
                    b.dims,
                    if count > 1 { " x 288 experts" } else { "" }
                )));
            }
            if a.dtype != b.dtype {
                return Err(bad(format!("{base}: lora_a and lora_b types differ")));
            }
            let scale = if alpha != 0. {
                strength as f64 * alpha / rank as f64
            } else {
                strength as f64
            } as f32;
            out.push((
                layer,
                name.to_owned(),
                Pair {
                    rank,
                    k,
                    n,
                    count,
                    dtype: a.dtype,
                    a: self.tensor(r, a)?,
                    b: self.tensor(r, b)?,
                    scale,
                },
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// (name, dims, F16 values) per tensor.
    pub(crate) type Tensors<'a> = Vec<(&'a str, Vec<u64>, Vec<f32>)>;
    /// A GGUF LoRA writer for tests.
    pub(crate) fn write_gguf(
        path: &std::path::Path,
        arch: &str,
        alpha: f32,
        tensors: &[(&str, Vec<u64>, Vec<f32>)],
    ) {
        fn s(out: &mut Vec<u8>, v: &str) {
            out.extend((v.len() as u64).to_le_bytes());
            out.extend(v.as_bytes());
        }
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend(4u64.to_le_bytes());
        for (k, v) in [
            ("general.architecture", arch),
            ("general.type", "adapter"),
            ("adapter.type", "lora"),
        ] {
            s(&mut out, k);
            out.extend(8u32.to_le_bytes());
            s(&mut out, v);
        }
        s(&mut out, "adapter.lora.alpha");
        out.extend(6u32.to_le_bytes());
        out.extend(alpha.to_le_bytes());
        let mut data = Vec::new();
        for (name, dims, values) in tensors {
            s(&mut out, name);
            out.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend(d.to_le_bytes());
            }
            out.extend(1u32.to_le_bytes());
            out.extend((data.len() as u64).to_le_bytes());
            for v in values {
                data.extend(half(*v).to_le_bytes());
            }
            data.resize(data.len().next_multiple_of(32), 0);
        }
        out.resize(out.len().next_multiple_of(32), 0);
        out.extend(data);
        std::fs::write(path, out).unwrap();
    }
    /// F32 -> F16 for exactly representable test values.
    fn half(v: f32) -> u16 {
        if v == 0. {
            return 0;
        }
        let bits = v.to_bits();
        let exp = ((bits >> 23) & 0xff) as i32 - 112;
        assert!((1..31).contains(&exp) && bits & 0x1fff == 0);
        ((bits >> 16) & 0x8000) as u16 | ((exp as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
    }
    fn dir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("glm53-lora-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn spec(path: &std::path::Path, strength: f32) -> Spec {
        Spec {
            path: path.into(),
            strength,
        }
    }

    #[test]
    fn glm53_lora_gguf_targets_shapes_and_scale() {
        let path = dir().join("ok.gguf");
        let routed_a: Vec<f32> = (0..288 * 2 * 2048).map(|i| (i % 7) as f32 * 0.25).collect();
        write_gguf(
            &path,
            "glm5-next",
            4.,
            &[
                (
                    "blk.7.attn_output.weight.lora_a",
                    vec![16384, 2],
                    vec![0.5; 32768],
                ),
                (
                    "blk.7.attn_output.weight.lora_b",
                    vec![2, 4096],
                    vec![-1.; 8192],
                ),
                (
                    "blk.4.attn_output.weight.lora_a",
                    vec![8192, 2],
                    vec![0.5; 16384],
                ),
                (
                    "blk.4.attn_output.weight.lora_b",
                    vec![2, 4096],
                    vec![1.; 8192],
                ),
                (
                    "blk.5.ffn_down_exps.weight.lora_a",
                    vec![2048, 2, 288],
                    routed_a,
                ),
                (
                    "blk.5.ffn_down_exps.weight.lora_b",
                    vec![2, 4096, 288],
                    vec![0.125; 288 * 8192],
                ),
                (
                    "blk.5.ffn_down_shexp.weight.lora_a",
                    vec![2048, 2],
                    vec![1.; 4096],
                ),
                (
                    "blk.5.ffn_down_shexp.weight.lora_b",
                    vec![2, 4096],
                    vec![1.; 8192],
                ),
                (
                    "blk.1.ffn_down.weight.lora_a",
                    vec![12288, 2],
                    vec![1.; 24576],
                ),
                (
                    "blk.1.ffn_down.weight.lora_b",
                    vec![2, 4096],
                    vec![1.; 8192],
                ),
                (
                    "blk.40.attn_output.weight.lora_a",
                    vec![8192, 2],
                    vec![1.; 16384],
                ),
                (
                    "blk.40.attn_output.weight.lora_b",
                    vec![2, 4096],
                    vec![1.; 8192],
                ),
            ],
        );
        let layers = load(&[spec(&path, 0.5)], 10).unwrap();
        assert_eq!(layers.len(), 10);
        let o = &layers[7]["self_attn.o_proj.weight"][0];
        assert_eq!(
            (o.rank, o.k, o.n, o.count, o.dtype),
            (2, 16384, 4096, 1, Dtype::F16)
        );
        // strength 0.5 x alpha 4 / rank 2.
        assert_eq!(o.scale, 1.);
        assert_eq!(o.a_f32()[3], 0.5);
        assert_eq!(o.b_f32()[3], -1.);
        assert_eq!(layers[4]["self_attn.o_proj.weight"][0].k, 8192);
        let r = &layers[5][ROUTED][0];
        assert_eq!(
            (r.count, r.a.len(), r.b.len()),
            (288, 288 * 2 * 2048 * 2, 288 * 8192 * 2)
        );
        assert!(layers[5].contains_key("mlp.shared_experts.down_proj.weight"));
        assert!(layers[1].contains_key("mlp.down_proj.weight"));
        // Layer 40 is past the loaded 10 layers: skipped, not an error.
        assert!(layers.iter().all(|l| l.len() <= 2));

        // CPU routed path: out += s B_e (A_e x) for expert 3.
        let host = HostExperts::new(&layers);
        assert!(!host.is_empty());
        let act: Vec<f32> = (0..2 * 2048).map(|i| ((i % 5) as f32 - 2.) * 0.5).collect();
        let mut out = vec![1f32; 2 * 4096];
        host.apply(5, 3, &act, &mut out);
        let a = r.a_f32();
        for row in 0..2 {
            let t: Vec<f32> = (0..2)
                .map(|j| {
                    (0..2048)
                        .map(|c| a[(3 * 2 + j) * 2048 + c] * act[row * 2048 + c])
                        .sum::<f32>()
                })
                .collect();
            let want = 1. + r.scale * 0.125 * (t[0] + t[1]);
            assert!((out[row * 4096 + 17] - want).abs() <= want.abs() * 1e-5);
        }
        // Layers without routed adapters are untouched.
        let mut same = vec![1f32; 4096];
        host.apply(7, 0, &act[..2048], &mut same);
        assert!(same.iter().all(|&v| v == 1.));
    }

    #[test]
    fn glm53_lora_gguf_refuses_other_models_and_targets() {
        let d = dir();
        let cases: [(&str, &str, Tensors, &str); 5] = [
            (
                "arch",
                "qwen3",
                vec![
                    (
                        "blk.4.attn_output.weight.lora_a",
                        vec![8192, 1],
                        vec![0.; 8192],
                    ),
                    (
                        "blk.4.attn_output.weight.lora_b",
                        vec![1, 4096],
                        vec![0.; 4096],
                    ),
                ],
                "glm5-next",
            ),
            (
                "fused",
                "glm5-next",
                vec![
                    (
                        "blk.4.ffn_gate_exps.weight.lora_a",
                        vec![4096, 1, 288],
                        vec![0.; 4096 * 288],
                    ),
                    (
                        "blk.4.ffn_gate_exps.weight.lora_b",
                        vec![1, 2048, 288],
                        vec![0.; 2048 * 288],
                    ),
                ],
                "unsupported target",
            ),
            (
                "shape",
                "glm5-next",
                vec![
                    (
                        "blk.3.attn_output.weight.lora_a",
                        vec![8192, 1],
                        vec![0.; 8192],
                    ),
                    (
                        "blk.3.attn_output.weight.lora_b",
                        vec![1, 4096],
                        vec![0.; 4096],
                    ),
                ],
                "do not fit",
            ),
            (
                "half",
                "glm5-next",
                vec![(
                    "blk.4.attn_output.weight.lora_a",
                    vec![8192, 1],
                    vec![0.; 8192],
                )],
                "both lora_a and lora_b",
            ),
            (
                "mtp",
                "glm5-next",
                vec![
                    (
                        "blk.45.attn_output.weight.lora_a",
                        vec![8192, 1],
                        vec![0.; 8192],
                    ),
                    (
                        "blk.45.attn_output.weight.lora_b",
                        vec![1, 4096],
                        vec![0.; 4096],
                    ),
                ],
                "MTP",
            ),
        ];
        for (name, arch, tensors, message) in cases {
            let path = d.join(format!("{name}.gguf"));
            write_gguf(&path, arch, 1., &tensors);
            let error = load(&[spec(&path, 1.)], 45).unwrap_err().to_string();
            assert!(error.contains(message), "{name}: {error}");
        }
        std::fs::write(d.join("junk.gguf"), b"not gguf").unwrap();
        assert!(load(&[spec(&d.join("junk.gguf"), 1.)], 45).is_err());
    }

    /// The user's adapter, when present: every tensor maps and the expected counts load.
    #[test]
    #[ignore = "reads GLM53_LORA (a real GGUF LoRA)"]
    fn glm53_lora_real_gguf() {
        let path = std::env::var("GLM53_LORA").unwrap();
        let layers = load(&[spec(path.as_ref(), 1.)], 45).unwrap();
        for (i, l) in layers.iter().enumerate() {
            if !l.is_empty() {
                let names: Vec<_> = l
                    .iter()
                    .map(|(n, p)| format!("{n} r{} s{}", p[0].rank, p[0].scale))
                    .collect();
                eprintln!("layer {i}: {}", names.join(", "));
            }
        }
    }
}
