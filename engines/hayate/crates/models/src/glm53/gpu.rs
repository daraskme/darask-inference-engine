use super::{EXPERTS, HC, HIDDEN, TOP_K, experts::matrix_stride, validate_raw};
use hayate_core::{DType, Error, MachineId, QuantSpec, Result, hyt::HytFile};
use hayate_kernels::{
    glm53::ffi as g,
    graph::Graph,
    runtime::{Device, DeviceBuffer, Event, PinnedHost, Stream, check},
    text::ffi::hayate_text_linear,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
    path::Path,
};

#[path = "device.rs"]
mod device;
#[path = "prefill_gpu.rs"]
mod prefill_gpu;
pub(in crate::glm53) use device::{Defer, DeviceRefs, LaneSync, PRE_META_BYTES};
#[path = "lora_gpu.rs"]
mod lora_gpu;
#[path = "session_state.rs"]
pub(crate) mod session_state;
#[path = "verification.rs"]
mod verification;

pub(super) type Ptr = *mut c_void;

/// Metadata of one layer's packed chunks, uploaded in one copy:
/// images[288] u64 | formats[288] 8 B | ranges[288] {rows, row in chunk} | assignments.
#[derive(Default)]
struct PackedMeta {
    images: Vec<u64>,
    formats: Vec<u8>,
    ranges: Vec<u32>,
    assignments: Vec<u32>,
    /// Per chunk: first expert, first assignment.
    chunks: Vec<(usize, usize)>,
}

impl PackedMeta {
    fn push(
        &mut self,
        experts: &[usize],
        images: &[Ptr],
        groups: &[Vec<u32>],
        formats: &[hayate_kernels::glm53::lowbit::ExpertFormat],
    ) {
        let row_base = self.assignments.len();
        self.chunks.push((self.images.len(), row_base));
        for (&e, &image) in experts.iter().zip(images) {
            self.images.push(image as u64);
            for m in [formats[e].gate_up, formats[e].down] {
                self.formats.extend([m.format, m.grid, m.group, m.reserved]);
            }
            self.ranges.extend([
                groups[e].len() as u32,
                (self.assignments.len() - row_base) as u32,
            ]);
            self.assignments.extend_from_slice(&groups[e]);
        }
    }
    #[cfg(feature = "cuda")]
    fn upload(&self, stream: &Stream, batch: &mut super::prefill::Prefill) -> Result<()> {
        let mut raw = vec![0u8; EXPERTS * 24 + self.assignments.len() * 4];
        for (i, v) in self.images.iter().enumerate() {
            raw[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        raw[EXPERTS * 8..EXPERTS * 8 + self.formats.len()].copy_from_slice(&self.formats);
        for (i, v) in self.ranges.iter().enumerate() {
            let at = EXPERTS * 16 + i * 4;
            raw[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in self.assignments.iter().enumerate() {
            let at = EXPERTS * 24 + i * 4;
            raw[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        stream.copy_h2d(batch.buffers.get_mut("packed_meta").unwrap(), &raw)
    }
}
/// Token splits of the decode MLA attention (combined afterwards).
const MLA_SPLITS: usize = 8;

/// Device buffers that carry decode state from one token to the next.
pub(in crate::glm53) const SESSION_BUFFERS: [&str; 7] = [
    "state",
    "hq",
    "hk",
    "hv",
    "pool_keys",
    "pool_gates",
    "pools",
];
/// One layer's saved state; see `Layer::save_session`.
#[derive(Debug, Clone)]
pub struct LayerSession {
    pub position: usize,
    pub buffers: Vec<(&'static str, Vec<u8>)>,
    pub latent: Vec<u8>,
    pub tail_keys: Vec<u8>,
    pub tail_gates: Vec<u8>,
}
#[derive(Debug)]
pub struct LayerOutput {
    pub tensors: BTreeMap<String, Vec<u8>>,
    /// Includes router D2H and host submission gaps, excludes input/output copies
    /// and weight loading. Warmup/repeats are chosen by the caller.
    pub cuda_ms: f32,
}

#[derive(Debug)]
pub struct Layer<'d> {
    device: &'d Device,
    stream: Stream,
    pub(super) index: usize,
    tokens: usize,
    weights: BTreeMap<String, DeviceBuffer>,
    buffers: BTreeMap<String, DeviceBuffer>,
    fp8: BTreeSet<String>,
    nvfp4: BTreeSet<String>,
    pub(super) hadamard_block: u32,
    /// The layer stores LOWBIT experts (decode through the format-aware kernels).
    pub(super) lowbit: bool,
    /// EXL3 experts (gate/up k2, down k2): the EXL3 kernels (`lowbit` is set too).
    pub(super) exl3: Option<(u8, u8)>,
    /// The PCIe lane's copy size: the largest expert image of the layer (16-byte multiple; 0: no lane).
    pub(super) pcie_bytes: usize,
    position: usize,
    context: usize,
    latent: Option<hayate_kernels::runtime::PinnedHost>,
    tail_keys: Vec<u8>,
    tail_gates: Vec<u8>,
    pub(super) done: Event,
    pub(super) reference: bool,
    pub(super) graphs: bool,
    before_graph: Option<Graph>,
    moe_graph: Option<Graph>,
    warmed: bool,
    profile: Option<super::profile::Profile>,
    pub(super) asynchronous: Option<super::async_decode::LayerAsync>,
    pub(super) async_active: bool,
    route_host: PinnedHost,
    image_host: PinnedHost,
    cpu_input: PinnedHost,
    cpu_output: PinnedHost,
    prediction_stream: Stream,
    prediction_host: PinnedHost,
    routing_ready: Event,
    verification: Option<verification::Verification>,
    pub(super) dense_formats: BTreeMap<String, u32>,
    lora: Option<lora_gpu::LayerLora>,
}

impl<'d> Layer<'d> {
    /// Stage 1 loads one whole layer, including its experts. This is bounded to
    /// about 15 GB restored BF16; full-model tier placement is a separate stage.
    pub fn load(device: &'d Device, path: &Path, index: usize, tokens: usize) -> Result<Self> {
        Self::load_inner(device, path, index, tokens, 0)
    }

    pub(super) fn load_decode(
        device: &'d Device,
        path: &Path,
        index: usize,
        context: usize,
    ) -> Result<Self> {
        Self::load_inner(device, path, index, 1, context)
    }

    fn load_inner(
        device: &'d Device,
        path: &Path,
        index: usize,
        tokens: usize,
        context: usize,
    ) -> Result<Self> {
        if index > 45 || !(1..=8192).contains(&tokens) {
            return Err(Error::InvalidShape("GLM layer/tokens out of range".into()));
        }
        let file = HytFile::open(path, MachineId::Main)?;
        Self::validate_weights(index, &file)?;
        let hadamard_block = super::experts::hadamard_block(&file)?;
        let stream = device.stream()?;
        let names: BTreeMap<_, _> = file
            .entries()
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name.as_str(), i))
            .collect();
        let mut uploaded = if context > 0 {
            let indices: Vec<_> = file
                .entries()
                .iter()
                .enumerate()
                .filter(|(_, e)| !e.name.starts_with("mlp.experts."))
                .map(|(i, _)| i)
                .collect();
            super::load::upload(device, &file, &indices)?
        } else {
            BTreeMap::new()
        };
        let mut weights = BTreeMap::new();
        let mut fp8 = BTreeSet::new();
        let mut nvfp4 = BTreeSet::new();
        let mut dense_formats = BTreeMap::new();
        for entry in file.entries() {
            if entry.name == "mlp.experts.hadamard_block" {
                continue;
            }
            if context > 0 && entry.name.starts_with("mlp.experts.") {
                continue;
            }
            if entry.name.ends_with("weight_scale_inv") {
                continue;
            }
            let source = if context > 0 {
                uploaded.remove(&entry.name).unwrap()
            } else {
                let mut data = vec![0; entry.bytes as usize];
                file.read_tensor(names[entry.name.as_str()], &mut data)?;
                if entry.quant == QuantSpec::Nvfp4 {
                    let dims = entry.shape.dims();
                    super::validate_nvfp4(&data, dims[0] as usize, dims[1] as usize)?;
                } else {
                    validate_raw(&data, entry.quant)?;
                }
                let mut buffer = device.alloc(data.len())?;
                stream.copy_h2d(&mut buffer, &data)?;
                buffer
            };
            let buffer = match entry.quant {
                QuantSpec::Nvfp4 if context > 0 => {
                    dense_formats.insert(entry.name.clone(), 3);
                    source
                }
                QuantSpec::PerTensor(DType::F8E4M3) if context > 0 => {
                    dense_formats.insert(entry.name.clone(), 2);
                    source
                }
                QuantSpec::PerTensor(DType::F8E4M3) => {
                    let dims = entry.shape.dims();
                    let mut out = device.alloc(dims[0] as usize * dims[1] as usize * 2)?;
                    check(
                        unsafe {
                            g::hayate_glm_dense_unpack(
                                out.as_mut_ptr(),
                                g::DenseWeight {
                                    data: source.as_ptr(),
                                    scale: std::ptr::null(),
                                    format: 2,
                                },
                                dims[0] as u32,
                                dims[1] as u32,
                                stream.as_raw(),
                            )
                        },
                        "glm oracle FP8 restore",
                    )?;
                    stream.synchronize()?;
                    out
                }
                QuantSpec::Nvfp4 => {
                    let dims = entry.shape.dims();
                    let mut out = device.alloc(dims[0] as usize * dims[1] as usize * 2)?;
                    check(
                        unsafe {
                            g::hayate_glm_nvfp4_restore(
                                out.as_mut_ptr(),
                                source.as_ptr(),
                                dims[0] as u32,
                                dims[1] as u32,
                                stream.as_raw(),
                            )
                        },
                        "glm oracle NVFP4 restore",
                    )?;
                    stream.synchronize()?;
                    nvfp4.insert(entry.name.clone());
                    out
                }
                QuantSpec::Plain(DType::Bf16 | DType::F32) => source,
                QuantSpec::Plain(DType::F8E4M3) => {
                    if entry.shape.dims().len() != 2 {
                        return Err(Error::Weights("GLM FP8 matrix rank".into()));
                    }
                    let n = entry.shape.dims()[0] as usize;
                    let k = entry.shape.dims()[1] as usize;
                    let scale_name = entry
                        .name
                        .strip_suffix("weight")
                        .ok_or_else(|| Error::Weights("GLM FP8 weight name".into()))?
                        .to_owned()
                        + "weight_scale_inv";
                    let i = *names
                        .get(scale_name.as_str())
                        .ok_or_else(|| Error::Weights(format!("missing {scale_name}")))?;
                    let e = &file.entries()[i];
                    if e.shape.dims() != [n.div_ceil(128) as u64, k.div_ceil(128) as u64]
                        || e.quant != QuantSpec::Plain(DType::F32)
                    {
                        return Err(Error::Weights("GLM FP8 scale shape/dtype".into()));
                    }
                    let scale = if context > 0 {
                        uploaded.remove(&scale_name).unwrap()
                    } else {
                        let mut raw = vec![0; e.bytes as usize];
                        file.read_tensor(i, &mut raw)?;
                        validate_raw(&raw, e.quant)?;
                        let mut scale = device.alloc(raw.len())?;
                        stream.copy_h2d(&mut scale, &raw)?;
                        scale
                    };
                    if context > 0 {
                        weights.insert(scale_name, scale);
                        fp8.insert(entry.name.clone());
                        weights.insert(entry.name.clone(), source);
                        continue;
                    }
                    let mut out = device.alloc(n * k * 2)?;
                    check(
                        unsafe {
                            g::hayate_glm_restore(
                                out.as_mut_ptr(),
                                source.as_ptr(),
                                scale.as_ptr(),
                                n as u32,
                                k as u32,
                                stream.as_raw(),
                            )
                        },
                        "glm restore",
                    )?;
                    stream.synchronize()?;
                    out
                }
                _ => {
                    return Err(Error::Weights(format!(
                        "GLM unsupported format: {}",
                        entry.name
                    )));
                }
            };
            weights.insert(entry.name.clone(), buffer);
        }
        let done = stream.event()?;
        let mut layer = Self {
            device,
            stream,
            index,
            tokens,
            weights,
            buffers: BTreeMap::new(),
            fp8,
            nvfp4,
            hadamard_block,
            lowbit: super::experts::is_mixed(&file),
            exl3: super::experts::exl3_bitrates(&file)?,
            pcie_bytes: 0,
            position: 0,
            context,
            latent: None,
            tail_keys: vec![0; 4 * 128 * 2],
            tail_gates: vec![0; 4 * 128 * 2],
            done,
            reference: false,
            graphs: true,
            before_graph: None,
            moe_graph: None,
            warmed: false,
            profile: None,
            asynchronous: None,
            async_active: false,
            route_host: device.alloc_pinned(TOP_K * 4)?,
            image_host: device.alloc_pinned(TOP_K * 8)?,
            cpu_input: device.alloc_pinned(HIDDEN * 2)?,
            cpu_output: device.alloc_pinned(TOP_K * HIDDEN * 2)?,
            dense_formats,
            verification: None,
            lora: None,
            prediction_stream: device.stream()?,
            prediction_host: device.alloc_pinned(TOP_K * 4)?,
            routing_ready: device.stream()?.event()?,
        };
        layer.allocate()?;
        if layer
            .dense_formats
            .contains_key("self_attn.kv_b_proj.weight")
            || layer.fp8.contains("self_attn.kv_b_proj.weight")
        {
            layer
                .buffers
                .insert("dense.scratch".into(), device.alloc(32768 * 512 * 2)?);
        }
        if context > 0 && index >= 3 {
            for (name, bytes) in [
                ("decode.images", TOP_K * 8),
                // LOWBIT layers: the ExpertFormat of each chosen expert.
                ("decode.formats", TOP_K * 8),
                ("prediction.logits", EXPERTS * 4),
                ("prediction.ids", TOP_K * 4),
                ("prediction.weights", TOP_K * 4),
                ("decode.product", 9 * 2048 * 2),
                ("decode.expert", 9 * HIDDEN * 2),
            ] {
                layer.buffers.insert(name.into(), device.alloc(bytes)?);
            }
        }
        if hadamard_block != 0 {
            layer
                .buffers
                .insert("rotated_input".into(), device.alloc(tokens * HIDDEN * 2)?);
        }
        if context > 0 {
            if index % 4 != 3 && index != 45 {
                for name in ["state", "hq", "hk", "hv"] {
                    if name != "state" {
                        layer
                            .buffers
                            .insert(name.into(), device.alloc(3 * 8192 * 2)?);
                    }
                    let zeros = vec![0; layer.buffers[name].bytes()];
                    layer
                        .stream
                        .copy_h2d(layer.buffers.get_mut(name).unwrap(), &zeros)?;
                }
            } else {
                for (name, bytes) in [
                    ("pools", context.div_ceil(4) * 128 * 2),
                    ("scores", context.div_ceil(4) * 4),
                    ("selected", 2051 * 4),
                    ("pool_keys", 4 * 128 * 2),
                    ("pool_gates", 4 * 128 * 2),
                    ("latent_cache", context * 512 * 2),
                    ("mla.wkt", 64 * 512 * 256 * 2),
                    ("mla.wv", 64 * 256 * 512 * 2),
                    ("mla.qabs", 64 * 4 * 512 * 2),
                    ("mla.olat", 64 * 4 * 512 * 2),
                    ("mla.part", MLA_SPLITS * 64 * 4 * 512 * 4),
                    ("mla.stats", MLA_SPLITS * 64 * 4 * 2 * 4),
                ] {
                    layer.buffers.insert(name.into(), device.alloc(bytes)?);
                }
                layer.prepare_mla()?;
            }
        }

        Ok(layer)
    }

    /// Absorbed-attention weights: W_K transposed per head and W_V per head
    /// (BF16, from kv_b_proj after any FP8 restore).
    fn prepare_mla(&mut self) -> Result<()> {
        let weight = "self_attn.kv_b_proj.weight";
        let kvb = if self.dense_formats.contains_key(weight) || self.fp8.contains(weight) {
            check(
                unsafe {
                    g::hayate_glm_dense_unpack(
                        self.p("dense.scratch"),
                        self.dense_weight(weight),
                        32768,
                        512,
                        self.stream.as_raw(),
                    )
                },
                "glm MLA kv_b restore",
            )?;
            self.p("dense.scratch")
        } else {
            self.w(weight)
        };
        check(
            unsafe {
                g::hayate_glm_mla_prepare(
                    self.p("mla.wkt"),
                    self.p("mla.wv"),
                    kvb,
                    self.stream.as_raw(),
                )
            },
            "glm MLA prepare",
        )?;
        self.stream.synchronize()
    }
    /// Device latent cache append for `rows` rows at the current position.
    fn latent_append(&self, latent: Ptr, rows: usize) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_copy(
                    self.p("latent_cache")
                        .cast::<u8>()
                        .add(self.position * 1024)
                        .cast(),
                    latent,
                    rows * 1024,
                    self.stream.as_raw(),
                )
            },
            "glm latent cache append",
        )
    }
    /// Absorbed sparse attention for 1..=4 decode rows (selection already in
    /// `selected`, rows at stride 2051): q -> attn_core.
    fn mla_decode(&self, q: Ptr, selected: Ptr, out: Ptr, rows: usize) -> Result<()> {
        self.mla_decode_on(self.stream.as_raw(), q, selected, out, rows)
    }
    fn mla_decode_on(
        &self,
        st: hayate_kernels::cuda::ffi::Stream,
        q: Ptr,
        selected: Ptr,
        out: Ptr,
        rows: usize,
    ) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_mla_absorb_q(
                    self.p("mla.qabs"),
                    q,
                    self.p("mla.wkt"),
                    rows as u32,
                    st,
                )
            },
            "glm MLA absorb q",
        )?;
        check(
            unsafe {
                g::hayate_glm_mla_attention(
                    self.p("mla.olat"),
                    self.p("mla.part"),
                    self.p("mla.stats"),
                    self.p("mla.qabs"),
                    self.p("latent_cache"),
                    selected,
                    rows as u32,
                    MLA_SPLITS as u32,
                    st,
                )
            },
            "glm MLA attention",
        )?;
        check(
            unsafe {
                g::hayate_glm_mla_absorb_out(
                    out,
                    self.p("mla.olat"),
                    self.p("mla.wv"),
                    rows as u32,
                    st,
                )
            },
            "glm MLA absorb out",
        )
    }
    fn allocate(&mut self) -> Result<()> {
        let t = self.tokens;
        for (name, count, bytes) in [
            ("input", t * HC * HIDDEN, 2),
            ("attn_map", t * 24, 4),
            ("ffn_map", t * 24, 4),
            ("attn_collapsed", t * HIDDEN, 2),
            ("attn_norm", t * HIDDEN, 2),
            ("attention", t * HIDDEN, 2),
            ("attention_residual", t * HC * HIDDEN, 2),
            ("ffn_collapsed", t * HIDDEN, 2),
            ("ffn_norm", t * HIDDEN, 2),
            ("mlp", t * HIDDEN, 2),
            ("output", t * HC * HIDDEN, 2),
            ("gate", t * 12288, 2),
            ("up", t * 12288, 2),
            ("product", t * 12288, 2),
            ("expert", t * HIDDEN, 2),
            ("router_logits", t * EXPERTS, 4),
            ("route_weights", t * TOP_K, 4),
            ("route_ids", t * TOP_K, 4),
        ] {
            self.buffers
                .insert(name.into(), self.device.alloc(count * bytes)?);
        }
        let shapes: Vec<(&str, usize, usize)> = if self.index % 4 != 3 && self.index != 45 {
            vec![
                ("q", t * 8192, 2),
                ("k", t * 8192, 2),
                ("v", t * 8192, 2),
                ("cq", t * 8192, 2),
                ("ck", t * 8192, 2),
                ("cv", t * 8192, 2),
                ("fa", t * 128, 2),
                ("fb", t * 8192, 2),
                ("decay", t * 8192, 4),
                ("beta", t * 64, 2),
                ("ga", t * 128, 2),
                ("gb", t * 8192, 2),
                ("kda", t * 8192, 2),
                ("state", 64 * 128 * 128, 4),
                ("gated", t * 8192, 2),
            ]
        } else {
            vec![
                ("qa", t * 1536, 2),
                ("qr", t * 1536, 2),
                ("q", t * 16384, 2),
                ("kva", t * 512, 2),
                ("kvr", t * 512, 2),
                ("kv", t * 32768, 2),
                ("k", t * 16384, 2),
                ("v", t * 16384, 2),
                ("iq", t * 4096, 2),
                ("ik", t * 128, 2),
                ("ikn", t * 128, 2),
                ("ig", t * 128, 2),
                ("iw", t * 32, 2),
                ("pools", t.div_ceil(4) * 128, 2),
                ("scores", (t * (t / 4)).max(1), 4),
                ("selected", t * 2051, 4),
                ("probabilities", t * t * 64, 2),
                ("attn_core", t * 16384, 2),
            ]
        };
        for (name, count, bytes) in shapes {
            self.buffers
                .insert(name.into(), self.device.alloc(count * bytes)?);
        }
        Ok(())
    }

    fn validate_weights(index: usize, file: &HytFile) -> Result<()> {
        let mut shapes: Vec<(String, Vec<u64>, DType)> = Vec::new();
        let mut add = |name: &str, shape: &[u64], dtype: DType| {
            shapes.push((name.into(), shape.to_vec(), dtype))
        };
        for site in if index == 45 {
            &[][..]
        } else {
            &["attn", "ffn"][..]
        } {
            add(&format!("hc_{site}_fn"), &[24, 16384], DType::Bf16);
            add(&format!("hc_{site}_base"), &[24], DType::F32);
            add(&format!("hc_{site}_scale"), &[3], DType::F32);
        }
        if index == 45 {
            for n in ["enorm.weight", "hnorm.weight", "shared_head.norm.weight"] {
                add(n, &[4096], DType::Bf16);
            }
            add("eh_proj.weight", &[4096, 8192], DType::Bf16);
        }
        for n in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            add(n, &[4096], DType::Bf16);
        }
        let mut linear =
            |name: &str, n: u64, k: u64| shapes.push((name.into(), vec![n, k], DType::Bf16));
        if index % 4 != 3 && index != 45 {
            for part in ["q", "k", "v"] {
                linear(&format!("self_attn.{part}_proj.weight"), 8192, 4096);
            }
            for part in ["f", "g"] {
                linear(&format!("self_attn.{part}_a_proj.weight"), 128, 4096);
                linear(&format!("self_attn.{part}_b_proj.weight"), 8192, 128);
            }
            linear("self_attn.b_proj.weight", 64, 4096);
            linear("self_attn.o_proj.weight", 4096, 8192);
            for part in ["q", "k", "v"] {
                shapes.push((
                    format!("self_attn.{part}_conv1d.weight"),
                    vec![8192, 1, 4],
                    DType::Bf16,
                ));
            }
            for (name, dims, ty) in [
                ("A_log", vec![64], DType::F32),
                ("dt_bias", vec![8192], DType::F32),
                ("o_norm.weight", vec![128], DType::Bf16),
            ] {
                shapes.push((format!("self_attn.{name}"), dims, ty));
            }
        } else {
            for (n, o, i) in [
                ("q_a_proj.weight", 1536, 4096),
                ("q_b_proj.weight", 16384, 1536),
                ("kv_a_proj_with_mqa.weight", 512, 4096),
                ("kv_b_proj.weight", 32768, 512),
                ("o_proj.weight", 4096, 16384),
                ("indexer.wq_b.weight", 4096, 1536),
                ("indexer.wk.weight", 128, 4096),
                ("indexer.weights_proj.weight", 32, 4096),
                ("indexer.index_kpool_compress_gate", 128, 4096),
            ] {
                linear(&format!("self_attn.{n}"), o, i);
            }
            for (n, dim) in [
                ("q_a_layernorm.weight", 1536),
                ("kv_a_layernorm.weight", 512),
                ("indexer.k_norm.weight", 128),
                ("indexer.k_norm.bias", 128),
            ] {
                shapes.push((format!("self_attn.{n}"), vec![dim], DType::Bf16));
            }
            shapes.push((
                "self_attn.indexer.index_kpool_compress_ape".into(),
                vec![4, 128],
                DType::Bf16,
            ));
        }
        let mlps = if index < 3 {
            vec![("mlp".to_owned(), 12288)]
        } else {
            shapes.push(("mlp.gate.weight".into(), vec![288, 4096], DType::Bf16));
            shapes.push((
                "mlp.gate.e_score_correction_bias".into(),
                vec![288],
                DType::F32,
            ));
            (0..EXPERTS)
                .map(|i| (format!("mlp.experts.{i}"), 2048))
                .chain(std::iter::once(("mlp.shared_experts".into(), 2048)))
                .collect()
        };
        for (stem, dim) in mlps {
            for (part, n, k) in [("gate", dim, 4096), ("up", dim, 4096), ("down", 4096, dim)] {
                shapes.push((
                    format!("{stem}.{part}_proj.weight"),
                    vec![n, k],
                    DType::Bf16,
                ));
            }
        }
        for (name, dims, dtype) in shapes {
            let entry = file
                .entries()
                .iter()
                .find(|e| e.name == name)
                .ok_or_else(|| Error::Weights(format!("missing {name}")))?;
            let valid = entry.quant == QuantSpec::Plain(dtype)
                || (dtype == DType::Bf16 && entry.quant == QuantSpec::Plain(DType::F8E4M3))
                || (name.starts_with("mlp.experts.")
                    && matches!(
                        entry.quant,
                        QuantSpec::Nvfp4 | QuantSpec::Lowbit { .. } | QuantSpec::Exl3 { .. }
                    ))
                || (dtype == DType::Bf16
                    && dims.len() == 2
                    && (name.ends_with("_proj.weight")
                        || matches!(
                            name.as_str(),
                            "self_attn.kv_a_proj_with_mqa.weight"
                                | "self_attn.indexer.wq_b.weight"
                                | "self_attn.indexer.wk.weight"
                                | "self_attn.indexer.weights_proj.weight"
                                | "self_attn.indexer.index_kpool_compress_gate"
                        ))
                    && matches!(
                        entry.quant,
                        QuantSpec::Nvfp4 | QuantSpec::PerTensor(DType::F8E4M3)
                    ));
            if entry.shape.dims() != dims || !valid {
                return Err(Error::Weights(format!("GLM shape/dtype mismatch: {name}")));
            }
        }
        Ok(())
    }

    fn p(&self, name: &str) -> Ptr {
        self.buffers[name].as_ptr().cast_mut()
    }
    pub(super) fn w(&self, name: &str) -> Ptr {
        self.weights[name].as_ptr().cast_mut()
    }
    pub(super) fn dense_weight(&self, name: &str) -> g::DenseWeight {
        let format = *self
            .dense_formats
            .get(name)
            .unwrap_or(&u32::from(self.fp8.contains(name)));
        g::DenseWeight {
            data: self.w(name),
            scale: if format == 1 {
                self.w(&(name.strip_suffix("weight").unwrap().to_owned() + "weight_scale_inv"))
            } else {
                std::ptr::null()
            },
            format,
        }
    }
    /// The base linear, then its LoRA side paths (if any).
    fn linear(&self, out: &str, input: &str, weight: &str, n: usize, k: usize) -> Result<()> {
        self.linear_base(out, input, weight, n, k)?;
        self.lora_linear(
            self.stream.as_raw(),
            weight,
            self.p(out),
            self.p(input),
            self.tokens,
        )
    }
    fn linear_base(&self, out: &str, input: &str, weight: &str, n: usize, k: usize) -> Result<()> {
        if self.context > 0 && self.tokens > 1 {
            return self.linear_ptrs(self.p(out), self.p(input), weight, self.tokens, n, k);
        }
        if let Some(v) = &self.verification
            && let Some(row) = v.row
            && weight.starts_with("self_attn.")
            && weight != "self_attn.kv_b_proj.weight"
        {
            return if weight == "self_attn.o_proj.weight" {
                self.copy(v.row_ptr(input, row, k * 2), self.p(input), k * 2)
            } else {
                self.copy(self.p(out), v.row_ptr(out, row, n * 2), n * 2)
            };
        }
        if self.context > 0
            && self.tokens == 1
            && (!self.reference || self.dense_formats.contains_key(weight))
        {
            return check(
                unsafe {
                    g::hayate_glm_dense(
                        self.p(out),
                        self.p(input),
                        self.dense_weight(weight),
                        n as u32,
                        k as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm dense GEMV",
            );
        }
        if self.fp8.contains(weight) {
            return check(
                unsafe {
                    g::hayate_glm_fp8_gemv(
                        self.p(out),
                        self.p(input),
                        self.w(weight),
                        self.w(&(weight.strip_suffix("weight").unwrap().to_owned()
                            + "weight_scale_inv")),
                        n as u32,
                        k as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm FP8 GEMV",
            );
        }
        check(
            unsafe {
                hayate_text_linear(
                    self.p(out),
                    self.p(input),
                    self.w(weight),
                    self.tokens as u32,
                    n as u32,
                    k as u32,
                    self.stream.as_raw(),
                )
            },
            "glm linear",
        )
    }
    fn linear_ptrs(
        &self,
        out: Ptr,
        input: Ptr,
        weight: &str,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let packed = self.dense_formats.contains_key(weight) || self.fp8.contains(weight);
        let matrix = if packed {
            check(
                unsafe {
                    g::hayate_glm_dense_unpack(
                        self.p("dense.scratch"),
                        self.dense_weight(weight),
                        n as u32,
                        k as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm dense prefill restore",
            )?;
            self.p("dense.scratch")
        } else {
            self.w(weight)
        };
        check(
            unsafe {
                hayate_text_linear(
                    out,
                    input,
                    matrix,
                    rows as u32,
                    n as u32,
                    k as u32,
                    self.stream.as_raw(),
                )
            },
            "glm dense GEMM",
        )
    }
    fn norm(&self, out: &str, input: &str, weight: &str, cols: usize) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_norm(
                    self.p(out),
                    self.p(input),
                    self.w(weight),
                    std::ptr::null(),
                    self.tokens as u32,
                    cols as u32,
                    0,
                    self.stream.as_raw(),
                )
            },
            "glm rms",
        )
    }
    fn hc(&self, site: &str, map: &str, out: &str, input: &str) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_hc(
                    self.p(map),
                    self.p(out),
                    self.p(input),
                    self.w(&format!("hc_{site}_fn")),
                    self.w(&format!("hc_{site}_base")),
                    self.w(&format!("hc_{site}_scale")),
                    self.tokens as u32,
                    HIDDEN as u32,
                    self.stream.as_raw(),
                )
            },
            "glm hc",
        )
    }
    fn expand(&self, out: &str, input: &str, branch: &str, map: &str) -> Result<()> {
        if self.index == 45 {
            return check(
                unsafe {
                    g::hayate_glm_mtp_add(
                        self.p(out),
                        self.p(input),
                        self.p(branch),
                        self.stream.as_raw(),
                    )
                },
                "glm MTP residual",
            );
        }
        check(
            unsafe {
                g::hayate_glm_expand(
                    self.p(out),
                    self.p(input),
                    self.p(branch),
                    self.p(map),
                    self.tokens as u32,
                    HIDDEN as u32,
                    self.stream.as_raw(),
                )
            },
            "glm expand",
        )
    }
    fn mlp(&self, stem: &str, out: &str, dim: usize) -> Result<()> {
        let input = if self.hadamard_block != 0 && stem.starts_with("mlp.experts.") {
            "rotated_input"
        } else {
            "ffn_norm"
        };
        self.linear(
            "gate",
            input,
            &format!("{stem}.gate_proj.weight"),
            dim,
            HIDDEN,
        )?;
        self.linear("up", input, &format!("{stem}.up_proj.weight"), dim, HIDDEN)?;
        check(
            unsafe {
                g::hayate_glm_point(
                    self.p("product"),
                    self.p("gate"),
                    self.p("up"),
                    self.tokens * dim,
                    1,
                    self.stream.as_raw(),
                )
            },
            "glm swiglu",
        )?;
        if self.hadamard_block != 0
            && stem.starts_with("mlp.experts.")
            && self.nvfp4.contains(&format!("{stem}.down_proj.weight"))
        {
            self.rotate(self.p("product"), self.p("product"), self.tokens, dim)?;
        }
        self.linear(
            out,
            "product",
            &format!("{stem}.down_proj.weight"),
            HIDDEN,
            dim,
        )
    }

    fn rotate(&self, out: Ptr, input: Ptr, rows: usize, cols: usize) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_hadamard(
                    out,
                    input,
                    rows as u32,
                    cols as u32,
                    self.hadamard_block,
                    self.stream.as_raw(),
                )
            },
            "glm Hadamard rotation",
        )
    }

    fn kda(&self) -> Result<()> {
        let t = self.tokens as u32;
        let st = self.stream.as_raw();
        for part in ["q", "k", "v"] {
            self.linear(
                part,
                "attn_norm",
                &format!("self_attn.{part}_proj.weight"),
                8192,
                HIDDEN,
            )?;
            if self.context > 0 && self.tokens > 1 {
                check(
                    unsafe {
                        g::hayate_glm_conv_chunk(
                            self.p(&format!("c{part}")),
                            self.p(&format!("h{part}")),
                            self.p(part),
                            self.w(&format!("self_attn.{part}_conv1d.weight")),
                            t,
                            8192,
                            self.stream.as_raw(),
                        )
                    },
                    "glm chunk conv",
                )?;
            } else if self.context > 0 {
                check(
                    unsafe {
                        g::hayate_glm_conv_step(
                            self.p(&format!("c{part}")),
                            self.p(&format!("h{part}")),
                            self.p(part),
                            self.w(&format!("self_attn.{part}_conv1d.weight")),
                            8192,
                            st,
                        )
                    },
                    "glm conv step",
                )?;
            } else {
                check(
                    unsafe {
                        g::hayate_glm_conv(
                            self.p(&format!("c{part}")),
                            self.p(part),
                            self.w(&format!("self_attn.{part}_conv1d.weight")),
                            t,
                            8192,
                            st,
                        )
                    },
                    "glm conv",
                )?;
            }
        }
        self.linear("fa", "attn_norm", "self_attn.f_a_proj.weight", 128, HIDDEN)?;
        self.linear("fb", "fa", "self_attn.f_b_proj.weight", 8192, 128)?;
        check(
            unsafe {
                g::hayate_glm_decay(
                    self.p("decay"),
                    self.p("fb"),
                    self.w("self_attn.dt_bias"),
                    self.w("self_attn.A_log"),
                    t,
                    64,
                    128,
                    st,
                )
            },
            "glm decay",
        )?;
        self.linear("beta", "attn_norm", "self_attn.b_proj.weight", 64, HIDDEN)?;
        check(
            unsafe {
                g::hayate_glm_point(
                    self.p("beta"),
                    self.p("beta"),
                    std::ptr::null(),
                    self.tokens * 64,
                    0,
                    st,
                )
            },
            "glm beta",
        )?;
        if self.context > 0 && self.tokens == 1 && !self.reference {
            self.linear("ga", "attn_norm", "self_attn.g_a_proj.weight", 128, HIDDEN)?;
            self.linear("gb", "ga", "self_attn.g_b_proj.weight", 8192, 128)?;
            self.mark(10)?;
            check(
                unsafe {
                    g::hayate_glm_kda_gated(
                        self.p("kda"),
                        self.p("gated"),
                        self.p("state"),
                        self.p("cq"),
                        self.p("ck"),
                        self.p("cv"),
                        self.p("decay"),
                        self.p("beta"),
                        self.p("gb"),
                        self.w("self_attn.o_norm.weight"),
                        st,
                    )
                },
                "glm fused KDA gated norm",
            )?;
            self.mark(11)?;
        } else {
            if self.context > 0 && self.tokens > 1 {
                self.mark(8)?;
            }
            self.mark(10)?;
            if self.buffers.contains_key("kda_nq") {
                check(
                    unsafe {
                        g::hayate_glm_kda_chunk(
                            self.p("kda"),
                            self.p("state"),
                            self.p("kda_nq"),
                            self.p("kda_nk"),
                            self.p("cq"),
                            self.p("ck"),
                            self.p("cv"),
                            self.p("decay"),
                            self.p("beta"),
                            t,
                            64,
                            128,
                            st,
                        )
                    },
                    "glm kda",
                )?;
            } else {
                check(
                    unsafe {
                        g::hayate_glm_kda(
                            self.p("kda"),
                            self.p("state"),
                            self.p("cq"),
                            self.p("ck"),
                            self.p("cv"),
                            self.p("decay"),
                            self.p("beta"),
                            t,
                            64,
                            128,
                            st,
                        )
                    },
                    "glm kda",
                )?;
            }
            self.mark(11)?;
            if self.context > 0 && self.tokens > 1 {
                self.mark(9)?;
            }
            self.linear("ga", "attn_norm", "self_attn.g_a_proj.weight", 128, HIDDEN)?;
            self.linear("gb", "ga", "self_attn.g_b_proj.weight", 8192, 128)?;
            check(
                unsafe {
                    g::hayate_glm_gated(
                        self.p("gated"),
                        self.p("kda"),
                        self.p("gb"),
                        self.w("self_attn.o_norm.weight"),
                        t * 64,
                        128,
                        st,
                    )
                },
                "glm gated norm",
            )?;
        }
        self.linear(
            "attention",
            "gated",
            "self_attn.o_proj.weight",
            HIDDEN,
            8192,
        )
    }

    fn sparse(&self) -> Result<()> {
        let t = self.tokens as u32;
        let st = self.stream.as_raw();
        self.linear("qa", "attn_norm", "self_attn.q_a_proj.weight", 1536, HIDDEN)?;
        self.norm("qr", "qa", "self_attn.q_a_layernorm.weight", 1536)?;
        self.linear("q", "qr", "self_attn.q_b_proj.weight", 16384, 1536)?;
        self.linear(
            "kva",
            "attn_norm",
            "self_attn.kv_a_proj_with_mqa.weight",
            512,
            HIDDEN,
        )?;
        self.norm("kvr", "kva", "self_attn.kv_a_layernorm.weight", 512)?;
        self.linear("kv", "kvr", "self_attn.kv_b_proj.weight", 32768, 512)?;
        check(
            unsafe { g::hayate_glm_split(self.p("k"), self.p("v"), self.p("kv"), t, 64, 256, st) },
            "glm split kv",
        )?;
        self.linear("iq", "qr", "self_attn.indexer.wq_b.weight", 4096, 1536)?;
        self.linear(
            "ik",
            "attn_norm",
            "self_attn.indexer.wk.weight",
            128,
            HIDDEN,
        )?;
        check(
            unsafe {
                g::hayate_glm_norm(
                    self.p("ikn"),
                    self.p("ik"),
                    self.w("self_attn.indexer.k_norm.weight"),
                    self.w("self_attn.indexer.k_norm.bias"),
                    t,
                    128,
                    1,
                    st,
                )
            },
            "glm index layernorm",
        )?;
        self.linear(
            "ig",
            "attn_norm",
            "self_attn.indexer.index_kpool_compress_gate",
            128,
            HIDDEN,
        )?;
        self.linear(
            "iw",
            "attn_norm",
            "self_attn.indexer.weights_proj.weight",
            32,
            HIDDEN,
        )?;
        check(
            unsafe {
                g::hayate_glm_pool(
                    self.p("pools"),
                    self.p("ikn"),
                    self.p("ig"),
                    self.w("self_attn.indexer.index_kpool_compress_ape"),
                    t / 4,
                    128,
                    st,
                )
            },
            "glm pools",
        )?;
        check(
            unsafe {
                g::hayate_glm_select(
                    self.p("selected"),
                    self.p("scores"),
                    self.p("iq"),
                    self.p("pools"),
                    self.p("iw"),
                    t,
                    32,
                    128,
                    2048,
                    st,
                )
            },
            "glm select",
        )?;
        check(
            unsafe {
                g::hayate_glm_attention(
                    self.p("attn_core"),
                    self.p("probabilities"),
                    self.p("q"),
                    self.p("k"),
                    self.p("v"),
                    self.p("selected"),
                    t,
                    64,
                    256,
                    2051,
                    st,
                )
            },
            "glm sparse attention",
        )?;
        self.linear(
            "attention",
            "attn_core",
            "self_attn.o_proj.weight",
            HIDDEN,
            16384,
        )
    }

    /// Session state of this layer at `self.position`. `full` adds the token
    /// history (sparse layers: latent KV prefix and index pools); without it
    /// only the fixed-size recurrent state is copied (KDA state, conv
    /// histories `hq`/`hk`/`hv`, index pool tails `pool_keys`/`pool_gates`).
    pub(in crate::glm53) fn save_session(&self, full: bool) -> Result<LayerSession> {
        self.stream.synchronize()?;
        let mut buffers = Vec::new();
        for name in SESSION_BUFFERS {
            if let Some(buffer) = self.buffers.get(name) {
                let bytes = if name == "pools" {
                    if !full {
                        continue;
                    }
                    self.position.div_ceil(4) * 128 * 2
                } else {
                    buffer.bytes()
                };
                let mut raw = vec![0; bytes];
                if bytes > 0 {
                    self.stream.copy_d2h(&mut raw, buffer)?;
                }
                buffers.push((name, raw));
            }
        }
        let mut latent = Vec::new();
        if full
            && self.position > 0
            && let Some(cache) = self.buffers.get("latent_cache")
        {
            latent = vec![0; self.position * 1024];
            self.stream.copy_d2h(&mut latent, cache)?;
        }
        Ok(LayerSession {
            position: self.position,
            buffers,
            latent,
            tail_keys: self.tail_keys.clone(),
            tail_gates: self.tail_gates.clone(),
        })
    }
    pub(in crate::glm53) fn restore_session(&mut self, session: &LayerSession) -> Result<()> {
        if session.position > self.context {
            return Err(Error::InvalidShape("GLM session exceeds context".into()));
        }
        self.stream.synchronize()?;
        for (name, raw) in &session.buffers {
            let buffer = self
                .buffers
                .get_mut(*name)
                .ok_or_else(|| Error::InvalidPlan(format!("GLM session buffer {name}")))?;
            if raw.len() > buffer.bytes() {
                return Err(Error::InvalidShape(format!("GLM session buffer {name}")));
            }
            if !raw.is_empty() {
                self.stream.copy_h2d(buffer, raw)?;
            }
        }
        if !session.latent.is_empty() {
            let cache = self
                .buffers
                .get_mut("latent_cache")
                .ok_or_else(|| Error::InvalidPlan("GLM session latent".into()))?;
            self.stream.copy_h2d(cache, &session.latent)?;
            if let Some(host) = self.latent.as_mut() {
                host.as_mut_slice()[..session.latent.len()].copy_from_slice(&session.latent);
            }
        }
        self.tail_keys.copy_from_slice(&session.tail_keys);
        self.tail_gates.copy_from_slice(&session.tail_gates);
        self.position = session.position;
        Ok(())
    }
    pub(super) fn reset(&mut self) -> Result<()> {
        self.stream.synchronize()?;
        self.position = 0;
        self.tail_keys.fill(0);
        self.tail_gates.fill(0);
        if self.index % 4 != 3 && self.index != 45 {
            for name in ["state", "hq", "hk", "hv"] {
                check(
                    unsafe {
                        g::hayate_glm_copy(
                            self.p(name),
                            std::ptr::null(),
                            self.buffers[name].bytes(),
                            self.stream.as_raw(),
                        )
                    },
                    "glm reset state",
                )?;
            }
        }
        self.stream.synchronize()
    }

    pub(in crate::glm53) fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
    pub(super) fn weight_bytes(&self) -> usize {
        self.weights
            .values()
            .map(DeviceBuffer::bytes)
            .sum::<usize>()
            + self.lora_bytes()
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.weights
            .values()
            .chain(self.buffers.values())
            .map(DeviceBuffer::bytes)
            .sum::<usize>()
            + self.verification.as_ref().map_or(0, |v| v.bytes())
            + self.lora_bytes()
    }

    pub(super) fn enable_profile(&mut self) -> Result<()> {
        if !self.async_active {
            self.stream.synchronize()?;
        }
        self.before_graph = None;
        self.moe_graph = None;
        self.graphs = false;
        self.profile = Some(super::profile::Profile::new(&self.stream, 14)?);
        Ok(())
    }
    fn mark(&self, index: usize) -> Result<()> {
        if let Some(profile) = &self.profile {
            profile.mark(index, &self.stream)?;
        }
        Ok(())
    }
    fn download(&self, dst: &mut [u8], name: &str) -> Result<()> {
        if let Some(profile) = &self.profile {
            profile.download(&self.stream, dst, &self.buffers[name])
        } else {
            self.stream.copy_d2h(dst, &self.buffers[name])
        }
    }
    pub(super) fn report_profile(&self, position: usize) -> Result<()> {
        let Some(p) = &self.profile else {
            return Ok(());
        };
        let waits = *p.waits.borrow();
        let kind = if self.index % 4 == 3 || self.index == 45 {
            "sparse"
        } else {
            "kda"
        };
        eprintln!(
            "glm_profile position={position} layer={} kind={kind} mhc_ms={:.6} attention_ms={:.6} dense_ffn_ms={:.6} routing_ms={:.6} transfer_ms={:.6} gate_up_ms={:.6} down_ms={:.6} reduce_ms={:.6} cpu_tail_wait_ms={:.6} cpu_merge_ms={:.6} d2h_count={} d2h_sync_count={} d2h_wait_ms={:.6}",
            self.index,
            p.ms(0, 1)? + p.ms(2, 3)? + p.ms(8, 9)?,
            p.ms(1, 2)?,
            if self.index < 3 { p.ms(3, 8)? } else { 0. },
            if self.index >= 3 { p.ms(3, 4)? } else { 0. },
            if self.index >= 3 { p.ms(4, 5)? } else { 0. },
            if self.index >= 3 { p.ms(5, 6)? } else { 0. },
            if self.index >= 3 { p.ms(6, 7)? } else { 0. },
            if self.index >= 3 { p.ms(7, 8)? } else { 0. },
            if self.async_active && self.index >= 3 {
                p.ms(7, 12)?
            } else {
                0.
            },
            if self.async_active && self.index >= 3 {
                p.ms(12, 13)?
            } else {
                0.
            },
            waits.d2h_count,
            waits.d2h_sync_count,
            waits.d2h_ms
        );
        Ok(())
    }
    pub(super) fn step_device(
        &mut self,
        input: Ptr,
        ready: &Event,
        cache: &mut super::experts::Experts,
    ) -> Result<Ptr> {
        cache.check_worker_error()?;
        if self.position >= self.context {
            return Err(Error::InvalidShape("GLM decode context".into()));
        }
        if let Some(p) = &self.profile {
            p.reset();
        }
        self.stream.wait_event(ready)?;
        check(
            unsafe {
                g::hayate_glm_copy(
                    self.p("input"),
                    input,
                    HC * HIDDEN * 2,
                    self.stream.as_raw(),
                )
            },
            "glm layer input D2D",
        )?;
        self.decode(cache)?;
        self.done.record(&self.stream)?;
        self.position += 1;
        Ok(self.p("output"))
    }
    pub(super) fn prefetch_from(
        &mut self,
        previous: &Layer<'_>,
        cache: &mut super::experts::Experts,
    ) -> Result<f64> {
        cache.check_worker_error()?;
        self.prediction_stream.wait_event(&previous.routing_ready)?;
        check(
            unsafe {
                g::hayate_glm_route_decode(
                    self.p("prediction.logits"),
                    self.p("prediction.weights"),
                    self.p("prediction.ids"),
                    previous.p("ffn_norm"),
                    self.w("mlp.gate.weight"),
                    self.w("mlp.gate.e_score_correction_bias"),
                    self.prediction_stream.as_raw(),
                )
            },
            "glm approximate prefetch router",
        )?;
        unsafe {
            self.prediction_stream
                .copy_d2h_async(&mut self.prediction_host, &self.buffers["prediction.ids"])
        }?;
        let wait = std::time::Instant::now();
        self.prediction_stream.synchronize()?;
        let prediction_ms = wait.elapsed().as_secs_f64() * 1000.;
        let ids: Vec<_> = self
            .prediction_host
            .as_slice()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        if ids.iter().any(|&id| id >= EXPERTS as u32) {
            return Err(Error::Weights("GLM prefetch ID".into()));
        }
        cache.prefetch(
            self.index,
            &ids.iter().map(|&id| id as u16).collect::<Vec<_>>(),
        )?;
        Ok(prediction_ms)
    }
    pub(super) fn record_prediction_wait(&self, wait: f64) {
        if let Some(p) = &self.profile {
            let mut waits = p.waits.borrow_mut();
            waits.d2h_count += 1;
            waits.d2h_sync_count += 1;
            waits.d2h_ms += wait;
        }
    }
    fn decode(&mut self, cache: &mut super::experts::Experts) -> Result<()> {
        if self.reference {
            return self.forward(Some(cache));
        }
        let graphs = self.graphs && self.profile.is_none();
        // Sparse attention still gathers latent KV from pinned RAM. KDA has
        // fixed addresses and updates state only when the captured graph runs.
        if graphs && self.index % 4 != 3 && self.index != 45 && self.warmed {
            if self.before_graph.is_none() {
                if !self.async_active {
                    self.stream.synchronize()?;
                }
                self.before_graph = Some(unsafe {
                    Graph::capture(self.stream.as_raw(), || self.attention_ffn_norm())
                }?);
            }
            unsafe {
                self.before_graph
                    .as_ref()
                    .unwrap()
                    .launch(self.stream.as_raw())
            }?;
        } else {
            self.attention_ffn_norm()?;
        }
        if self.index < 3 {
            self.mlp("mlp", "mlp", 12288)?;
            self.mark(8)?;
            self.expand("output", "attention_residual", "mlp", "ffn_map")?;
        } else {
            check(
                unsafe {
                    g::hayate_glm_route_decode(
                        self.p("router_logits"),
                        self.p("route_weights"),
                        self.p("route_ids"),
                        self.p("ffn_norm"),
                        self.w("mlp.gate.weight"),
                        self.w("mlp.gate.e_score_correction_bias"),
                        self.stream.as_raw(),
                    )
                },
                "glm decode router",
            )?;
            self.mark(4)?;
            self.routing_ready.record(&self.stream)?;
            if self.async_active {
                self.decode_async(cache)?;
                self.mark(9)?;
                self.warmed = true;
                return Ok(());
            }
            // Pinned route copy overlaps input rotation. Only the CPU cache
            // resolver consumes IDs; top-k and expert reduction stay on GPU.
            unsafe {
                self.stream
                    .copy_d2h_async(&mut self.route_host, &self.buffers["route_ids"])
            }?;
            if self.hadamard_block != 0 {
                self.rotate(self.p("rotated_input"), self.p("ffn_norm"), 1, HIDDEN)?;
            }
            if cache.cpu_enabled() {
                let name = if self.hadamard_block != 0 {
                    "rotated_input"
                } else {
                    "ffn_norm"
                };
                // Same stream as the router/rotation; this synchronization also
                // retires the previous token's CPU result upload for this layer.
                unsafe {
                    self.stream
                        .copy_d2h_async(&mut self.cpu_input, &self.buffers[name])
                }?;
            }
            let wait = std::time::Instant::now();
            self.stream.synchronize()?;
            if let Some(p) = &self.profile {
                p.add_wait(wait);
            }
            let ids: Vec<_> = self
                .route_host
                .as_slice()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            if ids.iter().any(|&e| e >= EXPERTS as u32) {
                return Err(Error::Weights("GLM routed expert ID".into()));
            }
            let ids: Vec<_> = ids.into_iter().map(|e| e as u16).collect();
            cache.route(self.index, &ids)?;
            let cpu_images = cache.cpu_images(self.index, &ids);
            let gpu_ids: Vec<_> = ids
                .iter()
                .zip(&cpu_images)
                .filter_map(|(&id, cpu)| cpu.is_none().then_some(id))
                .collect();
            let mut gpu_images = cache
                .decode_images(self.index, &gpu_ids, &self.stream)?
                .into_iter();
            let images: Vec<_> = cpu_images
                .iter()
                .map(|cpu| {
                    if cpu.is_some() {
                        std::ptr::null()
                    } else {
                        gpu_images.next().unwrap()
                    }
                })
                .collect();
            for (dst, image) in self
                .image_host
                .as_mut_slice()
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(images)
            {
                *dst = (image as u64).to_le_bytes();
            }
            unsafe {
                self.stream.copy_h2d_async(
                    self.buffers.get_mut("decode.images").unwrap(),
                    &self.image_host,
                )
            }?;
            // This marks submission completion, not a host synchronization.
            self.mark(5)?;
            let (nv, down_fp8) = cache.format(self.index);
            if graphs && self.warmed {
                if self.moe_graph.is_none() {
                    self.stream.synchronize()?;
                    self.moe_graph = Some(unsafe {
                        Graph::capture(self.stream.as_raw(), || self.decode_moe(nv, down_fp8))
                    }?);
                }
                unsafe {
                    self.moe_graph
                        .as_ref()
                        .unwrap()
                        .launch(self.stream.as_raw())
                }?;
            } else {
                self.decode_moe(nv, down_fp8)?;
            }
            // GPU cache hits/shared expert are already running. CPU workers
            // read pinned misses in place; only their BF16 outputs cross PCIe.
            cache.compute_cpu(
                self.index,
                &cpu_images,
                &ids,
                self.cpu_input.as_slice(),
                self.cpu_output.as_mut_slice(),
                self.hadamard_block,
            )?;
            for (i, image) in cpu_images.iter().enumerate() {
                if image.is_some() {
                    let offset = i * HIDDEN * 2;
                    // Source is owned by this layer until its next router wait;
                    // destination is consumed below on the same stream.
                    unsafe {
                        self.stream.copy_h2d_at_async(
                            self.buffers.get_mut("decode.expert").unwrap(),
                            offset,
                            self.cpu_output.range(offset..offset + HIDDEN * 2)?,
                        )
                    }?;
                }
            }
            self.decode_moe_reduce()?;
        }
        self.mark(9)?;
        self.warmed = true;
        Ok(())
    }
    fn publish_host(&self, out: Ptr, input: Ptr, bytes: usize) -> Result<()> {
        check(
            unsafe { g::hayate_glm_async_publish(out, input, bytes as u32, self.stream.as_raw()) },
            "glm GPU mailbox publication",
        )
    }
    fn decode_async(&mut self, cache: &super::experts::Experts) -> Result<()> {
        use super::async_decode::{IDS, INPUT, OUTPUT};
        if self.hadamard_block != 0 {
            self.rotate(self.p("rotated_input"), self.p("ffn_norm"), 1, HIDDEN)?;
        }
        let input = self.p(if self.hadamard_block != 0 {
            "rotated_input"
        } else {
            "ffn_norm"
        });
        let format = cache.format(self.index);
        let sequence = self.asynchronous.as_mut().unwrap().submit(
            cache.asynchronous.as_ref().unwrap(),
            self.index,
            1,
            format,
            self.hadamard_block,
        )?;
        let a = self.asynchronous.as_ref().unwrap();
        self.publish_host(a.mailbox.device(IDS), self.p("route_ids"), TOP_K * 4)?;
        self.publish_host(a.mailbox.device(INPUT), input, HIDDEN * 2)?;
        a.signal(&self.stream, sequence)?;
        check(
            unsafe {
                g::hayate_glm_async_images(
                    self.p("decode.images"),
                    self.p("route_ids"),
                    a.table.device_ptr(),
                    1,
                    self.stream.as_raw(),
                )
            },
            "glm async images",
        )?;
        self.mark(5)?;
        if self.graphs && self.profile.is_none() && self.warmed {
            if self.moe_graph.is_none() {
                self.moe_graph = Some(unsafe {
                    Graph::capture(self.stream.as_raw(), || self.decode_moe(format.0, format.1))
                }?);
            }
            unsafe {
                self.moe_graph
                    .as_ref()
                    .unwrap()
                    .launch(self.stream.as_raw())
            }?;
        } else {
            self.decode_moe(format.0, format.1)?;
        }
        a.wait(&self.stream, sequence)?;
        self.mark(12)?;
        check(
            unsafe {
                g::hayate_glm_async_merge(
                    self.p("decode.expert"),
                    a.mailbox.device(OUTPUT),
                    self.p("route_ids"),
                    a.table.device_ptr(),
                    1,
                    self.stream.as_raw(),
                )
            },
            "glm async CPU merge",
        )?;
        self.mark(13)?;
        self.decode_moe_reduce()
    }
    fn decode_moe(&self, nv: bool, down_fp8: bool) -> Result<()> {
        let st = self.stream.as_raw();
        let stem = "mlp.shared_experts";
        check(
            unsafe {
                g::hayate_glm_moe_gate_up(
                    self.p("decode.product"),
                    self.p(if self.hadamard_block != 0 {
                        "rotated_input"
                    } else {
                        "ffn_norm"
                    }),
                    self.p("ffn_norm"),
                    self.p("decode.images"),
                    self.dense_weight(&format!("{stem}.gate_proj.weight")),
                    self.dense_weight(&format!("{stem}.up_proj.weight")),
                    nv as i32,
                    down_fp8 as i32,
                    matrix_stride(nv),
                    st,
                )
            },
            "glm batched gate/up + shared",
        )?;
        if self.hadamard_block != 0 && nv && !down_fp8 {
            self.rotate(
                self.p("decode.product"),
                self.p("decode.product"),
                TOP_K,
                2048,
            )?;
        }
        self.mark(6)?;
        check(
            unsafe {
                g::hayate_glm_moe_down(
                    self.p("decode.expert"),
                    self.p("decode.product"),
                    self.p("decode.images"),
                    self.dense_weight(&format!("{stem}.down_proj.weight")),
                    nv as i32,
                    down_fp8 as i32,
                    matrix_stride(nv),
                    st,
                )
            },
            "glm batched down + shared",
        )?;
        self.lora_experts(
            st,
            self.p("decode.product"),
            self.p("decode.expert"),
            self.p("route_ids"),
            self.p("decode.images"),
        )?;
        self.mark(7)
    }
    fn decode_moe_reduce(&self) -> Result<()> {
        let st = self.stream.as_raw();
        check(
            unsafe {
                g::hayate_glm_moe_reduce(
                    self.p("mlp"),
                    self.p("decode.expert"),
                    self.p("route_ids"),
                    self.p("route_weights"),
                    st,
                )
            },
            "glm batched expert reduce",
        )?;
        self.mark(8)?;
        self.expand("output", "attention_residual", "mlp", "ffn_map")
    }

    pub(super) fn step(
        &mut self,
        input: &[u8],
        cache: &mut super::experts::Experts,
    ) -> Result<Vec<u8>> {
        if self.context == 0 || self.position >= self.context || input.len() != HC * HIDDEN * 2 {
            return Err(Error::InvalidShape("GLM decode context/input".into()));
        }
        if let Some(p) = &self.profile {
            p.reset();
        }
        self.stream
            .copy_h2d(self.buffers.get_mut("input").unwrap(), input)?;
        self.decode(cache)?;
        let mut output = vec![0; HC * HIDDEN * 2];
        self.download(&mut output, "output")?;
        self.position += 1;
        Ok(output)
    }

    pub(super) fn prefill_cached(
        &mut self,
        input: &[u8],
        cache: &mut super::experts::Experts,
        batch: &mut super::prefill::Prefill,
    ) -> Result<Vec<u8>> {
        let row_bytes = HC * HIDDEN * 2;
        let tokens = input.len() / row_bytes;
        if tokens == 0
            || !input.len().is_multiple_of(row_bytes)
            || tokens > batch.capacity
            || self.position + tokens > self.context
        {
            return Err(Error::InvalidShape("GLM prefill chunk/input".into()));
        }
        if self.index < 3 || tokens == 1 {
            let start = std::time::Instant::now();
            let mut attention = 0.;
            let mut dense = 0.;
            let mut scan_ms = 0.;
            let mut output = Vec::with_capacity(input.len());
            for row in input.chunks_exact(row_bytes) {
                output.extend(self.step(row, cache)?);
                if let Some(p) = &self.profile {
                    attention += p.ms(1, 2)? as f64;
                    dense += p.ms(3, 8)? as f64;
                    if self.index % 4 != 3 {
                        scan_ms += p.ms(10, 11)? as f64;
                    }
                }
            }
            if self.profile.is_some() {
                eprintln!(
                    "glm_prefill_layer position={} rows={tokens} layer={} kind=legacy attention_ms={attention:.6} ffn_ms={dense:.6} scan_attention_ms={scan_ms:.6} wall_ms={:.6}",
                    self.position - tokens,
                    self.index,
                    start.elapsed().as_secs_f64() * 1000.
                );
            }
            return Ok(output);
        }
        let legacy_start = std::time::Instant::now();
        let mut attention_ms = 0.;
        let mut scan_ms = 0.;
        let mut shared_ms = 0.;
        let st = self.stream.as_raw();
        for (t, row) in input.chunks_exact(row_bytes).enumerate() {
            self.stream
                .copy_h2d(self.buffers.get_mut("input").unwrap(), row)?;
            self.attention_ffn_norm()?;
            if let Some(p) = &self.profile {
                self.stream.synchronize()?;
                attention_ms += p.ms(1, 2)? as f64;
                if self.index % 4 != 3 {
                    scan_ms += p.ms(10, 11)? as f64;
                }
            }

            for (dst, src, bytes) in [
                ("norm", "ffn_norm", HIDDEN * 2),
                ("residual", "attention_residual", row_bytes),
                ("map", "ffn_map", 24 * 4),
            ] {
                check(
                    unsafe {
                        g::hayate_glm_copy(
                            batch.p(dst).cast::<u8>().add(t * bytes).cast(),
                            self.p(src),
                            bytes,
                            st,
                        )
                    },
                    "glm prefill snapshot",
                )?;
            }
            self.position += 1;
        }
        self.mark(3)?;
        self.prefill_experts(cache, batch, tokens)?;
        self.mark(4)?;
        let moe_ms = if let Some(p) = &self.profile {
            self.stream.synchronize()?;
            p.ms(3, 4)?
        } else {
            0.
        };
        let mut output = Vec::with_capacity(input.len());
        for t in 0..tokens {
            for (dst, src, bytes) in [
                ("ffn_norm", "norm", HIDDEN * 2),
                ("attention_residual", "residual", row_bytes),
                ("ffn_map", "map", 24 * 4),
                ("mlp", "mlp", HIDDEN * 2),
            ] {
                check(
                    unsafe {
                        g::hayate_glm_copy(
                            self.p(dst),
                            batch.p(src).cast::<u8>().add(t * bytes).cast(),
                            bytes,
                            st,
                        )
                    },
                    "glm prefill restore token",
                )?;
            }
            self.mark(4)?;
            self.mlp("mlp.shared_experts", "expert", 2048)?;
            check(
                unsafe {
                    g::hayate_glm_point(
                        self.p("mlp"),
                        self.p("mlp"),
                        self.p("expert"),
                        HIDDEN,
                        2,
                        st,
                    )
                },
                "glm prefill shared expert",
            )?;
            self.mark(5)?;
            self.expand("output", "attention_residual", "mlp", "ffn_map")?;
            let mut row = vec![0; row_bytes];
            self.download(&mut row, "output")?;
            output.extend(row);
            if let Some(p) = &self.profile {
                shared_ms += p.ms(4, 5)? as f64;
            }
        }
        if self.profile.is_some() {
            eprintln!(
                "glm_prefill_layer position={} rows={tokens} layer={} kind=legacy attention_ms={attention_ms:.6} scan_attention_ms={scan_ms:.6} routed_moe_ms={moe_ms:.6} shared_dense_ms={shared_ms:.6} wall_ms={:.6}",
                self.position - tokens,
                self.index,
                legacy_start.elapsed().as_secs_f64() * 1000.
            );
        }
        Ok(output)
    }

    fn prefill_experts(
        &mut self,
        cache: &mut super::experts::Experts,
        batch: &mut super::prefill::Prefill,
        tokens: usize,
    ) -> Result<()> {
        let st = self.stream.as_raw();
        let transfer_before = cache.transfer_stats.h2d_bytes;
        let prefetch_before = cache.transfer_stats.prefetch_hits;
        let mut transfer_ms = 0.;
        let mut host_bytes = 0;
        let mut disk_bytes = 0;

        check(
            unsafe {
                g::hayate_glm_route(
                    batch.p("logits"),
                    batch.p("route_weights"),
                    batch.p("ids"),
                    batch.p("norm"),
                    self.w("mlp.gate.weight"),
                    self.w("mlp.gate.e_score_correction_bias"),
                    tokens as u32,
                    HIDDEN as u32,
                    EXPERTS as u32,
                    TOP_K as u32,
                    st,
                )
            },
            "glm prefill route",
        )?;
        let mut raw = vec![0; tokens * TOP_K * 4];
        if let Some(p) = &self.profile {
            p.download(&self.stream, &mut raw, &batch.buffers["ids"])?;
        } else {
            self.stream.copy_d2h(&mut raw, &batch.buffers["ids"])?;
        }
        let ids: Vec<_> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let groups = super::prefill::group_routes(&ids)?;
        let gathered_input = if self.hadamard_block != 0 {
            self.rotate(batch.p("rotated_norm"), batch.p("norm"), tokens, HIDDEN)?;
            "rotated_norm"
        } else {
            "norm"
        };
        cache.route(
            self.index,
            &ids.iter().map(|&i| i as u16).collect::<Vec<_>>(),
        )?;
        let selected: Vec<_> = groups
            .iter()
            .enumerate()
            .filter_map(|(e, g)| (!g.is_empty()).then_some(e))
            .collect();
        check(
            unsafe {
                g::hayate_glm_copy(batch.p("mlp"), std::ptr::null(), tokens * HIDDEN * 2, st)
            },
            "glm prefill zero",
        )?;
        if let Some(ahead) = &batch.ahead {
            self.stream.wait_event(&ahead[self.index % 2].ready)?;
        }
        batch.previous[self.index] = selected.clone();
        // Groups of at most 8 experts and at most `tokens` routed rows, so
        // the grouped buffers need `tokens` rows (not 8 x tokens).
        let mut chunks: Vec<Vec<usize>> = Vec::new();
        // Packed experts (NVFP4/LOWBIT, no FP8 down) skip the BF16 restore: the
        // grouped GEMM reads the images in place. Experts already on the GPU
        // (resident or prefetched) go first in chunks of any size; the others
        // follow in uploaded banks of 8.
        let packed = self.tokens > 1 && {
            let (nv, fp8) = cache.format(self.index);
            nv && !fp8
        };
        let ahead = batch.ahead.as_ref().map(|a| &a[self.index % 2]);
        let mut ready: Vec<(usize, Ptr)> = Vec::new();
        let mut missing: Vec<usize> = Vec::new();
        if packed {
            for &e in &selected {
                match cache.prefill_image(self.index, e, ahead) {
                    Some(image) => ready.push((e, image)),
                    None => missing.push(e),
                }
            }
        }
        let ready_count = {
            let mut split = |list: &[usize], limit: usize| {
                let mut group_rows = 0;
                let first = chunks.len();
                for &e in list {
                    let n = groups[e].len();
                    if chunks.len() == first
                        || chunks
                            .last()
                            .is_none_or(|c| c.len() == limit || group_rows + n > tokens)
                    {
                        chunks.push(Vec::new());
                        group_rows = 0;
                    }
                    chunks.last_mut().unwrap().push(e);
                    group_rows += n;
                }
                chunks.len() - first
            };
            if packed {
                let ready_ids: Vec<usize> = ready.iter().map(|&(e, _)| e).collect();
                let count = split(&ready_ids, EXPERTS);
                split(&missing, 8);
                count
            } else {
                split(&selected, 8);
                0
            }
        };
        let timings = if self.profile.is_some() {
            Some(super::profile::Profile::new(
                &self.stream,
                chunks.len() * 3,
            )?)
        } else {
            None
        };
        if packed {
            let uploads = &chunks[ready_count..];
            if let Some(first) = uploads.first() {
                cache.prefill_images(
                    self.index,
                    first,
                    &mut batch.banks.as_mut().unwrap()[0],
                    &batch.upload[0],
                    batch.ahead.as_ref().map(|a| &a[self.index % 2]),
                )?;
            }
            // All chunks' metadata in one copy (one stream sync per layer).
            // Uploaded experts land in bank arena entry j (`prefill_images`).
            let mut meta = PackedMeta::default();
            for (group_index, experts) in chunks.iter().enumerate() {
                let images: Vec<Ptr> = if group_index < ready_count {
                    experts
                        .iter()
                        .map(|e| ready.iter().find(|r| r.0 == *e).unwrap().1)
                        .collect()
                } else {
                    let bank = &batch.banks.as_ref().unwrap()[(group_index - ready_count) % 2];
                    (0..experts.len())
                        .map(|j| bank.arena[j].as_ptr().cast_mut())
                        .collect()
                };
                meta.push(experts, &images, &groups, cache.layer_formats(self.index));
            }
            meta.upload(&self.stream, batch)?;
            for (group_index, experts) in chunks.iter().enumerate() {
                if group_index >= ready_count {
                    let u = group_index - ready_count;
                    let bank_index = u % 2;
                    self.stream
                        .wait_event(&batch.banks.as_ref().unwrap()[bank_index].ready)?;
                    if let Some(next) = uploads.get(u + 1) {
                        cache.prefill_images(
                            self.index,
                            next,
                            &mut batch.banks.as_mut().unwrap()[1 - bank_index],
                            &batch.upload[1 - bank_index],
                            batch.ahead.as_ref().map(|a| &a[self.index % 2]),
                        )?;
                    }
                    if self.profile.is_some() {
                        let bank = &batch.banks.as_ref().unwrap()[bank_index];
                        bank.ready.synchronize()?;
                        transfer_ms += bank.begin.elapsed_ms(&bank.ready)? as f64;
                        host_bytes += bank.host_bytes;
                        disk_bytes += bank.disk_bytes;
                    }
                    debug_assert!(
                        batch.banks.as_ref().unwrap()[bank_index]
                            .images
                            .iter()
                            .zip(&batch.banks.as_ref().unwrap()[bank_index].arena)
                            .all(|(&p, b)| p == b.as_ptr().cast_mut())
                    );
                }
                self.prefill_packed_chunk(
                    batch,
                    &meta,
                    group_index,
                    experts,
                    gathered_input,
                    timings.as_ref().map(|t| (t, group_index)),
                )?;
                if group_index >= ready_count {
                    batch.banks.as_ref().unwrap()[(group_index - ready_count) % 2]
                        .used
                        .record(&self.stream)?;
                }
            }
        }
        if !packed
            && self.tokens > 1
            && let Some(first) = chunks.first()
        {
            cache.prefill_images(
                self.index,
                first,
                &mut batch.banks.as_mut().unwrap()[0],
                &batch.upload[0],
                batch.ahead.as_ref().map(|a| &a[self.index % 2]),
            )?;
        }
        for (group_index, experts) in chunks.iter().enumerate().filter(|_| !packed) {
            let bank_index = group_index % 2;
            if self.tokens > 1 {
                self.stream
                    .wait_event(&batch.banks.as_ref().unwrap()[bank_index].ready)?;
            }
            if self.tokens > 1
                && let Some(next) = chunks.get(group_index + 1)
            {
                cache.prefill_images(
                    self.index,
                    next,
                    &mut batch.banks.as_mut().unwrap()[1 - bank_index],
                    &batch.upload[1 - bank_index],
                    batch.ahead.as_ref().map(|a| &a[self.index % 2]),
                )?;
            }

            let mut assignments = Vec::new();
            let mut ranges = Vec::new();
            for &e in experts.iter() {
                ranges.extend([groups[e].len() as u32, assignments.len() as u32]);
                assignments.extend_from_slice(&groups[e]);
            }
            let rows = assignments.len();
            self.stream.copy_h2d(
                batch.buffers.get_mut("assignments").unwrap(),
                &assignments
                    .iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            self.stream.copy_h2d(
                batch.buffers.get_mut("ranges").unwrap(),
                &ranges
                    .iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            if self.tokens > 1 && self.profile.is_some() {
                let bank = &batch.banks.as_ref().unwrap()[bank_index];
                transfer_ms += bank.begin.elapsed_ms(&bank.ready)? as f64;
                host_bytes += bank.host_bytes;
                disk_bytes += bank.disk_bytes;
            }
            check(
                unsafe {
                    g::hayate_glm_gather(
                        batch.p("input"),
                        batch.p(gathered_input),
                        batch.p("assignments"),
                        rows as u32,
                        HIDDEN as u32,
                        st,
                    )
                },
                "glm grouped gather",
            )?;
            if let Some(t) = &timings {
                t.mark(group_index * 3, &self.stream)?;
            }
            for (j, _) in experts.iter().enumerate() {
                if self.tokens == 1 {
                    self.mark(8)?;
                }
                let image = if self.tokens > 1 {
                    batch.banks.as_ref().unwrap()[bank_index].images[j]
                } else {
                    cache
                        .image(self.index, experts[j], &self.stream)?
                        .cast_mut()
                };
                if self.tokens == 1
                    && let Some(p) = &self.profile
                {
                    self.mark(9)?;
                    self.stream.synchronize()?;
                    transfer_ms += p.ms(8, 9)? as f64;
                }
                if let Some((gate_k2, down_k2)) = self.exl3 {
                    // EXL3: each matrix restored to BF16 rows; prefill_grouped takes the BF16 GEMM.
                    let gate_bytes = hayate_core::QuantSpec::Exl3 { k2: gate_k2, cb: 2 }
                        .matrix_bytes(2048, HIDDEN as u64)
                        .unwrap() as usize;
                    for (part, (target, n, k, k2)) in [
                        ("gate_w", 2048, HIDDEN, gate_k2),
                        ("up_w", 2048, HIDDEN, gate_k2),
                        ("down_w", HIDDEN, 2048, down_k2),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let dst = unsafe { batch.p(target).cast::<u8>().add(j * n * k * 2).cast() };
                        let weight = unsafe { image.cast::<u8>().add(part * gate_bytes).cast() };
                        check(
                            unsafe {
                                hayate_kernels::glm53::exl3_ffi::hayate_glm_exl3_restore(
                                    dst,
                                    weight,
                                    n as u32,
                                    k as u32,
                                    u32::from(k2),
                                    st,
                                )
                            },
                            "glm grouped EXL3 restore",
                        )?;
                    }
                    continue;
                }
                // LOWBIT experts: lossless conversion to an NVFP4 image first (the
                // matrices below then take the NVFP4 path).
                let image = if cache.is_mixed(self.index) {
                    let format = cache.layer_formats(self.index)[experts[j]];
                    if format.gate_up.format != 0 || format.down.format != 0 {
                        let scratch = batch.lowbit_scratch.as_ref().unwrap().as_ptr().cast_mut();
                        check(
                            unsafe {
                                hayate_kernels::glm53::lowbit::ffi::hayate_glm_lowbit_convert_expert(
                                    scratch,
                                    image,
                                    format,
                                    HIDDEN as u32,
                                    2048,
                                    st,
                                )
                            },
                            "glm prefill LOWBIT convert",
                        )?;
                        scratch
                    } else {
                        image
                    }
                } else {
                    image
                };
                let (nvfp4, down_fp8) = cache.format(self.index);
                let stride = matrix_stride(nvfp4);
                for (part, (target, n, k)) in [
                    ("gate_w", 2048, HIDDEN),
                    ("up_w", 2048, HIDDEN),
                    ("down_w", HIDDEN, 2048),
                ]
                .into_iter()
                .enumerate()
                {
                    let offset = part * stride;
                    let dst = unsafe { batch.p(target).cast::<u8>().add(j * n * k * 2).cast() };
                    let weight = unsafe { image.cast::<u8>().add(offset).cast() };
                    if nvfp4 && !(part == 2 && down_fp8) {
                        check(
                            unsafe {
                                g::hayate_glm_nvfp4_restore_unscaled(
                                    dst,
                                    batch
                                        .p("scales")
                                        .cast::<u8>()
                                        .add((part * 8 + j) * 4)
                                        .cast(),
                                    weight,
                                    n as u32,
                                    k as u32,
                                    st,
                                )
                            },
                            "glm grouped NVFP4 restore",
                        )?;
                    } else {
                        let scale = unsafe { image.cast::<u8>().add(offset + n * k).cast() };
                        check(
                            unsafe {
                                g::hayate_glm_restore(dst, weight, scale, n as u32, k as u32, st)
                            },
                            "glm grouped FP8 restore",
                        )?;
                    }
                }
            }
            if self.tokens > 1 {
                batch.banks.as_ref().unwrap()[bank_index]
                    .used
                    .record(&self.stream)?;
            }
            if let Some(t) = &timings {
                t.mark(group_index * 3 + 1, &self.stream)?;
            }
            for (part, (out, x, weight)) in [("gate", "input", "gate_w"), ("up", "input", "up_w")]
                .into_iter()
                .enumerate()
            {
                self.prefill_grouped(cache, batch, out, x, weight, part, experts.len(), &ranges)?;
            }
            check(
                unsafe {
                    g::hayate_glm_point(
                        batch.p("product"),
                        batch.p("gate"),
                        batch.p("up"),
                        rows * 2048,
                        1,
                        st,
                    )
                },
                "glm grouped swiglu",
            )?;
            if self.hadamard_block != 0 && !cache.format(self.index).1 {
                self.rotate(batch.p("product"), batch.p("product"), rows, 2048)?;
            }
            self.prefill_grouped(
                cache,
                batch,
                "expert",
                "product",
                "down_w",
                2,
                experts.len(),
                &ranges,
            )?;
            self.lora_grouped(batch.p("product"), batch.p("expert"), experts, &ranges)?;
            for (j, _) in experts.iter().enumerate() {
                let count = ranges[j * 2];
                let offset = ranges[j * 2 + 1] as usize;
                check(
                    unsafe {
                        g::hayate_glm_scatter(
                            batch.p("mlp"),
                            batch
                                .p("expert")
                                .cast::<u8>()
                                .add(offset * HIDDEN * 2)
                                .cast(),
                            batch.p("route_weights"),
                            batch.p("assignments").cast::<u8>().add(offset * 4).cast(),
                            count,
                            HIDDEN as u32,
                            st,
                        )
                    },
                    "glm grouped scatter",
                )?;
            }
            if let Some(t) = &timings {
                t.mark(group_index * 3 + 2, &self.stream)?;
            }
        }
        let mut restore_ms = 0.;
        let mut grouped_ms = 0.;
        if let Some(t) = &timings {
            self.stream.synchronize()?;
            for i in 0..chunks.len() {
                restore_ms += t.ms(i * 3, i * 3 + 1)? as f64;
                grouped_ms += t.ms(i * 3 + 1, i * 3 + 2)? as f64;
            }
        }
        if self.profile.is_some() {
            eprintln!(
                "glm_prefill_transfer position={} rows={tokens} layer={} transfer_ms={transfer_ms:.6} restore_ms={restore_ms:.6} grouped_ms={grouped_ms:.6} host_h2d_bytes={host_bytes} disk_h2d_bytes={disk_bytes} h2d_bytes={} overlap={} clock={}",
                self.position,
                self.index,
                cache.transfer_stats.h2d_bytes - transfer_before,
                self.tokens > 1,
                if self.tokens > 1 { "upload" } else { "compute" }
            );
        }
        if let Some(ahead) = &batch.ahead {
            let bank = &ahead[self.index % 2];
            bank.used.record(&self.stream)?;
            if self.profile.is_some() && !bank.ids.is_empty() {
                bank.ready.synchronize()?;
                eprintln!(
                    "glm_prefill_lookahead position={} rows={tokens} layer={} policy={} predicted={} h2d_bytes={} host_h2d_bytes={} disk_h2d_bytes={} hits={} transfer_ms={:.6}",
                    self.position,
                    self.index,
                    if batch.all_experts { "all" } else { "previous" },
                    bank.ids.len(),
                    bank.copied_bytes,
                    bank.host_bytes,
                    bank.disk_bytes,
                    cache.transfer_stats.prefetch_hits - prefetch_before,
                    bank.begin.elapsed_ms(&bank.ready)?
                );
            }
        }
        Ok(())
    }

    /// One chunk of the packed path: gather the chunk's routed rows, fused
    /// gate/up + SwiGLU and down on the packed images, scatter per expert.
    fn prefill_packed_chunk(
        &mut self,
        batch: &super::prefill::Prefill,
        meta: &PackedMeta,
        chunk: usize,
        experts: &[usize],
        gathered_input: &str,
        timings: Option<(&super::profile::Profile, usize)>,
    ) -> Result<()> {
        use hayate_kernels::glm53::packed_grouped::ffi as packed;
        let st = self.stream.as_raw();
        let count = experts.len();
        let (first, row_base) = meta.chunks[chunk];
        let ranges = &meta.ranges[first * 2..(first + count) * 2];
        let rows = ranges[(count - 1) * 2] + ranges[(count - 1) * 2 + 1];
        let base = batch.p("packed_meta").cast::<u8>();
        let at = |offset: usize| unsafe { base.add(offset).cast::<c_void>() };
        let images = at(first * 8);
        let formats = at(EXPERTS * 8 + first * 8);
        let ranges_ptr = at(EXPERTS * 16 + first * 8);
        let assignments = at(EXPERTS * 24 + row_base * 4);
        check(
            unsafe {
                g::hayate_glm_gather(
                    batch.p("input"),
                    batch.p(gathered_input),
                    assignments,
                    rows,
                    HIDDEN as u32,
                    st,
                )
            },
            "glm packed gather",
        )?;
        if let Some((t, i)) = timings {
            t.mark(i * 3, &self.stream)?;
            t.mark(i * 3 + 1, &self.stream)?;
        }
        if let Some((gate_k2, _)) = self.exl3 {
            check(
                unsafe {
                    hayate_kernels::glm53::exl3_ffi::hayate_glm_grouped_exl3_gate_up(
                        batch.p("product"),
                        batch.p("input"),
                        images,
                        ranges_ptr,
                        count as u32,
                        u32::from(gate_k2),
                        HIDDEN as u32,
                        2048,
                        st,
                    )
                },
                "glm EXL3 grouped gate/up",
            )?;
        } else {
            check(
                unsafe {
                    packed::hayate_glm_grouped_packed_gate_up(
                        batch.p("product"),
                        batch.p("input"),
                        images,
                        formats.cast(),
                        ranges_ptr,
                        count as u32,
                        HIDDEN as u32,
                        2048,
                        st,
                    )
                },
                "glm packed gate/up",
            )?;
        }
        if self.hadamard_block != 0 {
            self.rotate(batch.p("product"), batch.p("product"), rows as usize, 2048)?;
        }
        if let Some((gate_k2, down_k2)) = self.exl3 {
            check(
                unsafe {
                    hayate_kernels::glm53::exl3_ffi::hayate_glm_grouped_exl3_down(
                        batch.p("expert"),
                        batch.p("product"),
                        images,
                        ranges_ptr,
                        count as u32,
                        u32::from(gate_k2),
                        u32::from(down_k2),
                        HIDDEN as u32,
                        2048,
                        st,
                    )
                },
                "glm EXL3 grouped down",
            )?;
        } else {
            check(
                unsafe {
                    packed::hayate_glm_grouped_packed_down(
                        batch.p("expert"),
                        batch.p("product"),
                        images,
                        formats.cast(),
                        ranges_ptr,
                        count as u32,
                        HIDDEN as u32,
                        2048,
                        st,
                    )
                },
                "glm packed down",
            )?;
        }
        // LoRA の routed down(s·B_e(A_e·h)): SwiGLU の積は product に残っている
        self.lora_grouped(batch.p("product"), batch.p("expert"), experts, ranges)?;
        for j in 0..count {
            let n = ranges[j * 2];
            let offset = ranges[j * 2 + 1] as usize;
            check(
                unsafe {
                    g::hayate_glm_scatter(
                        batch.p("mlp"),
                        batch
                            .p("expert")
                            .cast::<u8>()
                            .add(offset * HIDDEN * 2)
                            .cast(),
                        batch.p("route_weights"),
                        assignments.cast::<u8>().add(offset * 4).cast(),
                        n,
                        HIDDEN as u32,
                        st,
                    )
                },
                "glm packed scatter",
            )?;
        }
        if let Some((t, i)) = timings {
            t.mark(i * 3 + 2, &self.stream)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_grouped(
        &self,
        cache: &super::experts::Experts,
        batch: &super::prefill::Prefill,
        out: &str,
        input: &str,
        weight: &str,
        part: usize,
        count: usize,
        ranges: &[u32],
    ) -> Result<()> {
        let (nv, down_fp8) = cache.format(self.index);
        let (n, k) = if part == 2 {
            (HIDDEN, 2048)
        } else {
            (2048, HIDDEN)
        };
        check(
            unsafe {
                if nv && !(part == 2 && down_fp8) && self.exl3.is_none() {
                    g::hayate_glm_grouped_nvfp4(
                        batch.p(out),
                        batch.p("accumulator"),
                        batch.p(input),
                        batch.p(weight),
                        batch.p("ranges"),
                        count as u32,
                        n as u32,
                        k as u32,
                        batch.p("metadata"),
                        batch.p("scales").cast::<u8>().add(part * 8 * 4).cast(),
                        *ranges.iter().step_by(2).max().unwrap(),
                        self.stream.as_raw(),
                    )
                } else {
                    g::hayate_glm_grouped(
                        batch.p(out),
                        batch.p(input),
                        batch.p(weight),
                        batch.p("ranges"),
                        count as u32,
                        n as u32,
                        k as u32,
                        batch.p("metadata"),
                        self.stream.as_raw(),
                    )
                }
            },
            "glm prefill grouped",
        )
    }

    fn expert_gemv(&self, image: *const c_void, nvfp4: bool, down_fp8: bool) -> Result<()> {
        let st = self.stream.as_raw();
        debug_assert!(
            super::experts::validate_image_alignment(image as usize, nvfp4, down_fp8).is_ok()
        );
        let stride = matrix_stride(nvfp4);
        for (part, (out, input, n, k)) in [
            (
                "gate",
                if self.hadamard_block != 0 {
                    "rotated_input"
                } else {
                    "ffn_norm"
                },
                2048,
                4096,
            ),
            (
                "up",
                if self.hadamard_block != 0 {
                    "rotated_input"
                } else {
                    "ffn_norm"
                },
                2048,
                4096,
            ),
            ("expert", "product", 4096, 2048),
        ]
        .into_iter()
        .enumerate()
        {
            let offset = part * stride;
            if out == "expert" {
                check(
                    unsafe {
                        g::hayate_glm_point(
                            self.p("product"),
                            self.p("gate"),
                            self.p("up"),
                            2048,
                            1,
                            st,
                        )
                    },
                    "glm expert swiglu",
                )?;
                if self.hadamard_block != 0 && !down_fp8 {
                    self.rotate(self.p("product"), self.p("product"), 1, 2048)?;
                }
            }
            let weight = unsafe { image.cast::<u8>().add(offset).cast() };
            if nvfp4 && !(part == 2 && down_fp8) {
                check(
                    unsafe {
                        hayate_kernels::llm::ffi::hayate_llm_gemv(
                            self.p(out),
                            self.p(input),
                            weight,
                            n as u32,
                            k as u32,
                            1,
                            0,
                            std::ptr::null_mut(),
                            4,
                            1,
                            0,
                            st,
                        )
                    },
                    "glm expert NVFP4 GEMV",
                )?;
            } else {
                let scale = unsafe { image.cast::<u8>().add(offset + n * k).cast() };
                check(
                    unsafe {
                        g::hayate_glm_fp8_gemv(
                            self.p(out),
                            self.p(input),
                            weight,
                            scale,
                            n as u32,
                            k as u32,
                            st,
                        )
                    },
                    "glm expert FP8 GEMV",
                )?;
            }
        }
        Ok(())
    }

    fn sparse_step(&mut self) -> Result<()> {
        self.sparse_select()?;
        self.sparse_attention()
    }

    fn sparse_select(&mut self) -> Result<()> {
        let st = self.stream.as_raw();
        self.linear("qa", "attn_norm", "self_attn.q_a_proj.weight", 1536, HIDDEN)?;
        self.norm("qr", "qa", "self_attn.q_a_layernorm.weight", 1536)?;
        self.linear("q", "qr", "self_attn.q_b_proj.weight", 16384, 1536)?;
        self.linear("iq", "qr", "self_attn.indexer.wq_b.weight", 4096, 1536)?;
        self.linear(
            "iw",
            "attn_norm",
            "self_attn.indexer.weights_proj.weight",
            32,
            HIDDEN,
        )?;
        self.sparse_cache(None)?;
        check(
            unsafe {
                g::hayate_glm_mla_select(
                    self.p("selected"),
                    self.p("scores"),
                    self.p("iq"),
                    self.p("pools"),
                    self.p("iw"),
                    self.position as u32,
                    1,
                    self.context.div_ceil(4),
                    st,
                )
            },
            "glm decode select",
        )
    }

    fn sparse_cache(&mut self, condition: Option<(Ptr, usize)>) -> Result<()> {
        let t = 1;
        let st = self.stream.as_raw();
        self.linear(
            "kva",
            "attn_norm",
            "self_attn.kv_a_proj_with_mqa.weight",
            512,
            HIDDEN,
        )?;
        self.norm("kvr", "kva", "self_attn.kv_a_layernorm.weight", 512)?;
        self.linear(
            "ik",
            "attn_norm",
            "self_attn.indexer.wk.weight",
            128,
            HIDDEN,
        )?;
        check(
            unsafe {
                g::hayate_glm_norm(
                    self.p("ikn"),
                    self.p("ik"),
                    self.w("self_attn.indexer.k_norm.weight"),
                    self.w("self_attn.indexer.k_norm.bias"),
                    t,
                    128,
                    1,
                    st,
                )
            },
            "glm index layernorm",
        )?;
        self.linear(
            "ig",
            "attn_norm",
            "self_attn.indexer.index_kpool_compress_gate",
            128,
            HIDDEN,
        )?;

        let offset = self.position * 512 * 2;
        let tail = self.position % 4 * 128 * 2;
        if self.reference {
            let mut latent = vec![0; 512 * 2];
            self.download(&mut latent, "kvr")?;
            if let Some(host) = self.latent.as_mut() {
                host.as_mut_slice()[offset..offset + latent.len()].copy_from_slice(&latent);
            }
            let mut key = vec![0; 128 * 2];
            let mut gate = vec![0; 128 * 2];
            self.download(&mut key, "ikn")?;
            self.download(&mut gate, "ig")?;
            self.tail_keys[tail..tail + key.len()].copy_from_slice(&key);
            self.tail_gates[tail..tail + gate.len()].copy_from_slice(&gate);
        } else {
            // Pinned mirror only for MTP verification (projected-KV path).
            if let Some(latent) = self.latent.as_mut() {
                let host = unsafe { latent.as_mut_slice().as_mut_ptr().add(offset).cast() };
                check(
                    unsafe { g::hayate_glm_latent_copy(host, self.p("kvr"), st) },
                    "glm latent append",
                )?;
            }

            if let Some(p) = &self.profile {
                p.enqueue_d2h();
            }
            for (dst, src) in [("pool_keys", "ikn"), ("pool_gates", "ig")] {
                let out = unsafe { self.p(dst).cast::<u8>().add(tail).cast() };
                check(
                    unsafe {
                        if let Some((result, row)) = condition {
                            g::hayate_glm_copy_if(out, self.p(src), 128 * 2, result, row as u32, st)
                        } else {
                            g::hayate_glm_copy(out, self.p(src), 128 * 2, st)
                        }
                    },
                    "glm index tail D2D",
                )?;
            }
        }
        self.latent_append(self.p("kvr"), 1)?;
        if self.position % 4 == 3 {
            if self.reference {
                self.stream
                    .copy_h2d(self.buffers.get_mut("pool_keys").unwrap(), &self.tail_keys)?;
                self.stream.copy_h2d(
                    self.buffers.get_mut("pool_gates").unwrap(),
                    &self.tail_gates,
                )?;
            }
            let out = unsafe {
                self.p("pools")
                    .cast::<u8>()
                    .add(self.position / 4 * 128 * 2)
                    .cast()
            };
            check(
                unsafe {
                    g::hayate_glm_pool(
                        out,
                        self.p("pool_keys"),
                        self.p("pool_gates"),
                        self.w("self_attn.indexer.index_kpool_compress_ape"),
                        1,
                        128,
                        st,
                    )
                },
                "glm decode pool",
            )?;
        }
        Ok(())
    }

    fn sparse_attention(&mut self) -> Result<()> {
        self.mla_decode(self.p("q"), self.p("selected"), self.p("attn_core"), 1)?;
        self.linear(
            "attention",
            "attn_core",
            "self_attn.o_proj.weight",
            HIDDEN,
            16384,
        )
    }

    fn sparse_count(length: usize) -> usize {
        (length / 4).min(512) * 4 + length % 4
    }
    fn sparse_gather_async(&self, selected: Ptr, rows: usize, bound: usize) -> Result<()> {
        let host = self.latent.as_ref().unwrap().as_ptr();
        check(
            unsafe {
                g::hayate_glm_sparse_gather(
                    self.p("latent_selected"),
                    self.p("sparse.sorted"),
                    self.p("sparse.ids"),
                    self.p("sparse.maps"),
                    self.p("sparse.counts"),
                    selected,
                    host.cast(),
                    rows as u32,
                    bound as u32,
                    self.stream.as_raw(),
                )
            },
            "glm mapped sparse KV gather",
        )?;
        if self.profile.is_some() {
            self.publish_host(
                self.asynchronous
                    .as_ref()
                    .unwrap()
                    .mailbox
                    .device(super::async_decode::SPARSE_COUNTS),
                self.p("sparse.counts"),
                20,
            )?;
        }
        Ok(())
    }
    pub fn prefill(&mut self, input: &[u8]) -> Result<LayerOutput> {
        if input.len() != self.tokens * HC * HIDDEN * 2 {
            return Err(Error::InvalidShape("GLM prefill input bytes".into()));
        }
        self.stream
            .copy_h2d(self.buffers.get_mut("input").unwrap(), input)?;
        if self.index % 4 != 3 && self.index != 45 {
            let zeros = vec![0; 64 * 128 * 128 * 4];
            self.stream
                .copy_h2d(self.buffers.get_mut("state").unwrap(), &zeros)?;
        }
        let start = self.stream.event()?;
        let end = self.stream.event()?;
        start.record(&self.stream)?;
        self.forward(None)?;
        end.record(&self.stream)?;
        end.synchronize()?;
        self.capture(start.elapsed_ms(&end)?)
    }

    fn hc_norm(
        &self,
        site: &str,
        map: &str,
        collapsed: &str,
        out: &str,
        input: &str,
        weight: &str,
    ) -> Result<()> {
        if self.index == 45 {
            return self.norm(out, input, weight, HIDDEN);
        }
        if self.context > 0 && self.tokens == 1 && !self.reference {
            check(
                unsafe {
                    g::hayate_glm_hc_norm(
                        self.p(map),
                        self.p(collapsed),
                        self.p(out),
                        self.p(input),
                        self.w(&format!("hc_{site}_fn")),
                        self.w(&format!("hc_{site}_base")),
                        self.w(&format!("hc_{site}_scale")),
                        self.w(weight),
                        self.stream.as_raw(),
                    )
                },
                "glm mHC collapse RMSNorm",
            )
        } else {
            self.hc(site, map, collapsed, input)?;
            self.norm(out, collapsed, weight, HIDDEN)
        }
    }

    fn attention_ffn_norm(&mut self) -> Result<()> {
        self.mark(0)?;
        self.hc_norm(
            "attn",
            "attn_map",
            "attn_collapsed",
            "attn_norm",
            "input",
            "input_layernorm.weight",
        )?;
        self.mark(1)?;
        if self.index % 4 != 3 && self.index != 45 {
            self.kda()?;
        } else {
            if self.context > 0 {
                self.sparse_step()?;
            } else {
                self.sparse()?;
            }
        }
        self.mark(2)?;
        self.expand("attention_residual", "input", "attention", "attn_map")?;
        self.hc_norm(
            "ffn",
            "ffn_map",
            "ffn_collapsed",
            "ffn_norm",
            "attention_residual",
            "post_attention_layernorm.weight",
        )?;
        self.mark(3)?;
        Ok(())
    }

    fn forward(&mut self, mut cache: Option<&mut super::experts::Experts>) -> Result<()> {
        self.attention_ffn_norm()?;
        if self.index < 3 {
            self.mlp("mlp", "mlp", 12288)?;
        } else {
            let t = self.tokens as u32;
            let st = self.stream.as_raw();
            check(
                unsafe {
                    g::hayate_glm_route(
                        self.p("router_logits"),
                        self.p("route_weights"),
                        self.p("route_ids"),
                        self.p("ffn_norm"),
                        self.w("mlp.gate.weight"),
                        self.w("mlp.gate.e_score_correction_bias"),
                        t,
                        HIDDEN as u32,
                        EXPERTS as u32,
                        TOP_K as u32,
                        st,
                    )
                },
                "glm route",
            )?;
            self.mark(4)?;
            let mut raw = vec![0; self.tokens * TOP_K * 4];
            self.download(&mut raw, "route_ids")?;
            let selected: BTreeSet<_> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            if selected.iter().any(|&e| e >= EXPERTS as u32) {
                return Err(Error::Weights("GLM routed expert ID".into()));
            }
            let zeros = vec![0; self.tokens * HIDDEN * 2];
            self.stream
                .copy_h2d(self.buffers.get_mut("mlp").unwrap(), &zeros)?;
            if let Some(cache) = cache.as_mut() {
                let ids: Vec<u16> = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b) as u16)
                    .collect();
                cache.route(self.index, &ids)?;
            }
            if self.hadamard_block != 0 {
                self.rotate(
                    self.p("rotated_input"),
                    self.p("ffn_norm"),
                    self.tokens,
                    HIDDEN,
                )?;
            }
            self.mark(5)?;
            for e in selected {
                if let Some(cache) = cache.as_mut() {
                    let image = cache.image(self.index, e as usize, &self.stream)?;
                    let (nvfp4, down_fp8) = cache.format(self.index);
                    self.expert_gemv(image, nvfp4, down_fp8)?;
                    self.lora_linear(
                        st,
                        &format!("mlp.experts.{e}.down_proj.weight"),
                        self.p("expert"),
                        self.p("product"),
                        1,
                    )?;
                } else {
                    self.mlp(&format!("mlp.experts.{e}"), "expert", 2048)?;
                }
                check(
                    unsafe {
                        g::hayate_glm_accumulate(
                            self.p("mlp"),
                            self.p("expert"),
                            self.p("route_ids"),
                            self.p("route_weights"),
                            t,
                            HIDDEN as u32,
                            TOP_K as u32,
                            e,
                            st,
                        )
                    },
                    "glm expert sum",
                )?;
            }
            self.mlp("mlp.shared_experts", "expert", 2048)?;
            check(
                unsafe {
                    g::hayate_glm_point(
                        self.p("mlp"),
                        self.p("mlp"),
                        self.p("expert"),
                        self.tokens * HIDDEN,
                        2,
                        st,
                    )
                },
                "glm shared expert",
            )?;
        }
        self.mark(6)?;
        self.mark(7)?;
        self.mark(8)?;
        self.expand("output", "attention_residual", "mlp", "ffn_map")?;
        self.mark(9)?;
        Ok(())
    }

    fn capture(&self, cuda_ms: f32) -> Result<LayerOutput> {
        let mut tensors = BTreeMap::new();
        let mut names = vec![
            "attn_map",
            "attn_collapsed",
            "attn_norm",
            "attention",
            "attention_residual",
            "ffn_map",
            "ffn_collapsed",
            "ffn_norm",
            "mlp",
            "output",
        ];
        if self.index >= 3 {
            names.extend(["router_logits", "route_weights", "route_ids"]);
        }
        if self.index % 4 != 3 && self.index != 45 {
            names.extend(["cq", "ck", "cv", "decay", "kda", "state", "gated"]);
        } else {
            names.extend(["qr", "q", "k", "v", "ikn", "selected", "attn_core"]);
        }
        for name in names {
            let buf = &self.buffers[name];
            let mut data = vec![0; buf.bytes()];
            self.stream.copy_d2h(&mut data, buf)?;
            tensors.insert(name.into(), data);
        }
        Ok(LayerOutput { tensors, cuda_ms })
    }
}

impl Drop for Layer<'_> {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        let _ = self.prediction_stream.synchronize();
        self.before_graph.take();
        self.moe_graph.take();
    }
}
