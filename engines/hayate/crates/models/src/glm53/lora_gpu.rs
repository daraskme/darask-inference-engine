//! LoRA side paths of one layer on the GPU (see `glm53::lora`): every adapter of a target is
//! one `hayate_glm_lora` call after the base linear, on the same stream, so the per-op,
//! device (CUDA graph) and batched prefill paths all add it the same way.
use super::*;
use crate::glm53::lora::{self as host, ROUTED};
use hayate_kernels::cuda::ffi::Stream as RawStream;

/// Rows of the largest call: one prefill chunk.
const ROWS: usize = 8192;
const NONE: (Ptr, Ptr) = (std::ptr::null_mut(), std::ptr::null_mut());
/// Targets whose output every decode path materializes (the fused ones are refused).
const SUPPORTED: [&str; 6] = [
    "self_attn.o_proj.weight",
    "mlp.shared_experts.down_proj.weight",
    ROUTED,
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

#[derive(Debug)]
struct Adapter {
    a: DeviceBuffer,
    b: DeviceBuffer,
    rank: usize,
    k: usize,
    n: usize,
    count: usize,
    dtype: u32,
    scale: f32,
}
impl Adapter {
    /// All experts (`expert` None) or one expert as a single matrix pair.
    fn ffi(&self, expert: Option<usize>) -> g::GlmLora {
        let (a_stride, b_stride) = (self.rank * self.k, self.n * self.rank);
        let width = if self.dtype == 0 { 4 } else { 2 };
        let (a, b, count) = match expert {
            Some(e) => unsafe {
                (
                    self.a
                        .as_ptr()
                        .cast::<u8>()
                        .add(e * a_stride * width)
                        .cast(),
                    self.b
                        .as_ptr()
                        .cast::<u8>()
                        .add(e * b_stride * width)
                        .cast(),
                    1,
                )
            },
            None => (self.a.as_ptr(), self.b.as_ptr(), self.count),
        };
        g::GlmLora {
            a,
            b,
            a_stride: a_stride as u64,
            b_stride: b_stride as u64,
            rank: self.rank as u32,
            k: self.k as u32,
            n: self.n as u32,
            dtype: self.dtype,
            count: count as u32,
            scale: self.scale,
        }
    }
}
#[derive(Debug)]
pub(in crate::glm53) struct LayerLora {
    targets: BTreeMap<String, Vec<Adapter>>,
    /// rows x max rank F32.
    t: DeviceBuffer,
}
impl LayerLora {
    fn bytes(&self) -> usize {
        self.targets
            .values()
            .flatten()
            .map(|a| a.a.bytes() + a.b.bytes())
            .sum::<usize>()
            + self.t.bytes()
    }
}

impl Layer<'_> {
    /// Uploads this layer's adapters (before any graph is captured).
    pub(in crate::glm53) fn set_lora(&mut self, adapters: &host::Layer) -> Result<()> {
        if adapters.is_empty() {
            self.lora = None;
            return Ok(());
        }
        if self.hadamard_block != 0 || self.verification.is_some() || self.before_graph.is_some() {
            return Err(Error::InvalidPlan(
                "GLM LoRA needs unrotated experts, no MTP verification and no captured graphs"
                    .into(),
            ));
        }
        let mut targets = BTreeMap::new();
        let mut rank = 0;
        for (name, pairs) in adapters {
            if !SUPPORTED.contains(&name.as_str())
                || (name != ROUTED && !self.weights.contains_key(name))
                || (name == ROUTED && self.index < 3)
            {
                return Err(Error::InvalidPlan(format!(
                    "GLM LoRA target {name} is not supported in layer {}",
                    self.index
                )));
            }
            let mut uploaded = Vec::new();
            for p in pairs {
                let upload = |bytes: &[u8]| -> Result<DeviceBuffer> {
                    let mut buffer = self.device.alloc(bytes.len())?;
                    self.stream.copy_h2d(&mut buffer, bytes)?;
                    Ok(buffer)
                };
                rank = rank.max(p.rank);
                uploaded.push(Adapter {
                    a: upload(&p.a)?,
                    b: upload(&p.b)?,
                    rank: p.rank,
                    k: p.k,
                    n: p.n,
                    count: p.count,
                    dtype: p.dtype.code(),
                    scale: p.scale,
                });
            }
            targets.insert(name.clone(), uploaded);
        }
        self.lora = Some(LayerLora {
            targets,
            t: self.device.alloc(ROWS * rank * 4)?,
        });
        Ok(())
    }
    pub(in crate::glm53) fn lora_bytes(&self) -> usize {
        self.lora.as_ref().map_or(0, LayerLora::bytes)
    }
    pub(in crate::glm53) fn has_lora(&self) -> bool {
        self.lora.is_some()
    }
    /// `routing`: per-row experts (i32) and images (null: none / skipped row).
    fn call(
        &self,
        st: RawStream,
        lora: g::GlmLora,
        out: Ptr,
        x: Ptr,
        rows: usize,
        (experts, images): (Ptr, Ptr),
    ) -> Result<()> {
        let l = self.lora.as_ref().unwrap();
        if rows > ROWS {
            return Err(Error::InvalidShape("GLM LoRA rows".into()));
        }
        check(
            unsafe {
                g::hayate_glm_lora(
                    out,
                    x,
                    l.t.as_ptr().cast_mut(),
                    lora,
                    rows as u32,
                    experts,
                    images,
                    st,
                )
            },
            "glm LoRA",
        )
    }
    /// After the base linear `weight` wrote `rows` rows of `out` from `x`. A routed
    /// expert's own weight name (`mlp.experts.E.down_proj.weight`, reference paths)
    /// takes that expert's matrices.
    pub(super) fn lora_linear(
        &self,
        st: RawStream,
        weight: &str,
        out: Ptr,
        x: Ptr,
        rows: usize,
    ) -> Result<()> {
        let Some(l) = &self.lora else {
            return Ok(());
        };
        let expert = weight
            .strip_prefix("mlp.experts.")
            .and_then(|s| s.strip_suffix(".down_proj.weight"))
            .and_then(|e| e.parse::<usize>().ok());
        let (key, expert) = match expert {
            Some(e) if e < EXPERTS => (ROUTED, Some(e)),
            _ => (weight, None),
        };
        for adapter in l.targets.get(key).into_iter().flatten() {
            self.call(st, adapter.ffi(expert), out, x, rows, NONE)?;
        }
        Ok(())
    }
    /// Decode MoE outputs `expert` [9][4096] from SwiGLU products `product` [9][2048]:
    /// routed slots 0..8 by `ids` (skipping CPU slots, null `images`), shared slot 8.
    pub(super) fn lora_experts(
        &self,
        st: RawStream,
        product: Ptr,
        expert: Ptr,
        ids: Ptr,
        images: Ptr,
    ) -> Result<()> {
        let Some(l) = &self.lora else {
            return Ok(());
        };
        for adapter in l.targets.get(ROUTED).into_iter().flatten() {
            self.call(st, adapter.ffi(None), expert, product, TOP_K, (ids, images))?;
        }
        let (x, out) = unsafe {
            (
                product.cast::<u8>().add(TOP_K * 2048 * 2).cast(),
                expert.cast::<u8>().add(TOP_K * HIDDEN * 2).cast(),
            )
        };
        self.lora_linear(st, "mlp.shared_experts.down_proj.weight", out, x, 1)
    }
    /// Grouped prefill: expert `experts[j]` owns the `ranges[2j]` rows from `ranges[2j + 1]`
    /// of `product` -> `expert` (one call per expert, no host synchronization).
    pub(super) fn lora_grouped(
        &self,
        product: Ptr,
        expert: Ptr,
        experts: &[usize],
        ranges: &[u32],
    ) -> Result<()> {
        let Some(l) = &self.lora else {
            return Ok(());
        };
        for adapter in l.targets.get(ROUTED).into_iter().flatten() {
            for (j, &e) in experts.iter().enumerate() {
                let (count, offset) = (ranges[2 * j] as usize, ranges[2 * j + 1] as usize);
                if count == 0 {
                    continue;
                }
                let (x, out) = unsafe {
                    (
                        product.cast::<u8>().add(offset * 2048 * 2).cast(),
                        expert.cast::<u8>().add(offset * HIDDEN * 2).cast(),
                    )
                };
                let st = self.stream.as_raw();
                self.call(st, adapter.ffi(Some(e)), out, x, count, NONE)?;
            }
        }
        Ok(())
    }
}
