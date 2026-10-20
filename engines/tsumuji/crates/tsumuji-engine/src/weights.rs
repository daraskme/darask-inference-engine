//! Loads a RadixArk-layout checkpoint (routed experts NVFP4, PLE table FP8, the rest BF16) into
//! device memory, plus the PLE table into pinned host memory. Projections that read the same
//! input are concatenated into one matrix so one GEMV serves them:
//!   GR       [input_mix_weight_down | block_inject_weight]
//!   GDN      [in_proj_qkv | in_proj_z | in_proj_b | in_proj_a]
//!   QSA      [q_proj | k_proj | v_proj | indexer.index_qk_proj]
//!   router   [mlp.gate | shared_expert_gate]
//!   shared   [gate_proj | up_proj]
//!   PLE      [key_proj | value_proj]
//! Routed experts of one layer live in one buffer, expert e at e * stride:
//!   [gate codes | up codes | down codes | gate scales | up scales | down scales]
//! with stride = 27/16 * F * H bytes, and their global scales in `gs` [E][3] (gate, up, down).

use crate::config::Dims;
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::sync_channel;
use tsumuji_cuda::{
    DeviceBuffer, Function, LaunchDims, MappedHostBuffer, Module, Stream, args, launch,
    synchronize_context,
};
use tsumuji_model::ple::Tables;
use tsumuji_pack::safetensors::{TensorInfo, read_dir};

pub type Res<T> = Result<T, String>;

fn cu<T>(r: tsumuji_cuda::Result<T>) -> Res<T> {
    r.map_err(|e| e.to_string())
}

/// A dense matrix [rows][cols]: bf16 (fmt 0), or 8-bit with one ue8m0 scale byte per 32 of a row in
/// `sc` — MXFP8 (fmt 1, e4m3) or MXINT8 (fmt 2, int8) — when its category is listed in TSJ_DENSE
/// (see `Quant`).
#[derive(Debug)]
pub struct Dense {
    pub w: DeviceBuffer,
    pub sc: Option<DeviceBuffer>,
    pub fmt: u8,
}

impl Dense {
    /// Elements per 16-byte vector.
    pub fn vw(&self) -> usize {
        if self.fmt == 0 { 8 } else { 16 }
    }

    pub fn sc_ptr(&self) -> u64 {
        self.sc.as_ref().map_or(0, |b| b.ptr)
    }
}

/// Load-time bf16 -> 8-bit conversion (kernels/decode/quant.cu) for the categories in TSJ_DENSE:
/// a comma list of gr, gdn, qsa, shared, ple, lm_head, mtp, router, "all" (all but router; the default)
/// or "none" (bf16 everywhere). `mtp` covers the whole MTP layer (its GR, QSA and shared expert too):
/// without it the MTP layer stays bf16. TSJ_DENSE_FMT picks the format: int8 (MXINT8, default) or fp8
/// (MXFP8). MXINT8 costs a third to a half of MXFP8's KL at the same bytes (docs/m2.md).
#[derive(Debug)]
struct Quant {
    _m: Module,
    f: Function,
    fmt: u8,
    stream: Stream,
    cats: Vec<String>,
}

impl Quant {
    fn new() -> Res<Self> {
        let m = cu(Module::load(tsumuji_kernels::DECODE_QUANT))?;
        let fmt = match std::env::var("TSJ_DENSE_FMT").as_deref() {
            Ok("fp8") => 1,
            Ok("int8") | Err(_) => 2,
            Ok(other) => return Err(format!("TSJ_DENSE_FMT={other}: expected int8 or fp8")),
        };
        let f = cu(m.function(if fmt == 1 {
            "quant_mxfp8"
        } else {
            "quant_mxint8"
        }))?;
        let spec = std::env::var("TSJ_DENSE").unwrap_or_else(|_| "all".into());
        let mut cats: Vec<String> = spec
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if cats.iter().any(|c| c == "all") {
            cats = ["gr", "gdn", "qsa", "shared", "ple", "lm_head", "mtp"]
                .map(String::from)
                .to_vec();
        }
        Ok(Self {
            _m: m,
            f,
            fmt,
            stream: cu(Stream::new())?,
            cats,
        })
    }

    /// The conversion for the MTP layer: everything enabled when `mtp` is, else nothing.
    fn for_mtp(&self) -> Res<Self> {
        let mut q = Self::new()?;
        if !self.cats.iter().any(|c| c == "mtp") {
            q.cats.clear();
        }
        Ok(q)
    }

    /// Wraps uploaded rows with `cols` columns: MXFP8 as stored, or bf16, converted when `cat` is
    /// enabled.
    fn dense(&self, cat: &str, rows: Rows, cols: usize) -> Res<Dense> {
        let bf16 = match rows {
            Rows::Mxfp8(w, sc) => {
                return Ok(Dense {
                    w,
                    sc: Some(sc),
                    fmt: 1,
                });
            }
            Rows::Bf16(b) => b,
        };
        if !self.cats.iter().any(|c| c == cat) {
            return Ok(Dense {
                w: bf16,
                sc: None,
                fmt: 0,
            });
        }
        assert!(
            cols.is_multiple_of(32),
            "MXFP8 needs rows of 32-element blocks"
        );
        let n = bf16.bytes / 2;
        let (q, sc) = (cu(DeviceBuffer::new(n))?, cu(DeviceBuffer::new(n / 32))?);
        let blocks = (n / 32) as u64;
        // The bf16 upload ran on the legacy stream and may still be in flight; this stream does not
        // wait for it (reading it early quantized stale data: results changed from run to run).
        cu(synchronize_context())?;
        let mut a = args!(bf16.ptr, q.ptr, sc.ptr, blocks);
        cu(unsafe {
            launch(
                self.f,
                LaunchDims::linear(blocks.div_ceil(256) as u32, 256),
                &self.stream,
                false,
                &mut a,
            )
        })?;
        cu(self.stream.synchronize())?;
        Ok(Dense {
            w: q,
            sc: Some(sc),
            fmt: self.fmt,
        })
    }
}

/// Rows as read from the checkpoint.
enum Rows {
    Bf16(DeviceBuffer),
    /// e4m3 values and ue8m0 scales per 32.
    Mxfp8(DeviceBuffer, DeviceBuffer),
}

impl Rows {
    fn bf16(self) -> Res<DeviceBuffer> {
        match self {
            Rows::Bf16(b) => Ok(b),
            Rows::Mxfp8(..) => Err("expected bf16 rows".into()),
        }
    }
}

#[derive(Debug)]
pub struct Gr {
    /// f32 [HC * H], the (1 + w) RMSNorm weights of the four streams.
    pub norm: DeviceBuffer,
    /// bf16 [rank + HC][HC * H] ([rank][HC * H] for the final mixer, which has no inject).
    pub down_inj: Dense,
    /// bf16 [HC * H][rank].
    pub up: Dense,
}

#[derive(Debug)]
pub struct Gdn {
    pub in_proj: Dense,
    /// f32 [C][4].
    pub conv: DeviceBuffer,
    pub a_log: DeviceBuffer,
    pub dt_bias: DeviceBuffer,
    pub norm: DeviceBuffer,
    pub out_proj: Dense,
}

#[derive(Debug)]
pub struct Qsa {
    pub proj: Dense,
    pub q_norm: DeviceBuffer,
    pub k_norm: DeviceBuffer,
    pub iq_norm: DeviceBuffer,
    pub ik_norm: DeviceBuffer,
    pub o_proj: Dense,
}

#[derive(Debug)]
pub enum Mixer {
    Gdn(Gdn),
    Qsa(Qsa),
}

#[derive(Debug)]
pub struct Moe {
    pub router: Dense,
    pub shared_gu: Dense,
    pub shared_down: Dense,
    pub experts: DeviceBuffer,
    pub gs: DeviceBuffer,
}

/// Mirrors `PleHash` in kernels/decode/ple.cu.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PleHash {
    pub mult: [u64; 3],
    pub size: [u64; 16],
    pub offset: [u64; 16],
}

#[derive(Debug)]
pub struct Ple {
    pub kv_proj: Dense,
    pub norm_key: DeviceBuffer,
    pub norm_query: DeviceBuffer,
    pub norm_conv: DeviceBuffer,
    /// f32 [HC * H][4].
    pub conv: DeviceBuffer,
    /// The dequantization scale: bf16 [1] for FP8 rows, the f32 global scale for NVFP4 rows.
    pub scale: DeviceBuffer,
    pub hash: PleHash,
    /// Rows of `row_bytes` bytes, mapped for zero-copy reads: FP8 values, or NVFP4 codes then their
    /// e4m3 scales per 16 (`nvfp4`).
    pub table: MappedHostBuffer,
    pub nvfp4: bool,
    pub row_bytes: usize,
}

#[derive(Debug)]
pub struct Layer {
    pub attn_gr: Gr,
    pub mixer: Mixer,
    pub mlp_gr: Gr,
    pub moe: Moe,
}

#[derive(Debug)]
pub struct Weights {
    pub embed: DeviceBuffer,
    pub lm_head: Dense,
    pub final_gr: Gr,
    pub layers: Vec<Layer>,
    pub ple: Option<Ple>,
    /// The MTP layer for drafts (TSJ_MTP=0 skips it).
    pub mtp: Option<Mtp>,
    /// The drafts' own NVFP4 copy of lm_head (with the MTP layer; TSJ_DRAFT_HEAD=int8 drafts with
    /// the main lm_head instead).
    pub draft_head: Option<DraftHead>,
}

/// An NVFP4 matrix [rows][cols]: e2m1 codes [rows][cols / 2] (low nibble first) then e4m3 scales
/// [rows][cols / 16] in `w` (`so` bytes in), and one fp32 global scale `gs`; quantized on the GPU
/// like the MTP experts (round to nearest, ModelOpt's scales).
#[derive(Debug)]
pub struct DraftHead {
    pub w: DeviceBuffer,
    pub so: usize,
    pub gs: DeviceBuffer,
}

impl DraftHead {
    /// Quantizes bf16 rows [rows][cols] already on the device.
    fn quantize(bf16: &DeviceBuffer, rows: usize, cols: usize) -> Res<Self> {
        let m = cu(Module::load(tsumuji_kernels::DECODE_QUANT))?;
        let (amax_f, quant_f) = (
            cu(m.function("nvfp4_amax"))?,
            cu(m.function("nvfp4_quant"))?,
        );
        let stream = cu(Stream::new())?;
        let n = rows * cols;
        let so = n / 2;
        let (w, gs, amax) = (
            cu(DeviceBuffer::zeroed(so + n / 16))?,
            cu(DeviceBuffer::zeroed(4))?,
            cu(DeviceBuffer::zeroed(4))?,
        );
        cu(synchronize_context())?; // the upload and memsets ran on the legacy stream
        let (zero, nn) = (0u64, n as u64);
        let mut a = args!(bf16.ptr, zero, nn, amax.ptr);
        cu(unsafe {
            launch(
                amax_f,
                LaunchDims {
                    grid: (1024, 1, 1),
                    block: (256, 1, 1),
                    smem: 0,
                },
                &stream,
                false,
                &mut a,
            )
        })?;
        let (ri, ci, sz, gstride) = (rows as i32, cols as i32, so as u64, 0i32);
        let mut a = args!(
            bf16.ptr, zero, ri, ci, amax.ptr, w.ptr, zero, zero, sz, gs.ptr, gstride
        );
        let blocks = (n / 16).div_ceil(256) as u32;
        cu(unsafe {
            launch(
                quant_f,
                LaunchDims {
                    grid: (blocks, 1, 1),
                    block: (256, 1, 1),
                    smem: 0,
                },
                &stream,
                false,
                &mut a,
            )
        })?;
        cu(stream.synchronize())?;
        Ok(Self { w, so, gs })
    }
}

pub fn expert_stride(d: &Dims) -> usize {
    d.ff * d.h * 27 / 16
}

struct Ckpt {
    t: HashMap<String, TensorInfo>,
    text: String,
}

impl Ckpt {
    fn get(&self, name: &str) -> Res<&TensorInfo> {
        self.t.get(name).ok_or(format!("missing tensor {name}"))
    }

    /// A text-model tensor (`model.language_model.` or `model.` prefix).
    fn txt(&self, name: &str) -> Res<&TensorInfo> {
        self.get(&format!("{}{name}", self.text))
    }

    fn check(t: &TensorInfo, dtype: &str, shape: &[usize]) -> Res<()> {
        let s: Vec<usize> = t.shape.iter().map(|&x| x as usize).collect();
        if t.dtype != dtype || s != shape {
            return Err(format!(
                "{}: {} {:?}, expected {dtype} {shape:?}",
                t.name, t.dtype, s
            ));
        }
        Ok(())
    }

    /// Matrices with `cols` columns, concatenated by rows, uploaded: MXFP8 as stored when every
    /// part is (e4m3 `weight` + ue8m0 `weight_scale` per 32; nvidia's FP4-Local), else bf16, with
    /// FP8 parts that carry a per-channel or per-tensor scale (compressed-tensors, e.g. orcarouter)
    /// dequantized to bf16 first (`fp8_to_bf16`).
    fn bf16_rows(&self, parts: &[(&str, usize)], cols: usize) -> Res<Rows> {
        let io = |e: std::io::Error| e.to_string();
        let scale_of = |name: &str| -> Res<&TensorInfo> {
            let base = name
                .strip_suffix(".weight")
                .ok_or(format!("{name}: not a .weight"))?;
            self.txt(&format!("{base}.weight_scale"))
        };
        let mx = parts.iter().all(|&(name, _)| {
            self.txt(name).is_ok_and(|t| t.dtype == "F8_E4M3")
                && scale_of(name).is_ok_and(|s| s.dtype == "U8")
        });
        if mx {
            let (mut w, mut sc) = (Vec::new(), Vec::new());
            for &(name, rows) in parts {
                let t = self.txt(name)?;
                Self::check(t, "F8_E4M3", &[rows, cols])?;
                w.extend(t.read().map_err(io)?);
                let s = scale_of(name)?;
                Self::check(s, "U8", &[rows, cols / 32])?;
                sc.extend(s.read().map_err(io)?);
            }
            return Ok(Rows::Mxfp8(
                cu(DeviceBuffer::from_slice(&w))?,
                cu(DeviceBuffer::from_slice(&sc))?,
            ));
        }
        let mut bytes = Vec::new();
        for &(name, rows) in parts {
            let t = self.txt(name)?;
            if t.dtype == "F8_E4M3" {
                Self::check(t, "F8_E4M3", &[rows, cols])?;
                bytes.extend(fp8_to_bf16(
                    &t.read().map_err(io)?,
                    scale_of(name)?,
                    rows,
                    cols,
                )?);
            } else {
                Self::check(t, "BF16", &[rows, cols])?;
                bytes.extend(t.read().map_err(io)?);
            }
        }
        Ok(Rows::Bf16(cu(DeviceBuffer::from_slice(&bytes))?))
    }

    /// One tensor as f32 (from BF16 or F32), uploaded; `len` elements.
    fn f32v(&self, name: &str, len: usize) -> Res<DeviceBuffer> {
        let t = self.txt(name)?;
        let v = t.read_as_f32().map_err(|e| e.to_string())?;
        if v.len() != len {
            return Err(format!("{name}: {} values, expected {len}", v.len()));
        }
        cu(DeviceBuffer::from_slice(&v))
    }

    fn gr(&self, pre: &str, d: &Dims, inject: bool, q: &Quant) -> Res<Gr> {
        let hcw = d.hcw();
        let mut parts = vec![(format!("{pre}.input_mix_weight_down.weight"), d.rank)];
        if inject {
            parts.push((format!("{pre}.block_inject_weight.weight"), d.hc));
        }
        let parts: Vec<(&str, usize)> = parts.iter().map(|(n, r)| (n.as_str(), *r)).collect();
        Ok(Gr {
            norm: self.f32v(&format!("{pre}.hc_norm.weight"), hcw)?,
            down_inj: q.dense("gr", self.bf16_rows(&parts, hcw)?, hcw)?,
            up: q.dense(
                "gr",
                self.bf16_rows(
                    &[(&format!("{pre}.input_mix_weight_up.weight"), hcw)],
                    d.rank,
                )?,
                d.rank,
            )?,
        })
    }
}

/// Packs the routed experts under `pre` (`layers.{l}.mlp.experts`; host side): returns (expert
/// buffer, gs [E][3]). Block scales are e4m3 tensors, or nvidia's CSF-compressed form (`csf_decode`).
fn pack_experts(ck: &Ckpt, d: &Dims, pre: &str) -> Res<(Vec<u8>, Vec<f32>)> {
    let (f, h) = (d.ff, d.h);
    let fh = f * h;
    let stride = expert_stride(d);
    let mut buf = vec![0u8; d.experts * stride];
    let mut gs = vec![0f32; d.experts * 3];
    let io = |e: std::io::Error| e.to_string();
    for e in 0..d.experts {
        let base = e * stride;
        // (projection, rows, cols, code offset, scale offset)
        let projs = [
            ("gate_proj", f, h, 0, fh * 3 / 2),
            ("up_proj", f, h, fh / 2, fh * 3 / 2 + fh / 16),
            ("down_proj", h, f, fh, fh * 3 / 2 + fh / 8),
        ];
        for (pi, &(proj, rows, cols, co, so)) in projs.iter().enumerate() {
            let p = format!("{pre}.{e}.{proj}");
            // Codes: `weight` (ModelOpt: RadixArk, nvidia) or `weight_packed` (compressed-tensors:
            // RedHatAI, orcarouter), both two e2m1 per byte, low nibble first.
            let packed = ck.txt(&format!("{p}.weight_packed")).is_ok();
            let w = ck.txt(&format!(
                "{p}.{}",
                if packed { "weight_packed" } else { "weight" }
            ))?;
            Ckpt::check(w, "U8", &[rows, cols / 2])?;
            w.read_into(&mut buf[base + co..base + co + rows * cols / 2])
                .map_err(io)?;
            let scales = &mut buf[base + so..base + so + rows * cols / 16];
            match ck.txt(&format!("{p}.weight_scale.nvfp4_csf_fixed")) {
                Ok(fixed) => {
                    Ckpt::check(fixed, "U8", &[rows / 16, 16 + 16 * cols / 32])?;
                    let exc = ck.txt(&format!("{p}.weight_scale.nvfp4_csf_exceptions"))?;
                    csf_decode(
                        &fixed.read().map_err(io)?,
                        &exc.read().map_err(io)?,
                        rows,
                        cols / 16,
                        scales,
                    )?;
                }
                Err(_) => {
                    let s = ck.txt(&format!("{p}.weight_scale"))?;
                    Ckpt::check(s, "F8_E4M3", &[rows, cols / 16])?;
                    s.read_into(scales).map_err(io)?;
                }
            }
            // Global scale: ModelOpt multiplies by `weight_scale_2`; compressed-tensors divides by
            // `weight_global_scale` (checked against the bf16 original, docs/formats.md).
            gs[e * 3 + pi] = if packed {
                1.0 / ck
                    .txt(&format!("{p}.weight_global_scale"))?
                    .read_f32()
                    .map_err(io)?[0]
            } else {
                ck.txt(&format!("{p}.weight_scale_2"))?
                    .read_f32()
                    .map_err(io)?[0]
            };
        }
    }
    Ok((buf, gs))
}

/// nvidia's lossless CSF form of NVFP4 block scales (`weight_scale.nvfp4_csf_*`; reverse-engineered
/// and checked against the bf16 original, docs/m3.md): per 16 rows, 16 base bytes (one per row),
/// then each row's 4-bit offsets (low nibble first); scale byte = base + offset, except at the
/// positions listed in `exceptions` (u32: byte << 24 | flat index), which hold the byte itself.
/// `out` is [rows][ns] e4m3 bytes.
fn csf_decode(fixed: &[u8], exceptions: &[u8], rows: usize, ns: usize, out: &mut [u8]) -> Res<()> {
    let stride = 16 + 16 * ns / 2;
    if fixed.len() != rows / 16 * stride || out.len() != rows * ns {
        return Err(format!(
            "CSF scales: {} bytes for {rows} x {ns}",
            fixed.len()
        ));
    }
    for (g, blk) in fixed.chunks_exact(stride).enumerate() {
        for r in 0..16 {
            let base = blk[r];
            let row = &mut out[(g * 16 + r) * ns..][..ns];
            for (i, &b) in blk[16 + r * ns / 2..][..ns / 2].iter().enumerate() {
                row[2 * i] = base.wrapping_add(b & 15);
                row[2 * i + 1] = base.wrapping_add(b >> 4);
            }
        }
    }
    for x in exceptions.chunks_exact(4) {
        let v = u32::from_le_bytes([x[0], x[1], x[2], x[3]]);
        let at = (v & 0xff_ffff) as usize;
        *out.get_mut(at)
            .ok_or(format!("CSF exception index {at} out of range"))? = (v >> 24) as u8;
    }
    Ok(())
}

/// The PLE layer: projections from `ck`, the n-gram table (and its hash tables) from `tk` (the same
/// checkpoint, or another one with the identical table, see `load`).
fn load_ple(ck: &Ckpt, tk: &Ckpt, d: &Dims, l: usize, q: &Quant) -> Res<Ple> {
    let hcw = d.hcw();
    let pre = format!("layers.{l}.ple");
    let tables = Tables::new(d.vocab as u64, d.ngram_base, 0, d.ple_seed);
    // The checkpoint stores the hash tables too; they must agree with ours.
    let io = |e: std::io::Error| e.to_string();
    let stored = |n: &str| -> Res<Vec<u64>> {
        Ok(tk
            .txt(&format!("{pre}.ple_embedding.{n}"))?
            .read_i64()
            .map_err(io)?
            .into_iter()
            .map(|v| v as u64)
            .collect())
    };
    if stored("layer_multipliers")? != tables.mult
        || stored("ngram_heads_vocab_sizes")? != tables.size
        || stored("ngram_heads_offsets")? != tables.offset
    {
        return Err("PLE hash tables in the checkpoint differ from tsumuji-model's".into());
    }
    let mut shards: Vec<&TensorInfo> = Vec::new();
    for i in 0.. {
        match tk.txt(&format!(
            "{pre}.ple_embedding.ngram_embedding.shard_{i}.weight"
        )) {
            Ok(t) => shards.push(t),
            Err(_) => break,
        }
    }
    if shards.is_empty() {
        return Err("PLE table: no FP8 or NVFP4 shards (a BF16 table is not supported)".into());
    }
    let rows: u64 = shards.iter().map(|s| s.shape[0]).sum();
    if rows != tables.padded_rows {
        return Err(format!(
            "PLE table has {rows} rows, expected {}",
            tables.padded_rows
        ));
    }
    // FP8 rows (RadixArk) as stored; NVFP4 rows (nvidia) as [codes W/2 | e4m3 scales W/16 | pad]
    // in 16-byte units, so one row is one contiguous read.
    let nvfp4 = shards[0].dtype == "U8";
    let row_bytes = if nvfp4 {
        (d.ple_row / 2 + d.ple_row / 16).div_ceil(16) * 16
    } else {
        d.ple_row
    };
    let mut parts = Vec::new();
    for (i, s) in shards.iter().enumerate() {
        let n = s.shape[0] as usize;
        if nvfp4 {
            Ckpt::check(s, "U8", &[n, d.ple_row / 2])?;
            let sc = tk.txt(&format!(
                "{pre}.ple_embedding.ngram_embedding.shard_{i}.weight_scale"
            ))?;
            Ckpt::check(sc, "F8_E4M3", &[n, d.ple_row / 16])?;
            parts.push((*s, Some(sc), n));
        } else {
            Ckpt::check(s, "F8_E4M3", &[n, d.ple_row])?;
            parts.push((*s, None, n));
        }
    }
    let mut table = cu(MappedHostBuffer::new((rows as usize) * row_bytes))?;
    // Fill the pinned table from all shards in parallel (disjoint slices).
    {
        let mut slices = Vec::new();
        let mut rest = table.as_mut_slice();
        for &(s, sc, n) in &parts {
            let (a, b) = rest.split_at_mut(n * row_bytes);
            slices.push((s, sc, a));
            rest = b;
        }
        let threads = 8.min(slices.len());
        let mut groups: Vec<Vec<(&TensorInfo, Option<&TensorInfo>, &mut [u8])>> =
            (0..threads).map(|_| Vec::new()).collect();
        for (i, s) in slices.into_iter().enumerate() {
            groups[i % threads].push(s);
        }
        let (cb, sb) = (d.ple_row / 2, d.ple_row / 16);
        let fill = move |(t, sc, dst): (&TensorInfo, Option<&TensorInfo>, &mut [u8])| -> Res<()> {
            let io = |e: std::io::Error| e.to_string();
            let Some(sc) = sc else {
                return t.read_into(dst).map_err(io);
            };
            let (codes, scales) = (t.read().map_err(io)?, sc.read().map_err(io)?);
            for (r, row) in dst.chunks_exact_mut(row_bytes).enumerate() {
                row[..cb].copy_from_slice(&codes[r * cb..][..cb]);
                row[cb..cb + sb].copy_from_slice(&scales[r * sb..][..sb]);
            }
            Ok(())
        };
        let results: Vec<Res<()>> = std::thread::scope(|scope| {
            let hs: Vec<_> = groups
                .into_iter()
                .map(|g| scope.spawn(move || g.into_iter().try_for_each(fill)))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        results.into_iter().collect::<Res<()>>()?;
    }
    // FP8: one bf16 dequantization scale; NVFP4: the f32 global scale (block scales per row).
    let scale = if nvfp4 {
        let g = tk
            .txt(&format!(
                "{pre}.ple_embedding.ngram_embedding.weight_scale_2"
            ))?
            .read_f32()
            .map_err(io)?;
        cu(DeviceBuffer::from_slice(&g[..1]))?
    } else {
        let s = tk.txt(&format!("{pre}.ple_embedding.ngram_embedding.weight_scale"))?;
        Ckpt::check(s, "BF16", &[1])?;
        cu(DeviceBuffer::from_slice(&s.read().map_err(io)?))?
    };
    Ok(Ple {
        kv_proj: q.dense(
            "ple",
            ck.bf16_rows(
                &[
                    (&format!("{pre}.key_proj.weight"), hcw),
                    (&format!("{pre}.value_proj.weight"), d.h),
                ],
                d.ple_heads * d.ple_row,
            )?,
            d.ple_heads * d.ple_row,
        )?,
        norm_key: ck.f32v(&format!("{pre}.norm_key.weight"), hcw)?,
        norm_query: ck.f32v(&format!("{pre}.norm_query.weight"), hcw)?,
        norm_conv: ck.f32v(&format!("{pre}.norm_conv.weight"), hcw)?,
        conv: ck.f32v(&format!("{pre}.conv1d.weight"), hcw * 4)?,
        scale,
        nvfp4,
        row_bytes,
        hash: PleHash {
            mult: tables.mult,
            size: tables.size,
            offset: tables.offset,
        },
        table,
    })
}

/// Loads everything. `progress` gets one line per finished stage.
pub fn load(dir: &Path, d: &Dims, mut progress: impl FnMut(&str)) -> Res<Weights> {
    let t: HashMap<String, TensorInfo> = read_dir(dir)?
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect();
    let text = if t.contains_key("model.language_model.embed_tokens.weight") {
        "model.language_model."
    } else {
        "model."
    };
    let ck = Ckpt {
        t,
        text: text.to_string(),
    };
    // Experiments: routed experts from another checkpoint (TSJ_EXPERTS=DIR), e.g. nvidia's QAD experts
    // with the rest from RadixArk (docs/formats.md).
    let ex_ck = match std::env::var("TSJ_EXPERTS") {
        Ok(other) => {
            let t: HashMap<String, TensorInfo> = read_dir(Path::new(&other))?
                .into_iter()
                .map(|t| (t.name.clone(), t))
                .collect();
            let text = if t.contains_key("model.language_model.embed_tokens.weight") {
                "model.language_model."
            } else {
                "model."
            };
            progress(&format!("routed experts from {other}"));
            Some(Ckpt {
                t,
                text: text.to_string(),
            })
        }
        Err(_) => None,
    };
    let quant = Quant::new()?;
    if !quant.cats.is_empty() {
        progress(&format!(
            "dense {} for: {}",
            if quant.fmt == 1 { "MXFP8" } else { "MXINT8" },
            quant.cats.join(", ")
        ));
    }
    let h = d.h;
    let t0 = std::time::Instant::now();

    // tsumuji.json in the model directory may name another checkpoint for the PLE n-gram table
    // ({"ple_from": DIR}): fine-tunes that keep the table (abliterations) can then ship without
    // it, and the FP8 table of another quantization of the same base model is used instead.
    let ple_ck = match std::fs::read_to_string(dir.join("tsumuji.json")) {
        Ok(text) => match tsumuji_pack::json::Json::parse(&text)?
            .get("ple_from")
            .and_then(|v| v.as_str())
        {
            Some(other) => {
                progress(&format!("PLE table from {other}"));
                let t: HashMap<String, TensorInfo> = read_dir(Path::new(other))?
                    .into_iter()
                    .map(|t| (t.name.clone(), t))
                    .collect();
                let text = if t.contains_key("model.language_model.embed_tokens.weight") {
                    "model.language_model."
                } else {
                    "model."
                };
                Some(Ckpt {
                    t,
                    text: text.to_string(),
                })
            }
            None => None,
        },
        Err(_) => None,
    };
    let ple = match d.ple_layer {
        Some(l) => Some(load_ple(&ck, ple_ck.as_ref().unwrap_or(&ck), d, l, &quant)?),
        None => None,
    };
    progress(&format!(
        "PLE table in pinned RAM ({:.1} s)",
        t0.elapsed().as_secs_f64()
    ));

    // Routed experts: pack layers on worker threads, upload on this one (bounded in flight).
    let (tx, rx) = sync_channel::<(usize, Res<(Vec<u8>, Vec<f32>)>)>(3);
    let mut experts: Vec<Option<(DeviceBuffer, DeviceBuffer)>> =
        (0..d.layers).map(|_| None).collect();
    std::thread::scope(|sc| -> Res<()> {
        let workers = 6.min(d.layers);
        for w in 0..workers {
            let tx = tx.clone();
            let ck = ex_ck.as_ref().unwrap_or(&ck);
            sc.spawn(move || {
                for l in (w..d.layers).step_by(workers) {
                    if tx
                        .send((l, pack_experts(ck, d, &format!("layers.{l}.mlp.experts"))))
                        .is_err()
                    {
                        return;
                    }
                }
            });
        }
        drop(tx);
        for (l, r) in rx {
            let (buf, gs) = r?;
            experts[l] = Some((
                cu(DeviceBuffer::from_slice(&buf))?,
                cu(DeviceBuffer::from_slice(&gs))?,
            ));
        }
        Ok(())
    })?;
    progress(&format!(
        "routed experts in VRAM ({:.1} s)",
        t0.elapsed().as_secs_f64()
    ));

    let mut layers = Vec::with_capacity(d.layers);
    for (l, ex) in experts.into_iter().enumerate() {
        let p = format!("layers.{l}");
        let mixer = if d.qsa[l] {
            Mixer::Qsa(load_qsa(&ck, &format!("{p}.self_attn"), d, &quant)?)
        } else {
            let a = format!("{p}.linear_attn");
            let vd = d.gdn_v * d.gdn_d;
            Mixer::Gdn(Gdn {
                in_proj: quant.dense(
                    "gdn",
                    ck.bf16_rows(
                        &[
                            (&format!("{a}.in_proj_qkv.weight"), d.gdn_c()),
                            (&format!("{a}.in_proj_z.weight"), vd),
                            (&format!("{a}.in_proj_b.weight"), d.gdn_v),
                            (&format!("{a}.in_proj_a.weight"), d.gdn_v),
                        ],
                        h,
                    )?,
                    h,
                )?,
                conv: ck.f32v(&format!("{a}.conv1d.weight"), d.gdn_c() * 4)?,
                a_log: ck.f32v(&format!("{a}.A_log"), d.gdn_v)?,
                dt_bias: ck.f32v(&format!("{a}.dt_bias"), d.gdn_v)?,
                norm: ck.f32v(&format!("{a}.norm.weight"), d.gdn_d)?,
                out_proj: quant.dense(
                    "gdn",
                    ck.bf16_rows(&[(&format!("{a}.out_proj.weight"), h)], vd)?,
                    vd,
                )?,
            })
        };
        let (ex, gs) = ex.ok_or(format!("layer {l}: experts missing"))?;
        layers.push(Layer {
            attn_gr: ck.gr(&format!("{p}.attn_hyper_connection"), d, true, &quant)?,
            mixer,
            mlp_gr: ck.gr(&format!("{p}.mlp_hyper_connection"), d, true, &quant)?,
            moe: load_moe(&ck, &format!("{p}.mlp"), d, &quant, ex, gs)?,
        });
    }
    let embed = ck
        .bf16_rows(&[("embed_tokens.weight", d.vocab)], h)?
        .bf16()?;
    let lm = ck.get("lm_head.weight")?;
    Ckpt::check(lm, "BF16", &[d.vocab, h])?;
    let lm_bf16 = cu(DeviceBuffer::from_slice(
        &lm.read().map_err(|e| e.to_string())?,
    ))?;
    let want_mtp = ck.t.contains_key("mtp.fc_embedding.weight")
        && std::env::var("TSJ_MTP").map_or(true, |v| v != "0");
    let draft_head = if want_mtp && std::env::var("TSJ_DRAFT_HEAD").map_or(true, |v| v != "int8") {
        Some(DraftHead::quantize(&lm_bf16, d.vocab, h)?)
    } else {
        None
    };
    let lm_head = quant.dense("lm_head", Rows::Bf16(lm_bf16), h)?;
    let final_gr = ck.gr("hyper_connection_mixer", d, false, &quant)?;
    progress(&format!(
        "dense weights in VRAM ({:.1} s)",
        t0.elapsed().as_secs_f64()
    ));
    let ck = Ckpt {
        t: ck.t,
        text: String::new(),
    }; // the MTP tensors are top-level (`mtp.`)
    let mtp = if want_mtp {
        let m = load_mtp(&ck, d, &quant.for_mtp()?)?;
        progress(&format!(
            "MTP layer in VRAM, experts NVFP4 ({:.1} s)",
            t0.elapsed().as_secs_f64()
        ));
        Some(m)
    } else {
        None
    };
    Ok(Weights {
        embed,
        lm_head,
        final_gr,
        layers,
        ple,
        mtp,
        draft_head,
    })
}

fn load_qsa(ck: &Ckpt, a: &str, d: &Dims, quant: &Quant) -> Res<Qsa> {
    let h = d.h;
    Ok(Qsa {
        proj: quant.dense(
            "qsa",
            ck.bf16_rows(
                &[
                    (&format!("{a}.q_proj.weight"), d.att_q * 2 * d.att_d),
                    (&format!("{a}.k_proj.weight"), d.att_kv * d.att_d),
                    (&format!("{a}.v_proj.weight"), d.att_kv * d.att_d),
                    (
                        &format!("{a}.indexer.index_qk_proj.weight"),
                        (d.idx_h + 1) * d.idx_d,
                    ),
                ],
                h,
            )?,
            h,
        )?,
        q_norm: ck.f32v(&format!("{a}.q_norm.weight"), d.att_d)?,
        k_norm: ck.f32v(&format!("{a}.k_norm.weight"), d.att_d)?,
        iq_norm: ck.f32v(&format!("{a}.indexer.q_layernorm.weight"), d.idx_d)?,
        ik_norm: ck.f32v(&format!("{a}.indexer.k_layernorm.weight"), d.idx_d)?,
        o_proj: quant.dense(
            "qsa",
            ck.bf16_rows(&[(&format!("{a}.o_proj.weight"), h)], d.att_q * d.att_d)?,
            d.att_q * d.att_d,
        )?,
    })
}

/// The MoE block at `m` (`layers.{l}.mlp`) around already packed routed experts.
fn load_moe(
    ck: &Ckpt,
    m: &str,
    d: &Dims,
    quant: &Quant,
    experts: DeviceBuffer,
    gs: DeviceBuffer,
) -> Res<Moe> {
    let h = d.h;
    Ok(Moe {
        router: quant.dense(
            "router",
            ck.bf16_rows(
                &[
                    (&format!("{m}.gate.weight"), d.experts),
                    (&format!("{m}.shared_expert_gate.weight"), 1),
                ],
                h,
            )?,
            h,
        )?,
        shared_gu: quant.dense(
            "shared",
            ck.bf16_rows(
                &[
                    (&format!("{m}.shared_expert.gate_proj.weight"), d.ff_shared),
                    (&format!("{m}.shared_expert.up_proj.weight"), d.ff_shared),
                ],
                h,
            )?,
            h,
        )?,
        shared_down: quant.dense(
            "shared",
            ck.bf16_rows(
                &[(&format!("{m}.shared_expert.down_proj.weight"), h)],
                d.ff_shared,
            )?,
            d.ff_shared,
        )?,
        experts,
        gs,
    })
}

/// The MTP layer (`mtp.*`): fuses the main model's residual streams at position p with the
/// embedding of token p + 1, runs one QSA + MoE layer and its own final mixer; the logits come from
/// the main lm_head (see engine.rs and docs/m3.md).
#[derive(Debug)]
pub struct Mtp {
    /// f32 [H]: (1 + w) RMSNorm of the token embedding.
    pub norm_emb: DeviceBuffer,
    /// f32 [HC * H]: (1 + w) RMSNorm over all four streams together.
    pub norm_hidden: DeviceBuffer,
    /// [H][H]: on the normed embedding.
    pub fc_emb: Dense,
    /// [H][H]: on each normed stream.
    pub fc_hidden: Dense,
    pub layer: Layer,
    pub final_gr: Gr,
}

/// The MTP layer's bf16 routed experts (fused gate_up [E][2F][H], down [E][H][F]) quantized to
/// NVFP4 on the GPU (kernels/decode/quant.cu) into the main layers' packed layout. Only the drafts
/// depend on them, so the rounding can only change the acceptance rate, never the output.
fn quant_experts_nvfp4(ck: &Ckpt, pre: &str, d: &Dims) -> Res<(DeviceBuffer, DeviceBuffer)> {
    let (e_n, f, h) = (d.experts, d.ff, d.h);
    let fh = f * h;
    let stride = expert_stride(d);
    let gu = ck.get(&format!("{pre}.gate_up_proj"))?;
    let dn = ck.get(&format!("{pre}.down_proj"))?;
    Ckpt::check(gu, "BF16", &[e_n, 2 * f, h])?;
    Ckpt::check(dn, "BF16", &[e_n, h, f])?;
    let m = cu(Module::load(tsumuji_kernels::DECODE_QUANT))?;
    let (amax_f, quant_f) = (
        cu(m.function("nvfp4_amax"))?,
        cu(m.function("nvfp4_quant"))?,
    );
    let stream = cu(Stream::new())?;
    let out = cu(DeviceBuffer::zeroed(e_n * stride))?;
    let gs = cu(DeviceBuffer::zeroed(e_n * 3 * 4))?;
    const CHUNK: usize = 64; // experts per upload
    let (gu_e, dn_e) = (2 * fh * 2, fh * 2); // bytes per expert
    let mut host = vec![0u8; CHUNK * gu_e];
    let io = |e: std::io::Error| e.to_string();
    for c0 in (0..e_n).step_by(CHUNK) {
        let ne = CHUNK.min(e_n - c0);
        let amax = cu(DeviceBuffer::zeroed(ne * 3 * 4))?;
        // (bf16 source, matrix stride (elements), rows, cols, code offset, scale offset, gs slot)
        gu.read_range_into((c0 * gu_e) as u64, &mut host[..ne * gu_e])
            .map_err(io)?;
        let gu_dev = cu(DeviceBuffer::from_slice(&host[..ne * gu_e]))?;
        dn.read_range_into((c0 * dn_e) as u64, &mut host[..ne * dn_e])
            .map_err(io)?;
        let dn_dev = cu(DeviceBuffer::from_slice(&host[..ne * dn_e]))?;
        cu(synchronize_context())?; // the uploads ran on the legacy stream
        let mats = [
            (gu_dev.ptr, 2 * fh, f, h, 0, fh * 3 / 2, 0usize),
            (
                gu_dev.ptr + (fh * 2) as u64,
                2 * fh,
                f,
                h,
                fh / 2,
                fh * 3 / 2 + fh / 16,
                1,
            ),
            (dn_dev.ptr, fh, h, f, fh, fh * 3 / 2 + fh / 8, 2),
        ];
        for &(src, mstride, rows, cols, co, so, slot) in &mats {
            let (ms, n) = (mstride as u64, (rows * cols) as u64);
            let am = amax.ptr + (slot * ne * 4) as u64;
            let mut a = args!(src, ms, n, am);
            cu(unsafe {
                launch(
                    amax_f,
                    LaunchDims {
                        grid: (64, ne as u32, 1),
                        block: (256, 1, 1),
                        smem: 0,
                    },
                    &stream,
                    false,
                    &mut a,
                )
            })?;
            let (ri, ci) = (rows as i32, cols as i32);
            let dst = out.ptr + (c0 * stride) as u64;
            let (ds, co, so) = (stride as u64, co as u64, so as u64);
            let g = gs.ptr + ((c0 * 3 + slot) * 4) as u64;
            let gstride = 3i32;
            let mut a = args!(src, ms, ri, ci, am, dst, ds, co, so, g, gstride);
            let blocks = (rows * cols / 16).div_ceil(256) as u32;
            cu(unsafe {
                launch(
                    quant_f,
                    LaunchDims {
                        grid: (blocks, ne as u32, 1),
                        block: (256, 1, 1),
                        smem: 0,
                    },
                    &stream,
                    false,
                    &mut a,
                )
            })?;
        }
        cu(stream.synchronize())?;
    }
    Ok((out, gs))
}

fn load_mtp(ck: &Ckpt, d: &Dims, quant: &Quant) -> Res<Mtp> {
    let (h, hcw) = (d.h, d.hcw());
    let p = "mtp.layers.0";
    // nvidia's FP4-Local ships them as NVFP4 per expert; others as fused bf16, quantized here.
    let (ex, gs) = if ck
        .t
        .contains_key(&format!("{p}.mlp.experts.0.gate_proj.weight"))
    {
        let (buf, gs) = pack_experts(ck, d, &format!("{p}.mlp.experts"))?;
        (
            cu(DeviceBuffer::from_slice(&buf))?,
            cu(DeviceBuffer::from_slice(&gs))?,
        )
    } else {
        quant_experts_nvfp4(ck, &format!("{p}.mlp.experts"), d)?
    };
    Ok(Mtp {
        norm_emb: ck.f32v("mtp.pre_fc_norm_embedding.weight", h)?,
        norm_hidden: ck.f32v("mtp.pre_fc_norm_hidden.weight", hcw)?,
        fc_emb: quant.dense(
            "mtp",
            ck.bf16_rows(&[("mtp.fc_embedding.weight", h)], h)?,
            h,
        )?,
        fc_hidden: quant.dense("mtp", ck.bf16_rows(&[("mtp.fc_hidden.weight", h)], h)?, h)?,
        layer: Layer {
            attn_gr: ck.gr(&format!("{p}.attn_hyper_connection"), d, true, quant)?,
            mixer: Mixer::Qsa(load_qsa(ck, &format!("{p}.self_attn"), d, quant)?),
            mlp_gr: ck.gr(&format!("{p}.mlp_hyper_connection"), d, true, quant)?,
            moe: load_moe(ck, &format!("{p}.mlp"), d, quant, ex, gs)?,
        },
        final_gr: ck.gr("mtp.hyper_connection_mixer", d, false, quant)?,
    })
}

/// e4m3 byte -> f32 (exact; NaN for 0x7f / 0xff).
fn e4m3_to_f32(b: u8) -> f32 {
    let (s, e, m) = (b >> 7, i32::from((b >> 3) & 15), f32::from(b & 7));
    let v = if e == 15 && m == 7.0 {
        f32::NAN
    } else if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if s != 0 { -v } else { v }
}

/// An FP8 (e4m3) matrix [rows][cols] with a per-channel ([rows] / [rows, 1]) or per-tensor ([1] / [])
/// f32 or bf16 `scale` (compressed-tensors float-quantized weights), as bf16 bytes (round to
/// nearest even).
fn fp8_to_bf16(w: &[u8], scale: &TensorInfo, rows: usize, cols: usize) -> Res<Vec<u8>> {
    let s = scale.read_as_f32().map_err(|e| e.to_string())?;
    let per_row = match s.len() {
        1 => false,
        n if n == rows => true,
        n => {
            return Err(format!(
                "{}: {n} scales for {rows} x {cols} (only per-channel or per-tensor FP8 is supported)",
                scale.name
            ));
        }
    };
    let lut: Vec<f32> = (0..=255u8).map(e4m3_to_f32).collect();
    let mut out = Vec::with_capacity(rows * cols * 2);
    for (r, row) in w.chunks_exact(cols).enumerate() {
        let sc = if per_row { s[r] } else { s[0] };
        for &b in row {
            let x = (lut[b as usize] * sc).to_bits();
            let bf = ((x + 0x7fff + ((x >> 16) & 1)) >> 16) as u16;
            out.extend_from_slice(&bf.to_le_bytes());
        }
    }
    Ok(out)
}
