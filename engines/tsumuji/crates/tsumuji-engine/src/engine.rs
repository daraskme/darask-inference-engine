//! The decode step: N tokens (1..=N_MAX) at consecutive positions, batch 1. Each N is captured
//! once into a CUDA graph and replayed, every kernel launched with PDL; positions and tokens live
//! on the device. Plain decode is N = 1; a prompt goes through in steps of N_MAX tokens (prefill
//! by decode; M4 adds a real prefill), and speculative verification (M3) uses N = drafts + 1.
//! A token's computation does not depend on N (same split-K, lanes and summation order), so a
//! step of N tokens gives the same bits as N steps of one.
//!
//! Per layer (GDN): GR down (norm fused) -> GR up + read-out -> in_proj -> conv + prep -> delta
//! rule -> gated norm -> out_proj -> write-back; then GR down -> GR up + read-out -> router ->
//! top-k -> experts gate/up -> experts down -> combine. Split-K partial sums are added by the
//! consumer; the residual writers leave per-stream sum-of-squares partials for the next GR norm.

use crate::adapter::{Adapter, LayerAdapter, Rank1Dev};
use crate::config::Dims;
use crate::weights::{Dense, Gr, Layer, Mixer, Mtp, Qsa, Res, Weights, expert_stride};
use std::collections::HashMap;
use std::ffi::c_void;
use tsumuji_cuda::{
    Device, DeviceBuffer, Function, Graph, LaunchDims, Module, Stream, args, launch,
};

fn cu<T>(r: tsumuji_cuda::Result<T>) -> Res<T> {
    r.map_err(|e| e.to_string())
}

/// Most tokens one step takes.
pub const N_MAX: usize = 8;

/// Sampling fast path (kernels/decode/sample.cu CHUNKS, TOPK_MAX).
const SAMPLE_CHUNKS: usize = 128;
const SAMPLE_TOPK: usize = 64;

/// Token capacities the kernels are compiled for (the smallest one >= N runs a step of N).
const NTS: [usize; 4] = [1, 2, 4, 8];

fn nt_index(n: usize) -> usize {
    NTS.iter().position(|&nt| nt >= n).expect("N <= N_MAX")
}

/// Mirrors `QsaDims` in kernels/decode/qsa.cu.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct QsaDims {
    aq: i32,
    akv: i32,
    ad: i32,
    rope: i32,
    ih: i32,
    id: i32,
    ir: i32,
    topb: i32,
    theta: f32,
    eps: f32,
}

/// Mirrors `Rank1` in kernels/include/rank1.cuh: one matrix's runtime adapter (a = 0: none).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Rank1 {
    a: u64,
    b: u64,
    scale: u64,
}

/// Mirrors `Stage` in kernels/decode/gemv2.cu: how a GEMV stages its input.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Stage {
    x: u64,
    ss: u64,
    w: u64,
    n: i32,
    h: i32,
    hc: i32,
    eps: f32,
}

impl Stage {
    fn plain(x: u64) -> Self {
        Self {
            x,
            ss: 0,
            w: 0,
            n: 0,
            h: 0,
            hc: 0,
            eps: 0.0,
        }
    }
}

/// GEMV kernels by (format, norm, rows per warp, prefetch, token capacity).
type Variants = Vec<((u8, bool, usize, usize, usize), Function)>;

/// Dynamic shared memory the GEMVs may use (8 tokens of a 2560-wide bf16 input: 80 KiB).
const GEMV_SMEM: u32 = 96 * 1024;

#[derive(Debug)]
struct Kernels {
    _mods: Vec<Module>,
    /// (function handle, kernel name), for per-kernel launch options.
    names: Vec<(usize, String)>,
    gemv: Variants,
    gemv_reduce: Function,
    /// gr_up_mix by weight format (bf16, MXFP8, MXINT8) and token capacity.
    gr_up_mix: [[Function; 4]; 3],
    begin_step: Function,
    embed_streams: Function,
    inject_stats: Function,
    router_topk: Function,
    /// MoE kernels by the shared expert's weight format and token capacity (N = 1, then the
    /// expert-union kernels for 2, 4, 8 tokens).
    moe_gate_up: [[Function; 4]; 3],
    moe_down: [[Function; 4]; 3],
    moe_combine: Function,
    argmax: Function,
    accept: Function,
    gdn_fused: Function,
    gdn_fused_ref: Function,
    gdn_window: Function,
    gdn_window_norm: Function,
    ple_gather: Function,
    ple_gather_nvfp4: Function,
    ple_mix: Function,
    qsa_prep_attn: Function,
    qsa_scores: Function,
    qsa_select: Function,
    qsa_attn: Function,
    mtp_fuse_norm: Function,
    mtp_fuse_out: Function,
    mtp_select: Function,
    /// The NVFP4 draft head (4 or 2 row pairs per warp) and its final reduction.
    draft_head: [Function; 2],
    /// Rescoring of its best candidates with the main lm_head, by that head's format.
    draft_pick: [Function; 3],
    sample_candidates: Function,
    sample_tokens: Function,
}

impl Kernels {
    fn load() -> Res<Self> {
        let gemv = cu(Module::load(tsumuji_kernels::DECODE_GEMV))?;
        let gemv2 = cu(Module::load(tsumuji_kernels::DECODE_GEMV2))?;
        let ops = cu(Module::load(tsumuji_kernels::DECODE_OPS))?;
        let gdn = cu(Module::load(tsumuji_kernels::DECODE_GDN))?;
        let ple = cu(Module::load(tsumuji_kernels::DECODE_PLE))?;
        let qsa = cu(Module::load(tsumuji_kernels::DECODE_QSA))?;
        let mtp = cu(Module::load(tsumuji_kernels::DECODE_MTP))?;
        let sample = cu(Module::load(tsumuji_kernels::DECODE_SAMPLE))?;
        let names = std::cell::RefCell::new(Vec::new());
        let f = |m: &Module, n: &str| -> Res<Function> {
            let func = cu(m.function(n))?;
            names.borrow_mut().push((func.raw as usize, n.to_string()));
            Ok(func)
        };
        let mut gemv_v = Vec::new();
        for (fmt, norm, prefix) in [
            (0, false, "gemv_bf16"),
            (0, true, "gemv_norm"),
            (1, false, "gemv_fp8"),
            (1, true, "gemv_fp8norm"),
            (2, false, "gemv_i8"),
            (2, true, "gemv_i8norm"),
        ] {
            for (r, p) in [(1, 4), (1, 8), (2, 8), (4, 2)] {
                for nt in NTS {
                    let func = f(&gemv2, &format!("{prefix}_r{r}_p{p}_n{nt}"))?;
                    cu(func.set_max_dynamic_smem(GEMV_SMEM))?;
                    gemv_v.push(((fmt, norm, r, p, nt), func));
                }
            }
        }
        let up = |prefix: &str| -> Res<[Function; 4]> {
            Ok([
                f(&gemv2, &format!("{prefix}_n1"))?,
                f(&gemv2, &format!("{prefix}_n2"))?,
                f(&gemv2, &format!("{prefix}_n4"))?,
                f(&gemv2, &format!("{prefix}_n8"))?,
            ])
        };
        let moe = |prefix: &str| -> Res<[Function; 4]> {
            let fs = [
                f(&gemv, prefix)?,
                f(&gemv, &format!("{prefix}_union_n2"))?,
                f(&gemv, &format!("{prefix}_union_n4"))?,
                f(&gemv, &format!("{prefix}_union_n8"))?,
            ];
            for func in &fs[1..] {
                cu(func.set_max_dynamic_smem(GEMV_SMEM))?;
            }
            Ok(fs)
        };
        Ok(Self {
            gemv: gemv_v,
            gemv_reduce: f(&gemv2, "gemv_reduce")?,
            gr_up_mix: [up("gr_up_mix")?, up("gr_up_mix_fp8")?, up("gr_up_mix_i8")?],
            begin_step: f(&ops, "begin_step")?,
            embed_streams: f(&ops, "embed_streams")?,
            inject_stats: f(&ops, "inject_stats")?,
            router_topk: f(&ops, "router_topk")?,
            moe_gate_up: [
                moe("moe_gate_up")?,
                moe("moe_gate_up_fp8")?,
                moe("moe_gate_up_i8")?,
            ],
            moe_down: [moe("moe_down")?, moe("moe_down_fp8")?, moe("moe_down_i8")?],
            moe_combine: f(&ops, "moe_combine")?,
            argmax: f(&ops, "argmax")?,
            accept: f(&ops, "accept")?,
            gdn_fused: f(&gdn, "gdn_fused")?,
            gdn_fused_ref: f(&gdn, "gdn_fused_ref")?,
            gdn_window: f(&gdn, "gdn_window")?,
            gdn_window_norm: f(&gdn, "gdn_window_norm")?,
            ple_gather: f(&ple, "ple_gather")?,
            ple_gather_nvfp4: f(&ple, "ple_gather_nvfp4")?,
            ple_mix: f(&ple, "ple_mix")?,
            qsa_prep_attn: f(&qsa, "qsa_prep_attn")?,
            qsa_scores: f(&qsa, "qsa_scores")?,
            qsa_select: f(&qsa, "qsa_select")?,
            qsa_attn: f(&qsa, "qsa_attn")?,
            mtp_fuse_norm: f(&mtp, "mtp_fuse_norm")?,
            mtp_fuse_out: f(&mtp, "mtp_fuse_out")?,
            mtp_select: f(&mtp, "mtp_select")?,
            draft_head: [f(&gemv2, "draft_head")?, f(&gemv2, "draft_head_rg2")?],
            draft_pick: [
                f(&gemv2, "draft_pick_bf16")?,
                f(&gemv2, "draft_pick_fp8")?,
                f(&gemv2, "draft_pick_i8")?,
            ],
            sample_candidates: f(&sample, "sample_candidates")?,
            sample_tokens: f(&sample, "sample_tokens")?,
            names: names.take(),
            _mods: vec![gemv, gemv2, ops, gdn, ple, qsa, mtp, sample],
        })
    }
}

/// Largest split-K factor `gemv_plan` uses.
const MAX_SLICES: usize = 16;

/// Per-sequence state and scratch, all allocated up front for N_MAX tokens per step.
/// Per-token buffers are [N][..] (split-K partials [slices][N][..]).
#[derive(Debug)]
struct State {
    /// [t1, t2, next position, parity]: committed PLE n-gram context and position (advanced by
    /// `accept`), step parity (flipped by `begin_step`).
    ctx4: DeviceBuffer,
    /// How many of the previous window's records the state kernels apply first (docs/m3.md §4.1).
    napply: DeviceBuffer,
    /// Position of the step's first token.
    pos: DeviceBuffer,
    /// Parity of the step: state buffers alternate per step (read [par], write [par ^ 1]).
    par: DeviceBuffer,
    /// Residual streams [N][HC][H].
    r: DeviceBuffer,
    /// Per-stream sum-of-squares partials of the residual [N][rows][HC].
    ss: DeviceBuffer,
    /// GR down (+ inject) split-K partials [slices][N][rank + HC].
    di_part: DeviceBuffer,
    /// GR read-out (the sub-layer input) [N][H].
    inp: DeviceBuffer,
    /// Mixer input projection partials [slices][N][rows].
    mix_part: DeviceBuffer,
    /// Mixer output projection partials [slices][N][H].
    op_part: DeviceBuffer,
    /// Generic GEMV partials (PLE, lm_head).
    part: DeviceBuffer,
    // GDN
    gdn_y: DeviceBuffer,
    gdn_s: Vec<DeviceBuffer>,
    /// Committed conv history [2 parities][3][C] per layer.
    gdn_conv: Vec<DeviceBuffer>,
    /// Window records [2 parities][rec_len] per layer (see gdn_fused).
    gdn_rec: Vec<DeviceBuffer>,
    // QSA
    q: DeviceBuffer,
    gate: DeviceBuffer,
    iq: DeviceBuffer,
    kc: Vec<DeviceBuffer>,
    vc: Vec<DeviceBuffer>,
    raw: Vec<DeviceBuffer>,
    bkeys: Vec<DeviceBuffer>,
    /// Block scores [N][sb].
    scores: DeviceBuffer,
    /// Selected tokens [N][max tokens] and their counts [N].
    toks: DeviceBuffer,
    ntok: DeviceBuffer,
    attn: DeviceBuffer,
    // MoE
    rlogits: DeviceBuffer,
    sel: DeviceBuffer,
    rw: DeviceBuffer,
    sgate: DeviceBuffer,
    eh: DeviceBuffer,
    ey: DeviceBuffer,
    // PLE
    ple_e: DeviceBuffer,
    ple_kv: DeviceBuffer,
    /// Committed conv history [2 parities][9][HC * H].
    ple_conv: DeviceBuffer,
    /// The window's conv values [2 parities][N_MAX][HC * H].
    ple_win: DeviceBuffer,
    /// [N][token, t1, t2].
    tok3: DeviceBuffer,
    // head
    logits: DeviceBuffer,
    /// The step's input tokens [N]; after a step, [0] is the next committed input (written by `accept`).
    next: DeviceBuffer,
    /// Greedy token after each of the step's tokens [N].
    out: DeviceBuffer,
    /// Token selection of the main head (`SampleParams` in kernels/decode/sample.cu).
    sampling: DeviceBuffer,
    /// Runtime adapter scale (f32; 0: adapter off), read by the kernels (rank1.cuh).
    adapter_scale: DeviceBuffer,
    /// Sampling candidates [N][SAMPLE_CHUNKS * SAMPLE_TOPK] (values, token ids; sample.cu).
    cand_v: DeviceBuffer,
    cand_i: DeviceBuffer,
    /// [committed count, committed tokens [N]] after a step (written by `accept`).
    acc_out: DeviceBuffer,
    /// The token after each window position for the MTP rows [N], and the draft steps' positions [N_MAX].
    mtp_tok: DeviceBuffer,
    chain_pos: DeviceBuffer,
    /// The MTP layer's own residual, caches and scratch (when the checkpoint has one).
    mtp: Option<MtpState>,
    /// Block scores per token (`scores` stride).
    sb: usize,
}

impl State {
    fn new(d: &Dims, ctx: usize, mtp: bool) -> Res<Self> {
        let f = |n: usize| cu(DeviceBuffer::zeroed(n * 4));
        let kv = |n: usize| cu(DeviceBuffer::zeroed(n * tsumuji_kernels::KV_BYTES));
        let hcw = d.hcw();
        let n_gdn = d.qsa.iter().filter(|q| !**q).count();
        let n_qsa = d.layers - n_gdn;
        let sb = ctx.div_ceil(d.idx_r);
        let nm = N_MAX;
        let part = [(d.vocab, d.h), (hcw + d.h, d.ple_heads * d.ple_row)]
            .iter()
            .map(|&(o, k)| gemv_plan(o, k, 0).2 * o)
            .max()
            .unwrap();
        Ok(Self {
            ctx4: f(4)?,
            napply: f(1)?,
            pos: f(1)?,
            par: f(1)?,
            r: f(nm * hcw)?,
            ss: f(nm * stat_rows(d) * d.hc)?,
            di_part: f(MAX_SLICES * nm * (d.rank + d.hc))?,
            inp: f(nm * d.h)?,
            mix_part: f(MAX_SLICES * nm * d.gdn_in_rows().max(d.qsa_rows()))?,
            op_part: f(MAX_SLICES * nm * d.h)?,
            part: f(nm * part)?,
            gdn_y: f(nm * d.gdn_v * d.gdn_d)?,
            gdn_s: (0..n_gdn)
                .map(|_| f(d.gdn_v * d.gdn_d * d.gdn_d))
                .collect::<Res<_>>()?,
            // Double-buffered by step parity (see gdn_fused).
            gdn_conv: (0..n_gdn)
                .map(|_| f(2 * 3 * d.gdn_c()))
                .collect::<Res<_>>()?,
            gdn_rec: (0..n_gdn)
                .map(|_| f(2 * gdn_rec_len(d)))
                .collect::<Res<_>>()?,
            q: f(nm * d.att_q * d.att_d)?,
            gate: f(nm * d.att_q * d.att_d)?,
            iq: f(nm * d.idx_h * d.idx_d)?,
            kc: (0..n_qsa)
                .map(|_| kv(ctx * d.att_kv * d.att_d))
                .collect::<Res<_>>()?,
            vc: (0..n_qsa)
                .map(|_| kv(ctx * d.att_kv * d.att_d))
                .collect::<Res<_>>()?,
            raw: (0..n_qsa).map(|_| kv(ctx * d.idx_d)).collect::<Res<_>>()?,
            bkeys: (0..n_qsa).map(|_| kv(sb * d.idx_d)).collect::<Res<_>>()?,
            scores: f(nm * sb)?,
            toks: f(nm * d.qsa_max_tokens())?,
            ntok: f(nm)?,
            attn: f(nm * d.att_q * d.att_d)?,
            rlogits: f(nm * (d.experts + 1))?,
            sel: f(nm * d.topk)?,
            rw: f(nm * d.topk)?,
            sgate: f(nm)?,
            eh: f(nm * (d.topk + 1) * d.ff)?,
            ey: f(nm * (d.topk + 1) * d.h)?,
            ple_e: f(nm * d.ple_heads * d.ple_row)?,
            ple_kv: f(nm * (hcw + d.h))?,
            ple_conv: f(2 * 9 * hcw)?,
            ple_win: f(2 * nm * hcw)?,
            tok3: f(nm * 3)?,
            logits: f(nm * d.vocab)?,
            // + 1: the lookahead slot of a prompt chunk (see `accept`).
            next: f(nm + 1)?,
            out: f(nm)?,
            sampling: cu(DeviceBuffer::from_slice(&Sampling::default().raw()))?,
            adapter_scale: f(1)?,
            cand_v: f(nm * SAMPLE_CHUNKS * SAMPLE_TOPK)?,
            cand_i: f(nm * SAMPLE_CHUNKS * SAMPLE_TOPK)?,
            acc_out: f(1 + nm)?,
            mtp_tok: f(nm)?,
            chain_pos: f(nm)?,
            mtp: if mtp {
                Some(MtpState::new(d, ctx)?)
            } else {
                None
            },
            sb,
        })
    }
}

/// Floats of one parity's GDN window records (gdn.cu): conv inputs [N_MAX][C], then (k, delta)
/// [N_MAX][V][2][D] and decay [N_MAX][V].
fn gdn_rec_len(d: &Dims) -> usize {
    N_MAX * d.gdn_c() + N_MAX * d.gdn_v * (2 * d.gdn_d + 1)
}

/// The MTP layer's residual, QSA caches and scratch.
#[derive(Debug)]
struct MtpState {
    /// Residual [N][HC][H] and its statistics [N][rows][HC].
    r: DeviceBuffer,
    ss: DeviceBuffer,
    /// The row the next draft step starts from [HC * H], and its statistics.
    chain_h: DeviceBuffer,
    chain_ss: DeviceBuffer,
    /// Fusion scratch: normed embeddings [N][H], normed streams [N * HC][H], their projections.
    en: DeviceBuffer,
    hn: DeviceBuffer,
    ye: DeviceBuffer,
    yh: DeviceBuffer,
    logits: DeviceBuffer,
    kc: DeviceBuffer,
    vc: DeviceBuffer,
    raw: DeviceBuffer,
    bkeys: DeviceBuffer,
}

impl MtpState {
    fn new(d: &Dims, ctx: usize) -> Res<Self> {
        let f = |n: usize| cu(DeviceBuffer::zeroed(n * 4));
        let kv = |n: usize| cu(DeviceBuffer::zeroed(n * tsumuji_kernels::KV_BYTES));
        let (nm, hcw) = (N_MAX, d.hcw());
        Ok(Self {
            r: f(nm * hcw)?,
            ss: f(nm * stat_rows(d) * d.hc)?,
            chain_h: f(hcw)?,
            chain_ss: f(stat_rows(d) * d.hc)?,
            en: f(nm * d.h)?,
            hn: f(nm * hcw)?,
            ye: f(nm * d.h)?,
            yh: f(nm * hcw)?,
            logits: f(d.vocab)?,
            kc: kv(ctx * d.att_kv * d.att_d)?,
            vc: kv(ctx * d.att_kv * d.att_d)?,
            raw: kv(ctx * d.idx_d)?,
            bkeys: kv(ctx.div_ceil(d.idx_r) * d.idx_d)?,
        })
    }
}

/// Rows of stream statistics the residual writers leave per token: one per 256 positions for the
/// element-wise writers, one per stream for the PLE mix.
fn stat_rows(d: &Dims) -> usize {
    d.h.div_ceil(256).max(d.hc)
}

/// Bytes of the state buffers that grow with the context (`State::new`, `MtpState::new`): K, V and
/// raw indexer keys per position and block keys per block of every QSA layer (and the MTP's), and
/// the block scores.
fn ctx_bytes(d: &Dims, ctx: usize, mtp: bool) -> usize {
    let layers = d.qsa.iter().filter(|q| **q).count() + usize::from(mtp);
    let sb = ctx.div_ceil(d.idx_r);
    layers * (ctx * (2 * d.att_kv * d.att_d + d.idx_d) + sb * d.idx_d) * tsumuji_kernels::KV_BYTES
        + N_MAX * sb * 4
}

/// VRAM left free when the context is sized automatically (ctx = 0): CUDA graphs, snapshots,
/// other processes' small allocations.
const CTX_RESERVE: usize = 2 << 30;

/// The longest context whose state fits in free VRAM after the weights, leaving CTX_RESERVE:
/// a probe state (1024 tokens) measures the context-independent part.
fn auto_ctx(dev: &Device, d: &Dims, mtp: bool) -> Res<usize> {
    const PROBE: usize = 1024;
    const STEP: usize = 256;
    let (free0, _) = cu(dev.mem_info())?;
    let probe = State::new(d, PROBE, mtp)?;
    cu(dev.synchronize())?;
    let (free1, _) = cu(dev.mem_info())?;
    drop(probe);
    let fixed = free0
        .saturating_sub(free1)
        .saturating_sub(ctx_bytes(d, PROBE, mtp));
    let avail = free0.saturating_sub(fixed + CTX_RESERVE);
    let per = ctx_bytes(d, 1 << 20, mtp) >> 20;
    let mut ctx = (avail / per.max(1)).min(d.max_pos) / STEP * STEP;
    while ctx > 0 && ctx_bytes(d, ctx, mtp) > avail {
        ctx -= STEP;
    }
    if ctx < STEP {
        return Err(format!(
            "not enough free VRAM for a context ({} MiB free after the weights)",
            free0 >> 20
        ));
    }
    Ok(ctx)
}

#[derive(Debug)]
pub struct Engine {
    pub dims: Dims,
    pub weights: Weights,
    pub stream: Stream,
    k: Kernels,
    s: State,
    /// Next position to write (host-side bookkeeping; the device keeps its own in `ctx4`).
    pub pos: usize,
    pub ctx: usize,
    /// Tokens in the last step.
    last_n: usize,
    /// One captured step per (tokens, verify, drafts), replayed.
    graphs: HashMap<(usize, bool, usize), Graph>,
    /// Drafts the last step left in next[1..] (the next window is 1 + this).
    drafts_ready: usize,
    /// `spec_auto`: per draft depth, decayed counts [accepted, tried]; and the cost of a step with k
    /// drafts in single-step units (measured, docs/m3.md §12; tune when kernels change).
    pub spec_acc: [[f64; 2]; N_MAX],
    /// Runtime rank-1 adapter (heretic / abliteration), see `load_adapter`.
    adapter: Option<Adapter>,
    pub spec_cost: Vec<f64>,
    /// Launch with PDL (TSJ_NO_PDL=1 turns it off).
    pub pdl: bool,
    /// Replay the captured step (TSJ_NO_GRAPH=1, profiling or capture launch directly).
    pub use_graph: bool,
    _dev: std::rc::Rc<Device>,
    /// Debugging: when set, the residual (N x HC x H floats) after each layer of every step is appended.
    pub capture: Option<Vec<Vec<f32>>>,
    /// Debugging: with `capture`, the QSA token list and block scores of every QSA layer and token.
    pub capture_qsa: Vec<(Vec<i32>, Vec<f32>)>,
    /// Per-op timing of every step when set (see `profile`).
    pub profile: Option<crate::profile::Profiler>,
    /// Timing experiments only: op groups left out of the step (the output is then wrong).
    skip: Vec<String>,
    /// Debugging: kernels launched without PDL.
    no_pdl: Vec<String>,
    /// Experiments, off by default (both measured slower, docs/m3.md §8): R = 1 GEMVs for windows
    /// of 4+ tokens (TSJ_WIDE_RP=1) and the MoE expert-union kernels (TSJ_MOE_UNION=1).
    pub wide_rp: bool,
    pub moe_union: bool,
    /// Sol round 4's column-split GDN for windows (`gdn_window`, TSJ_GDN_SPLIT=1): bit-identical,
    /// not faster on the real model (docs/m3.md §11), so off by default.
    gdn_split: bool,
    /// The original one-token-at-a-time GDN kernel (TSJ_GDN_REF=1, for A/B timing; same bits).
    gdn_ref: bool,
    /// Buffers of dropped snapshots, reused by `save_state` (cuMemAlloc is slow).
    snap_pool: SnapPool,
    /// The NVFP4 draft head with 2 row pairs per warp (default, 2 blocks per SM) or 4 (TSJ_DRAFT_RG=4).
    draft_rg2: bool,
}

const B: u32 = 256;

fn grid(n: usize, per_block: usize) -> u32 {
    n.div_ceil(per_block) as u32
}

/// (rows per warp, prefetched vectors, K slices) for a decode GEMV of shape [o][k] in weight format
/// `fmt`, from `tsumuji-bench gemv-shapes [int8]` on the real model's shapes (docs/m2.md). S must
/// not depend on the token count (the slices fix each token's summation order); `gemv_parts` may
/// change R and P, which do not.
fn gemv_plan(o: usize, k: usize, fmt: u8) -> (usize, usize, usize) {
    // The int8-only optimum of the isolated bench (S = 1 for most) was slower in the engine, where
    // the consumer adds the split-K partials for free (docs/m2.md); one plan serves both formats.
    let _ = fmt;
    if o >= 100_000 {
        (4, 2, 2) // lm_head
    } else if o >= 8192 && k >= 2048 {
        (4, 2, if o >= 16_000 { 4 } else { 2 }) // GDN in, QSA proj, PLE kv
    } else if o >= 8192 {
        (4, 2, 1) // GR up (k = rank)
    } else if k >= 8192 {
        // GR down + inject. R = 2 (was 1; R does not change results): 3.63 -> 2.95 us at N = 1,
        // 5.84 -> 5.68 us at N = 4 (gemv_win), GEMV lab on the real shape.
        (2, 8, 8)
    } else if k >= 4096 {
        (4, 2, 8) // GDN out, QSA o
    } else if o >= 1024 {
        (2, 8, 1) // shared expert
    } else {
        (1, 8, 1) // router
    }
}

/// A residual being run through layers: R [N][HC][H] and its stream statistics.
struct Seq {
    r: u64,
    ss: u64,
}

/// Debug state taken out of the engine while a step is enqueued directly.
struct Debug {
    profile: Option<crate::profile::Profiler>,
    capture: Option<Vec<Vec<f32>>>,
    capture_qsa: Vec<(Vec<i32>, Vec<f32>)>,
}

impl Engine {
    pub fn new(dev: std::rc::Rc<Device>, dims: Dims, weights: Weights, ctx: usize) -> Res<Self> {
        let k = Kernels::load()?;
        // Weight uploads and memsets ran on the legacy stream; the engine stream does not wait for it.
        cu(dev.synchronize())?;
        cu(k.qsa_attn
            .set_max_dynamic_smem(((dims.att_d + dims.qsa_max_tokens()) * 4) as u32))?;
        cu(k.qsa_prep_attn
            .set_max_dynamic_smem(((4 * dims.att_d + dims.qsa_max_tokens()) * 4) as u32))?;
        let mtp = weights.mtp.is_some();
        // ctx = 0: as many tokens as fit (docs/m3.md §13).
        let ctx = if ctx == 0 {
            auto_ctx(&dev, &dims, mtp)?
        } else {
            ctx
        };
        let s = State::new(&dims, ctx, mtp)?;
        cu(dev.synchronize())?;
        let env_off = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
        let mut e = Self {
            s,
            stream: cu(Stream::new())?,
            dims,
            weights,
            k,
            pos: 0,
            ctx,
            last_n: 0,
            graphs: HashMap::new(),
            drafts_ready: 0,
            spec_acc: [[0.0; 2]; N_MAX],
            adapter: None,
            spec_cost: vec![1.12, 1.58, 2.05, 2.33, 2.84, 3.11],
            pdl: !env_off("TSJ_NO_PDL"),
            use_graph: !env_off("TSJ_NO_GRAPH"),
            _dev: dev,
            capture: None,
            capture_qsa: Vec::new(),
            profile: None,
            skip: Vec::new(),
            no_pdl: Vec::new(),
            wide_rp: std::env::var("TSJ_WIDE_RP").is_ok_and(|v| v == "1"),
            moe_union: std::env::var("TSJ_MOE_UNION").is_ok_and(|v| v == "1"),
            gdn_split: std::env::var("TSJ_GDN_SPLIT").is_ok_and(|v| v == "1"),
            gdn_ref: std::env::var("TSJ_GDN_REF").is_ok_and(|v| v == "1"),
            snap_pool: Default::default(),
            draft_rg2: std::env::var("TSJ_DRAFT_RG").map_or(true, |v| v != "4"),
        };
        e.reset()?;
        Ok(e)
    }

    /// Leaves the op groups in `skip` out of the step, for timing experiments (`tsumuji bench`).
    /// Groups: grdown grup mixin mixmid mixout inject router experts combine lm_head ple.
    pub fn set_skip(&mut self, skip: &[&str]) {
        self.skip = skip.iter().map(|s| s.to_string()).collect();
        self.graphs.clear();
    }

    fn on(&self, group: &str) -> bool {
        !self.skip.iter().any(|s| s == group)
    }

    fn go(&self, f: Function, dims: LaunchDims, a: &mut [*mut c_void]) -> Res<()> {
        let pdl = self.pdl
            && (self.no_pdl.is_empty()
                || !self
                    .no_pdl
                    .iter()
                    .any(|n| Some(n.as_str()) == self.kernel_name(f)));
        cu(unsafe { launch(f, dims, &self.stream, pdl, a) })
    }

    fn kernel_name(&self, f: Function) -> Option<&str> {
        self.k
            .names
            .iter()
            .find(|(p, _)| *p == f.raw as usize)
            .map(|(_, n)| n.as_str())
    }

    /// Names of all loaded kernels.
    pub fn kernel_names(&self) -> Vec<String> {
        self.k.names.iter().map(|(_, n)| n.clone()).collect()
    }

    /// Debugging: launch these kernels without PDL (the graphs are re-captured).
    pub fn set_no_pdl(&mut self, names: &[String]) {
        self.no_pdl = names.to_vec();
        self.graphs.clear();
    }

    /// GEMV of a matrix [o][k] over n tokens, writing S split-K slices out[s][n][o]; returns S
    /// (the consumer adds them).
    #[allow(clippy::too_many_arguments)]
    fn gemv_parts(
        &self,
        w: &Dense,
        stage: Stage,
        norm: bool,
        out: u64,
        o: usize,
        k: usize,
        n: usize,
    ) -> Res<usize> {
        let vw = w.vw();
        debug_assert!(w.w.bytes >= o * k * 16 / vw / 8);
        let (mut r, mut p, sl) = gemv_plan(o, k, w.fmt);
        let nt = NTS[nt_index(n)];
        // Fewer live accumulators and prefetched rows for wide token windows. S and the lane -> K
        // map stay fixed, so this changes occupancy, not any token's arithmetic.
        if self.wide_rp && nt >= 4 && r > 1 {
            (r, p) = (1, 4);
        }
        let f = self
            .k
            .gemv
            .iter()
            .find(|(key, _)| *key == (w.fmt, norm, r, p, nt))
            .ok_or("no GEMV variant")?
            .1;
        let per = (k / vw).div_ceil(sl);
        let (oi, ki, ni, sc) = (o as i32, k as i32, n as i32, w.sc_ptr());
        let mut a = args!(w.w.ptr, sc, stage, out, oi, ki, ni);
        let dims = LaunchDims {
            grid: (grid(o, 8 * r), sl as u32, 1),
            block: (B, 1, 1),
            smem: (nt * per * vw * 4) as u32,
        };
        self.go(f, dims, &mut a)?;
        Ok(sl)
    }

    /// `gemv_parts` unless `group` is skipped; returns the slice count either way.
    #[allow(clippy::too_many_arguments)]
    fn parts_in(
        &self,
        group: &str,
        w: &Dense,
        stage: Stage,
        norm: bool,
        out: u64,
        o: usize,
        k: usize,
        n: usize,
    ) -> Res<usize> {
        if self.on(group) {
            self.gemv_parts(w, stage, norm, out, o, k, n)
        } else {
            Ok(gemv_plan(o, k, w.fmt).2)
        }
    }

    /// y[n][o] = W x[n] (partials reduced into y when the plan splits K).
    fn gemv(&self, w: &Dense, x: u64, y: u64, o: usize, k: usize, n: usize) -> Res<()> {
        let sl = gemv_plan(o, k, w.fmt).2;
        if sl == 1 {
            self.gemv_parts(w, Stage::plain(x), false, y, o, k, n)?;
            return Ok(());
        }
        debug_assert!(self.s.part.bytes >= sl * n * o * 4);
        self.gemv_parts(w, Stage::plain(x), false, self.s.part.ptr, o, k, n)?;
        let (oi, si) = ((n * o) as i32, sl as i32);
        let mut a = args!(self.s.part.ptr, y, oi, si);
        self.go(
            self.k.gemv_reduce,
            LaunchDims::linear(grid(n * o, B as usize).min(1024), B),
            &mut a,
        )
    }

    /// GR read of n tokens of the residual `seq`: down (+ inject) over the normed residual into
    /// `di_part`, then up + read-out into `inp`. `stats` = rows of stream statistics per token left
    /// by the last residual writer. Returns the number of down slices (the write-back reads the
    /// inject logits from them).
    fn gr_read(&self, g: &Gr, inject: bool, stats: usize, n: usize, seq: &Seq) -> Res<usize> {
        let d = &self.dims;
        let s = &self.s;
        let (h, hc, rank, eps, ni) = (d.h as i32, d.hc as i32, d.rank as i32, d.eps, n as i32);
        let rows = d.rank + if inject { d.hc } else { 0 };
        let stage = Stage {
            x: seq.r,
            ss: seq.ss,
            w: g.norm.ptr,
            n: stats as i32,
            h,
            hc,
            eps,
        };
        let sd = self.parts_in(
            "grdown",
            &g.down_inj,
            stage,
            true,
            s.di_part.ptr,
            rows,
            d.hcw(),
            n,
        )?;
        let (sdi, ld, p) = (sd as i32, rows as i32, stats as i32);
        let up_sc = g.up.sc_ptr();
        let mut a = args!(
            g.up.w.ptr,
            up_sc,
            s.di_part.ptr,
            sdi,
            ld,
            seq.r,
            seq.ss,
            p,
            g.norm.ptr,
            s.inp.ptr,
            h,
            hc,
            rank,
            eps,
            ni
        );
        let ti = nt_index(n);
        let dims = LaunchDims {
            grid: (grid(d.h, 8), 1, 1),
            block: (B, 1, 1),
            smem: (NTS[ti] * d.rank * 4) as u32,
        };
        if self.on("grup") {
            self.go(self.k.gr_up_mix[usize::from(g.up.fmt)][ti], dims, &mut a)?;
        }
        Ok(sd)
    }

    /// Runs one token through the model; the logits and the greedy next token stay on the
    /// device. `Some(token)` feeds a given token (a prompt); `None` feeds the previous step's
    /// greedy token without a host round trip.
    pub fn step(&mut self, token: Option<u32>) -> Res<()> {
        match token {
            Some(t) => self.step_tokens(&[t]),
            None => self.run(1, false, 0),
        }
    }

    /// Runs the given tokens (1..=N_MAX) through the model in one step, at the next positions;
    /// all of them are committed.
    pub fn step_tokens(&mut self, tokens: &[u32]) -> Res<()> {
        self.put_window(tokens, None)?;
        self.run(tokens.len(), false, 0)
    }

    /// Runs a prompt through the model in steps of up to N_MAX tokens. Each step also tells the
    /// MTP layer the token after its last one (the next step's first), so its cache is exact.
    pub fn prefill(&mut self, tokens: &[u32]) -> Res<()> {
        for (c, chunk) in tokens.chunks(N_MAX).enumerate() {
            self.put_window(chunk, tokens.get((c + 1) * N_MAX).copied())?;
            self.run(chunk.len(), false, 0)?;
        }
        Ok(())
    }

    /// Speculative verification of a given window: [the next token, drafts...] (1..=N_MAX tokens).
    /// Draft j is accepted while all drafts before it were and it equals the greedy token after
    /// the token before it; the window's token 0 and the accepted drafts are committed. Returns the
    /// number committed (1 + accepted drafts); `next_token` is then the greedy token after the
    /// last committed one, and `logits_all` holds the window's logits (valid for the committed
    /// positions).
    pub fn verify(&mut self, window: &[u32]) -> Res<usize> {
        self.put_window(window, None)?;
        Ok(self.verify_on_device(window.len(), 0)?.len())
    }

    /// `spec_step` with the draft count chosen from the recent acceptance. Per draft depth j,
    /// a_j = P(draft j accepted | drafts 1..j-1 were), from decayed counts and shrunk toward
    /// a_{j-1} (0.75 for j = 1), so depths not tried lately inherit the one before. k drafts commit
    /// 1 + a_1 + a_1 a_2 + ... + a_1..a_k tokens on average for a step costing `spec_cost[k]`
    /// (window of k + 1 plus k MTP drafts, in single-step units); the best ratio wins.
    /// (One acceptance rate for all depths overrated deep drafts: docs/m3.md §12.)
    pub fn spec_auto(&mut self) -> Res<Vec<u32>> {
        let (mut prev, mut reach, mut gain) = (0.75, 1.0, vec![1.0]);
        for st in self
            .spec_acc
            .iter()
            .take(self.spec_cost.len().saturating_sub(1))
        {
            let a = (st[0] + 2.0 * prev) / (st[1] + 2.0);
            reach *= a;
            gain.push(gain[gain.len() - 1] + reach);
            prev = a;
        }
        let k = (0..self.spec_cost.len())
            .max_by(|&x, &y| {
                (gain[x] / self.spec_cost[x]).total_cmp(&(gain[y] / self.spec_cost[y]))
            })
            .unwrap_or(0);
        let proposed = self.drafts_ready;
        let out = self.spec_step(k)?;
        let accepted = out.len() - 1;
        for st in &mut self.spec_acc {
            st[0] *= 0.97;
            st[1] *= 0.97;
        }
        for (j, st) in self
            .spec_acc
            .iter_mut()
            .enumerate()
            .take(proposed.min(accepted + 1))
        {
            st[1] += 1.0;
            st[0] += f64::from(u8::from(j < accepted));
        }
        Ok(out)
    }

    /// One step of speculative decoding with the MTP layer: verifies the window already on the
    /// device (the next token and the drafts the previous step made), commits its accepted
    /// prefix, and has the MTP draft `drafts` tokens (0..N_MAX) for the next window. Returns the
    /// committed tokens; `next_token` is the one after them.
    pub fn spec_step(&mut self, drafts: usize) -> Res<Vec<u32>> {
        if drafts > 0 && self.weights.mtp.is_none() {
            return Err("this checkpoint has no MTP layer".into());
        }
        if drafts >= N_MAX {
            return Err(format!("at most {} drafts", N_MAX - 1));
        }
        self.verify_on_device(1 + self.drafts_ready, drafts)
    }

    /// Verifies the n-token window in `next`, then drafts; returns the committed tokens.
    fn verify_on_device(&mut self, n: usize, drafts: usize) -> Res<Vec<u32>> {
        if self.pos + n + drafts > self.ctx {
            return Err(format!("context full ({} tokens)", self.ctx));
        }
        self.run(n, true, drafts)?;
        cu(self.stream.synchronize())?;
        let acc = cu(self.s.acc_out.to_vec::<i32>(1 + n))?;
        let committed = acc[0] as usize;
        self.pos = self.pos - n + committed;
        self.drafts_ready = drafts;
        Ok(acc[1..=committed].iter().map(|&t| t as u32).collect())
    }

    /// Uploads a window (and the prompt lookahead for the MTP, or none).
    fn put_window(&mut self, tokens: &[u32], look: Option<u32>) -> Res<()> {
        if tokens.is_empty() || tokens.len() > N_MAX {
            return Err(format!(
                "a step takes 1..={N_MAX} tokens, not {}",
                tokens.len()
            ));
        }
        if let Some(&t) = tokens
            .iter()
            .chain(&look)
            .find(|&&t| t as usize >= self.dims.vocab)
        {
            return Err(format!("token {t} out of vocabulary"));
        }
        let mut ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        ids.push(look.map_or(-1, |t| t as i32));
        self.drafts_ready = 0;
        cu(self.s.next.upload_on(&ids, &self.stream))
    }

    /// One step of the n tokens in `next` (`verify`: commit by the accept rule, else all), then
    /// `drafts` MTP drafts.
    fn run(&mut self, n: usize, verify: bool, drafts: usize) -> Res<()> {
        if self.pos + n > self.ctx {
            return Err(format!("context full ({} tokens)", self.ctx));
        }
        if !verify {
            self.drafts_ready = 0;
        }
        let direct = !self.use_graph || self.profile.is_some() || self.capture.is_some();
        if direct {
            let mut dbg = Debug {
                profile: self.profile.take(),
                capture: self.capture.take(),
                capture_qsa: std::mem::take(&mut self.capture_qsa),
            };
            let r = self.enqueue(n, verify, drafts, &mut dbg);
            (self.profile, self.capture, self.capture_qsa) =
                (dbg.profile, dbg.capture, dbg.capture_qsa);
            r?;
        } else {
            let key = (n, verify, drafts);
            if !self.graphs.contains_key(&key) {
                let mut none = Debug {
                    profile: None,
                    capture: None,
                    capture_qsa: Vec::new(),
                };
                let g = self
                    .stream
                    .capture(|_| self.enqueue(n, verify, drafts, &mut none))?;
                self.graphs.insert(key, g);
            }
            cu(self.graphs[&key].launch(&self.stream))?;
        }
        self.pos += n;
        self.last_n = n;
        Ok(())
    }

    /// Enqueues one step: n tokens through the model, the commit, then the MTP's cache rows and
    /// `drafts` drafts (captured once per (n, verify, drafts) into a graph).
    fn enqueue(&self, n: usize, verify: bool, drafts: usize, dbg: &mut Debug) -> Res<()> {
        let d = &self.dims;
        let s = &self.s;
        let w = &self.weights;
        let (h, hc, eps, ni) = (d.h as i32, d.hc as i32, d.eps, n as i32);
        let elem_rows = d.h.div_ceil(B as usize);
        let elementwise = LaunchDims {
            grid: (elem_rows as u32, n as u32, 1),
            block: (B, 1, 1),
            smem: 0,
        };
        let main = Seq {
            r: s.r.ptr,
            ss: s.ss.ptr,
        };
        macro_rules! mark {
            ($label:expr) => {
                if let Some(pr) = dbg.profile.as_mut() {
                    cu(pr.mark(&self.stream, $label))?;
                }
            };
        }
        mark!("begin");

        let eos = d.eos as i32;
        let mut a = args!(
            s.next.ptr, s.ctx4.ptr, s.tok3.ptr, s.pos.ptr, s.par.ptr, eos, ni
        );
        self.go(self.k.begin_step, LaunchDims::linear(1, 32), &mut a)?;
        let mut a = args!(w.embed.ptr, s.tok3.ptr, s.r.ptr, s.ss.ptr, h, hc);
        self.go(self.k.embed_streams, elementwise, &mut a)?;
        let mut stats = elem_rows;
        mark!("embed");

        let (mut gi, mut qi) = (0usize, 0usize);
        for (l, layer) in w.layers.iter().enumerate() {
            if d.ple_layer == Some(l) && self.on("ple") {
                let ple = w.ple.as_ref().unwrap();
                let row = d.ple_row as i32;
                let hash = ple.hash;
                if ple.nvfp4 {
                    let rb = ple.row_bytes as i32;
                    let mut a = args!(
                        ple.table.dev,
                        hash,
                        s.tok3.ptr,
                        ple.scale.ptr,
                        s.ple_e.ptr,
                        row,
                        rb
                    );
                    self.go(
                        self.k.ple_gather_nvfp4,
                        LaunchDims::linear(n as u32, 512),
                        &mut a,
                    )?;
                } else {
                    let mut a = args!(
                        ple.table.dev,
                        hash,
                        s.tok3.ptr,
                        ple.scale.ptr,
                        s.ple_e.ptr,
                        row
                    );
                    self.go(self.k.ple_gather, LaunchDims::linear(n as u32, 512), &mut a)?;
                }
                self.gemv(
                    &ple.kv_proj,
                    s.ple_e.ptr,
                    s.ple_kv.ptr,
                    d.hcw() + d.h,
                    d.ple_heads * d.ple_row,
                    n,
                )?;
                let mut a = args!(
                    s.r.ptr,
                    s.ple_kv.ptr,
                    ple.norm_key.ptr,
                    ple.norm_query.ptr,
                    ple.norm_conv.ptr,
                    ple.conv.ptr,
                    s.ple_conv.ptr,
                    s.ple_win.ptr,
                    s.par.ptr,
                    s.napply.ptr,
                    s.ss.ptr,
                    h,
                    hc,
                    eps,
                    ni
                );
                self.go(self.k.ple_mix, LaunchDims::linear(d.hc as u32, 512), &mut a)?;
                stats = d.hc;
                mark!("ple");
            }

            let sd = self.gr_read(&layer.attn_gr, true, stats, n, &main)?;
            mark!("gr");
            let so = match &layer.mixer {
                Mixer::Gdn(g) => {
                    let (qk, v, dd) = (d.gdn_qk as i32, d.gdn_v as i32, d.gdn_d as i32);
                    let rows = d.gdn_in_rows();
                    let sm = self.parts_in(
                        "mixin",
                        &g.in_proj,
                        Stage::plain(s.inp.ptr),
                        false,
                        s.mix_part.ptr,
                        rows,
                        d.h,
                        n,
                    )? as i32;
                    mark!("gdn in");
                    let ld = rows as i32;
                    let (st, cs, rec) = (&s.gdn_s[gi], &s.gdn_conv[gi], &s.gdn_rec[gi]);
                    let mut a = args!(
                        s.mix_part.ptr,
                        sm,
                        ld,
                        cs.ptr,
                        s.par.ptr,
                        s.napply.ptr,
                        rec.ptr,
                        g.conv.ptr,
                        g.a_log.ptr,
                        g.dt_bias.ptr,
                        st.ptr,
                        g.norm.ptr,
                        s.gdn_y.ptr,
                        qk,
                        v,
                        dd,
                        eps,
                        ni
                    );
                    let smem = (((4 * n + 1) * d.gdn_d
                        + 2 * (d.gdn_d / 32) * d.gdn_d
                        + 2 * N_MAX * d.gdn_d)
                        * 4) as u32;
                    let dims = LaunchDims {
                        grid: (d.gdn_v as u32, 1, 1),
                        block: (4 * d.gdn_d as u32, 1, 1),
                        smem,
                    };
                    if self.on("mixmid") && self.on("gdnmid") {
                        if self.gdn_split && n > 1 && d.gdn_d <= 128 && d.gdn_d.is_multiple_of(32) {
                            let smem = ((2 * n * d.gdn_d + n * 32 + 32 + 2 * d.gdn_d) * 4) as u32;
                            let dims = LaunchDims {
                                grid: (d.gdn_v as u32, (d.gdn_d / 32) as u32, 1),
                                block: (d.gdn_d as u32, 1, 1),
                                smem,
                            };
                            self.go(self.k.gdn_window, dims, &mut a)?;
                            let mut a = args!(
                                s.gdn_y.ptr,
                                s.mix_part.ptr,
                                sm,
                                ld,
                                g.norm.ptr,
                                qk,
                                v,
                                dd,
                                eps,
                                ni
                            );
                            self.go(
                                self.k.gdn_window_norm,
                                LaunchDims {
                                    grid: (d.gdn_v as u32, n as u32, 1),
                                    block: (d.gdn_d as u32, 1, 1),
                                    smem: 0,
                                },
                                &mut a,
                            )?;
                        } else {
                            let f = if self.gdn_ref {
                                self.k.gdn_fused_ref
                            } else {
                                self.k.gdn_fused
                            };
                            let dims = if self.gdn_ref {
                                LaunchDims {
                                    smem: (((4 * n + 1) * d.gdn_d + d.gdn_d / 32 * d.gdn_d) * 4)
                                        as u32,
                                    ..dims
                                }
                            } else {
                                dims
                            };
                            self.go(f, dims, &mut a)?;
                        }
                    }
                    mark!("gdn mid");
                    let so = self.parts_in(
                        "mixout",
                        &g.out_proj,
                        Stage::plain(s.gdn_y.ptr),
                        false,
                        s.op_part.ptr,
                        d.h,
                        d.gdn_v * d.gdn_d,
                        n,
                    )?;
                    gi += 1;
                    mark!("gdn");
                    so
                }
                Mixer::Qsa(q) => {
                    let caches = [&s.kc[qi], &s.vc[qi], &s.raw[qi], &s.bkeys[qi]];
                    let so = self.qsa_mixer(q, n, s.pos.ptr, caches, Some(dbg))?;
                    qi += 1;
                    mark!("qsa");
                    so
                }
            };
            let mix = if d.qsa[l] {
                (s.attn.ptr, d.att_q * d.att_d)
            } else {
                (s.gdn_y.ptr, d.gdn_v * d.gdn_d)
            };
            self.mlp_half(
                layer,
                sd,
                so,
                n,
                &main,
                mix,
                self.adapter.as_ref().map(|a| &a.layers[l]),
            )?;
            stats = elem_rows;
            mark!("moe");
            if let Some(cap) = dbg.capture.as_mut() {
                cap.push(cu(self
                    .stream
                    .synchronize()
                    .and_then(|()| s.r.to_vec::<f32>(n * d.hcw())))?);
            }
        }
        debug_assert_eq!(expert_stride(d), d.ff * d.h * 27 / 16);

        // Head: final mixer (no inject), lm_head, the next token (greedy or sampled).
        self.gr_read(&w.final_gr, false, stats, n, &main)?;
        mark!("gr");
        if self.on("lm_head") {
            self.gemv(&w.lm_head, s.inp.ptr, s.logits.ptr, d.vocab, d.h, n)?;
            let v = d.vocab as i32;
            let mut a = args!(s.logits.ptr, v, s.sampling.ptr, s.cand_v.ptr, s.cand_i.ptr);
            let dims = LaunchDims {
                grid: (SAMPLE_CHUNKS as u32, n as u32, 1),
                block: (B, 1, 1),
                smem: 0,
            };
            self.go(self.k.sample_candidates, dims, &mut a)?;
            let mut a = args!(
                s.logits.ptr,
                v,
                s.sampling.ptr,
                s.pos.ptr,
                s.cand_v.ptr,
                s.cand_i.ptr,
                s.out.ptr
            );
            self.go(
                self.k.sample_tokens,
                LaunchDims::linear(n as u32, 1024),
                &mut a,
            )?;
        }
        // Commit: all n tokens, or the verified prefix of the window.
        let vf = i32::from(verify);
        let mut a = args!(
            s.next.ptr,
            s.out.ptr,
            s.ctx4.ptr,
            s.napply.ptr,
            s.mtp_tok.ptr,
            s.chain_pos.ptr,
            s.acc_out.ptr,
            eos,
            ni,
            vf
        );
        self.go(self.k.accept, LaunchDims::linear(1, 32), &mut a)?;
        mark!("lm_head");

        // MTP: fill its cache for this window's positions (row t pairs h_{p+t} with the token
        // after it), then draft from the last committed row into next[1..=drafts].
        if let (Some(m), Some(ms)) = (&w.mtp, &s.mtp)
            && self.on("mtp")
        {
            self.mtp_layer(m, ms, n, s.r.ptr, s.mtp_tok.ptr, s.pos.ptr)?;
            mark!("mtp cache");
            if drafts > 0 {
                self.mtp_head(m, ms, Some(s.napply.ptr), s.next.ptr + 4)?;
                mark!("mtp head");
                for i in 1..drafts {
                    self.mtp_layer(
                        m,
                        ms,
                        1,
                        ms.chain_h.ptr,
                        s.next.ptr + 4 * i as u64,
                        s.chain_pos.ptr + 4 * (i - 1) as u64,
                    )?;
                    mark!("mtp draft layer");
                    self.mtp_head(m, ms, None, s.next.ptr + 4 * (i + 1) as u64)?;
                    mark!("mtp head");
                }
            }
        }
        if let Some(pr) = dbg.profile.as_mut() {
            cu(pr.end_step())?;
        }
        Ok(())
    }

    /// The QSA mixer of n tokens at positions *pos.. with the given caches [K, V, raw keys, block
    /// keys]; returns the output projection's slice count. `dbg` (main layers) records the
    /// selections when capturing.
    fn qsa_mixer(
        &self,
        q: &Qsa,
        n: usize,
        pos: u64,
        caches: [&DeviceBuffer; 4],
        dbg: Option<&mut Debug>,
    ) -> Res<usize> {
        let d = &self.dims;
        let s = &self.s;
        let ni = n as i32;
        let rows = d.qsa_rows();
        let sm = self.parts_in(
            "mixin",
            &q.proj,
            Stage::plain(s.inp.ptr),
            false,
            s.mix_part.ptr,
            rows,
            d.h,
            n,
        )? as i32;
        let ld = rows as i32;
        let qd = QsaDims {
            aq: d.att_q as i32,
            akv: d.att_kv as i32,
            ad: d.att_d as i32,
            rope: d.rope as i32,
            ih: d.idx_h as i32,
            id: d.idx_d as i32,
            ir: d.idx_r as i32,
            topb: d.idx_topb as i32,
            theta: d.theta,
            eps: d.eps,
        };
        let [kc, vc, raw, bk] = caches;
        let mut a = args!(
            s.mix_part.ptr,
            sm,
            ld,
            qd,
            pos,
            q.q_norm.ptr,
            q.k_norm.ptr,
            q.iq_norm.ptr,
            q.ik_norm.ptr,
            s.q.ptr,
            s.gate.ptr,
            kc.ptr,
            vc.ptr,
            s.iq.ptr,
            raw.ptr,
            bk.ptr,
            s.attn.ptr,
            ni
        );
        // Query heads (with N = 1, attending directly while every token is visible; with N > 1,
        // qsa_attn attends after this grid), indexer, raw key.
        let blocks = (d.att_q + 2) as u32;
        let smem = ((4 * d.att_d + d.qsa_max_tokens()) * 4) as u32;
        let dims = LaunchDims {
            grid: (blocks, n as u32, 1),
            block: (d.att_d as u32, 1, 1),
            smem,
        };
        let mid = self.on("mixmid") && self.on("qsamid");
        if mid {
            self.go(self.k.qsa_prep_attn, dims, &mut a)?;
        }
        let (ir, topb, ih, id) = (
            d.idx_r as i32,
            d.idx_topb as i32,
            d.idx_h as i32,
            d.idx_d as i32,
        );
        let (sb, mt) = (s.sb as i32, d.qsa_max_tokens() as i32);
        let mut a = args!(s.iq.ptr, bk.ptr, s.scores.ptr, pos, ir, topb, ih, id, sb);
        if mid {
            self.go(
                self.k.qsa_scores,
                LaunchDims {
                    grid: (376, n as u32, 1),
                    block: (B, 1, 1),
                    smem: 0,
                },
                &mut a,
            )?;
        }
        let mut a = args!(s.scores.ptr, pos, topb, ir, s.toks.ptr, s.ntok.ptr, sb, mt);
        if mid {
            self.go(
                self.k.qsa_select,
                LaunchDims::linear(n as u32, 1024),
                &mut a,
            )?;
        }
        if let Some(dbg) = dbg
            && dbg.capture.is_some()
        {
            cu(self.stream.synchronize())?;
            let counts = cu(s.ntok.to_vec::<i32>(n))?;
            let all_toks = cu(s.toks.to_vec::<i32>(n * d.qsa_max_tokens()))?;
            let all_sc = cu(s.scores.to_vec::<f32>(n * s.sb))?;
            for (t, &c) in counts.iter().enumerate() {
                let toks = all_toks[t * d.qsa_max_tokens()..][..c as usize].to_vec();
                let nb = (self.pos + t + 1) / d.idx_r;
                let sc = if nb > d.idx_topb {
                    all_sc[t * s.sb..][..nb].to_vec()
                } else {
                    Vec::new()
                };
                dbg.capture_qsa.push((toks, sc));
            }
        }
        let (aq, akv, ad) = (d.att_q as i32, d.att_kv as i32, d.att_d as i32);
        let mut a = args!(
            s.q.ptr, s.gate.ptr, kc.ptr, vc.ptr, s.toks.ptr, s.ntok.ptr, s.attn.ptr, aq, akv, ad,
            pos, ir, topb, mt
        );
        let dims = LaunchDims {
            grid: (d.att_q as u32, n as u32, 1),
            block: (d.att_d as u32, 1, 1),
            smem: ((d.att_d + d.qsa_max_tokens()) * 4) as u32,
        };
        if mid {
            self.go(self.k.qsa_attn, dims, &mut a)?;
        }
        self.parts_in(
            "mixout",
            &q.o_proj,
            Stage::plain(s.attn.ptr),
            false,
            s.op_part.ptr,
            d.h,
            d.att_q * d.att_d,
            n,
        )
    }

    /// The rest of a layer after its mixer: the mixer's GR write-back (sd GR down slices, so
    /// output projection slices; `mix` = the output projection's input [N][K] for its adapter),
    /// then GR read, router, experts and their write-back. `ad`: this layer's runtime adapters.
    #[allow(clippy::too_many_arguments)]
    fn mlp_half(
        &self,
        layer: &Layer,
        sd: usize,
        so: usize,
        n: usize,
        seq: &Seq,
        mix: (u64, usize),
        ad: Option<&LayerAdapter>,
    ) -> Res<()> {
        let d = &self.dims;
        let s = &self.s;
        let (h, hc, rank) = (d.h as i32, d.hc as i32, d.rank as i32);
        let elem_rows = d.h.div_ceil(B as usize);
        let elementwise = LaunchDims {
            grid: (elem_rows as u32, n as u32, 1),
            block: (B, 1, 1),
            smem: 0,
        };
        let ld_d = rank + hc;
        let (sdi, soi) = (sd as i32, so as i32);
        let r1 = |m: Option<&Rank1Dev>| Rank1 {
            a: m.map_or(0, |r| r.a.ptr),
            b: m.map_or(0, |r| r.b.ptr),
            scale: s.adapter_scale.ptr,
        };
        let (ad_attn, ad_sh, ad_ex) = (
            r1(ad.and_then(|a| a.attn.as_ref())),
            r1(ad.and_then(|a| a.shared.as_ref())),
            r1(ad.and_then(|a| a.experts.as_ref())),
        );
        let (mx, mk) = (mix.0, mix.1 as i32);
        let mut a = args!(
            seq.r,
            s.di_part.ptr,
            sdi,
            ld_d,
            rank,
            s.op_part.ptr,
            soi,
            seq.ss,
            h,
            hc,
            mx,
            mk,
            ad_attn
        );
        if self.on("inject") {
            self.go(self.k.inject_stats, elementwise, &mut a)?;
        }

        let sd = self.gr_read(&layer.mlp_gr, true, elem_rows, n, seq)? as i32;
        let m = &layer.moe;
        let router = self.on("router");
        if router {
            self.gemv(&m.router, s.inp.ptr, s.rlogits.ptr, d.experts + 1, d.h, n)?;
        }
        let (e, topk, ff) = (d.experts as i32, d.topk as i32, d.ff as i32);
        let mut a = args!(s.rlogits.ptr, e, topk, s.sel.ptr, s.rw.ptr, s.sgate.ptr);
        if router {
            self.go(self.k.router_topk, LaunchDims::linear(n as u32, 32), &mut a)?;
        }
        // The topk routed experts and the shared expert (slot topk) in one launch per projection:
        // 16 lanes per gate/up row (2 rows per warp), 4 lanes per down row (8 rows per warp).
        let (gu_sc, dn_sc) = (m.shared_gu.sc_ptr(), m.shared_down.sc_ptr());
        let fi = usize::from(m.shared_gu.fmt);
        // Per-token kernels (grid z = token), or the expert-union kernels for windows (TSJ_MOE_UNION=1).
        let ti = if self.moe_union { nt_index(n) } else { 0 };
        let nt = NTS[ti];
        let (k_gu, k_dn) = (self.k.moe_gate_up[fi][ti], self.k.moe_down[fi][ti]);
        let mut a = args!(
            m.experts.ptr,
            m.gs.ptr,
            s.sel.ptr,
            m.shared_gu.w.ptr,
            gu_sc,
            s.inp.ptr,
            s.eh.ptr,
            h,
            ff,
            topk
        );
        let slots = d.topk as u32 + u32::from(self.on("shared"));
        let dims = LaunchDims {
            grid: (grid(d.ff, 16), slots, n as u32),
            block: (B, 1, 1),
            smem: (nt * d.h / 32 * 33 * 4) as u32,
        };
        let experts = self.on("experts");
        if experts {
            self.go(k_gu, dims, &mut a)?;
        }
        let dims = LaunchDims {
            grid: (grid(d.h, 64), slots, n as u32),
            block: (B, 1, 1),
            smem: (nt * d.ff / 32 * 33 * 4) as u32,
        };
        if experts && ti == 0 {
            let mut a = args!(
                m.experts.ptr,
                m.gs.ptr,
                s.sel.ptr,
                m.shared_down.w.ptr,
                dn_sc,
                s.eh.ptr,
                s.ey.ptr,
                h,
                ff,
                topk,
                ad_ex,
                ad_sh
            );
            self.go(k_dn, dims, &mut a)?;
        } else if experts {
            if ad.is_some() {
                return Err(
                    "the expert-union kernels (TSJ_MOE_UNION=1) do not support adapters".into(),
                );
            }
            let mut a = args!(
                m.experts.ptr,
                m.gs.ptr,
                s.sel.ptr,
                m.shared_down.w.ptr,
                dn_sc,
                s.eh.ptr,
                s.ey.ptr,
                h,
                ff,
                topk
            );
            self.go(k_dn, dims, &mut a)?;
        }
        let mut a = args!(
            seq.r,
            s.di_part.ptr,
            sd,
            ld_d,
            rank,
            s.ey.ptr,
            s.rw.ptr,
            s.sgate.ptr,
            seq.ss,
            h,
            hc,
            topk
        );
        if self.on("combine") {
            self.go(self.k.moe_combine, elementwise, &mut a)?;
        }
        Ok(())
    }

    /// One MTP pass over n rows at positions *pos..: row t fuses hidden row t (HC * H floats apart)
    /// with the embedding of tok[t], then the MTP's QSA + MoE layer; the result is in `ms.r`.
    fn mtp_layer(
        &self,
        m: &Mtp,
        ms: &MtpState,
        n: usize,
        hidden: u64,
        tok: u64,
        pos: u64,
    ) -> Res<()> {
        let d = &self.dims;
        let (h, hc, eps, ld) = (d.h as i32, d.hc as i32, d.eps, d.hcw() as i32);
        let mut a = args!(
            self.weights.embed.ptr,
            tok,
            hidden,
            ld,
            m.norm_emb.ptr,
            m.norm_hidden.ptr,
            ms.en.ptr,
            ms.hn.ptr,
            h,
            hc,
            eps
        );
        self.go(
            self.k.mtp_fuse_norm,
            LaunchDims::linear(n as u32, 512),
            &mut a,
        )?;
        self.gemv(&m.fc_emb, ms.en.ptr, ms.ye.ptr, d.h, d.h, n)?;
        // fc_hidden is shared by the streams: N * HC rows, at most N_MAX per GEMV.
        for c in (0..n * d.hc).step_by(N_MAX) {
            let rows = N_MAX.min(n * d.hc - c);
            let off = (c * d.h * 4) as u64;
            self.gemv(
                &m.fc_hidden,
                ms.hn.ptr + off,
                ms.yh.ptr + off,
                d.h,
                d.h,
                rows,
            )?;
        }
        let elem_rows = d.h.div_ceil(B as usize);
        let mut a = args!(ms.ye.ptr, ms.yh.ptr, ms.r.ptr, ms.ss.ptr, h, hc);
        self.go(
            self.k.mtp_fuse_out,
            LaunchDims {
                grid: (elem_rows as u32, n as u32, 1),
                block: (B, 1, 1),
                smem: 0,
            },
            &mut a,
        )?;
        let seq = Seq {
            r: ms.r.ptr,
            ss: ms.ss.ptr,
        };
        let layer = &m.layer;
        let sd = self.gr_read(&layer.attn_gr, true, elem_rows, n, &seq)?;
        let Mixer::Qsa(q) = &layer.mixer else {
            return Err("MTP layer is not QSA".into());
        };
        let so = self.qsa_mixer(q, n, pos, [&ms.kc, &ms.vc, &ms.raw, &ms.bkeys], None)?;
        self.mlp_half(
            layer,
            sd,
            so,
            n,
            &seq,
            (self.s.attn.ptr, d.att_q * d.att_d),
            None,
        )
    }

    /// MTP head on one row of the last MTP pass (row *napply - 1 when given, else 0): keeps that
    /// row as the hidden input of the next draft step and writes the greedy draft to `out`.
    fn mtp_head(&self, m: &Mtp, ms: &MtpState, napply: Option<u64>, out: u64) -> Res<()> {
        let d = &self.dims;
        let s = &self.s;
        let (wi, rows, hc) = (d.hcw() as i32, stat_rows(d) as i32, d.hc as i32);
        let sel = napply.unwrap_or(0);
        let mut a = args!(
            ms.r.ptr,
            ms.ss.ptr,
            sel,
            wi,
            rows,
            hc,
            ms.chain_h.ptr,
            ms.chain_ss.ptr
        );
        self.go(
            self.k.mtp_select,
            LaunchDims::linear(grid(d.hcw(), B as usize), B),
            &mut a,
        )?;
        let seq = Seq {
            r: ms.chain_h.ptr,
            ss: ms.chain_ss.ptr,
        };
        self.gr_read(&m.final_gr, false, d.h.div_ceil(B as usize), 1, &seq)?;
        let v = d.vocab as i32;
        if let Some(dh) = &self.weights.draft_head {
            // The NVFP4 copy of lm_head with the argmax folded in (per-block bests in ms.logits),
            // then its best candidates rescored with the main lm_head (docs/m3.md §12).
            let ri = usize::from(self.draft_rg2);
            let rows_per_block = 16 * if self.draft_rg2 { 2 } else { 4 };
            let blocks = d.vocab.div_ceil(rows_per_block);
            let (so, hi, nb) = (dh.so as u64, d.h as i32, blocks as i32);
            let mut a = args!(dh.w.ptr, so, dh.gs.ptr, s.inp.ptr, ms.logits.ptr, v, hi);
            let smem = (d.h / 32 * 33 * 4) as u32;
            self.go(
                self.k.draft_head[ri],
                LaunchDims {
                    grid: (blocks as u32, 1, 1),
                    block: (B, 1, 1),
                    smem,
                },
                &mut a,
            )?;
            let lm = &self.weights.lm_head;
            let lsc = lm.sc_ptr();
            let mut a = args!(ms.logits.ptr, nb, lm.w.ptr, lsc, s.inp.ptr, hi, out);
            return self.go(
                self.k.draft_pick[usize::from(lm.fmt)],
                LaunchDims::linear(1, 1024),
                &mut a,
            );
        }
        self.gemv(
            &self.weights.lm_head,
            s.inp.ptr,
            ms.logits.ptr,
            d.vocab,
            d.h,
            1,
        )?;
        let mut a = args!(ms.logits.ptr, v, out);
        self.go(self.k.argmax, LaunchDims::linear(1, 1024), &mut a)
    }

    /// Starts a new sequence: position 0, zero recurrent and conv states, nothing pending, fresh
    /// PLE context. KV and indexer caches need no clearing; positions are rewritten before they
    /// are read.
    pub fn reset(&mut self) -> Res<()> {
        let s = &self.s;
        for b in s
            .gdn_s
            .iter()
            .chain(&s.gdn_conv)
            .chain([&s.ple_conv, &s.napply])
        {
            cu(b.zero_on(&self.stream))?;
        }
        let eos = self.dims.eos as i32;
        cu(s.ctx4.upload_on(&[eos, eos, 0, 0], &self.stream))?;
        cu(self.stream.synchronize())?;
        self.pos = 0;
        self.last_n = 0;
        self.drafts_ready = 0;
        self.spec_acc = [[0.0; 2]; N_MAX];
        Ok(())
    }

    /// Everything a later step reads besides the KV and indexer caches.
    fn seq_state(&self) -> Vec<&DeviceBuffer> {
        let s = &self.s;
        s.gdn_s
            .iter()
            .chain(&s.gdn_conv)
            .chain(&s.gdn_rec)
            .chain([&s.ple_conv, &s.ple_win, &s.ctx4, &s.napply, &s.next])
            .collect()
    }

    /// A copy (on the device) of the sequence state after the last step, for prefix reuse:
    /// `restore_state` continues the sequence from here. The KV and indexer caches are not copied;
    /// their entries below this position stay valid until a step at those positions rewrites them.
    /// The buffers come from the pool of dropped snapshots when there is one (a dropped
    /// `Snapshot` returns its buffers to the engine), so only the first few saves allocate.
    pub fn save_state(&self) -> Res<Snapshot> {
        let bufs = self
            .snap_pool
            .lock()
            .map_err(|_| "snapshot pool poisoned")?
            .pop()
            .unwrap_or_default();
        let mut snap = Snapshot {
            pos: 0,
            bufs,
            drafts_ready: 0,
            pool: Some(self.snap_pool.clone()),
        };
        self.save_state_into(&mut snap)?;
        Ok(snap)
    }

    /// `save_state` into an existing snapshot, overwriting it in place (its buffers are reused;
    /// an empty `Snapshot::default()` allocates them once). About 170 MB of device copies.
    pub fn save_state_into(&self, snap: &mut Snapshot) -> Res<()> {
        let src = self.seq_state();
        if snap.bufs.len() != src.len()
            || snap.bufs.iter().zip(&src).any(|(a, b)| a.bytes != b.bytes)
        {
            snap.bufs = src
                .iter()
                .map(|b| cu(DeviceBuffer::new(b.bytes)))
                .collect::<Res<_>>()?;
        }
        for (dst, b) in snap.bufs.iter().zip(src) {
            cu(dst.copy_from_on(b, &self.stream))?;
        }
        cu(self.stream.synchronize())?;
        snap.pos = self.pos;
        snap.drafts_ready = self.drafts_ready;
        Ok(())
    }

    /// Frees the buffers of dropped snapshots that `save_state` would otherwise reuse.
    pub fn trim_snapshot_pool(&self) {
        if let Ok(mut p) = self.snap_pool.lock() {
            p.clear();
        }
    }

    /// Continues the sequence from `snap` (see `save_state`).
    pub fn restore_state(&mut self, snap: &Snapshot) -> Res<()> {
        let dsts = self.seq_state();
        if dsts.len() != snap.bufs.len()
            || dsts.iter().zip(&snap.bufs).any(|(d, s)| d.bytes != s.bytes)
        {
            return Err("restore_state: the snapshot holds no state of this engine".into());
        }
        for (dst, src) in dsts.into_iter().zip(&snap.bufs) {
            cu(dst.copy_from_on(src, &self.stream))?;
        }
        cu(self.stream.synchronize())?;
        self.pos = snap.pos;
        self.last_n = 0;
        self.drafts_ready = snap.drafts_ready;
        Ok(())
    }

    /// Loads a rank-1 LoRA adapter GGUF (heretic, or an abliteration converted to that form) and
    /// switches it on at scale 1. It applies to the main layers' output projections and down
    /// projections in fp32 at run time (adapter.rs); graphs are re-captured once.
    pub fn load_adapter(&mut self, path: &std::path::Path) -> Res<()> {
        let a = Adapter::load_gguf(path, &self.dims)?;
        self.adapter = Some(a);
        self.graphs.clear();
        self.set_adapter_scale(1.0)
    }

    /// Removes the adapter (graphs are re-captured once). To switch it off for a while, use
    /// `set_adapter_scale(0.0)` instead.
    pub fn drop_adapter(&mut self) {
        self.adapter = None;
        self.graphs.clear();
    }

    /// The adapter's strength from the next step on: 0 off, 1 as trained (no graph re-capture).
    pub fn set_adapter_scale(&mut self, scale: f32) -> Res<()> {
        cu(self.s.adapter_scale.upload_on(&[scale], &self.stream))?;
        cu(self.stream.synchronize())
    }

    /// The loaded adapter's name (its file stem).
    pub fn adapter_name(&self) -> Option<&str> {
        self.adapter.as_ref().map(|a| a.name.as_str())
    }

    /// Sets how the main head picks tokens from the next step on (no graph re-capture).
    pub fn set_sampling(&mut self, s: Sampling) -> Res<()> {
        cu(self.s.sampling.upload_on(&s.raw(), &self.stream))?;
        cu(self.stream.synchronize())
    }

    /// The token after the last step (greedy or sampled, see `set_sampling`).
    pub fn next_token(&self) -> Res<u32> {
        cu(self.stream.synchronize())?;
        Ok(cu(self.s.next.to_vec::<i32>(1))?[0] as u32)
    }

    /// The greedy token after each token of the last step.
    pub fn greedy_all(&self) -> Res<Vec<u32>> {
        cu(self.stream.synchronize())?;
        Ok(cu(self.s.out.to_vec::<i32>(self.last_n))?
            .into_iter()
            .map(|t| t as u32)
            .collect())
    }

    /// The logits after the last step's last token, on the host.
    pub fn logits(&self) -> Res<Vec<f32>> {
        let v = self.dims.vocab;
        Ok(self.logits_all()?.split_off((self.last_n.max(1) - 1) * v))
    }

    /// The logits after each token of the last step, [N][vocab] on the host.
    pub fn logits_all(&self) -> Res<Vec<f32>> {
        cu(self.stream.synchronize())?;
        cu(self.s.logits.to_vec(self.last_n.max(1) * self.dims.vocab))
    }

    pub fn synchronize(&self) -> Res<()> {
        cu(self.stream.synchronize())
    }
}

/// The sequence state at a position (see `Engine::save_state`). Dropping a snapshot made by
/// `save_state` hands its buffers back to the engine for the next save.
#[derive(Debug, Default)]
pub struct Snapshot {
    /// Tokens the sequence holds.
    pub pos: usize,
    bufs: Vec<DeviceBuffer>,
    drafts_ready: usize,
    pool: Option<SnapPool>,
}

impl Snapshot {
    /// Device bytes the snapshot holds (0 for an empty `Snapshot::default()`).
    pub fn bytes(&self) -> usize {
        self.bufs.iter().map(|b| b.bytes).sum()
    }
}

/// Buffer sets of dropped snapshots, shared by the engine and its snapshots.
type SnapPool = std::sync::Arc<std::sync::Mutex<Vec<Vec<DeviceBuffer>>>>;

/// Most dropped snapshots' buffer sets kept for reuse (~170 MB each on the real model).
const SNAP_POOL_MAX: usize = 4;

impl Drop for Snapshot {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool
            && !self.bufs.is_empty()
            && let Ok(mut p) = pool.lock()
            && p.len() < SNAP_POOL_MAX
        {
            p.push(std::mem::take(&mut self.bufs));
        }
    }
}

/// How the main head picks tokens (mirrors `SampleParams` in kernels/decode/sample.cu).
/// temperature <= 0 is greedy; top_k 0 and top_p >= 1 turn those filters off. A draw depends only
/// on (seed, position, logits), so speculative decoding gives the same tokens as one-token steps.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

impl Sampling {
    fn raw(&self) -> [u32; 6] {
        [
            self.temperature.to_bits(),
            self.top_k,
            self.top_p.to_bits(),
            0,
            self.seed as u32,
            (self.seed >> 32) as u32,
        ]
    }
}
