#[cfg(feature = "cuda")]
use super::transfer::{GpuImage, HostImage, Maintenance, ReadJob, Readers, SwapJob};
#[cfg(feature = "cuda")]
use crate::{
    expert_cache::{ExpertCache, ExpertId},
    offload::{DiskPolicy, ExpertStat, Tier},
};
use hayate_core::{DType, Error, QuantSpec, Result, hyt::HytFile};
use hayate_kernels::glm53::lowbit::{ExpertFormat, MatrixFormat};
#[cfg(feature = "cuda")]
use hayate_kernels::runtime::{Device, DeviceBuffer, Event, PinnedHost, Stream};
#[cfg(feature = "cuda")]
use std::{ffi::c_void, sync::Arc};

pub(super) const IMAGE_BYTES: usize = 3 * (2048 * 4096 + 16 * 32 * 4);
const IMAGE_ALIGN: usize = 256;

#[cfg(any(feature = "cuda", test))]
pub(super) fn gpu_budget(resident: usize, free: usize) -> Result<u64> {
    // Keep the existing 8 GiB temporary margin and staging image. Plan from
    // owned bytes, never from driver/context/other-process allocation jitter.
    let planned = hayate_core::machine::MAIN.planned_vram();
    let requested = resident as u64 + (8 << 30) + IMAGE_BYTES as u64;
    let gpu = planned.checked_sub(requested).ok_or(Error::OutOfMemory {
        buffer: "GLM resident weights/state/workspace",
        requested,
        available: planned,
    })?;
    if gpu > free as u64 {
        return Err(Error::OutOfMemory {
            buffer: "GLM fixed expert cache (free VRAM cannot change placement)",
            requested: gpu,
            available: free as u64,
        });
    }
    Ok(gpu)
}

/// Expert-cache bytes inside an explicit total budget: the budget minus the
/// resident bytes, 2 GiB of driver/cuBLAS/prefill headroom and the transfer
/// images (staging, 16 scratch, one spare per maintenance swap).
#[cfg(any(feature = "cuda", test))]
pub(super) fn budget_experts(
    total: u64,
    resident: usize,
    image: usize,
    swaps: usize,
    free: usize,
) -> Result<u64> {
    let fixed = resident as u64 + (2 << 30) + ((17 + swaps) * image) as u64;
    let gpu = total.checked_sub(fixed).ok_or(Error::OutOfMemory {
        buffer: "GLM VRAM budget (resident weights/state/workspace)",
        requested: fixed,
        available: total,
    })?;
    if gpu > free as u64 {
        return Err(Error::OutOfMemory {
            buffer: "GLM VRAM budget exceeds free VRAM",
            requested: gpu,
            available: free as u64,
        });
    }
    Ok(gpu - gpu % image as u64)
}

// Gate/up share a stride; down can retain block-FP8. Keep the FP32 matrix
// scale in the HYT payload; only pad after it, before the next matrix.
pub(super) fn matrix_stride(nvfp4: bool) -> usize {
    if nvfp4 {
        (QuantSpec::Nvfp4.matrix_bytes(2048, 4096).unwrap() as usize).next_multiple_of(IMAGE_ALIGN)
    } else {
        IMAGE_BYTES / 3
    }
}

pub(super) fn expert_format(file: &HytFile) -> (bool, bool) {
    let nvfp4 = file
        .entries()
        .iter()
        .find(|e| e.name == "mlp.experts.0.gate_proj.weight")
        .is_some_and(|e| e.quant == QuantSpec::Nvfp4);
    let down_fp8 = nvfp4
        && file
            .entries()
            .iter()
            .find(|e| e.name == "mlp.experts.0.down_proj.weight")
            .is_some_and(|e| e.quant == QuantSpec::Plain(DType::F8E4M3));
    (nvfp4, down_fp8)
}

/// Bytes of the largest expert image of a layer (three matrices at 256-byte strides).
pub(super) fn layer_image_bytes(file: &HytFile) -> Result<usize> {
    if !is_mixed(file) {
        let (nvfp4, down_fp8) = expert_format(file);
        return Ok(2 * matrix_stride(nvfp4) + matrix_stride(nvfp4 && !down_fp8));
    }
    let mut bytes = 0;
    for expert in 0..288 {
        bytes = bytes.max(mixed_expert(file, expert)?.2);
    }
    Ok(bytes)
}

/// Whether the layer's experts carry per-expert formats: any LOWBIT (HYT tag
/// 6) or EXL3 (tag 7) expert matrix.
pub(super) fn is_mixed(file: &HytFile) -> bool {
    file.entries().iter().any(|e| {
        matches!(e.quant, QuantSpec::Lowbit { .. } | QuantSpec::Exl3 { .. })
            && e.name.starts_with("mlp.experts.")
    })
}
/// Whether the layer's routed experts are EXL3 (the loader checks that all are).
pub(super) fn is_exl3(file: &HytFile) -> bool {
    file.entries()
        .iter()
        .any(|e| matches!(e.quant, QuantSpec::Exl3 { .. }) && e.name.starts_with("mlp.experts."))
}
/// An EXL3 layer's (gate/up k2, down k2): one bitrate each for all its routed experts (the kernels take the
/// layer's). `None` for other layers.
pub(super) fn exl3_bitrates(file: &HytFile) -> Result<Option<(u8, u8)>> {
    if !is_exl3(file) {
        return Ok(None);
    }
    let mut found: [Option<u8>; 2] = [None, None];
    for e in file.entries() {
        let Some(rest) = e.name.strip_prefix("mlp.experts.") else {
            continue;
        };
        let slot = if rest.ends_with(".down_proj.weight") {
            1
        } else {
            0
        };
        let QuantSpec::Exl3 { k2, cb: 2 } = e.quant else {
            return Err(Error::Weights(format!(
                "GLM EXL3 layer: {} is not mul1 EXL3",
                e.name
            )));
        };
        if *found[slot].get_or_insert(k2) != k2 {
            return Err(Error::Weights(
                "GLM EXL3 layer: mixed bitrates per layer are not supported".into(),
            ));
        }
    }
    match found {
        [Some(g), Some(d)] => Ok(Some((g, d))),
        _ => Err(Error::Weights("GLM EXL3 layer without experts".into())),
    }
}

fn cpu_format(format: MatrixFormat) -> hayate_kernels::glm53::cpu::Format {
    if format.format == 0 {
        hayate_kernels::glm53::cpu::Format::Nvfp4
    } else if format.is_exl3() {
        hayate_kernels::glm53::cpu::Format::Exl3 { k2: format.grid }
    } else {
        hayate_kernels::glm53::cpu::Format::Lowbit {
            grid: format.grid,
            group: format.group as usize,
        }
    }
}

fn matrix_format(quant: QuantSpec) -> Option<MatrixFormat> {
    match quant {
        QuantSpec::Nvfp4 => Some(MatrixFormat::NVFP4),
        QuantSpec::Lowbit { grid, group } => MatrixFormat::new(grid, group).ok(),
        // The kernels decode the mul1 codebook (exllamav3 1.6's default and only
        // one for half-integer K) at K 2, 2.5, 3, 3.5 and 4.
        QuantSpec::Exl3 {
            k2: k2 @ 4..=8,
            cb: 2,
        } => MatrixFormat::exl3(k2, 2).ok(),
        _ => None,
    }
}

/// Entry indices (gate, up, down), formats and image bytes (three matrices at
/// 256-byte strides) of one expert of a layer with LOWBIT experts.
pub(super) fn mixed_expert(
    file: &HytFile,
    expert: usize,
) -> Result<(Vec<usize>, ExpertFormat, usize)> {
    let mut indices = Vec::new();
    let mut formats = Vec::new();
    let mut bytes = 0;
    for (stem, dims) in [
        ("gate", [2048u64, 4096]),
        ("up", [2048, 4096]),
        ("down", [4096, 2048]),
    ] {
        let name = format!("mlp.experts.{expert}.{stem}_proj.weight");
        let index = file
            .entries()
            .iter()
            .position(|e| e.name == name)
            .ok_or_else(|| Error::Weights(format!("missing {name}")))?;
        let e = &file.entries()[index];
        let format = matrix_format(e.quant)
            .filter(|_| e.shape.dims() == dims)
            .ok_or_else(|| Error::Weights(format!("GLM expert layout: {name}")))?;
        indices.push(index);
        formats.push(format);
        bytes += (e.bytes as usize).next_multiple_of(IMAGE_ALIGN);
    }
    if formats[0] != formats[1] {
        return Err(Error::Weights(format!(
            "GLM expert {expert}: gate and up formats differ"
        )));
    }
    if formats[0].is_exl3() != formats[2].is_exl3() {
        return Err(Error::Weights(format!(
            "GLM expert {expert}: EXL3 and other matrices mixed"
        )));
    }
    Ok((
        indices,
        ExpertFormat {
            gate_up: formats[0],
            down: formats[2],
        },
        bytes,
    ))
}

pub(super) fn hadamard_block(file: &HytFile) -> Result<u32> {
    let Some((i, entry)) = file
        .entries()
        .iter()
        .enumerate()
        .find(|(_, e)| e.name == "mlp.experts.hadamard_block")
    else {
        return Ok(0);
    };
    if entry.quant != QuantSpec::Plain(hayate_core::DType::F32) || entry.shape.dims() != [1] {
        return Err(Error::Weights("GLM Hadamard metadata shape/type".into()));
    }
    let mut raw = [0; 4];
    file.read_tensor(i, &mut raw)?;
    let block = f32::from_le_bytes(raw);
    if ![32., 64., 128., 256.].contains(&block) {
        return Err(Error::Weights(
            "GLM Hadamard block must be 32/64/128/256".into(),
        ));
    }
    if file
        .entries()
        .iter()
        .filter(|e| {
            e.name.starts_with("mlp.experts.")
                && (e.name.ends_with("gate_proj.weight") || e.name.ends_with("up_proj.weight"))
        })
        .any(|e| e.quant != QuantSpec::Nvfp4)
    {
        return Err(Error::Weights("GLM Hadamard gate/up must be NVFP4".into()));
    }
    Ok(block as u32)
}

pub(super) fn validate_image_alignment(base: usize, nvfp4: bool, down_fp8: bool) -> Result<()> {
    let stride = matrix_stride(nvfp4);
    for (part, (n, k)) in [(2048usize, 4096usize), (2048, 4096), (4096, 2048)]
        .into_iter()
        .enumerate()
    {
        let weight = base + part * stride;
        let part_nvfp4 = nvfp4 && !(part == 2 && down_fp8);
        let scales = weight + n * k / if part_nvfp4 { 2 } else { 1 };
        let global = scales + n.div_ceil(128) * 128 * (k / 16).next_multiple_of(4);
        if !weight.is_multiple_of(IMAGE_ALIGN)
            || !scales.is_multiple_of(IMAGE_ALIGN)
            || (part_nvfp4 && !global.is_multiple_of(IMAGE_ALIGN))
        {
            return Err(Error::Weights(
                "GLM expert image alignment (256 bytes)".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct TransferStats {
    pub host_h2d: u64,
    pub disk_h2d: u64,
    pub h2d_bytes: u64,
    pub disk_bytes: u64,
    pub read_wait_ms: f64,
    pub reader_ms: f64,
    pub reader_io_ms: f64,
    pub reader_bytes: u64,
    pub reader_pack_ms: f64,
    pub reader_upload_ms: f64,
    pub predicted: u64,
    pub prefetch_hits: u64,
    pub maintenance_h2d: u64,
    pub maintenance_d2h: u64,
    pub maintenance_disk: u64,
    pub maintenance_bytes: u64,
    pub maintenance_wait_ms: f64,
    pub cpu_experts: u64,
    pub cpu_bytes: u64,
    pub cpu_ms: f64,
    pub cpu_worker_ms: f64,
    pub cpu_dispatch_ms: f64,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Transfers {
    failure: Arc<std::sync::atomic::AtomicI32>,
    readers: Readers,
    maintenance: Maintenance,
    copy: [Stream; 2],
    ready: [Event; 2],
    scratch: Vec<Option<DeviceBuffer>>,
    pending: [bool; 16],
    bank: usize,
    predicted_layer: Option<usize>,
    predicted: Vec<u16>,
    predicted_slots: Vec<usize>,
    swaps: usize,
    /// Pinned buffers for the evicted expert of an in-place swap.
    spare: Vec<PinnedHost>,
    interval: usize,
    window: usize,
    hysteresis: usize,
    asynchronous: bool,
    tokens: usize,
}

#[cfg(feature = "cuda")]
impl Transfers {
    fn new(
        device: &Device,
        files: Arc<Vec<HytFile>>,
        direct: Arc<Vec<std::fs::File>>,
        entries: Arc<Vec<Vec<usize>>>,
        image_bytes: usize,
        options: &super::Options,
    ) -> Result<Self> {
        if options.maintenance_async {
            // Swaps overwrite the slot in place: only at token boundaries.
            return Err(Error::InvalidPlan(
                "GLM --maintenance-async was removed (swaps run in place at token boundaries)"
                    .into(),
            ));
        }
        let interval = options.maintenance_interval;
        let reader_count = options.expert_readers;
        let window = options.maintenance_window;
        let max_swaps = options.maintenance_swaps;
        let failure = Arc::new(std::sync::atomic::AtomicI32::new(0));
        let copy = [device.stream()?, device.stream()?];
        let ready = [copy[0].event()?, copy[1].event()?];
        Ok(Self {
            readers: Readers::new(
                device,
                files.clone(),
                direct.clone(),
                entries.clone(),
                image_bytes,
                failure.clone(),
                reader_count,
            )?,
            maintenance: Maintenance::new(device, files, direct, entries, failure.clone())?,
            failure,
            copy,
            ready,
            scratch: (0..16)
                .map(|_| device.alloc(image_bytes).map(Some))
                .collect::<Result<_>>()?,
            pending: [false; 16],
            bank: 0,
            predicted_layer: None,
            predicted: Vec::new(),
            predicted_slots: Vec::new(),
            swaps: 0,
            spare: (0..max_swaps)
                .map(|_| device.alloc_pinned(image_bytes))
                .collect::<Result<_>>()?,
            interval,
            window,
            hysteresis: options.maintenance_hysteresis,
            asynchronous: options.maintenance_async,
            tokens: 0,
        })
    }
}

#[cfg(any(test, feature = "cuda"))]
fn batch_locations(
    ids: &[u16],
    resident: &[Option<usize>],
    predicted: &[u16],
    slots: &[usize],
    bank: usize,
) -> Vec<(bool, usize)> {
    let mut used = [false; 8];
    let mut locations: Vec<_> = ids
        .iter()
        .zip(resident)
        .map(|(&id, &slot)| {
            if let Some(slot) = slot {
                return (true, slot);
            }
            if let Some(index) = predicted.iter().position(|&p| p == id) {
                let slot = slots[index];
                used[slot % 8] = true;
                return (false, slot);
            }
            (false, usize::MAX)
        })
        .collect();
    // Reserve every successful prediction before assigning slots to actual
    // misses, even when a miss occurs earlier in the routing order.
    for location in &mut locations {
        if location.1 != usize::MAX {
            continue;
        }
        let slot = used.iter().position(|&u| !u).unwrap();
        used[slot] = true;
        *location = (false, bank * 8 + slot);
    }
    locations
}

#[cfg(any(test, feature = "cuda"))]
fn maintenance_due(tokens: usize, interval: usize, pending: bool) -> bool {
    interval != 0 && tokens.is_multiple_of(interval) && !pending
}

#[cfg(any(test, feature = "cuda"))]
fn host_limit(meminfo: &str) -> Result<u64> {
    let total = meminfo
        .lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next()? == "MemTotal:")
                .then(|| fields.next()?.parse::<u64>().ok())
                .flatten()
        })
        .ok_or_else(|| Error::InvalidPlan("GLM MemTotal unavailable".into()))?;
    // Leave at least 24 GiB for OS, resident weights/KV and staging/page cache.
    Ok(total
        .checked_mul(1024)
        .ok_or_else(|| Error::InvalidPlan("GLM MemTotal overflow".into()))?
        .saturating_sub(24 << 30))
}

fn host_budget(requested: Option<u64>, required: u64, limit: u64, allow_disk: bool) -> Result<u64> {
    let host = requested.unwrap_or(required.min(limit));
    if host > limit {
        return Err(Error::InvalidPlan(format!(
            "GLM host budget {host} exceeds safe limit {limit} (24 GiB reserved)"
        )));
    }
    if !allow_disk && required > host {
        return Err(Error::InvalidPlan(format!(
            "GLM disk-free placement needs {required} host bytes; budget={host}, safe_limit={limit}. Increase --host-gb within safe_limit or convert original weights to a smaller expert format offline."
        )));
    }
    Ok(host)
}

fn expert_placement(
    stats: &[crate::offload::ExpertStat],
    gpu: u64,
    host: u64,
    disk: crate::offload::DiskPolicy,
    mtp_cpu: bool,
) -> Result<crate::offload::Placement> {
    use crate::offload::{Budget, Tier, place_experts};
    if !mtp_cpu {
        return place_experts(stats, Budget { gpu, host }, disk);
    }
    let split = stats
        .iter()
        .position(|s| s.layer == 45)
        .ok_or_else(|| Error::InvalidPlan("GLM missing CPU MTP expert stats".into()))?;
    if stats[split..]
        .iter()
        .any(|s| s.layer != 45 || !(0.0..=1.0).contains(&s.freq))
    {
        return Err(Error::InvalidPlan("GLM CPU MTP expert stats".into()));
    }
    let reserved = stats[split..].iter().map(|s| s.host_bytes).sum::<u64>();
    let remaining = host.checked_sub(reserved).ok_or(Error::OutOfMemory {
        buffer: "GLM CPU MTP experts",
        requested: reserved,
        available: host,
    })?;
    let mut placement = place_experts(
        &stats[..split],
        Budget {
            gpu,
            host: remaining,
        },
        disk,
    )?;
    placement
        .tiers
        .extend(std::iter::repeat_n(Tier::HostCompute, stats.len() - split));
    placement.host_used += reserved;
    placement.host_bytes_per_token += stats[split..]
        .iter()
        .map(|s| f64::from(s.freq) * s.host_bytes as f64)
        .sum::<f64>();
    Ok(placement)
}

/// Original block-FP8 or HYT NVFP4 bytes, never a restored expert matrix. Misses use one
/// bounded GPU staging banks; host images contain only the planned RAM tier.
#[cfg(feature = "cuda")]
pub(super) struct PrefillLoan {
    pub slots: Vec<(usize, usize, bool)>,
    pub buffers: Vec<DeviceBuffer>,
    pub copied_bytes: usize,
    pub copy_events: [(Event, Event); 2],
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub(super) struct Experts {
    files: Arc<Vec<HytFile>>,
    entries: Arc<Vec<Vec<usize>>>,
    pub formats: Vec<(bool, bool)>,
    /// Per expert (statistics order): matrix formats of its image. NVFP4 for
    /// uniform NVFP4/FP8 layers; LOWBIT checkpoints mix NVFP4 and LOWBIT
    /// matrices per expert (`mixed` layers, decode through the LOWBIT kernels).
    pub expert_formats: Vec<ExpertFormat>,
    /// Exact image bytes per expert (statistics order). Host images are pinned
    /// buffers of `image_bytes` (the largest) of which this prefix is used;
    /// GPU slots hold exactly their expert's bytes, so smaller LOWBIT images
    /// fit more experts in the same VRAM.
    pub expert_bytes: Vec<usize>,
    /// Capacity of each cache slot (its buffer's bytes), in slot order.
    slot_caps: Vec<usize>,
    /// Per MoE layer: any expert is LOWBIT or EXL3 (per-expert formats).
    pub mixed: Vec<bool>,
    cpu_formats: Vec<Option<Arc<Vec<[hayate_kernels::glm53::cpu::Format; 2]>>>>,
    pub image_bytes: usize,
    pub counts: Vec<u64>,
    pub host_hits: u64,
    cache: ExpertCache,
    slots: Vec<Option<Arc<GpuImage>>>,
    host: Vec<Option<Arc<HostImage>>>,
    staging: PinnedHost,
    scratch: DeviceBuffer,
    pub hits: u64,
    pub main_hits: u64,
    pub main_routes: u64,
    mtp_cpu: bool,
    pub routes: u64,
    pub disk_reads: u64,
    pub d2h_count: u64,
    pub d2h_wait_ms: f64,
    pub host_bytes: u64,
    pub gpu_bytes: u64,
    pub disk_bytes: u64,
    transfers: Option<Transfers>,
    pub transfer_stats: TransferStats,
    cpu: Option<Arc<std::sync::Mutex<super::cpu_experts::CpuExperts>>>,
    pub asynchronous: Option<super::async_decode::Worker>,
    split: super::cpu_experts::Split,
    /// Load-time routing frequencies (expected selections per token).
    prior: Vec<f32>,
    host_limit: u64,
    allow_disk: bool,
    /// Route trace (`--save-routes`): per call u32 layer, count, expert ids.
    pub trace: Option<Vec<u32>>,
    /// Routes of the last `tail_tokens` context tokens per MoE layer (prefill
    /// and decode), ranked by `place_for_context` at the end of a prefill.
    tail: Vec<std::collections::VecDeque<u16>>,
    tail_tokens: usize,
    probe: Option<Probe>,
    probe_tokens: usize,
}

/// Decoded tokens since the last context placement (`probe_token`).
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Probe {
    tokens: usize,
    decode: Vec<u32>,
    /// Routes of the decoded tokens on the GPU set they ran with / in the
    /// context's top set.
    hits: u64,
    context_hits: u64,
    context_top: Vec<bool>,
    context: Vec<f64>,
    max_swaps: usize,
}

/// Result of a context placement (`Experts::place_for_context`).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default)]
pub struct Placed {
    pub swaps: usize,
    /// Bytes moved in each direction (H2D and D2H).
    pub bytes: u64,
    pub seconds: f64,
}

/// Result of an elastic budget change (`Experts::resize`).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default)]
pub struct Resize {
    pub slots_before: usize,
    pub slots_after: usize,
    /// GPU expert bytes released (positive) or newly allocated (negative).
    pub gpu_delta: i64,
    pub d2h_bytes: u64,
    pub h2d_bytes: u64,
    pub disk_reads: u64,
    /// Experts moved on the GPU (D2D) so that emptied split blocks are freed.
    pub packed: usize,
    /// Device memory actually returned to CUDA (free memory after - before).
    pub vram_freed: i64,
    pub seconds: f64,
}

/// Buffers per shared allocation for GPU images of `bytes`. cudaMalloc rounds
/// to 2 MiB pages: sizes that would waste 8% or more on their own are carved
/// from blocks of the count that wastes under 1% (others: one each).
#[cfg(feature = "cuda")]
fn split_group(bytes: usize) -> usize {
    let waste = |n: usize| n.next_multiple_of(2 << 20) - n;
    (1..=16usize)
        .find(|&g| waste(g * bytes) * 100 < g * bytes)
        .filter(|_| waste(bytes) * 100 >= 8 * bytes)
        .unwrap_or(1)
}

/// A GPU buffer for `pack_moves`: its size and split block (`None`: own allocation).
#[cfg(any(test, feature = "cuda"))]
#[derive(Debug, Clone, Copy)]
struct BufferUse {
    bytes: usize,
    block: Option<usize>,
}

/// Pairs that empty split blocks: (kept index, free index). `kept` are the
/// GPU images that stay (with whether they may move), `free` the released
/// buffers. Per buffer size, the smallest blocks are emptied into the free
/// buffers of the biggest ones, as many whole blocks as those can take (a
/// block that cannot be emptied entirely moves nothing).
#[cfg(any(test, feature = "cuda"))]
fn pack_moves(kept: &[(BufferUse, bool)], free: &[BufferUse]) -> Vec<(usize, usize)> {
    use std::collections::BTreeMap;
    #[derive(Default)]
    struct BlockUse {
        kept: Vec<usize>,
        free: Vec<usize>,
        pinned: bool,
    }
    let mut sizes: BTreeMap<usize, BTreeMap<usize, BlockUse>> = BTreeMap::new();
    for (k, &(buffer, movable)) in kept.iter().enumerate() {
        if let Some(block) = buffer.block {
            let b = sizes
                .entry(buffer.bytes)
                .or_default()
                .entry(block)
                .or_default();
            b.kept.push(k);
            b.pinned |= !movable;
        }
    }
    for (f, buffer) in free.iter().enumerate() {
        if let (Some(block), Some(blocks)) = (buffer.block, sizes.get_mut(&buffer.bytes))
            && let Some(b) = blocks.get_mut(&block)
        {
            b.free.push(f);
        }
    }
    let mut moves = Vec::new();
    for blocks in sizes.values() {
        // Destinations first: blocks holding an image in use (never emptied),
        // then the biggest blocks (kept + free), the fullest among equals.
        let mut order: Vec<&BlockUse> = blocks.values().collect();
        order.sort_by_key(|b| {
            use std::cmp::Reverse;
            (
                Reverse(b.pinned),
                Reverse(b.kept.len() + b.free.len()),
                Reverse(b.kept.len()),
            )
        });
        let n = order.len();
        let mut emptied = 0;
        while emptied < n && !order[n - emptied - 1].pinned {
            let src: usize = order[n - emptied - 1..].iter().map(|b| b.kept.len()).sum();
            let dst: usize = order[..n - emptied - 1].iter().map(|b| b.free.len()).sum();
            if src > dst {
                break;
            }
            emptied += 1;
        }
        let sources = order[n - emptied..]
            .iter()
            .flat_map(|b| b.kept.iter().copied());
        let dests = order[..n - emptied]
            .iter()
            .flat_map(|b| b.free.iter().copied());
        moves.extend(sources.zip(dests));
    }
    moves
}
#[cfg(feature = "cuda")]
impl Experts {
    pub fn load(
        device: &Device,
        files: Vec<HytFile>,
        gpu: u64,
        root: &std::path::Path,
        options: &super::Options,
    ) -> Result<Self> {
        let frequencies = options.frequencies.as_deref();
        let allow_disk = options.allow_disk;
        let interval = options.maintenance_interval;
        if frequencies.is_some_and(|f| f.len() != files.len().saturating_sub(3) * 288) {
            return Err(Error::InvalidPlan("GLM frequency count".into()));
        }
        let mixed: Vec<bool> = files.iter().skip(3).map(is_mixed).collect();
        let exl3: Vec<bool> = files.iter().skip(3).map(is_exl3).collect();
        let formats: Vec<_> = files
            .iter()
            .skip(3)
            .zip(&mixed)
            .map(|(file, &mixed)| {
                if mixed {
                    (true, false)
                } else {
                    expert_format(file)
                }
            })
            .collect();
        // Sensitivity runs can restore individual layers to FP8. Uniform cache
        // slots use the largest image; homogeneous runs retain their exact size.
        let mut image_bytes = formats
            .iter()
            .zip(&mixed)
            .filter(|&(_, &mixed)| !mixed)
            .map(|(&(nvfp4, down_fp8), _)| {
                2 * matrix_stride(nvfp4) + matrix_stride(nvfp4 && !down_fp8)
            })
            .max()
            .unwrap_or(0);
        let mut stats = Vec::new();
        let mut entries = Vec::new();
        let mut expert_formats = Vec::new();
        let mut expert_bytes = Vec::new();
        for (layer, file) in files.iter().enumerate().skip(3) {
            let (nvfp4, down_fp8) = formats[layer - 3];
            for expert in 0..288 {
                if mixed[layer - 3] {
                    let (indices, format, bytes) = mixed_expert(file, expert)?;
                    if format.gate_up.is_exl3() != exl3[layer - 3] {
                        return Err(Error::Weights(format!(
                            "GLM layer {layer}: EXL3 and other expert formats mixed"
                        )));
                    }
                    image_bytes = image_bytes.max(bytes);
                    entries.push(indices);
                    expert_formats.push(format);
                    expert_bytes.push(bytes);
                    stats.push(ExpertStat {
                        layer: layer as u16,
                        expert: expert as u16,
                        freq: frequencies.map_or(8.0 / 288.0, |f| f[stats.len()]),
                        gpu_bytes: 0,
                        host_bytes: 0,
                    });
                    continue;
                }
                expert_formats.push(ExpertFormat {
                    gate_up: MatrixFormat::NVFP4,
                    down: MatrixFormat::NVFP4,
                });
                // Uniform layers keep the uniform (largest) slot size.
                expert_bytes.push(usize::MAX);
                let mut indices = Vec::new();
                for (part, stem) in ["gate", "up", "down"].into_iter().enumerate() {
                    let part_nvfp4 = nvfp4 && !(part == 2 && down_fp8);
                    let (n, k) = if part == 2 {
                        (4096, 2048)
                    } else {
                        (2048, 4096)
                    };
                    for (suffix, dims, quant, at) in [
                        (
                            "weight",
                            vec![n, k],
                            if part_nvfp4 {
                                QuantSpec::Nvfp4
                            } else {
                                QuantSpec::Plain(DType::F8E4M3)
                            },
                            part * 2,
                        ),
                        (
                            "weight_scale_inv",
                            vec![n / 128, k / 128],
                            QuantSpec::Plain(DType::F32),
                            part * 2 + 1,
                        ),
                    ] {
                        if part_nvfp4 && at % 2 == 1 {
                            continue;
                        }
                        let name = format!("mlp.experts.{expert}.{stem}_proj.{suffix}");
                        let index = file
                            .entries()
                            .iter()
                            .position(|e| e.name == name)
                            .ok_or_else(|| Error::Weights(format!("missing {name}")))?;
                        let e = &file.entries()[index];
                        if e.shape.dims() != dims || e.quant != quant {
                            return Err(Error::Weights(format!("GLM expert layout: {name}")));
                        }
                        indices.push(index);
                    }
                }
                entries.push(indices);
                stats.push(ExpertStat {
                    layer: layer as u16,
                    expert: expert as u16,
                    freq: frequencies.map_or(8.0 / 288.0, |f| f[stats.len()]),
                    gpu_bytes: image_bytes as u64,
                    host_bytes: image_bytes as u64,
                });
            }
        }
        // Pinned images and staging buffers use the largest image; a GPU slot
        // only its expert's bytes (uniform layers: the largest).
        for (s, bytes) in stats.iter_mut().zip(&mut expert_bytes) {
            *bytes = (*bytes).min(image_bytes);
            s.gpu_bytes = *bytes as u64;
            s.host_bytes = image_bytes as u64;
        }
        let limit = host_limit(&std::fs::read_to_string("/proc/meminfo")?)?;
        let eligible = stats
            .iter()
            .filter(|s| !options.mtp_cpu_experts || s.layer != 45)
            .count();
        // Experts the GPU budget takes in frequency order (by their own bytes);
        // the rest need a pinned image of the largest size each.
        let on_gpu = {
            let mut order: Vec<usize> = (0..stats.len())
                .filter(|&i| !options.mtp_cpu_experts || stats[i].layer != 45)
                .collect();
            order.sort_by(|&a, &b| stats[b].freq.total_cmp(&stats[a].freq).then(a.cmp(&b)));
            let mut used = 0u64;
            order
                .into_iter()
                .take_while(|&i| {
                    used += stats[i].gpu_bytes;
                    used <= gpu
                })
                .count()
                .min(eligible)
        };
        let required_host = (stats.len() - on_gpu) as u64 * image_bytes as u64;
        let host = host_budget(options.host_bytes, required_host, limit, allow_disk)?;
        let placement = expert_placement(
            &stats,
            gpu,
            host,
            if allow_disk {
                DiskPolicy::Allow
            } else {
                DiskPolicy::Refuse
            },
            options.mtp_cpu_experts,
        )?;
        let cache =
            ExpertCache::with_disk_sizes(&stats, &placement, 0.9, options.maintenance_swaps)?;
        // Slots in cache order (the k-th GPU-tier expert by index), each as big as
        // its expert. cudaMalloc rounds to 2 MiB: images are carved from shared
        // blocks of the size that wastes under 1% (bigger images: one each).
        let slot_caps: Vec<usize> = placement
            .tiers
            .iter()
            .zip(&expert_bytes)
            .filter(|&(&tier, _)| tier == Tier::Gpu)
            .map(|(_, &bytes)| bytes)
            .collect();
        let mut by_size: std::collections::BTreeMap<usize, usize> = Default::default();
        for &bytes in &slot_caps {
            *by_size.entry(bytes).or_default() += 1;
        }
        let mut pools: std::collections::BTreeMap<usize, Vec<DeviceBuffer>> = Default::default();
        for (&bytes, &count) in &by_size {
            let group = split_group(bytes);
            let pool = pools.entry(bytes).or_default();
            while pool.len() < count {
                let n = group.min(count - pool.len());
                pool.extend(device.alloc_split(n, bytes)?);
            }
        }
        let slots: Vec<Option<DeviceBuffer>> = slot_caps
            .iter()
            .map(|bytes| pools.get_mut(bytes).and_then(Vec::pop))
            .collect();
        let host_images = (0..stats.len()).map(|_| None).collect();
        let files = Arc::new(files);
        let entries = Arc::new(entries);
        let mut result = Self {
            files: files.clone(),
            entries: entries.clone(),
            formats,
            cpu_formats: mixed
                .iter()
                .enumerate()
                .map(|(l, &mixed)| {
                    mixed.then(|| {
                        Arc::new(
                            expert_formats[l * 288..][..288]
                                .iter()
                                .map(|f| [cpu_format(f.gate_up), cpu_format(f.down)])
                                .collect(),
                        )
                    })
                })
                .collect(),
            expert_formats,
            expert_bytes,
            slot_caps,
            mixed,
            image_bytes,
            counts: vec![0; stats.len()],
            host_hits: 0,
            cache,
            slots: Vec::new(),
            host: host_images,
            staging: device.alloc_pinned(image_bytes)?,
            scratch: device.alloc(image_bytes)?,
            hits: 0,
            main_hits: 0,
            main_routes: 0,
            mtp_cpu: options.mtp_cpu_experts,
            routes: 0,
            disk_reads: 0,
            d2h_count: 0,
            d2h_wait_ms: 0.,
            host_bytes: placement.host_used,
            gpu_bytes: placement.gpu_used + image_bytes as u64,
            disk_bytes: placement.tiers.iter().filter(|&&t| t == Tier::Disk).count() as u64
                * image_bytes as u64,
            transfers: None,
            transfer_stats: TransferStats::default(),
            cpu: None,
            asynchronous: None,
            split: crate::glm53::cpu_experts::Split::new(1.)?,
            prior: stats.iter().map(|s| s.freq).collect(),
            host_limit: options.host_bytes.unwrap_or(limit).max(host),
            allow_disk,
            trace: None,
            tail: Vec::new(),
            // A frozen cache (maintenance interval 0) also skips context placement.
            tail_tokens: if interval == 0 {
                0
            } else {
                options.placement_tail
            },
            probe: None,
            probe_tokens: options.placement_probe,
        };
        for slot in &slots {
            for &(nvfp4, down_fp8) in &result.formats {
                validate_image_alignment(
                    slot.as_ref().unwrap().as_ptr() as usize,
                    nvfp4,
                    down_fp8,
                )?;
            }
        }
        for &(nvfp4, down_fp8) in &result.formats {
            validate_image_alignment(result.scratch.as_ptr() as usize, nvfp4, down_fp8)?;
            validate_image_alignment(result.staging.as_slice().as_ptr() as usize, nvfp4, down_fp8)?;
        }
        // Four E-core readers fill the RAM tier directly and use one pinned
        // staging image each for GPU slots. No duplicate host expert copies.
        let mut targets = Vec::new();
        let mut gpu_slots = slots.into_iter();
        for (i, tier) in placement.tiers.iter().enumerate() {
            match tier {
                Tier::Gpu => targets.push((i, gpu_slots.next().unwrap(), None)),
                Tier::HostCompute => {
                    targets.push((i, None, Some(device.alloc_pinned(image_bytes)?)))
                }
                Tier::Disk => {}
            }
        }
        let chunk = targets.len().div_ceil(4).max(1);
        std::thread::scope(|scope| -> Result<()> {
            let workers: Vec<_> = targets
                .chunks_mut(chunk)
                .enumerate()
                .map(|(worker, jobs)| {
                    let files = &result.files;
                    let entries = &result.entries;
                    let sizes = &result.expert_bytes;
                    scope.spawn(move || -> Result<()> {
                        hayate_kernels::cpu::pin_thread(
                            hayate_core::machine::MAIN.cpu.service[worker],
                        )?;
                        let stream = device.stream()?;
                        let mut staging = device.alloc_pinned(image_bytes)?;
                        for (i, gpu, host) in jobs {
                            if let Some(host) = host {
                                read_image(
                                    &files[*i / 288 + 3],
                                    &entries[*i],
                                    host.as_mut_slice(),
                                )?;
                            } else if let Some(gpu) = gpu {
                                read_image(
                                    &files[*i / 288 + 3],
                                    &entries[*i],
                                    staging.as_mut_slice(),
                                )?;
                                stream.copy_h2d(gpu, &staging.as_slice()[..sizes[*i]])?;
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            for worker in workers {
                worker
                    .join()
                    .map_err(|_| Error::Weights("GLM expert load worker panicked".into()))??;
            }
            Ok(())
        })?;
        for (i, gpu, host) in targets {
            if let Some(gpu) = gpu {
                result.slots.push(Some(Arc::new(GpuImage(gpu))));
            }
            result.host[i] = host.map(|h| Arc::new(HostImage(h)));
        }
        use std::os::unix::fs::OpenOptionsExt;
        let direct = Arc::new(
            (0..files.len())
                .map(|layer| {
                    std::fs::OpenOptions::new()
                        .read(true)
                        .custom_flags(0o40000)
                        .open(root.join(format!("layer{layer}/weights.hyt")))
                })
                .collect::<std::io::Result<Vec<_>>>()?,
        );
        result.transfers = Some(Transfers::new(
            device,
            files,
            direct,
            entries,
            image_bytes,
            options,
        )?);
        result.gpu_bytes += ((16 + options.maintenance_swaps) * image_bytes) as u64;
        result.disk_reads = 0;
        result.split = super::cpu_experts::Split::new(options.pcie_frac)?;
        if options.pcie_frac < 1. && result.host_bytes != 0 {
            result.cpu = Some(Arc::new(std::sync::Mutex::new(
                super::cpu_experts::CpuExperts::with_threads(
                    options.cpu_threads,
                    options.cpu_spin,
                )?,
            )));
        }
        eprintln!(
            "GLM CPU experts: pcie_frac={} compute_cores={:?} spin={} host_core={:?} maintenance_window={} maintenance_swaps={} maintenance_hysteresis={} maintenance_async={} disk_policy={:?}",
            options.pcie_frac,
            &hayate_core::machine::MAIN.cpu.compute[..options.cpu_threads],
            options.cpu_spin,
            options.host_core,
            options.maintenance_window,
            options.maintenance_swaps,
            options.maintenance_hysteresis,
            options.maintenance_async,
            if allow_disk {
                DiskPolicy::Allow
            } else {
                DiskPolicy::Refuse
            }
        );
        eprintln!(
            "GLM expert placement: image_bytes={image_bytes} gpu_slots={} gpu_bytes={} host_bytes={} disk_bytes={} readers={} scratch_images={} pinned_staging_bytes={} pinned_expert_bytes={} maintenance_interval={interval}",
            result.slots.len(),
            placement.gpu_used,
            result.host_bytes,
            result.disk_bytes,
            options.expert_readers,
            17 + options.maintenance_swaps,
            (options.expert_readers + 1 + options.maintenance_swaps) * image_bytes
                + (options.expert_readers + 1) * (super::transfer::DIRECT_BYTES + 4095),
            result.host.iter().filter(|h| h.is_some()).count() * image_bytes,
        );
        Ok(result)
    }
    fn index(layer: usize, expert: usize) -> usize {
        (layer - 3) * 288 + expert
    }
    pub fn slot_count(&self) -> usize {
        self.cache.slot_count()
    }
    /// Elastic VRAM budget: keep the `slots` highest-ranked experts on the GPU
    /// (rolling-window routing score plus the load-time frequency prior).
    /// Call only at a token boundary with no async snapshot or prefill loan
    /// alive. Experts leaving the GPU get a pinned RAM copy while the host
    /// limit allows, otherwise they fall back to the disk tier (requires
    /// `allow_disk`); experts entering the GPU release their RAM copy.
    pub fn resize(&mut self, device: &Device, budget: u64) -> Result<Resize> {
        let start = std::time::Instant::now();
        self.drain()?;
        let free_before = device.memory_info()?.0;
        let before = self.cache.slot_count();
        let before_bytes: u64 = self.slot_caps.iter().map(|&b| b as u64).sum();
        let window = self.transfers.as_ref().map_or(64, |t| t.window) as f64;
        let image = self.image_bytes as u64;
        let eligible: Vec<usize> = (0..self.host.len())
            .filter(|&i| !(self.mtp_cpu && i / 288 + 3 == 45))
            .collect();
        let rank = |i: usize| self.cache.score(i) + f64::from(self.prior[i]) * window;
        let mut ranked = eligible.clone();
        ranked.sort_by(|&a, &b| rank(b).total_cmp(&rank(a)).then(a.cmp(&b)));
        // Best-ranked experts while their images fit the byte budget (a big
        // one that no longer fits does not stop smaller ones behind it).
        let mut order = Vec::new();
        let mut taken = 0u64;
        for &i in &ranked {
            let bytes = self.expert_bytes[i] as u64;
            if taken + bytes <= budget {
                taken += bytes;
                order.push(i);
            }
        }
        let mut desired = vec![false; self.host.len()];
        for &i in &order {
            desired[i] = true;
        }
        let current: Vec<usize> = (0..before)
            .map(|s| {
                let id = self.cache.resident(s).unwrap();
                Self::index(id.layer as usize, id.expert as usize)
            })
            .collect();
        let leaving: Vec<usize> = current
            .iter()
            .enumerate()
            .filter(|&(_, &i)| !desired[i])
            .map(|(s, _)| s)
            .collect();
        let mut on_gpu = vec![false; self.host.len()];
        for &i in &current {
            on_gpu[i] = true;
        }
        let entering: Vec<usize> = order.iter().copied().filter(|&i| !on_gpu[i]).collect();
        // Plan RAM first so a refused shrink changes nothing.
        let held = self.host.iter().filter(|h| h.is_some()).count() as u64 * image;
        let need = leaving
            .iter()
            .filter(|&&s| self.host[current[s]].is_none())
            .count() as u64;
        let room = self.host_limit.saturating_sub(held) / image;
        let spill = need.saturating_sub(room);
        if spill > 0 && !self.allow_disk {
            return Err(Error::OutOfMemory {
                buffer: "GLM elastic shrink (RAM copies of evicted experts; enable --allow-disk)",
                requested: need * image,
                available: room * image,
            });
        }
        let mut report = Resize {
            slots_before: before,
            slots_after: order.len(),
            ..Resize::default()
        };
        // 1. Evict: RAM copy (D2H on both copy streams) or disk tier.
        {
            let copies = &self.transfers.as_ref().unwrap().copy;
            let targets: Vec<(usize, usize)> = leaving
                .iter()
                .map(|&s| (s, current[s]))
                .filter(|&(_, i)| self.host[i].is_none())
                .take(room as usize)
                .collect();
            // Page pinning dominates (one cudaMallocHost per image); allocate
            // every destination first, on several threads, then copy.
            let image_bytes = self.image_bytes;
            let mut hosts: Vec<PinnedHost> =
                std::thread::scope(|scope| {
                    let workers: Vec<_> = (0..4)
                        .map(|w| {
                            let count = targets.len() / 4 + usize::from(w < targets.len() % 4);
                            scope.spawn(move || {
                                (0..count)
                                    .map(|_| device.alloc_pinned(image_bytes))
                                    .collect::<Result<Vec<_>>>()
                            })
                        })
                        .collect();
                    let mut all = Vec::new();
                    for worker in workers {
                        all.extend(worker.join().map_err(|_| {
                            Error::Weights("GLM pinned allocation thread".into())
                        })??);
                    }
                    Ok::<_, Error>(all)
                })?;
            let mut backups = Vec::new();
            let mut submitted = Ok(());
            for (j, &(s, i)) in targets.iter().enumerate() {
                let mut host = hosts.pop().unwrap();
                let bytes = self.expert_bytes[i];
                submitted = submitted.and_then(|()| unsafe {
                    copies[j % 2].copy_d2h_prefix_async(
                        &mut host,
                        &self.slots[s].as_ref().unwrap().0,
                        bytes,
                    )
                });
                report.d2h_bytes += bytes as u64;
                backups.push((i, host));
            }
            for copy in copies {
                copy.synchronize()?;
            }
            submitted?;
            for (i, host) in backups {
                self.host[i] = Some(Arc::new(HostImage(host)));
            }
        }
        let mut freed: Vec<DeviceBuffer> = Vec::new();
        for &s in &leaving {
            let gpu = self.slots[s]
                .take()
                .ok_or_else(|| Error::InvalidPlan("GLM resize: empty slot".into()))?;
            freed.push(
                Arc::try_unwrap(gpu)
                    .map_err(|_| Error::InvalidPlan("GLM resize: GPU image still in use".into()))?
                    .0,
            );
        }
        // Reuse released buffers for entering experts (the smallest that
        // fits) before allocating.
        let mut kept: Vec<(usize, Arc<GpuImage>)> = current
            .iter()
            .enumerate()
            .filter_map(|(s, &i)| self.slots[s].take().map(|g| (i, g)))
            .collect();
        {
            let copies = &self.transfers.as_ref().unwrap().copy;
            let mut uploads = 0usize;
            for (pos, &i) in entering.iter().enumerate() {
                let bytes = self.expert_bytes[i];
                let fit = freed
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| b.bytes() >= bytes)
                    .min_by_key(|(_, b)| b.bytes())
                    .map(|(k, _)| k);
                let mut gpu = match fit {
                    Some(k) => freed.swap_remove(k),
                    None => {
                        // A new block as at load; its spare buffers serve the
                        // next entering experts of this size.
                        let left = entering[pos..]
                            .iter()
                            .filter(|&&j| self.expert_bytes[j] == bytes)
                            .count();
                        let group = split_group(bytes).min(left);
                        if group > 1 {
                            let mut block = device.alloc_split(group, bytes)?;
                            let gpu = block.pop().unwrap();
                            freed.extend(block);
                            gpu
                        } else {
                            device.alloc(bytes)?
                        }
                    }
                };
                if let Some(host) = &self.host[i] {
                    unsafe { copies[uploads % 2].copy_h2d_prefix_async(&mut gpu, &host.0, bytes)? };
                    uploads += 1;
                    report.h2d_bytes += bytes as u64;
                } else {
                    copies[uploads % 2].synchronize()?;
                    read_image(
                        &self.files[i / 288 + 3],
                        &self.entries[i],
                        self.staging.as_mut_slice(),
                    )?;
                    copies[uploads % 2].copy_h2d(&mut gpu, &self.staging.as_slice()[..bytes])?;
                    report.disk_reads += 1;
                    report.h2d_bytes += bytes as u64;
                }
                kept.push((i, Arc::new(GpuImage(gpu))));
            }
            for copy in copies {
                copy.synchronize()?;
            }
            // A split block returns to CUDA only when all its buffers are
            // dropped, and a shrink leaves scattered holes: move the images of
            // sparsely used blocks into the free buffers of fuller ones (D2D),
            // so the emptied blocks go back with `freed`.
            let uses: Vec<(BufferUse, bool)> = kept
                .iter()
                .map(|(_, g)| {
                    let use_ = BufferUse {
                        bytes: g.0.bytes(),
                        block: g.0.block(),
                    };
                    (use_, Arc::strong_count(g) == 1)
                })
                .collect();
            let free_uses: Vec<BufferUse> = freed
                .iter()
                .map(|b| BufferUse {
                    bytes: b.bytes(),
                    block: b.block(),
                })
                .collect();
            let moves = pack_moves(&uses, &free_uses);
            if !moves.is_empty() {
                let mut spare: Vec<Option<DeviceBuffer>> =
                    std::mem::take(&mut freed).into_iter().map(Some).collect();
                let mut old = Vec::with_capacity(moves.len());
                let mut submitted = Ok(());
                for (j, &(k, f)) in moves.iter().enumerate() {
                    let mut dst = spare[f].take().unwrap();
                    let bytes = self.expert_bytes[kept[k].0];
                    submitted = submitted.and_then(|()| unsafe {
                        copies[j % 2].copy_d2d_prefix_async(&mut dst, &kept[k].1.0, bytes)
                    });
                    old.push(std::mem::replace(&mut kept[k].1, Arc::new(GpuImage(dst))));
                }
                for copy in copies {
                    copy.synchronize()?;
                }
                submitted?;
                drop(old);
                freed = spare.into_iter().flatten().collect();
                report.packed = moves.len();
            }
        }
        drop(freed);
        kept.sort_by_key(|&(i, _)| i);
        let order: Vec<usize> = kept.iter().map(|&(i, _)| i).collect();
        self.cache.reassign(&order)?;
        self.slot_caps = kept.iter().map(|(_, g)| g.0.bytes()).collect();
        self.slots = kept.into_iter().map(|(_, g)| Some(g)).collect();
        // Entering experts no longer need RAM copies (same invariant as load).
        for &i in &entering {
            if let Some(host) = self.host[i].take()
                && Arc::strong_count(&host) != 1
            {
                self.host[i] = Some(host);
            }
        }
        if let Some(t) = self.transfers.as_mut() {
            t.predicted_layer = None;
            t.predicted.clear();
            t.predicted_slots.clear();
        }
        let after = self.slots.len();
        let after_bytes: u64 = self.slot_caps.iter().map(|&b| b as u64).sum();
        report.gpu_delta = before_bytes as i64 - after_bytes as i64;
        report.vram_freed = device.memory_info()?.0 as i64 - free_before as i64;
        self.gpu_bytes = (self.gpu_bytes as i64 - report.gpu_delta) as u64;
        self.host_bytes = self.host.iter().filter(|h| h.is_some()).count() as u64 * image;
        self.disk_bytes = (self.host.len() as u64 - after as u64) * image - self.host_bytes;
        report.seconds = start.elapsed().as_secs_f64();
        Ok(report)
    }
    /// Slots a prefill may borrow (their buffers are image-sized).
    pub fn loanable_slots(&self) -> usize {
        self.slot_caps
            .iter()
            .filter(|&&b| b >= self.image_bytes)
            .count()
    }
    /// Bytes of the GPU cache slots (sum of their capacities).
    pub fn slot_bytes(&self) -> u64 {
        self.slot_caps.iter().map(|&b| b as u64).sum()
    }
    /// Routes of the last context tokens (for session checkpoints).
    pub fn tail_routes(&self) -> Vec<Vec<u16>> {
        self.tail
            .iter()
            .map(|row| row.iter().copied().collect())
            .collect()
    }
    pub fn set_tail(&mut self, tail: &[Vec<u16>]) {
        self.tail = tail
            .iter()
            .map(|row| row.iter().copied().collect())
            .collect();
    }
    pub fn clear_tail(&mut self) {
        self.tail.clear();
        self.probe = None;
    }
    /// End of a prefill (or a restore without a saved placement): starts a
    /// probe over the next `probe_tokens` decoded tokens and restarts the
    /// rolling maintenance window; no swaps here. The context score is
    /// beta x routes per token over the last `tail_tokens` context tokens +
    /// (1 - beta) x load-time frequency. Placing by it right away backfires
    /// when the reply is unlike its prompt (an agent reply after tool
    /// definitions), so `probe_token` decides from the decoded tokens.
    pub fn place_for_context(&mut self, beta: f64, max_swaps: usize) {
        self.probe = None;
        if self.tail_tokens == 0 || max_swaps == 0 || self.tail.is_empty() {
            return;
        }
        self.cache.reset_recent();
        let mut rate = vec![0f64; self.host.len()];
        for (l, row) in self.tail.iter().enumerate() {
            let tokens = (row.len() / super::TOP_K).max(1) as f64;
            for &e in row {
                rate[l * 288 + e as usize] += 1. / tokens;
            }
        }
        let score: Vec<f64> = rate
            .iter()
            .zip(&self.prior)
            .map(|(&r, &p)| beta * r + (1. - beta) * f64::from(p))
            .collect();
        let mut order: Vec<usize> = (0..score.len()).collect();
        order.sort_by(|&a, &b| self.better(&score, b, a));
        let mut context_top = vec![false; score.len()];
        for &i in &order[..self.cache.slot_count().min(order.len())] {
            context_top[i] = true;
        }
        self.probe = Some(Probe {
            tokens: 0,
            decode: vec![0; self.host.len()],
            hits: 0,
            context_hits: 0,
            context_top,
            context: score,
            max_swaps,
        });
    }
    /// After each decoded token: at `probe_tokens` tokens after a context
    /// placement, place once (uncapped up to `max_swaps`) by the decoded
    /// tokens' routes per token, ties broken by the context score when the
    /// context's top set would have served those tokens at least as well as
    /// the current set, otherwise by the load-time frequency.
    pub fn probe_token(&mut self, device: &Device) -> Result<Placed> {
        let Some(probe) = self.probe.as_mut() else {
            return Ok(Placed::default());
        };
        probe.tokens += 1;
        if probe.tokens < self.probe_tokens {
            return Ok(Placed::default());
        }
        let probe = self.probe.take().unwrap();
        let tie: Vec<f64> = if probe.context_hits >= probe.hits {
            probe.context
        } else {
            self.prior.iter().map(|&p| f64::from(p)).collect()
        };
        let top = tie.iter().copied().fold(0., f64::max).max(1e-30);
        // Most experts already on the GPU (2-bit mixes hold ~90% or more): the
        // context ranking only reshuffles the cold tail and the swaps (seconds)
        // outweigh what a short reply gains. Bring in only experts the probe
        // tokens actually used.
        let eligible = (0..self.host.len())
            .filter(|&i| !(self.mtp_cpu && i / 288 + 3 == 45))
            .count();
        let observed_only = self.cache.slot_count() * 100 >= eligible * 85;
        let score: Vec<f64> = probe
            .decode
            .iter()
            .zip(&tie)
            .enumerate()
            .map(|(i, (&d, &t))| {
                let resident = self
                    .cache
                    .slot(ExpertId {
                        layer: (i / 288 + 3) as u16,
                        expert: (i % 288) as u16,
                    })
                    .is_some();
                if observed_only && d == 0 && !resident {
                    -1.
                } else {
                    f64::from(d) / probe.tokens as f64 + 0.01 * t / top
                }
            })
            .collect();
        self.place_scores(device, &score, probe.max_swaps)
    }
    /// Order for placement: higher score, then load-time frequency, then lower index.
    fn better(&self, score: &[f64], a: usize, b: usize) -> std::cmp::Ordering {
        score[a]
            .total_cmp(&score[b])
            .then(self.prior[a].total_cmp(&self.prior[b]))
            .then(b.cmp(&a))
    }
    /// Moves the GPU set toward the highest `score` (ties by load-time
    /// frequency, then index), at most `max_swaps` pairs, best entering
    /// against worst leaving while the order improves; an uncapped call
    /// leaves exactly the top scores on the GPU whatever came before, and the
    /// rolling maintenance window restarts. A pair uploads the entering
    /// expert's pinned RAM image and then downloads the leaving one into that
    /// same buffer (same stream, so in order); the leaving slot's buffer
    /// receives the next pair's upload. No allocation beyond one image per
    /// copy stream. Call at a token boundary with no snapshot, loan or CPU
    /// job holding images (held ones are skipped).
    fn place_scores(&mut self, device: &Device, score: &[f64], max_swaps: usize) -> Result<Placed> {
        let start = std::time::Instant::now();
        self.drain()?;
        self.cache.reset_recent();
        let better = |a: usize, b: usize| self.better(score, a, b);
        let slots = self.cache.slot_count();
        let current: Vec<usize> = (0..slots)
            .map(|s| {
                let id = self.cache.resident(s).unwrap();
                Self::index(id.layer as usize, id.expert as usize)
            })
            .collect();
        let mut on_gpu = vec![false; self.host.len()];
        for &i in &current {
            on_gpu[i] = true;
        }
        // Best experts outside the GPU (with an unshared RAM image), and the
        // weakest residents (unshared slot), both in rank order.
        let mut entering: Vec<usize> = (0..self.host.len())
            .filter(|&i| {
                !on_gpu[i]
                    && !(self.mtp_cpu && i / 288 + 3 == 45)
                    && self.host[i]
                        .as_ref()
                        .is_some_and(|h| Arc::strong_count(h) == 1)
            })
            .collect();
        entering.sort_by(|&a, &b| better(b, a));
        let mut leaving: Vec<usize> = (0..slots)
            .filter(|&s| {
                self.slots[s]
                    .as_ref()
                    .is_some_and(|g| Arc::strong_count(g) == 1)
            })
            .collect();
        leaving.sort_by(|&a, &b| better(current[a], current[b]));
        // Each entering expert takes the smallest slot it fits among the
        // residents it beats (the weakest first among equals).
        let mut used = vec![false; slots];
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for &e in &entering {
            if pairs.len() >= max_swaps {
                break;
            }
            let mut best: Option<usize> = None;
            for &s in leaving.iter().filter(|&&s| !used[s]) {
                if better(e, current[s]).is_le() {
                    break;
                }
                if self.slot_caps[s] >= self.expert_bytes[e]
                    && best.is_none_or(|b| self.slot_caps[s] < self.slot_caps[b])
                {
                    best = Some(s);
                }
            }
            match best {
                Some(s) => {
                    used[s] = true;
                    pairs.push((s, e));
                }
                // The weakest unused resident beats it: nothing left to improve.
                None if leaving
                    .iter()
                    .find(|&&s| !used[s])
                    .is_none_or(|&s| better(e, current[s]).is_le()) =>
                {
                    break;
                }
                None => {}
            }
        }
        if pairs.is_empty() {
            return Ok(Placed::default());
        }
        // In place: the evicted expert is copied into a pinned buffer, then the
        // entering one (from its RAM image) overwrites the slot, on one stream
        // per pair. A stream's next pair reuses the buffer the previous entering
        // expert was read from (stream order makes that safe).
        let copies = &self.transfers.as_ref().unwrap().copy;
        let mut spare = [
            Some(device.alloc_pinned(self.image_bytes)?),
            if pairs.len() > 1 {
                Some(device.alloc_pinned(self.image_bytes)?)
            } else {
                None
            },
        ];
        let mut done = Vec::new();
        let mut submitted = Ok(());
        for (j, &(s, e)) in pairs.iter().enumerate() {
            let k = j % 2;
            let mut evicted = spare[k]
                .take()
                .ok_or_else(|| Error::InvalidPlan("GLM placement: no spare buffer".into()))?;
            let source = Arc::try_unwrap(self.host[e].take().unwrap())
                .map_err(|_| Error::InvalidPlan("GLM placement: RAM image in use".into()))?
                .0;
            let mut gpu = Arc::try_unwrap(self.slots[s].take().unwrap())
                .map_err(|_| Error::InvalidPlan("GLM placement: GPU image in use".into()))?
                .0;
            let (evict_bytes, load_bytes) = (self.expert_bytes[current[s]], self.expert_bytes[e]);
            submitted = submitted.and_then(|()| unsafe {
                copies[k].copy_d2h_prefix_async(&mut evicted, &gpu, evict_bytes)?;
                copies[k].copy_h2d_prefix_async(&mut gpu, &source, load_bytes)
            });
            spare[k] = Some(source);
            done.push((s, current[s], gpu, evicted));
        }
        for copy in copies {
            copy.synchronize()?;
        }
        let mut loaded = 0;
        for ((s, evicted, gpu, image), &(_, e)) in done.into_iter().zip(&pairs) {
            self.slots[s] = Some(Arc::new(GpuImage(gpu)));
            self.host[evicted] = Some(Arc::new(HostImage(image)));
            loaded += self.expert_bytes[e] as u64;
        }
        submitted?;
        drop(spare);
        let mut placed = current;
        for &(s, e) in &pairs {
            placed[s] = e;
        }
        self.cache.reassign(&placed)?;
        if let Some(t) = self.transfers.as_mut() {
            t.predicted_layer = None;
            t.predicted.clear();
            t.predicted_slots.clear();
        }
        Ok(Placed {
            swaps: pairs.len(),
            bytes: loaded,
            seconds: start.elapsed().as_secs_f64(),
        })
    }
    pub fn enable_async(&mut self) -> Result<()> {
        self.asynchronous = Some(super::async_decode::Worker::new(
            self.cpu.clone(),
            Some((self.files.clone(), self.entries.clone(), self.image_bytes)),
        )?);
        Ok(())
    }
    pub fn snapshot(&self, layer: usize) -> super::async_decode::Snapshot {
        let host = self.host[Self::index(layer, 0)..Self::index(layer, 0) + 288].to_vec();
        let gpu = (0..288)
            .map(|expert| {
                self.cache
                    .slot(ExpertId {
                        layer: layer as u16,
                        expert,
                    })
                    .and_then(|s| self.slots[s].as_ref())
                    .map_or(0, |g| g.0.as_ptr() as u64)
            })
            .collect();
        super::async_decode::Snapshot {
            host,
            gpu,
            formats: self.cpu_formats(layer),
        }
    }
    /// CPU formats of a layer with LOWBIT experts (`None`: the layer format applies).
    pub fn cpu_formats(
        &self,
        layer: usize,
    ) -> Option<Arc<Vec<[hayate_kernels::glm53::cpu::Format; 2]>>> {
        self.cpu_formats[layer - 3].clone()
    }
    /// GPU image address of every expert of layers 3.. (0: not on the GPU),
    /// `[layer - 3][expert]`, in one pass over the slots.
    pub fn gpu_table(&self, table: &mut [u64]) {
        table.fill(0);
        for (slot, image) in self.slots.iter().enumerate() {
            if let (Some(image), Some(id)) = (image, self.cache.resident(slot)) {
                let i = Self::index(id.layer as usize, id.expert as usize);
                if let Some(entry) = table.get_mut(i) {
                    *entry = image.0.as_ptr() as u64;
                }
            }
        }
    }
    /// RAM images of one layer (doorbell worker snapshot).
    pub fn host_row(&self, layer: usize) -> Vec<Option<Arc<HostImage>>> {
        self.host[Self::index(layer, 0)..Self::index(layer, 0) + 288].to_vec()
    }
    pub fn retire_async(&mut self) -> Result<()> {
        let completed: Vec<_> = self
            .asynchronous
            .as_ref()
            .map(|w| w.done.try_iter().collect())
            .unwrap_or_default();
        for done in completed {
            done.result?;
            for ids in done.ids.as_chunks::<8>().0 {
                self.route(
                    done.layer,
                    &ids.iter().map(|&e| e as u16).collect::<Vec<_>>(),
                )?;
            }
            self.transfer_stats.cpu_experts += done.stats.cpu_experts;
            self.transfer_stats.cpu_bytes += done.stats.cpu_bytes;
            self.transfer_stats.cpu_ms += done.stats.cpu_ms;
            self.transfer_stats.cpu_worker_ms += done.stats.cpu_worker_ms;
            self.transfer_stats.cpu_dispatch_ms += done.stats.cpu_dispatch_ms;
        }
        Ok(())
    }
    /// Device decode: block until the doorbell worker reported `count` layers
    /// of the token (layers without misses are not awaited by the GPU), then
    /// record their routes like `retire_async`.
    pub fn retire_device(&mut self, count: usize) -> Result<()> {
        let mut first = Ok(());
        for _ in 0..count {
            let done = self
                .asynchronous
                .as_ref()
                .ok_or_else(|| Error::InvalidPlan("GLM device decode needs the worker".into()))?
                .done
                .recv()
                .map_err(|_| Error::Weights("GLM doorbell worker stopped".into()))?;
            if let Err(error) = done.result {
                if first.is_ok() {
                    first = Err(error);
                }
                continue;
            }
            self.route(
                done.layer,
                &done.ids.iter().map(|&e| e as u16).collect::<Vec<_>>(),
            )?;
            self.transfer_stats.cpu_experts += done.stats.cpu_experts;
            self.transfer_stats.cpu_bytes += done.stats.cpu_bytes;
            self.transfer_stats.cpu_ms += done.stats.cpu_ms;
            self.transfer_stats.cpu_worker_ms += done.stats.cpu_worker_ms;
            self.transfer_stats.cpu_dispatch_ms += done.stats.cpu_dispatch_ms;
        }
        first
    }
    pub fn main_placement(&self) -> (u64, u64) {
        let mut hash = 0xcbf29ce484222325u64;
        let mut bytes = 0;
        for slot in 0..self.cache.slot_count() {
            let id = self.cache.resident(slot).unwrap();
            if id.layer == 45 {
                continue;
            }
            bytes += self.slot_caps[slot] as u64;
            for byte in id
                .layer
                .to_le_bytes()
                .into_iter()
                .chain(id.expert.to_le_bytes())
            {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        (bytes, hash)
    }
    pub fn layer_gpu_bytes(&self, layer: usize) -> u64 {
        (0..288)
            .filter(|&expert| {
                self.cache
                    .slot(ExpertId {
                        layer: layer as u16,
                        expert,
                    })
                    .is_some()
            })
            .map(|expert| self.expert_bytes[Self::index(layer, expert as usize)] as u64)
            .sum()
    }
    pub fn maintenance_async(&self) -> bool {
        self.transfers.as_ref().is_some_and(|t| t.asynchronous)
    }
    pub fn maintenance_interval(&self) -> usize {
        self.transfers.as_ref().unwrap().interval
    }
    pub fn layer_count(&self) -> usize {
        self.formats.len() + 3
    }
    pub fn format(&self, layer: usize) -> (bool, bool) {
        self.formats[layer - 3]
    }
    /// The layer has LOWBIT experts: decode and prefill use the per-expert formats.
    pub fn is_mixed(&self, layer: usize) -> bool {
        self.mixed[layer - 3]
    }
    /// Paths that predate LOWBIT (synchronous decode, per-expert CPU calls)
    /// assume one format per layer.
    fn require_uniform(&self, layer: usize) -> Result<()> {
        if self.mixed[layer - 3] {
            Err(Error::InvalidPlan(
                "GLM LOWBIT experts need device decode (--decode-host device)".into(),
            ))
        } else {
            Ok(())
        }
    }
    /// Formats of the 288 experts of a MoE layer.
    pub fn layer_formats(&self, layer: usize) -> &[ExpertFormat] {
        &self.expert_formats[(layer - 3) * 288..][..288]
    }
    fn read(&mut self, i: usize) -> Result<()> {
        read_image(
            &self.files[i / 288 + 3],
            &self.entries[i],
            self.staging.as_mut_slice(),
        )?;
        self.disk_reads += 1;
        Ok(())
    }
    pub fn check_worker_error(&self) -> Result<()> {
        self.transfers
            .as_ref()
            .map_or(Ok(()), |t| super::transfer::worker_status(&t.failure))
    }
    pub fn route(&mut self, layer: usize, ids: &[u16]) -> Result<()> {
        if let Some(trace) = &mut self.trace {
            trace.extend([layer as u32, ids.len() as u32]);
            trace.extend(ids.iter().map(|&e| u32::from(e)));
        }
        // CPU-only MTP never records promotion scores in the main GPU cache.
        let hits = if self.mtp_cpu && layer == 45 {
            if ids.iter().any(|&id| id >= 288) {
                return Err(Error::InvalidShape("GLM CPU MTP expert ID".into()));
            }
            0
        } else {
            self.cache
                .route(layer as u16, ids)?
                .gpu
                .iter()
                .filter(|g| self.slots[g.slot].is_some())
                .count() as u64
        };
        self.routes += ids.len() as u64;
        self.hits += hits;
        if layer != 45 {
            self.main_routes += ids.len() as u64;
            self.main_hits += hits;
        }
        for &expert in ids {
            let i = Self::index(layer, expert as usize);
            self.counts[i] += 1;
            if self.host[i].is_some() {
                self.host_hits += 1;
            }
        }
        if let Some(p) = self.probe.as_mut() {
            p.hits += hits;
            for &expert in ids {
                let i = Self::index(layer, expert as usize);
                p.decode[i] += 1;
                p.context_hits += u64::from(p.context_top[i]);
            }
        }
        if self.tail_tokens > 0 {
            if self.tail.len() <= layer - 3 {
                self.tail.resize_with(layer - 2, Default::default);
            }
            let keep = self.tail_tokens * super::TOP_K;
            let row = &mut self.tail[layer - 3];
            row.extend(&ids[ids.len().saturating_sub(keep)..]);
            if row.len() > keep {
                row.drain(..row.len() - keep);
            }
        }
        Ok(())
    }
    pub fn image(&mut self, layer: usize, expert: usize, stream: &Stream) -> Result<*const c_void> {
        self.check_worker_error()?;
        if let Some(slot) = self
            .cache
            .slot(ExpertId {
                layer: layer as u16,
                expert: expert as u16,
            })
            .filter(|&s| self.slots[s].is_some())
        {
            return Ok(self.slots[slot].as_ref().unwrap().0.as_ptr());
        }
        let i = Self::index(layer, expert);
        // Retire the previous scratch use before overwriting it.
        stream.synchronize()?;
        if let Some(image) = &self.host[i] {
            stream.copy_h2d(&mut self.scratch, image.as_slice())?;
            self.transfer_stats.host_h2d += 1;
            self.transfer_stats.h2d_bytes += self.image_bytes as u64;
        } else {
            self.read(i)?;
            stream.copy_h2d(&mut self.scratch, self.staging.as_slice())?;
            self.transfer_stats.disk_h2d += 1;
            self.transfer_stats.h2d_bytes += self.image_bytes as u64;
            self.transfer_stats.disk_bytes += self.image_bytes as u64;
        }
        Ok(self.scratch.as_ptr())
    }
    pub fn borrow_prefill(
        &mut self,
        device: &Device,
        stream: &Stream,
        count: usize,
        wait: bool,
    ) -> Result<PrefillLoan> {
        // With wait=false only the independent dense layer may run before
        // wait_prefill_loan. Neither the returned buffers nor new host images
        // may be accessed until that wait, including on an error path.
        self.drain()?;
        // Only slots of the full image size: their buffers become prefill
        // workspace and image banks (smaller LOWBIT slots stay).
        let mut candidates: Vec<_> = (0..self.slots.len())
            .filter(|&slot| self.slot_caps[slot] >= self.image_bytes)
            .map(|slot| {
                let id = self.cache.resident(slot).unwrap();
                (slot, Self::index(id.layer as usize, id.expert as usize))
            })
            .collect();
        let count = count.min(candidates.len());
        candidates.sort_by_key(|&(slot, i)| (self.host[i].is_none(), std::cmp::Reverse(slot)));
        let mut saved = Vec::new();
        let copies = &self.transfers.as_ref().unwrap().copy;
        let copy_events = [
            (copies[0].event()?, copies[0].event()?),
            (copies[1].event()?, copies[1].event()?),
        ];
        let ready = stream.event()?;
        ready.record(stream)?;
        for copy in copies {
            copy.wait_event(&ready)?;
        }
        let result = (|| -> Result<()> {
            for &(slot, i) in &candidates[..count] {
                let image = self.slots[slot].as_ref().unwrap();
                if Arc::strong_count(image) != 1 {
                    return Err(Error::InvalidPlan("GLM prefill loan still in use".into()));
                }
                if self.host[i].is_some() {
                    saved.push((slot, i, None));
                } else {
                    // Keep every destination alive even if submission fails.
                    saved.push((slot, i, Some(device.alloc_pinned(self.image_bytes)?)));
                }
            }
            for (copy, (begin, _)) in copies.iter().zip(&copy_events) {
                begin.record(copy)?;
            }
            for (j, (slot, i, backup)) in saved.iter_mut().enumerate() {
                if let Some(backup) = backup {
                    unsafe {
                        copies[j % 2].copy_d2h_prefix_async(
                            backup,
                            &self.slots[*slot].as_ref().unwrap().0,
                            self.expert_bytes[*i],
                        )?;
                    }
                }
            }
            for (copy, (_, end)) in copies.iter().zip(&copy_events) {
                end.record(copy)?;
            }
            Ok(())
        })();
        if wait || result.is_err() {
            self.wait_prefill_loan()?;
        }
        result?;
        let copied_bytes = saved.iter().filter(|(_, _, h)| h.is_some()).count() * self.image_bytes;
        let mut pool = Vec::new();
        let mut loan = Vec::new();
        for (slot, i, backup) in saved {
            let temporary = backup.is_some();
            if let Some(backup) = backup {
                self.host[i] = Some(Arc::new(HostImage(backup)));
            }
            pool.push(Arc::try_unwrap(self.slots[slot].take().unwrap()).unwrap().0);
            loan.push((slot, i, temporary));
        }
        Ok(PrefillLoan {
            slots: loan,
            buffers: pool,
            copied_bytes,
            copy_events,
        })
    }
    pub fn wait_prefill_loan(&self) -> Result<()> {
        let sync: Vec<_> = self
            .transfers
            .as_ref()
            .unwrap()
            .copy
            .iter()
            .map(Stream::synchronize)
            .collect();
        for result in sync {
            result?;
        }
        Ok(())
    }
    pub fn return_prefill(
        &mut self,
        device: &Device,
        _stream: &Stream,
        loan: &mut Vec<(usize, usize, bool)>,
    ) -> Result<()> {
        // Keep the backup published until the complete upload succeeds. On an
        // error a later reset/drop retains every still-needed source allocation.
        let copies = &self.transfers.as_ref().unwrap().copy;
        let mut restored = loan
            .iter()
            .map(|&(slot, _, _)| device.alloc(self.slot_caps[slot]))
            .collect::<Result<Vec<_>>>()?;
        let result = (|| -> Result<()> {
            for (j, (&(_, i, _), image)) in loan.iter().zip(&mut restored).enumerate() {
                unsafe {
                    copies[j % 2].copy_h2d_prefix_async(
                        image,
                        &self.host[i].as_ref().unwrap().0,
                        self.expert_bytes[i],
                    )?;
                }
            }
            Ok(())
        })();
        let sync: Vec<_> = copies.iter().map(Stream::synchronize).collect();
        for result in sync {
            result?;
        }
        result?;
        for ((slot, i, temporary), image) in loan.drain(..).zip(restored) {
            self.slots[slot] = Some(Arc::new(GpuImage(image)));
            if temporary {
                self.host[i] = None;
            }
        }
        Ok(())
    }
    /// Most experts of one MoE layer that are not on the GPU (lookahead size).
    pub fn max_missing(&self) -> usize {
        (3..self.layer_count())
            .map(|layer| (0..288).filter(|&e| !self.on_gpu(layer, e)).count())
            .max()
            .unwrap_or(0)
    }
    pub fn on_gpu(&self, layer: usize, expert: usize) -> bool {
        self.cache
            .slot(ExpertId {
                layer: layer as u16,
                expert: expert as u16,
            })
            .is_some()
    }
    /// The image of `expert` if it is already on the GPU for a prefill: its
    /// cache slot, or the lookahead bank.
    pub fn prefill_image(
        &mut self,
        layer: usize,
        expert: usize,
        ahead: Option<&super::prefill::Bank>,
    ) -> Option<*mut c_void> {
        if let Some(slot) = self
            .cache
            .slot(ExpertId {
                layer: layer as u16,
                expert: expert as u16,
            })
            .filter(|&s| self.slots[s].is_some())
        {
            return Some(self.slots[slot].as_ref().unwrap().0.as_ptr().cast_mut());
        }
        let ahead = ahead?;
        let at = ahead.ids.iter().position(|&id| id == expert)?;
        self.transfer_stats.prefetch_hits += 1;
        Some(ahead.images[at])
    }
    pub fn prefill_images(
        &mut self,
        layer: usize,
        ids: &[usize],
        bank: &mut super::prefill::Bank,
        upload: &Stream,
        ahead: Option<&super::prefill::Bank>,
    ) -> Result<()> {
        self.check_worker_error()?;
        // Host images stay immutable for the whole prefill. Retain their Arcs
        // until release, so recycling a bank needs no host-side CUDA wait.
        upload.wait_event(&bank.used)?;
        bank.begin.record(upload)?;
        bank.images.clear();
        bank.ids = ids.to_vec();
        bank.copied_bytes = 0;
        bank.host_bytes = 0;
        bank.disk_bytes = 0;
        let result = (|| -> Result<()> {
            for (j, &expert) in ids.iter().enumerate() {
                if let Some(slot) = self
                    .cache
                    .slot(ExpertId {
                        layer: layer as u16,
                        expert: expert as u16,
                    })
                    .filter(|&s| self.slots[s].is_some())
                {
                    bank.images
                        .push(self.slots[slot].as_ref().unwrap().0.as_ptr().cast_mut());
                    continue;
                }
                if let Some(ahead) = ahead
                    && let Some(at) = ahead.ids.iter().position(|&id| id == expert)
                {
                    bank.images.push(ahead.images[at]);
                    self.transfer_stats.prefetch_hits += 1;
                    continue;
                }
                let i = Self::index(layer, expert);
                if let Some(image) = &self.host[i] {
                    let image = image.clone();
                    unsafe {
                        upload.copy_h2d_at_async(
                            &mut bank.arena[j],
                            0,
                            image.0.range(0..self.expert_bytes[i])?,
                        )?;
                    }
                    if !bank.holds.iter().any(|held| Arc::ptr_eq(held, &image)) {
                        bank.holds.push(image);
                    }
                    self.transfer_stats.host_h2d += 1;
                    bank.host_bytes += self.expert_bytes[i];
                } else {
                    // Disk is explicit opt-in. Only this path needs staging.
                    upload.synchronize()?;
                    read_image(
                        &self.files[layer],
                        &self.entries[i],
                        bank.staging.as_mut_slice(),
                    )?;
                    self.disk_reads += 1;
                    unsafe {
                        upload.copy_h2d_at_async(
                            &mut bank.arena[j],
                            0,
                            bank.staging.range(0..self.expert_bytes[i])?,
                        )?;
                    }
                    self.transfer_stats.disk_h2d += 1;
                    self.transfer_stats.disk_bytes += self.expert_bytes[i] as u64;
                    bank.disk_bytes += self.expert_bytes[i];
                }
                self.transfer_stats.h2d_bytes += self.expert_bytes[i] as u64;
                bank.copied_bytes += self.expert_bytes[i];
                bank.images.push(bank.arena[j].as_ptr().cast_mut());
            }
            Ok(())
        })();
        if let Err(error) = result {
            upload.synchronize()?;
            return Err(error);
        }
        bank.ready.record(upload)
    }
    pub fn cpu_enabled(&self) -> bool {
        self.cpu.is_some()
    }
    /// Routed-expert LoRA for the CPU experts (the async/device workers share the pool).
    pub fn set_lora(&mut self, lora: Arc<super::lora::HostExperts>) {
        if let Some(cpu) = &self.cpu {
            cpu.lock().unwrap().lora = (!lora.is_empty()).then_some(lora);
        }
    }
    pub fn prefill_transfer_bytes(&self, rows: usize) -> u64 {
        // At most eight different experts per token, capped at all 288.
        let host = self.host.iter().filter(|h| h.is_some()).count() as u64;
        host * self.image_bytes as u64 * (rows * 8).min(288) as u64 / 288
    }
    /// Arcs keep images immutable through concurrent promotion. Publication
    /// only occurs at a later token boundary, after this layer has finished.
    pub fn cpu_images(&mut self, layer: usize, ids: &[u16]) -> Vec<Option<Arc<HostImage>>> {
        ids.iter()
            .map(|&expert| {
                let id = ExpertId {
                    layer: layer as u16,
                    expert,
                };
                if self.cpu.is_none() || self.cache.slot(id).is_some() {
                    return None;
                }
                if self.split.pcie() {
                    return None;
                }
                self.host[Self::index(layer, expert as usize)].clone()
            })
            .collect()
    }
    /// `images` and `ids` are per routed slot (`None`: computed on the GPU).
    pub fn compute_cpu(
        &mut self,
        layer: usize,
        images: &[Option<Arc<HostImage>>],
        ids: &[u16],
        input: &[u8],
        output: &mut [u8],
        hadamard: u32,
    ) -> Result<()> {
        self.require_uniform(layer)?;
        let (nv, down_fp8) = self.format(layer);
        let start = std::time::Instant::now();
        let before = self
            .cpu
            .as_ref()
            .map(|c| c.lock().unwrap().timing())
            .unwrap_or_default();
        let mut misses: Vec<_> = images
            .iter()
            .zip(ids)
            .zip(output.as_chunks_mut::<8192>().0)
            .filter_map(|((image, &id), output)| {
                image.as_ref().map(|image| super::cpu_experts::Miss {
                    image: image.as_slice(),
                    input,
                    output,
                    expert: Some((layer, id as usize)),
                })
            })
            .collect();
        let count = misses.len() as u64;
        if count != 0 {
            self.cpu.as_ref().unwrap().lock().unwrap().forward_layer(
                &mut misses,
                nv,
                down_fp8,
                hadamard,
            )?;
        }
        if count != 0 {
            self.transfer_stats.cpu_experts += count;
            self.transfer_stats.cpu_bytes +=
                count * (2 * matrix_stride(nv) + matrix_stride(nv && !down_fp8)) as u64;
            self.transfer_stats.cpu_ms += start.elapsed().as_secs_f64() * 1000.;
            let after = self.cpu.as_ref().unwrap().lock().unwrap().timing();
            self.transfer_stats.cpu_worker_ms += after.0 - before.0;
            self.transfer_stats.cpu_dispatch_ms += after.1 - before.1;
        }
        Ok(())
    }
    pub fn compute_cpu_group(
        &mut self,
        layer: usize,
        image: &[u8],
        input: &[u8],
        output: &mut [u8],
        hadamard: u32,
    ) -> Result<()> {
        self.require_uniform(layer)?;
        let (nv, down_fp8) = self.format(layer);
        let start = std::time::Instant::now();
        let before = self
            .cpu
            .as_ref()
            .map(|c| c.lock().unwrap().timing())
            .unwrap_or_default();
        self.cpu
            .as_mut()
            .ok_or_else(|| Error::InvalidPlan("GLM CPU experts disabled".into()))?
            .lock()
            .unwrap()
            .forward(image, input, output, nv, down_fp8, hadamard)?;
        self.transfer_stats.cpu_experts += 1;
        self.transfer_stats.cpu_bytes +=
            (2 * matrix_stride(nv) + matrix_stride(nv && !down_fp8)) as u64;
        self.transfer_stats.cpu_ms += start.elapsed().as_secs_f64() * 1000.;
        let after = self.cpu.as_ref().unwrap().lock().unwrap().timing();
        self.transfer_stats.cpu_worker_ms += after.0 - before.0;
        self.transfer_stats.cpu_dispatch_ms += after.1 - before.1;
        Ok(())
    }
    pub fn decode_images(
        &mut self,
        layer: usize,
        ids: &[u16],
        stream: &Stream,
    ) -> Result<Vec<*const c_void>> {
        self.batch_images(layer, ids, stream)
    }
    fn accept_swap(&mut self, done: super::transfer::SwapDone) -> Result<()> {
        let t = self.transfers.as_mut().unwrap();
        t.swaps -= 1;
        // The swap ran in place: on failure the slot's content is unreliable
        // (the error stops decoding), but its buffer and the spare return.
        let swap = done.replacement;
        self.slots[swap.slot] = Some(Arc::new(GpuImage(done.gpu)));
        if let Err(error) = done.result {
            t.spare.push(done.image);
            return Err(error);
        }
        self.cache.apply_prepared(swap)?;
        let load = Self::index(swap.load.layer as usize, swap.load.expert as usize);
        let evict = Self::index(swap.evict.layer as usize, swap.evict.expert as usize);
        let spare = if done.from_host {
            let source = self.host[load].take().unwrap();
            self.host[evict] = Some(Arc::new(HostImage(done.image)));
            self.d2h_count += 1;
            Arc::try_unwrap(source)
                .map_err(|_| Error::Weights("GLM host image still borrowed".into()))?
                .0
        } else {
            done.image
        };
        self.d2h_wait_ms += done.worker_ms;
        t.spare.push(spare);
        if done.disk_bytes != 0 {
            self.disk_reads += 1;
        }
        self.transfer_stats.disk_bytes += done.disk_bytes;
        Ok(())
    }
    fn poll_swaps(&mut self) -> Result<()> {
        loop {
            let Some(t) = &self.transfers else {
                return Ok(());
            };
            match t.maintenance.done.try_recv() {
                Ok(done) => self.accept_swap(done)?,
                Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(()),
                Err(_) => return Err(Error::Weights("GLM maintenance stopped".into())),
            }
        }
    }
    fn retire_reads(&mut self, bank: usize) -> Result<()> {
        let t = self.transfers.as_mut().unwrap();
        if !t.pending[bank * 8..bank * 8 + 8].iter().any(|&p| p) {
            return Ok(());
        }
        let start = std::time::Instant::now();
        while t.pending[bank * 8..bank * 8 + 8].iter().any(|&p| p) {
            let done = t
                .readers
                .done
                .recv()
                .map_err(|_| Error::Weights("GLM disk reader stopped".into()))?;
            t.pending[done.scratch] = false;
            t.scratch[done.scratch] = Some(done.gpu);
            self.transfer_stats.reader_ms += done.read_ms;
            self.transfer_stats.reader_io_ms += done.io_ms;
            self.transfer_stats.reader_bytes += done.bytes;
            self.transfer_stats.reader_pack_ms += done.pack_ms;
            self.transfer_stats.reader_upload_ms += done.upload_ms;
            self.transfer_stats.disk_bytes += done.bytes;
            done.result?;
        }
        self.transfer_stats.read_wait_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
    fn enqueue_image(&mut self, layer: usize, expert: u16, scratch: usize) -> Result<()> {
        self.check_worker_error()?;
        let i = Self::index(layer, expert as usize);
        let t = self.transfers.as_mut().unwrap();
        if let Some(image) = &self.host[i] {
            unsafe {
                t.copy[scratch / 8].copy_h2d_async(t.scratch[scratch].as_mut().unwrap(), &image.0)
            }?;
            self.transfer_stats.host_h2d += 1;
        } else {
            let gpu = t.scratch[scratch].take().unwrap();
            t.readers.submit(ReadJob {
                index: i,
                scratch,
                gpu,
            })?;
            t.pending[scratch] = true;
            self.disk_reads += 1;
            self.transfer_stats.disk_h2d += 1;
        }
        self.transfer_stats.h2d_bytes += self.image_bytes as u64;
        Ok(())
    }
    // IDs from the next layer's router on the current normalized activation are
    // hints only. Actual top-k IDs/weights are always recomputed by that layer.
    pub fn prefetch(&mut self, layer: usize, ids: &[u16]) -> Result<()> {
        if self.transfers.is_none() {
            return Ok(());
        }
        let bank = self.transfers.as_ref().unwrap().bank;
        self.retire_reads(bank)?;
        self.transfers.as_mut().unwrap().predicted.clear();
        self.transfers.as_mut().unwrap().predicted_slots.clear();
        for &expert in ids {
            if self
                .cache
                .slot(ExpertId {
                    layer: layer as u16,
                    expert,
                })
                .is_some()
            {
                continue;
            }
            let slot = bank * 8 + self.transfers.as_ref().unwrap().predicted.len();
            self.enqueue_image(layer, expert, slot)?;
            let t = self.transfers.as_mut().unwrap();
            t.predicted.push(expert);
            t.predicted_slots.push(slot);
            self.transfer_stats.predicted += 1;
        }
        let t = self.transfers.as_mut().unwrap();
        t.predicted_layer = Some(layer);
        t.ready[bank].record(&t.copy[bank])
    }
    fn batch_images(
        &mut self,
        layer: usize,
        ids: &[u16],
        stream: &Stream,
    ) -> Result<Vec<*const c_void>> {
        if ids.len() > 8 {
            return Err(Error::InvalidShape("GLM transfer top-k".into()));
        }
        let bank = self.transfers.as_ref().unwrap().bank;
        // Complete this bank's speculative reads once, then recycle only unused
        // entries. The previous layer's router wait retired this bank's old use.
        self.retire_reads(bank)?;
        if self.transfers.as_ref().unwrap().predicted_layer == Some(layer) {
            self.transfers.as_ref().unwrap().ready[bank].synchronize()?;
        }
        let resident: Vec<_> = ids
            .iter()
            .map(|&expert| {
                self.cache.slot(ExpertId {
                    layer: layer as u16,
                    expert,
                })
            })
            .collect();
        let t = self.transfers.as_ref().unwrap();
        let (predicted, predicted_slots) = if t.predicted_layer == Some(layer) {
            (&t.predicted[..], &t.predicted_slots[..])
        } else {
            (&[][..], &[][..])
        };
        let locations = batch_locations(ids, &resident, predicted, predicted_slots, bank);
        let missed: Vec<_> = ids
            .iter()
            .zip(&locations)
            .filter_map(|(&expert, &(gpu, slot))| {
                if gpu {
                    None
                } else if predicted.contains(&expert) {
                    self.transfer_stats.prefetch_hits += 1;
                    None
                } else {
                    Some((expert, slot))
                }
            })
            .collect();
        for (expert, slot) in missed {
            self.enqueue_image(layer, expert, slot)?;
        }
        self.retire_reads(bank)?;
        let t = self.transfers.as_mut().unwrap();
        t.ready[bank].record(&t.copy[bank])?;
        stream.wait_event(&t.ready[bank])?;
        let images = locations
            .iter()
            .map(|&(gpu, slot)| {
                if gpu {
                    self.slots[slot].as_ref().unwrap().0.as_ptr()
                } else {
                    t.scratch[slot].as_ref().unwrap().as_ptr()
                }
            })
            .collect();
        t.predicted_layer = None;
        t.bank ^= 1;
        Ok(images)
    }
    fn maintain(&mut self) -> Result<()> {
        // Head D2H has retired all previous kernels and exact/predicted H2D.
        // Publish here only: layer graphs may still use old slot pointers earlier.
        self.poll_swaps()?;
        self.plan_swaps()?;
        if !self.transfers.as_ref().unwrap().asynchronous {
            // Complete and publish at this boundary before the next token's
            // CPU experts can read DRAM. The worker synchronizes each copy.
            self.wait_swaps()?;
        }
        Ok(())
    }
    /// Device decode, at a token boundary (GPU idle, doorbell jobs retired):
    /// publish the swaps planned during the previous token.
    pub fn accept_swaps(&mut self) -> Result<()> {
        self.poll_swaps()?;
        if !self.transfers.as_ref().unwrap().asynchronous {
            self.wait_swaps()?;
        }
        Ok(())
    }
    /// Records the last token's routes and submits the due swaps without
    /// waiting (device decode calls it while the next token runs: a swap only
    /// writes spare images, publication waits for `accept_swaps`).
    pub fn plan_swaps(&mut self) -> Result<()> {
        self.check_worker_error()?;
        self.cache
            .record_recent_token(self.transfers.as_ref().unwrap().window);
        let t = self.transfers.as_mut().unwrap();
        t.tokens += 1;
        if !maintenance_due(t.tokens, t.interval, t.swaps != 0) {
            return Ok(());
        }
        // Each spare owns one outstanding swap; publication stays at token
        // boundaries. The limit includes both H2D and D2H traffic.
        let (expert_bytes, slot_caps) = (&self.expert_bytes, &self.slot_caps);
        let plan = self.cache.plan_recent_fit(
            2 * self.image_bytes * t.spare.len(),
            t.hysteresis,
            |slot, expert| expert_bytes[expert] <= slot_caps[slot],
        );
        for swap in plan {
            let load = Self::index(swap.load.layer as usize, swap.load.expert as usize);
            let evict = Self::index(swap.evict.layer as usize, swap.evict.expert as usize);
            let source = self.host[load].clone();
            self.transfer_stats.maintenance_h2d += 1;
            if source.is_some() {
                self.transfer_stats.maintenance_d2h += 1;
            } else {
                self.transfer_stats.maintenance_disk += 1;
            }
            self.transfer_stats.maintenance_bytes += self.expert_bytes[load] as u64
                + if source.is_some() {
                    self.expert_bytes[evict] as u64
                } else {
                    0
                };
            // In place: the slot's own buffer goes to the worker (the GPU is
            // idle at the token boundary where swaps run and complete).
            let gpu = Arc::try_unwrap(self.slots[swap.slot].take().unwrap())
                .map_err(|_| Error::InvalidPlan("GLM swap: GPU image still in use".into()))?
                .0;
            let image = t.spare.pop().unwrap();
            t.maintenance.submit(SwapJob {
                replacement: swap,
                load,
                source,
                gpu,
                image,
                evict_bytes: self.expert_bytes[evict],
                load_bytes: self.expert_bytes[load],
            })?;
            t.swaps += 1;
        }
        Ok(())
    }
    fn wait_swaps(&mut self) -> Result<()> {
        while self.transfers.as_ref().unwrap().swaps != 0 {
            let done = self
                .transfers
                .as_ref()
                .unwrap()
                .maintenance
                .done
                .recv()
                .map_err(|_| Error::Weights("GLM maintenance stopped".into()))?;
            self.accept_swap(done)?;
        }
        Ok(())
    }
    pub fn drain(&mut self) -> Result<()> {
        if self.transfers.is_none() {
            return Ok(());
        }
        let result = (|| {
            self.retire_reads(0)?;
            self.retire_reads(1)?;
            self.wait_swaps()
        })();
        let mut sync = Ok(());
        for stream in &self.transfers.as_ref().unwrap().copy {
            sync = stream.synchronize().and(sync);
        }
        sync.and(result)
    }
    pub fn finish_token(&mut self, _stream: &Stream) -> Result<()> {
        // The caller's head read has retired all layer work before reserving slots.
        self.maintain()
    }
}

#[cfg(feature = "cuda")]
impl Drop for Experts {
    fn drop(&mut self) {
        let _ = self.drain();
    }
}

pub(super) fn read_image(file: &HytFile, entries: &[usize], dst: &mut [u8]) -> Result<()> {
    let lowbit = entries.iter().any(|&i| {
        matches!(
            file.entries()[i].quant,
            QuantSpec::Lowbit { .. } | QuantSpec::Exl3 { .. }
        )
    });
    let nvfp4 = file.entries()[entries[0]].quant == QuantSpec::Nvfp4;
    let expected: usize = entries
        .iter()
        .map(|&i| (file.entries()[i].bytes as usize).next_multiple_of(IMAGE_ALIGN))
        .sum();
    if dst.len() < expected {
        return Err(Error::Weights("GLM expert image length".into()));
    }
    if lowbit {
        // Matrices sit at 256-byte strides; the kernels derive offsets from the format.
        if !(dst.as_ptr() as usize).is_multiple_of(IMAGE_ALIGN) {
            return Err(Error::Weights(
                "GLM expert image alignment (256 bytes)".into(),
            ));
        }
    } else {
        let down_fp8 = nvfp4 && file.entries()[entries[2]].quant != QuantSpec::Nvfp4;
        validate_image_alignment(dst.as_ptr() as usize, nvfp4, down_fp8)?;
    }
    let mut offset = 0;
    for &entry in entries {
        let e = &file.entries()[entry];
        let end = offset + e.bytes as usize;
        file.read_tensor(entry, &mut dst[offset..end])?;
        if e.quant == QuantSpec::Nvfp4 {
            let dims = e.shape.dims();
            super::validate_nvfp4(&dst[offset..end], dims[0] as usize, dims[1] as usize)?;
        } else if let QuantSpec::Lowbit { grid, group } = e.quant {
            let dims = e.shape.dims();
            super::validate_lowbit(&dst[offset..end], (dims[0] * dims[1]) as usize, grid, group)?;
        } else if let QuantSpec::Exl3 { .. } = e.quant {
            let dims = e.shape.dims();
            super::validate_exl3(&dst[offset..end], dims[0] as usize, dims[1] as usize)?;
        } else {
            super::validate_raw(&dst[offset..end], e.quant)?;
        }
        offset = end.next_multiple_of(IMAGE_ALIGN);
        dst[end..offset].fill(0);
    }
    dst[offset..].fill(0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use crate::offload::{Budget, place_experts};
    use crate::offload::{DiskPolicy, ExpertStat, Tier};

    #[test]
    fn pack_moves_empties_the_sparsest_split_blocks() {
        let b = |bytes, block| BufferUse { bytes, block };
        // Size 100: block 1 keeps 3 (one free), block 2 keeps 2 (one free),
        // block 3 keeps 1 (two free). Only block 3 can be emptied: its image
        // goes to block 1's free buffer. Size 200 and own allocations stay.
        // (Block 3 keeps 1 of 3, block 2 keeps 2 of 3: block 3 is the source.)
        let kept = [
            (b(100, Some(1)), true),
            (b(100, Some(1)), true),
            (b(100, Some(1)), true),
            (b(100, Some(2)), true),
            (b(100, Some(2)), true),
            (b(100, Some(3)), true),
            (b(200, Some(9)), true),
            (b(100, None), true),
        ];
        let free = [
            b(100, Some(3)),
            b(100, Some(1)),
            b(100, Some(2)),
            b(100, Some(3)),
            b(200, Some(8)),
        ];
        assert_eq!(pack_moves(&kept, &free), vec![(5, 1)]);
        // Block 1 (1 kept of 4) is the biggest: blocks 2 and 3 empty into it,
        // although block 2 keeps more images.
        let free = [
            b(100, Some(1)),
            b(100, Some(1)),
            b(100, Some(1)),
            b(100, Some(2)),
            b(100, Some(3)),
        ];
        let kept = [
            (b(100, Some(1)), true),
            (b(100, Some(2)), true),
            (b(100, Some(2)), true),
            (b(100, Some(3)), true),
        ];
        let moves = pack_moves(&kept, &free);
        assert_eq!(moves.len(), 3);
        assert!(moves.iter().all(|&(k, f)| k != 0 && f < 3), "{moves:?}");
        // A block with an image in use is never a source.
        let kept = [
            (b(100, Some(1)), true),
            (b(100, Some(1)), true),
            (b(100, Some(2)), false),
        ];
        assert!(pack_moves(&kept, &[b(100, Some(1))]).is_empty());
    }

    #[test]
    fn fixed_placement_survives_free_vram_jitter_and_refuses_to_shrink() -> Result<()> {
        let resident = 20 << 30;
        let planned = hayate_core::machine::MAIN.planned_vram();
        let gpu = gpu_budget(resident, planned as usize)?;
        let image = (3 * matrix_stride(true)) as u64;
        let stats: Vec<_> = (0..42 * 288)
            .map(|i| ExpertStat {
                layer: (i / 288 + 3) as u16,
                expert: (i % 288) as u16,
                // Include equal frequencies, as well as a calibrated order.
                freq: (i % 19) as f32 / 19.,
                gpu_bytes: image,
                host_bytes: image,
            })
            .collect();
        let expected = expert_placement(&stats, gpu, u64::MAX, DiskPolicy::Refuse, false)?;
        for extra in [0, 1, image - 1, image, 512 << 20, 2 << 30] {
            let actual = expert_placement(
                &stats,
                gpu_budget(resident, (gpu + extra) as usize)?,
                u64::MAX,
                DiskPolicy::Refuse,
                false,
            )?;
            assert_eq!(actual, expected);
        }
        assert_eq!(expected.gpu_used / image, gpu / image);
        assert!(gpu_budget(resident, (gpu - 1) as usize).is_err());
        assert!(gpu_budget(planned as usize, planned as usize).is_err());
        Ok(())
    }

    #[test]
    fn mtp_cpu_placement_reserves_ram_and_returns_all_gpu_slots_to_main() -> Result<()> {
        use crate::expert_cache::{ExpertCache, ExpertId};
        let stats: Vec<_> = [(3, 0.8), (3, 0.2), (45, 1.0), (45, 0.9)]
            .into_iter()
            .enumerate()
            .map(|(i, (layer, freq))| ExpertStat {
                layer,
                expert: (i % 2) as u16,
                freq,
                gpu_bytes: 10,
                host_bytes: 12,
            })
            .collect();
        let gpu = expert_placement(&stats, 20, 24, DiskPolicy::Refuse, false)?;
        assert_eq!(
            gpu.tiers,
            [Tier::HostCompute, Tier::HostCompute, Tier::Gpu, Tier::Gpu]
        );
        let cpu = expert_placement(&stats, 20, 24, DiskPolicy::Refuse, true)?;
        assert_eq!(
            cpu.tiers,
            [Tier::Gpu, Tier::Gpu, Tier::HostCompute, Tier::HostCompute]
        );
        assert_eq!((cpu.gpu_used, cpu.host_used), (20, 24));
        assert!((cpu.host_bytes_per_token - 1.9 * 12.).abs() < 1e-6);
        // Even explicit disk permission cannot spill CPU-only MTP images.
        assert!(expert_placement(&stats, 20, 23, DiskPolicy::Allow, true).is_err());
        let mut cache = ExpertCache::with_disk(&stats, &cpu, 0.9, 1)?;
        for _ in 0..128 {
            // CPU-only MTP is deliberately absent from GPU promotion statistics.
            cache.route(3, &[0, 1])?;
            cache.record_recent_token(64);
            assert!(cache.plan_recent(20, 0).is_empty());
            for expert in 0..2 {
                assert!(cache.slot(ExpertId { layer: 45, expert }).is_none());
            }
        }
        Ok(())
    }

    #[test]
    fn vram_budget_keeps_resident_headroom_and_whole_images() -> Result<()> {
        let image = 14_156_544;
        let gib = 1u64 << 30;
        let gpu = budget_experts(60 * gib, 12 << 30, image, 1, usize::MAX)?;
        assert_eq!(gpu % image as u64, 0);
        assert!(gpu <= 60 * gib - (12 << 30) - (2 << 30) - 18 * image as u64);
        assert!(gpu + image as u64 > 60 * gib - (12 << 30) - (2 << 30) - 18 * image as u64);
        assert!(budget_experts(10 * gib, 12 << 30, image, 1, usize::MAX).is_err());
        assert!(budget_experts(60 * gib, 12 << 30, image, 1, 1 << 30).is_err());
        Ok(())
    }
    #[test]
    fn host_budget_leaves_os_and_staging_room() -> Result<()> {
        let limit = host_limit("MemTotal: 131072000 kB\nMemFree: 1 kB\n")?;
        assert!(88_000_000_000 < limit);
        assert!(110_000_000_000 > limit);
        assert_eq!(host_limit("MemTotal: 1 kB")?, 0);
        assert!(host_limit("MemFree: 1 kB").is_err());
        assert!(host_limit("MemTotal: 18446744073709551615 kB").is_err());
        Ok(())
    }

    #[test]
    fn automatic_host_budget_fits_mtp_and_safe_limit() -> Result<()> {
        let limit = 109_054_394_368;
        for required in [0, 101_389_168_128, limit] {
            assert_eq!(host_budget(None, required, limit, false)?, required);
        }
        assert!(host_budget(None, limit + 1, limit, false).is_err());
        assert_eq!(host_budget(None, limit + 1, limit, true)?, limit);
        Ok(())
    }

    #[test]
    fn explicit_host_budget_is_preserved() -> Result<()> {
        let required = 101_389_168_128;
        let limit = 109_054_394_368;
        assert!(host_budget(Some(100_000_000_000), required, limit, false).is_err());
        assert_eq!(host_budget(Some(0), required, limit, true)?, 0);
        assert_eq!(host_budget(Some(limit), required, limit, false)?, limit);
        assert!(host_budget(Some(limit + 1), required, limit, true).is_err());
        Ok(())
    }

    #[test]
    fn exact_routes_keep_prediction_hits_when_earlier_predictions_miss() {
        let ids = [70, 4, 3, 9, 2, 1, 8, 7];
        let resident = [None, None, None, Some(19), None, None, None, None];
        let predicted = [1, 2, 3, 4, 100, 101, 102, 103];
        let slots = [8, 9, 10, 11, 12, 13, 14, 15];
        let got = batch_locations(&ids, &resident, &predicted, &slots, 1);
        assert_eq!(
            got,
            [
                (false, 12),
                (false, 11),
                (false, 10),
                (true, 19),
                (false, 9),
                (false, 8),
                (false, 13),
                (false, 14)
            ]
        );
        // Predictions from another layer are discarded, preserving route order.
        assert_eq!(
            batch_locations(&ids, &resident, &[], &[], 0),
            [
                (false, 0),
                (false, 1),
                (false, 2),
                (true, 19),
                (false, 3),
                (false, 4),
                (false, 5),
                (false, 6)
            ]
        );
    }
    #[test]
    fn maintenance_runs_at_boundaries_and_can_be_frozen() {
        assert_eq!(
            (1..=127)
                .filter(|&n| maintenance_due(n, 32, false))
                .collect::<Vec<_>>(),
            [32, 64, 96]
        );
        assert!(!maintenance_due(32, 32, true));
        assert!(!maintenance_due(32, 0, false));
        assert!(maintenance_due(33, 1, false));
    }

    fn nvfp4_file() -> Result<HytFile> {
        expert_file(false, None)
    }

    fn expert_file(down_fp8: bool, rotation: Option<f32>) -> Result<HytFile> {
        use hayate_core::{
            MachineId, Shape,
            hyt::{FORMAT_VERSION, Header, TensorEntry, encode_table},
        };
        let bytes = QuantSpec::Nvfp4.matrix_bytes(2048, 4096).unwrap() as usize;
        let disk_stride = bytes.next_multiple_of(4096);
        let mut entries: Vec<_> = ["gate", "up", "down"]
            .into_iter()
            .enumerate()
            .map(|(part, name)| TensorEntry {
                name: format!("mlp.experts.0.{name}_proj.weight"),
                quant: QuantSpec::Nvfp4,
                shape: Shape::new(if part == 2 {
                    &[4096, 2048]
                } else {
                    &[2048, 4096]
                })
                .unwrap(),
                offset: (part * disk_stride) as u64,
                bytes: bytes as u64,
            })
            .collect();
        if down_fp8 {
            entries[2].quant = QuantSpec::Plain(hayate_core::DType::F8E4M3);
            entries[2].bytes = (2048 * 4096) as u64;
            entries.push(TensorEntry {
                name: "mlp.experts.0.down_proj.weight_scale_inv".into(),
                quant: QuantSpec::Plain(hayate_core::DType::F32),
                shape: Shape::new(&[32, 16])?,
                offset: entries[2].offset + entries[2].bytes,
                bytes: 2048,
            });
        }
        if rotation.is_some() {
            let last = entries.last().unwrap();
            entries.push(TensorEntry {
                name: "mlp.experts.hadamard_block".into(),
                quant: QuantSpec::Plain(hayate_core::DType::F32),
                shape: Shape::new(&[1])?,
                offset: (last.offset + last.bytes).next_multiple_of(4096),
                bytes: 4,
            });
        }
        let mut raw = Header {
            format_version: FORMAT_VERSION,
            machine: 0,
            tensor_count: entries.len() as u32,
            table_offset: 64,
            data_offset: 4096,
            source_hash: [0; 16],
        }
        .encode()
        .to_vec();
        raw.extend(encode_table(&entries)?);
        let last = entries.last().unwrap();
        raw.resize(4096 + (last.offset + last.bytes) as usize, 0);
        for part in 0..3 {
            let start = 4096 + part * disk_stride;
            if part == 2 && down_fp8 {
                raw[start..start + 2048 * 4096].fill(0x38);
                let scale_start = 4096 + entries[3].offset as usize;
                for s in raw[scale_start..scale_start + 2048].as_chunks_mut::<4>().0 {
                    *s = 0.25f32.to_le_bytes();
                }
                continue;
            }
            raw[start..start + 2048 * 4096 / 2].fill([0x22, 0x44, 0x66][part]);
            raw[start + 2048 * 4096 / 2..start + bytes - 4].fill(0x38);
            raw[start + bytes - 4..start + bytes].copy_from_slice(&0.25f32.to_le_bytes());
        }
        if let Some(block) = rotation {
            let at = 4096 + entries.last().unwrap().offset as usize;
            raw[at..at + 4].copy_from_slice(&block.to_le_bytes());
        }
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "hayate-glm-alignment-{}-{}.hyt",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, raw)?;
        let file = HytFile::open(&path, MachineId::Main);
        std::fs::remove_file(path)?;
        file
    }

    #[test]
    fn rotated_mixed_down_image_keeps_fp8_and_block_scales() -> Result<()> {
        let file = expert_file(true, Some(64.))?;
        assert_eq!(expert_format(&file), (true, true));
        assert_eq!(hadamard_block(&file)?, 64);
        let image_bytes = 2 * matrix_stride(true) + matrix_stride(false);
        assert_eq!(image_bytes, 17_828_352);
        assert_eq!(
            (image_bytes - 3 * matrix_stride(true)) * 42 * 288,
            44_414_189_568
        );
        let mut raw = vec![0xff; image_bytes + IMAGE_ALIGN];
        let start = (IMAGE_ALIGN - raw.as_ptr() as usize % IMAGE_ALIGN) % IMAGE_ALIGN;
        let dst = &mut raw[start..start + image_bytes];
        read_image(&file, &[0, 1, 2, 3], dst)?;
        let down = 2 * matrix_stride(true);
        assert!(dst[down..down + 4096 * 2048].iter().all(|&b| b == 0x38));
        assert!(
            dst[down + 4096 * 2048..]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|s| f32::from_le_bytes(*s) == 0.25)
        );
        for block in [0., 48., f32::NAN] {
            assert!(hadamard_block(&expert_file(false, Some(block))?).is_err());
        }
        Ok(())
    }

    #[test]
    fn sensitivity_slots_zero_pad_smaller_nvfp4_images() -> Result<()> {
        let file = nvfp4_file()?;
        assert_eq!(expert_format(&file), (true, false));
        let bytes = IMAGE_BYTES;
        let mut raw = vec![0xff; bytes + IMAGE_ALIGN];
        let start = (IMAGE_ALIGN - raw.as_ptr() as usize % IMAGE_ALIGN) % IMAGE_ALIGN;
        let dst = &mut raw[start..start + bytes];
        read_image(&file, &[0, 1, 2], dst)?;
        assert!(dst[3 * matrix_stride(true)..].iter().all(|&b| b == 0));
        Ok(())
    }

    #[test]
    fn nvfp4_images_preserve_payload_and_align_every_matrix_and_slot() -> Result<()> {
        let bytes = QuantSpec::Nvfp4.matrix_bytes(2048, 4096).unwrap() as usize;
        assert_eq!(bytes, 4_718_596);
        assert_eq!(matrix_stride(true), 4_718_848);
        let image_bytes = 3 * matrix_stride(true);
        assert_eq!(image_bytes, 14_156_544);
        assert_eq!(3 * matrix_stride(false), IMAGE_BYTES);
        let file = nvfp4_file()?;
        assert_eq!(hadamard_block(&file)?, 0);
        for entry in file.entries() {
            let disk_weight = file.header().data_offset + entry.offset;
            assert!(disk_weight.is_multiple_of(4096));
            assert!((disk_weight + 2048 * 4096 / 2).is_multiple_of(256));
            assert!((disk_weight + bytes as u64 - 4).is_multiple_of(256));
        }
        // Simulate an arena and recycled pinned staging with dirty padding.
        let mut arena = vec![0xff; 3 * image_bytes + IMAGE_ALIGN];
        let start = (IMAGE_ALIGN - arena.as_ptr() as usize % IMAGE_ALIGN) % IMAGE_ALIGN;
        for slot in 0..3 {
            let dst = &mut arena[start + slot * image_bytes..start + (slot + 1) * image_bytes];
            read_image(&file, &[0, 1, 2], dst)?;
            validate_image_alignment(dst.as_ptr() as usize, true, false)?;
            for part in 0..3 {
                let offset = part * matrix_stride(true);
                let mut payload = vec![0; bytes];
                file.read_tensor(part, &mut payload)?;
                assert_eq!(&dst[offset..offset + bytes], payload);
                assert!(
                    dst[offset + bytes..offset + matrix_stride(true)]
                        .iter()
                        .all(|&b| b == 0)
                );
            }
            assert!(read_image(&file, &[0, 1, 2], &mut dst[..image_bytes - 1]).is_err());
        }
        // The old tight offsets violate GEMV's 16-byte vector loads even
        // when cudaMalloc aligned the start of each separate slot.
        assert_eq!(bytes % 16, 4);
        assert_eq!(2 * bytes % 16, 8);
        for bad in [4, 8, 12, 16, 128] {
            assert!(validate_image_alignment(256 + bad, true, false).is_err());
        }
        assert!(validate_image_alignment(256, false, false).is_ok());
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires supported CUDA GPU; checkpoint-free padded expert images"]
    fn glm53_nvfp4_padded_images_gemv() -> Result<()> {
        use hayate_kernels::{llm::ffi::hayate_llm_gemv, runtime::check};
        let file = nvfp4_file()?;
        let device = Device::select()?;
        let stream = device.stream()?;
        let stride = matrix_stride(true);
        let image_bytes = 3 * stride;
        let mut host = device.alloc_pinned(3 * image_bytes)?;
        for image in host.as_mut_slice().chunks_exact_mut(image_bytes) {
            read_image(&file, &[0, 1, 2], image)?;
        }
        let mut arena = device.alloc(3 * image_bytes)?;
        let mut x = device.alloc(4096 * 2)?;
        let mut out = device.alloc(4096 * 2)?;
        stream.copy_h2d(&mut arena, host.as_slice())?;
        stream.copy_h2d(&mut x, &0x3f80u16.to_le_bytes().repeat(4096))?;
        for slot in 0..3 {
            validate_image_alignment(arena.as_ptr() as usize + slot * image_bytes, true, false)?;
            for (part, (n, k, expected)) in [
                (2048, 4096, 1024f32),
                (2048, 4096, 2048.),
                (4096, 2048, 2048.),
            ]
            .into_iter()
            .enumerate()
            {
                let weight = unsafe {
                    arena
                        .as_ptr()
                        .cast::<u8>()
                        .add(slot * image_bytes + part * stride)
                        .cast()
                };
                check(
                    unsafe {
                        hayate_llm_gemv(
                            out.as_mut_ptr(),
                            x.as_ptr(),
                            weight,
                            n,
                            k,
                            1,
                            0,
                            std::ptr::null_mut(),
                            4,
                            1,
                            0,
                            stream.as_raw(),
                        )
                    },
                    "GLM padded expert GEMV test",
                )?;
                let mut raw = vec![0; out.bytes()];
                stream.copy_d2h(&mut raw, &out)?;
                for b in raw[..n as usize * 2].as_chunks::<2>().0 {
                    assert_eq!(
                        f32::from_bits((u16::from_le_bytes(*b) as u32) << 16),
                        expected
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires supported CUDA GPU; no checkpoint needed"]
    fn glm53_boundary_expert_transfers() -> Result<()> {
        expert_transfer_test(false)
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires supported CUDA GPU; no checkpoint needed"]
    fn glm53_context_placement_swaps_images() -> Result<()> {
        // 16 experts of layer 3, 4 GPU slots (experts 0-3 by frequency), the
        // rest in RAM; image e is filled with 17 + e.
        let image_bytes = 256;
        let device = Device::select()?;
        let stream = device.stream()?;
        let stats: Vec<_> = (0..16)
            .map(|expert| ExpertStat {
                layer: 3,
                expert,
                freq: 1.0 / (expert + 1) as f32,
                gpu_bytes: image_bytes as u64,
                host_bytes: image_bytes as u64,
            })
            .collect();
        let placement = place_experts(
            &stats,
            Budget {
                gpu: 4 * image_bytes as u64,
                host: 12 * image_bytes as u64,
            },
            DiskPolicy::Refuse,
        )?;
        let cache = ExpertCache::with_disk(&stats, &placement, 0., 1)?;
        let mut slots = Vec::new();
        let mut host = Vec::new();
        for expert in 0..16u8 {
            let mut image = device.alloc_pinned(image_bytes)?;
            image.as_mut_slice().fill(17 + expert);
            match cache.slot(ExpertId {
                layer: 3,
                expert: expert.into(),
            }) {
                Some(slot) => {
                    let mut gpu = device.alloc(image_bytes)?;
                    stream.copy_h2d(&mut gpu, image.as_slice())?;
                    slots.push((slot, Some(Arc::new(GpuImage(gpu)))));
                    host.push(None);
                }
                None => host.push(Some(Arc::new(HostImage(image)))),
            }
        }
        slots.sort_by_key(|&(slot, _)| slot);
        let mut experts = Experts {
            files: Arc::new(Vec::new()),
            entries: Arc::new(Vec::new()),
            formats: vec![(false, false)],
            expert_formats: Vec::new(),
            expert_bytes: vec![image_bytes; 16],
            slot_caps: vec![image_bytes; 4],
            mixed: vec![false],
            cpu_formats: vec![None],
            image_bytes,
            counts: vec![0; 16],
            host_hits: 0,
            cache,
            slots: slots.into_iter().map(|(_, g)| g).collect(),
            host,
            staging: device.alloc_pinned(image_bytes)?,
            scratch: device.alloc(image_bytes)?,
            hits: 0,
            main_hits: 0,
            main_routes: 0,
            mtp_cpu: false,
            routes: 0,
            disk_reads: 0,
            d2h_count: 0,
            d2h_wait_ms: 0.,
            host_bytes: (12 * image_bytes) as u64,
            gpu_bytes: (4 * image_bytes) as u64,
            disk_bytes: 0,
            transfers: None,
            transfer_stats: TransferStats::default(),
            cpu: None,
            asynchronous: None,
            split: crate::glm53::cpu_experts::Split::new(1.)?,
            prior: stats.iter().map(|s| s.freq).collect(),
            host_limit: 0,
            allow_disk: false,
            trace: None,
            tail: Vec::new(),
            tail_tokens: 3,
            probe: None,
            probe_tokens: 0,
        };
        experts.transfers = Some(Transfers::new(
            &device,
            experts.files.clone(),
            Arc::new(Vec::new()),
            experts.entries.clone(),
            image_bytes,
            &super::super::Options::default(),
        )?);
        // An older token falls out of the 3-token tail; then 8-11 three times,
        // 12-15 twice and 0-3 once.
        experts.route(3, &[4, 5, 6, 7, 12, 13, 14, 15])?;
        experts.route(3, &[8, 9, 10, 11, 12, 13, 14, 15])?;
        experts.route(3, &[8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 0, 1, 2, 3])?;
        let check = |experts: &Experts| -> Result<Vec<usize>> {
            let mut gpu = Vec::new();
            for slot in 0..4 {
                let id = experts.cache.resident(slot).unwrap();
                let mut bytes = vec![0; image_bytes];
                stream.copy_d2h(&mut bytes, &experts.slots[slot].as_ref().unwrap().0)?;
                assert!(
                    bytes.iter().all(|&b| b == 17 + id.expert as u8),
                    "slot {slot}"
                );
                gpu.push(id.expert as usize);
            }
            for (expert, image) in experts.host.iter().enumerate() {
                assert_eq!(image.is_some(), !gpu.contains(&expert), "expert {expert}");
                if let Some(image) = image {
                    assert!(image.as_slice().iter().all(|&b| b == 17 + expert as u8));
                }
            }
            gpu.sort();
            Ok(gpu)
        };
        // The tail's score: 8-11 three times, 12-15 twice, 0-3 once.
        let mut score = vec![0.; 16];
        for e in 0..16 {
            score[e] = [1. / 3., 0., 1., 2. / 3.][e / 4];
        }
        // At most 3 pairs: the best entering (8, 9, 10) for the lowest of 0-3
        // (3, 2, 1: ties by frequency).
        let placed = experts.place_scores(&device, &score, 3)?;
        assert_eq!((placed.swaps, placed.bytes), (3, 3 * image_bytes as u64));
        assert_eq!(check(&experts)?, [0, 8, 9, 10]);
        assert_eq!(experts.place_scores(&device, &score, 8)?.swaps, 1);
        assert_eq!(check(&experts)?, [8, 9, 10, 11]);
        assert_eq!(experts.place_scores(&device, &score, 8)?.swaps, 0);
        // Probe over 2 tokens: no swaps at the context placement; the decoded
        // tokens use 4-7 and 12-15, which the context's top set (8-11) serves
        // as badly as the current set, so ties go by the context score: 12-15.
        experts.probe_tokens = 2;
        experts.place_for_context(1., 8);
        assert_eq!(check(&experts)?, [8, 9, 10, 11]);
        experts.route(3, &[4, 5, 6, 7, 12, 13, 14, 15])?;
        assert_eq!(experts.probe_token(&device)?.swaps, 0);
        experts.route(3, &[4, 5, 6, 7, 12, 13, 14, 15])?;
        assert_eq!(experts.probe_token(&device)?.swaps, 4);
        assert_eq!(check(&experts)?, [12, 13, 14, 15]);
        assert_eq!(experts.probe_token(&device)?.swaps, 0);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires supported CUDA GPU; no checkpoint needed"]
    fn glm53_context_placement_variable_slots() -> Result<()> {
        // 8 experts of layer 3: even ones 256 bytes, odd ones 128; slots hold
        // experts 0-3 with exactly their sizes; image e is filled with 17 + e.
        let image_bytes = 256;
        let device = Device::select()?;
        let stream = device.stream()?;
        let bytes_of = |e: usize| if e.is_multiple_of(2) { 256 } else { 128 };
        let stats: Vec<_> = (0..8)
            .map(|expert| ExpertStat {
                layer: 3,
                expert,
                freq: 1.0 / (expert + 1) as f32,
                gpu_bytes: bytes_of(expert as usize) as u64,
                host_bytes: image_bytes as u64,
            })
            .collect();
        let placement = place_experts(
            &stats,
            Budget {
                gpu: (256 + 128 + 256 + 128) as u64,
                host: (4 * image_bytes) as u64,
            },
            DiskPolicy::Refuse,
        )?;
        let cache = ExpertCache::with_disk_sizes(&stats, &placement, 0., 1)?;
        let mut slots = Vec::new();
        let mut host = Vec::new();
        let mut slot_caps = Vec::new();
        for expert in 0..8usize {
            let mut image = device.alloc_pinned(image_bytes)?;
            image.as_mut_slice().fill(17 + expert as u8);
            match cache.slot(ExpertId {
                layer: 3,
                expert: expert as u16,
            }) {
                Some(slot) => {
                    let mut gpu = device.alloc(bytes_of(expert))?;
                    stream.copy_h2d(&mut gpu, &image.as_slice()[..bytes_of(expert)])?;
                    slot_caps.push((slot, bytes_of(expert)));
                    slots.push((slot, Some(Arc::new(GpuImage(gpu)))));
                    host.push(None);
                }
                None => host.push(Some(Arc::new(HostImage(image)))),
            }
        }
        slots.sort_by_key(|&(slot, _)| slot);
        slot_caps.sort();
        let mut experts = Experts {
            files: Arc::new(Vec::new()),
            entries: Arc::new(Vec::new()),
            formats: vec![(false, false)],
            expert_formats: Vec::new(),
            expert_bytes: (0..8).map(bytes_of).collect(),
            slot_caps: slot_caps.iter().map(|&(_, b)| b).collect(),
            mixed: vec![false],
            cpu_formats: vec![None],
            image_bytes,
            counts: vec![0; 8],
            host_hits: 0,
            cache,
            slots: slots.into_iter().map(|(_, g)| g).collect(),
            host,
            staging: device.alloc_pinned(image_bytes)?,
            scratch: device.alloc(image_bytes)?,
            hits: 0,
            main_hits: 0,
            main_routes: 0,
            mtp_cpu: false,
            routes: 0,
            disk_reads: 0,
            d2h_count: 0,
            d2h_wait_ms: 0.,
            host_bytes: (4 * image_bytes) as u64,
            gpu_bytes: 768,
            disk_bytes: 0,
            transfers: None,
            transfer_stats: TransferStats::default(),
            cpu: None,
            asynchronous: None,
            split: crate::glm53::cpu_experts::Split::new(1.)?,
            prior: stats.iter().map(|s| s.freq).collect(),
            host_limit: 0,
            allow_disk: false,
            trace: None,
            tail: Vec::new(),
            tail_tokens: 3,
            probe: None,
            probe_tokens: 0,
        };
        experts.transfers = Some(Transfers::new(
            &device,
            experts.files.clone(),
            Arc::new(Vec::new()),
            experts.entries.clone(),
            image_bytes,
            &super::super::Options::default(),
        )?);
        // Every slot holds its expert's bytes and every other expert has a RAM image.
        let check = |experts: &Experts| -> Result<Vec<usize>> {
            let mut gpu = Vec::new();
            for slot in 0..4 {
                let id = experts.cache.resident(slot).unwrap();
                let e = id.expert as usize;
                let buffer = &experts.slots[slot].as_ref().unwrap().0;
                assert_eq!(buffer.bytes(), experts.slot_caps[slot], "slot {slot} size");
                assert!(buffer.bytes() >= bytes_of(e), "expert {e} in slot {slot}");
                let mut bytes = vec![0; bytes_of(e)];
                stream.copy_d2h(&mut bytes, buffer)?;
                assert!(
                    bytes.iter().all(|&b| b == 17 + e as u8),
                    "slot {slot} holds {e}"
                );
                gpu.push(e);
            }
            for (e, image) in experts.host.iter().enumerate() {
                assert_eq!(image.is_some(), !gpu.contains(&e), "expert {e}");
                if let Some(image) = image {
                    assert!(
                        image.as_slice()[..bytes_of(e)]
                            .iter()
                            .all(|&b| b == 17 + e as u8)
                    );
                }
            }
            gpu.sort();
            Ok(gpu)
        };
        assert_eq!(check(&experts)?, [0, 1, 2, 3]);
        // Experts 4-7 outrank the residents. Each takes the smallest slot it
        // fits: 4 (256) and 6 (256) the 256-byte slots, 5 and 7 (128) the others.
        let mut score = vec![0.; 8];
        for (e, v) in [
            (0, 0.4),
            (1, 0.3),
            (2, 0.2),
            (3, 0.1),
            (4, 4.),
            (5, 3.),
            (6, 2.),
            (7, 1.),
        ] {
            score[e] = v;
        }
        let placed = experts.place_scores(&device, &score, 8)?;
        assert_eq!(placed.swaps, 4);
        assert_eq!(check(&experts)?, [4, 5, 6, 7]);
        for slot in 0..4 {
            let e = experts.cache.resident(slot).unwrap().expert as usize;
            assert_eq!(experts.slot_caps[slot], bytes_of(e), "best fit: expert {e}");
        }
        // Expert 0 (256 bytes) now beats the others: it must take a 256 slot
        // (of 4 or 6, the weaker, 6), never a 128 one, even though the weakest
        // resident (7) has a smaller slot.
        let mut score = vec![0.; 8];
        for (e, v) in [(0, 10.), (4, 5.), (5, 4.), (6, 3.), (7, 1.)] {
            score[e] = v;
        }
        let placed = experts.place_scores(&device, &score, 8)?;
        assert_eq!(placed.swaps, 1);
        assert_eq!(check(&experts)?, [0, 4, 5, 7]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn expert_transfer_test(asynchronous: bool) -> Result<()> {
        let image_bytes = 256;
        let device = Device::select()?;
        let stream = device.stream()?;
        let stats: Vec<_> = (0..4)
            .map(|expert| ExpertStat {
                layer: 3,
                expert,
                freq: 1.0 / (expert + 1) as f32,
                gpu_bytes: image_bytes as u64,
                host_bytes: image_bytes as u64,
            })
            .collect();
        let placement = place_experts(
            &stats,
            Budget {
                gpu: image_bytes as u64,
                host: (3 * image_bytes) as u64,
            },
            DiskPolicy::Refuse,
        )?;
        let cache = ExpertCache::with_disk(&stats, &placement, 0., 1)?;
        let mut slot = device.alloc(image_bytes)?;
        let mut staging = device.alloc_pinned(image_bytes)?;
        staging.as_mut_slice().fill(17);
        stream.copy_h2d(&mut slot, staging.as_slice())?;
        let mut host = vec![None];
        for expert in 1..4 {
            let mut image = device.alloc_pinned(image_bytes)?;
            image.as_mut_slice().fill(17 + expert);
            host.push(Some(Arc::new(HostImage(image))));
        }
        let mut experts = Experts {
            files: Arc::new(Vec::new()),
            entries: Arc::new(Vec::new()),
            formats: vec![(false, false)],
            expert_formats: Vec::new(),
            expert_bytes: vec![image_bytes; 4],
            slot_caps: vec![image_bytes],
            mixed: vec![false],
            cpu_formats: vec![None],
            image_bytes,
            counts: vec![0; 4],
            host_hits: 0,
            cache,
            slots: vec![Some(Arc::new(GpuImage(slot)))],
            host,
            staging,
            scratch: device.alloc(image_bytes)?,
            hits: 0,
            main_hits: 0,
            main_routes: 0,
            mtp_cpu: false,
            routes: 0,
            disk_reads: 0,
            d2h_count: 0,
            d2h_wait_ms: 0.,
            host_bytes: (3 * image_bytes) as u64,
            gpu_bytes: (2 * image_bytes) as u64,
            disk_bytes: 0,
            transfers: None,
            transfer_stats: TransferStats::default(),
            cpu: None,
            asynchronous: None,
            split: crate::glm53::cpu_experts::Split::new(1.)?,
            prior: Vec::new(),
            host_limit: 0,
            allow_disk: false,
            trace: None,
            tail: Vec::new(),
            tail_tokens: 0,
            probe: None,
            probe_tokens: 0,
        };
        experts.transfers = Some(Transfers::new(
            &device,
            experts.files.clone(),
            Arc::new(Vec::new()),
            experts.entries.clone(),
            image_bytes,
            &super::super::Options {
                maintenance_swaps: 1,
                maintenance_hysteresis: 0,
                maintenance_async: asynchronous,
                ..Default::default()
            },
        )?);
        experts.route(3, &[3])?;
        experts.finish_token(&stream)?;
        let mut old = vec![0; image_bytes];
        if asynchronous {
            // An in-flight promotion leaves both old tiers usable without a wait.
            assert_eq!(
                experts.cache.slot(ExpertId {
                    layer: 3,
                    expert: 3
                }),
                None
            );
            assert!(experts.host[3].is_some());
            stream.copy_d2h(&mut old, &experts.slots[0].as_ref().unwrap().0)?;
            assert!(old.iter().all(|&b| b == 17));
            experts.image(3, 3, &stream)?;
            stream.copy_d2h(&mut old, &experts.scratch)?;
            assert!(old.iter().all(|&b| b == 20));
        } else {
            // All copies and publication finish before the next CPU/GPU layer.
            assert_eq!(experts.transfers.as_ref().unwrap().swaps, 0);
            assert_eq!(experts.transfers.as_ref().unwrap().spare.len(), 1);
            assert_eq!(experts.d2h_count, 1);
        }
        experts.drain()?;
        assert_eq!(
            experts.cache.slot(ExpertId {
                layer: 3,
                expert: 3
            }),
            Some(0)
        );
        assert!(experts.host[3].is_none());
        assert!(
            experts.host[0]
                .as_ref()
                .unwrap()
                .as_slice()
                .iter()
                .all(|&b| b == 17)
        );
        assert_eq!(experts.host.iter().filter(|h| h.is_some()).count(), 3);
        let mut raw = vec![0; image_bytes];
        stream.copy_d2h(&mut raw, &experts.slots[0].as_ref().unwrap().0)?;
        assert!(raw.iter().all(|&b| b == 20));
        experts.image(3, 0, &stream)?;
        stream.copy_d2h(&mut raw, &experts.scratch)?;
        assert!(raw.iter().all(|&b| b == 17));
        for (prediction, actual) in [
            (&[0u16, 1, 2][..], &[3u16, 2, 0, 1][..]),
            (&[1][..], &[0, 3, 2, 1][..]),
        ] {
            experts.prefetch(3, prediction)?;
            experts.route(3, actual)?;
            let images = experts.decode_images(3, actual, &stream)?;
            for (&expert, image) in actual.iter().zip(images) {
                let mut out = device.alloc(image_bytes)?;
                hayate_kernels::runtime::check(
                    unsafe {
                        hayate_kernels::glm53::ffi::hayate_glm_copy(
                            out.as_mut_ptr(),
                            image,
                            image_bytes,
                            stream.as_raw(),
                        )
                    },
                    "glm transfer test image",
                )?;
                stream.copy_d2h(&mut raw, &out)?;
                assert!(raw.iter().all(|&b| b == 17 + expert as u8));
            }
        }
        assert_eq!(experts.transfer_stats.prefetch_hits, 4);
        assert_eq!(experts.transfer_stats.maintenance_d2h, 1);
        assert_eq!(
            experts.transfer_stats.maintenance_bytes,
            2 * image_bytes as u64
        );
        let original = experts.slots[0].as_ref().unwrap().0.as_ptr();
        let borrowed = experts.borrow_prefill(&device, &stream, 1, true)?;
        assert_eq!(borrowed.copied_bytes, image_bytes);
        let (mut loan, mut pool) = (borrowed.slots, borrowed.buffers);
        assert_eq!(pool[0].as_ptr(), original);
        assert!(experts.slots[0].is_none());
        assert!(
            experts.host[3]
                .as_ref()
                .unwrap()
                .as_slice()
                .iter()
                .all(|&b| b == 20)
        );
        let mut bank =
            super::super::prefill::Bank::new(&device, &stream, image_bytes, 4, &mut pool)?;
        assert_eq!(bank.arena[0].as_ptr(), original);
        let upload = device.stream()?;
        experts.prefill_images(3, &[0, 1, 2, 3], &mut bank, &upload, None)?;
        stream.wait_event(&bank.ready)?;
        for (expert, &image) in bank.images.iter().enumerate() {
            let mut result = device.alloc(image_bytes)?;
            hayate_kernels::runtime::check(
                unsafe {
                    hayate_kernels::glm53::ffi::hayate_glm_copy(
                        result.as_mut_ptr(),
                        image,
                        image_bytes,
                        stream.as_raw(),
                    )
                },
                "GLM borrowed prefill image",
            )?;
            let mut bytes = vec![0; image_bytes];
            stream.copy_d2h(&mut bytes, &result)?;
            assert!(bytes.iter().all(|&b| b == 17 + expert as u8));
        }
        let mut actual =
            super::super::prefill::Bank::new(&device, &stream, image_bytes, 2, &mut Vec::new())?;
        experts.prefill_images(3, &[3, 1], &mut actual, &upload, Some(&bank))?;
        stream.wait_event(&actual.ready)?;
        assert_eq!(actual.images, [bank.images[3], bank.images[1]]);
        assert_eq!(experts.transfer_stats.prefetch_hits, 6);
        actual.used.record(&stream)?;
        bank.used.record(&stream)?;
        stream.synchronize()?;
        upload.synchronize()?;
        drop(actual);
        drop(bank);
        drop(pool);
        experts.return_prefill(&device, &stream, &mut loan)?;
        assert!(loan.is_empty());
        assert!(experts.host[3].is_none());
        let mut restored = vec![0; image_bytes];
        stream.copy_d2h(&mut restored, &experts.slots[0].as_ref().unwrap().0)?;
        assert!(restored.iter().all(|&b| b == 20));
        let mut shadow = device.alloc_pinned(image_bytes)?;
        stream.copy_d2h(shadow.as_mut_slice(), &experts.slots[0].as_ref().unwrap().0)?;
        experts.host[3] = Some(Arc::new(HostImage(shadow)));
        let borrowed = experts.borrow_prefill(&device, &stream, 1, true)?;
        assert_eq!(borrowed.copied_bytes, 0);
        assert!(!borrowed.slots[0].2);
        let mut loan = borrowed.slots;
        drop(borrowed.buffers);
        experts.return_prefill(&device, &stream, &mut loan)?;
        assert!(experts.host[3].is_some(), "keep the original host image");
        assert_eq!(experts.disk_reads, 0);
        Ok(())
    }
}
