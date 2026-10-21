#[path = "device_step.rs"]
mod device_step;
#[path = "session.rs"]
mod session;
#[path = "speculation_gpu.rs"]
mod speculation_gpu;
pub use session::SessionState;

use super::{EXPERTS, HC, HIDDEN, Layer, experts::Experts, tokenizer::Tokenizer};
use hayate_core::{DType, Error, MachineId, QuantSpec, Result, hyt::HytFile};
use hayate_kernels::{
    glm53::ffi as g,
    runtime::{Device, DeviceBuffer, Event, Stream, check},
    text::ffi::hayate_text_linear,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

/// Default CPU share of the n RAM-tier misses of a layer (n = 0..8) on device decode's PCIe lane: max(k t_cpu,
/// (n - k) t_pcie) is least for t_cpu 0.40 ms (EXL3 K2.5 on 8 P-cores) and t_pcie 0.30 ms (fetch 0.28 +
/// compute), at most 4 on the lane.
pub(super) const PCIE_TAKE: [u8; 9] = [0, 0, 1, 1, 2, 2, 3, 3, 4];
/// The CPU share table as nibbles; each take[n] <= n and leaves at most 4 experts to the lane.
pub(super) fn pcie_take(take: [u8; 9]) -> Result<u64> {
    let mut packed = 0u64;
    for (n, &k) in take.iter().enumerate() {
        if usize::from(k) > n || n - usize::from(k) > 4 {
            return Err(Error::InvalidPlan(format!(
                "GLM pcie-take: take[{n}] = {k} must be <= {n} and leave at most 4 to the lane"
            )));
        }
        packed |= u64::from(k) << (4 * n);
    }
    Ok(packed)
}

#[derive(Debug)]
pub struct Options {
    pub layers: usize,
    pub decode_async: bool,
    /// Whole-token CUDA graph with device routing and CPU doorbells (stage 19).
    pub decode_device: bool,
    /// Device decode: CPU experts of MoE layer k with a routing weight below this
    /// join at layer k+1 (approximation; 0 off, infinity all).
    pub defer_cpu_below: f32,
    /// Device decode: drop CPU experts whose routing weight is below this
    /// (approximation; 0 keeps all).
    pub skip_cpu_below: f32,
    pub context: usize,
    pub host_bytes: Option<u64>,
    pub allow_disk: bool,
    pub frequencies: Option<Vec<f32>>,
    pub prefill_chunk: usize,
    pub prefill_legacy: bool,
    pub prefill_all_experts: bool,
    pub maintenance_interval: usize,
    pub maintenance_window: usize,
    pub maintenance_swaps: usize,
    pub maintenance_hysteresis: usize,
    pub maintenance_async: bool,
    pub expert_prefetch: bool,
    pub expert_readers: usize,
    pub pcie_frac: f64,
    /// Device decode's PCIe lane: CPU share of the n RAM-tier misses of a layer (None: every miss on the CPU).
    pub pcie_take: Option<[u8; 9]>,
    /// The lane's prefetch: per MoE layer, up to this many of the next layer's predicted RAM-tier misses are
    /// fetched on a side stream during this layer's experts and the next attention (0: none, at most 4).
    pub pcie_prefetch: u32,
    pub cpu_threads: usize,
    pub cpu_spin: bool,
    pub host_core: Option<usize>,
    pub mtp: bool,
    pub mtp_reserve_only: bool,
    pub mtp_cpu_experts: bool,
    /// Total GLM VRAM budget in bytes (resident weights, state, workspace,
    /// expert cache). `None` plans from the machine's VRAM with 8 GiB spare.
    pub vram_budget: Option<u64>,
    /// End of each prefill: move the GPU experts toward the routes of the
    /// last `placement_tail` context tokens (0 off), rank = beta x those
    /// routes per token + (1 - beta) x load-time frequency, at most
    /// `placement_swaps` pairs (stage 20).
    pub placement_tail: usize,
    pub placement_beta: f32,
    pub placement_swaps: usize,
    /// Decoded tokens after a context placement before the one re-placement
    /// from their routes (0 off).
    pub placement_probe: usize,
    /// LoRA adapters (llama.cpp GGUF), added as side paths to the converted weights.
    pub loras: Vec<super::lora::Spec>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            layers: 45,
            decode_async: false,
            decode_device: false,
            defer_cpu_below: 0.,
            skip_cpu_below: 0.,
            context: 4096,
            host_bytes: None,
            allow_disk: false,
            frequencies: None,
            prefill_chunk: 0,
            prefill_legacy: false,
            prefill_all_experts: false,
            maintenance_interval: 1,
            maintenance_window: 64,
            maintenance_swaps: 1,
            maintenance_hysteresis: 16,
            maintenance_async: false,
            expert_prefetch: false,
            expert_readers: super::transfer::DEFAULT_READERS,
            pcie_frac: 0.,
            pcie_take: Some(PCIE_TAKE),
            pcie_prefetch: 1,
            cpu_threads: 8,
            cpu_spin: false,
            host_core: Some(23),
            mtp: false,
            mtp_reserve_only: false,
            mtp_cpu_experts: false,
            vram_budget: None,
            placement_tail: 32768,
            placement_beta: 1.,
            placement_swaps: usize::MAX,
            placement_probe: 4,
            loras: Vec::new(),
        }
    }
}
/// Prompts of at least this many tokens borrow cache slots for a full-size
/// prefill chunk instead of shrinking the chunk to the free VRAM.
const BIG_PREFILL_TOKENS: usize = 6144;
/// Saved decode state for prefix reuse (`Executor::export_session`).
#[derive(Debug, Clone)]
pub struct Session {
    pub position: usize,
    pub layers: Vec<super::LayerSession>,
    pub mtp: Option<super::LayerSession>,
}
/// Result of `Executor::prefill_cancellable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillOutcome {
    /// Every token was consumed; the greedy next token.
    Done(u32),
    /// Stopped by the cancel flag. State is valid up to `position` (a chunk
    /// or token boundary); the caller may continue with `ids[position - start..]`.
    Cancelled { position: usize },
}
const CANCELLED: &str = "GLM prefill cancelled";
#[derive(Debug)]
pub struct MemoryUsage {
    pub weights: usize,
    pub state_and_workspace: usize,
    pub latent_kv_host: usize,
    pub experts_gpu: u64,
    pub experts_host: u64,
    pub experts_disk: u64,
    pub vram_peak: usize,
    pub mtp_added: usize,
    pub mtp_experts_gpu: u64,
}
#[derive(Debug)]
pub struct Executor<'d> {
    image_embeddings: BTreeMap<usize, Vec<u8>>,
    device: &'d Device,
    stream: Stream,
    layers: Vec<Layer<'d>>,
    mtp: Option<Layer<'d>>,
    mtp_reserve_only: bool,
    spec: BTreeMap<String, DeviceBuffer>,
    experts: Experts,
    decode_async: bool,
    async_prepared: bool,
    device_decode: Option<device_step::DeviceDecode>,
    /// Layer streams are idle; device tokens may run on `stream`.
    device_ready: bool,
    host_syncs: u64,
    decode_steps: u64,
    prefill: super::prefill::Prefill,
    globals: BTreeMap<String, DeviceBuffer>,
    input: DeviceBuffer,
    mean: DeviceBuffer,
    normalized: DeviceBuffer,
    logits: DeviceBuffer,
    next: DeviceBuffer,
    vocab: usize,
    context: usize,
    position: usize,
    source_hash: [u8; 16],
    ready: Event,
    reference: bool,
    expert_prefetch: bool,
    head_format: u32,
    profile: Option<super::profile::Profile>,
    pub memory: MemoryUsage,
    vram_budget: Option<u64>,
    swaps: usize,
    /// VRAM kept free inside an explicit budget for one 8K prefill chunk.
    prefill_reserve: u64,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    placement: (f32, usize),
    /// Context placements so far (swaps, bytes each way, seconds).
    pub placed: super::experts::Placed,
}
impl<'d> Executor<'d> {
    pub fn load(device: &'d Device, root: &Path, mut options: Options) -> Result<Self> {
        if device.machine != MachineId::Main
            || !(5..=45).contains(&options.layers)
            || !(1..=131072).contains(&options.context)
            || options.prefill_chunk > 8192
        {
            return Err(Error::InvalidPlan(
                "GLM needs main PC, 5..45 layers and context 1..131072".into(),
            ));
        }
        if options.mtp && (options.layers != 45 || ![0., 1.].contains(&options.pcie_frac)) {
            return Err(Error::InvalidPlan(
                "GLM MTP needs 45 layers and pcie-frac 0 or 1".into(),
            ));
        }
        if options.mtp_cpu_experts && (!options.mtp || options.pcie_frac != 0.) {
            return Err(Error::InvalidPlan(
                "GLM mtp-experts=cpu requires MTP allocation and pcie-frac 0".into(),
            ));
        }
        super::cpu_experts::Split::new(options.pcie_frac)?;
        if ((options.defer_cpu_below != 0. || options.skip_cpu_below != 0.)
            && !options.decode_device)
            || options.defer_cpu_below.is_nan()
            || options.defer_cpu_below < 0.
            || !(options.skip_cpu_below >= 0. && options.skip_cpu_below < 2.5)
            || (options.decode_device
                && (options.decode_async || options.mtp || options.mtp_reserve_only))
        {
            return Err(Error::InvalidPlan(
                "GLM device decode excludes --decode-host async and MTP; --defer-cpu-experts and --skip-cpu-below (0..2.5) need it".into(),
            ));
        }
        if (options.decode_async || options.decode_device)
            && (options.pcie_frac != 0. || options.expert_prefetch || options.cpu_spin)
        {
            return Err(Error::InvalidPlan("GLM async decode requires pcie-frac=0, no expert-prefetch, no cpu-spin (disk-tier misses are read and computed on the CPU); select --decode-host sync for transfer experiments".into()));
        }

        if !(1..=8).contains(&options.cpu_threads)
            || !(1..=128).contains(&options.maintenance_window)
            || !(1..=64).contains(&options.maintenance_swaps)
            || options.host_core.is_some_and(|core| core > 23)
        {
            return Err(Error::InvalidPlan(
                "GLM cpu-threads=1..8, host-core=none/0..23, maintenance-window=1..128, maintenance-swaps=1..64".into(),
            ));
        }
        if options.pcie_frac < 1. && options.expert_prefetch {
            return Err(Error::InvalidPlan(
                "GLM expert-prefetch requires pcie-frac 1".into(),
            ));
        }
        if let Some(core) = options.host_core {
            hayate_kernels::cpu::pin_thread(core as u16)?;
        }
        let (initial_free, total) = device.memory_info()?;
        let stream = device.stream()?;
        if !(1..=16).contains(&options.expert_readers) {
            return Err(Error::InvalidPlan(
                "GLM expert readers must be 1..=16".into(),
            ));
        }
        let resident_start = Instant::now();
        let file = HytFile::open(root.join("globals.hyt"), MachineId::Main)?;
        let source_hash = file.header().source_hash;
        let mut globals = BTreeMap::new();
        let mut vocab = 0;
        let mut head_format = 0;
        let mut indices = Vec::new();
        for name in ["embed_tokens.weight", "norm.weight", "lm_head.weight"] {
            let index = file
                .entries()
                .iter()
                .position(|e| e.name == name)
                .ok_or_else(|| Error::Weights(format!("missing GLM {name}")))?;
            let entry = &file.entries()[index];
            let dims = entry.shape.dims();
            if name == "embed_tokens.weight" {
                if dims.len() != 2 || dims[1] != HIDDEN as u64 {
                    return Err(Error::Weights("GLM embedding shape".into()));
                }
                vocab = dims[0] as usize;
            }
            let expected = if name == "norm.weight" {
                vec![HIDDEN as u64]
            } else {
                vec![vocab as u64, HIDDEN as u64]
            };
            let valid = entry.quant == QuantSpec::Plain(DType::Bf16)
                || (name == "lm_head.weight"
                    && matches!(
                        entry.quant,
                        QuantSpec::Nvfp4 | QuantSpec::PerTensor(DType::F8E4M3)
                    ));
            if name == "lm_head.weight" {
                head_format = match entry.quant {
                    QuantSpec::Nvfp4 => 3,
                    QuantSpec::PerTensor(DType::F8E4M3) => 2,
                    _ => 0,
                };
            }
            if dims != expected || !valid {
                return Err(Error::Weights(format!("GLM global shape/dtype: {name}")));
            }
            indices.push(index);
        }
        globals.extend(super::load::upload(device, &file, &indices)?);
        let mut layers = Vec::new();
        let mut files = Vec::new();
        for i in 0..options.layers {
            let path = root.join(format!("layer{i}/weights.hyt"));
            let file = HytFile::open(&path, MachineId::Main)?;
            if file.header().source_hash != source_hash {
                return Err(Error::Weights(
                    "GLM layer/global source hashes differ".into(),
                ));
            }
            layers.push(Layer::load_decode(device, &path, i, options.context)?);
            files.push(file);
        }
        let lora_experts = if options.loras.is_empty() {
            None
        } else {
            if options.mtp || options.mtp_reserve_only {
                return Err(Error::InvalidPlan("GLM LoRA does not support MTP".into()));
            }
            let adapters = super::lora::load(&options.loras, options.layers)?;
            for (layer, adapters) in layers.iter_mut().zip(&adapters) {
                layer.set_lora(adapters)?;
            }
            eprintln!(
                "GLM LoRA: layers={} vram_bytes={}",
                layers.iter().filter(|l| l.has_lora()).count(),
                layers.iter().map(Layer::lora_bytes).sum::<usize>()
            );
            Some(std::sync::Arc::new(super::lora::HostExperts::new(
                &adapters,
            )))
        };
        let mut spec = BTreeMap::new();
        let mut mtp = None;
        let mut mtp_added = 0;
        if options.mtp {
            let path = root.join("layer45/weights.hyt");
            let file = HytFile::open(&path, MachineId::Main)?;
            if file.header().source_hash != source_hash {
                return Err(Error::Weights("GLM MTP source hash differs".into()));
            }
            let mut layer = Layer::load_decode(device, &path, 45, options.context)?;
            layer.graphs = false;
            layer.prepare_verification()?;
            mtp_added += layer.device_bytes();
            mtp = Some(layer);
            files.push(file);
            if let Some(f) = &mut options.frequencies
                && f.len() == 42 * 288
            {
                f.extend(std::iter::repeat_n(8. / 288., 288));
            }
            for layer in &mut layers {
                let before = layer.device_bytes();
                layer.prepare_verification()?;
                mtp_added += layer.device_bytes() - before;
            }
            for (name, bytes) in [
                ("target", HIDDEN * 2),
                ("mtp_hidden", HIDDEN * 2),
                ("hidden", 4 * HIDDEN * 2),
                ("input", 4 * HC * HIDDEN * 2),
                ("logits", 4 * vocab * 2),
                ("predictions", 4 * 4),
                ("drafts", 3 * 4),
                ("result", 5 * 4),
            ] {
                spec.insert(name.into(), device.alloc(bytes)?);
                mtp_added += bytes;
            }
        }
        let input = device.alloc(HC * HIDDEN * 2)?;
        let mean = device.alloc(HIDDEN * 2)?;
        let normalized = device.alloc(HIDDEN * 2)?;
        let logits = device.alloc(vocab * 2)?;
        let next = device.alloc(4)?;
        let weights = globals.values().map(DeviceBuffer::bytes).sum::<usize>()
            + layers.iter().map(Layer::weight_bytes).sum::<usize>()
            + mtp.as_ref().map_or(0, Layer::weight_bytes);
        let mut state_and_workspace = layers
            .iter()
            .map(|l| l.device_bytes() - l.weight_bytes())
            .sum::<usize>()
            + input.bytes()
            + mean.bytes()
            + normalized.bytes()
            + logits.bytes()
            + next.bytes()
            + spec.values().map(DeviceBuffer::bytes).sum::<usize>()
            + mtp
                .as_ref()
                .map_or(0, |l| l.device_bytes() - l.weight_bytes());
        let prefill_budget = (hayate_core::machine::MAIN.planned_vram() as usize)
            .saturating_sub(weights + state_and_workspace);
        let requested = if options.prefill_chunk == 0 && (options.mtp || options.prefill_legacy) {
            128
        } else {
            options.prefill_chunk
        };
        let image_bytes = files
            .iter()
            .skip(3)
            .map(super::experts::layer_image_bytes)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap();
        let prefill_capacity =
            super::prefill::capacity(requested, options.context, prefill_budget / 4, image_bytes)?;
        let mut prefill = super::prefill::Prefill::new(
            device,
            prefill_capacity,
            options.context,
            layers.iter().any(|l| l.hadamard_block != 0),
            image_bytes,
            !options.prefill_legacy && !options.mtp,
        )?;
        prefill.all_experts = options.prefill_all_experts;
        if layers.iter().any(|l| l.lowbit) {
            if !prefill.batched {
                return Err(Error::InvalidPlan(
                    "GLM LOWBIT experts need the batched prefill".into(),
                ));
            }
            prefill.lowbit_scratch = Some(device.alloc(3 * super::experts::matrix_stride(true))?);
        }
        let resident_seconds = resident_start.elapsed().as_secs_f64();
        state_and_workspace += prefill.bytes();
        // The device decode's PCIe lane: scratch slots of the largest image, before the expert budget.
        let pcie = match options.pcie_take {
            Some(take) if options.decode_device => {
                let pcie = device_step::Pcie::new(
                    device,
                    options.layers,
                    image_bytes,
                    take,
                    options.pcie_prefetch,
                )?;
                state_and_workspace += pcie.bytes();
                Some(pcie)
            }
            _ => None,
        };
        let (free, _) = device.memory_info()?;
        let prefill_reserve = if prefill.batched {
            super::prefill::reserve_bytes(
                options.context,
                image_bytes,
                layers.iter().any(|l| l.hadamard_block != 0),
            )
        } else {
            0
        };
        let gpu = match options.vram_budget {
            Some(total) => super::experts::budget_experts(
                total,
                weights + state_and_workspace + prefill_reserve as usize,
                image_bytes,
                options.maintenance_swaps,
                free,
            )?,
            None => super::experts::gpu_budget(weights + state_and_workspace, free)?,
        };
        eprintln!(
            "GLM fixed memory plan: resident_bytes={} prefill_capacity={prefill_capacity} prefill_reserve={prefill_reserve} expert_budget={gpu} free_bytes={free}",
            weights + state_and_workspace,
        );
        let expert_start = Instant::now();
        if !options.mtp
            && let Some(f) = &mut options.frequencies
            && f.len() == 43 * 288
            && options.layers == 45
        {
            f.truncate(42 * 288);
        }
        let mut experts = Experts::load(device, files, gpu, root, &options)?;
        if let Some(lora) = lora_experts {
            experts.set_lora(lora);
        }
        if options.decode_async || options.decode_device {
            experts.enable_async()?;
        }
        eprintln!(
            "GLM load breakdown: resident_s={resident_seconds:.3} expert_cache_s={:.3}",
            expert_start.elapsed().as_secs_f64()
        );
        let (free, _) = device.memory_info()?;
        let mtp_experts_gpu = if options.mtp {
            experts.layer_gpu_bytes(45)
        } else {
            0
        };
        let memory = MemoryUsage {
            weights,
            state_and_workspace,
            latent_kv_host: (options.layers / 4 + usize::from(options.mtp))
                * options.context
                * 512
                * 2,
            experts_gpu: experts.gpu_bytes,
            experts_host: experts.host_bytes,
            experts_disk: experts.disk_bytes,
            vram_peak: total - free,
            mtp_added,
            mtp_experts_gpu,
        };
        eprintln!("GLM load: initial_free={initial_free}, memory={memory:?}");
        let ready = stream.event()?;
        if options.decode_async || options.decode_device {
            for layer in layers.iter_mut().chain(mtp.iter_mut()) {
                if layer.index >= 3 {
                    layer.asynchronous = Some(super::async_decode::LayerAsync::new(device)?);
                }
            }
        }
        if options.decode_device && !layers.iter().all(Layer::device_supported) {
            return Err(Error::InvalidPlan(
                "GLM device decode needs unrotated experts".into(),
            ));
        }
        let mut device_decode = if options.decode_device {
            Some(device_step::DeviceDecode::new(
                device,
                options.layers,
                options.defer_cpu_below,
                options.skip_cpu_below,
            )?)
        } else {
            None
        };
        if let (Some(dd), Some(pcie)) = (device_decode.as_mut(), pcie) {
            // The PCIe lane: per layer, its largest expert image is the copy size.
            for layer in layers.iter_mut().filter(|l| l.index >= 3) {
                let bytes = experts.expert_bytes[(layer.index - 3) * 288..][..288]
                    .iter()
                    .map(|&b| b.min(experts.image_bytes))
                    .max()
                    .unwrap_or(0)
                    .next_multiple_of(16);
                layer.pcie_bytes = bytes;
            }
            dd.set_pcie(pcie);
        }
        if layers.iter().any(|l| l.lowbit) {
            // LOWBIT experts decode through the format-aware kernels of the
            // device path only (the per-op path and MTP know NVFP4 and FP8).
            let Some(dd) = device_decode.as_mut().filter(|_| !options.mtp) else {
                return Err(Error::InvalidPlan(
                    "GLM LOWBIT experts need --decode-host device without MTP".into(),
                ));
            };
            dd.set_formats(
                device,
                &experts.expert_formats[..(options.layers - 3) * 288],
            )?;
        }
        Ok(Self {
            image_embeddings: BTreeMap::new(),
            device,
            stream,
            layers,
            mtp,
            mtp_reserve_only: options.mtp_reserve_only,
            spec,
            experts,
            decode_async: options.decode_async,
            async_prepared: false,
            device_decode,
            device_ready: false,
            host_syncs: 0,
            decode_steps: 0,
            prefill,
            globals,
            input,
            mean,
            normalized,
            logits,
            next,
            vocab,
            context: options.context,
            position: 0,
            source_hash,
            ready,
            reference: false,
            head_format,
            profile: None,
            expert_prefetch: options.expert_prefetch,
            memory,
            vram_budget: options.vram_budget,
            swaps: options.maintenance_swaps,
            prefill_reserve,
            cancel: Default::default(),
            placement: (options.placement_beta, options.placement_swaps),
            placed: Default::default(),
        })
    }
    fn prepare_async(&mut self) -> Result<()> {
        if !self.decode_async || self.reference || self.async_prepared {
            return Ok(());
        }
        for layer in self.layers.iter_mut().chain(self.mtp.iter_mut()) {
            layer.async_active = true;
            if layer.index >= 3 {
                layer
                    .asynchronous
                    .as_mut()
                    .unwrap()
                    .refresh(self.experts.snapshot(layer.index));
            }
        }
        self.async_prepared = true;
        Ok(())
    }
    fn retire_async(&mut self) -> Result<()> {
        if !self.async_prepared {
            return Ok(());
        }
        self.experts.retire_async()?;
        for layer in self.layers.iter_mut().chain(self.mtp.iter_mut()) {
            if let Some(a) = &mut layer.asynchronous {
                a.snapshot = None;
            }
        }
        self.async_prepared = false;
        Ok(())
    }
    /// Elastic VRAM budget (bytes, everything GLM owns on the GPU). Shrinks or
    /// grows the expert cache in place without a reload; resident weights,
    /// KV/state and workspace stay. Call between requests or tokens, never
    /// concurrently with `step`/`prefill`. Shrinking demotes the lowest-ranked
    /// GPU experts to pinned RAM (or to the disk tier when RAM is full and
    /// `allow_disk` was set) and frees their VRAM before returning; the
    /// returned `Resize::gpu_delta` is the released byte count (negative when
    /// growing). Fails without changes when the budget cannot hold the
    /// resident part, or RAM/disk cannot take the evicted experts.
    pub fn set_vram_budget(&mut self, bytes: u64) -> Result<super::experts::Resize> {
        self.device_ready = false;
        if self.async_prepared {
            self.stream.synchronize()?;
            self.retire_async()?;
        }
        let (free, _) = self.device.memory_info()?;
        let owned =
            self.memory.weights + self.memory.state_and_workspace + self.prefill_reserve as usize;
        let image = self.experts.image_bytes;
        let current = self.experts.slot_bytes() as usize;
        let target =
            super::experts::budget_experts(bytes, owned, image, self.swaps, free + current)?;
        let report = self.experts.resize(self.device, target)?;
        self.vram_budget = Some(bytes);
        self.memory.experts_gpu = self.experts.gpu_bytes;
        self.memory.experts_host = self.experts.host_bytes;
        self.memory.experts_disk = self.experts.disk_bytes;
        eprintln!(
            "glm_vram_budget bytes={bytes} slots_before={} slots_after={} gpu_delta={} vram_freed={} packed={} d2h_bytes={} h2d_bytes={} disk_reads={} seconds={:.3} host_bytes={} disk_bytes={}",
            report.slots_before,
            report.slots_after,
            report.gpu_delta,
            report.vram_freed,
            report.packed,
            report.d2h_bytes,
            report.h2d_bytes,
            report.disk_reads,
            report.seconds,
            self.experts.host_bytes,
            self.experts.disk_bytes,
        );
        Ok(report)
    }
    /// Flag checked by `prefill_cancellable` between tokens (short prompts) and
    /// between layers (batched chunks). Set it from any thread; the prefill
    /// returns within about one layer and clears it.
    pub fn cancel_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        self.cancel.clone()
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(std::sync::atomic::Ordering::Acquire)
    }
    pub fn position(&self) -> usize {
        self.position
    }
    /// Full decode state at the current position: per layer the KDA state
    /// (`state`), conv histories (`hq`, `hk`, `hv`), index pool tails
    /// (`pool_keys`, `pool_gates`), the index pools prefix (`pools`) and the
    /// latent KV prefix (pinned RAM); MTP layer included when loaded.
    /// Restore with `import_session`, then `prefill` the new suffix.
    pub fn export_session(&mut self) -> Result<Session> {
        if self.async_prepared {
            self.stream.synchronize()?;
            self.retire_async()?;
        }
        Ok(Session {
            position: self.position,
            layers: self
                .layers
                .iter()
                .map(|l| l.save_session(true))
                .collect::<Result<_>>()?,
            mtp: self
                .mtp
                .as_ref()
                .map(|l| l.save_session(true))
                .transpose()?,
        })
    }
    pub fn import_session(&mut self, session: &Session) -> Result<()> {
        self.device_ready = false;
        if session.layers.len() != self.layers.len()
            || session.mtp.is_some() != self.mtp.is_some()
            || session.position > self.context
        {
            return Err(Error::InvalidPlan("GLM session layout".into()));
        }
        self.reset()?;
        for (layer, saved) in self.layers.iter_mut().zip(&session.layers) {
            layer.restore_session(saved)?;
        }
        if let (Some(layer), Some(saved)) = (&mut self.mtp, &session.mtp) {
            layer.restore_session(saved)?;
        }
        self.position = session.position;
        Ok(())
    }
    fn snapshot_layers(&self) -> Result<Vec<super::LayerSession>> {
        self.layers.iter().map(|l| l.save_session(false)).collect()
    }
    fn restore_layers(&mut self, saved: &[super::LayerSession]) -> Result<()> {
        self.device_ready = false;
        for layer in &self.layers {
            layer.synchronize()?;
        }
        self.stream.synchronize()?;
        for (layer, s) in self.layers.iter_mut().zip(saved) {
            layer.restore_session(s)?;
        }
        Ok(())
    }
    /// Prefill that can be interrupted through `cancel_handle`.
    pub fn prefill_cancellable(&mut self, ids: &[u32]) -> Result<PrefillOutcome> {
        let result = self.prefill(ids);
        let cancelled = self.cancel.swap(false, std::sync::atomic::Ordering::AcqRel);
        match result {
            Ok(next) => Ok(PrefillOutcome::Done(next)),
            Err(Error::InvalidPlan(message)) if cancelled && message == CANCELLED => {
                Ok(PrefillOutcome::Cancelled {
                    position: self.position,
                })
            }
            Err(error) => Err(error),
        }
    }
    /// Bytes GLM currently owns on the GPU (planned, not driver-measured).
    pub fn vram_bytes(&self) -> u64 {
        (self.memory.weights + self.memory.state_and_workspace) as u64 + self.experts.gpu_bytes
    }
    /// Free VRAM that temporary prefill buffers may use: the driver's free
    /// memory, capped by the remaining budget when one is set.
    fn usable_free(&self, free: usize) -> usize {
        match self.vram_budget {
            Some(budget) => free.min(budget.saturating_sub(self.vram_bytes()) as usize),
            None => free,
        }
    }
    pub fn has_images(&self) -> bool {
        !self.image_embeddings.is_empty()
    }
    /// Projected image rows, before the language model expands its hyper-connection streams.
    pub fn set_image_embeddings(&mut self, images: Vec<(usize, Vec<f32>)>) -> Result<()> {
        self.image_embeddings.clear();
        for (start, values) in images {
            if values.len() % HIDDEN != 0
                || start + values.len() / HIDDEN > self.context
                || values.iter().any(|x| !x.is_finite())
            {
                return Err(Error::InvalidShape("GLM visual embeddings/context".into()));
            }
            for (i, row) in values.chunks_exact(HIDDEN).enumerate() {
                let mut bytes = Vec::with_capacity(HIDDEN * HC * 2);
                for _ in 0..HC {
                    for &v in row {
                        let b = v.to_bits();
                        let bf = ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16;
                        bytes.extend(bf.to_le_bytes());
                    }
                }
                self.image_embeddings.insert(start + i, bytes);
            }
        }
        self.device_ready = false;
        Ok(())
    }
    pub fn step(&mut self, token: u32) -> Result<u32> {
        self.experts.check_worker_error()?;
        if token as usize >= self.vocab || self.position >= self.context {
            return Err(Error::InvalidShape("GLM token/context out of range".into()));
        }
        if self.device_decode.is_some()
            && !self.reference
            && !self.image_embeddings.contains_key(&self.position)
        {
            let host_syncs = hayate_kernels::runtime::host_sync_count();
            let next = self.step_device(token)?;
            self.host_syncs += hayate_kernels::runtime::host_sync_count() - host_syncs;
            self.decode_steps += 1;
            self.position += 1;
            self.probe_placement()?;
            return Ok(next);
        }
        self.device_ready = false;
        self.prepare_async()?;
        let host_syncs = hayate_kernels::runtime::host_sync_count();
        let token_start = Instant::now();
        let maintenance_before = (self.experts.d2h_count, self.experts.d2h_wait_ms);
        let transfer_before = self.experts.transfer_stats;
        check(
            unsafe {
                g::hayate_glm_embed(
                    self.input.as_mut_ptr(),
                    self.globals["embed_tokens.weight"].as_ptr(),
                    token,
                    HIDDEN as u32,
                    self.stream.as_raw(),
                )
            },
            "glm embedding",
        )?;
        if let Some(bytes) = self.image_embeddings.get(&self.position) {
            self.stream.copy_h2d(&mut self.input, bytes)?;
        }
        if let Some(p) = &self.profile {
            p.reset();
        }
        let next = if self.reference {
            let mut hidden = vec![0; HC * HIDDEN * 2];
            self.download(&mut hidden, &self.input)?;
            for layer in &mut self.layers {
                hidden = layer.step(&hidden, &mut self.experts)?;
            }
            self.head(&hidden)?
        } else {
            self.ready.record(&self.stream)?;
            let mut input = self.input.as_mut_ptr();
            for index in 0..self.layers.len() {
                let (previous, current) = self.layers.split_at_mut(index);
                let prediction_wait = if self.expert_prefetch && index >= 4 {
                    Some(current[0].prefetch_from(&previous[index - 1], &mut self.experts)?)
                } else {
                    None
                };
                let dependency = previous.last().map_or(&self.ready, |layer| &layer.done);
                input = current[0].step_device(input, dependency, &mut self.experts)?;
                if let Some(wait) = prediction_wait {
                    current[0].record_prediction_wait(wait);
                }
            }
            let ready = &self.layers.last().unwrap().done;
            self.stream.wait_event(ready)?;
            check(
                unsafe {
                    g::hayate_glm_copy(
                        self.input.as_mut_ptr(),
                        input,
                        HC * HIDDEN * 2,
                        self.stream.as_raw(),
                    )
                },
                "glm head input D2D",
            )?;
            self.head_device()?
        };
        if self.mtp.is_some() {
            self.spec_copy("target", self.normalized.as_ptr().cast_mut(), HIDDEN * 2)?;
        }
        self.retire_async()?;
        let maintenance_start = Instant::now();
        self.experts.finish_token(&self.stream)?;
        let maintenance_ms = maintenance_start.elapsed().as_secs_f64() * 1000.;
        let (free, total) = self.device.memory_info()?;
        self.memory.vram_peak = self.memory.vram_peak.max(total - free);
        if let Some(p) = &self.profile {
            eprintln!(
                "glm_profile position={} layer=maintenance kind=maintenance maintenance_ms={maintenance_ms:.6} d2h_count={} d2h_sync_count={} d2h_wait_ms={:.6} token_wall_ms={:.6} maintenance_worker_ms={:.6} host_h2d={} disk_h2d={} h2d_bytes={} disk_bytes={} read_wait_ms={:.6} reader_ms={:.6} reader_io_ms={:.6} reader_bytes={} reader_pack_ms={:.6} reader_upload_ms={:.6} predicted={} prefetch_hits={} maintenance_h2d={} maintenance_d2h={} maintenance_disk={} maintenance_bytes={} maintenance_wait_ms={:.6} cpu_experts={} cpu_bytes={} cpu_ms={:.6} cpu_worker_ms={:.6} cpu_dispatch_ms={:.6}",
                self.position,
                self.experts.d2h_count - maintenance_before.0,
                0,
                0.,
                token_start.elapsed().as_secs_f64() * 1000.,
                self.experts.d2h_wait_ms - maintenance_before.1,
                self.experts.transfer_stats.host_h2d - transfer_before.host_h2d,
                self.experts.transfer_stats.disk_h2d - transfer_before.disk_h2d,
                self.experts.transfer_stats.h2d_bytes - transfer_before.h2d_bytes,
                self.experts.transfer_stats.disk_bytes - transfer_before.disk_bytes,
                self.experts.transfer_stats.read_wait_ms - transfer_before.read_wait_ms,
                self.experts.transfer_stats.reader_ms - transfer_before.reader_ms,
                self.experts.transfer_stats.reader_io_ms - transfer_before.reader_io_ms,
                self.experts.transfer_stats.reader_bytes - transfer_before.reader_bytes,
                self.experts.transfer_stats.reader_pack_ms - transfer_before.reader_pack_ms,
                self.experts.transfer_stats.reader_upload_ms - transfer_before.reader_upload_ms,
                self.experts.transfer_stats.predicted - transfer_before.predicted,
                self.experts.transfer_stats.prefetch_hits - transfer_before.prefetch_hits,
                self.experts.transfer_stats.maintenance_h2d - transfer_before.maintenance_h2d,
                self.experts.transfer_stats.maintenance_d2h - transfer_before.maintenance_d2h,
                self.experts.transfer_stats.maintenance_disk - transfer_before.maintenance_disk,
                self.experts.transfer_stats.maintenance_bytes - transfer_before.maintenance_bytes,
                self.experts.transfer_stats.maintenance_wait_ms
                    - transfer_before.maintenance_wait_ms,
                self.experts.transfer_stats.cpu_experts - transfer_before.cpu_experts,
                self.experts.transfer_stats.cpu_bytes - transfer_before.cpu_bytes,
                self.experts.transfer_stats.cpu_ms - transfer_before.cpu_ms,
                self.experts.transfer_stats.cpu_worker_ms - transfer_before.cpu_worker_ms,
                self.experts.transfer_stats.cpu_dispatch_ms - transfer_before.cpu_dispatch_ms,
            );
            for layer in &self.layers {
                layer.report_profile(self.position)?;
            }
            let waits = *p.waits.borrow();
            eprintln!(
                "glm_profile position={} layer=head kind=head mhc_ms={:.6} lm_head_ms={:.6} greedy_ms={:.6} d2h_count={} d2h_sync_count={} d2h_wait_ms={:.6}",
                self.position,
                p.ms(0, 1)?,
                p.ms(1, 2)?,
                p.ms(2, 3)?,
                waits.d2h_count,
                waits.d2h_sync_count,
                waits.d2h_ms
            );
        }
        self.host_syncs += hayate_kernels::runtime::host_sync_count() - host_syncs;
        self.decode_steps += 1;
        self.position += 1;
        self.probe_placement()?;
        Ok(next)
    }
    fn head(&mut self, hidden: &[u8]) -> Result<u32> {
        self.stream.copy_h2d(&mut self.input, hidden)?;
        self.head_device()
    }
    fn head_device(&mut self) -> Result<u32> {
        self.mark(0)?;
        check(
            unsafe {
                g::hayate_glm_head_mean(
                    self.mean.as_mut_ptr(),
                    self.input.as_ptr(),
                    HIDDEN as u32,
                    self.stream.as_raw(),
                )
            },
            "glm head mean",
        )?;
        check(
            unsafe {
                g::hayate_glm_norm(
                    self.normalized.as_mut_ptr(),
                    self.mean.as_ptr(),
                    self.globals["norm.weight"].as_ptr(),
                    std::ptr::null(),
                    1,
                    HIDDEN as u32,
                    0,
                    self.stream.as_raw(),
                )
            },
            "glm final norm",
        )?;
        self.mark(1)?;
        if self.reference && self.head_format == 0 {
            check(
                unsafe {
                    hayate_text_linear(
                        self.logits.as_mut_ptr(),
                        self.normalized.as_ptr(),
                        self.globals["lm_head.weight"].as_ptr(),
                        1,
                        self.vocab as u32,
                        HIDDEN as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm lm head",
            )?;
        } else {
            check(
                unsafe {
                    g::hayate_glm_dense(
                        self.logits.as_mut_ptr(),
                        self.normalized.as_ptr(),
                        g::DenseWeight {
                            data: self.globals["lm_head.weight"].as_ptr(),
                            scale: std::ptr::null(),
                            format: self.head_format,
                        },
                        self.vocab as u32,
                        HIDDEN as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm dense lm head",
            )?;
        }
        self.mark(2)?;
        check(
            unsafe {
                g::hayate_glm_greedy(
                    self.next.as_mut_ptr(),
                    self.logits.as_ptr(),
                    self.vocab as u32,
                    self.stream.as_raw(),
                )
            },
            "glm greedy",
        )?;
        self.mark(3)?;
        let mut raw = [0; 4];
        self.download(&mut raw, &self.next)?;
        Ok(u32::from_le_bytes(raw))
    }
    pub fn decode_reference(&mut self, enabled: bool) {
        self.reference = enabled;
        for layer in &mut self.layers {
            layer.reference = enabled;
            layer.graphs = !enabled;
        }
    }
    pub fn disable_graphs(&mut self) {
        for layer in &mut self.layers {
            layer.graphs = false;
        }
    }
    pub fn enable_profile(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.enable_profile()?;
        }
        self.device_profile(true)?;
        self.profile = Some(super::profile::Profile::new(&self.stream, 4)?);
        Ok(())
    }
    fn mark(&self, index: usize) -> Result<()> {
        if let Some(p) = &self.profile {
            p.mark(index, &self.stream)?;
        }
        Ok(())
    }
    fn download(&self, dst: &mut [u8], src: &DeviceBuffer) -> Result<()> {
        if let Some(p) = &self.profile {
            p.download(&self.stream, dst, src)
        } else {
            self.stream.copy_d2h(dst, src)
        }
    }
    #[cfg(test)]
    pub(super) fn prefill_all_experts(&mut self, all: bool) {
        self.prefill.all_experts = all;
    }
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32> {
        self.device_ready = false;
        if ids.is_empty() || ids.len() > self.context - self.position {
            return Err(Error::InvalidShape("GLM empty/oversized prefill".into()));
        }
        if ids.iter().any(|&id| id as usize >= self.vocab) {
            return Err(Error::InvalidShape("GLM token out of range".into()));
        }
        let transfer_bytes = self.experts.prefill_transfer_bytes(ids.len());
        if self.prefill.batched
            && !self.reference
            && !self.prefill.all_experts
            && self.experts.cpu_enabled()
            && super::prefill::cpu_short_prompt(ids.len(), transfer_bytes)
        {
            let start = Instant::now();
            let mut next = 0;
            for &id in ids {
                if self.cancelled() {
                    return Err(Error::InvalidPlan(CANCELLED.into()));
                }
                next = self.step(id)?;
            }
            if self.profile.is_some() {
                eprintln!(
                    "glm_prefill_policy method=cpu rows={} estimated_h2d_bytes={transfer_bytes}",
                    ids.len()
                );
                eprintln!(
                    "glm_prefill_layer position={} rows={} layer=all kind=cpu wall_ms={:.6}",
                    self.position - ids.len(),
                    ids.len(),
                    start.elapsed().as_secs_f64() * 1000.
                );
            }
            // A short prompt also tells which experts the reply will use
            // (the probe decides after a few decoded tokens).
            if !self.reference {
                self.place_for_context()?;
            }
            return Ok(next);
        }
        if self.async_prepared {
            self.stream.synchronize()?;
            self.retire_async()?;
        }
        for layer in self.layers.iter_mut().chain(self.mtp.iter_mut()) {
            layer.async_active = false;
        }
        let capacity = self.prefill.capacity;
        self.prefill.capacity = capacity.min(ids.len());
        let result = self.prefill_allocated(ids);
        self.prefill.capacity = capacity;
        if result.is_ok() && !self.reference {
            self.place_for_context()?;
        }
        result
    }
    /// `Experts::place_for_context` at the end of a prefill (no-op when
    /// `placement_tail` is 0). Device decode rebuilds its expert table and
    /// host rows from the slots at the next token.
    fn place_for_context(&mut self) -> Result<()> {
        let (beta, swaps) = self.placement;
        for layer in &mut self.layers {
            if let Some(a) = layer.asynchronous.as_mut() {
                a.snapshot = None;
            }
        }
        self.experts.place_for_context(f64::from(beta), swaps);
        Ok(())
    }
    /// `Experts::probe_token` after a decoded token (no-op unless due).
    fn probe_placement(&mut self) -> Result<()> {
        if self.reference {
            return Ok(());
        }
        if self.async_prepared {
            self.stream.synchronize()?;
            self.retire_async()?;
        }
        for layer in &mut self.layers {
            if let Some(a) = layer.asynchronous.as_mut() {
                a.snapshot = None;
            }
        }
        let placed = self.experts.probe_token(self.device)?;
        self.note_placement("probe", placed);
        Ok(())
    }
    fn note_placement(&mut self, kind: &str, placed: super::experts::Placed) {
        if placed.swaps > 0 {
            eprintln!(
                "glm_context_placement kind={kind} position={} swaps={} bytes={} seconds={:.3}",
                self.position, placed.swaps, placed.bytes, placed.seconds
            );
        }
        self.placed.swaps += placed.swaps;
        self.placed.bytes += placed.bytes;
        self.placed.seconds += placed.seconds;
    }
    fn prefill_allocated(&mut self, ids: &[u32]) -> Result<u32> {
        let borrowed = self.prefill.batched && self.prefill.capacity > 1 && ids.len() > 1;
        let loan_start = Instant::now();
        let mut d2h_bytes = 0;
        let mut first_layer_done = false;
        if borrowed {
            let desired = if self.prefill.all_experts || ids.len() > self.prefill.capacity {
                EXPERTS
            } else {
                self.prefill
                    .previous
                    .iter()
                    .map(Vec::len)
                    .max()
                    .unwrap_or(0)
            }
            .min(self.experts.max_missing());
            // Never borrow cache slots when avoidable (the D2H/H2D of a loan
            // costs seconds and cannot be interrupted): shrink the chunk until
            // the workspace fits the free VRAM, then give the next-layer
            // lookahead whatever is left. Keep 2 GiB for cuBLAS/driver.
            // A big prompt is the exception: every halving of the chunk doubles
            // the expert uploads (34K tokens: 31 s at 8K chunks with 8 GiB free,
            // 53 s with 2 GiB), more than a loan costs (~2 s), so it keeps the
            // chunk and borrows up to half of the full-size slots.
            let (free, _) = self.device.memory_info()?;
            let free = self.usable_free(free);
            let free_slots = free.saturating_sub(2 << 30) / self.experts.image_bytes;
            let loanable = if ids.len() >= BIG_PREFILL_TOKENS {
                self.experts.loanable_slots() / 2
            } else {
                0
            };
            self.prefill.prediction_capacity = 0;
            while self.prefill.capacity > 512 && self.prefill.loan_slots() > free_slots + loanable {
                self.prefill.capacity = self.prefill.capacity.div_ceil(2);
            }
            let spare = free_slots.saturating_sub(self.prefill.loan_slots());
            self.prefill.prediction_capacity = desired.min(spare / 2);
            let layer_bytes = super::prefill::layer_layout(self.prefill.capacity, self.context)
                .iter()
                .map(|(_, bytes)| bytes)
                .sum::<usize>()
                + self.prefill.capacity * 8 * 4;
            let overlap = !self.reference
                && self.prefill.loan_slots() > free_slots
                && layer_bytes <= free.saturating_sub(2 << 30);
            if overlap && let Err(error) = self.prefill.prepare_first_layer(self.device) {
                self.prefill.release()?;
                return Err(error);
            }
            let (free, _) = self.device.memory_info()?;
            let free = self.usable_free(free);
            let free_slots = free.saturating_sub(2 << 30) / self.experts.image_bytes;
            let loan = match self.experts.borrow_prefill(
                self.device,
                &self.stream,
                self.prefill.loan_slots().saturating_sub(free_slots),
                !overlap,
            ) {
                Ok(loan) => loan,
                Err(error) => {
                    self.prefill.release()?;
                    return Err(error);
                }
            };
            d2h_bytes = loan.copied_bytes;
            self.prefill.loan = loan.slots;
            let mut pool = loan.buffers;
            // Layer 0 is dense and touches neither borrowed slots nor host
            // experts. Its independent arena can run while pinned D2H is busy.
            let first = if overlap {
                self.prefill_embed(&ids[..self.prefill.capacity])
                    .and_then(|()| {
                        self.layers[0].prefill_device(
                            self.prefill.capacity,
                            &mut self.experts,
                            &mut self.prefill,
                            &self.ready,
                        )
                    })
            } else {
                Ok(())
            };
            let retired = self.experts.wait_prefill_loan();
            let allocated = retired
                .and(first)
                .and_then(|()| self.prefill.allocate(self.device, &mut pool));
            if allocated.is_ok() && self.profile.is_some() && d2h_bytes > 0 {
                let origin = &loan.copy_events[0].0;
                let begin = origin.elapsed_ms(&loan.copy_events[1].0)?.min(0.);
                let end = origin
                    .elapsed_ms(&loan.copy_events[0].1)?
                    .max(origin.elapsed_ms(&loan.copy_events[1].1)?);
                let overlap_ms = if overlap {
                    self.layers[0].done.synchronize()?;
                    (end.min(origin.elapsed_ms(&self.layers[0].done)?)
                        - begin.max(origin.elapsed_ms(&self.ready)?))
                    .max(0.)
                } else {
                    0.
                };
                eprintln!(
                    "glm_prefill_workspace stage=backup d2h_bytes={d2h_bytes} dma_ms={:.6} first_layer_overlap_ms={overlap_ms:.6}",
                    end - begin
                );
            }
            first_layer_done = overlap;
            drop(pool);
            if let Err(error) = allocated {
                for layer in &self.layers {
                    layer.synchronize()?;
                }
                self.prefill.release()?;
                self.experts
                    .return_prefill(self.device, &self.stream, &mut self.prefill.loan)?;
                return Err(error);
            }
        }
        if borrowed && self.profile.is_some() {
            eprintln!(
                "glm_prefill_workspace stage=borrow capacity={} loan_slots={} d2h_bytes={} workspace_bytes={} elapsed_ms={:.6}",
                self.prefill.capacity,
                self.prefill.loan.len(),
                d2h_bytes,
                self.prefill.bytes(),
                loan_start.elapsed().as_secs_f64() * 1000.
            );
        }
        let before = self.experts.transfer_stats.clone();
        let chunks_start = Instant::now();
        let result = self.prefill_chunks(ids, borrowed, first_layer_done);
        let after = &self.experts.transfer_stats;
        eprintln!(
            "glm_prefill_summary rows={} capacity={} chunks={} loan_slots={} loan_ms={:.1} chunks_ms={:.1} h2d_bytes={} disk_bytes={}",
            ids.len(),
            self.prefill.capacity,
            ids.len().div_ceil(self.prefill.capacity.max(1)),
            self.prefill.loan.len(),
            (chunks_start - loan_start).as_secs_f64() * 1000.,
            chunks_start.elapsed().as_secs_f64() * 1000.,
            after.h2d_bytes - before.h2d_bytes,
            after.disk_bytes - before.disk_bytes,
        );
        let return_start = Instant::now();
        if borrowed {
            let restore_bytes = self.prefill.loan.len() * self.experts.image_bytes;
            // A failed layer may have submitted kernels after its last event.
            for layer in &self.layers {
                layer.synchronize()?;
            }
            self.stream.synchronize()?;
            self.prefill.release()?;
            self.experts
                .return_prefill(self.device, &self.stream, &mut self.prefill.loan)?;
            if self.profile.is_some() {
                eprintln!(
                    "glm_prefill_workspace stage=return h2d_bytes={restore_bytes} elapsed_ms={:.6}",
                    return_start.elapsed().as_secs_f64() * 1000.
                );
            }
            if result.is_ok() {
                self.experts.finish_token(&self.stream)?;
            }
        }
        result
    }
    fn prefill_chunks(
        &mut self,
        ids: &[u32],
        borrowed: bool,
        first_layer_done: bool,
    ) -> Result<u32> {
        let mut next = 0;
        let capacity = self.prefill.capacity;
        let mut prompt_offset = 0;
        for (chunk_index, chunk) in ids.chunks(capacity).enumerate() {
            if self.cancelled() {
                return Err(Error::InvalidPlan(CANCELLED.into()));
            }
            if borrowed && !self.reference {
                // Layer 0 of the first chunk may already have run during the
                // cache loan; that chunk is cancellable only at its end.
                let overlapped = first_layer_done && chunk_index == 0;
                let saved = if overlapped {
                    Vec::new()
                } else {
                    self.snapshot_layers()?
                };
                next = match self.prefill_batch(chunk, overlapped) {
                    Err(Error::InvalidPlan(message)) if message == CANCELLED => {
                        self.restore_layers(&saved)?;
                        return Err(Error::InvalidPlan(CANCELLED.into()));
                    }
                    other => other?,
                };
                self.position += chunk.len();
                let (free, total) = self.device.memory_info()?;
                self.memory.vram_peak = self.memory.vram_peak.max(total - free);
                continue;
            }

            let mut hidden = Vec::with_capacity(chunk.len() * HC * HIDDEN * 2);
            for (row_index, &id) in chunk.iter().enumerate() {
                check(
                    unsafe {
                        g::hayate_glm_embed(
                            self.input.as_mut_ptr(),
                            self.globals["embed_tokens.weight"].as_ptr(),
                            id,
                            HIDDEN as u32,
                            self.stream.as_raw(),
                        )
                    },
                    "glm prefill embed",
                )?;
                if let Some(bytes) = self.image_embeddings.get(&(self.position + row_index)) {
                    self.stream.copy_h2d(&mut self.input, bytes)?;
                }
                let mut row = vec![0; HC * HIDDEN * 2];
                self.stream.copy_d2h(&mut row, &self.input)?;
                hidden.extend(row);
            }
            for layer in &mut self.layers {
                hidden = layer.prefill_cached(&hidden, &mut self.experts, &mut self.prefill)?;
            }
            if self.mtp.is_some() && !self.mtp_reserve_only {
                for row in 0..chunk.len() {
                    next =
                        self.head(&hidden[row * HC * HIDDEN * 2..(row + 1) * HC * HIDDEN * 2])?;
                    self.spec_copy("target", self.normalized.as_ptr().cast_mut(), HIDDEN * 2)?;
                    let next_id = ids.get(prompt_offset + row + 1).copied();
                    if let Some(token) = next_id {
                        self.draft_prefill(token)?;
                    }
                }
            } else {
                next = self.head(&hidden[hidden.len() - HC * HIDDEN * 2..])?;
            }
            prompt_offset += chunk.len();
            if !borrowed {
                self.experts.finish_token(&self.stream)?;
            }
            self.position += chunk.len();
            let (free, total) = self.device.memory_info()?;
            self.memory.vram_peak = self.memory.vram_peak.max(total - free);
        }
        Ok(next)
    }
    fn prefill_embed(&mut self, ids: &[u32]) -> Result<()> {
        self.stream.copy_h2d(
            self.prefill.buffers.get_mut("ids").unwrap(),
            &ids.iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        check(
            unsafe {
                g::hayate_glm_embed_batch(
                    self.prefill.layer["input"].as_ptr().cast_mut(),
                    self.globals["embed_tokens.weight"].as_ptr(),
                    self.prefill.buffers["ids"].as_ptr(),
                    ids.len() as u32,
                    self.stream.as_raw(),
                )
            },
            "glm batch embed",
        )?;
        for (&position, bytes) in self
            .image_embeddings
            .range(self.position..self.position + ids.len())
        {
            self.stream.copy_h2d_at(
                self.prefill.layer.get_mut("input").unwrap(),
                (position - self.position) * HC * HIDDEN * 2,
                bytes,
            )?;
        }
        self.ready.record(&self.stream)
    }
    fn prefill_batch(&mut self, ids: &[u32], first_layer_done: bool) -> Result<u32> {
        if !first_layer_done {
            self.prefill_embed(ids)?;
        }
        for i in usize::from(first_layer_done)..self.layers.len() {
            if !first_layer_done && self.cancelled() {
                return Err(Error::InvalidPlan(CANCELLED.into()));
            }
            let (previous, current) = self.layers.split_at_mut(i);
            let ready = previous.last().map_or(&self.ready, |layer| &layer.done);
            current[0].prefill_device(ids.len(), &mut self.experts, &mut self.prefill, ready)?;
        }
        self.stream.wait_event(&self.layers.last().unwrap().done)?;
        check(
            unsafe {
                g::hayate_glm_copy(
                    self.input.as_mut_ptr(),
                    self.prefill.layer["output"]
                        .as_ptr()
                        .cast::<u8>()
                        .add((ids.len() - 1) * HC * HIDDEN * 2)
                        .cast(),
                    HC * HIDDEN * 2,
                    self.stream.as_raw(),
                )
            },
            "glm batch head input",
        )?;
        self.head_device()
    }
    pub fn reset(&mut self) -> Result<()> {
        self.device_ready = false;
        if self.async_prepared {
            self.stream.synchronize()?;
            self.retire_async()?;
        }
        self.experts.drain()?;
        self.experts.clear_tail();
        self.prefill.previous.iter_mut().for_each(Vec::clear);
        for layer in &mut self.layers {
            layer.reset()?;
        }
        if let Some(mtp) = &mut self.mtp {
            mtp.reset()?;
        }
        self.retire_async()?;
        for layer in self.layers.iter_mut().chain(self.mtp.iter_mut()) {
            layer.async_active = false;
        }
        self.position = 0;
        Ok(())
    }
    pub fn save_frequencies(&self, path: &Path) -> Result<()> {
        let raw: Vec<_> = self
            .experts
            .counts
            .as_chunks::<288>()
            .0
            .iter()
            .flat_map(|counts| {
                let total = counts.iter().sum::<u64>().max(1) as f32;
                counts
                    .iter()
                    .flat_map(move |&n| (n as f32 * 8. / total).to_le_bytes())
            })
            .collect();
        std::fs::write(path, raw)?;
        Ok(())
    }
    pub fn logits(&self) -> Result<Vec<f64>> {
        let raw = self.logits_raw()?;
        Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16) as f64)
            .collect())
    }
    fn logits_raw(&self) -> Result<Vec<u8>> {
        let mut raw = vec![0; self.vocab * 2];
        self.stream.copy_d2h(&mut raw, &self.logits)?;
        Ok(raw)
    }
    pub(super) fn evaluate(&mut self, ids: &[u32], prefix: usize, out: &Path) -> Result<()> {
        self.evaluate_mode(ids, prefix, out, false)
    }
    fn evaluate_mode(
        &mut self,
        ids: &[u32],
        prefix: usize,
        out: &Path,
        step_prefix: bool,
    ) -> Result<()> {
        use std::io::Write;
        if prefix == 0
            || prefix >= ids.len().saturating_sub(1)
            || ids.len() > self.context
            || ids.iter().any(|&id| id as usize >= self.vocab)
        {
            return Err(Error::InvalidShape(
                "GLM eval token outside vocabulary".into(),
            ));
        }
        // A fresh directory and a manifest written last prevent reuse of partial runs.
        if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::create_dir(out)?;
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(out.join("eval_logits.bf16.bin"))?);
        self.reset()?;
        let prefix_start = Instant::now();
        if step_prefix {
            for &id in &ids[..prefix] {
                self.step(id)?;
            }
        } else {
            self.prefill(&ids[..prefix])?;
        }
        let prefix_seconds = prefix_start.elapsed().as_secs_f64();
        let mut prefix_file = std::fs::File::create(out.join("prefix_logits.bf16.bin"))?;
        super::eval::write_logits(&mut prefix_file, &self.logits_raw()?)?;
        prefix_file.sync_all()?;
        eprintln!(
            "eval_prefix={prefix} prefill_s={prefix_seconds:.6} prefill_tok_s={:.6}",
            prefix as f64 / prefix_seconds
        );
        let start = Instant::now();
        for (row, &id) in ids[prefix..].iter().enumerate() {
            self.step(id)?;
            super::eval::write_logits(&mut file, &self.logits_raw()?)?;
            if (row + 1).is_multiple_of(32) {
                eprintln!(
                    "eval_rows={}/{} elapsed_s={:.3}",
                    row + 1,
                    ids.len() - prefix,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        file.flush()?;
        file.get_ref().sync_all()?;
        std::fs::write(
            out.join("eval_ids.i32.bin"),
            ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )?;
        let source_hash = self
            .source_hash
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect::<String>();
        std::fs::write(
            out.join("eval.json"),
            format!(
                "{{\"prefix\":{prefix},\"rows\":{},\"vocab\":{},\"source_hash\":\"{source_hash}\",\"layers\":{},\"prefill_chunk\":{},\"dtype\":\"bf16\",\"eval_mode\":\"{}\",\"prefix_seconds\":{prefix_seconds:.9}}}\n",
                ids.len() - prefix,
                self.vocab,
                self.layers.len(),
                self.prefill.capacity,
                if step_prefix { "step" } else { "prefill" }
            ),
        )?;
        eprintln!(
            "eval_rows={} eval_s={:.3}",
            ids.len() - prefix,
            start.elapsed().as_secs_f64()
        );
        Ok(())
    }
    /// Long-context bench only (`--bench-tile`): after a prefill of `prefix`
    /// tokens, repeat the sparse layers' latent KV and index pools `times`
    /// times so decode runs at `prefix * times` (KDA states stay the prefix's).
    fn tile_history(&mut self, prefix: usize, times: usize) -> Result<()> {
        if !prefix.is_multiple_of(4) || self.position != prefix || prefix * times > self.context {
            return Err(Error::InvalidPlan("GLM bench tile".into()));
        }
        for layer in &mut self.layers {
            layer.tile_history(prefix, times)?;
        }
        self.position = prefix * times;
        self.device_ready = false;
        Ok(())
    }
    pub fn cache_counts(&self) -> (u64, u64, u64) {
        (
            self.experts.hits,
            self.experts.routes,
            self.experts.disk_reads,
        )
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let mut request_start = Instant::now();
    let mut resize_at: Vec<(usize, u64)> = Vec::new();
    let mut options = Options::default();
    let mut source = PathBuf::from(
        "/mnt/NVME4TB/darask-inference-engine/local/hayate/src/GLM-5.3-Flash-Uncensored-FP8",
    );
    let mut hyt = PathBuf::from("/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53-fp8");
    let mut prompt = None;
    let mut bench_text: Option<PathBuf> = None;
    let mut bench_split: Option<(usize, usize)> = None;
    let mut bench_tile = 1usize;
    let mut max_tokens = 32usize;
    let mut dump = None;
    let mut raw = false;
    let mut warmup = 0usize;
    let mut save_frequencies = None;
    let mut save_routes: Option<PathBuf> = None;
    let mut profile = false;
    let mut reference = false;
    let mut no_graphs = false;
    let mut spec_n = 0usize;
    let mut profile_spec = false;
    let mut spec_adaptive = true;
    let mut spec_validate = None;
    let mut ignore_eos = false;
    let mut output_ids = None;
    let mut eval_ids = None;
    let mut eval_prefix = 64usize;
    let mut eval_step = false;
    let mut out = None;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--cpu-spin" {
            options.cpu_spin = true;
            i += 1;
            continue;
        }
        if matches!(
            flag,
            "--profile"
                | "--decode-reference"
                | "--no-graphs"
                | "--expert-prefetch"
                | "--profile-spec"
                | "--spec-no-adaptive"
                | "--ignore-eos"
                | "--maintenance-async"
                | "--prefill-legacy"
                | "--spec-reserve-mtp"
        ) {
            match flag {
                "--profile" => profile = true,
                "--spec-reserve-mtp" => options.mtp_reserve_only = true,
                "--prefill-legacy" => options.prefill_legacy = true,
                "--profile-spec" => profile_spec = true,
                "--spec-no-adaptive" => spec_adaptive = false,
                "--ignore-eos" => ignore_eos = true,
                "--decode-reference" => reference = true,
                "--expert-prefetch" => options.expert_prefetch = true,
                "--maintenance-async" => options.maintenance_async = true,
                _ => no_graphs = true,
            }
            i += 1;
            continue;
        }
        if flag == "--allow-disk" {
            options.allow_disk = true;
            i += 1;
            continue;
        }
        if flag == "--defer-cpu-experts" {
            options.defer_cpu_below = f32::INFINITY;
            i += 1;
            continue;
        }
        if flag == "--raw" {
            raw = true;
            i += 1;
            continue;
        }
        let value = args
            .get(i + 1)
            .ok_or_else(|| Error::InvalidPlan(format!("missing value for {flag}")))?;
        let number = || {
            value
                .parse::<usize>()
                .map_err(|_| Error::InvalidPlan(format!("invalid {flag}")))
        };
        match flag {
            "--decode-host" => {
                (options.decode_async, options.decode_device) = match value.as_str() {
                    "async" => (true, false),
                    "sync" => (false, false),
                    "device" => (false, true),
                    _ => {
                        return Err(Error::InvalidPlan(
                            "GLM decode-host=async|sync|device".into(),
                        ));
                    }
                }
            }
            "--prompt" => prompt = Some(value.clone()),
            "--lora" => options.loras.push(super::lora::Spec::parse(value)?),
            "--bench-text" => bench_text = Some(PathBuf::from(value)),
            "--bench-split" => {
                bench_split = value
                    .split_once(',')
                    .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
                    .filter(|&(a, b): &(usize, usize)| a > 0 && b > 0);
                if bench_split.is_none() {
                    return Err(Error::InvalidPlan("--bench-split PREFIX,TURN".into()));
                }
            }
            "--spec-draft-n" => spec_n = number()?,
            "--bench-tile" => bench_tile = number()?.max(1),
            "--mtp-experts" => {
                options.mtp_cpu_experts = match value.as_str() {
                    "gpu" => false,
                    "cpu" => true,
                    _ => return Err(Error::InvalidPlan("GLM mtp-experts=gpu|cpu".into())),
                };
            }
            "--spec-validate" => spec_validate = Some(number()?),
            "--output-ids" => output_ids = Some(PathBuf::from(value)),
            "--max-tokens" => max_tokens = number()?,
            "--layers" => options.layers = number()?,
            "--context" => options.context = number()?,
            "--prefill-chunk" => options.prefill_chunk = number()?,
            "--prefill-experts" => {
                options.prefill_all_experts = match value.as_str() {
                    "previous" => false,
                    "all" => true,
                    _ => {
                        return Err(Error::InvalidPlan(
                            "GLM prefill-experts=previous|all".into(),
                        ));
                    }
                }
            }
            "--warmup" => warmup = number()?,
            "--expert-readers" => options.expert_readers = number()?,
            "--cpu-threads" => options.cpu_threads = number()?,
            "--maintenance-window" => options.maintenance_window = number()?,
            "--maintenance-swaps" => options.maintenance_swaps = number()?,
            "--maintenance-hysteresis" => options.maintenance_hysteresis = number()?,
            "--host-core" => {
                options.host_core = if value == "none" {
                    None
                } else {
                    Some(number()?)
                };
            }
            "--pcie-take" => {
                options.pcie_take = if value == "off" {
                    None
                } else {
                    let v: Vec<u8> = value
                        .split(',')
                        .map(|t| t.parse::<u8>())
                        .collect::<std::result::Result<_, _>>()
                        .map_err(|_| Error::InvalidPlan("invalid pcie-take".into()))?;
                    let take: [u8; 9] = v.try_into().map_err(|_| {
                        Error::InvalidPlan("pcie-take needs 9 values (n = 0..8) or off".into())
                    })?;
                    pcie_take(take)?;
                    Some(take)
                }
            }
            "--pcie-prefetch" => {
                options.pcie_prefetch = number()? as u32;
                if options.pcie_prefetch > 4 {
                    return Err(Error::InvalidPlan("pcie-prefetch must be 0..4".into()));
                }
            }
            "--pcie-frac" => {
                options.pcie_frac = value
                    .parse()
                    .map_err(|_| Error::InvalidPlan("invalid pcie-frac".into()))?
            }
            "--maintenance-interval" => options.maintenance_interval = number()?,
            "--vram-gib" => {
                options.vram_budget = Some(
                    value
                        .parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite() && *v > 0.)
                        .map(|v| (v * (1u64 << 30) as f64) as u64)
                        .ok_or_else(|| Error::InvalidPlan("invalid vram-gib".into()))?,
                )
            }
            "--resize-at" => {
                let (token, gib) = value
                    .split_once(':')
                    .and_then(|(t, g)| Some((t.parse::<usize>().ok()?, g.parse::<f64>().ok()?)))
                    .filter(|(_, g)| g.is_finite() && *g > 0.)
                    .ok_or_else(|| Error::InvalidPlan("--resize-at TOKEN:GIB".into()))?;
                resize_at.push((token, (gib * (1u64 << 30) as f64) as u64));
            }
            "--save-frequencies" => save_frequencies = Some(PathBuf::from(value)),
            "--save-routes" => save_routes = Some(PathBuf::from(value)),
            "--defer-below" => {
                options.defer_cpu_below = value
                    .parse()
                    .map_err(|_| Error::InvalidPlan("invalid defer-below".into()))?
            }
            "--context-placement" => {
                options.placement_tail = value
                    .parse()
                    .map_err(|_| Error::InvalidPlan("invalid context-placement".into()))?
            }
            "--placement-beta" => {
                options.placement_beta = value
                    .parse()
                    .map_err(|_| Error::InvalidPlan("invalid placement-beta".into()))?
            }
            "--placement-probe" => {
                options.placement_probe = value
                    .parse()
                    .map_err(|_| Error::InvalidPlan("invalid placement-probe".into()))?
            }
            "--placement-swaps" => {
                options.placement_swaps = value
                    .parse()
                    .map_err(|_| Error::InvalidPlan("invalid placement-swaps".into()))?
            }
            "--skip-cpu-below" => {
                options.skip_cpu_below = value
                    .parse()
                    .map_err(|_| Error::InvalidPlan("invalid skip-cpu-below".into()))?
            }
            "--host-gb" => {
                options.host_bytes = Some(
                    value
                        .parse::<u64>()
                        .ok()
                        .and_then(|n| n.checked_mul(1_000_000_000))
                        .ok_or_else(|| Error::InvalidPlan("invalid host-gb".into()))?,
                )
            }
            "--model-dir" => source = PathBuf::from(value),
            "--hyt-dir" => hyt = PathBuf::from(value),
            "--dump-logits" => dump = Some(PathBuf::from(value)),
            "--eval-ids" => eval_ids = Some(PathBuf::from(value)),
            "--eval-prefix" => eval_prefix = number()?,
            "--eval-mode" => {
                eval_step = match value.as_str() {
                    "step" => true,
                    "prefill" => false,
                    _ => return Err(Error::InvalidPlan("GLM eval-mode=prefill|step".into())),
                }
            }
            "--out" => out = Some(PathBuf::from(value)),
            "--frequencies" => {
                let bytes = std::fs::read(value)?;
                if !bytes.len().is_multiple_of(4) {
                    return Err(Error::Weights(
                        "GLM frequencies require F32 LE layer3.. expert order".into(),
                    ));
                }
                options.frequencies = Some(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect(),
                );
            }
            _ => return Err(Error::InvalidPlan(format!("unknown GLM argument {flag}"))),
        }
        i += 2;
    }
    if spec_n > 3
        || (profile_spec && spec_n == 0)
        || (spec_n > 0 && (reference || profile || eval_ids.is_some()))
    {
        return Err(Error::InvalidPlan(
            "GLM speculation needs n=1..3 without reference/profile/eval".into(),
        ));
    }
    if options.mtp_reserve_only && (spec_n != 0 || spec_validate.is_some()) {
        return Err(Error::InvalidPlan(
            "GLM spec-reserve-mtp requires n=0 without spec-validate".into(),
        ));
    }
    options.mtp = spec_n > 0 || spec_validate.is_some() || options.mtp_reserve_only;
    if spec_validate.is_some() {
        options.maintenance_interval = usize::MAX;
    }
    if let Some(path) = eval_ids {
        if prompt.is_some() || raw || dump.is_some() || save_frequencies.is_some() {
            return Err(Error::InvalidPlan(
                "GLM eval cannot use generation/dump/frequency collection flags".into(),
            ));
        }
        let out = out.ok_or_else(|| Error::InvalidPlan("--eval-ids requires --out".into()))?;
        if out.exists() {
            return Err(Error::InvalidPlan(
                "GLM eval --out must be a new directory".into(),
            ));
        }
        let ids = super::eval::read_ids(&path, eval_prefix, options.context)?;
        let device = Device::select()?;
        let mut executor = Executor::load(&device, &hyt, options)?;
        executor.decode_reference(reference);
        if no_graphs {
            executor.disable_graphs();
        }
        for _ in 0..warmup {
            executor.prefill(&ids[..eval_prefix])?;
            executor.reset()?;
        }
        if profile {
            executor.enable_profile()?;
        }
        return if eval_step {
            executor.evaluate_mode(&ids, eval_prefix, &out, true)
        } else {
            executor.evaluate(&ids, eval_prefix, &out)
        };
    }
    if out.is_some()
        || args
            .iter()
            .any(|s| s == "--eval-prefix" || s == "--eval-mode")
    {
        return Err(Error::InvalidPlan(
            "--out/--eval-prefix require --eval-ids".into(),
        ));
    }
    if let Some(path) = bench_text {
        // Agent-turn benchmark: cold prefill of PREFIX tokens (the reused
        // context), then a TURN-token prefill on top (its time is the TTFT of
        // a turn with PREFIX reused), then decode at that context length.
        let (prefix, turn) = bench_split
            .ok_or_else(|| Error::InvalidPlan("--bench-text needs --bench-split".into()))?;
        let tokenizer = Tokenizer::open(source.join("tokenizer.json"))?;
        let text = std::fs::read_to_string(&path)?;
        let mut ids = tokenizer.encode(&text)?;
        if ids.len() < prefix + turn {
            return Err(Error::InvalidShape(format!(
                "bench text has {} tokens, needs {}",
                ids.len(),
                prefix + turn
            )));
        }
        ids.truncate(prefix + turn);
        if prefix * bench_tile + turn + max_tokens > options.context {
            return Err(Error::InvalidShape("bench exceeds --context".into()));
        }
        let device = Device::select()?;
        let mut executor = Executor::load(&device, &hyt, options)?;
        if no_graphs {
            executor.disable_graphs();
        }
        if profile {
            executor.enable_profile()?;
        }
        if save_routes.is_some() {
            executor.experts.trace = Some(Vec::new());
        }
        let trace_len = |e: &Executor| e.experts.trace.as_ref().map_or(0, Vec::len);
        let start = Instant::now();
        executor.prefill(&ids[..prefix])?;
        let cold = start.elapsed().as_secs_f64();
        let trace_prefix = trace_len(&executor);
        if bench_tile > 1 {
            executor.tile_history(prefix, bench_tile)?;
        }
        let start = Instant::now();
        let mut next = executor.prefill(&ids[prefix..])?;
        let ttft = start.elapsed().as_secs_f64();
        // --resize-at in the bench: every budget applies after the turn prefill.
        for &(_, bytes) in &resize_at {
            executor.set_vram_budget(bytes)?;
        }
        if let Some(device) = executor.device_decode.as_mut() {
            device.stats = Default::default();
        }
        let (cpu_before, routes_before) =
            (executor.experts.transfer_stats, executor.cache_counts());
        let trace_turn = trace_len(&executor);
        let mut latencies = Vec::new();
        for _ in 1..max_tokens {
            let start = Instant::now();
            next = executor.step(next)?;
            latencies.push(start.elapsed().as_secs_f64());
        }
        let decode: f64 = latencies.iter().sum();
        latencies.sort_by(f64::total_cmp);
        eprintln!(
            "glm_bench prefix={prefix} tile={bench_tile} turn={turn} cold_prefill_s={cold:.3} cold_prefill_tok_s={:.1} turn_ttft_s={ttft:.3} turn_prefill_tok_s={:.1} decode_context={} decode_tok_s={:.3} ITL_p50_ms={:.1} ITL_p95_ms={:.1} decode_cpu_gb_s={:.2} decode_cpu_ms_per_token={:.3} decode_gpu_hit_rate={:.4}",
            prefix as f64 / cold,
            turn as f64 / ttft,
            prefix * bench_tile + turn,
            latencies.len() as f64 / decode.max(1e-9),
            latencies.get(latencies.len() / 2).copied().unwrap_or(0.) * 1000.,
            latencies
                .get(latencies.len() * 95 / 100)
                .copied()
                .unwrap_or(0.)
                * 1000.,
            (executor.experts.transfer_stats.cpu_bytes - cpu_before.cpu_bytes) as f64
                / ((executor.experts.transfer_stats.cpu_ms - cpu_before.cpu_ms).max(1e-9) / 1000.)
                / 1e9,
            (executor.experts.transfer_stats.cpu_ms - cpu_before.cpu_ms)
                / latencies.len().max(1) as f64,
            {
                let (hits, routes, _) = executor.cache_counts();
                (hits - routes_before.0) as f64 / (routes - routes_before.1).max(1) as f64
            },
        );
        executor.device_report();
        eprintln!("glm_bench_memory {:?}", executor.memory);
        if let (Some(path), Some(trace)) = (&save_routes, &executor.experts.trace) {
            // Word offsets: the cold prefix ends at trace_prefix, the turn
            // prefill at trace_turn; decode follows.
            eprintln!("glm_bench_trace prefix_words={trace_prefix} turn_words={trace_turn}");
            std::fs::write(
                path,
                trace
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
        }
        return Ok(());
    }
    let prompt = prompt.ok_or_else(|| Error::InvalidPlan("GLM requires --prompt".into()))?;
    if max_tokens == 0 {
        return Err(Error::InvalidShape("max-tokens must be positive".into()));
    }
    let tokenizer = Tokenizer::open(source.join("tokenizer.json"))?;
    let ids = if raw {
        tokenizer.encode(&prompt)?
    } else {
        tokenizer.chat(&prompt, &source.join("chat_template.jinja"))?
    };
    if ids
        .len()
        .checked_add(max_tokens - 1)
        .is_none_or(|n| n > options.context)
    {
        return Err(Error::InvalidShape(
            "prompt + generation exceeds context".into(),
        ));
    }
    let eos_config = super::json::parse(&std::fs::read_to_string(
        source.join("generation_config.json"),
    )?)?;
    let eos: Vec<_> = eos_config
        .get("eos_token_id")?
        .array()?
        .iter()
        .map(super::json::Value::number)
        .collect::<Result<_>>()?;
    let device = Device::select()?;
    let cold = Instant::now();
    let mut executor = Executor::load(&device, &hyt, options)?;
    let load_seconds = cold.elapsed().as_secs_f64();
    executor.decode_reference(reference);
    if no_graphs {
        executor.disable_graphs();
    }
    if let Some(tokens) = spec_validate {
        return executor.validate_speculation(&ids, tokens);
    }
    let warmup_start = Instant::now();
    for _ in 0..warmup {
        let mut next = executor.prefill(&ids)?;
        let mut emitted = 1;
        while emitted < max_tokens && (ignore_eos || !eos.contains(&next)) {
            if spec_n > 0 && executor.context - executor.position > spec_n {
                let output = executor.spec_step(next, spec_n, None, spec_n, false)?;
                emitted += output.len();
                next = *output.last().unwrap();
            } else {
                next = executor.step(next)?;
                emitted += 1;
            }
        }
        executor.reset()?;
    }
    let warmup_seconds = warmup_start.elapsed().as_secs_f64();
    if warmup > 0 {
        request_start = Instant::now();
    }
    executor.experts.main_hits = 0;
    executor.experts.main_routes = 0;
    executor.experts.hits = 0;
    executor.experts.routes = 0;
    executor.experts.host_hits = 0;
    executor.experts.disk_reads = 0;
    if save_routes.is_some() {
        executor.experts.trace = Some(Vec::new());
    }
    if profile {
        executor.enable_profile()?;
    }
    let prefill_before = executor.cache_counts();
    let start = Instant::now();
    let mut next = executor.prefill(&ids)?;
    let prefill_seconds = start.elapsed().as_secs_f64();
    let prefill_counts = executor.cache_counts();
    let prefill_host_hits = executor.experts.host_hits;
    let prefill_main = (executor.experts.main_hits, executor.experts.main_routes);
    let (main_gpu_bytes, main_placement_hash) = executor.experts.main_placement();
    let mtp_expert_gpu_bytes = if executor.mtp.is_some() {
        executor.experts.layer_gpu_bytes(45)
    } else {
        0
    };
    let cpu_before = executor.experts.transfer_stats;
    let maintenance_worker_before = executor.experts.d2h_wait_ms;
    let ttft_seconds = request_start.elapsed().as_secs_f64();
    if profile {
        executor.enable_profile()?;
    }
    eprintln!(
        "decode_reference={reference} decode_graphs={} profile={profile} expert_prefetch={} maintenance_interval={}",
        !reference && !no_graphs && !profile,
        executor.expert_prefetch,
        executor.experts.maintenance_interval(),
    );
    executor.host_syncs = 0;
    executor.decode_steps = 0;
    if let Some(device) = executor.device_decode.as_mut() {
        device.stats = Default::default();
    }
    let mut generated = vec![next];
    let mut latencies = Vec::new();
    if let Some(path) = &dump {
        dump_logits(path, 0, &executor)?;
    }
    let mut spec_metrics = super::speculation::Metrics::default();
    let mut resize_ms = 0.;
    while generated.len() < max_tokens && (ignore_eos || !eos.contains(&next)) {
        if let Some(at) = resize_at.iter().position(|&(t, _)| t <= generated.len()) {
            let (_, bytes) = resize_at.remove(at);
            let resize = Instant::now();
            executor.set_vram_budget(bytes)?;
            resize_ms += resize.elapsed().as_secs_f64() * 1000.;
            let (free, total) = device.memory_info()?;
            eprintln!(
                "glm_resize_at token={} vram_bytes={} device_used={}",
                generated.len(),
                executor.vram_bytes(),
                total - free
            );
        }
        let start = Instant::now();
        let n = spec_n
            .min(max_tokens - generated.len() - 1)
            .min(executor.context - executor.position - 1);
        if n > 0 && !spec_metrics.disabled && (!spec_adaptive || spec_metrics.baseline_tokens >= 8)
        {
            let mut output = executor.spec_step(next, n, None, n, profile_spec)?;
            let accepted = output.len() - 1;
            if !ignore_eos && let Some(end) = output.iter().position(|t| eos.contains(t)) {
                output.truncate(end + 1);
            }
            let seconds = start.elapsed().as_secs_f64();
            spec_metrics.record(n, accepted, output.len(), seconds)?;
            next = *output.last().unwrap();
            latencies.extend(std::iter::repeat_n(
                seconds / output.len() as f64,
                output.len(),
            ));
            if let Some(path) = &dump {
                executor.dump_spec_logits(path, generated.len(), output.len())?;
            }
            generated.extend(output);
            if spec_adaptive && spec_metrics.should_disable() {
                spec_metrics.disabled = true;
            }
        } else {
            // Keep draft KV current during the eight baseline decode samples.
            // Baseline comparison excludes this initialization cost.
            if spec_n > 0 && !spec_metrics.disabled {
                executor.draft_one(next, false)?;
                if !executor.decode_async {
                    executor.stream.synchronize()?;
                }
            }
            let plain_start = Instant::now();
            next = executor.step(next)?;
            if spec_n > 0 && spec_metrics.baseline_tokens < 8 {
                spec_metrics.baseline_tokens += 1;
                spec_metrics.baseline_seconds += plain_start.elapsed().as_secs_f64();
            }
            latencies.push(start.elapsed().as_secs_f64());
            generated.push(next);
            if let Some(path) = &dump {
                dump_logits(path, generated.len() - 1, &executor)?;
            }
        }
    }
    if resize_ms > 0. {
        eprintln!("glm_resize_total_ms={resize_ms:.3} (excluded from ITL)");
    }
    eprintln!(
        "glm_spec_summary n={spec_n} steps={} proposed={} accepted={} acceptance={:.6} emitted={} seconds={:.6} disabled={} MTP_added_VRAM_bytes={} MTP_expert_GPU_bytes={} peak_VRAM_bytes={}",
        spec_metrics.steps,
        spec_metrics.proposed,
        spec_metrics.accepted,
        spec_metrics.accepted as f64 / spec_metrics.proposed.max(1) as f64,
        spec_metrics.emitted,
        spec_metrics.seconds,
        spec_metrics.disabled,
        executor.memory.mtp_added,
        mtp_expert_gpu_bytes,
        executor.memory.vram_peak,
    );
    println!("{}", tokenizer.decode(&generated)?);
    eprintln!(
        "glm_decode_sync host_mode={} steps={} host_syncs={} host_syncs_per_step={:.6}",
        if executor.device_enabled() {
            "device"
        } else if executor.decode_async {
            "async"
        } else {
            "sync"
        },
        executor.decode_steps,
        executor.host_syncs,
        executor.host_syncs as f64 / executor.decode_steps.max(1) as f64
    );
    executor.device_report();
    let decode_seconds: f64 = latencies.iter().sum();
    latencies.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        latencies
            .get(((latencies.len().saturating_sub(1)) as f64 * p).round() as usize)
            .copied()
            .unwrap_or(0.)
            * 1000.
    };
    let drain_start = Instant::now();
    executor.experts.drain()?;
    eprintln!(
        "expert_final_drain_ms={:.6}",
        drain_start.elapsed().as_secs_f64() * 1000.
    );
    let (hits, routes, disk_reads) = executor.cache_counts();
    eprintln!(
        "gpu_host_hit_rate={:.6}",
        if routes == 0 {
            0.
        } else {
            (hits + executor.experts.host_hits) as f64 / routes as f64
        }
    );
    eprintln!(
        "expert_quant={} down_fp8={} expert_formats_nvfp4_down_fp8={:?} hadamard_blocks={:?} expert_image_bytes={} warmup={warmup} warmup_s={warmup_seconds:.3} prefill_chunk={}",
        if executor
            .experts
            .formats
            .iter()
            .any(|f| *f != executor.experts.formats[0])
        {
            "mixed"
        } else if executor.experts.formats[0].0 {
            "nvfp4"
        } else {
            "block_fp8"
        },
        executor.experts.formats.iter().all(|f| f.1),
        executor.experts.formats,
        executor
            .layers
            .iter()
            .skip(3)
            .map(|l| l.hadamard_block)
            .collect::<Vec<_>>(),
        executor.experts.image_bytes,
        executor.prefill.capacity
    );
    eprintln!(
        "prefill_cache_hits={}/{} prefill_host_hits={prefill_host_hits} prefill_disk_reads={} decode_cache_hits={}/{} decode_host_hits={} decode_disk_reads={}",
        prefill_counts.0 - prefill_before.0,
        prefill_counts.1 - prefill_before.1,
        prefill_counts.2 - prefill_before.2,
        hits - prefill_counts.0,
        routes - prefill_counts.1,
        executor.experts.host_hits - prefill_host_hits,
        disk_reads - prefill_counts.2
    );
    eprintln!(
        "glm_mtp_cache main_decode_hits={} main_decode_routes={} main_expert_GPU_bytes={} mtp_expert_GPU_bytes={} main_placement_hash={main_placement_hash}",
        executor.experts.main_hits - prefill_main.0,
        executor.experts.main_routes - prefill_main.1,
        main_gpu_bytes,
        mtp_expert_gpu_bytes,
    );
    let decode_routes = routes - prefill_counts.1;
    let decode_hits = hits - prefill_counts.0;
    let decode_host_hits = executor.experts.host_hits - prefill_host_hits;

    let cpu = executor.experts.transfer_stats;
    let cpu_ms = cpu.cpu_ms - cpu_before.cpu_ms;
    let cpu_bytes = cpu.cpu_bytes - cpu_before.cpu_bytes;
    eprintln!(
        "decode_cpu_experts={} decode_cpu_bytes={cpu_bytes} decode_cpu_ms={cpu_ms:.6} decode_cpu_gb_s={:.6} decode_cpu_ms_per_token={:.6} decode_cpu_worker_ms={:.6} decode_cpu_dispatch_ms={:.6} decode_maintenance_bytes={} decode_maintenance_worker_ms={:.6}",
        cpu.cpu_experts - cpu_before.cpu_experts,
        if cpu_ms > 0. {
            cpu_bytes as f64 / cpu_ms / 1e6
        } else {
            0.
        },
        cpu_ms / latencies.len().max(1) as f64,
        cpu.cpu_worker_ms - cpu_before.cpu_worker_ms,
        cpu.cpu_dispatch_ms - cpu_before.cpu_dispatch_ms,
        cpu.maintenance_bytes - cpu_before.maintenance_bytes,
        executor.experts.d2h_wait_ms - maintenance_worker_before,
    );
    eprintln!(
        "decode_gpu_hit_rate={:.6} decode_gpu_host_hit_rate={:.6} frequencies_loaded={}",
        decode_hits as f64 / decode_routes.max(1) as f64,
        (decode_hits + decode_host_hits) as f64 / decode_routes.max(1) as f64,
        args.iter().any(|s| s == "--frequencies")
    );
    if let Some(path) = save_frequencies {
        executor.save_frequencies(&path)?;
    }
    if let (Some(path), Some(trace)) = (save_routes, &executor.experts.trace) {
        std::fs::write(
            path,
            trace
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
    }
    eprintln!(
        "load_s={load_seconds:.3} TTFT_s={ttft_seconds:.3} prefill_s={prefill_seconds:.3} prefill_tok_s={:.3} decode_tok_s={:.3} decode_steps={} ITL_p50_ms={:.3} ITL_p95_ms={:.3} cache_hits={hits}/{routes} cache_hit_rate={:.6} disk_reads={disk_reads} memory={:?}",
        ids.len() as f64 / prefill_seconds,
        if decode_seconds == 0. {
            0.
        } else {
            latencies.len() as f64 / decode_seconds
        },
        latencies.len(),
        percentile(0.5),
        percentile(0.95),
        if routes == 0 {
            0.
        } else {
            hits as f64 / routes as f64
        },
        executor.memory
    );
    if let Some(path) = output_ids {
        std::fs::write(
            path,
            format!("{{\"input\":{ids:?},\"generated\":{generated:?}}}\n"),
        )?;
    }
    if let Some(path) = dump {
        std::fs::write(
            path.with_extension("ids.json"),
            format!("{{\"input\":{ids:?},\"generated\":{generated:?}}}\n"),
        )?;
    }
    Ok(())
}
fn dump_logits(prefix: &Path, step: usize, executor: &Executor<'_>) -> Result<()> {
    let values = executor.logits()?;
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|v| (*v as f32).to_le_bytes())
        .collect();
    std::fs::write(format!("{}.step{step}.bin", prefix.display()), bytes)?;
    Ok(())
}
