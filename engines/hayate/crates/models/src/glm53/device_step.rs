//! `--decode-host device` (stage 19): one decode token is one CUDA graph on
//! the executor stream. Routing, the expert pointer table, the CPU doorbells
//! (mapped mailboxes, spin-wait kernel) and greedy sampling run on the GPU;
//! the host writes {token, position, seq}, hands the doorbell worker one job
//! per MoE layer, launches the graph and synchronizes once.
use super::*;
use crate::glm53::gpu::{Defer, DeviceRefs};
use hayate_kernels::runtime::PinnedHost;
use hayate_kernels::{graph::Graph, runtime::MappedHost};

/// GPU wait for one CPU doorbell before the token fails (worker stuck).
const DOORBELL_LIMIT_NS: u64 = 120_000_000_000;
/// Events per layer in profile mode (phase boundaries, see `profile_token`).
const MARKS: usize = 8;

#[derive(Debug)]
pub(in crate::glm53) struct DeviceDecode {
    ctl: DeviceBuffer,
    host_ctl: MappedHost,
    host_out: MappedHost,
    table: DeviceBuffer,
    table_host: Vec<u64>,
    table_next: Vec<u64>,
    table_stage: PinnedHost,
    /// [42][288] ExpertFormat (8 bytes) of LOWBIT checkpoints; null elsewhere.
    formats: Option<DeviceBuffer>,
    misses: DeviceBuffer,
    counter: DeviceBuffer,
    logits: DeviceBuffer,
    greedy_values: DeviceBuffer,
    greedy_ids: DeviceBuffer,
    graph: Option<Graph>,
    seq: u32,
    eager_tokens: usize,
    events: Option<Vec<Event>>,
    /// Expert deferral below this routing weight (approximation, 0 off;
    /// `--defer-cpu-experts` = all, `--defer-below W`).
    defer_below: f32,
    /// CPU experts below this routing weight are dropped (approximation, 0 off).
    skip_below: f32,
    /// The PCIe lane (EXL3.md §6): RAM-tier images copied into scratch slots and computed on the GPU.
    pcie: Option<Pcie>,
    pub(super) stats: DeviceStats,
}
/// The PCIe lane's device state: [42][288] RAM-tier image pointers (uploaded when they change), [8] scratch
/// slot pointers (4 slots), [8] fetch sources the router writes, and the CPU share table. Allocated before the
/// expert budget is planned (`Pcie::new`), attached to the device decode later.
#[derive(Debug)]
pub(super) struct Pcie {
    host_table: DeviceBuffer,
    host: Vec<u64>,
    stage: PinnedHost,
    _slots: Vec<DeviceBuffer>,
    scratch: DeviceBuffer,
    fetch: DeviceBuffer,
    take: u64,
    lane: crate::glm53::gpu::LaneSync,
    /// Prefetch: `prefetch` slots per layer parity (two sets), [2][8] slot pointers and [2][8] sources (the
    /// predicted host images, null unused), and u32 [4] counters (see `device.cu` Pcie).
    prefetch: u32,
    _pre_slots: Vec<DeviceBuffer>,
    pre_scratch: DeviceBuffer,
    pre_src: DeviceBuffer,
    pre_meta: DeviceBuffer,
    stats: DeviceBuffer,
}
impl Pcie {
    /// Four scratch slots of `slot_bytes` (the largest expert image) and the tables; `bytes()` is their VRAM.
    pub(super) fn new(
        device: &Device,
        layers: usize,
        slot_bytes: usize,
        take: [u8; 9],
        prefetch: u32,
    ) -> Result<Self> {
        let take = super::pcie_take(take)?;
        let stream = device.stream()?;
        let slots: Vec<DeviceBuffer> = (0..4)
            .map(|_| device.alloc(slot_bytes.next_multiple_of(256)))
            .collect::<Result<_>>()?;
        let mut pointers: Vec<u8> = Vec::with_capacity(64);
        for i in 0..8 {
            pointers.extend((slots[i % 4].as_ptr() as u64).to_le_bytes());
        }
        let mut scratch = device.alloc(64)?;
        stream.copy_h2d(&mut scratch, &pointers)?;
        let mut fetch = device.alloc(64)?;
        stream.copy_h2d(&mut fetch, &[0u8; 64])?;
        let moe = layers - 3;
        let mut host_table = device.alloc(moe * EXPERTS * 8)?;
        stream.copy_h2d(&mut host_table, &vec![0u8; moe * EXPERTS * 8])?;
        let lane = crate::glm53::gpu::LaneSync {
            events: (0..layers)
                .map(|_| Ok((stream.event()?, stream.event()?)))
                .collect::<Result<_>>()?,
            stream: device.stream()?,
        };
        let pre_slots: Vec<DeviceBuffer> = (0..2 * prefetch as usize)
            .map(|_| device.alloc(slot_bytes.next_multiple_of(256)))
            .collect::<Result<_>>()?;
        let mut pointers: Vec<u8> = Vec::with_capacity(128);
        for set in 0..2 {
            for s in 0..8 {
                let slot = (s < prefetch as usize).then(|| &pre_slots[set * prefetch as usize + s]);
                pointers.extend(slot.map_or(0, |b| b.as_ptr() as u64).to_le_bytes());
            }
        }
        let mut pre_scratch = device.alloc(128)?;
        stream.copy_h2d(&mut pre_scratch, &pointers)?;
        let mut pre_src = device.alloc(128)?;
        stream.copy_h2d(&mut pre_src, &[0u8; 128])?;
        let mut pre_meta = device.alloc(crate::glm53::gpu::PRE_META_BYTES)?;
        stream.copy_h2d(&mut pre_meta, &[0u8; crate::glm53::gpu::PRE_META_BYTES])?;
        let mut stats = device.alloc(16)?;
        stream.copy_h2d(&mut stats, &[0u8; 16])?;
        Ok(Self {
            host_table,
            host: vec![0; moe * EXPERTS],
            stage: device.alloc_pinned(moe * EXPERTS * 8)?,
            _slots: slots,
            scratch,
            fetch,
            take,
            lane,
            prefetch,
            _pre_slots: pre_slots,
            pre_scratch,
            pre_src,
            pre_meta,
            stats,
        })
    }
    pub(super) fn bytes(&self) -> usize {
        self._slots
            .iter()
            .chain(&self._pre_slots)
            .map(DeviceBuffer::bytes)
            .sum::<usize>()
            + 128
            + 272
            + crate::glm53::gpu::PRE_META_BYTES
            + self.host.len() * 8
    }
}

/// Per-token accumulators (milliseconds; profile phases only in profile mode).
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct DeviceStats {
    pub tokens: u64,
    pub prep_ms: f64,
    pub retire_ms: f64,
    pub maintenance_ms: f64,
    pub gpu_ms: f64,
    pub attention_ms: f64,
    pub router_ms: f64,
    pub experts_ms: f64,
    pub cpu_wait_ms: f64,
    pub finish_ms: f64,
    pub head_ms: f64,
    /// Sparse layers: top-512 selection and absorbed MLA (inside attention_ms).
    pub select_ms: f64,
    pub mla_ms: f64,
    pub profiled: u64,
}

impl DeviceDecode {
    pub(super) fn new(
        device: &Device,
        layers: usize,
        defer_below: f32,
        skip_below: f32,
    ) -> Result<Self> {
        let moe = layers - 3;
        let zero = |stream: &Stream, buffer: &mut DeviceBuffer| -> Result<()> {
            check(
                unsafe {
                    g::hayate_glm_copy(
                        buffer.as_mut_ptr(),
                        std::ptr::null(),
                        buffer.bytes(),
                        stream.as_raw(),
                    )
                },
                "glm device zero",
            )
        };
        let stream = device.stream()?;
        let mut ctl = device.alloc(32)?;
        let mut misses = device.alloc(1024)?;
        let mut counter = device.alloc(16)?;
        let mut table = device.alloc(moe * EXPERTS * 8)?;
        for buffer in [&mut ctl, &mut misses, &mut counter, &mut table] {
            zero(&stream, buffer)?;
        }
        stream.synchronize()?;
        Ok(Self {
            ctl,
            host_ctl: device.alloc_mapped(32)?,
            host_out: device.alloc_mapped(32)?,
            table,
            table_host: vec![0; moe * EXPERTS],
            table_next: vec![0; moe * EXPERTS],
            table_stage: device.alloc_pinned(moe * EXPERTS * 8)?,
            formats: None,
            misses,
            counter,
            logits: device.alloc(24 * 4)?,
            greedy_values: device.alloc(128 * 4)?,
            greedy_ids: device.alloc(128 * 4)?,
            graph: None,
            seq: 0,
            eager_tokens: 0,
            events: None,
            defer_below,
            skip_below,
            pcie: None,
            stats: DeviceStats::default(),
        })
    }
    /// Turns the PCIe lane on.
    pub(super) fn set_pcie(&mut self, pcie: Pcie) {
        self.pcie = Some(pcie);
    }
    /// Uploads the per-expert matrix formats of a checkpoint with LOWBIT experts.
    pub(super) fn set_formats(
        &mut self,
        device: &Device,
        formats: &[hayate_kernels::glm53::lowbit::ExpertFormat],
    ) -> Result<()> {
        let mut bytes = Vec::with_capacity(formats.len() * 8);
        for f in formats {
            for m in [f.gate_up, f.down] {
                bytes.extend([m.format, m.grid, m.group, m.reserved]);
            }
        }
        let mut buffer = device.alloc(bytes.len())?;
        device.stream()?.copy_h2d(&mut buffer, &bytes)?;
        self.formats = Some(buffer);
        Ok(())
    }
    fn refs(&self) -> DeviceRefs {
        DeviceRefs {
            ctl: self.ctl.as_ptr().cast_mut(),
            table: self.table.as_ptr().cast_mut(),
            formats: self
                .formats
                .as_ref()
                .map_or(std::ptr::null_mut(), |f| f.as_ptr().cast_mut()),
            plans: self.misses.as_ptr().cast_mut(),
            counter: self.counter.as_ptr().cast_mut(),
            logits: self.logits.as_ptr().cast_mut(),
            limit_ns: DOORBELL_LIMIT_NS,
            skip_below: self.skip_below,
            host_table: self
                .pcie
                .as_ref()
                .map_or(std::ptr::null_mut(), |p| p.host_table.as_ptr().cast_mut()),
            scratch: self
                .pcie
                .as_ref()
                .map_or(std::ptr::null_mut(), |p| p.scratch.as_ptr().cast_mut()),
            fetch: self
                .pcie
                .as_ref()
                .map_or(std::ptr::null_mut(), |p| p.fetch.as_ptr().cast_mut()),
            take: self.pcie.as_ref().map_or(0, |p| p.take),
            prefetch: self.pcie.as_ref().map_or(0, |p| p.prefetch),
            pre_scratch: self
                .pcie
                .as_ref()
                .map_or(std::ptr::null_mut(), |p| p.pre_scratch.as_ptr().cast_mut()),
            pre_src: self
                .pcie
                .as_ref()
                .map_or(std::ptr::null_mut(), |p| p.pre_src.as_ptr().cast_mut()),
            pre_meta: self
                .pcie
                .as_ref()
                .map_or(std::ptr::null_mut(), |p| p.pre_meta.as_ptr().cast_mut()),
            pcie_stats: self
                .pcie
                .as_ref()
                .map_or(std::ptr::null_mut(), |p| p.stats.as_ptr().cast_mut()),
        }
    }
}

impl Executor<'_> {
    pub(super) fn device_enabled(&self) -> bool {
        self.device_decode.is_some()
    }
    /// Test hook: park (None) or restore the device path; parked, `step`
    /// runs the per-op synchronous path on the same executor.
    #[cfg(test)]
    pub(in crate::glm53) fn swap_device(
        &mut self,
        decode: Option<DeviceDecode>,
    ) -> Option<DeviceDecode> {
        self.device_ready = false;
        std::mem::replace(&mut self.device_decode, decode)
    }
    /// Profile mode runs device tokens eagerly with phase events.
    pub(super) fn device_profile(&mut self, enabled: bool) -> Result<()> {
        if let Some(dev) = self.device_decode.as_mut() {
            dev.graph = None;
            dev.events = if enabled {
                Some(
                    (0..self.layers.len() * MARKS + 2)
                        .map(|_| self.stream.event())
                        .collect::<Result<_>>()?,
                )
            } else {
                None
            };
        }
        Ok(())
    }
    /// Enqueue one token: embed, all layers, head and greedy sampling.
    fn device_submit(&self) -> Result<()> {
        let dev = self.device_decode.as_ref().unwrap();
        let refs = dev.refs();
        let st = self.stream.as_raw();
        let mark = |i: usize| -> Result<()> {
            match &dev.events {
                Some(events) => events[i].record(&self.stream),
                None => Ok(()),
            }
        };
        check(
            unsafe {
                g::hayate_glm_dev_embed(
                    self.input.as_ptr().cast_mut(),
                    self.globals["embed_tokens.weight"].as_ptr(),
                    refs.ctl,
                    dev.host_ctl.device_ptr(),
                    self.vocab as u32,
                    st,
                )
            },
            "glm device embed",
        )?;
        let mut input = self.input.as_ptr().cast_mut();
        let last = self.layers.len() - 1;
        let mut prev = None;
        for (i, layer) in self.layers.iter().enumerate() {
            mark(layer.index * MARKS)?;
            let (mailbox, format) = if layer.index >= 3 {
                (
                    layer.asynchronous.as_ref().unwrap().mailbox.device(0),
                    self.experts.format(layer.index),
                )
            } else {
                (std::ptr::null_mut(), (true, false))
            };
            let defer = Defer {
                below: if layer.index == last {
                    0.
                } else {
                    dev.defer_below
                },
                prev,
            };
            input = layer.device_step(
                &self.stream,
                input,
                &refs,
                mailbox,
                format,
                defer,
                dev.pcie.as_ref().map(|p| &p.lane),
                self.layers.get(i + 1).filter(|next| next.index >= 4),
                &|phase| mark(layer.index * MARKS + phase),
            )?;
            if dev.defer_below > 0. && layer.index >= 3 {
                prev = Some(layer.deferred(&refs, mailbox));
            }
        }
        mark(self.layers.len() * MARKS)?;
        check(
            unsafe {
                g::hayate_glm_dev_head_norm(
                    self.mean.as_ptr().cast_mut(),
                    self.normalized.as_ptr().cast_mut(),
                    input,
                    self.globals["norm.weight"].as_ptr(),
                    st,
                )
            },
            "glm device head norm",
        )?;
        check(
            unsafe {
                g::hayate_glm_dense(
                    self.logits.as_ptr().cast_mut(),
                    self.normalized.as_ptr(),
                    g::DenseWeight {
                        data: self.globals["lm_head.weight"].as_ptr(),
                        scale: std::ptr::null(),
                        format: self.head_format,
                    },
                    self.vocab as u32,
                    HIDDEN as u32,
                    st,
                )
            },
            "glm device lm head",
        )?;
        check(
            unsafe {
                g::hayate_glm_dev_greedy(
                    dev.greedy_values.as_ptr().cast_mut(),
                    dev.greedy_ids.as_ptr().cast_mut(),
                    self.logits.as_ptr(),
                    self.vocab as u32,
                    refs.counter,
                    refs.ctl,
                    dev.host_out.device_ptr(),
                    st,
                )
            },
            "glm device greedy",
        )?;
        if let (Some(pcie), Some(last)) = (&dev.pcie, self.layers.last()) {
            // Join the prefetch stream (its last copy targets the last layer) before the token ends.
            if refs.prefetch > 0 && last.index >= 4 {
                self.stream.wait_event(&pcie.lane.events[last.index].1)?;
            }
        }
        mark(self.layers.len() * MARKS + 1)
    }
    /// One decode token on the device path; returns the greedy next token.
    pub(super) fn step_device(&mut self, token: u32) -> Result<u32> {
        let host_start = Instant::now();
        if !self.device_ready {
            // Prefill, resets and session restores run on the layer streams.
            for layer in &self.layers {
                layer.synchronize()?;
            }
            self.stream.synchronize()?;
            self.device_ready = true;
        }
        // --maintenance-async: swaps planned while the previous token ran are
        // published now (otherwise they completed at the previous boundary).
        let overlap = self.experts.maintenance_async();
        if overlap {
            self.experts.accept_swaps()?;
        }
        let worker = self.experts.asynchronous.as_ref().unwrap();
        let dev = self.device_decode.as_mut().unwrap();
        dev.seq = dev.seq.wrapping_add(1).max(1);
        let seq = dev.seq;
        let mut table = std::mem::take(&mut dev.table_next);
        self.experts.gpu_table(&mut table);
        let changed = table != dev.table_host;
        std::mem::swap(&mut table, &mut dev.table_host);
        dev.table_next = table;
        let mut jobs = 0;
        let mut queued = Ok(());
        for layer in self.layers.iter_mut().skip(3) {
            let gpu = dev.table_host[(layer.index - 3) * EXPERTS..][..EXPERTS].to_vec();
            let snapshot = crate::glm53::async_decode::Snapshot {
                host: self.experts.host_row(layer.index),
                gpu,
                formats: self.experts.cpu_formats(layer.index),
            };
            let a = layer.asynchronous.as_mut().unwrap();
            a.snapshot = Some(std::sync::Arc::new(snapshot));
            queued = a
                .submit_with(
                    worker,
                    layer.index,
                    1,
                    self.experts.format(layer.index),
                    0,
                    seq,
                )
                .map(|_| ());
            if queued.is_err() {
                break;
            }
            jobs += 1;
        }
        if changed {
            for (dst, value) in dev
                .table_stage
                .as_mut_slice()
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(&dev.table_host)
            {
                *dst = value.to_le_bytes();
            }
            // The stage is rewritten only after this token's synchronization.
            queued =
                queued.and(unsafe { self.stream.copy_h2d_async(&mut dev.table, &dev.table_stage) });
        }
        if let Some(pcie) = dev.pcie.as_mut() {
            // RAM-tier image pointers for the PCIe lane (they change with the swaps only).
            let mut changed = false;
            for layer in 3..3 + pcie.host.len() / EXPERTS {
                for (e, image) in self.experts.host_row(layer).iter().enumerate() {
                    let pointer = image.as_ref().map_or(0, |i| i.0.as_ptr() as u64);
                    let at = (layer - 3) * EXPERTS + e;
                    changed |= pcie.host[at] != pointer;
                    pcie.host[at] = pointer;
                }
            }
            if changed {
                for (dst, value) in pcie
                    .stage
                    .as_mut_slice()
                    .as_chunks_mut::<8>()
                    .0
                    .iter_mut()
                    .zip(&pcie.host)
                {
                    *dst = value.to_le_bytes();
                }
                queued = queued.and(unsafe {
                    self.stream
                        .copy_h2d_async(&mut pcie.host_table, &pcie.stage)
                });
            }
        }
        unsafe {
            let ctl = dev.host_ctl.host_ptr().cast::<u32>();
            ctl.write_volatile(token);
            ctl.add(1).write_volatile(self.position as u32);
            ctl.add(2).write_volatile(seq);
        }
        let capture = dev.graph.is_none() && dev.events.is_none() && dev.eager_tokens > 0;
        let host_ms = host_start.elapsed().as_secs_f64() * 1000.;
        let gpu_start = Instant::now();
        if capture && queued.is_ok() {
            // Same work as the eager token; nothing outside the stream.
            match unsafe { Graph::capture(self.stream.as_raw(), || self.device_submit()) } {
                Ok(graph) => self.device_decode.as_mut().unwrap().graph = Some(graph),
                Err(error) => queued = Err(error),
            }
        }
        let dev = self.device_decode.as_ref().unwrap();
        let submitted = queued.and_then(|()| match &dev.graph {
            Some(graph) => unsafe { graph.launch(self.stream.as_raw()) },
            None => self.device_submit(),
        });
        // Plan the next swaps from the routes so far while the GPU runs.
        let maintenance = Instant::now();
        let planned = if submitted.is_ok() && overlap {
            self.experts.plan_swaps()
        } else {
            Ok(())
        };
        let mut maintenance_ms = maintenance.elapsed().as_secs_f64() * 1000.;
        let synced = self.stream.synchronize();
        if submitted.is_err() || synced.is_err() {
            // Doorbells the GPU never rang: let the worker finish those jobs.
            for layer in self.layers.iter().skip(3).take(jobs) {
                layer.asynchronous.as_ref().unwrap().ring(seq);
            }
        }
        let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.;
        let retire_start = Instant::now();
        let retired = self.experts.retire_device(jobs);
        for layer in self.layers.iter_mut().skip(3) {
            layer.asynchronous.as_mut().unwrap().snapshot = None;
        }
        submitted.and(synced).and(retired).and(planned)?;
        let dev = self.device_decode.as_mut().unwrap();
        if dev.graph.is_none() {
            dev.eager_tokens += 1;
        }
        let (next, error) = unsafe {
            let out = dev.host_out.host_ptr().cast::<u32>();
            (out.read_volatile(), out.add(1).read_volatile())
        };
        if error != 0 {
            return Err(Error::Weights(
                "GLM device decode: CPU expert doorbell timed out".into(),
            ));
        }
        if next as usize >= self.vocab {
            return Err(Error::Weights(
                "GLM device decode: invalid next token".into(),
            ));
        }
        dev.stats.tokens += 1;
        dev.stats.gpu_ms += gpu_ms;
        if let Some(events) = &dev.events {
            let ms = |a: usize, b: usize| events[a].elapsed_ms(&events[b]).map(f64::from);
            let s = &mut dev.stats;
            s.profiled += 1;
            for l in 0..self.layers.len() {
                let base = l * MARKS;
                let end = base + MARKS;
                s.attention_ms += ms(base, base + 1)?;
                s.router_ms += ms(base + 1, base + 2)?;
                s.experts_ms += ms(base + 2, base + 3)?;
                s.cpu_wait_ms += ms(base + 3, base + 4)?;
                s.finish_ms += ms(base + 4, end)?;
                if l % 4 == 3 {
                    s.select_ms += ms(base + 5, base + 6)?;
                    s.mla_ms += ms(base + 6, base + 7)?;
                }
            }
            s.head_ms += ms(self.layers.len() * MARKS, self.layers.len() * MARKS + 1)?;
        }
        for layer in &mut self.layers {
            layer.device_advance();
        }
        let retire_ms = retire_start.elapsed().as_secs_f64() * 1000.;
        if !overlap {
            // Same boundary as the per-op path: publish before the next token.
            let maintenance = Instant::now();
            self.experts.finish_token(&self.stream)?;
            maintenance_ms += maintenance.elapsed().as_secs_f64() * 1000.;
        }
        let s = &mut self.device_decode.as_mut().unwrap().stats;
        s.prep_ms += host_ms;
        s.retire_ms += retire_ms;
        s.maintenance_ms += maintenance_ms;
        Ok(next)
    }
    /// Summary line of the device decode counters.
    pub(super) fn device_report(&self) {
        let Some(dev) = &self.device_decode else {
            return;
        };
        let s = dev.stats;
        let n = s.tokens.max(1) as f64;
        let p = s.profiled.max(1) as f64;
        eprintln!(
            "glm_device tokens={} graph={} prep_ms={:.3} retire_ms={:.3} maintenance_ms={:.3} gpu_wall_ms_per_token={:.3} profiled={} attention_ms={:.3} router_ms={:.3} experts_ms={:.3} cpu_wait_ms={:.3} finish_ms={:.3} head_ms={:.3} sparse_select_ms={:.3} sparse_mla_ms={:.3}",
            s.tokens,
            dev.graph.is_some(),
            s.prep_ms / n,
            s.retire_ms / n,
            s.maintenance_ms / n,
            s.gpu_ms / n,
            s.profiled,
            s.attention_ms / p,
            s.router_ms / p,
            s.experts_ms / p,
            s.cpu_wait_ms / p,
            s.finish_ms / p,
            s.head_ms / p,
            s.select_ms / p,
            s.mla_ms / p,
        );
        if let Some(pcie) = &dev.pcie {
            let mut counts = [0u8; 16];
            if self.stream.copy_d2h(&mut counts, &pcie.stats).is_ok() {
                let c: Vec<u32> = counts
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                eprintln!(
                    "glm_pcie prefetch={} prefetched={} prefetch_used={} lane_serial={} ram_misses_cpu={}",
                    pcie.prefetch, c[0], c[1], c[2], c[3]
                );
            }
        }
    }
}
