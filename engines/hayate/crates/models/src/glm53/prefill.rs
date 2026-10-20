use super::{EXPERTS, TOP_K};
use hayate_core::{Error, Result};

/// Short prompts avoid a cache loan and model-wide expert uploads. The full
/// batch path remains explicitly selectable with --prefill-experts all.
#[cfg(any(feature = "cuda", test))]
pub(super) fn cpu_short_prompt(rows: usize, transfer_bytes: u64) -> bool {
    (1..32).contains(&rows) && transfer_bytes > (rows * super::HIDDEN * 2) as u64
}

/// Route indices in ascending expert order; each group has unique token rows.
pub(super) fn groups(ids: &[u32]) -> Result<Vec<Vec<u32>>> {
    if ids.is_empty() || !ids.len().is_multiple_of(TOP_K) {
        return Err(Error::InvalidShape("GLM prefill route count".into()));
    }
    let mut groups = vec![Vec::new(); EXPERTS];
    for (token, row) in ids.as_chunks::<TOP_K>().0.iter().enumerate() {
        for (j, &expert) in row.iter().enumerate() {
            if expert as usize >= EXPERTS || row[..j].contains(&expert) {
                return Err(Error::Weights("GLM invalid/duplicate expert route".into()));
            }
            groups[expert as usize].push((token * TOP_K + j) as u32);
        }
    }
    Ok(groups)
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use crate::glm53::gpu::Ptr;
    use crate::glm53::transfer::HostImage;
    use hayate_kernels::runtime::{Device, DeviceBuffer, Event, PinnedHost, Stream};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn take_buffer(
        device: &Device,
        bytes: usize,
        pool: &mut Vec<DeviceBuffer>,
    ) -> Result<DeviceBuffer> {
        if let Some(i) = pool.iter().position(|b| b.bytes() >= bytes) {
            Ok(pool.swap_remove(i))
        } else {
            device.alloc(bytes)
        }
    }
    #[derive(Debug)]
    pub(crate) struct Bank {
        pub arena: Vec<DeviceBuffer>,
        pub ids: Vec<usize>,
        pub copied_bytes: usize,
        pub host_bytes: usize,
        pub disk_bytes: usize,
        pub staging: PinnedHost,
        pub images: Vec<Ptr>,
        pub holds: Vec<Arc<HostImage>>,
        pub ready: Event,
        pub begin: Event,
        pub used: Event,
    }
    impl Bank {
        pub fn new(
            device: &Device,
            stream: &Stream,
            image_bytes: usize,
            count: usize,
            pool: &mut Vec<DeviceBuffer>,
        ) -> Result<Self> {
            let ready = stream.event()?;
            let used = stream.event()?;
            ready.record(stream)?;
            used.record(stream)?;
            Ok(Self {
                arena: (0..count)
                    .map(|_| take_buffer(device, image_bytes, pool))
                    .collect::<Result<_>>()?,
                ids: Vec::new(),
                copied_bytes: 0,
                host_bytes: 0,
                disk_bytes: 0,
                staging: device.alloc_pinned(image_bytes)?,
                images: Vec::new(),
                holds: Vec::new(),
                ready,
                begin: stream.event()?,
                used,
            })
        }
    }
    #[derive(Debug)]
    pub(crate) struct Prefill {
        pub capacity: usize,
        pub buffers: BTreeMap<String, DeviceBuffer>,
        pub layer: BTreeMap<String, DeviceBuffer>,
        pub banks: Option<[Bank; 2]>,
        pub upload: [Stream; 2],
        pub ahead: Option<[Bank; 2]>,
        pub previous: Vec<Vec<usize>>,
        pub all_experts: bool,
        pub prediction_capacity: usize,
        pub loan: Vec<(usize, usize, bool)>,
        /// One NVFP4 expert image: LOWBIT experts are converted here (lossless)
        /// before their matrices are dequantized for the grouped GEMM.
        pub lowbit_scratch: Option<DeviceBuffer>,
        context: usize,
        rotated: bool,
        image_bytes: usize,
        pub batched: bool,
    }
    impl Drop for Prefill {
        fn drop(&mut self) {
            for stream in &self.upload {
                let _ = stream.synchronize();
            }
        }
    }
    impl Prefill {
        pub fn new(
            device: &Device,
            capacity: usize,
            context: usize,
            rotated: bool,
            image_bytes: usize,
            batched: bool,
        ) -> Result<Self> {
            let mut result = Self {
                capacity,
                buffers: BTreeMap::new(),
                layer: BTreeMap::new(),
                banks: None,
                upload: [device.stream()?, device.stream()?],
                ahead: None,
                previous: vec![Vec::new(); 46],
                all_experts: false,
                prediction_capacity: EXPERTS,
                loan: Vec::new(),
                context,
                rotated,
                image_bytes,
                batched,
                lowbit_scratch: None,
            };
            if !batched {
                result.allocate(device, &mut Vec::new())?;
            }
            Ok(result)
        }
        pub fn allocate(&mut self, device: &Device, pool: &mut Vec<DeviceBuffer>) -> Result<()> {
            let small = 16
                + 2 * self.prediction_capacity
                + layer_layout(self.capacity, self.context)
                    .into_iter()
                    .chain(moe_layout(self.capacity, self.rotated))
                    .filter(|(_, b)| *b <= self.image_bytes)
                    .count();
            pool.truncate(small);
            let capacity = self.capacity;
            let context = self.context;
            let rotated = self.rotated;
            let image_bytes = self.image_bytes;
            let batched = self.batched;
            let upload = &self.upload;
            if capacity == 1 {
                return Ok(());
            }
            let buffers = &mut self.buffers;
            let layer = &mut self.layer;
            if batched {
                for (name, bytes) in layer_layout(capacity, context) {
                    if !layer.contains_key(name) {
                        layer.insert(name.into(), take_buffer(device, bytes, pool)?);
                    }
                }
            }
            for (name, bytes) in moe_layout(capacity, rotated) {
                if !buffers.contains_key(name) {
                    buffers.insert(name.into(), take_buffer(device, bytes, pool)?);
                }
            }
            let banks = if batched {
                Some([
                    Bank::new(device, &upload[0], image_bytes, 8, pool)?,
                    Bank::new(device, &upload[1], image_bytes, 8, pool)?,
                ])
            } else {
                None
            };
            self.banks = banks;
            if batched {
                self.ahead = Some([
                    Bank::new(
                        device,
                        &upload[0],
                        image_bytes,
                        self.prediction_capacity,
                        pool,
                    )?,
                    Bank::new(
                        device,
                        &upload[1],
                        image_bytes,
                        self.prediction_capacity,
                        pool,
                    )?,
                ]);
            }
            Ok(())
        }
        pub fn release(&mut self) -> Result<()> {
            for stream in &self.upload {
                stream.synchronize()?;
            }
            self.banks = None;
            self.ahead = None;
            self.buffers.clear();
            self.layer.clear();
            Ok(())
        }
        pub fn loan_slots(&self) -> usize {
            // Small buffers use whole cache allocations; larger ones use the
            // capacity released by the remaining slots, without aliasing arenas.
            workspace_slots(self.capacity, self.context, self.image_bytes, self.rotated)
                - 2 * (EXPERTS - self.prediction_capacity)
                - self
                    .layer
                    .values()
                    .chain(self.buffers.values())
                    .map(|b| b.bytes().div_ceil(self.image_bytes))
                    .sum::<usize>()
        }
        pub fn prepare_first_layer(&mut self, device: &Device) -> Result<()> {
            for (name, bytes) in layer_layout(self.capacity, self.context) {
                self.layer.insert(name.into(), device.alloc(bytes)?);
            }
            self.buffers
                .insert("ids".into(), device.alloc(self.capacity * TOP_K * 4)?);
            Ok(())
        }

        pub fn p(&self, name: &str) -> Ptr {
            self.buffers[name].as_ptr().cast_mut()
        }
        pub fn bytes(&self) -> usize {
            self.ahead.as_ref().map_or(0, |banks| {
                banks
                    .iter()
                    .flat_map(|b| &b.arena)
                    .map(DeviceBuffer::bytes)
                    .sum::<usize>()
            }) + self
                .buffers
                .values()
                .chain(self.layer.values())
                .map(DeviceBuffer::bytes)
                .sum::<usize>()
                + self.banks.as_ref().map_or(0, |banks| {
                    banks
                        .iter()
                        .map(|bank| bank.arena.iter().map(DeviceBuffer::bytes).sum::<usize>())
                        .sum()
                })
        }
    }
}
#[cfg(feature = "cuda")]
pub(super) use gpu::{Bank, Prefill};
#[cfg(feature = "cuda")]
pub(super) use groups as group_routes;

#[cfg(any(feature = "cuda", test))]
pub(super) fn layer_layout(capacity: usize, context: usize) -> Vec<(&'static str, usize)> {
    let t = capacity;
    let mut sizes = vec![
        ("input", t * 16384 * 2),
        ("output", t * 16384 * 2),
        ("attn_map", t * 24 * 4),
        ("ffn_map", t * 24 * 4),
        ("attn_collapsed", t * 4096 * 2),
        ("attn_norm", t * 4096 * 2),
        ("attention", t * 4096 * 2),
        ("attention_residual", t * 16384 * 2),
        ("ffn_collapsed", t * 4096 * 2),
        ("ffn_norm", t * 4096 * 2),
        ("mlp", t * 4096 * 2),
        ("gate", t * 12288 * 2),
        ("up", t * 12288 * 2),
        ("product", t * 12288 * 2),
        ("expert", t * 4096 * 2),
        ("router_logits", t * EXPERTS * 4),
        ("route_ids", t * TOP_K * 4),
        ("route_weights", t * TOP_K * 4),
        ("dense.scratch", 4096 * 16384 * 2),
        ("q", t * 16384 * 2),
        ("k", t * 8192 * 2),
        ("v", t * 8192 * 2),
        ("cq", t * 8192 * 2),
        ("ck", t * 8192 * 2),
        ("cv", t * 8192 * 2),
        ("fa", t * 128 * 2),
        ("fb", t * 8192 * 2),
        ("ga", t * 128 * 2),
        ("gb", t * 8192 * 2),
        ("beta", t * 64 * 2),
        ("decay", t * 8192 * 4),
        ("kda", t * 8192 * 2),
        ("kda_nq", t * 8192 * 4),
        ("kda_nk", t * 8192 * 4),
        ("gated", t * 8192 * 2),
        ("qa", t * 1536 * 2),
        ("qr", t * 1536 * 2),
        ("kva", t * 512 * 2),
        ("kvr", t * 512 * 2),
        ("iq", t * 4096 * 2),
        ("ik", t * 128 * 2),
        ("ikn", t * 128 * 2),
        ("ig", t * 128 * 2),
        ("iw", t * 32 * 2),
        ("scores", (t * context.div_ceil(4) * 4).max(4)),
        ("selected", t * 2051 * 4),
        ("attn_core", t * 16384 * 2),
        ("selected_counts", t * 4),
        ("kv", t.max(2051) * 32768 * 2),
        ("latent_selected", t.max(2051) * 1024),
    ];
    sizes.retain(|(_, bytes)| *bytes > 0);
    sizes
}

#[cfg(any(feature = "cuda", test))]
fn moe_layout(capacity: usize, rotated: bool) -> Vec<(&'static str, usize)> {
    use super::HIDDEN;
    let mut layout = [
        ("norm", capacity * HIDDEN * 2),
        ("residual", capacity * 4 * HIDDEN * 2),
        ("map", capacity * 24 * 4),
        ("mlp", capacity * HIDDEN * 2),
        ("ids", capacity * TOP_K * 4),
        ("route_weights", capacity * TOP_K * 4),
        ("logits", capacity * EXPERTS * 4),
        ("assignments", capacity * TOP_K * 4),
        ("ranges", EXPERTS * 2 * 4),
        ("packed_meta", EXPERTS * 24 + capacity * TOP_K * 4),
        ("metadata", 608),
        ("scales", 3 * 8 * 4),
        ("accumulator", capacity * HIDDEN * 4),
        ("input", capacity * HIDDEN * 2),
        ("gate", capacity * 2048 * 2),
        ("up", capacity * 2048 * 2),
        ("product", capacity * 2048 * 2),
        ("expert", capacity * HIDDEN * 2),
        ("gate_w", 8 * 2048 * HIDDEN * 2),
        ("up_w", 8 * 2048 * HIDDEN * 2),
        ("down_w", 8 * HIDDEN * 2048 * 2),
    ]
    .to_vec();
    if rotated {
        layout.push(("rotated_norm", capacity * HIDDEN * 2));
    }
    layout
}

#[cfg(any(feature = "cuda", test))]
pub(super) fn capacity(
    requested: usize,
    context: usize,
    budget: usize,
    image_bytes: usize,
) -> Result<usize> {
    if requested > 8192 || context == 0 || image_bytes == 0 {
        return Err(Error::InvalidShape("GLM prefill capacity".into()));
    }
    let mut cap = if requested == 0 {
        8192.min(context)
    } else {
        requested.min(context)
    };
    while cap > 1 {
        // The legacy grouped buffers are still shared by the batched MoE.
        let bytes = workspace_slots(cap, context, image_bytes, true) * image_bytes;
        if bytes <= budget {
            return Ok(cap);
        }
        if requested != 0 {
            return Err(Error::OutOfMemory {
                buffer: "GLM prefill workspace",
                requested: bytes as u64,
                available: budget as u64,
            });
        }
        cap = cap.div_ceil(2);
    }
    Ok(1)
}

/// Bytes a batched prefill chunk of `min(8192, context)` rows needs beyond
/// the load-time allocation (layer arena, grouped MoE buffers, 16 staging
/// images; no lookahead). Budgets reserve it so prefill never borrows slots.
#[cfg(feature = "cuda")]
pub(super) fn reserve_bytes(context: usize, image_bytes: usize, rotated: bool) -> u64 {
    ((workspace_slots(8192.min(context), context, image_bytes, rotated) - 2 * EXPERTS)
        * image_bytes) as u64
}

fn workspace_slots(capacity: usize, context: usize, image_bytes: usize, rotated: bool) -> usize {
    16 + 2 * EXPERTS
        + layer_layout(capacity, context)
            .into_iter()
            .chain(moe_layout(capacity, rotated))
            .map(|(_, b)| b.div_ceil(image_bytes))
            .sum::<usize>()
}

#[cfg(any(feature = "cuda", test))]
pub(super) fn predictions(all: bool, previous: &[usize]) -> Vec<usize> {
    if all {
        (0..EXPERTS).collect()
    } else {
        previous.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefill_capacity_respects_context_and_memory() -> Result<()> {
        let capacity = |requested, context, budget| {
            super::capacity(
                requested,
                context,
                budget,
                super::super::experts::IMAGE_BYTES,
            )
        };
        assert_eq!(capacity(0, 131072, 100 << 30)?, 8192);
        assert_eq!(capacity(0, 127, 100 << 30)?, 127);
        assert_eq!(capacity(0, 8192, 0)?, 1);
        assert!(capacity(8192, 8192, 1 << 30).is_err());
        assert!(capacity(8193, 8192, usize::MAX).is_err());
        let smaller = capacity(0, 8192, 20 << 30)?;
        assert!((2..8192).contains(&smaller));
        let layout = layer_layout(8192, 131072);
        assert_eq!(
            layout.iter().find(|(n, _)| *n == "kv").unwrap().1,
            8192 * 32768 * 2
        );
        assert_eq!(
            layout.iter().find(|(n, _)| *n == "scores").unwrap().1,
            8192 * 32768 * 4
        );
        let names: std::collections::BTreeSet<_> = layout.iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), layout.len());
        Ok(())
    }
    #[test]
    fn prefill_prediction_and_loan_budget_cover_all_allocations() -> Result<()> {
        let ids = [7, 2, 3, 4, 5, 6, 1, 0, 8, 2, 3, 4, 5, 6, 1, 0];
        let previous = groups(&ids)?
            .iter()
            .enumerate()
            .filter_map(|(e, g)| (!g.is_empty()).then_some(e))
            .collect::<Vec<_>>();
        assert_eq!(predictions(false, &[]), []);
        assert_eq!(predictions(false, &previous), (0..9).collect::<Vec<_>>());
        assert_eq!(predictions(true, &[]), (0..EXPERTS).collect::<Vec<_>>());
        for image in [
            3 * super::super::experts::matrix_stride(true),
            super::super::experts::IMAGE_BYTES,
        ] {
            for rows in [2, 127, 2048, 8192] {
                let layout = layer_layout(rows, 131072)
                    .into_iter()
                    .chain(moe_layout(rows, true))
                    .collect::<Vec<_>>();
                let small = 16 + 2 * EXPERTS + layout.iter().filter(|(_, b)| *b <= image).count();
                let large = layout
                    .iter()
                    .filter(|(_, b)| *b > image)
                    .map(|(_, b)| b)
                    .sum::<usize>();
                let loan = workspace_slots(rows, 131072, image, true);
                assert!(loan >= small);
                assert!((loan - small) * image >= large);
            }
        }
        Ok(())
    }
    #[test]
    fn ragged_routes_preserve_assignment_indices() {
        let ids = [7, 2, 3, 4, 5, 6, 1, 0, 8, 2, 3, 4, 5, 6, 1, 0];
        let grouped = groups(&ids).unwrap();
        assert_eq!(grouped[0], [7, 15]);
        assert_eq!(grouped[7], [0]);
        assert_eq!(grouped[8], [8]);
        assert_eq!(grouped[2], [1, 9]);
        assert_eq!(grouped.iter().map(Vec::len).sum::<usize>(), ids.len());
        assert!(groups(&[]).is_err());
        assert!(groups(&ids[..9]).is_err());
        assert!(groups(&[288, 2, 3, 4, 5, 6, 1, 0]).is_err());
        assert!(groups(&[2, 2, 3, 4, 5, 6, 1, 0]).is_err());
    }
    #[test]
    fn short_prompt_avoids_uploads_only_when_there_are_misses() {
        assert!(cpu_short_prompt(1, 1 << 30));
        assert!(cpu_short_prompt(31, 1 << 30));
        assert!(!cpu_short_prompt(32, 1 << 30));
        assert!(!cpu_short_prompt(2048, 1 << 30));
        assert!(!cpu_short_prompt(17, 0));
        assert!(!cpu_short_prompt(17, 17 * 4096 * 2));
    }
}
