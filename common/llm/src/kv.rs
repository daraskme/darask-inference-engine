//! Per-layer KV placement and ordered transfer plans, without GPU allocation.
//!
//! A cache owns one sequence; shared prefixes must be copied before writing.
//! Slot numbers are local to this layer and refer to executor-owned arenas.
//! Append plans advance the logical state immediately. The executor must finish
//! them in order before using another plan or reusing its staging workspace.

use std::collections::{HashSet, VecDeque};

use hayate_core::{DType, Error, Result};

use crate::LayerCache;

pub const DEFAULT_BLOCK_TOKENS: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvShape {
    pub heads: u32,
    pub head_dim: u32,
    pub dtype: DType,
}

impl KvShape {
    /// KV defaults to INT8 (§4.4); head geometry comes from the model config.
    pub const fn new(heads: u32, head_dim: u32) -> Self {
        Self {
            heads,
            head_dim,
            dtype: DType::I8,
        }
    }
}

/// K and V payload bytes per token, using core's format size definition.
/// Sliding KV has the same rate but retains only its window; recurrent layers
/// have no growing KV. Fixed recurrent state, indexer caches, quantization
/// scales and workspaces are separate from this payload estimate (§4.1).
pub fn kv_bytes_per_token(layer: LayerCache, shape: KvShape) -> Result<u64> {
    if layer == LayerCache::Recurrent {
        return Ok(0);
    }
    if shape.heads == 0 || shape.head_dim == 0 {
        return Err(Error::InvalidShape("KV dimensions must be nonzero".into()));
    }
    if !matches!(
        shape.dtype,
        DType::I8 | DType::F8E4M3 | DType::F8E5M2 | DType::Bf16
    ) {
        return Err(Error::InvalidShape("KV requires INT8, FP8 or BF16".into()));
    }
    let elements = u64::from(shape.heads)
        .checked_mul(u64::from(shape.head_dim))
        .and_then(|n| n.checked_mul(2));
    elements
        .and_then(|n| shape.dtype.payload_bytes(n))
        .ok_or_else(|| Error::InvalidShape("KV token size overflows".into()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    Vram { slot: usize },
    PinnedRam { slot: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub location: Location,
    /// Valid prefix of the page; the final page can be incomplete.
    pub tokens: u32,
}

/// Execute in vector order: a spill must finish before its VRAM slot is
/// overwritten, and a reload before its pinned slot is reused. Transfers copy
/// only `tokens` valid entries, with the page stride determined by the shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppendOp {
    Spill {
        block: usize,
        vram_slot: usize,
        ram_slot: usize,
        tokens: u32,
    },
    Reload {
        block: usize,
        ram_slot: usize,
        vram_slot: usize,
        tokens: u32,
    },
    /// Write newly computed K/V at `offset` within the VRAM page.
    Write {
        block: usize,
        vram_slot: usize,
        offset: u32,
        tokens: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionRequest<'a> {
    All,
    SlidingWindow {
        tokens: u32,
    },
    /// Page numbers in indexer order. Include QSA's incomplete tail here when
    /// needed; the planner neither expands the selection nor runs the indexer.
    Indexed(&'a [usize]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionSource {
    Vram { slot: usize },
    Staging { slot: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionBlock {
    pub block: usize,
    pub source: AttentionSource,
    /// Absolute position of the page's first token, for masks / RoPE.
    pub first_token: u32,
    /// Token range within the page, excluding the window's old prefix and
    /// uninitialized entries in the final page.
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gather {
    pub block: usize,
    pub ram_slot: usize,
    pub staging_slot: usize,
    pub tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionPlan {
    /// Logical attention order, including pages already resident in VRAM.
    pub blocks: Vec<AttentionBlock>,
    /// RAM pages only, packed into staging in their requested order.
    pub gathers: Vec<Gather>,
    /// Staging is separate from resident VRAM; each gather uses a full slot.
    pub staging_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryUsage {
    /// Active full-page allocations, including unused tail capacity. Does not
    /// include staging, scales, indexer cache or fixed recurrent state.
    pub vram: u64,
    pub ram: u64,
}

/// PagedKv / SparseIndexedKv storage for a single layer. Sliding rings and
/// recurrent snapshots remain separate because they overwrite state (§9.1).
#[derive(Debug)]
pub struct KvCache {
    block_tokens: u32,
    block_bytes: u64,
    resident_limit: usize,
    tokens: u32,
    blocks: Vec<Block>,
    resident: VecDeque<usize>,
    free_vram: Vec<usize>,
    next_vram: usize,
    free_ram: Vec<usize>,
    next_ram: usize,
}

impl KvCache {
    /// `kv_resident` is rounded down to full pages and must hold at least one
    /// page, so the append tail always has a VRAM slot.
    pub fn new(
        layer: LayerCache,
        shape: KvShape,
        block_tokens: u32,
        kv_resident: u32,
    ) -> Result<Self> {
        if layer.rollback_needs_snapshot() {
            return Err(Error::InvalidPlan(
                "paged KV requires a truncatable layer".into(),
            ));
        }
        if let LayerCache::SparseIndexedKv { compress, budget } = layer
            && (compress == 0 || budget == 0)
        {
            return Err(Error::InvalidPlan(
                "sparse KV compression and budget must be nonzero".into(),
            ));
        }
        if block_tokens == 0 || kv_resident < block_tokens {
            return Err(Error::InvalidPlan(
                "KV resident budget must hold a nonzero page".into(),
            ));
        }
        let block_bytes = kv_bytes_per_token(layer, shape)?
            .checked_mul(u64::from(block_tokens))
            .ok_or_else(|| Error::InvalidPlan("KV page size overflows".into()))?;
        Ok(Self {
            block_tokens,
            block_bytes,
            resident_limit: (kv_resident / block_tokens) as usize,
            tokens: 0,
            blocks: Vec::new(),
            resident: VecDeque::new(),
            free_vram: Vec::new(),
            next_vram: 0,
            free_ram: Vec::new(),
            next_ram: 0,
        })
    }

    pub fn tokens(&self) -> u32 {
        self.tokens
    }

    /// Page numbers are indices into this slice and remain stable until
    /// truncated. Removed pages and physical slots may subsequently be reused.
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    fn vram_slot(&mut self, ops: &mut Vec<AppendOp>) -> usize {
        if let Some(slot) = self.free_vram.pop() {
            return slot;
        }
        if self.next_vram < self.resident_limit {
            let slot = self.next_vram;
            self.next_vram += 1;
            return slot;
        }
        let block = self.resident.pop_front().unwrap();
        let Location::Vram { slot } = self.blocks[block].location else {
            unreachable!();
        };
        let ram_slot = self.free_ram.pop().unwrap_or_else(|| {
            let slot = self.next_ram;
            self.next_ram += 1;
            slot
        });
        ops.push(AppendOp::Spill {
            block,
            vram_slot: slot,
            ram_slot,
            tokens: self.blocks[block].tokens,
        });
        self.blocks[block].location = Location::PinnedRam { slot: ram_slot };
        slot
    }

    /// Append a token or prefill chunk. Returned writes split at page boundaries
    /// and spills precede slot reuse even when a chunk exceeds the resident cap.
    /// Overflow is checked before changing placement.
    pub fn append(&mut self, tokens: u32) -> Result<Vec<AppendOp>> {
        let end = self
            .tokens
            .checked_add(tokens)
            .ok_or_else(|| Error::InvalidPlan("KV sequence length overflows".into()))?;
        let mut ops = Vec::new();
        while self.tokens < end {
            let block = (self.tokens / self.block_tokens) as usize;
            let offset = self.tokens % self.block_tokens;
            let vram_slot = if offset == 0 {
                let slot = self.vram_slot(&mut ops);
                self.blocks.push(Block {
                    location: Location::Vram { slot },
                    tokens: 0,
                });
                self.resident.push_back(block);
                slot
            } else {
                match self.blocks[block].location {
                    Location::Vram { slot } => slot,
                    Location::PinnedRam { slot: ram_slot } => {
                        // Truncation may leave an old RAM page as the new tail.
                        let slot = self.vram_slot(&mut ops);
                        ops.push(AppendOp::Reload {
                            block,
                            ram_slot,
                            vram_slot: slot,
                            tokens: self.blocks[block].tokens,
                        });
                        self.free_ram.push(ram_slot);
                        self.blocks[block].location = Location::Vram { slot };
                        self.resident.push_back(block);
                        slot
                    }
                }
            };
            let count = (end - self.tokens).min(self.block_tokens - offset);
            ops.push(AppendOp::Write {
                block,
                vram_slot,
                offset,
                tokens: count,
            });
            self.blocks[block].tokens = offset + count;
            self.tokens += count;
        }
        Ok(ops)
    }

    /// Drop a rejected speculative suffix, including partial pages. No transfer
    /// is needed; subsequent plans mask invalid tail entries and reuse slots.
    pub fn truncate(&mut self, tokens: u32) -> Result<()> {
        if tokens > self.tokens {
            return Err(Error::InvalidPlan("cannot extend KV by truncation".into()));
        }
        let keep = tokens.div_ceil(self.block_tokens) as usize;
        while self.blocks.len() > keep {
            match self.blocks.pop().unwrap().location {
                Location::Vram { slot } => {
                    let removed = self.resident.pop_back();
                    debug_assert_eq!(removed, Some(self.blocks.len()));
                    self.free_vram.push(slot);
                }
                Location::PinnedRam { slot } => self.free_ram.push(slot),
            }
        }
        if let Some(tail) = self.blocks.last_mut() {
            tail.tokens = tokens - (keep as u32 - 1) * self.block_tokens;
        }
        self.tokens = tokens;
        Ok(())
    }

    pub fn memory_usage(&self) -> Result<MemoryUsage> {
        let bytes = |pages: usize| {
            (pages as u64)
                .checked_mul(self.block_bytes)
                .ok_or_else(|| Error::InvalidPlan("KV memory size overflows".into()))
        };
        Ok(MemoryUsage {
            vram: bytes(self.resident.len())?,
            ram: bytes(self.blocks.len() - self.resident.len())?,
        })
    }

    /// Build a read-only plan. Gathering leaves the canonical RAM placement
    /// intact, so sparse reads do not evict the recent append tail.
    pub fn attention(&self, request: AttentionRequest<'_>) -> Result<AttentionPlan> {
        let mut plan = AttentionPlan {
            blocks: Vec::new(),
            gathers: Vec::new(),
            staging_bytes: 0,
        };
        let first_token = match request {
            AttentionRequest::SlidingWindow { tokens } => self.tokens.saturating_sub(tokens),
            _ => 0,
        };
        let mut add = |block: usize| -> Result<()> {
            let page = self
                .blocks
                .get(block)
                .ok_or_else(|| Error::InvalidPlan(format!("KV block {block} does not exist")))?;
            let source = match page.location {
                Location::Vram { slot } => AttentionSource::Vram { slot },
                Location::PinnedRam { slot } => {
                    let staging_slot = plan.gathers.len();
                    plan.gathers.push(Gather {
                        block,
                        ram_slot: slot,
                        staging_slot,
                        tokens: page.tokens,
                    });
                    plan.staging_bytes = plan
                        .staging_bytes
                        .checked_add(self.block_bytes)
                        .ok_or_else(|| Error::InvalidPlan("KV staging size overflows".into()))?;
                    AttentionSource::Staging { slot: staging_slot }
                }
            };
            let page_start = block as u32 * self.block_tokens;
            plan.blocks.push(AttentionBlock {
                block,
                source,
                first_token: page_start,
                start: first_token.saturating_sub(page_start),
                end: page.tokens,
            });
            Ok(())
        };
        match request {
            AttentionRequest::Indexed(indices) => {
                let mut seen = HashSet::with_capacity(indices.len());
                for &block in indices {
                    if !seen.insert(block) {
                        return Err(Error::InvalidPlan(format!("duplicate KV block {block}")));
                    }
                    add(block)?;
                }
            }
            AttentionRequest::All | AttentionRequest::SlidingWindow { .. } => {
                if first_token < self.tokens {
                    for block in (first_token / self.block_tokens) as usize..self.blocks.len() {
                        add(block)?;
                    }
                }
            }
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(layer: LayerCache, block_tokens: u32, resident: u32) -> KvCache {
        KvCache::new(layer, KvShape::new(2, 256), block_tokens, resident).unwrap()
    }

    fn sparse() -> LayerCache {
        LayerCache::SparseIndexedKv {
            compress: 4,
            budget: 2048,
        }
    }

    #[test]
    fn bytes_follow_layer_and_dtype() {
        for layer in [
            LayerCache::PagedKv,
            LayerCache::SlidingKv { window: 1024 },
            sparse(),
        ] {
            for (dtype, bytes) in [
                (DType::I8, 1024),
                (DType::F8E4M3, 1024),
                (DType::F8E5M2, 1024),
                (DType::Bf16, 2048),
            ] {
                assert_eq!(
                    kv_bytes_per_token(
                        layer,
                        KvShape {
                            dtype,
                            ..KvShape::new(2, 256)
                        }
                    )
                    .unwrap(),
                    bytes
                );
            }
        }
        assert_eq!(
            kv_bytes_per_token(LayerCache::Recurrent, KvShape::new(2, 256)).unwrap(),
            0
        );
        // Gemma local/global layers can have different head geometry.
        assert_eq!(
            kv_bytes_per_token(LayerCache::PagedKv, KvShape::new(4, 128)).unwrap(),
            1024
        );
    }

    #[test]
    fn rejects_invalid_shapes_and_budgets() {
        for shape in [
            KvShape::new(0, 256),
            KvShape::new(2, 0),
            KvShape::new(u32::MAX, u32::MAX),
            KvShape {
                dtype: DType::F32,
                ..KvShape::new(2, 256)
            },
        ] {
            assert!(kv_bytes_per_token(LayerCache::PagedKv, shape).is_err());
        }
        for (page, resident) in [(0, 256), (256, 0), (256, 255)] {
            assert!(
                KvCache::new(LayerCache::PagedKv, KvShape::new(2, 256), page, resident).is_err()
            );
        }
        for layer in [
            LayerCache::Recurrent,
            LayerCache::SlidingKv { window: 1024 },
            LayerCache::SparseIndexedKv {
                compress: 0,
                budget: 2048,
            },
            LayerCache::SparseIndexedKv {
                compress: 4,
                budget: 0,
            },
        ] {
            assert!(KvCache::new(layer, KvShape::new(2, 256), 256, 2048).is_err());
        }
        assert!(
            KvCache::new(
                LayerCache::PagedKv,
                KvShape::new(u32::MAX, 2),
                u32::MAX,
                u32::MAX
            )
            .is_err()
        );
    }

    #[test]
    fn token_appends_spill_oldest_before_reusing_slot() {
        let mut c = cache(LayerCache::PagedKv, 4, 8);
        for _ in 0..8 {
            assert!(matches!(
                c.append(1).unwrap().as_slice(),
                [AppendOp::Write { .. }]
            ));
        }
        assert_eq!(
            c.append(1).unwrap(),
            [
                AppendOp::Spill {
                    block: 0,
                    vram_slot: 0,
                    ram_slot: 0,
                    tokens: 4
                },
                AppendOp::Write {
                    block: 2,
                    vram_slot: 0,
                    offset: 0,
                    tokens: 1
                },
            ]
        );
        assert_eq!(
            c.blocks(),
            [
                Block {
                    location: Location::PinnedRam { slot: 0 },
                    tokens: 4
                },
                Block {
                    location: Location::Vram { slot: 1 },
                    tokens: 4
                },
                Block {
                    location: Location::Vram { slot: 0 },
                    tokens: 1
                },
            ]
        );
        assert_eq!(
            c.memory_usage().unwrap(),
            MemoryUsage {
                vram: 8192,
                ram: 4096
            }
        );
    }

    #[test]
    fn resident_cap_rounds_down_and_pages_are_layer_local() {
        let mut a = cache(LayerCache::PagedKv, 4, 7);
        let mut b = cache(sparse(), 4, 7);
        a.append(5).unwrap();
        b.append(1).unwrap();
        assert_eq!(a.blocks()[0].location, Location::PinnedRam { slot: 0 });
        assert_eq!(b.blocks()[0].location, Location::Vram { slot: 0 });
        assert_eq!(
            a.memory_usage().unwrap(),
            MemoryUsage {
                vram: 4096,
                ram: 4096
            }
        );
        assert_eq!(b.tokens(), 1);
    }

    #[test]
    fn all_attention_only_gathers_ram_pages() {
        let mut c = cache(LayerCache::PagedKv, 4, 8);
        c.append(17).unwrap();
        let before = c.blocks().to_vec();
        let p = c.attention(AttentionRequest::All).unwrap();
        assert_eq!(
            p.blocks.iter().map(|b| b.block).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4]
        );
        assert_eq!(
            p.gathers
                .iter()
                .map(|g| (g.block, g.staging_slot))
                .collect::<Vec<_>>(),
            [(0, 0), (1, 1), (2, 2)]
        );
        assert_eq!(p.staging_bytes, 3 * 4 * 1024);
        assert_eq!(p.blocks[3].source, AttentionSource::Vram { slot: 1 });
        assert_eq!((p.blocks[4].start, p.blocks[4].end), (0, 1));
        assert_eq!(c.blocks(), before);
    }

    #[test]
    fn sliding_window_masks_partial_first_and_last_pages() {
        let mut c = cache(LayerCache::PagedKv, 4, 8);
        c.append(17).unwrap();
        let p = c
            .attention(AttentionRequest::SlidingWindow { tokens: 8 })
            .unwrap();
        assert_eq!(
            p.blocks
                .iter()
                .map(|b| (b.block, b.first_token, b.start, b.end))
                .collect::<Vec<_>>(),
            [(2, 8, 1, 4), (3, 12, 0, 4), (4, 16, 0, 1)]
        );
        assert_eq!(
            p.gathers,
            [Gather {
                block: 2,
                ram_slot: 2,
                staging_slot: 0,
                tokens: 4
            }]
        );
        assert_eq!(
            c.attention(AttentionRequest::SlidingWindow { tokens: 1 })
                .unwrap()
                .staging_bytes,
            0
        );
        assert!(
            c.attention(AttentionRequest::SlidingWindow { tokens: 0 })
                .unwrap()
                .blocks
                .is_empty()
        );
        assert_eq!(
            c.attention(AttentionRequest::SlidingWindow { tokens: 100 })
                .unwrap(),
            c.attention(AttentionRequest::All).unwrap()
        );
    }

    #[test]
    fn sparse_attention_preserves_indexer_order_and_selection() {
        let mut c = cache(sparse(), 4, 8);
        c.append(17).unwrap();
        let p = c.attention(AttentionRequest::Indexed(&[2, 4, 0])).unwrap();
        assert_eq!(
            p.blocks
                .iter()
                .map(|b| (b.block, b.source))
                .collect::<Vec<_>>(),
            [
                (2, AttentionSource::Staging { slot: 0 }),
                (4, AttentionSource::Vram { slot: 0 }),
                (0, AttentionSource::Staging { slot: 1 }),
            ]
        );
        assert_eq!(
            p.gathers,
            [
                Gather {
                    block: 2,
                    ram_slot: 2,
                    staging_slot: 0,
                    tokens: 4
                },
                Gather {
                    block: 0,
                    ram_slot: 0,
                    staging_slot: 1,
                    tokens: 4
                },
            ]
        );
        assert_eq!(p.staging_bytes, 8192);
        // The planner does not implicitly fetch unselected pages or the tail.
        assert_eq!(
            c.attention(AttentionRequest::Indexed(&[1]))
                .unwrap()
                .blocks
                .len(),
            1
        );
    }

    #[test]
    fn empty_and_invalid_attention_requests() {
        let mut c = cache(sparse(), 4, 8);
        assert!(
            c.attention(AttentionRequest::All)
                .unwrap()
                .blocks
                .is_empty()
        );
        assert!(
            c.attention(AttentionRequest::Indexed(&[]))
                .unwrap()
                .gathers
                .is_empty()
        );
        assert!(c.attention(AttentionRequest::Indexed(&[0])).is_err());
        c.append(5).unwrap();
        assert!(c.attention(AttentionRequest::Indexed(&[2])).is_err());
        assert!(c.attention(AttentionRequest::Indexed(&[0, 0])).is_err());
    }

    #[test]
    fn truncation_handles_ram_tail_page_boundaries_and_zero() {
        for layer in [LayerCache::PagedKv, sparse()] {
            let mut c = cache(layer, 4, 8);
            c.append(17).unwrap();
            assert!(c.truncate(18).is_err());
            assert_eq!(c.tokens(), 17);
            c.truncate(6).unwrap();
            assert_eq!(c.blocks().len(), 2);
            assert_eq!(c.blocks()[1].tokens, 2);
            assert_eq!(
                c.append(1).unwrap(),
                [
                    AppendOp::Reload {
                        block: 1,
                        ram_slot: 1,
                        vram_slot: 1,
                        tokens: 2
                    },
                    AppendOp::Write {
                        block: 1,
                        vram_slot: 1,
                        offset: 2,
                        tokens: 1
                    },
                ]
            );
            c.truncate(4).unwrap();
            assert_eq!(c.blocks()[0].tokens, 4);
            assert!(matches!(
                c.append(1).unwrap().as_slice(),
                [AppendOp::Write {
                    block: 1,
                    offset: 0,
                    ..
                }]
            ));
            c.truncate(0).unwrap();
            assert!(c.blocks().is_empty());
            assert_eq!(c.memory_usage().unwrap(), MemoryUsage { vram: 0, ram: 0 });
            c.append(17).unwrap();
            assert_eq!(
                c.memory_usage().unwrap(),
                MemoryUsage {
                    vram: 8192,
                    ram: 12288
                }
            );
        }
    }

    #[test]
    fn append_overflow_does_not_change_state() {
        let mut c = cache(LayerCache::PagedKv, u32::MAX, u32::MAX);
        c.append(u32::MAX).unwrap();
        let before = c.blocks().to_vec();
        assert!(c.append(1).is_err());
        assert_eq!(c.tokens(), u32::MAX);
        assert_eq!(c.blocks(), before);
        assert!(c.append(0).unwrap().is_empty());
        assert_eq!(
            c.attention(AttentionRequest::All).unwrap().blocks[0].end,
            u32::MAX
        );
    }

    #[test]
    fn qwen38_payload_and_tier_sizes_match_architecture() {
        // §4.1 uses decimal 1M = 1,000,000, not 1,048,576 tokens.
        for (tokens, dtype, payload, paged, vram, ram) in [
            (
                262_144,
                DType::Bf16,
                6_442_450_944,
                6_442_450_944,
                50_331_648,
                6_392_119_296,
            ),
            (
                1_000_000,
                DType::F8E4M3,
                12_288_000_000,
                12_290_359_296,
                25_165_824,
                12_265_193_472,
            ),
        ] {
            let shape = KvShape {
                dtype,
                ..KvShape::new(2, 256)
            };
            let rate = 12 * kv_bytes_per_token(sparse(), shape).unwrap();
            assert_eq!(rate * u64::from(tokens), payload);
            let mut c = KvCache::new(sparse(), shape, DEFAULT_BLOCK_TOKENS, 2048).unwrap();
            c.append(tokens).unwrap();
            let usage = c.memory_usage().unwrap();
            assert_eq!((12 * usage.vram, 12 * usage.ram), (vram, ram));
            assert_eq!(12 * (usage.vram + usage.ram), paged);
            // A full-residency configuration reproduces the §4.1 VRAM figure.
            let mut resident =
                KvCache::new(sparse(), shape, 256, tokens.next_multiple_of(256)).unwrap();
            resident.append(tokens).unwrap();
            assert_eq!(resident.memory_usage().unwrap().vram * 12, paged);
            println!(
                "Qwen3.8 {tokens} {dtype:?}: {rate} B/token, payload {payload} B, paged {paged} B, resident(2048) {vram} B, RAM {ram} B"
            );
        }
    }

    // Execute plans against small CPU buffers to check data, not just metadata.
    #[derive(Default)]
    struct CpuPages {
        vram: Vec<Vec<u32>>,
        ram: Vec<Vec<u32>>,
        next_token: u32,
    }

    impl CpuPages {
        fn execute(&mut self, ops: &[AppendOp], page_tokens: u32) {
            for &op in ops {
                match op {
                    AppendOp::Spill {
                        vram_slot,
                        ram_slot,
                        tokens,
                        ..
                    } => {
                        self.ram
                            .resize_with(self.ram.len().max(ram_slot + 1), Vec::new);
                        self.ram[ram_slot] = self.vram[vram_slot][..tokens as usize].to_vec();
                    }
                    AppendOp::Reload {
                        ram_slot,
                        vram_slot,
                        tokens,
                        ..
                    } => {
                        self.vram[vram_slot][..tokens as usize]
                            .copy_from_slice(&self.ram[ram_slot][..tokens as usize]);
                    }
                    AppendOp::Write {
                        vram_slot,
                        offset,
                        tokens,
                        ..
                    } => {
                        self.vram
                            .resize_with(self.vram.len().max(vram_slot + 1), || {
                                vec![u32::MAX; page_tokens as usize]
                            });
                        for token in
                            &mut self.vram[vram_slot][offset as usize..(offset + tokens) as usize]
                        {
                            *token = self.next_token;
                            self.next_token += 1;
                        }
                    }
                }
            }
        }

        fn read(&self, plan: &AttentionPlan) -> Vec<u32> {
            let staging: Vec<_> = plan
                .gathers
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    assert_eq!(i, g.staging_slot);
                    self.ram[g.ram_slot][..g.tokens as usize].to_vec()
                })
                .collect();
            plan.blocks
                .iter()
                .flat_map(|b| {
                    let page = match b.source {
                        AttentionSource::Vram { slot } => &self.vram[slot],
                        AttentionSource::Staging { slot } => &staging[slot],
                    };
                    page[b.start as usize..b.end as usize].iter().copied()
                })
                .collect()
        }
    }

    #[test]
    fn cpu_transfers_preserve_all_window_and_sparse_data() {
        let mut c = cache(sparse(), 4, 8);
        let mut pages = CpuPages::default();
        pages.execute(&c.append(23).unwrap(), 4);
        assert_eq!(
            pages.read(&c.attention(AttentionRequest::All).unwrap()),
            (0..23).collect::<Vec<_>>()
        );
        assert_eq!(
            pages.read(
                &c.attention(AttentionRequest::SlidingWindow { tokens: 10 })
                    .unwrap()
            ),
            (13..23).collect::<Vec<_>>()
        );
        assert_eq!(
            pages.read(&c.attention(AttentionRequest::Indexed(&[3, 0, 5])).unwrap()),
            [12, 13, 14, 15, 0, 1, 2, 3, 20, 21, 22]
        );
    }

    #[test]
    fn every_mtp_acceptance_length_preserves_prefix_and_future_writes() {
        for layer in [LayerCache::PagedKv, sparse()] {
            for prefix in 1..=13 {
                for accepted in 0..=4 {
                    let mut c = cache(layer, 4, 4);
                    let mut pages = CpuPages::default();
                    pages.execute(&c.append(prefix + 4).unwrap(), 4);
                    c.truncate(prefix + accepted).unwrap();
                    assert_eq!(
                        pages.read(&c.attention(AttentionRequest::All).unwrap()),
                        (0..prefix + accepted).collect::<Vec<_>>()
                    );
                    // New tokens replace rejected data, even across spills.
                    pages.next_token = 100;
                    pages.execute(&c.append(9).unwrap(), 4);
                    let expected: Vec<_> = (0..prefix + accepted).chain(100..109).collect();
                    assert_eq!(
                        pages.read(&c.attention(AttentionRequest::All).unwrap()),
                        expected
                    );
                    c.truncate(0).unwrap();
                    pages.next_token = 200;
                    pages.execute(&c.append(17).unwrap(), 4);
                    assert_eq!(
                        pages.read(&c.attention(AttentionRequest::All).unwrap()),
                        (200..217).collect::<Vec<_>>()
                    );
                    assert!(c.next_ram <= 6, "freed RAM slots should be reused");
                }
            }
        }
    }
}
