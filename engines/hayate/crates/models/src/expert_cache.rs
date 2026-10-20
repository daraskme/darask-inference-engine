//! Fixed-size expert slots, interval routing statistics, and bounded promotion.
//! GPU-format images are copied as bytes; CPU Q4Bf16 weights stay with the executor.

use std::collections::HashMap;

use hayate_core::{Error, Result};

use crate::offload::{ExpertStat, Placement, Tier};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertId {
    pub layer: u16,
    pub expert: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuExpert {
    pub expert: u16,
    pub slot: usize,
}

/// Each list preserves routing order, including repeated selections. The
/// executor retains the router's weights when joining CPU and GPU outputs.
#[derive(Debug, PartialEq, Eq)]
pub struct RoutedExperts {
    pub gpu: Vec<GpuExpert>,
    pub cpu: Vec<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Replacement {
    pub slot: usize,
    pub evict: ExpertId,
    pub load: ExpertId,
}

/// Only the cache creates plans. A routing call, interval boundary, or applied
/// plan invalidates older plans so a slot cannot be reassigned twice.
#[derive(Debug)]
pub struct SwapPlan {
    revision: u64,
    replacements: Vec<Replacement>,
}

impl SwapPlan {
    #[cfg(test)]
    pub fn replacements(&self) -> &[Replacement] {
        &self.replacements
    }
}

#[derive(Debug)]
pub struct ExpertCache {
    ids: Vec<ExpertId>,
    indices: HashMap<ExpertId, usize>,
    slots: Vec<usize>,
    expert_slots: Vec<Option<usize>>,
    counts: Vec<u64>,
    scores: Vec<f64>,
    slot_bytes: usize,
    #[cfg(test)]
    decay: f64,
    max_swaps: usize,
    revision: u64,
    #[cfg(any(test, feature = "cuda"))]
    recent: std::collections::VecDeque<Vec<u64>>,
}

impl ExpertCache {
    /// `stats` and `placement.tiers` have the same order. All experts have one
    /// GPU image size; every expert must also remain in RAM for CPU execution.
    #[cfg(test)]
    pub fn new(
        stats: &[ExpertStat],
        placement: &Placement,
        decay: f64,
        max_swaps: usize,
    ) -> Result<Self> {
        Self::build(stats, placement, decay, max_swaps, false, true)
    }

    /// The executor keeps only the selected host tier pinned and reads
    /// the remaining images from NVMe. Nonresident routes are returned in `cpu`.
    #[cfg(test)]
    pub fn with_disk(
        stats: &[ExpertStat],
        placement: &Placement,
        decay: f64,
        max_swaps: usize,
    ) -> Result<Self> {
        Self::build(stats, placement, decay, max_swaps, true, true)
    }

    /// Like `with_disk` for experts of different image sizes (GLM LOWBIT
    /// mixes): `slot_bytes` is the largest, and the caller pairs promotions
    /// with slots large enough (`plan_recent`'s `fits`).
    pub fn with_disk_sizes(
        stats: &[ExpertStat],
        placement: &Placement,
        decay: f64,
        max_swaps: usize,
    ) -> Result<Self> {
        Self::build(stats, placement, decay, max_swaps, true, false)
    }

    fn build(
        stats: &[ExpertStat],
        placement: &Placement,
        decay: f64,
        max_swaps: usize,
        allow_disk: bool,
        uniform: bool,
    ) -> Result<Self> {
        if stats.is_empty() || stats.len() != placement.tiers.len() {
            return Err(Error::InvalidPlan(
                "expert statistics/placement length mismatch".into(),
            ));
        }
        if !(0.0..=1.0).contains(&decay) {
            return Err(Error::InvalidPlan("expert decay must be in [0, 1]".into()));
        }
        let largest = stats.iter().map(|s| s.gpu_bytes).max().unwrap();
        let slot_bytes = usize::try_from(largest)
            .map_err(|_| Error::InvalidPlan("expert slot size overflow".into()))?;
        if slot_bytes == 0
            || stats.iter().any(|s| s.gpu_bytes == 0)
            || (uniform && stats.iter().any(|s| s.gpu_bytes != stats[0].gpu_bytes))
        {
            return Err(Error::InvalidPlan(
                "expert slots must have one nonzero size".into(),
            ));
        }
        let mut cache = Self {
            ids: Vec::with_capacity(stats.len()),
            indices: HashMap::with_capacity(stats.len()),
            slots: Vec::new(),
            expert_slots: vec![None; stats.len()],
            counts: vec![0; stats.len()],
            scores: vec![0.0; stats.len()],
            slot_bytes,
            #[cfg(test)]
            decay,
            max_swaps,
            revision: 0,
            #[cfg(any(test, feature = "cuda"))]
            recent: std::collections::VecDeque::new(),
        };
        for (i, (stat, tier)) in stats.iter().zip(&placement.tiers).enumerate() {
            let id = ExpertId {
                layer: stat.layer,
                expert: stat.expert,
            };
            if cache.indices.insert(id, i).is_some() {
                return Err(Error::InvalidPlan("duplicate expert ID".into()));
            }
            cache.ids.push(id);
            match tier {
                Tier::Gpu => {
                    cache.expert_slots[i] = Some(cache.slots.len());
                    cache.slots.push(i);
                }
                Tier::HostCompute => {}
                Tier::Disk if allow_disk => {}
                _ => {
                    return Err(Error::InvalidPlan(
                        "expert cache requires all experts in RAM".into(),
                    ));
                }
            }
        }
        cache.arena_bytes()?;
        Ok(cache)
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn arena_bytes(&self) -> Result<usize> {
        self.slot_bytes
            .checked_mul(self.slots.len())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| Error::InvalidPlan("expert arena size overflow".into()))
    }

    pub fn slot(&self, id: ExpertId) -> Option<usize> {
        self.indices.get(&id).and_then(|&i| self.expert_slots[i])
    }

    pub fn resident(&self, slot: usize) -> Option<ExpertId> {
        self.slots.get(slot).map(|&i| self.ids[i])
    }

    /// Records each routing occurrence and splits a layer's selections. Unknown
    /// IDs reject the entire call without changing statistics.
    pub fn route(&mut self, layer: u16, selected: &[u16]) -> Result<RoutedExperts> {
        let indices: Vec<usize> = selected
            .iter()
            .map(|&expert| {
                self.indices
                    .get(&ExpertId { layer, expert })
                    .copied()
                    .ok_or_else(|| Error::InvalidPlan(format!("unknown expert {layer}/{expert}")))
            })
            .collect::<Result<_>>()?;
        let mut routed = RoutedExperts {
            gpu: Vec::new(),
            cpu: Vec::new(),
        };
        for i in indices {
            self.counts[i] = self.counts[i].saturating_add(1);
            if let Some(slot) = self.expert_slots[i] {
                routed.gpu.push(GpuExpert {
                    expert: self.ids[i].expert,
                    slot,
                });
            } else {
                routed.cpu.push(self.ids[i].expert);
            }
        }
        self.revision += 1;
        Ok(routed)
    }

    /// score = decay * previous score + this interval's routing count. Only
    /// unused residents may be evicted; a promotion must improve the score.
    /// Planning leaves placement unchanged until `apply`/`enqueue` succeeds.
    #[cfg(test)]
    pub fn finish_interval(&mut self) -> SwapPlan {
        self.finish_interval_with_hysteresis(1.0)
    }

    #[cfg(test)]
    pub(crate) fn finish_interval_with_hysteresis(&mut self, gain: f64) -> SwapPlan {
        for (score, &count) in self.scores.iter_mut().zip(&self.counts) {
            *score = self.decay * *score + count as f64;
        }
        let mut victims: Vec<usize> = (0..self.slots.len())
            .filter(|&slot| self.counts[self.slots[slot]] == 0)
            .collect();
        victims.sort_by(|&a, &b| {
            self.scores[self.slots[a]]
                .total_cmp(&self.scores[self.slots[b]])
                .then(a.cmp(&b))
        });
        let mut candidates: Vec<usize> = (0..self.ids.len())
            .filter(|&i| self.expert_slots[i].is_none() && self.counts[i] != 0)
            .collect();
        candidates.sort_by(|&a, &b| self.scores[b].total_cmp(&self.scores[a]).then(a.cmp(&b)));
        let replacements = victims
            .into_iter()
            .zip(candidates)
            .take(self.max_swaps)
            .take_while(|&(slot, i)| self.scores[i] > gain * self.scores[self.slots[slot]])
            .map(|(slot, i)| Replacement {
                slot,
                evict: self.ids[self.slots[slot]],
                load: self.ids[i],
            })
            .collect();
        self.counts.fill(0);
        self.revision += 1;
        SwapPlan {
            revision: self.revision,
            replacements,
        }
    }

    // GLM records every token, including while a prepared batch is in flight.
    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn record_recent_token(&mut self, window: usize) {
        if self.recent.len() == window {
            let old = self.recent.pop_front().unwrap();
            for (score, count) in self.scores.iter_mut().zip(old) {
                *score -= count as f64;
            }
        }
        for (score, &count) in self.scores.iter_mut().zip(&self.counts) {
            *score += count as f64;
        }
        self.recent.push_back(self.counts.clone());
        self.counts.fill(0);
    }

    /// Forgets the rolling window (GLM context placement starts a new one,
    /// so the state after a placement depends on the context only).
    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn reset_recent(&mut self) {
        self.recent.clear();
        self.scores.fill(0.);
        self.counts.fill(0);
    }

    #[cfg(any(test, feature = "cuda"))]
    // GLM's rolling-window counts require an absolute gain as well as a strict
    // improvement. Zero preserves the previous policy for comparison runs.
    #[cfg(test)]
    pub(crate) fn plan_recent(&self, max_bytes: usize, min_gain: usize) -> Vec<Replacement> {
        self.plan_recent_fit(max_bytes, min_gain, |_, _| true)
    }

    /// `plan_recent` for slots of different sizes: the best candidates in
    /// order, each against the lowest-scored resident whose slot `fits(slot,
    /// expert index)` it (the first fit; equal scores by slot).
    pub(crate) fn plan_recent_fit(
        &self,
        max_bytes: usize,
        min_gain: usize,
        fits: impl Fn(usize, usize) -> bool,
    ) -> Vec<Replacement> {
        let mut victims: Vec<_> = (0..self.slots.len()).collect();
        victims.sort_by(|&a, &b| {
            self.scores[self.slots[a]]
                .total_cmp(&self.scores[self.slots[b]])
                .then(a.cmp(&b))
        });
        let mut candidates: Vec<_> = (0..self.ids.len())
            .filter(|&i| self.expert_slots[i].is_none() && self.scores[i] > 0.)
            .collect();
        candidates.sort_by(|&a, &b| self.scores[b].total_cmp(&self.scores[a]).then(a.cmp(&b)));
        let limit = self.max_swaps.min(max_bytes / (2 * self.slot_bytes));
        let mut taken = vec![false; self.slots.len()];
        let mut plan = Vec::new();
        for i in candidates {
            if plan.len() >= limit {
                break;
            }
            let Some(&slot) = victims.iter().find(|&&s| !taken[s] && fits(s, i)) else {
                continue;
            };
            let gain = self.scores[i] - self.scores[self.slots[slot]];
            if gain > 0. && gain >= min_gain as f64 {
                taken[slot] = true;
                plan.push(Replacement {
                    slot,
                    evict: self.ids[self.slots[slot]],
                    load: self.ids[i],
                });
            } else if victims.iter().find(|&&s| !taken[s]) == Some(&slot) {
                // The weakest resident is not beaten, and candidates only get worse.
                break;
            }
        }
        plan
    }

    /// Rolling-window score of expert `i` (statistics order).
    #[cfg(feature = "cuda")]
    pub(crate) fn score(&self, i: usize) -> f64 {
        self.scores[i]
    }

    /// Replaces the resident set (elastic VRAM budget). `gpu` lists expert
    /// indices in statistics order, one per new slot. Routing statistics are
    /// kept; outstanding plans become stale.
    #[cfg(feature = "cuda")]
    pub(crate) fn reassign(&mut self, gpu: &[usize]) -> Result<()> {
        let mut seen = vec![false; self.ids.len()];
        for &i in gpu {
            if i >= self.ids.len() || std::mem::replace(&mut seen[i], true) {
                return Err(Error::InvalidPlan("invalid expert reassignment".into()));
            }
        }
        self.expert_slots.fill(None);
        for (slot, &i) in gpu.iter().enumerate() {
            self.expert_slots[i] = Some(slot);
        }
        self.slots = gpu.to_vec();
        self.revision += 1;
        Ok(())
    }

    // A prepared copy may span routing calls. Validate the unchanged old mapping,
    // then publish only after the executor has retired every previous GPU use.
    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn apply_prepared(&mut self, swap: Replacement) -> Result<()> {
        let plan = SwapPlan {
            revision: self.revision,
            replacements: vec![swap],
        };
        self.apply(&plan)
    }

    fn validate_plan(&self, plan: &SwapPlan) -> Result<()> {
        if plan.revision != self.revision {
            return Err(Error::InvalidPlan("stale expert swap plan".into()));
        }
        // Revision numbers alone do not distinguish plans from different caches.
        let mut slots = std::collections::HashSet::new();
        let mut loads = std::collections::HashSet::new();
        for swap in &plan.replacements {
            if self.resident(swap.slot) != Some(swap.evict)
                || !self.indices.contains_key(&swap.load)
                || self.slot(swap.load).is_some()
                || !slots.insert(swap.slot)
                || !loads.insert(swap.load)
            {
                return Err(Error::InvalidPlan("invalid expert swap plan".into()));
            }
        }
        if plan.replacements.len() > self.max_swaps {
            return Err(Error::InvalidPlan("expert swap limit exceeded".into()));
        }
        Ok(())
    }

    /// Commits metadata after the caller has arranged the slot copies. CPU-only
    /// tests can use this directly; GLM execution copies the planned slots before committing the plan.
    pub fn apply(&mut self, plan: &SwapPlan) -> Result<()> {
        self.validate_plan(plan)?;
        for swap in &plan.replacements {
            let old = self.slots[swap.slot];
            let new = self.indices[&swap.load];
            self.expert_slots[old] = None;
            self.expert_slots[new] = Some(swap.slot);
            self.slots[swap.slot] = new;
        }
        self.revision += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offload::{Budget, DiskPolicy, place_experts};

    fn stats(count: u16) -> Vec<ExpertStat> {
        (0..count)
            .map(|expert| ExpertStat {
                layer: 0,
                expert,
                freq: 1.0 / (expert as f32 + 1.0),
                gpu_bytes: 32,
                host_bytes: 36,
            })
            .collect()
    }

    fn cache(count: u16, slots: u64, decay: f64, limit: usize) -> ExpertCache {
        let stats = stats(count);
        let p = place_experts(
            &stats,
            Budget {
                gpu: slots * 32,
                host: count as u64 * 36,
            },
            DiskPolicy::Refuse,
        )
        .unwrap();
        ExpertCache::new(&stats, &p, decay, limit).unwrap()
    }

    fn id(expert: u16) -> ExpertId {
        ExpertId { layer: 0, expert }
    }

    #[test]
    fn recent_window_prefers_hottest_miss_over_used_resident_and_expires() {
        let mut c = cache(4, 1, 0.9, 8);
        c.route(0, &[0, 1, 1, 2]).unwrap();
        c.record_recent_token(2);
        let plan = c.plan_recent(64, 0);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].evict, id(0));
        assert_eq!(plan[0].load, id(1));
        // Still usable until a prepared copy is published; routes invalidate
        // synchronous plans but do not invalidate the unchanged prepared slot.
        assert_eq!(c.slot(id(0)), Some(0));
        c.route(0, &[0]).unwrap();
        c.record_recent_token(2);
        c.apply_prepared(plan[0]).unwrap();
        assert_eq!(c.slot(id(1)), Some(0));
        assert!(c.apply_prepared(plan[0]).is_err());
        c.route(0, &[3, 3, 3]).unwrap();
        c.record_recent_token(2);
        assert_eq!(c.scores, [1., 0., 0., 3.]);
        assert_eq!(c.plan_recent(64, 0)[0].load, id(3));
    }

    #[test]
    fn recent_promotions_obey_transfer_bytes_and_keep_counts_during_pending_batch() {
        let mut c = cache(6, 2, 0.9, 8);
        c.route(0, &[2, 3, 3]).unwrap();
        c.record_recent_token(32);
        assert!(c.plan_recent(63, 0).is_empty());
        let pending = c.plan_recent(64, 0);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].load, id(3));
        // Three further tokens while the worker is busy must still contribute.
        for _ in 0..3 {
            c.route(0, &[4, 4]).unwrap();
            c.record_recent_token(32);
        }
        c.apply_prepared(pending[0]).unwrap();
        let next = c.plan_recent(128, 0);
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].load, id(4));
        assert_eq!(next[0].evict, id(1));
    }

    #[test]
    fn glm53_recent_hysteresis_suppresses_churn_and_promotes_sustained_routes() {
        let mut c = cache(6, 2, 0.9, 1);
        for _ in 0..32 {
            c.route(0, &[0, 1, 2, 3]).unwrap();
            c.record_recent_token(64);
        }
        // All four experts have equal counts; ties never trigger a swap.
        assert!(c.plan_recent(128, 0).is_empty());
        for gain in 1..=4 {
            c.route(0, &[2, 3]).unwrap();
            c.record_recent_token(64);
            let plan = c.plan_recent(128, 4);
            if gain < 4 {
                assert!(plan.is_empty());
                assert_eq!(c.plan_recent(128, 0).len(), 1);
            } else {
                // Even with a two-image byte budget, max_swaps remains one.
                assert_eq!(plan.len(), 1);
                assert_eq!((plan[0].evict, plan[0].load), (id(0), id(2)));
                assert!(c.plan_recent(63, 4).is_empty());
                c.apply_prepared(plan[0]).unwrap();
            }
        }
        // A sustained miss can replace the next cold resident, then the newly
        // resident hot experts remain in place. Expired history must not churn.
        let next = c.plan_recent(64, 4);
        assert_eq!((next[0].evict, next[0].load), (id(1), id(3)));
        c.apply_prepared(next[0]).unwrap();
        for _ in 0..128 {
            c.route(0, &[2, 3]).unwrap();
            c.record_recent_token(64);
            assert!(c.plan_recent(64, 4).is_empty());
        }
        assert_eq!(c.scores, [0., 0., 64., 64., 0., 0.]);
    }

    #[test]
    fn glm53_short_window_follows_a_route_shift_and_larger_batch_catches_up() {
        // A phase shift with equal hot-set sizes; publish only at the next
        // token boundary, as the asynchronous GLM executor does.
        let replay = |window, limit| {
            let mut c = cache(128, 64, 0.9, limit);
            let mut pending = Vec::new();
            let mut hits = 0;
            for token in 0..64 {
                for swap in pending.drain(..) {
                    c.apply_prepared(swap).unwrap();
                }
                let ids: Vec<_> = if token < 32 {
                    (0..64).collect()
                } else {
                    (64..128).collect()
                };
                let routes = c.route(0, &ids).unwrap();
                if token >= 32 {
                    hits += routes.gpu.len();
                }
                c.record_recent_token(window);
                pending = c.plan_recent(2 * 32 * limit, 0);
                assert!(pending.len() <= limit);
                let loads: std::collections::HashSet<_> = pending.iter().map(|s| s.load).collect();
                assert_eq!(loads.len(), pending.len());
            }
            hits
        };
        let old = replay(32, 8);
        let window_only = replay(8, 8);
        let new = replay(8, 32);
        assert!(window_only > old);
        assert!(new > window_only);
        println!(
            "GLM synthetic shift: stage8_hits={old}/2048 window8_hits={window_only}/2048 window8_swaps32_hits={new}/2048"
        );
    }

    #[test]
    fn disk_backed_cache_can_promote_without_pinning_all_images() {
        let stats = stats(3);
        let placement =
            place_experts(&stats, Budget { gpu: 32, host: 0 }, DiskPolicy::Allow).unwrap();
        let mut cache = ExpertCache::with_disk(&stats, &placement, 0., 1).unwrap();
        assert_eq!(cache.route(0, &[2, 2]).unwrap().cpu, [2, 2]);
        let plan = cache.finish_interval();
        assert_eq!(plan.replacements()[0].load, id(2));
        cache.apply(&plan).unwrap();
        assert_eq!(cache.slot(id(2)), Some(0));
        assert!(ExpertCache::new(&stats, &placement, 0., 1).is_err());
    }

    #[test]
    fn initializes_from_placement_and_splits_routing_by_layer() {
        let mut stats = stats(4);
        stats[2].layer = 1;
        stats[2].expert = 0;
        let p = place_experts(&stats, Budget { gpu: 64, host: 144 }, DiskPolicy::Refuse).unwrap();
        let mut c = ExpertCache::new(&stats, &p, 0.5, 2).unwrap();
        assert_eq!(c.arena_bytes().unwrap(), 64);
        assert_eq!(
            c.route(0, &[3, 0, 1, 3]).unwrap(),
            RoutedExperts {
                gpu: vec![
                    GpuExpert { expert: 0, slot: 0 },
                    GpuExpert { expert: 1, slot: 1 }
                ],
                cpu: vec![3, 3],
            }
        );
        assert_eq!(c.route(1, &[0]).unwrap().cpu, [0]);
        assert_eq!(
            c.slot(ExpertId {
                layer: 1,
                expert: 0
            }),
            None
        );
    }

    #[test]
    fn hysteresis_retains_calibrated_residents_until_gain_is_large_enough() {
        let mut c = cache(3, 1, 0.5, 1);
        c.route(0, &[0; 8]).unwrap();
        c.finish_interval();
        c.route(0, &[1; 5]).unwrap();
        assert!(
            c.finish_interval_with_hysteresis(1.25)
                .replacements()
                .is_empty()
        );
        c.route(0, &[1; 5]).unwrap();
        let plan = c.finish_interval_with_hysteresis(1.25);
        assert_eq!(plan.replacements()[0].load, id(1));
        c.apply(&plan).unwrap();
        assert_eq!(c.slot(id(1)), Some(0));
    }

    #[test]
    fn keeps_used_residents_and_promotes_hottest_misses() {
        let mut c = cache(5, 2, 0.5, 2);
        c.route(0, &[0, 2, 2, 2, 3, 3, 4]).unwrap();
        let plan = c.finish_interval();
        assert_eq!(
            plan.replacements(),
            &[Replacement {
                slot: 1,
                evict: id(1),
                load: id(2)
            }]
        );
        assert_eq!(c.slot(id(2)), None);
        c.apply(&plan).unwrap();
        assert_eq!(c.slot(id(0)), Some(0));
        assert_eq!(c.slot(id(2)), Some(1));
        assert_eq!(c.slot(id(1)), None);
    }

    #[test]
    fn bounds_swaps_and_never_reuses_a_slot_or_expert() {
        let mut c = cache(8, 3, 0.5, 2);
        c.route(0, &[3, 4, 4, 5, 5, 5, 6, 6, 6, 6]).unwrap();
        let plan = c.finish_interval();
        assert_eq!(plan.replacements().len(), 2);
        assert_eq!(plan.replacements()[0].load, id(6));
        assert_eq!(plan.replacements()[1].load, id(5));
        c.apply(&plan).unwrap();
        let residents: std::collections::HashSet<_> =
            (0..3).map(|s| c.resident(s).unwrap()).collect();
        assert_eq!(residents.len(), 3);
        for s in 0..3 {
            assert_eq!(c.slot(c.resident(s).unwrap()), Some(s));
        }
    }

    #[test]
    fn decay_retains_recent_hot_experts_then_allows_a_shift() {
        let mut c = cache(3, 1, 0.5, 1);
        c.route(0, &[0; 16]).unwrap();
        assert!(c.finish_interval().replacements().is_empty());
        c.route(0, &[1]).unwrap();
        assert!(c.finish_interval().replacements().is_empty());
        c.route(0, &[1; 8]).unwrap();
        let plan = c.finish_interval();
        assert_eq!(plan.replacements()[0].load, id(1));
        c.apply(&plan).unwrap();
        assert_eq!(c.slot(id(1)), Some(0));
    }

    #[test]
    fn no_churn_on_empty_intervals_or_equal_scores() {
        let mut c = cache(3, 1, 1.0, 1);
        assert!(c.finish_interval().replacements().is_empty());
        c.route(0, &[0]).unwrap();
        c.finish_interval();
        c.route(0, &[1]).unwrap();
        assert!(c.finish_interval().replacements().is_empty());
    }

    #[test]
    fn handles_zero_slots_zero_limit_and_full_residency() {
        for (slots, limit) in [(0, 2), (1, 0), (3, 2)] {
            let mut c = cache(3, slots, 0.0, limit);
            c.route(0, &[2]).unwrap();
            assert!(c.finish_interval().replacements().is_empty());
        }
    }

    #[test]
    fn rejects_stale_plans_before_mutation() {
        let mut c = cache(3, 1, 0.0, 1);
        c.route(0, &[2]).unwrap();
        let plan = c.finish_interval();
        c.route(0, &[0]).unwrap();
        assert!(c.apply(&plan).is_err());
        assert_eq!(c.resident(0), Some(id(0)));
        let plan = c.finish_interval();
        c.apply(&plan).unwrap();
        assert!(c.apply(&plan).is_err());
    }

    #[test]
    fn rejects_unknown_routes_without_partial_counts() {
        let mut c = cache(3, 1, 0.0, 1);
        assert!(c.route(0, &[1, 99]).is_err());
        assert!(c.finish_interval().replacements().is_empty());
    }

    #[test]
    fn validates_uniform_sizes_ids_tiers_and_decay() {
        let mut stats = stats(3);
        let p = place_experts(&stats, Budget { gpu: 32, host: 108 }, DiskPolicy::Refuse).unwrap();
        for decay in [f64::NAN, -0.1, 1.1] {
            assert!(ExpertCache::new(&stats, &p, decay, 1).is_err());
        }
        stats[1].gpu_bytes = 64;
        assert!(ExpertCache::new(&stats, &p, 0.5, 1).is_err());
        stats[1].gpu_bytes = 32;
        stats[1].expert = 0;
        assert!(ExpertCache::new(&stats, &p, 0.5, 1).is_err());
        stats[1].expert = 1;
        let mut invalid = p.clone();
        invalid.tiers[2] = Tier::Disk;
        assert!(ExpertCache::new(&stats, &invalid, 0.5, 1).is_err());
        invalid.tiers.pop();
        assert!(ExpertCache::new(&stats, &invalid, 0.5, 1).is_err());
        for stat in &mut stats {
            stat.gpu_bytes = 0;
        }
        assert!(ExpertCache::new(&stats, &p, 0.5, 1).is_err());
    }

    #[test]
    fn repeated_intervals_preserve_bijection_and_used_residents() {
        let mut c = cache(32, 8, 0.75, 3);
        let mut random = 17u32;
        for _ in 0..128 {
            let selected: Vec<u16> = (0..24)
                .map(|_| {
                    random ^= random << 13;
                    random ^= random >> 17;
                    random ^= random << 5;
                    (random % 32) as u16
                })
                .collect();
            let routed = c.route(0, &selected).unwrap();
            let plan = c.finish_interval();
            assert!(plan.replacements().len() <= 3);
            for used in routed.gpu {
                assert!(!plan.replacements().iter().any(|s| s.slot == used.slot));
            }
            c.apply(&plan).unwrap();
            let residents: std::collections::HashSet<_> =
                (0..8).map(|s| c.resident(s).unwrap()).collect();
            assert_eq!(residents.len(), 8);
            for expert in 0..32 {
                if let Some(slot) = c.slot(id(expert)) {
                    assert_eq!(c.resident(slot), Some(id(expert)));
                }
            }
        }
    }

    #[test]
    #[ignore = "CPU timing: run in release mode with --ignored --nocapture"]
    fn measure_glm_interval_planning() {
        let stats: Vec<_> = (0..42)
            .flat_map(|layer| {
                (0..288).map(move |expert| ExpertStat {
                    layer,
                    expert,
                    freq: 1.0 / (expert as f32 + 1.0),
                    gpu_bytes: 32,
                    host_bytes: 36,
                })
            })
            .collect();
        let p = place_experts(
            &stats,
            Budget {
                gpu: 5806 * 32,
                host: 12096 * 36,
            },
            DiskPolicy::Refuse,
        )
        .unwrap();
        let mut c = ExpertCache::new(&stats, &p, 0.75, 8).unwrap();
        let mut timings = Vec::new();
        for interval in 0..110 {
            for _ in 0..128 {
                for layer in 0..42 {
                    let selected: Vec<_> = (0..8)
                        .map(|e| (interval * 7 + layer * 3 + e) % 288)
                        .collect();
                    c.route(layer, &selected).unwrap();
                }
            }
            let start = std::time::Instant::now();
            let plan = c.finish_interval();
            c.apply(&plan).unwrap();
            if interval >= 10 {
                timings.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        timings.sort_by(f64::total_cmp);
        println!(
            "GLM planning+apply: 12096 experts, 5806 slots, 128 tokens/interval, max 8 swaps; 10 warmup + 100 repeats; p50={:.3} ms p95={:.3} ms",
            timings[50], timings[95]
        );
    }
}
