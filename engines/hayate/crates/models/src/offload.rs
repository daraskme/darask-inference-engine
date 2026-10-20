//! Placement planning for GLM-5.3-Flash experts that do not fit the GPU.
//!
//! This crate only decides where bytes live. Executing CPU experts and
//! overlapping them with GPU work is done by the executor in `hayate-models`
//! (ARCHITECTURE.md §5.1), so this crate depends on nothing but `core`.

use hayate_core::{Error, Result};

/// Where a weight block lives during inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Resident in VRAM, computed on GPU.
    Gpu,
    /// Host RAM, computed on CPU — cold MoE experts.
    HostCompute,
    /// Memory-mapped from NVMe. Only with `DiskPolicy::Allow`.
    Disk,
}

/// Routing frequency of one expert, measured by `python-tools/calibrate`.
#[derive(Clone, Copy, Debug)]
pub struct ExpertStat {
    pub layer: u16,
    pub expert: u16,
    /// Fraction of tokens routed to this expert, in [0, 1].
    pub freq: f32,
    /// Bytes in the GPU format (NVFP4).
    pub gpu_bytes: u64,
    /// Bytes in the CPU format (`QuantSpec::Q4Bf16`).
    pub host_bytes: u64,
}

/// Disk-resident experts stall whole layers when selected, so they are
/// refused unless explicitly allowed (e.g. for a throughput-insensitive run).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiskPolicy {
    Refuse,
    Allow,
}

/// Budgets left *after* KV/state caches, shared weights and workspaces.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub gpu: u64,
    pub host: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    pub tiers: Vec<Tier>,
    pub gpu_used: u64,
    pub host_used: u64,
    /// Expected bytes read from host RAM per token by CPU experts:
    /// Σ freq × host_bytes over `HostCompute` experts, summed over layers.
    /// Divide host bandwidth by this for the CPU-side decode ceiling (§9.3).
    pub host_bytes_per_token: f64,
}

/// Fill VRAM with the hottest experts first; the rest go to host RAM.
pub fn place_experts(stats: &[ExpertStat], budget: Budget, disk: DiskPolicy) -> Result<Placement> {
    if let Some(s) = stats.iter().find(|s| !(0.0..=1.0).contains(&s.freq)) {
        return Err(Error::InvalidPlan(format!(
            "expert {}/{}: frequency {} outside [0, 1]",
            s.layer, s.expert, s.freq
        )));
    }

    let mut order: Vec<usize> = (0..stats.len()).collect();
    order.sort_by(|&a, &b| stats[b].freq.total_cmp(&stats[a].freq));

    let mut p = Placement {
        tiers: vec![Tier::Disk; stats.len()],
        gpu_used: 0,
        host_used: 0,
        host_bytes_per_token: 0.0,
    };
    for i in order {
        let s = &stats[i];
        if s.gpu_bytes <= budget.gpu - p.gpu_used {
            p.gpu_used += s.gpu_bytes;
            p.tiers[i] = Tier::Gpu;
        } else if s.host_bytes <= budget.host - p.host_used {
            p.host_used += s.host_bytes;
            p.tiers[i] = Tier::HostCompute;
            p.host_bytes_per_token += s.freq as f64 * s.host_bytes as f64;
        } else if disk == DiskPolicy::Refuse {
            return Err(Error::OutOfMemory {
                buffer: "moe experts (host tier)",
                requested: s.host_bytes,
                available: budget.host - p.host_used,
            });
        }
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(expert: u16, freq: f32) -> ExpertStat {
        ExpertStat {
            layer: 0,
            expert,
            freq,
            gpu_bytes: 10,
            host_bytes: 12,
        }
    }

    #[test]
    fn hottest_experts_go_to_gpu() {
        let stats = [stat(0, 0.05), stat(1, 0.20), stat(2, 0.01)];
        let p = place_experts(&stats, Budget { gpu: 10, host: 24 }, DiskPolicy::Refuse).unwrap();
        assert_eq!(p.tiers, [Tier::HostCompute, Tier::Gpu, Tier::HostCompute]);
        assert_eq!((p.gpu_used, p.host_used), (10, 24));
        assert!((p.host_bytes_per_token - 0.06 * 12.0).abs() < 1e-6);
    }

    #[test]
    fn overflow_is_an_error_unless_disk_allowed() {
        let stats = [stat(0, 0.2), stat(1, 0.1), stat(2, 0.05)];
        let budget = Budget { gpu: 10, host: 12 };
        assert!(place_experts(&stats, budget, DiskPolicy::Refuse).is_err());
        let p = place_experts(&stats, budget, DiskPolicy::Allow).unwrap();
        assert_eq!(p.tiers[2], Tier::Disk);
    }

    #[test]
    fn rejects_bad_frequencies() {
        let budget = Budget {
            gpu: 100,
            host: 100,
        };
        assert!(place_experts(&[stat(0, f32::NAN)], budget, DiskPolicy::Allow).is_err());
        assert!(place_experts(&[stat(0, -0.1)], budget, DiskPolicy::Allow).is_err());
    }
}
