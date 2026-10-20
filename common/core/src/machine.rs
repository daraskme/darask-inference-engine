//! The two machines Hayate is built for, and nothing else.
//!
//! Instead of probing capabilities and branching, Hayate hard-codes its
//! targets and refuses to start anywhere else. Probing only *verifies* that
//! the running host matches one of these profiles; actual free VRAM is
//! re-measured after CUDA context creation and used for admission.
//!
//! The machines are separate hosts on the LAN, not two GPUs in one box, so
//! they never share tensors at layer granularity — only whole jobs or whole
//! pipeline stages (see `docs/ARCHITECTURE.md` §3).
//!
//! Units: all `*_bytes` fields are bytes; GiB = 2^30. Bandwidths are bytes/s
//! as published by the vendor (10^9-based).

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineId {
    /// RTX PRO 6000 Blackwell (96 GB) + i9-12900KS + 128 GB DDR4-3200 (4x32 GB, 2 channels).
    Main,
    /// `ai-workstation` (192.168.1.4): RTX 5060 Ti 16 GB + Ryzen 7 8700F + 32 GB.
    Sub,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuSpec {
    /// Substring matched against `cudaDeviceProp::name`.
    pub name_match: &'static str,
    /// Both GPUs are compute capability 12.0; kernels are built for `sm_120a`.
    pub sm: (u32, u32),
    /// Total VRAM as reported by the driver (`nvidia-smi memory.total`).
    pub vram_bytes: u64,
    /// Planning reserve for CUDA context, cuBLAS/NVENC workspaces and display.
    /// A budget, not a measurement: admission uses `cudaMemGetInfo` after init.
    pub reserve_bytes: u64,
    /// Vendor-spec DRAM bandwidth. Measured values live in `bench/`.
    pub dram_bw: u64,
    /// Host link (generation, lanes) as reported by the driver.
    pub pcie: (u8, u8),
}

/// SIMD level the CPU kernels are compiled for on each machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuIsa {
    /// Alder Lake: AVX2 + AVX-VNNI (256-bit), no AVX-512.
    Avx2Vnni,
    /// Zen 4: AVX-512 F/BW/VL/VNNI/BF16 (executed as 2×256-bit).
    Avx512Zen4,
}

/// Which logical CPUs run hot loops (CPU expert GEMV, n-gram gather) and
/// which take I/O, prefetch and the HTTP server.
#[derive(Clone, Copy, Debug)]
pub struct CpuCores {
    pub isa: CpuIsa,
    pub compute: &'static [u16],
    pub service: &'static [u16],
}

#[derive(Clone, Copy, Debug)]
pub struct Machine {
    pub id: MachineId,
    pub gpu: GpuSpec,
    pub cpu: CpuCores,
    pub host_ram_bytes: u64,
    /// Negotiated LAN link speed, bits/s.
    pub lan_bps: u64,
}

pub const MAIN: Machine = Machine {
    id: MachineId::Main,
    gpu: GpuSpec {
        name_match: "RTX PRO 6000 Blackwell",
        sm: (12, 0),
        vram_bytes: 97_887 * MIB,
        reserve_bytes: 2 * GIB,
        dram_bw: 1_792_000_000_000,
        pcie: (5, 16),
    },
    // i9-12900KS, verified with `lscpu -e` on this host: P-core hyperthread
    // pairs are (0,1)…(14,15), E-cores are 16-23. One thread per P-core does
    // compute; E-cores do everything else.
    cpu: CpuCores {
        isa: CpuIsa::Avx2Vnni,
        compute: &[0, 2, 4, 6, 8, 10, 12, 14],
        service: &[16, 17, 18, 19, 20, 21, 22, 23],
    },
    host_ram_bytes: 128 * GIB,
    lan_bps: 2_500_000_000,
};

pub const SUB: Machine = Machine {
    id: MachineId::Sub,
    gpu: GpuSpec {
        name_match: "RTX 5060 Ti",
        sm: (12, 0),
        vram_bytes: 16_311 * MIB,
        reserve_bytes: GIB,
        dram_bw: 448_000_000_000,
        // Driver reports max x16, but GB206 is electrically x8 and the 8700F is
        // Gen4: plan with Gen4 x8 (~15.75 GB/s) until the negotiated link is read.
        pcie: (4, 8),
    },
    // Ryzen 7 8700F, 8 cores / 16 threads. TODO(sub-pc): confirm SMT sibling
    // numbering with `lscpu -e`; assumed 0-7 = first thread of each core.
    cpu: CpuCores {
        isa: CpuIsa::Avx512Zen4,
        compute: &[0, 1, 2, 3, 4, 5, 6],
        service: &[7],
    },
    host_ram_bytes: 31 * GIB,
    // Sub PC NIC is 1GbE; the main PC's 2.5GbE link is limited by it.
    lan_bps: 1_000_000_000,
};

impl Machine {
    pub const fn get(id: MachineId) -> &'static Machine {
        match id {
            MachineId::Main => &MAIN,
            MachineId::Sub => &SUB,
        }
    }

    /// Planning budget before measurement; see `GpuSpec::reserve_bytes`.
    pub const fn planned_vram(&self) -> u64 {
        self.gpu.vram_bytes - self.gpu.reserve_bytes
    }

    /// One-way LAN line rate between the two machines, bytes/s (before
    /// TCP/IP overhead; expect ~94% of this in practice).
    pub const fn lan_link_bytes_per_s() -> u64 {
        let slower = if MAIN.lan_bps < SUB.lan_bps {
            MAIN.lan_bps
        } else {
            SUB.lan_bps
        };
        slower / 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planned_vram() {
        assert_eq!(
            Machine::get(MachineId::Main).planned_vram(),
            97_887 * MIB - 2 * GIB
        );
        assert_eq!(
            Machine::get(MachineId::Sub).planned_vram(),
            16_311 * MIB - GIB
        );
    }

    #[test]
    fn lan_is_limited_by_sub_nic() {
        assert_eq!(Machine::lan_link_bytes_per_s(), 125_000_000);
    }
}
