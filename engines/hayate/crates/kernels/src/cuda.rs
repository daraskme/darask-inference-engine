//! CUDA kernels for `sm_120a` and their Rust bindings.
//!
//! Kernel sources live in `cu/`, grouped by family. Every kernel is written
//! for the shapes of the six supported models; generic fallbacks are not
//! kept around. `docs/ARCHITECTURE.md` §6 has the full inventory.

/// Kernel families, one directory each under `cu/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// Dense FP8/NVFP4/BF16 GEMM, MoE grouped GEMM, W4A4 + low-rank correction.
    Gemm,
    /// Prefill/decode attention, SWA, QSA, GLM sparse attention, video block-sparse, FP8/FP4 attention.
    Attention,
    /// Gated DeltaNet (chunked prefill + recurrent decode), GLM linear attention.
    LinearAttention,
    /// Qwen Gated Residual (4 streams, FP8 state) and GLM mHC.
    Residual,
    /// Router top-k, permute/unpermute, shared-expert fusion.
    Moe,
    /// AdaLN modulation, 3D MM-RoPE, gated sigmoid attention epilogues.
    Dit,
    /// Conv2d/3d, spatio-temporal tiling, upscale decode, audio VAE.
    Vae,
    /// RMSNorm/LayerNorm, RoPE, activations, sampling.
    Elementwise,
}

#[cfg(feature = "cuda")]
pub mod ffi {
    //! Raw entry points exported by `cu/`. Contract for kernel launches:
    //! - enqueues on `stream` and never allocates or synchronizes;
    //! - pointers must stay valid until the stream passes this launch;
    //! - the return value is `cudaGetLastError()` after the launch;
    //!   but `cudaGetLastError` can also return an earlier asynchronous error,
    //!   so a non-zero code is classified by value, not by where it was seen;
    //!   sticky errors end the process (see `Error::CudaAsync`).

    use std::ffi::c_void;

    pub type Stream = *mut c_void;

    unsafe extern "C" {
        /// Startup-only query: SM count × resident blocks for the bandwidth kernel.
        /// `blocks` must be writable. Returns a CUDA runtime status.
        pub fn hayate_bandwidth_blocks(blocks: *mut i32) -> i32;

        /// DRAM bandwidth probe: copies `bytes` (positive multiple of 16).
        /// `blocks` is the positive startup result of `hayate_bandwidth_blocks`.
        /// Both allocations must be at least `bytes` long, 16-byte aligned,
        /// and disjoint (`__restrict__`). Reports read + write traffic.
        pub fn hayate_bandwidth_copy(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            blocks: i32,
            stream: Stream,
        ) -> i32;

        /// `y[r] = x[r] / rms(x[r]) * w`, bf16 in/out, fp32 accumulate.
        /// `rows, cols > 0`; no pointer may alias another (`__restrict__`).
        pub fn hayate_rmsnorm_bf16(
            y: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            rows: u32,
            cols: u32,
            eps: f32,
            stream: Stream,
        ) -> i32;
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::ffi;
    use crate::cpu::bf16_to_f32;
    use crate::runtime::{Device, check};
    use hayate_core::Result;

    fn bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        // Round to nearest, ties to even, matching __float2bfloat16.
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn bytes(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    /// Run explicitly with:
    /// cargo test -p hayate-kernels --features cuda rmsnorm_gpu -- --ignored --nocapture
    #[test]
    #[ignore = "requires one of Hayate's CC 12.0 GPUs; see test source for command"]
    fn rmsnorm_gpu() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let rows = 5;
        let eps = 1e-5;
        let mut max_error = 0.0f32;
        for cols in [1, 31, 32, 255, 256, 257, 4096] {
            let x: Vec<u16> = (0..rows * cols)
                .map(|i| {
                    let row = i / cols;
                    let value = match row {
                        0 => 0.0,
                        1 => ((i % 17) as f32 - 8.0) * 1e-4,
                        2 => ((i % 71) as f32 - 35.0) / 8.0,
                        3 => ((i % 97) as f32 - 48.0) * 32.0,
                        _ => ((i % 31) as f32 - 15.0) / 7.0,
                    };
                    bf16(value)
                })
                .collect();
            let w: Vec<u16> = (0..cols)
                .map(|i| bf16(((i % 23) as f32 - 11.0) / 8.0))
                .collect();
            let mut dx = device.alloc(x.len() * 2)?;
            let mut dw = device.alloc(w.len() * 2)?;
            let mut dy = device.alloc(x.len() * 2)?;
            stream.copy_h2d(&mut dx, &bytes(&x))?;
            stream.copy_h2d(&mut dw, &bytes(&w))?;
            // Distinct allocations stay alive until the following synchronization.
            let launch = check(
                unsafe {
                    ffi::hayate_rmsnorm_bf16(
                        dy.as_mut_ptr(),
                        dx.as_ptr(),
                        dw.as_ptr(),
                        rows as u32,
                        cols as u32,
                        eps,
                        stream.as_raw(),
                    )
                },
                "hayate_rmsnorm_bf16",
            );
            stream.synchronize()?;
            launch?;
            let mut output = vec![0; x.len() * 2];
            stream.copy_d2h(&mut output, &dy)?;
            for row in 0..rows {
                let xr = &x[row * cols..(row + 1) * cols];
                let sum: f32 = xr.iter().map(|&v| bf16_to_f32(v).powi(2)).sum();
                let inv = (sum / cols as f32 + eps).sqrt().recip();
                for col in 0..cols {
                    let index = row * cols + col;
                    let actual = bf16_to_f32(u16::from_ne_bytes(
                        output[index * 2..index * 2 + 2].try_into().unwrap(),
                    ));
                    let expected = bf16_to_f32(xr[col]) * inv * bf16_to_f32(w[col]);
                    // BF16 has 7 fraction bits: round-to-nearest relative error
                    // <= 2^-8 = 0.00390625. 0.004 allows that rounding plus
                    // f32 reduction/rsqrt differences; 1e-6 covers near zero.
                    let error = (actual - expected).abs();
                    max_error = max_error.max(error);
                    assert!(
                        actual.is_finite() && error <= 0.004 * expected.abs() + 1e-6,
                        "cols={cols} row={row} col={col}: {actual} vs {expected}"
                    );
                }
            }
        }
        eprintln!("rmsnorm_gpu: 7 shapes × {rows} rows passed, max absolute error={max_error}");
        Ok(())
    }
}
