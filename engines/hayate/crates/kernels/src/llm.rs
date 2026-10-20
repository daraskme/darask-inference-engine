//! Shared matrix primitive used by GLM-5.3-Flash.
#[cfg(feature = "cuda")]
pub mod ffi {
    use crate::cuda::ffi::Stream;
    use std::ffi::c_void;
    // Caller owns validated, disjoint storage through stream completion.
    unsafe extern "C" {
        pub fn hayate_llm_gemv(
            y: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            n: u32,
            k: u32,
            format: i32,
            logits: i32,
            partial: *mut f32,
            rows: u32,
            splits: u32,
            async_load: i32,
            s: Stream,
        ) -> i32;
    }
}
