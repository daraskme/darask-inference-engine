//! Shared matrix primitive used by GLM-5.3-Flash.
#[cfg(feature = "cuda")]
pub mod ffi {
    use crate::cuda::ffi::Stream;
    use std::ffi::c_void;
    // Caller owns validated, disjoint storage through stream completion.
    unsafe extern "C" {
        pub fn hayate_text_linear(
            out: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            m: u32,
            n: u32,
            k: u32,
            stream: Stream,
        ) -> i32;
    }
}
