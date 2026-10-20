//! CUDA Graph capture and replay for the GLM runtime.
use crate::{cuda::ffi::Stream, runtime::check};
use hayate_core::Result;
use std::ffi::c_void;

unsafe extern "C" {
    fn cudaStreamBeginCapture(stream: Stream, mode: i32) -> i32;
    fn cudaStreamEndCapture(stream: Stream, graph: *mut *mut c_void) -> i32;
    fn cudaGraphInstantiateWithFlags(exec: *mut *mut c_void, graph: *mut c_void, flags: u64)
    -> i32;
    fn cudaGraphDestroy(graph: *mut c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
    fn cudaGraphLaunch(exec: *mut c_void, stream: Stream) -> i32;
}
#[derive(Debug)]
pub struct Graph {
    raw: *mut c_void,
    exec: *mut c_void,
}
impl Graph {
    /// # Safety
    /// Captured pointers/events must live until this graph is destroyed and
    /// every replay completes. Closure submissions must use this stream.
    /// Warm up kernels before capture (including module/attribute setup).
    pub unsafe fn capture(stream: Stream, submit: impl FnOnce() -> Result<()>) -> Result<Self> {
        check(
            unsafe { cudaStreamBeginCapture(stream, 0) },
            "cudaStreamBeginCapture",
        )?;
        let status = submit();
        let mut raw = std::ptr::null_mut();
        let end = check(
            unsafe { cudaStreamEndCapture(stream, &mut raw) },
            "cudaStreamEndCapture",
        );
        if let Err(e) = status.and(end) {
            if !raw.is_null() {
                let _ = unsafe { cudaGraphDestroy(raw) };
            }
            return Err(e);
        }
        let mut g = Self {
            raw,
            exec: std::ptr::null_mut(),
        };
        check(
            unsafe { cudaGraphInstantiateWithFlags(&mut g.exec, raw, 0) },
            "cudaGraphInstantiateWithFlags",
        )?;
        Ok(g)
    }
    /// # Safety
    /// All captured storage must be valid, with no conflicting work until
    /// stream completion. Synchronize before releasing captured storage.
    pub unsafe fn launch(&self, stream: Stream) -> Result<()> {
        check(
            unsafe { cudaGraphLaunch(self.exec, stream) },
            "cudaGraphLaunch",
        )
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            let _ = unsafe { cudaGraphExecDestroy(self.exec) };
        }
        let _ = unsafe { cudaGraphDestroy(self.raw) };
    }
}
