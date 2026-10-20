//! Thin CUDA Driver API layer: one device, its primary context, device/pinned memory,
//! cubin modules, PDL launches and stream-captured graphs. No allocation on hot paths;
//! everything here is set up once and reused.

pub mod sys;

use std::ffi::{CStr, CString, c_void};
use std::fmt;
use sys::*;

#[derive(Debug, Clone)]
pub struct Error {
    pub call: &'static str,
    pub code: CUresult,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut name = std::ptr::null();
        let name = if unsafe { cuGetErrorName(self.code, &mut name) } == CUDA_SUCCESS {
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        } else {
            "unknown".into()
        };
        write!(f, "{} failed: {} ({})", self.call, name, self.code)
    }
}

impl std::error::Error for Error {}

impl From<Error> for String {
    fn from(e: Error) -> String {
        e.to_string()
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[inline]
pub fn check(code: CUresult, call: &'static str) -> Result<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error { call, code })
    }
}

/// The device and its primary context, made current on the calling thread.
#[derive(Debug)]
pub struct Device {
    pub dev: CUdevice,
    ctx: CUcontext,
}

impl Device {
    pub fn open(ordinal: i32) -> Result<Self> {
        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut dev = 0;
            check(cuDeviceGet(&mut dev, ordinal), "cuDeviceGet")?;
            let mut ctx = std::ptr::null_mut();
            check(
                cuDevicePrimaryCtxRetain(&mut ctx, dev),
                "cuDevicePrimaryCtxRetain",
            )?;
            check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            Ok(Self { dev, ctx })
        }
    }

    /// Makes the context current on another thread (the engine thread).
    pub fn bind_thread(&self) -> Result<()> {
        check(unsafe { cuCtxSetCurrent(self.ctx) }, "cuCtxSetCurrent")
    }

    pub fn attr(&self, attrib: i32) -> Result<i32> {
        let mut v = 0;
        check(
            unsafe { cuDeviceGetAttribute(&mut v, attrib, self.dev) },
            "cuDeviceGetAttribute",
        )?;
        Ok(v)
    }

    pub fn name(&self) -> Result<String> {
        let mut buf = [0i8; 256];
        check(
            unsafe { cuDeviceGetName(buf.as_mut_ptr(), 256, self.dev) },
            "cuDeviceGetName",
        )?;
        Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0, 0);
        check(
            unsafe { cuMemGetInfo_v2(&mut free, &mut total) },
            "cuMemGetInfo",
        )?;
        Ok((free, total))
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { cuCtxSynchronize() }, "cuCtxSynchronize")
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = cuDevicePrimaryCtxRelease_v2(self.dev);
        }
    }
}

/// Waits for all work on the current context (legacy-stream copies included: a synchronous
/// `cuMemcpyHtoD` from pageable memory can return before its DMA lands).
pub fn synchronize_context() -> Result<()> {
    check(unsafe { cuCtxSynchronize() }, "cuCtxSynchronize")
}

/// Device memory. Freed on drop.
#[derive(Debug)]
pub struct DeviceBuffer {
    pub ptr: CUdeviceptr,
    pub bytes: usize,
}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut ptr = 0;
        check(
            unsafe { cuMemAlloc_v2(&mut ptr, bytes.max(1)) },
            "cuMemAlloc",
        )?;
        Ok(Self { ptr, bytes })
    }

    pub fn zeroed(bytes: usize) -> Result<Self> {
        let b = Self::new(bytes)?;
        check(unsafe { cuMemsetD8_v2(b.ptr, 0, bytes) }, "cuMemsetD8")?;
        Ok(b)
    }

    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self> {
        let b = Self::new(size_of_val(data))?;
        b.upload(data)?;
        Ok(b)
    }

    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<()> {
        assert!(size_of_val(data) <= self.bytes);
        check(
            unsafe { cuMemcpyHtoD_v2(self.ptr, data.as_ptr().cast(), size_of_val(data)) },
            "cuMemcpyHtoD",
        )
    }

    /// Zeroes the buffer, ordered on `stream`.
    pub fn zero_on(&self, stream: &Stream) -> Result<()> {
        check(
            unsafe { cuMemsetD8Async(self.ptr, 0, self.bytes, stream.raw) },
            "cuMemsetD8Async",
        )
    }

    /// Copy ordered on `stream` (unlike `upload`, which runs on the legacy default stream and
    /// does not wait for work queued on non-blocking streams).
    pub fn upload_on<T: Copy>(&self, data: &[T], stream: &Stream) -> Result<()> {
        assert!(size_of_val(data) <= self.bytes);
        check(
            unsafe {
                cuMemcpyHtoDAsync_v2(
                    self.ptr,
                    data.as_ptr().cast(),
                    size_of_val(data),
                    stream.raw,
                )
            },
            "cuMemcpyHtoDAsync",
        )
    }

    /// Copies all of `src` (no larger than this buffer) to the start of this buffer, ordered on `stream`.
    pub fn copy_from_on(&self, src: &DeviceBuffer, stream: &Stream) -> Result<()> {
        assert!(src.bytes <= self.bytes);
        check(
            unsafe { cuMemcpyDtoDAsync_v2(self.ptr, src.ptr, src.bytes, stream.raw) },
            "cuMemcpyDtoDAsync",
        )
    }

    pub fn download<T: Copy>(&self, out: &mut [T]) -> Result<()> {
        assert!(size_of_val(out) <= self.bytes);
        check(
            unsafe { cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.ptr, size_of_val(out)) },
            "cuMemcpyDtoH",
        )
    }

    pub fn to_vec<T: Copy + Default>(&self, n: usize) -> Result<Vec<T>> {
        let mut v = vec![T::default(); n];
        self.download(&mut v)?;
        Ok(v)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = cuMemFree_v2(self.ptr);
        }
    }
}

/// Pinned host memory mapped into the device address space (zero-copy reads from kernels).
#[derive(Debug)]
pub struct MappedHostBuffer {
    pub host: *mut u8,
    pub dev: CUdeviceptr,
    pub bytes: usize,
}

impl MappedHostBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut host = std::ptr::null_mut();
        check(
            unsafe {
                cuMemHostAlloc(
                    &mut host,
                    bytes,
                    CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE,
                )
            },
            "cuMemHostAlloc",
        )?;
        let mut dev = 0;
        check(
            unsafe { cuMemHostGetDevicePointer_v2(&mut dev, host, 0) },
            "cuMemHostGetDevicePointer",
        )?;
        Ok(Self {
            host: host.cast(),
            dev,
            bytes,
        })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.host, self.bytes) }
    }
}

impl Drop for MappedHostBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = cuMemFreeHost(self.host.cast());
        }
    }
}

/// A loaded cubin.
#[derive(Debug)]
pub struct Module {
    raw: CUmodule,
}

impl Module {
    pub fn load(cubin: &[u8]) -> Result<Self> {
        // `include_bytes!` data is only byte-aligned; give the ELF loader an 8-byte-aligned copy.
        let mut image = vec![0u64; cubin.len().div_ceil(8)];
        unsafe {
            std::ptr::copy_nonoverlapping(cubin.as_ptr(), image.as_mut_ptr().cast(), cubin.len())
        };
        let mut raw = std::ptr::null_mut();
        check(
            unsafe { cuModuleLoadData(&mut raw, image.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        Ok(Self { raw })
    }

    pub fn function(&self, name: &str) -> Result<Function> {
        let cname = CString::new(name).unwrap();
        let mut f = std::ptr::null_mut();
        check(
            unsafe { cuModuleGetFunction(&mut f, self.raw, cname.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Function { raw: f })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            let _ = cuModuleUnload(self.raw);
        }
    }
}

/// A kernel handle. Valid while its `Module` lives.
#[derive(Debug, Clone, Copy)]
pub struct Function {
    pub raw: CUfunction,
}

impl Function {
    pub fn set_max_dynamic_smem(&self, bytes: u32) -> Result<()> {
        check(
            unsafe {
                cuFuncSetAttribute(
                    self.raw,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
            },
            "cuFuncSetAttribute",
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LaunchDims {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem: u32,
}

impl LaunchDims {
    pub fn linear(grid: u32, block: u32) -> Self {
        Self {
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            smem: 0,
        }
    }
}

/// Launches `f` on `stream`. With `pdl`, the kernel may start before the previous kernel in
/// the stream finishes; it must call `griddepcontrol.wait` before reading that kernel's output.
///
/// # Safety
/// `args` must match the kernel signature: one pointer per parameter, each pointing at a value
/// of the parameter's type, alive for the duration of the call.
pub unsafe fn launch(
    f: Function,
    dims: LaunchDims,
    stream: &Stream,
    pdl: bool,
    args: &mut [*mut c_void],
) -> Result<()> {
    let mut attr = CUlaunchAttribute {
        id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 4],
        value: CUlaunchAttributeValue { pad: [0; 64] },
    };
    attr.value.pad[..4].copy_from_slice(&1i32.to_ne_bytes());
    let cfg = CUlaunchConfig {
        gridDimX: dims.grid.0,
        gridDimY: dims.grid.1,
        gridDimZ: dims.grid.2,
        blockDimX: dims.block.0,
        blockDimY: dims.block.1,
        blockDimZ: dims.block.2,
        sharedMemBytes: dims.smem,
        hStream: stream.raw,
        attrs: &mut attr,
        numAttrs: u32::from(pdl),
    };
    check(
        unsafe { cuLaunchKernelEx(&cfg, f.raw, args.as_mut_ptr(), std::ptr::null_mut()) },
        "cuLaunchKernelEx",
    )
}

/// Builds the `void**` argument array for `launch` from references to the argument values.
#[macro_export]
macro_rules! args {
    ($($a:expr),* $(,)?) => {
        [$((&$a as *const _ as *mut ::std::ffi::c_void)),*]
    };
}

#[derive(Debug)]
pub struct Stream {
    pub raw: CUstream,
}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check(
            unsafe { cuStreamCreate(&mut raw, CU_STREAM_NON_BLOCKING) },
            "cuStreamCreate",
        )?;
        Ok(Self { raw })
    }

    pub fn synchronize(&self) -> Result<()> {
        check(
            unsafe { cuStreamSynchronize(self.raw) },
            "cuStreamSynchronize",
        )
    }

    /// Captures everything `body` enqueues on this stream into an executable graph.
    pub fn capture<E: From<Error>>(
        &self,
        body: impl FnOnce(&Stream) -> std::result::Result<(), E>,
    ) -> std::result::Result<Graph, E> {
        check(
            unsafe { cuStreamBeginCapture_v2(self.raw, CU_STREAM_CAPTURE_MODE_RELAXED) },
            "cuStreamBeginCapture",
        )?;
        let body_result = body(self);
        let mut graph = std::ptr::null_mut();
        let end = check(
            unsafe { cuStreamEndCapture(self.raw, &mut graph) },
            "cuStreamEndCapture",
        );
        body_result?;
        end?;
        let mut exec = std::ptr::null_mut();
        let inst = check(
            unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) },
            "cuGraphInstantiateWithFlags",
        );
        unsafe {
            let _ = cuGraphDestroy(graph);
        }
        inst?;
        Ok(Graph { exec })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            let _ = cuStreamDestroy_v2(self.raw);
        }
    }
}

#[derive(Debug)]
pub struct Graph {
    exec: CUgraphExec,
}

impl Graph {
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        check(
            unsafe { cuGraphLaunch(self.exec, stream.raw) },
            "cuGraphLaunch",
        )
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            let _ = cuGraphExecDestroy(self.exec);
        }
    }
}

#[derive(Debug)]
pub struct Event {
    raw: CUevent,
}

impl Event {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check(
            unsafe { cuEventCreate(&mut raw, CU_EVENT_DEFAULT) },
            "cuEventCreate",
        )?;
        Ok(Self { raw })
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        check(
            unsafe { cuEventRecord(self.raw, stream.raw) },
            "cuEventRecord",
        )
    }

    /// Milliseconds from `self` to `end`; waits for `end`.
    pub fn elapsed_ms(&self, end: &Event) -> Result<f32> {
        check(unsafe { cuEventSynchronize(end.raw) }, "cuEventSynchronize")?;
        let mut ms = 0.0;
        check(
            unsafe { cuEventElapsedTime_v2(&mut ms, self.raw, end.raw) },
            "cuEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            let _ = cuEventDestroy_v2(self.raw);
        }
    }
}

/// Times `iters` runs of `body` on `stream` after `warmup` untimed runs; returns ms per run.
pub fn time_ms(
    stream: &Stream,
    warmup: usize,
    iters: usize,
    mut body: impl FnMut(usize) -> Result<()>,
) -> Result<f64> {
    for i in 0..warmup {
        body(i)?;
    }
    let (start, end) = (Event::new()?, Event::new()?);
    start.record(stream)?;
    for i in 0..iters {
        body(warmup + i)?;
    }
    end.record(stream)?;
    Ok(f64::from(start.elapsed_ms(&end)?) / iters as f64)
}
