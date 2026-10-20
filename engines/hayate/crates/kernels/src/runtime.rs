//! Minimal CUDA 13 runtime for startup allocations and planned arenas.
//! One process uses one GPU, selected once at startup. Safe copies synchronize;
//! asynchronous copies and kernel FFI require the caller to preserve lifetimes.

use std::cell::Cell;
use std::ffi::{c_int, c_void};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;

use hayate_core::machine::{MAIN, SUB};
use hayate_core::{Error, MachineId, Result};

use crate::cuda::ffi;

unsafe extern "C" {
    fn cudaSetDevice(device: c_int) -> c_int;
    fn hayate_device_info(
        device: c_int,
        name: *mut u8,
        name_len: c_int,
        major: *mut c_int,
        minor: *mut c_int,
    ) -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaFree(ptr: *mut c_void) -> c_int;
    fn cudaMallocHost(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaHostAlloc(ptr: *mut *mut c_void, bytes: usize, flags: u32) -> c_int;
    fn cudaHostGetDevicePointer(device: *mut *mut c_void, host: *mut c_void, flags: u32) -> c_int;
    fn hayate_host_wait(stream: ffi::Stream, addr: *mut c_void, value: u32) -> c_int;
    fn hayate_host_signal(stream: ffi::Stream, addr: *mut c_void, value: u32) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    fn cudaStreamCreateWithFlags(stream: *mut ffi::Stream, flags: u32) -> c_int;
    fn cudaStreamDestroy(stream: ffi::Stream) -> c_int;
    fn cudaStreamSynchronize(stream: ffi::Stream) -> c_int;
    fn cudaThreadExchangeStreamCaptureMode(mode: *mut c_int) -> c_int;
    fn cudaStreamWaitEvent(stream: ffi::Stream, event: *mut c_void, flags: u32) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: c_int,
        stream: ffi::Stream,
    ) -> c_int;
    fn cudaEventCreate(event: *mut *mut c_void) -> c_int;
    fn cudaEventDestroy(event: *mut c_void) -> c_int;
    fn cudaEventRecord(event: *mut c_void, stream: ffi::Stream) -> c_int;
    fn cudaEventSynchronize(event: *mut c_void) -> c_int;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> c_int;
}

thread_local! { static HOST_SYNCS: Cell<u64> = const { Cell::new(0) }; }

/// Explicit blocking CUDA calls on this host thread, including safe copies.
pub fn host_sync_count() -> u64 {
    HOST_SYNCS.get()
}

fn count_sync() {
    HOST_SYNCS.set(HOST_SYNCS.get() + 1);
}

/// CUDA 13 `driver_types.h`: unrecoverable hardware/context errors and sticky
/// execution errors. Resource/configuration errors (e.g. 701, 720) stay ordinary.
/// https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__TYPES.html
pub const fn is_persistent_error(code: i32) -> bool {
    matches!(code, 214 | 220 | 700 | 702 | 709 | 710 | 714..=719 | 721)
}

/// Classifies by value, including errors observed by `cudaGetLastError` at launch.
pub fn check(code: i32, call: &'static str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else if is_persistent_error(code) {
        Err(Error::CudaAsync {
            code,
            sync_point: call,
        })
    } else {
        Err(Error::Cuda { code, call })
    }
}

#[derive(Clone, Debug)]
pub struct Device {
    pub name: String,
    pub sm: (u32, u32),
    pub machine: MachineId,
}

impl Device {
    /// Call once at startup: selects GPU 0 and validates the two supported profiles.
    pub fn select() -> Result<Self> {
        check(unsafe { cudaSetDevice(0) }, "cudaSetDevice")?;
        let mut name = [0u8; 256];
        let (mut major, mut minor) = (0, 0);
        check(
            unsafe {
                hayate_device_info(
                    0,
                    name.as_mut_ptr(),
                    name.len() as c_int,
                    &mut major,
                    &mut minor,
                )
            },
            "cudaGetDeviceProperties",
        )?;
        let name_end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
        let name = String::from_utf8_lossy(&name[..name_end]).into_owned();
        let sm = (major as u32, minor as u32);
        let machine = validate_gpu(&name, sm)?;
        Ok(Self { name, sm, machine })
    }

    pub fn memory_info(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0, 0);
        check(
            unsafe { cudaMemGetInfo(&mut free, &mut total) },
            "cudaMemGetInfo",
        )?;
        Ok((free, total))
    }

    pub fn synchronize(&self) -> Result<()> {
        count_sync();
        check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
    }

    /// Allocates a byte arena at startup, never on the kernel submission path.
    pub fn alloc(&self, bytes: usize) -> Result<DeviceBuffer> {
        allocation_size(bytes)?;
        let mut ptr = std::ptr::null_mut();
        check(unsafe { cudaMalloc(&mut ptr, bytes) }, "cudaMalloc")?;
        Ok(DeviceBuffer {
            ptr: NonNull::new(ptr).expect("successful cudaMalloc returned null"),
            bytes,
            block: None,
        })
    }

    /// `count` buffers of `bytes` carved from one allocation (cudaMalloc rounds
    /// to 2 MiB pages: many small images lose up to a page each). The block is
    /// freed when the last buffer is dropped.
    pub fn alloc_split(&self, count: usize, bytes: usize) -> Result<Vec<DeviceBuffer>> {
        let total = count
            .checked_mul(bytes)
            .ok_or_else(|| Error::InvalidShape("device split allocation overflow".into()))?;
        let whole = self.alloc(total)?;
        let block = Arc::new(Block(whole.ptr));
        std::mem::forget(whole);
        Ok((0..count)
            .map(|i| DeviceBuffer {
                ptr: NonNull::new(unsafe { block.0.as_ptr().cast::<u8>().add(i * bytes) }.cast())
                    .expect("split of a non-null block"),
                bytes,
                block: Some(block.clone()),
            })
            .collect())
    }

    pub fn alloc_pinned(&self, bytes: usize) -> Result<PinnedHost> {
        allocation_size(bytes)?;
        let mut ptr = std::ptr::null_mut();
        check(unsafe { cudaMallocHost(&mut ptr, bytes) }, "cudaMallocHost")?;
        let ptr = NonNull::new(ptr).expect("successful cudaMallocHost returned null");
        // Initialize before exposing safe Rust byte slices.
        unsafe { ptr.as_ptr().cast::<u8>().write_bytes(0, bytes) };
        Ok(PinnedHost { ptr, bytes })
    }

    /// Mapped pinned storage for externally synchronized CPU/GPU mailboxes.
    pub fn alloc_mapped(&self, bytes: usize) -> Result<MappedHost> {
        allocation_size(bytes)?;
        let mut host = std::ptr::null_mut();
        check(
            unsafe { cudaHostAlloc(&mut host, bytes, 2) },
            "cudaHostAllocMapped",
        )?;
        let storage = PinnedHost {
            ptr: NonNull::new(host).unwrap(),
            bytes,
        };
        unsafe { host.cast::<u8>().write_bytes(0, bytes) };
        let mut device = std::ptr::null_mut();
        check(
            unsafe { cudaHostGetDevicePointer(&mut device, host, 0) },
            "cudaHostGetDevicePointer",
        )?;
        Ok(MappedHost { storage, device })
    }

    pub fn stream(&self) -> Result<Stream> {
        let mut raw = std::ptr::null_mut();
        // cudaStreamNonBlocking: no implicit dependency on the legacy stream.
        check(
            unsafe { cudaStreamCreateWithFlags(&mut raw, 1) },
            "cudaStreamCreateWithFlags",
        )?;
        Ok(Stream { raw })
    }
}

/// No safe slices: concurrent access must follow the caller's mailbox protocol.
#[derive(Debug)]
pub struct MappedHost {
    storage: PinnedHost,
    device: *mut c_void,
}
unsafe impl Send for MappedHost {}
unsafe impl Sync for MappedHost {}
impl MappedHost {
    pub fn host_ptr(&self) -> *mut c_void {
        self.storage.ptr.as_ptr()
    }
    pub fn device_ptr(&self) -> *mut c_void {
        self.device
    }
    pub fn bytes(&self) -> usize {
        self.storage.bytes
    }
}

fn validate_gpu(name: &str, sm: (u32, u32)) -> Result<MachineId> {
    [MAIN, SUB]
        .iter()
        .find(|m| name.contains(m.gpu.name_match) && sm == m.gpu.sm && sm == (12, 0))
        .map(|m| m.id)
        .ok_or_else(|| Error::UnsupportedMachine(format!("GPU {name}, CC {}.{}", sm.0, sm.1)))
}

fn allocation_size(bytes: usize) -> Result<()> {
    if bytes == 0 || bytes > isize::MAX as usize {
        return Err(Error::InvalidShape(
            "CUDA allocation size must be in 1..=isize::MAX".into(),
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub struct DeviceBuffer {
    ptr: NonNull<c_void>,
    bytes: usize,
    /// Set for buffers from `alloc_split`: the shared allocation owns the memory.
    block: Option<Arc<Block>>,
}

#[derive(Debug)]
struct Block(NonNull<c_void>);
unsafe impl Send for Block {}
unsafe impl Sync for Block {}
impl Drop for Block {
    fn drop(&mut self) {
        let _ = unsafe { cudaFree(self.0.as_ptr()) };
    }
}

// CUDA allocations belong to GPU 0, not the allocating host thread. Moving
// ownership does not submit work; asynchronous users still retain lifetimes.
unsafe impl Send for DeviceBuffer {}

impl DeviceBuffer {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The shared allocation of an `alloc_split` buffer (its address; `None`
    /// for a buffer with its own allocation). The block returns to CUDA only
    /// when every buffer carved from it is dropped.
    pub fn block(&self) -> Option<usize> {
        self.block.as_ref().map(|b| b.0.as_ptr() as usize)
    }

    /// Raw arena base; callers of kernel FFI manage offsets and lifetimes.
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr()
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.block.is_none() {
            let _ = unsafe { cudaFree(self.ptr.as_ptr()) };
        }
    }
}

#[derive(Debug)]
pub struct PinnedHost {
    ptr: NonNull<c_void>,
    bytes: usize,
}

unsafe impl Send for PinnedHost {}

impl PinnedHost {
    /// Raw pointer without creating a slice over an in-flight DMA destination.
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }
    /// A checked view retaining the pinned allocation's lifetime.
    pub fn range(&self, range: Range<usize>) -> Result<PinnedRange<'_>> {
        let bytes = range
            .end
            .checked_sub(range.start)
            .ok_or_else(|| Error::InvalidShape("inverted pinned range".into()))?;
        copy_range(self.bytes, range.start, bytes)?;
        Ok(PinnedRange {
            bytes: &self.as_slice()[range],
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast(), self.bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.bytes) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PinnedRange<'a> {
    bytes: &'a [u8],
}

fn copy_range(capacity: usize, offset: usize, bytes: usize) -> Result<()> {
    if offset > capacity || bytes > capacity - offset {
        return Err(Error::InvalidShape(
            "CUDA copy range exceeds allocation".into(),
        ));
    }
    Ok(())
}

impl Drop for PinnedHost {
    fn drop(&mut self) {
        let _ = unsafe { cudaFreeHost(self.ptr.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct Stream {
    raw: ffi::Stream,
}

// An owned stream can move between host threads; it is not shared concurrently.
unsafe impl Send for Stream {}

impl Stream {
    /// Permit synchronization of this independent, non-capturing stream while
    /// another host thread captures a graph. No buffers/events may be shared
    /// with the captured work. The thread's previous policy is restored on exit.
    pub fn independent_work<T>(&self, work: impl FnOnce() -> Result<T>) -> Result<T> {
        struct CaptureMode(c_int);
        impl Drop for CaptureMode {
            fn drop(&mut self) {
                let _ = unsafe { cudaThreadExchangeStreamCaptureMode(&mut self.0) };
            }
        }
        // cudaStreamCaptureModeThreadLocal ignores captures in other threads.
        let mut mode = 1;
        check(
            unsafe { cudaThreadExchangeStreamCaptureMode(&mut mode) },
            "cudaThreadExchangeStreamCaptureMode",
        )?;
        let _restore = CaptureMode(mode);
        work()
    }

    pub fn as_raw(&self) -> ffi::Stream {
        self.raw
    }

    pub fn synchronize(&self) -> Result<()> {
        count_sync();
        check(
            unsafe { cudaStreamSynchronize(self.raw) },
            "cudaStreamSynchronize",
        )
    }

    /// # Safety
    /// `addr` must be an aligned mapped device address, alive until completion.
    /// The CPU must publish the value with release ordering. Cross-stream GPU
    /// producers must also be ordered by CUDA events, never just this flag.
    pub unsafe fn wait_value32(&self, addr: *mut c_void, value: u32) -> Result<()> {
        check(
            unsafe { hayate_host_wait(self.raw, addr, value) },
            "cuStreamWaitValue32",
        )
    }

    /// Publish preceding stream writes to a polling CPU without blocking.
    /// # Safety
    /// `addr` must be an aligned mapped device address, alive until completion;
    /// the CPU must acquire the value before accessing the published payload.
    pub unsafe fn write_value32(&self, addr: *mut c_void, value: u32) -> Result<()> {
        check(
            unsafe { hayate_host_signal(self.raw, addr, value) },
            "cuStreamWriteValue32",
        )
    }

    /// Orders future work after the event's most recent record without blocking the host.
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        check(
            unsafe { cudaStreamWaitEvent(self.raw, event.raw, 0) },
            "cudaStreamWaitEvent",
        )
    }

    fn copy_size(&self, buffer: &DeviceBuffer, bytes: usize) -> Result<()> {
        if bytes > buffer.bytes {
            return Err(Error::InvalidShape("CUDA copy exceeds arena".into()));
        }
        Ok(())
    }

    /// Synchronous H2D; pageable or pinned source is allowed.
    pub fn copy_h2d(&self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<()> {
        self.copy_size(dst, src.len())?;
        let result = check(
            unsafe {
                cudaMemcpyAsync(
                    dst.ptr.as_ptr(),
                    src.as_ptr().cast(),
                    src.len(),
                    1,
                    self.raw,
                )
            },
            "cudaMemcpyAsync H2D",
        );
        // Drain even if submission reports an earlier asynchronous error.
        let sync = self.synchronize();
        sync.and(result)
    }

    /// Synchronous D2H; output is readable when this returns successfully.
    pub fn copy_d2h(&self, dst: &mut [u8], src: &DeviceBuffer) -> Result<()> {
        self.copy_size(src, dst.len())?;
        let result = check(
            unsafe {
                cudaMemcpyAsync(
                    dst.as_mut_ptr().cast(),
                    src.ptr.as_ptr(),
                    dst.len(),
                    2,
                    self.raw,
                )
            },
            "cudaMemcpyAsync D2H",
        );
        let sync = self.synchronize();
        sync.and(result)
    }

    /// Synchronous D2H of `dst.len()` bytes starting at `offset` in the arena.
    pub fn copy_d2h_at(&self, dst: &mut [u8], src: &DeviceBuffer, offset: usize) -> Result<()> {
        copy_range(src.bytes, offset, dst.len())?;
        let result = check(
            unsafe {
                cudaMemcpyAsync(
                    dst.as_mut_ptr().cast(),
                    src.ptr.as_ptr().cast::<u8>().add(offset).cast(),
                    dst.len(),
                    2,
                    self.raw,
                )
            },
            "cudaMemcpyAsync D2H",
        );
        let sync = self.synchronize();
        sync.and(result)
    }

    /// Synchronous H2D of `src` into the arena at `offset`; pageable source allowed.
    pub fn copy_h2d_at(&self, dst: &mut DeviceBuffer, offset: usize, src: &[u8]) -> Result<()> {
        copy_range(dst.bytes, offset, src.len())?;
        let result = check(
            unsafe {
                cudaMemcpyAsync(
                    dst.ptr.as_ptr().cast::<u8>().add(offset).cast(),
                    src.as_ptr().cast(),
                    src.len(),
                    1,
                    self.raw,
                )
            },
            "cudaMemcpyAsync H2D",
        );
        let sync = self.synchronize();
        sync.and(result)
    }

    /// Enqueues a pinned H2D copy without synchronizing.
    ///
    /// # Safety
    /// Keep both allocations alive and do not modify `src` or access `dst`
    /// from another stream until this stream is synchronized, including on error.
    pub unsafe fn copy_h2d_async(&self, dst: &mut DeviceBuffer, src: &PinnedHost) -> Result<()> {
        unsafe { self.copy_h2d_at_async(dst, 0, src.range(0..src.bytes)?) }
    }

    /// Enqueues a pinned subrange copy into a byte offset in the device arena.
    ///
    /// # Safety
    /// Keep both allocations alive and do not modify the source range or access
    /// the destination range until the copy completes, including on error. Other
    /// streams must be ordered with events before reading or overwriting it.
    pub unsafe fn copy_h2d_at_async(
        &self,
        dst: &mut DeviceBuffer,
        offset: usize,
        src: PinnedRange<'_>,
    ) -> Result<()> {
        copy_range(dst.bytes, offset, src.bytes.len())?;
        check(
            unsafe {
                cudaMemcpyAsync(
                    dst.ptr.as_ptr().cast::<u8>().add(offset).cast(),
                    src.bytes.as_ptr().cast(),
                    src.bytes.len(),
                    1,
                    self.raw,
                )
            },
            "cudaMemcpyAsync H2D",
        )
    }

    /// Enqueues a pinned D2H copy without synchronizing.
    ///
    /// # Safety
    /// Keep both allocations alive and do not access `dst` or modify `src`
    /// from another stream until this stream is synchronized, including on error.
    pub unsafe fn copy_d2h_async(&self, dst: &mut PinnedHost, src: &DeviceBuffer) -> Result<()> {
        self.copy_size(src, dst.bytes)?;
        check(
            unsafe { cudaMemcpyAsync(dst.ptr.as_ptr(), src.ptr.as_ptr(), dst.bytes, 2, self.raw) },
            "cudaMemcpyAsync D2H",
        )
    }

    /// Enqueues a D2H copy of the first `bytes` of `src` into the start of `dst`
    /// (both may be larger).
    ///
    /// # Safety
    /// Same contract as `copy_d2h_async`.
    pub unsafe fn copy_d2h_prefix_async(
        &self,
        dst: &mut PinnedHost,
        src: &DeviceBuffer,
        bytes: usize,
    ) -> Result<()> {
        self.copy_size(src, bytes)?;
        if bytes > dst.bytes {
            return Err(Error::InvalidShape(
                "CUDA copy exceeds pinned buffer".into(),
            ));
        }
        check(
            unsafe { cudaMemcpyAsync(dst.ptr.as_ptr(), src.ptr.as_ptr(), bytes, 2, self.raw) },
            "cudaMemcpyAsync D2H",
        )
    }

    /// Enqueues an H2D copy of the first `bytes` of `src` into the start of `dst`.
    ///
    /// # Safety
    /// Same contract as `copy_h2d_async`.
    pub unsafe fn copy_h2d_prefix_async(
        &self,
        dst: &mut DeviceBuffer,
        src: &PinnedHost,
        bytes: usize,
    ) -> Result<()> {
        unsafe { self.copy_h2d_at_async(dst, 0, src.range(0..bytes)?) }
    }

    /// Enqueues a D2D copy of the first `bytes` of `src` into the start of `dst`.
    ///
    /// # Safety
    /// Both buffers must stay alive, and no other stream may use them, until
    /// the copy completes.
    pub unsafe fn copy_d2d_prefix_async(
        &self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
        bytes: usize,
    ) -> Result<()> {
        self.copy_size(src, bytes)?;
        self.copy_size(dst, bytes)?;
        check(
            unsafe { cudaMemcpyAsync(dst.ptr.as_ptr(), src.ptr.as_ptr(), bytes, 3, self.raw) },
            "cudaMemcpyAsync D2D",
        )
    }

    pub fn event(&self) -> Result<Event> {
        let mut raw = std::ptr::null_mut();
        check(unsafe { cudaEventCreate(&mut raw) }, "cudaEventCreate")?;
        Ok(Event { raw })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let _ = unsafe { cudaStreamDestroy(self.raw) };
    }
}

/// Timing event allocated before measurement.
#[derive(Debug)]
pub struct Event {
    raw: *mut c_void,
}

unsafe impl Send for Event {}

impl Event {
    pub fn record(&self, stream: &Stream) -> Result<()> {
        check(
            unsafe { cudaEventRecord(self.raw, stream.raw) },
            "cudaEventRecord",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        count_sync();
        check(
            unsafe { cudaEventSynchronize(self.raw) },
            "cudaEventSynchronize",
        )
    }

    /// Both events must have completed before reading their elapsed time.
    pub fn elapsed_ms(&self, end: &Event) -> Result<f32> {
        let mut ms = 0.0;
        check(
            unsafe { cudaEventElapsedTime(&mut ms, self.raw, end.raw) },
            "cudaEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let _ = unsafe { cudaEventDestroy(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_copy_ranges_without_cuda() {
        assert!(copy_range(16, 4, 12).is_ok());
        assert!(copy_range(16, 16, 0).is_ok());
        assert!(copy_range(16, 4, 13).is_err());
        assert!(copy_range(16, 17, 0).is_err());
        assert!(copy_range(16, usize::MAX, 2).is_err());
        assert!(copy_range(16, 1, usize::MAX).is_err());
    }

    #[test]
    fn classifies_errors_by_code_not_observation_site() {
        for code in [
            214, 220, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719, 721,
        ] {
            for call in ["cudaGetLastError", "cudaStreamSynchronize"] {
                assert!(
                    matches!(check(code, call), Err(Error::CudaAsync { code: c, sync_point }) if c == code && sync_point == call)
                );
            }
        }
        for code in [1, 2, 9, 35, 100, 701, 703, 704, 711, 712, 713, 720, 999] {
            assert!(
                matches!(check(code, "launch"), Err(Error::Cuda { code: c, call: "launch" }) if c == code)
            );
        }
        assert!(check(0, "sync").is_ok());
    }

    #[test]
    fn validates_only_the_two_gpu_profiles() {
        assert_eq!(
            validate_gpu("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", (12, 0)).unwrap(),
            MachineId::Main
        );
        assert_eq!(
            validate_gpu("NVIDIA GeForce RTX 5060 Ti", (12, 0)).unwrap(),
            MachineId::Sub
        );
        assert!(validate_gpu(MAIN.gpu.name_match, (12, 1)).is_err());
        assert!(validate_gpu("NVIDIA RTX 5090", (12, 0)).is_err());
    }
}
