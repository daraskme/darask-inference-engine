//! Raw CUDA Driver API declarations (CUDA 13.x `cuda.h`). Only what Tsumuji calls.
//! Versioned symbol names (`_v2`) are the ones the `cuda.h` macros resolve to.
#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::{c_char, c_int, c_uint, c_void};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: c_int = 13;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: c_int = 36;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
pub const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: c_int = 38;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: c_int = 81;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
pub const CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE: c_int = 108;

pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
pub const CU_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;
pub const CU_EVENT_DEFAULT: c_uint = 0;

pub const CU_MEMHOSTALLOC_PORTABLE: c_uint = 0x01;
pub const CU_MEMHOSTALLOC_DEVICEMAP: c_uint = 0x02;
pub const CU_MEMHOSTREGISTER_PORTABLE: c_uint = 0x01;
pub const CU_MEMHOSTREGISTER_DEVICEMAP: c_uint = 0x02;

pub const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: c_int = 6;

/// `CUlaunchAttributeValue` is a 64-byte union aligned to 8 (it holds pointers).
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct CUlaunchAttributeValue {
    pub pad: [u8; 64],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUlaunchAttribute {
    pub id: c_int,
    pub pad: [u8; 4],
    pub value: CUlaunchAttributeValue,
}

#[repr(C)]
pub struct CUlaunchConfig {
    pub gridDimX: c_uint,
    pub gridDimY: c_uint,
    pub gridDimZ: c_uint,
    pub blockDimX: c_uint,
    pub blockDimY: c_uint,
    pub blockDimZ: c_uint,
    pub sharedMemBytes: c_uint,
    pub hStream: CUstream,
    pub attrs: *mut CUlaunchAttribute,
    pub numAttrs: c_uint,
}

const _: () = assert!(size_of::<CUlaunchAttribute>() == 72);
const _: () = assert!(size_of::<CUlaunchConfig>() == 56);

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuGetErrorName(error: CUresult, name: *mut *const c_char) -> CUresult;
    pub fn cuDriverGetVersion(version: *mut c_int) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(value: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;

    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytes: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
    pub fn cuMemcpyHtoDAsync_v2(
        dst: CUdeviceptr,
        src: *const c_void,
        bytes: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoHAsync_v2(
        dst: *mut c_void,
        src: CUdeviceptr,
        bytes: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoDAsync_v2(
        dst: CUdeviceptr,
        src: CUdeviceptr,
        bytes: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    pub fn cuMemsetD8_v2(dst: CUdeviceptr, value: u8, n: usize) -> CUresult;
    pub fn cuMemsetD8Async(dst: CUdeviceptr, value: u8, n: usize, stream: CUstream) -> CUresult;
    pub fn cuMemHostAlloc(pp: *mut *mut c_void, bytes: usize, flags: c_uint) -> CUresult;
    pub fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    pub fn cuMemHostRegister_v2(p: *mut c_void, bytes: usize, flags: c_uint) -> CUresult;
    pub fn cuMemHostUnregister(p: *mut c_void) -> CUresult;
    pub fn cuMemHostGetDevicePointer_v2(
        dptr: *mut CUdeviceptr,
        p: *mut c_void,
        flags: c_uint,
    ) -> CUresult;

    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
    pub fn cuFuncSetAttribute(f: CUfunction, attrib: c_int, value: c_int) -> CUresult;
    pub fn cuLaunchKernelEx(
        config: *const CUlaunchConfig,
        f: CUfunction,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;

    pub fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;
    pub fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    pub fn cuStreamBeginCapture_v2(stream: CUstream, mode: c_int) -> CUresult;
    pub fn cuStreamEndCapture(stream: CUstream, graph: *mut CUgraph) -> CUresult;
    pub fn cuGraphInstantiateWithFlags(
        exec: *mut CUgraphExec,
        graph: CUgraph,
        flags: u64,
    ) -> CUresult;
    pub fn cuGraphLaunch(exec: CUgraphExec, stream: CUstream) -> CUresult;
    pub fn cuGraphExecDestroy(exec: CUgraphExec) -> CUresult;
    pub fn cuGraphDestroy(graph: CUgraph) -> CUresult;

    pub fn cuEventCreate(event: *mut CUevent, flags: c_uint) -> CUresult;
    pub fn cuEventDestroy_v2(event: CUevent) -> CUresult;
    pub fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;
    pub fn cuEventSynchronize(event: CUevent) -> CUresult;
    pub fn cuEventElapsedTime_v2(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;
}
