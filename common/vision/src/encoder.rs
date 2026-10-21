//! Resident native ViT; linear/attention GEMMs use cuBLAS, normalisation/rope/activation CUDA.
use super::{Image, Kind, Result};
use darask_formats::safetensors::{TensorInfo, read_header};
use std::{
    collections::HashMap,
    ffi::c_void,
    path::{Path, PathBuf},
};
use tsumuji_cuda::{Device, DeviceBuffer as Buf, LaunchDims, Module, Stream, args, launch};
unsafe extern "C" {
    fn cublasCreate_v2(h: *mut *mut c_void) -> i32;
    fn cublasDestroy_v2(h: *mut c_void) -> i32;
    fn cublasSetStream_v2(h: *mut c_void, s: *mut c_void) -> i32;
    fn cublasSgemm_v2(
        h: *mut c_void,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const f32,
        x: *const f32,
        lda: i32,
        y: *const f32,
        ldb: i32,
        b: *const f32,
        z: *mut f32,
        ldc: i32,
    ) -> i32;
    fn cublasSgemmStridedBatched(
        h: *mut c_void,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const f32,
        x: *const f32,
        lda: i32,
        sx: i64,
        y: *const f32,
        ldb: i32,
        sy: i64,
        b: *const f32,
        z: *mut f32,
        ldc: i32,
        sz: i64,
        batch: i32,
    ) -> i32;
}
fn blas(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("cuBLAS status {code}"))
    }
}
#[derive(Debug)]
struct Weight {
    data: Buf,
    shape: Vec<u64>,
}
#[derive(Debug)]
pub struct Encoder {
    weights: HashMap<String, Weight>,
    module: Module,
    stream: Stream,
    handle: *mut c_void,
    _device: Device,
    pub kind: Kind,
    h: usize,
    ff: usize,
    out: usize,
    depth: usize,
    heads: usize,
    eps: f32,
}
impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.handle);
        }
    }
}
fn headers(path: &Path) -> Result<Vec<TensorInfo>> {
    if path.is_file() {
        return read_header(path);
    }
    let index = path.join("model.safetensors.index.json");
    let files: Vec<PathBuf> = if index.is_file() {
        let v = darask_json::parse(&std::fs::read_to_string(index).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let mut f = v
            .get("weight_map")
            .and_then(|v| v.object())
            .map_err(|e| e.to_string())?
            .iter()
            .filter(|(k, _)| k.contains("visual."))
            .map(|(_, v)| v.string().map(|v| path.join(v)).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>>>()?;
        f.sort();
        f.dedup();
        f
    } else {
        let mut f = std::fs::read_dir(path)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect::<Vec<_>>();
        f.sort();
        f
    };
    let mut all = Vec::new();
    for f in files {
        all.extend(read_header(&f)?);
    }
    Ok(all)
}
impl Encoder {
    pub fn load(path: &Path, kind: Kind) -> Result<Self> {
        let device = Device::open(0)?;
        let stream = Stream::new()?;
        let module = Module::load(include_bytes!(concat!(env!("OUT_DIR"), "/vision.cubin")))?;
        let mut weights = HashMap::new();
        for t in headers(path)? {
            let Some((_, name)) = t.name.split_once("visual.") else {
                continue;
            };
            let raw = t.read().map_err(|e| e.to_string())?;
            let floats: Vec<f32> = match t.dtype.as_str() {
                "BF16" => raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| f32::from_bits((u16::from_le_bytes([v[0], v[1]]) as u32) << 16))
                    .collect(),
                "F16" => raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| darask_formats::gguf::f16_to_f32(u16::from_le_bytes([v[0], v[1]])))
                    .collect(),
                "F32" => raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect(),
                x => return Err(format!("vision tensor {}: unsupported dtype {x}", t.name)),
            };
            if floats.len() != t.shape.iter().product::<u64>() as usize
                || floats.iter().any(|x| !x.is_finite())
            {
                return Err(format!("invalid vision tensor {}", t.name));
            }
            weights.insert(
                name.to_string(),
                Weight {
                    data: Buf::from_slice(&floats)?,
                    shape: t.shape,
                },
            );
        }
        let (h, ff, out, depth, eps) = if kind == Kind::Glm {
            (1024, 4096, 4096, 24, 1e-5)
        } else {
            (1152, 4304, 0, 27, 1e-6)
        };
        let out = if out == 0 {
            weights
                .get("merger.linear_fc2.weight")
                .and_then(|w| w.shape.first())
                .copied()
                .ok_or("vision merger weight missing")? as usize
        } else {
            out
        };
        device.synchronize()?;
        let mut handle = std::ptr::null_mut();
        blas(unsafe { cublasCreate_v2(&mut handle) })?;
        blas(unsafe { cublasSetStream_v2(handle, stream.raw) })?;
        let encoder = Self {
            weights,
            module,
            stream,
            handle,
            _device: device,
            kind,
            h,
            ff,
            out,
            depth,
            heads: 16,
            eps,
        };
        encoder.validate()?;
        Ok(encoder)
    }
    fn w(&self, n: &str) -> Result<&Weight> {
        self.weights
            .get(n)
            .ok_or_else(|| format!("missing vision weight: {n}"))
    }
    fn ptr(&self, n: &str) -> Result<u64> {
        Ok(self.w(n)?.data.ptr)
    }
    fn optional(&self, n: &str) -> u64 {
        self.weights.get(n).map_or(0, |w| w.data.ptr)
    }
    fn validate(&self) -> Result<()> {
        let expect = |name: &str, shape: &[u64]| -> Result<()> {
            let w = self.w(name)?;
            if w.shape != shape {
                Err(format!("vision {name}: {:?}, expected {shape:?}", w.shape))
            } else {
                Ok(())
            }
        };
        let (h, ff) = (self.h as u64, self.ff as u64);
        let patch = if self.kind == Kind::Glm { 14 } else { 16 };
        expect("patch_embed.proj.weight", &[h, 3, 2, patch, patch])?;
        expect("patch_embed.proj.bias", &[h])?;
        for i in 0..self.depth {
            let p = format!("blocks.{i}");
            for n in ["norm1", "norm2"] {
                expect(&format!("{p}.{n}.weight"), &[h])?;
                if self.kind == Kind::Qwen {
                    expect(&format!("{p}.{n}.bias"), &[h])?;
                }
            }
            for (n, s) in [("attn.qkv", vec![3 * h, h]), ("attn.proj", vec![h, h])] {
                expect(&format!("{p}.{n}.weight"), &s)?;
                expect(&format!("{p}.{n}.bias"), &[s[0]])?;
            }
            if self.kind == Kind::Qwen {
                expect(&format!("{p}.mlp.linear_fc1.weight"), &[ff, h])?;
                expect(&format!("{p}.mlp.linear_fc2.weight"), &[h, ff])?;
                expect(&format!("{p}.mlp.linear_fc1.bias"), &[ff])?;
                expect(&format!("{p}.mlp.linear_fc2.bias"), &[h])?;
            } else {
                for n in ["gate_proj", "up_proj"] {
                    expect(&format!("{p}.mlp.{n}.weight"), &[ff, h])?;
                    expect(&format!("{p}.mlp.{n}.bias"), &[ff])?;
                }
                expect(&format!("{p}.mlp.down_proj.weight"), &[h, ff])?;
                expect(&format!("{p}.mlp.down_proj.bias"), &[h])?;
                for n in ["q_norm", "k_norm"] {
                    expect(&format!("{p}.attn.{n}.weight"), &[h / 16])?;
                }
            }
        }
        if self.kind == Kind::Qwen {
            expect("pos_embed.weight", &[2304, h])?;
            expect("merger.norm.weight", &[h])?;
            expect("merger.norm.bias", &[h])?;
            expect("merger.linear_fc1.bias", &[4 * h])?;
            expect("merger.linear_fc2.bias", &[self.out as u64])?;
            expect("merger.linear_fc1.weight", &[4 * h, 4 * h])?;
            expect("merger.linear_fc2.weight", &[self.out as u64, 4 * h])?;
        } else {
            expect("post_layernorm.weight", &[h])?;
            expect("downsample.bias", &[self.out as u64])?;
            expect("merger.post_projection_norm.weight", &[4096])?;
            expect("merger.post_projection_norm.bias", &[4096])?;
            expect("downsample.weight", &[self.out as u64, h, 2, 2])?;
            expect("merger.proj.weight", &[4096, 4096])?;
            expect("merger.gate_proj.weight", &[10240, 4096])?;
            expect("merger.up_proj.weight", &[10240, 4096])?;
            expect("merger.down_proj.weight", &[4096, 10240])?;
        }
        Ok(())
    }
    fn go(&self, name: &str, grid: u32, params: &mut [*mut c_void]) -> Result<()> {
        unsafe {
            launch(
                self.module.function(name)?,
                LaunchDims::linear(grid, 256),
                &self.stream,
                false,
                params,
            )
        }?;
        Ok(())
    }
    fn linear(&self, x: u64, y: u64, n: usize, name: &str, act: i32) -> Result<()> {
        let w = self.w(&format!("{name}.weight"))?;
        let o = w.shape[0] as i32;
        let k = w.shape[1..].iter().product::<u64>() as i32;
        blas(unsafe {
            cublasSgemm_v2(
                self.handle,
                1,
                0,
                o,
                n as i32,
                k,
                &1.0,
                w.data.ptr as *const f32,
                k,
                x as *const f32,
                k,
                &0.0,
                y as *mut f32,
                o,
            )
        })?;
        let len = n * o as usize;
        self.go(
            "bias_act",
            len.div_ceil(256) as u32,
            &mut args![
                y,
                self.optional(&format!("{name}.bias")),
                o,
                len as i32,
                act
            ],
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        x: u64,
        y: u64,
        n: usize,
        h: usize,
        name: &str,
        rms: bool,
        eps: f32,
    ) -> Result<()> {
        self.go(
            "vision_norm",
            n as u32,
            &mut args![
                x,
                y,
                self.ptr(&format!("{name}.weight"))?,
                self.optional(&format!("{name}.bias")),
                h as i32,
                eps,
                i32::from(rms)
            ],
        )
    }
    fn add(&self, x: u64, y: u64, n: usize) -> Result<()> {
        self.go("add", n.div_ceil(256) as u32, &mut args![x, y, n as i32])
    }
    pub fn encode(&mut self, image: &Image) -> Result<Vec<f32>> {
        let (n, h, nt) = (image.grid_h * image.grid_w, self.h, image.tokens());
        let pd = if self.kind == Kind::Glm {
            6 * 14 * 14
        } else {
            6 * 16 * 16
        };
        if n == 0
            || n > 16384
            || !image.grid_h.is_multiple_of(2)
            || !image.grid_w.is_multiple_of(2)
            || image.patches.len() != n * pd
        {
            return Err("invalid vision patch grid".into());
        }
        let b = |len| Buf::new(len * 4).map_err(String::from);
        let x = b(n * h)?;
        let norm = b(n * h)?;
        let qkv = b(n * h * 3)?;
        let packed = b(n * h * 3)?;
        let att = b(n * h)?;
        let mixed = b(n * h)?;
        let tile = n.min(512);
        let scores = b(self.heads * n * tile)?;
        let ff = b(n * self.ff)?;
        let up = b(n * self.ff)?;
        let patches = Buf::from_slice(&image.patches)?;
        tsumuji_cuda::synchronize_context()?;
        self.linear(patches.ptr, x.ptr, n, "patch_embed.proj", 0)?;
        let mut coords = Vec::with_capacity(n * 2);
        for by in 0..image.grid_h / 2 {
            for bx in 0..image.grid_w / 2 {
                for y in 0..2 {
                    for z in 0..2 {
                        coords.extend([(by * 2 + y) as i32, (bx * 2 + z) as i32]);
                    }
                }
            }
        }
        if self.kind == Kind::Qwen {
            // Learned 48x48 position grid, bilinear interpolation with align_corners=True.
            let table = self.w("pos_embed.weight")?.data.to_vec::<f32>(2304 * h)?;
            let mut pos = vec![0f32; n * h];
            for (i, c) in coords.as_chunks::<2>().0.iter().enumerate() {
                let fy = c[0] as f32 * 47.0 / (image.grid_h - 1) as f32;
                let fx = c[1] as f32 * 47.0 / (image.grid_w - 1) as f32;
                let (y, z) = (fy.floor() as usize, fx.floor() as usize);
                let (dy, dx) = (fy - y as f32, fx - z as f32);
                for (yy, wy) in [(y, 1.0 - dy), ((y + 1).min(47), dy)] {
                    for (xx, wx) in [(z, 1.0 - dx), ((z + 1).min(47), dx)] {
                        for j in 0..h {
                            pos[i * h + j] += table[(yy * 48 + xx) * h + j] * wy * wx;
                        }
                    }
                }
            }
            let pos = Buf::from_slice(&pos)?;
            tsumuji_cuda::synchronize_context()?;
            self.add(x.ptr, pos.ptr, n * h)?;
            self.stream.synchronize()?;
        }
        let coords = Buf::from_slice(&coords)?;
        tsumuji_cuda::synchronize_context()?;
        for i in 0..self.depth {
            let p = format!("blocks.{i}");
            self.norm(
                x.ptr,
                norm.ptr,
                n,
                h,
                &format!("{p}.norm1"),
                self.kind == Kind::Glm,
                self.eps,
            )?;
            self.linear(norm.ptr, qkv.ptr, n, &format!("{p}.attn.qkv"), 0)?;
            unsafe {
                launch(
                    self.module.function("qkv_rope")?,
                    LaunchDims {
                        grid: (n as u32, self.heads as u32, 3),
                        block: (256, 1, 1),
                        smem: 0,
                    },
                    &self.stream,
                    false,
                    &mut args![
                        qkv.ptr,
                        packed.ptr,
                        self.optional(&format!("{p}.attn.q_norm.weight")),
                        self.optional(&format!("{p}.attn.k_norm.weight")),
                        coords.ptr,
                        n as i32,
                        h as i32,
                        self.heads as i32,
                        self.eps
                    ],
                )
            }?;
            let d = (h / self.heads) as i32;
            let nh = (n * h * 4) as u64;
            let stride = (n * h / self.heads) as i64;
            for first in (0..n).step_by(tile) {
                let count = tile.min(n - first);
                let offset = (first * (d as usize) * 4) as u64;
                blas(unsafe {
                    cublasSgemmStridedBatched(
                        self.handle,
                        1,
                        0,
                        n as i32,
                        count as i32,
                        d,
                        &(1.0 / (d as f32).sqrt()),
                        (packed.ptr + nh) as *const f32,
                        d,
                        stride,
                        (packed.ptr + offset) as *const f32,
                        d,
                        stride,
                        &0.0,
                        scores.ptr as *mut f32,
                        n as i32,
                        (n * count) as i64,
                        self.heads as i32,
                    )
                })?;
                self.go(
                    "softmax",
                    (count * self.heads) as u32,
                    &mut args![scores.ptr, n as i32],
                )?;
                blas(unsafe {
                    cublasSgemmStridedBatched(
                        self.handle,
                        0,
                        0,
                        d,
                        count as i32,
                        n as i32,
                        &1.0,
                        (packed.ptr + 2 * nh) as *const f32,
                        d,
                        stride,
                        scores.ptr as *const f32,
                        n as i32,
                        (n * count) as i64,
                        &0.0,
                        (att.ptr + offset) as *mut f32,
                        d,
                        stride,
                        self.heads as i32,
                    )
                })?;
            }
            self.go(
                "unpack",
                (n * h).div_ceil(256) as u32,
                &mut args![att.ptr, mixed.ptr, n as i32, h as i32, self.heads as i32],
            )?;
            self.linear(mixed.ptr, norm.ptr, n, &format!("{p}.attn.proj"), 0)?;
            self.add(x.ptr, norm.ptr, n * h)?;
            self.norm(
                x.ptr,
                norm.ptr,
                n,
                h,
                &format!("{p}.norm2"),
                self.kind == Kind::Glm,
                self.eps,
            )?;
            if self.kind == Kind::Glm {
                self.linear(norm.ptr, ff.ptr, n, &format!("{p}.mlp.gate_proj"), 0)?;
                self.linear(norm.ptr, up.ptr, n, &format!("{p}.mlp.up_proj"), 0)?;
                self.go(
                    "swiglu",
                    (n * self.ff).div_ceil(256) as u32,
                    &mut args![ff.ptr, up.ptr, (n * self.ff) as i32],
                )?;
                self.linear(ff.ptr, norm.ptr, n, &format!("{p}.mlp.down_proj"), 0)?;
            } else {
                self.linear(norm.ptr, ff.ptr, n, &format!("{p}.mlp.linear_fc1"), 1)?;
                self.linear(ff.ptr, norm.ptr, n, &format!("{p}.mlp.linear_fc2"), 0)?;
            }
            self.add(x.ptr, norm.ptr, n * h)?;
        }
        let result = b(nt * self.out)?;
        if self.kind == Kind::Qwen {
            self.norm(x.ptr, norm.ptr, n, h, "merger.norm", false, 1e-6)?;
            let middle = b(nt * 4 * h)?;
            self.linear(norm.ptr, middle.ptr, nt, "merger.linear_fc1", 2)?;
            self.linear(middle.ptr, result.ptr, nt, "merger.linear_fc2", 0)?;
            self.stream.synchronize()?;
        } else {
            self.norm(x.ptr, norm.ptr, n, h, "post_layernorm", true, self.eps)?;
            let merge = b(n * h)?;
            self.go(
                "downsample_pack",
                (n * h).div_ceil(256) as u32,
                &mut args![norm.ptr, merge.ptr, nt as i32, h as i32],
            )?;
            let down = b(nt * self.out)?;
            let proj = b(nt * self.out)?;
            let normed = b(nt * self.out)?;
            let gate = b(nt * 10240)?;
            let up = b(nt * 10240)?;
            self.linear(merge.ptr, down.ptr, nt, "downsample", 0)?;
            self.linear(down.ptr, proj.ptr, nt, "merger.proj", 0)?;
            self.norm(
                proj.ptr,
                normed.ptr,
                nt,
                self.out,
                "merger.post_projection_norm",
                false,
                1e-5,
            )?;
            self.go(
                "bias_act",
                (nt * self.out).div_ceil(256) as u32,
                &mut args![
                    normed.ptr,
                    0u64,
                    self.out as i32,
                    (nt * self.out) as i32,
                    2i32
                ],
            )?;
            self.linear(normed.ptr, gate.ptr, nt, "merger.gate_proj", 0)?;
            self.linear(normed.ptr, up.ptr, nt, "merger.up_proj", 0)?;
            self.go(
                "swiglu",
                (nt * 10240).div_ceil(256) as u32,
                &mut args![gate.ptr, up.ptr, (nt * 10240) as i32],
            )?;
            self.linear(gate.ptr, result.ptr, nt, "merger.down_proj", 0)?;
            self.stream.synchronize()?;
        }
        let values = result.to_vec::<f32>(nt * self.out).map_err(String::from)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err("vision encoder produced a non-finite value".into());
        }
        Ok(values)
    }
    pub fn hidden(&self) -> usize {
        self.out
    }
}
