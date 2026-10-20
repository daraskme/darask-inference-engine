//! Resident Q4_K/Q6_K weights, FP32 recurrent state/KV, one captured decode graph.
use crate::config::{Config, tensor_bytes};
use darask_formats::gguf::Gguf;
use std::{collections::HashMap, fs::File, os::unix::fs::FileExt, path::Path, time::Instant};
use tsumuji_cuda::{
    Device, DeviceBuffer, Function, Graph, LaunchDims, Module, Stream, args, launch,
};
type Res<T> = Result<T, String>;
fn cu<T>(r: tsumuji_cuda::Result<T>) -> Res<T> {
    r.map_err(|e| e.to_string())
}
#[derive(Debug)]
struct Weight {
    data: DeviceBuffer,
    ty: i32,
    k: i32,
    rows: i32,
    ne: Vec<u64>,
}
#[derive(Debug)]
pub struct Engine {
    graph: Option<Graph>,
    module: Module,
    weights: HashMap<String, Weight>,
    buf: HashMap<&'static str, DeviceBuffer>,
    state: Vec<Vec<DeviceBuffer>>,
    stream: Stream,
    _device: Device,
    pub config: Config,
    pub context: usize,
    pub position: usize,
}
impl Engine {
    pub fn load(path: &Path, context: usize) -> Res<Self> {
        let g = Gguf::open(path)?;
        let config = Config::from_gguf(&g)?;
        if context == 0 || context > config.max_ctx {
            return Err("context outside model limits".into());
        }
        let device = cu(Device::open(0))?;
        let stream = cu(Stream::new())?;
        let module = cu(Module::load(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/native.cubin"
        ))))?;
        let file = File::open(path).map_err(|e| e.to_string())?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        let mut weights = HashMap::new();
        let start = Instant::now();
        let mut total = 0;
        // MTP weights remain in the file; the first native path implements exact plain decode.
        let trunk = g.tensors.iter().filter(|t| !t.name.starts_with("blk.48."));
        let bytes = trunk
            .clone()
            .try_fold(0usize, |n, t| tensor_bytes(t.ty, &t.ne).map(|b| n + b))?;
        let (free, _) = cu(device.mem_info())?;
        let kv_bytes = 12usize
            .checked_mul(2 * 2 * 256 * 4)
            .and_then(|b| b.checked_mul(context))
            .ok_or("KV size overflow")?;
        if bytes + kv_bytes + (1 << 30) > free {
            return Err(format!(
                "need {:.2} GiB including state/headroom, {:.2} GiB free",
                (bytes + kv_bytes + (1 << 30)) as f64 / (1u64 << 30) as f64,
                free as f64 / (1u64 << 30) as f64
            ));
        }
        let mut chunk = vec![0u8; 64 << 20];
        for t in trunk {
            let bytes = tensor_bytes(t.ty, &t.ne)?;
            if t.offset
                .checked_add(bytes as u64)
                .is_none_or(|end| end > file_len)
            {
                return Err(format!("{}: tensor exceeds file", t.name));
            }
            let data = cu(DeviceBuffer::new(bytes))?;
            for off in (0..bytes).step_by(chunk.len()) {
                let n = chunk.len().min(bytes - off);
                file.read_exact_at(&mut chunk[..n], t.offset + off as u64)
                    .map_err(|e| e.to_string())?;
                cu(tsumuji_cuda::check(
                    unsafe {
                        tsumuji_cuda::sys::cuMemcpyHtoD_v2(
                            data.ptr + off as u64,
                            chunk.as_ptr().cast(),
                            n,
                        )
                    },
                    "upload GGUF",
                ))?;
            }
            total += bytes;
            weights.insert(
                t.name.clone(),
                Weight {
                    data,
                    ty: t.ty as i32,
                    k: t.ne[0] as i32,
                    rows: t.ne.get(1).copied().unwrap_or(1) as i32,
                    ne: t.ne.clone(),
                },
            );
        }
        eprintln!(
            "Ornith native: {:.3} GiB loaded in {:.3}s",
            total as f64 / (1u64 << 30) as f64,
            start.elapsed().as_secs_f64()
        );
        let mut buf = HashMap::new();
        for (name, n) in [
            ("x", 3072),
            ("n", 3072),
            ("out", 3072),
            ("qkv", 16384),
            ("z", 8192),
            ("alpha", 64),
            ("beta", 64),
            ("mix", 8192),
            ("qg", 16384),
            ("k", 512),
            ("v", 512),
            ("scores", 32 * context),
            ("router", 256),
            ("ids", 8),
            ("prob", 8),
            ("gate", 8192),
            ("up", 8192),
            ("ff", 8192),
            ("down", 8 * 3072),
            ("sgate", 1024),
            ("sup", 1024),
            ("sff", 1024),
            ("shared", 3072),
            ("sg", 1),
            ("logits", config.vocab),
            ("token", 1),
            ("pos", 1),
        ] {
            buf.insert(name, cu(DeviceBuffer::zeroed(n * 4))?);
        }
        let mut state = Vec::new();
        for l in 0..48 {
            let sizes = if config.recurrent(l) {
                vec![12288 * 3 * 4, 64 * 128 * 128 * 4]
            } else {
                vec![2 * context * 256 * 4, 2 * context * 256 * 4]
            };
            state.push(
                sizes
                    .into_iter()
                    .map(|n| cu(DeviceBuffer::zeroed(n)))
                    .collect::<Res<Vec<_>>>()?,
            );
        }
        let mut engine = Self {
            graph: None,
            module,
            weights,
            buf,
            state,
            stream,
            _device: device,
            config,
            context,
            position: 0,
        };
        // Resolve every name before capture, and report missing tensors as an error.
        engine.validate()?;
        cu(tsumuji_cuda::check(
            unsafe { tsumuji_cuda::sys::cuCtxSynchronize() },
            "initialize native buffers",
        ))?;
        if std::env::var_os("DARASK_DUMP_DIR").is_none() {
            engine.graph = Some(
                engine
                    .stream
                    .capture::<String>(|stream| engine.record(stream))?,
            );
        }
        Ok(engine)
    }
    fn validate(&self) -> Res<()> {
        let shape = |name: &str, ne: &[u64], f32_only: bool| -> Res<()> {
            let w = self.w(name)?;
            if w.ne != ne || (f32_only && w.ty != 0) {
                return Err(format!(
                    "{name}: incompatible shape/type {:?}/{}",
                    w.ne, w.ty
                ));
            }
            Ok(())
        };
        for key in ["token_embd.weight", "output.weight"] {
            shape(key, &[3072, 248320], false)?;
        }
        shape("output_norm.weight", &[3072], true)?;
        for l in 0..48 {
            let n = |s: &str| format!("blk.{l}.{s}");
            for key in [
                "attn_norm.weight",
                "post_attention_norm.weight",
                "ffn_gate_inp_shexp.weight",
            ] {
                shape(&n(key), &[3072], true)?;
            }
            shape(&n("ffn_gate_inp.weight"), &[3072, 256], true)?;
            for key in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
                shape(&n(key), &[3072, 1024, 256], false)?;
            }
            shape(&n("ffn_down_exps.weight"), &[1024, 3072, 256], false)?;
            for key in ["ffn_gate_shexp.weight", "ffn_up_shexp.weight"] {
                shape(&n(key), &[3072, 1024], false)?;
            }
            shape(&n("ffn_down_shexp.weight"), &[1024, 3072], false)?;
            if self.config.recurrent(l) {
                shape(&n("attn_qkv.weight"), &[3072, 12288], false)?;
                shape(&n("attn_gate.weight"), &[3072, 8192], false)?;
                for key in ["ssm_alpha.weight", "ssm_beta.weight"] {
                    shape(&n(key), &[3072, 64], false)?;
                }
                for key in ["ssm_a", "ssm_dt.bias"] {
                    shape(&n(key), &[64], true)?;
                }
                shape(&n("ssm_conv1d.weight"), &[4, 12288], true)?;
                shape(&n("ssm_norm.weight"), &[128], true)?;
                shape(&n("ssm_out.weight"), &[8192, 3072], false)?;
            } else {
                shape(&n("attn_q.weight"), &[3072, 16384], false)?;
                for key in ["attn_k.weight", "attn_v.weight"] {
                    shape(&n(key), &[3072, 512], false)?;
                }
                for key in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                    shape(&n(key), &[256], true)?;
                }
                shape(&n("attn_output.weight"), &[8192, 3072], false)?;
            }
        }
        Ok(())
    }
    fn w(&self, name: &str) -> Res<&Weight> {
        self.weights
            .get(name)
            .ok_or_else(|| format!("missing {name}"))
    }
    fn b(&self, name: &str) -> u64 {
        self.buf[name].ptr
    }
    fn call(
        &self,
        name: &str,
        grid: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
        stream: &Stream,
    ) -> Res<()> {
        let f: Function = cu(self.module.function(name))?;
        cu(unsafe { launch(f, LaunchDims::linear(grid, block), stream, false, args) })
    }
    fn mv(
        &self,
        name: &str,
        x: u64,
        y: u64,
        ids: u64,
        selected: i32,
        count: u32,
        stream: &Stream,
    ) -> Res<()> {
        let w = self.w(name)?;
        let f = cu(self.module.function("gemv"))?;
        let dims = LaunchDims {
            grid: ((w.rows as u32).div_ceil(4), count, 1),
            block: (128, 1, 1),
            smem: 0,
        };
        cu(unsafe {
            launch(
                f,
                dims,
                stream,
                false,
                &mut args![w.data.ptr, w.ty, x, y, w.k, w.rows, ids, selected],
            )
        })
    }
    fn trace(&self, key: &str, label: &str, s: &Stream) -> Res<()> {
        if let Some(dir) = std::env::var_os("DARASK_DUMP_DIR") {
            cu(s.synchronize())?;
            let data = cu(self.buf[key].to_vec::<f32>(self.buf[key].bytes / 4))?;
            let path = Path::new(&dir).join(format!("{label}.f32"));
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let raw: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(path, raw).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn record(&self, s: &Stream) -> Res<()> {
        let c = &self.config;
        let h = 3072i32;
        let ctx = self.context as i32;
        let zero = 0u64;
        let w = self.w("token_embd.weight")?;
        self.call(
            "embed",
            12,
            256,
            &mut args![w.data.ptr, w.ty, self.b("x"), self.b("token"), h],
            s,
        )?;
        self.trace("x", "embed", s)?;
        for l in 0..48 {
            let name = |suffix: &str| format!("blk.{l}.{suffix}");
            self.call(
                "rms_native",
                1,
                256,
                &mut args![
                    self.b("x"),
                    self.w(&name("attn_norm.weight"))?.data.ptr,
                    self.b("n"),
                    h,
                    c.eps
                ],
                s,
            )?;
            if l == 0 {
                self.trace("n", "attn_norm-0", s)?;
            }
            if c.recurrent(l) {
                for (suffix, out) in [
                    ("attn_qkv.weight", "qkv"),
                    ("attn_gate.weight", "z"),
                    ("ssm_alpha.weight", "alpha"),
                    ("ssm_beta.weight", "beta"),
                ] {
                    self.mv(&name(suffix), self.b("n"), self.b(out), zero, 0, 1, s)?;
                }
                if l == 0 {
                    self.trace("qkv", "qkv-0", s)?;
                    self.trace("z", "z-0", s)?;
                }
                self.call(
                    "conv",
                    48,
                    256,
                    &mut args![
                        self.b("qkv"),
                        self.w(&name("ssm_conv1d.weight"))?.data.ptr,
                        self.state[l][0].ptr,
                        12288i32
                    ],
                    s,
                )?;
                if l == 0 {
                    self.trace("qkv", "conv-0", s)?;
                }
                self.call(
                    "qk_l2",
                    32,
                    128,
                    &mut args![self.b("qkv"), 128i32, c.eps],
                    s,
                )?;
                self.call(
                    "delta",
                    64 * 32,
                    128,
                    &mut args![
                        self.b("qkv"),
                        self.b("alpha"),
                        self.b("beta"),
                        self.w(&name("ssm_a"))?.data.ptr,
                        self.w(&name("ssm_dt.bias"))?.data.ptr,
                        self.state[l][1].ptr,
                        self.b("mix")
                    ],
                    s,
                )?;
                if l == 0 {
                    self.trace("mix", "delta-0", s)?;
                }
                self.call(
                    "gated_norm",
                    64,
                    128,
                    &mut args![
                        self.b("mix"),
                        self.b("z"),
                        self.w(&name("ssm_norm.weight"))?.data.ptr,
                        c.eps
                    ],
                    s,
                )?;
                if l == 0 {
                    self.trace("mix", "gated_norm-0", s)?;
                }
                self.mv(
                    &name("ssm_out.weight"),
                    self.b("mix"),
                    self.b("out"),
                    zero,
                    0,
                    1,
                    s,
                )?;
            } else {
                for (suffix, out) in [
                    ("attn_q.weight", "qg"),
                    ("attn_k.weight", "k"),
                    ("attn_v.weight", "v"),
                ] {
                    self.mv(&name(suffix), self.b("n"), self.b(out), zero, 0, 1, s)?;
                }
                self.call(
                    "qk_rope",
                    34,
                    256,
                    &mut args![
                        self.b("qg"),
                        self.b("k"),
                        self.b("v"),
                        self.w(&name("attn_q_norm.weight"))?.data.ptr,
                        self.w(&name("attn_k_norm.weight"))?.data.ptr,
                        self.state[l][0].ptr,
                        self.state[l][1].ptr,
                        self.b("pos"),
                        ctx,
                        c.theta,
                        c.eps
                    ],
                    s,
                )?;
                let f = cu(self.module.function("scores"))?;
                let dims = LaunchDims {
                    grid: ((ctx as u32).div_ceil(4).min(32), 32, 1),
                    block: (128, 1, 1),
                    smem: 0,
                };
                cu(unsafe {
                    launch(
                        f,
                        dims,
                        s,
                        false,
                        &mut args![
                            self.b("qg"),
                            self.state[l][0].ptr,
                            self.b("scores"),
                            self.b("pos"),
                            ctx
                        ],
                    )
                })?;
                self.call(
                    "softmax",
                    32,
                    256,
                    &mut args![self.b("scores"), self.b("pos"), ctx],
                    s,
                )?;
                self.call(
                    "attention",
                    32,
                    256,
                    &mut args![
                        self.b("scores"),
                        self.state[l][1].ptr,
                        self.b("qg"),
                        self.b("mix"),
                        self.b("pos"),
                        ctx
                    ],
                    s,
                )?;
                self.mv(
                    &name("attn_output.weight"),
                    self.b("mix"),
                    self.b("out"),
                    zero,
                    0,
                    1,
                    s,
                )?;
            }
            if l == 0 {
                self.trace("out", "linear_attn_out-0", s)?;
            }
            self.call(
                "residual",
                12,
                256,
                &mut args![self.b("x"), self.b("out"), h],
                s,
            )?;
            self.call(
                "rms_native",
                1,
                256,
                &mut args![
                    self.b("x"),
                    self.w(&name("post_attention_norm.weight"))?.data.ptr,
                    self.b("n"),
                    h,
                    c.eps
                ],
                s,
            )?;
            if l == 0 {
                self.trace("n", "attn_post_norm-0", s)?;
            }
            self.mv(
                &name("ffn_gate_inp.weight"),
                self.b("n"),
                self.b("router"),
                zero,
                0,
                1,
                s,
            )?;
            self.call(
                "route",
                1,
                32,
                &mut args![self.b("router"), self.b("ids"), self.b("prob")],
                s,
            )?;
            for (suffix, out) in [
                ("ffn_gate_exps.weight", "gate"),
                ("ffn_up_exps.weight", "up"),
            ] {
                self.mv(
                    &name(suffix),
                    self.b("n"),
                    self.b(out),
                    self.b("ids"),
                    0,
                    8,
                    s,
                )?;
            }
            self.call(
                "activate",
                32,
                256,
                &mut args![self.b("gate"), self.b("up"), self.b("ff"), 8192i32],
                s,
            )?;
            self.mv(
                &name("ffn_down_exps.weight"),
                self.b("ff"),
                self.b("down"),
                self.b("ids"),
                1,
                8,
                s,
            )?;
            self.mv(
                &name("ffn_gate_shexp.weight"),
                self.b("n"),
                self.b("sgate"),
                zero,
                0,
                1,
                s,
            )?;
            self.mv(
                &name("ffn_up_shexp.weight"),
                self.b("n"),
                self.b("sup"),
                zero,
                0,
                1,
                s,
            )?;
            self.call(
                "activate",
                4,
                256,
                &mut args![self.b("sgate"), self.b("sup"), self.b("sff"), 1024i32],
                s,
            )?;
            self.mv(
                &name("ffn_down_shexp.weight"),
                self.b("sff"),
                self.b("shared"),
                zero,
                0,
                1,
                s,
            )?;
            self.mv(
                &name("ffn_gate_inp_shexp.weight"),
                self.b("n"),
                self.b("sg"),
                zero,
                0,
                1,
                s,
            )?;
            self.call(
                "combine",
                12,
                256,
                &mut args![
                    self.b("x"),
                    self.b("down"),
                    self.b("prob"),
                    self.b("shared"),
                    self.b("sg"),
                    h
                ],
                s,
            )?;
            self.trace("x", &format!("l_out-{l}"), s)?;
        }
        self.call(
            "rms_native",
            1,
            256,
            &mut args![
                self.b("x"),
                self.w("output_norm.weight")?.data.ptr,
                self.b("n"),
                h,
                c.eps
            ],
            s,
        )?;
        self.mv(
            "output.weight",
            self.b("n"),
            self.b("logits"),
            zero,
            0,
            1,
            s,
        )?;
        self.call(
            "greedy",
            1,
            256,
            &mut args![
                self.b("logits"),
                self.b("token"),
                self.b("pos"),
                c.vocab as i32
            ],
            s,
        )
    }
    pub fn step(&mut self, token: u32) -> Res<u32> {
        if self.position >= self.context || token as usize >= self.config.vocab {
            return Err("token/context out of bounds".into());
        }
        cu(self.buf["token"].upload(&[token]))?;
        if let Some(g) = &self.graph {
            cu(g.launch(&self.stream))?;
        } else {
            self.record(&self.stream)?;
        }
        cu(self.stream.synchronize())?;
        let next = cu(self.buf["token"].to_vec::<u32>(1))?[0];
        self.position += 1;
        Ok(next)
    }
    pub fn logits(&self) -> Res<Vec<f32>> {
        cu(self.buf["logits"].to_vec(self.config.vocab))
    }
    pub fn reset(&mut self) -> Res<()> {
        cu(self.stream.synchronize())?;
        for layer in &self.state {
            for b in layer {
                cu(b.zero_on(&self.stream))?;
            }
        }
        for buffer in self.buf.values() {
            cu(buffer.zero_on(&self.stream))?;
        }
        cu(self.stream.synchronize())?;
        self.position = 0;
        Ok(())
    }
}
