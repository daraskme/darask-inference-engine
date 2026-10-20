//! Runtime rank-1 adapters on the matrices that write into the residual: each layer's mixer output
//! projection (QSA `o_proj`, GDN `out_proj`), the shared expert's `down_proj` and the routed experts'
//! `down_proj`. A matrix W gets y += s (a . x) b for its input x, computed in fp32 next to the
//! quantized matrix (kernels/include/rank1.cuh), so nothing is requantized and the adapter can be
//! switched per request through the device scale s.
//!
//! Source: a llama.cpp LoRA adapter GGUF of rank 1, as heretic writes them (tensor names
//! `blk.{l}.attn_output.weight.lora_{a,b}` / `ssm_out` / `ffn_down_shexp` / `ffn_down_exps`;
//! ggml dims: lora_a [in, 1(, E)], lora_b [1, out(, E)]); its scale alpha / rank is folded in.
//! Abliterations such as huihui-ai's (W' = W - c r r^T W) are this form with a = r^T W, b = -c r.

use crate::config::Dims;
use crate::weights::Res;
use std::path::Path;
use tsumuji_cuda::DeviceBuffer;
use tsumuji_pack::gguf::Gguf;

/// One matrix's adapter on the device: a [in] (routed experts [E][in]), b [out] ([E][out]).
#[derive(Debug)]
pub struct Rank1Dev {
    pub a: DeviceBuffer,
    pub b: DeviceBuffer,
}

#[derive(Debug, Default)]
pub struct LayerAdapter {
    pub attn: Option<Rank1Dev>,
    pub shared: Option<Rank1Dev>,
    pub experts: Option<Rank1Dev>,
}

#[derive(Debug)]
pub struct Adapter {
    pub name: String,
    pub layers: Vec<LayerAdapter>,
}

impl Adapter {
    /// Loads a rank-1 LoRA adapter GGUF for the model `d`.
    pub fn load_gguf(path: &Path, d: &Dims) -> Res<Self> {
        let g = Gguf::open(path)?;
        if g.str("general.type").is_some_and(|t| t != "adapter") {
            return Err(format!("{}: not an adapter GGUF", path.display()));
        }
        let mut layers: Vec<LayerAdapter> =
            (0..d.layers).map(|_| LayerAdapter::default()).collect();
        let mut rank_seen = None;
        let up = |v: &[f32]| -> Res<DeviceBuffer> {
            DeviceBuffer::from_slice(v).map_err(|e| e.to_string())
        };
        for t in &g.tensors {
            let Some(base) = t.name.strip_suffix(".lora_a") else {
                continue;
            };
            let lb = g
                .tensor(&format!("{base}.lora_b"))
                .ok_or(format!("{base}: lora_a without lora_b"))?;
            let rest = base
                .strip_prefix("blk.")
                .ok_or(format!("{base}: unexpected tensor"))?;
            let (l, what) = rest
                .split_once('.')
                .ok_or(format!("{base}: unexpected tensor"))?;
            let l: usize = l.parse().map_err(|_| format!("{base}: bad layer"))?;
            if l >= d.layers {
                return Err(format!("{base}: layer {l} out of range"));
            }
            let rank = t.ne[1];
            if rank != 1 || lb.ne[0] != 1 {
                return Err(format!(
                    "{base}: rank {rank} (only rank-1 adapters are supported)"
                ));
            }
            rank_seen = Some(rank);
            let (a, mut b) = (g.read_f32(t)?, g.read_f32(lb)?);
            let (inp, out) = (t.ne[0] as usize, lb.ne[1] as usize);
            let experts = if t.ne.len() == 3 { t.ne[2] as usize } else { 1 };
            let expect = |name: &str, i: usize, o: usize, e: usize| -> Res<()> {
                if (inp, out, experts) != (i, o, e) {
                    return Err(format!(
                        "{base}: shape in {inp} out {out} experts {experts}, expected {name} {i} x {o} x {e}"
                    ));
                }
                Ok(())
            };
            let slot = match what {
                "attn_output.weight" | "ssm_out.weight" => {
                    let k = if d.qsa[l] {
                        d.att_q * d.att_d
                    } else {
                        d.gdn_v * d.gdn_d
                    };
                    expect("mixer output", k, d.h, 1)?;
                    &mut layers[l].attn
                }
                "ffn_down_shexp.weight" => {
                    expect("shared down", d.ff_shared, d.h, 1)?;
                    &mut layers[l].shared
                }
                "ffn_down_exps.weight" => {
                    expect("experts down", d.ff, d.h, d.experts)?;
                    &mut layers[l].experts
                }
                other => return Err(format!("{base}: adapter on {other} is not supported")),
            };
            // llama.cpp applies B A with scale alpha / rank (heretic: alpha = rank).
            let alpha = g.f64("adapter.lora.alpha").unwrap_or(1.0) as f32;
            if alpha != 1.0 {
                b.iter_mut().for_each(|v| *v *= alpha);
            }
            *slot = Some(Rank1Dev {
                a: up(&a)?,
                b: up(&b)?,
            });
        }
        if rank_seen.is_none() {
            return Err(format!("{}: no LoRA tensors", path.display()));
        }
        let name = path
            .file_stem()
            .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
        Ok(Self { name, layers })
    }
}
