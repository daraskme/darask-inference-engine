//! Device decode of one layer (stage 19): every kernel on one stream, with the
//! position, the doorbell sequence and the expert pointer table read from
//! device memory, so a whole token can be one CUDA graph. The arithmetic is the
//! per-op decode path's (bit-identical logits); see cu/glm53/device.cu.
use super::*;
use hayate_kernels::cuda::ffi::Stream as RawStream;
use hayate_kernels::glm53::lowbit::ffi as lowbit;

/// Device addresses shared by every layer of a device token.
#[derive(Debug, Clone, Copy)]
pub(in crate::glm53) struct DeviceRefs {
    /// Control block {token, position, seq, error, next} (u32 each).
    pub ctl: Ptr,
    /// [42][288] GPU image pointers of layers 3.. (null: CPU expert).
    pub table: Ptr,
    /// [42][288] ExpertFormat (8 bytes) per expert for LOWBIT layers (null: none).
    pub formats: Ptr,
    /// [46] CPU-expert plans {now, later, skip mask, defer mask} (u32 each).
    pub plans: Ptr,
    /// CPU experts with a routing weight below this are dropped (0: none).
    pub skip_below: f32,
    /// Last-block counter (zero between launches).
    pub counter: Ptr,
    /// 24 F32 mHC logits.
    pub logits: Ptr,
    /// GPU wait limit for a CPU doorbell before the error flag is raised.
    pub limit_ns: u64,
    /// The PCIe lane (null `host_table`: none): [42][288] RAM-tier image pointers, [8] device scratch slots
    /// (4 used), [8] fetch sources the router writes (4..8 stay null), and take[n] as nibble n.
    pub host_table: Ptr,
    pub scratch: Ptr,
    pub fetch: Ptr,
    pub take: u64,
    /// The lane's prefetch (0: none): slots per layer, [2][8] slot pointers and sources by layer parity, the
    /// copy state (`PRE_*` offsets: per parity u32 [8] done, u32 [8] bad, u64 [8] tags; the router's u64
    /// progress) and the u32 [4] counters.
    pub prefetch: u32,
    pub pre_scratch: Ptr,
    pub pre_src: Ptr,
    pub pre_meta: Ptr,
    pub pcie_stats: Ptr,
}
/// Blocks per slot of the prefetch copy: few, so the main stream's kernels keep the SMs while it streams.
const PREFETCH_BLOCKS: u32 = 8;
/// `DeviceRefs::pre_meta` layout (bytes): done [2][8] u32, bad [2][8] u32, tags [2][8] u64, progress u64.
pub(in crate::glm53) const PRE_DONE: usize = 0;
pub(in crate::glm53) const PRE_BAD: usize = 64;
pub(in crate::glm53) const PRE_TAGS: usize = 128;
pub(in crate::glm53) const PRE_PROGRESS: usize = 256;
pub(in crate::glm53) const PRE_META_BYTES: usize = 264;

/// The PCIe lane's prefetch stream and fork/join events: layer k's pick for layer k+1 is copied there during layer
/// k's experts and layer k+1's attention; nothing waits for it until the token's end (EXL3.md §6).
#[derive(Debug)]
pub(in crate::glm53) struct LaneSync {
    pub stream: Stream,
    /// Per target layer index: (pick done on the main stream, copy done on the side stream).
    pub events: Vec<(Event, Event)>,
}

/// Expert deferral (KTransformers): CPU experts of MoE layer k with a routing
/// weight below `below` join the residual at layer k+1, so the CPU overlaps the
/// next attention (0: none); `prev` adds the previous layer's.
#[derive(Debug, Clone, Copy)]
pub(in crate::glm53) struct Defer {
    pub below: f32,
    pub prev: Option<Prev>,
}
#[derive(Debug, Clone, Copy)]
pub(in crate::glm53) struct Prev {
    mailbox: Ptr,
    ids: Ptr,
    weights: Ptr,
    images: Ptr,
    post: Ptr,
    plan: Ptr,
}

impl Layer<'_> {
    /// Hadamard-rotated experts keep the per-op path.
    pub(in crate::glm53) fn device_supported(&self) -> bool {
        self.context > 0 && self.index < 45 && self.hadamard_block == 0
    }
    fn segment(&self, weight: &str, out: &str, rows: usize) -> g::Segment {
        g::Segment {
            w: self.dense_weight(weight),
            out: self.p(out),
            rows: rows as u32,
        }
    }
    fn device_gemv(&self, st: RawStream, segments: &[g::Segment], x: Ptr, k: usize) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_dev_gemv(segments.as_ptr(), segments.len() as u32, x, k as u32, st)
            },
            "glm device GEMV",
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn device_hc(
        &self,
        st: RawStream,
        site: &str,
        map: &str,
        collapsed: &str,
        out: &str,
        input: Ptr,
        weight: &str,
        refs: &DeviceRefs,
    ) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_dev_hc_norm(
                    self.p(map),
                    self.p(collapsed),
                    self.p(out),
                    input,
                    self.w(&format!("hc_{site}_fn")),
                    self.w(&format!("hc_{site}_base")),
                    self.w(&format!("hc_{site}_scale")),
                    self.w(weight),
                    refs.logits,
                    refs.counter,
                    st,
                )
            },
            "glm device mHC",
        )
    }
    fn device_expand(
        &self,
        st: RawStream,
        out: &str,
        input: Ptr,
        branch: &str,
        map: &str,
    ) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_expand(
                    self.p(out),
                    input,
                    self.p(branch),
                    self.p(map),
                    1,
                    HIDDEN as u32,
                    st,
                )
            },
            "glm device expand",
        )
    }
    fn device_kda(&self, st: RawStream) -> Result<()> {
        let x = self.p("attn_norm");
        self.device_gemv(
            st,
            &[
                self.segment("self_attn.q_proj.weight", "q", 8192),
                self.segment("self_attn.k_proj.weight", "k", 8192),
                self.segment("self_attn.v_proj.weight", "v", 8192),
                self.segment("self_attn.f_a_proj.weight", "fa", 128),
                self.segment("self_attn.b_proj.weight", "beta", 64),
                self.segment("self_attn.g_a_proj.weight", "ga", 128),
            ],
            x,
            HIDDEN,
        )?;
        check(
            unsafe {
                g::hayate_glm_dev_kda(
                    self.p("kda"),
                    self.p("gated"),
                    self.p("state"),
                    self.p("hq"),
                    self.p("hk"),
                    self.p("hv"),
                    self.p("q"),
                    self.p("k"),
                    self.p("v"),
                    self.w("self_attn.q_conv1d.weight"),
                    self.w("self_attn.k_conv1d.weight"),
                    self.w("self_attn.v_conv1d.weight"),
                    self.p("fa"),
                    self.dense_weight("self_attn.f_b_proj.weight"),
                    self.p("ga"),
                    self.dense_weight("self_attn.g_b_proj.weight"),
                    self.w("self_attn.dt_bias"),
                    self.w("self_attn.A_log"),
                    self.p("beta"),
                    self.w("self_attn.o_norm.weight"),
                    st,
                )
            },
            "glm device KDA",
        )?;
        self.device_gemv(
            st,
            &[self.segment("self_attn.o_proj.weight", "attention", HIDDEN)],
            self.p("gated"),
            8192,
        )?;
        self.lora_linear(
            st,
            "self_attn.o_proj.weight",
            self.p("attention"),
            self.p("gated"),
            1,
        )
    }
    /// `mark` 5 / 6 / 7: before the top-512 selection, before and after MLA.
    fn device_sparse(
        &self,
        st: RawStream,
        refs: &DeviceRefs,
        mark: &dyn Fn(usize) -> Result<()>,
    ) -> Result<()> {
        self.device_gemv(
            st,
            &[
                self.segment("self_attn.q_a_proj.weight", "qa", 1536),
                self.segment("self_attn.kv_a_proj_with_mqa.weight", "kva", 512),
                self.segment("self_attn.indexer.wk.weight", "ik", 128),
                self.segment("self_attn.indexer.index_kpool_compress_gate", "ig", 128),
                self.segment("self_attn.indexer.weights_proj.weight", "iw", 32),
            ],
            self.p("attn_norm"),
            HIDDEN,
        )?;
        check(
            unsafe {
                g::hayate_glm_dev_sparse_prep(
                    self.p("qr"),
                    self.p("qa"),
                    self.w("self_attn.q_a_layernorm.weight"),
                    self.p("kvr"),
                    self.p("kva"),
                    self.w("self_attn.kv_a_layernorm.weight"),
                    self.p("latent_cache"),
                    self.p("ikn"),
                    self.p("ik"),
                    self.w("self_attn.indexer.k_norm.weight"),
                    self.w("self_attn.indexer.k_norm.bias"),
                    self.p("ig"),
                    self.p("pool_keys"),
                    self.p("pool_gates"),
                    self.p("pools"),
                    self.w("self_attn.indexer.index_kpool_compress_ape"),
                    refs.ctl,
                    st,
                )
            },
            "glm device sparse prep",
        )?;
        self.device_gemv(
            st,
            &[
                self.segment("self_attn.q_b_proj.weight", "q", 16384),
                self.segment("self_attn.indexer.wq_b.weight", "iq", 4096),
            ],
            self.p("qr"),
            1536,
        )?;
        mark(5)?;
        check(
            unsafe {
                g::hayate_glm_mla_select_device(
                    self.p("selected"),
                    self.p("scores"),
                    self.p("iq"),
                    self.p("pools"),
                    self.p("iw"),
                    refs.ctl.cast::<u8>().add(4).cast(),
                    self.context.div_ceil(4),
                    st,
                )
            },
            "glm device select",
        )?;
        mark(6)?;
        self.mla_decode_on(st, self.p("q"), self.p("selected"), self.p("attn_core"), 1)?;
        mark(7)?;
        self.device_gemv(
            st,
            &[self.segment("self_attn.o_proj.weight", "attention", HIDDEN)],
            self.p("attn_core"),
            16384,
        )?;
        self.lora_linear(
            st,
            "self_attn.o_proj.weight",
            self.p("attention"),
            self.p("attn_core"),
            1,
        )
    }
    /// Enqueues this layer's decode step at the device position on `stream`
    /// and returns its output (input of the next layer). `mailbox` is this
    /// MoE layer's CPU doorbell; `format` the expert image format. `mark(i)`
    /// is called after phase i: 1 attention, 2 FFN mHC + router, 3 GPU
    /// experts (or dense FFN), 4 CPU doorbell wait; sparse layers also 5-7.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm53) fn device_step(
        &self,
        stream: &Stream,
        input: Ptr,
        refs: &DeviceRefs,
        mailbox: Ptr,
        format: (bool, bool),
        defer: Defer,
        lane: Option<&LaneSync>,
        next: Option<&Layer<'_>>,
        mark: &dyn Fn(usize) -> Result<()>,
    ) -> Result<Ptr> {
        let st = stream.as_raw();
        self.device_hc(
            st,
            "attn",
            "attn_map",
            "attn_collapsed",
            "attn_norm",
            input,
            "input_layernorm.weight",
            refs,
        )?;
        if self.index % 4 != 3 {
            self.device_kda(st)?;
        } else {
            self.device_sparse(st, refs, mark)?;
        }
        self.device_expand(st, "attention_residual", input, "attention", "attn_map")?;
        mark(1)?;
        self.device_hc(
            st,
            "ffn",
            "ffn_map",
            "ffn_collapsed",
            "ffn_norm",
            self.p("attention_residual"),
            "post_attention_layernorm.weight",
            refs,
        )?;
        if self.index < 3 {
            mark(2)?;
            self.device_gemv(
                st,
                &[
                    self.segment("mlp.gate_proj.weight", "gate", 12288),
                    self.segment("mlp.up_proj.weight", "up", 12288),
                ],
                self.p("ffn_norm"),
                HIDDEN,
            )?;
            for (weight, out) in [
                ("mlp.gate_proj.weight", "gate"),
                ("mlp.up_proj.weight", "up"),
            ] {
                self.lora_linear(st, weight, self.p(out), self.p("ffn_norm"), 1)?;
            }
            check(
                unsafe {
                    g::hayate_glm_point(
                        self.p("product"),
                        self.p("gate"),
                        self.p("up"),
                        12288,
                        1,
                        st,
                    )
                },
                "glm device SwiGLU",
            )?;
            self.device_gemv(
                st,
                &[self.segment("mlp.down_proj.weight", "mlp", HIDDEN)],
                self.p("product"),
                12288,
            )?;
            self.lora_linear(
                st,
                "mlp.down_proj.weight",
                self.p("mlp"),
                self.p("product"),
                1,
            )?;
            mark(3)?;
            mark(4)?;
            self.device_expand(st, "output", self.p("attention_residual"), "mlp", "ffn_map")?;
            return Ok(self.p("output"));
        }
        let plan: Ptr = unsafe { refs.plans.cast::<u32>().add(4 * self.index).cast() };
        let table = unsafe {
            refs.table
                .cast::<u64>()
                .add((self.index - 3) * EXPERTS)
                .cast()
        };
        let (nv, down_fp8) = format;
        let stem = "mlp.shared_experts";
        // The lane's prefetch: layer k's pick for k+1 lands in set (k+1) % 2; the first MoE layer has none.
        let prefetch = refs.prefetch > 0 && !refs.host_table.is_null() && lane.is_some();
        let pre = |index: usize, base: Ptr| -> Ptr {
            unsafe { base.cast::<u64>().add(index % 2 * 8).cast() }
        };
        let meta = |offset: usize, index: usize, width: usize| -> Ptr {
            unsafe {
                refs.pre_meta
                    .cast::<u8>()
                    .add(offset + index % 2 * 8 * width)
                    .cast()
            }
        };
        let prefetched = prefetch && self.index >= 4;
        check(
            unsafe {
                g::hayate_glm_dev_route(
                    self.p("router_logits"),
                    self.p("route_weights"),
                    self.p("route_ids"),
                    self.p("decode.images"),
                    plan,
                    self.p("ffn_norm"),
                    self.w("mlp.gate.weight"),
                    self.w("mlp.gate.e_score_correction_bias"),
                    table,
                    mailbox,
                    refs.ctl,
                    refs.counter,
                    refs.skip_below,
                    defer.below,
                    if self.lowbit {
                        refs.formats
                            .cast::<u64>()
                            .add((self.index - 3) * EXPERTS)
                            .cast()
                    } else {
                        std::ptr::null()
                    },
                    if self.lowbit {
                        self.p("decode.formats")
                    } else {
                        std::ptr::null_mut()
                    },
                    if refs.host_table.is_null() {
                        std::ptr::null()
                    } else {
                        refs.host_table
                            .cast::<u64>()
                            .add((self.index - 3) * EXPERTS)
                            .cast()
                    },
                    refs.scratch,
                    refs.fetch,
                    refs.take,
                    if prefetched {
                        pre(self.index, refs.pre_src)
                    } else {
                        std::ptr::null_mut()
                    },
                    if prefetched {
                        pre(self.index, refs.pre_scratch)
                    } else {
                        std::ptr::null_mut()
                    },
                    if prefetched {
                        meta(PRE_TAGS, self.index, 8)
                    } else {
                        std::ptr::null_mut()
                    },
                    if prefetch {
                        meta(PRE_PROGRESS, 0, 0)
                    } else {
                        std::ptr::null_mut()
                    },
                    self.index as u32,
                    refs.pcie_stats,
                    st,
                )
            },
            "glm device router",
        )?;
        let fetch = |dst: Ptr, src: Ptr, bytes: usize, stream: RawStream| {
            // The PCIe lane: images from RAM into scratch slots (no work for null sources).
            check(
                unsafe {
                    hayate_kernels::glm53::exl3_ffi::hayate_glm_exl3_fetch(
                        dst, src, bytes, 188, stream,
                    )
                },
                "glm device PCIe fetch",
            )
        };
        if !refs.host_table.is_null() {
            // The router's own lane picks, on this stream.
            fetch(refs.scratch, refs.fetch, self.pcie_bytes, st)?;
        }
        if let (true, Some(next), Some(lane)) = (prefetch, next, lane) {
            // Pick layer k+1's likely RAM-tier misses from this layer's input and fork their fetch.
            let host = unsafe {
                refs.host_table
                    .cast::<u64>()
                    .add((next.index - 3) * EXPERTS)
                    .cast()
            };
            let next_table = unsafe {
                refs.table
                    .cast::<u64>()
                    .add((next.index - 3) * EXPERTS)
                    .cast()
            };
            check(
                unsafe {
                    g::hayate_glm_dev_predict(
                        self.p("prediction.logits"),
                        self.p("ffn_norm"),
                        next.w("mlp.gate.weight"),
                        next.w("mlp.gate.e_score_correction_bias"),
                        next_table,
                        host,
                        pre(next.index, refs.pre_src),
                        refs.prefetch,
                        refs.counter,
                        refs.pcie_stats,
                        st,
                    )
                },
                "glm device prefetch pick",
            )?;
            let (picked, copied) = &lane.events[next.index];
            picked.record(stream)?;
            lane.stream.wait_event(picked)?;
            check(
                unsafe {
                    g::hayate_glm_dev_prefetch(
                        pre(next.index, refs.pre_scratch),
                        pre(next.index, refs.pre_src),
                        next.pcie_bytes,
                        PREFETCH_BLOCKS,
                        meta(PRE_DONE, next.index, 4),
                        meta(PRE_BAD, next.index, 4),
                        meta(PRE_TAGS, next.index, 8),
                        meta(PRE_PROGRESS, 0, 0),
                        refs.ctl,
                        next.index as u32,
                        lane.stream.as_raw(),
                    )
                },
                "glm device prefetch copy",
            )?;
            copied.record(&lane.stream)?;
        }
        mark(2)?;
        if let Some((gate_k2, down_k2)) = self.exl3 {
            // EXL3 experts: the same product / expert buffers as the LOWBIT kernels.
            use hayate_kernels::glm53::exl3_ffi as x3;
            check(
                unsafe {
                    x3::hayate_glm_exl3_gate_up(
                        self.p("decode.product"),
                        self.p("ffn_norm"),
                        self.p("decode.images"),
                        self.dense_weight(&format!("{stem}.gate_proj.weight")),
                        self.dense_weight(&format!("{stem}.up_proj.weight")),
                        1,
                        u32::from(gate_k2),
                        HIDDEN as u32,
                        2048,
                        st,
                    )
                },
                "glm device EXL3 gate/up",
            )?;
            check(
                unsafe {
                    x3::hayate_glm_exl3_down(
                        self.p("decode.expert"),
                        self.p("decode.product"),
                        self.p("decode.images"),
                        self.dense_weight(&format!("{stem}.down_proj.weight")),
                        1,
                        u32::from(gate_k2),
                        u32::from(down_k2),
                        HIDDEN as u32,
                        2048,
                        st,
                    )
                },
                "glm device EXL3 down",
            )?;
        } else if self.lowbit {
            // Per-expert formats (NVFP4 and LOWBIT matrices mixed in one layer).
            check(
                unsafe {
                    lowbit::hayate_glm_lowbit_gate_up(
                        self.p("decode.product"),
                        self.p("ffn_norm"),
                        self.p("ffn_norm"),
                        self.p("decode.images"),
                        self.p("decode.formats").cast(),
                        self.dense_weight(&format!("{stem}.gate_proj.weight")),
                        self.dense_weight(&format!("{stem}.up_proj.weight")),
                        1,
                        HIDDEN as u32,
                        2048,
                        st,
                    )
                },
                "glm device LOWBIT gate/up",
            )?;
            check(
                unsafe {
                    lowbit::hayate_glm_lowbit_down(
                        self.p("decode.expert"),
                        self.p("decode.product"),
                        self.p("decode.images"),
                        self.p("decode.formats").cast(),
                        self.dense_weight(&format!("{stem}.down_proj.weight")),
                        1,
                        HIDDEN as u32,
                        2048,
                        st,
                    )
                },
                "glm device LOWBIT down",
            )?;
        } else {
            check(
                unsafe {
                    g::hayate_glm_moe_gate_up(
                        self.p("decode.product"),
                        self.p("ffn_norm"),
                        self.p("ffn_norm"),
                        self.p("decode.images"),
                        self.dense_weight(&format!("{stem}.gate_proj.weight")),
                        self.dense_weight(&format!("{stem}.up_proj.weight")),
                        nv as i32,
                        down_fp8 as i32,
                        matrix_stride(nv),
                        st,
                    )
                },
                "glm device gate/up",
            )?;
            check(
                unsafe {
                    g::hayate_glm_moe_down(
                        self.p("decode.expert"),
                        self.p("decode.product"),
                        self.p("decode.images"),
                        self.dense_weight(&format!("{stem}.down_proj.weight")),
                        nv as i32,
                        down_fp8 as i32,
                        matrix_stride(nv),
                        st,
                    )
                },
                "glm device down",
            )?;
        }
        self.lora_experts(
            st,
            self.p("decode.product"),
            self.p("decode.expert"),
            self.p("route_ids"),
            self.p("decode.images"),
        )?;
        mark(3)?;
        // count: plan word 0 (this layer, DONE) or 1 (deferred, DONE2).
        let wait = |plan: Ptr, word: usize, mailbox: Ptr| {
            let count = unsafe { plan.cast::<u32>().add(word).cast() };
            check(
                unsafe {
                    g::hayate_glm_dev_moe_wait(
                        count,
                        mailbox,
                        64 + 32 * word as u32,
                        refs.ctl,
                        refs.limit_ns,
                        st,
                    )
                },
                "glm device CPU wait",
            )
        };
        let mut previous = g::Deferred {
            mailbox: std::ptr::null(),
            ids: std::ptr::null(),
            weights: std::ptr::null(),
            images: std::ptr::null(),
            post: std::ptr::null(),
            plan: std::ptr::null(),
        };
        if let Some(prev) = defer.prev {
            wait(prev.plan, 1, prev.mailbox)?;
            previous = g::Deferred {
                mailbox: prev.mailbox,
                ids: prev.ids,
                weights: prev.weights,
                images: prev.images,
                post: prev.post,
                plan: prev.plan,
            };
        }
        wait(plan, 0, mailbox)?;
        mark(4)?;
        check(
            unsafe {
                g::hayate_glm_dev_moe_finish(
                    self.p("output"),
                    self.p("mlp"),
                    self.p("decode.expert"),
                    self.p("attention_residual"),
                    self.p("ffn_map"),
                    self.p("attn_map"),
                    self.p("route_ids"),
                    self.p("route_weights"),
                    self.p("decode.images"),
                    mailbox,
                    plan,
                    previous,
                    st,
                )
            },
            "glm device expert reduce",
        )?;
        Ok(self.p("output"))
    }
    /// This MoE layer's CPU-expert results as the next layer's `Defer::prev`.
    pub(in crate::glm53) fn deferred(&self, refs: &DeviceRefs, mailbox: Ptr) -> Prev {
        Prev {
            mailbox,
            ids: self.p("route_ids"),
            weights: self.p("route_weights"),
            images: self.p("decode.images"),
            post: self.p("ffn_map"),
            plan: unsafe { refs.plans.cast::<u32>().add(4 * self.index).cast() },
        }
    }
    /// Long-context bench only: repeat this layer's first `prefix` positions of
    /// latent KV and index pools `times` times and move to `prefix * times`.
    pub(in crate::glm53) fn tile_history(&mut self, prefix: usize, times: usize) -> Result<()> {
        self.stream.synchronize()?;
        if self.index % 4 == 3 {
            let st = self.stream.as_raw();
            for (name, row) in [("latent_cache", 1024), ("pools", 64)] {
                let base = self.p(name).cast::<u8>();
                for k in 1..times {
                    check(
                        unsafe {
                            g::hayate_glm_copy(
                                base.add(k * prefix * row).cast(),
                                base.cast(),
                                prefix * row,
                                st,
                            )
                        },
                        "glm bench tile history",
                    )?;
                }
            }
            self.stream.synchronize()?;
        }
        self.position = prefix * times;
        Ok(())
    }
    /// Host-side bookkeeping after a device token (positions live on the host).
    pub(in crate::glm53) fn device_advance(&mut self) {
        self.position += 1;
    }
}
