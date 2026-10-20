use super::*;
use crate::glm53::prefill::Prefill;
use std::time::Instant;

impl Layer<'_> {
    /// The caller orders the shared arena after the preceding layer's done event.
    /// State, conv history and index pools remain owned by this layer.
    pub(in crate::glm53) fn prefill_device(
        &mut self,
        rows: usize,
        cache: &mut super::super::experts::Experts,
        batch: &mut Prefill,
        ready: &Event,
    ) -> Result<()> {
        self.stream.wait_event(ready)?;
        if rows == 0 || rows > batch.capacity || self.position + rows > self.context {
            return Err(Error::InvalidShape("GLM batch prefill rows/context".into()));
        }
        let start = Instant::now();
        let mut saved = BTreeMap::new();
        for (name, buffer) in std::mem::take(&mut batch.layer) {
            saved.insert(name.clone(), self.buffers.insert(name, buffer));
        }
        self.tokens = rows;
        let before = cache.transfer_stats;
        let result = self.prefill_forward(cache, batch);
        let result = result
            .and_then(|()| self.copy(self.p("input"), self.p("output"), rows * HC * HIDDEN * 2));
        self.tokens = 1;
        // Restore even on failure, so a reset or drop sees the original arenas.
        for (name, old) in saved {
            batch
                .layer
                .insert(name.clone(), self.buffers.remove(&name).unwrap());
            if let Some(buffer) = old {
                self.buffers.insert(name, buffer);
            }
        }
        result?;
        self.position += rows;
        self.done.record(&self.stream)?;
        if let Some(p) = &self.profile {
            self.done.synchronize()?;
            eprintln!(
                "glm_prefill_layer position={} rows={rows} layer={} kind={} hc_attn_ms={:.6} attention_ms={:.6} hc_ffn_ms={:.6} routed_moe_ms={:.6} shared_dense_ms={:.6} expand_ms={:.6} scan_attention_ms={:.6} wall_ms={:.6} h2d_bytes={} cpu_ms={:.6} d2h_wait_ms={:.6}",
                self.position - rows,
                self.index,
                if self.index % 4 == 3 { "sparse" } else { "kda" },
                p.ms(0, 1)?,
                p.ms(1, 2)?,
                p.ms(2, 3)?,
                p.ms(3, 4)?,
                p.ms(4, 5)?,
                p.ms(5, 6)?,
                p.ms(8, 9)?,
                start.elapsed().as_secs_f64() * 1000.,
                cache.transfer_stats.h2d_bytes - before.h2d_bytes,
                cache.transfer_stats.cpu_ms - before.cpu_ms,
                p.waits.borrow().d2h_ms
            );
        }
        Ok(())
    }
    fn prefill_forward(
        &mut self,
        cache: &mut super::super::experts::Experts,
        batch: &mut Prefill,
    ) -> Result<()> {
        if let Some(p) = &self.profile {
            p.reset();
        }
        self.mark(0)?;
        self.hc_norm(
            "attn",
            "attn_map",
            "attn_collapsed",
            "attn_norm",
            "input",
            "input_layernorm.weight",
        )?;
        self.mark(1)?;
        if self.index + 1 >= 3
            && self.index + 1 < cache.layer_count()
            && batch.prediction_capacity > 0
        {
            let next = self.index + 1;
            let mut ids =
                super::super::prefill::predictions(batch.all_experts, &batch.previous[next]);
            // Only experts that are not on the GPU need a bank entry. Lookahead
            // uses only the VRAM that is free; a partial bank prefetches the first ids.
            ids.retain(|&e| !cache.on_gpu(next, e));
            ids.truncate(batch.prediction_capacity);
            cache.prefill_images(
                next,
                &ids,
                &mut batch.ahead.as_mut().unwrap()[next % 2],
                &batch.upload[next % 2],
                None,
            )?;
        }
        if self.index % 4 == 3 {
            self.prefill_sparse()?;
        } else {
            self.kda()?;
        }
        self.mark(2)?;
        self.expand("attention_residual", "input", "attention", "attn_map")?;
        self.hc_norm(
            "ffn",
            "ffn_map",
            "ffn_collapsed",
            "ffn_norm",
            "attention_residual",
            "post_attention_layernorm.weight",
        )?;
        self.mark(3)?;
        if self.index >= 3 {
            self.copy(
                batch.p("norm"),
                self.p("ffn_norm"),
                self.tokens * HIDDEN * 2,
            )?;
            self.prefill_experts(cache, batch, self.tokens)?;
            self.copy(self.p("mlp"), batch.p("mlp"), self.tokens * HIDDEN * 2)?;
        }
        self.mark(4)?;
        if self.index < 3 {
            self.mlp("mlp", "mlp", 12288)?;
        } else {
            self.mlp("mlp.shared_experts", "expert", 2048)?;
            check(
                unsafe {
                    g::hayate_glm_point(
                        self.p("mlp"),
                        self.p("mlp"),
                        self.p("expert"),
                        self.tokens * HIDDEN,
                        2,
                        self.stream.as_raw(),
                    )
                },
                "glm batch shared add",
            )?;
        }
        self.mark(5)?;
        self.expand("output", "attention_residual", "mlp", "ffn_map")?;
        self.mark(6)
    }
    /// Absorbed sparse attention for a prefill chunk: per head
    /// q_abs = q W_K^T (GEMM), tensor-core attention over the selected
    /// latents, then W_V. Uses the arena's `kv` for q_abs and `q` for the
    /// per-head outputs once q has been split.
    fn mla_prefill(&mut self, rows: usize) -> Result<()> {
        use hayate_kernels::text::ffi::hayate_text_linear;
        let st = self.stream.as_raw();
        let at =
            |p: Ptr, elements: usize| unsafe { p.cast::<u8>().add(elements * 2).cast::<c_void>() };
        check(
            unsafe { g::hayate_glm_mla_heads(self.p("kda_nq"), self.p("q"), rows as u32, 0, st) },
            "glm MLA split heads",
        )?;
        for h in 0..64 {
            check(
                unsafe {
                    hayate_text_linear(
                        at(self.p("kv"), h * rows * 512),
                        at(self.p("kda_nq"), h * rows * 256),
                        at(self.p("mla.wkt"), h * 512 * 256),
                        rows as u32,
                        512,
                        256,
                        st,
                    )
                },
                "glm MLA q_abs GEMM",
            )?;
        }
        check(
            unsafe {
                // In place: each block reads its q_abs rows before writing them.
                g::hayate_glm_mla_attention(
                    self.p("kv"),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    self.p("kv"),
                    self.p("latent_cache"),
                    self.p("selected"),
                    rows as u32,
                    1,
                    st,
                )
            },
            "glm MLA prefill attention",
        )?;
        for h in 0..64 {
            check(
                unsafe {
                    hayate_text_linear(
                        at(self.p("q"), h * rows * 256),
                        at(self.p("kv"), h * rows * 512),
                        at(self.p("mla.wv"), h * 256 * 512),
                        rows as u32,
                        256,
                        512,
                        st,
                    )
                },
                "glm MLA output GEMM",
            )?;
        }
        check(
            unsafe {
                g::hayate_glm_mla_heads(self.p("attn_core"), self.p("q"), rows as u32, 1, st)
            },
            "glm MLA merge heads",
        )
    }
    fn prefill_sparse(&mut self) -> Result<()> {
        let rows = self.tokens;
        let st = self.stream.as_raw();
        self.linear("qa", "attn_norm", "self_attn.q_a_proj.weight", 1536, HIDDEN)?;
        self.norm("qr", "qa", "self_attn.q_a_layernorm.weight", 1536)?;
        for (out, input, weight, n, k) in [
            ("q", "qr", "q_b_proj.weight", 16384, 1536),
            ("kva", "attn_norm", "kv_a_proj_with_mqa.weight", 512, HIDDEN),
            ("iq", "qr", "indexer.wq_b.weight", 4096, 1536),
            ("ik", "attn_norm", "indexer.wk.weight", 128, HIDDEN),
            (
                "ig",
                "attn_norm",
                "indexer.index_kpool_compress_gate",
                128,
                HIDDEN,
            ),
            ("iw", "attn_norm", "indexer.weights_proj.weight", 32, HIDDEN),
        ] {
            self.linear(out, input, &format!("self_attn.{weight}"), n, k)?;
        }
        self.norm("kvr", "kva", "self_attn.kv_a_layernorm.weight", 512)?;
        check(
            unsafe {
                g::hayate_glm_norm(
                    self.p("ikn"),
                    self.p("ik"),
                    self.w("self_attn.indexer.k_norm.weight"),
                    self.w("self_attn.indexer.k_norm.bias"),
                    rows as u32,
                    128,
                    1,
                    st,
                )
            },
            "glm batch index norm",
        )?;
        check(
            unsafe {
                g::hayate_glm_pool_chunk(
                    self.p("pools"),
                    self.p("pool_keys"),
                    self.p("pool_gates"),
                    self.p("ikn"),
                    self.p("ig"),
                    self.w("self_attn.indexer.index_kpool_compress_ape"),
                    self.position as u32,
                    rows as u32,
                    st,
                )
            },
            "glm batch pool",
        )?;
        if let Some(latent) = self.latent.as_mut() {
            let host = unsafe {
                latent
                    .as_mut_slice()
                    .as_mut_ptr()
                    .add(self.position * 1024)
                    .cast()
            };
            check(
                unsafe { g::hayate_glm_prefill_latent(host, self.p("kvr"), rows * 1024, st) },
                "glm batch latent append",
            )?;
        }
        self.latent_append(self.p("kvr"), rows)?;
        check(
            unsafe {
                g::hayate_glm_mla_select(
                    self.p("selected"),
                    self.p("scores"),
                    self.p("iq"),
                    self.p("pools"),
                    self.p("iw"),
                    self.position as u32,
                    rows as u32,
                    self.context.div_ceil(4),
                    st,
                )
            },
            "glm batch select",
        )?;
        self.mark(8)?;
        self.mla_prefill(rows)?;
        self.mark(9)?;
        self.linear(
            "attention",
            "attn_core",
            "self_attn.o_proj.weight",
            HIDDEN,
            16384,
        )
    }
}
