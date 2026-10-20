//! Small-M verification: decode arithmetic, one weight load for all token rows.
use super::super::speculation::{SPARSE_WIDTH, sparse_union};
use super::*;

#[derive(Debug)]
pub(super) struct Verification {
    buffers: BTreeMap<String, DeviceBuffer>,
    route_host: PinnedHost,
    cpu_input: PinnedHost,
    cpu_output: PinnedHost,
    sparse_upload: Option<(PinnedHost, PinnedHost)>,
    sparse_counts: (usize, usize),
    pub(super) row: Option<usize>,
    start: usize,
    rows: usize,
}
impl Verification {
    pub(in crate::glm53) fn row_ptr(&self, name: &str, row: usize, bytes: usize) -> Ptr {
        unsafe {
            self.buffers[name]
                .as_ptr()
                .cast::<u8>()
                .add(row * bytes)
                .cast_mut()
                .cast()
        }
    }
    fn p(&self, name: &str) -> Ptr {
        self.row_ptr(name, 0, 0)
    }
    pub(in crate::glm53) fn bytes(&self) -> usize {
        self.buffers.values().map(DeviceBuffer::bytes).sum()
    }
}
impl Layer<'_> {
    pub(in crate::glm53) fn prepare_verification(&mut self) -> Result<()> {
        if self.verification.is_some() {
            return Ok(());
        }
        if self.index % 4 == 3 || self.index == 45 {
            if self.latent.is_none() {
                self.latent = Some(self.device.alloc_pinned(self.context * 512 * 2)?);
            }
            // MTP verification keeps the projected-KV sparse path (3 GB over
            // the sparse layers); plain decode uses the absorbed MLA kernels.
            for (name, bytes) in [
                ("latent_selected", 2051 * 512 * 2),
                ("sparse.sorted", 4 * 4096 * 4),
                ("sparse.ids", 4 * 2051 * 4),
                ("sparse.maps", 4 * 2051 * 4),
                ("sparse.counts", 5 * 4),
                ("kv", 2051 * 32768 * 2),
                ("k", 2051 * 16384 * 2),
                ("v", 2051 * 16384 * 2),
                ("probabilities", 64 * 2051 * 2),
            ] {
                self.buffers.insert(name.into(), self.device.alloc(bytes)?);
            }
        }
        let mut buffers = BTreeMap::new();
        for (name, bytes) in [
            ("input", HC * HIDDEN * 2),
            ("output", HC * HIDDEN * 2),
            ("attention_residual", HC * HIDDEN * 2),
            ("attn_norm", HIDDEN * 2),
            ("ffn_norm", HIDDEN * 2),
            ("ffn_map", 24 * 4),
            ("attention", HIDDEN * 2),
            ("mlp", HIDDEN * 2),
            ("rotated_input", HIDDEN * 2),
            ("route_ids", TOP_K * 4),
            ("route_weights", TOP_K * 4),
            ("router_logits", EXPERTS * 4),
            ("expert_input", HIDDEN * 2),
            ("expert_gate", 12288 * 2),
            ("expert_up", 12288 * 2),
            ("expert_product", 12288 * 2),
            ("expert_output", HIDDEN * 2),
            ("expert_all", 9 * HIDDEN * 2),
            ("async.groups", 8 * 32),
            ("async.count", 4),
            ("async.input", 8 * 4 * HIDDEN * 2),
            ("async.product", 8 * 4 * 2048 * 2),
        ] {
            buffers.insert(name.into(), self.device.alloc(4 * bytes)?);
        }
        let names: &[&str] = if self.index % 4 == 3 || self.index == 45 {
            &[
                "qa",
                "qr",
                "q",
                "kva",
                "kvr",
                "iq",
                "ik",
                "ikn",
                "ig",
                "iw",
                "attn_core",
            ]
        } else {
            &["q", "k", "v", "fa", "fb", "ga", "gb", "beta", "gated"]
        };
        for &name in names {
            buffers.insert(
                name.into(),
                self.device.alloc(4 * self.buffers[name].bytes())?,
            );
        }
        let states: &[&str] = if self.index % 4 == 3 || self.index == 45 {
            &["pool_keys", "pool_gates"]
        } else {
            &["state", "hq", "hk", "hv"]
        };
        for &name in states {
            buffers.insert(
                format!("snapshot.{name}"),
                self.device.alloc(5 * self.buffers[name].bytes())?,
            );
        }
        // The MTP draft layer only uses snapshots. Main-model verification
        // projects the union once, then reuses the ordinary per-query K/V arenas.
        let sparse_upload = if self.index % 4 == 3 {
            let capacity = self.context.clamp(SPARSE_WIDTH, 4 * SPARSE_WIDTH);
            for (name, bytes) in [("kv", 32768 * 2), ("latent_selected", 512 * 2)] {
                if capacity > SPARSE_WIDTH {
                    self.buffers
                        .insert(name.into(), self.device.alloc(capacity * bytes)?);
                }
            }
            for name in ["selected", "selected_indices"] {
                buffers.insert(name.into(), self.device.alloc(4 * SPARSE_WIDTH * 4)?);
            }
            Some((
                self.device.alloc_pinned(capacity * 512 * 2)?,
                self.device.alloc_pinned(4 * SPARSE_WIDTH * 4)?,
            ))
        } else {
            None
        };
        self.verification = Some(Verification {
            buffers,
            route_host: self.device.alloc_pinned(4 * TOP_K * 4)?,
            cpu_input: self.device.alloc_pinned(4 * HIDDEN * 2)?,
            cpu_output: self.device.alloc_pinned(4 * 9 * HIDDEN * 2)?,
            sparse_upload,
            sparse_counts: (0, 0),
            row: None,
            start: 0,
            rows: 0,
        });
        Ok(())
    }
    fn v(&self, name: &str) -> Ptr {
        self.verification.as_ref().unwrap().p(name)
    }
    fn vr(&self, name: &str, row: usize, bytes: usize) -> Ptr {
        self.verification
            .as_ref()
            .unwrap()
            .row_ptr(name, row, bytes)
    }
    pub(in crate::glm53) fn copy(&self, dst: Ptr, src: Ptr, bytes: usize) -> Result<()> {
        check(
            unsafe { g::hayate_glm_copy(dst, src, bytes, self.stream.as_raw()) },
            "glm verification D2D",
        )
    }
    fn snapshot(&self, slot: usize, restore: bool) -> Result<()> {
        let names: &[&str] = if self.index % 4 == 3 || self.index == 45 {
            &["pool_keys", "pool_gates"]
        } else {
            &["state", "hq", "hk", "hv"]
        };
        for &name in names {
            let bytes = self.buffers[name].bytes();
            let saved = self.vr(&format!("snapshot.{name}"), slot, bytes);
            if restore {
                self.copy(self.p(name), saved, bytes)?;
            } else {
                self.copy(saved, self.p(name), bytes)?;
            }
        }
        Ok(())
    }
    pub(in crate::glm53) fn save_spec_start(&mut self) -> Result<()> {
        self.prepare_verification()?;
        let v = self.verification.as_mut().unwrap();
        v.start = self.position;
        v.rows = 0;
        self.snapshot(0, false)
    }
    pub(in crate::glm53) fn rollback(&mut self, keep: usize) -> Result<()> {
        let v = self
            .verification
            .as_ref()
            .ok_or_else(|| Error::InvalidPlan("GLM missing snapshots".into()))?;
        if keep > v.rows {
            return Err(Error::InvalidShape("GLM snapshot prefix".into()));
        }
        self.position = v.start + keep;
        // Latent KV and completed pools beyond position remain inaccessible and
        // are overwritten on append. The partial pool needs its own snapshot.
        self.snapshot(keep, true)?;
        self.done.record(&self.stream)
    }
    pub(in crate::glm53) fn rollback_device(&self, result: Ptr, ready: &Event) -> Result<()> {
        self.stream.wait_event(ready)?;
        let names: &[&str] = if self.index % 4 == 3 {
            &["pool_keys", "pool_gates"]
        } else {
            &["state", "hq", "hk", "hv"]
        };
        for &name in names {
            check(
                unsafe {
                    g::hayate_glm_restore_prefix(
                        self.p(name),
                        self.v(&format!("snapshot.{name}")),
                        self.buffers[name].bytes(),
                        result,
                        1,
                        self.stream.as_raw(),
                    )
                },
                "glm GPU prefix restore",
            )?;
        }
        self.done.record(&self.stream)
    }
    pub(in crate::glm53) fn commit_spec_position(&mut self, keep: usize) {
        self.position = self.verification.as_ref().unwrap().start + keep;
    }
    pub(in crate::glm53) fn mtp_repair_start(&mut self, ready: &Event) -> Result<()> {
        self.stream.wait_event(ready)?;
        self.rollback(0)
    }
    pub(in crate::glm53) fn mtp_cache_append(&mut self, result: Ptr, row: usize) -> Result<()> {
        if self.index != 45 || self.position >= self.context {
            return Err(Error::InvalidShape("GLM MTP cache append".into()));
        }
        self.norm("attn_norm", "input", "input_layernorm.weight", HIDDEN)?;
        self.sparse_cache(Some((result, row)))?;
        self.position += 1;
        self.done.record(&self.stream)
    }
    #[cfg(test)]
    pub(in crate::glm53) fn mtp_cache_bytes(&self) -> Result<Vec<u8>> {
        self.stream.synchronize()?;
        let mut raw = self.latent.as_ref().unwrap().as_slice()[..self.position * 1024].to_vec();
        for (name, bytes) in [
            ("pool_keys", 1024),
            ("pool_gates", 1024),
            ("pools", self.position / 4 * 256),
        ] {
            let start = raw.len();
            raw.resize(start + bytes, 0);
            if bytes != 0 {
                self.stream
                    .copy_d2h(&mut raw[start..], &self.buffers[name])?;
            }
        }
        Ok(raw)
    }
    fn project(
        &self,
        out: &str,
        input: &str,
        weight: &str,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.project_ptrs(
            self.v(out),
            self.v(input),
            self.dense_weight(weight),
            m,
            n,
            k,
            true,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn project_ptrs(
        &self,
        out: Ptr,
        input: Ptr,
        weight: g::DenseWeight,
        m: usize,
        n: usize,
        k: usize,
        rounded: bool,
    ) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_dense_batch(
                    out,
                    input,
                    weight,
                    m as u32,
                    n as u32,
                    k as u32,
                    rounded as i32,
                    self.stream.as_raw(),
                )
            },
            "glm W4A16 verification projection",
        )
    }
    fn spec_norm(&self, out: &str, input: &str, weight: &str, m: usize, k: usize) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_norm(
                    self.v(out),
                    self.v(input),
                    self.w(weight),
                    std::ptr::null(),
                    m as u32,
                    k as u32,
                    0,
                    self.stream.as_raw(),
                )
            },
            "glm verification norm",
        )
    }
    pub(in crate::glm53) fn verify_device(
        &mut self,
        input: Ptr,
        ready: &Event,
        m: usize,
        cache: &mut super::super::experts::Experts,
        profile: bool,
    ) -> Result<Ptr> {
        if !(1..=4).contains(&m)
            || self.position + m > self.context
            || self.reference
            || self.index == 45
        {
            return Err(Error::InvalidShape(
                "GLM verification context/rows/reference".into(),
            ));
        }
        self.prepare_verification()?;
        if profile && self.profile.is_none() {
            self.enable_profile()?;
        }
        if let Some(p) = &self.profile {
            p.reset();
        }
        self.stream.wait_event(ready)?;
        self.mark(0)?;
        self.save_spec_start()?;
        self.copy(self.v("input"), input, m * HC * HIDDEN * 2)?;
        self.verification.as_mut().unwrap().sparse_counts = (0, 0);
        for row in 0..m {
            self.copy(
                self.p("input"),
                self.vr("input", row, HC * HIDDEN * 2),
                HC * HIDDEN * 2,
            )?;
            self.hc_norm(
                "attn",
                "attn_map",
                "attn_collapsed",
                "attn_norm",
                "input",
                "input_layernorm.weight",
            )?;
            self.copy(
                self.vr("attn_norm", row, HIDDEN * 2),
                self.p("attn_norm"),
                HIDDEN * 2,
            )?;
            self.copy(self.vr("ffn_map", row, 24 * 4), self.p("attn_map"), 24 * 4)?;
        }
        self.mark(1)?;
        if self.index % 4 == 3 || self.index == 45 {
            self.project(
                "qa",
                "attn_norm",
                "self_attn.q_a_proj.weight",
                m,
                1536,
                HIDDEN,
            )?;
            self.spec_norm("qr", "qa", "self_attn.q_a_layernorm.weight", m, 1536)?;
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
                self.project(out, input, &format!("self_attn.{weight}"), m, n, k)?;
            }
        } else {
            for part in ["q", "k", "v"] {
                self.project(
                    part,
                    "attn_norm",
                    &format!("self_attn.{part}_proj.weight"),
                    m,
                    8192,
                    HIDDEN,
                )?;
            }
            for part in ["f", "g"] {
                self.project(
                    &format!("{part}a"),
                    "attn_norm",
                    &format!("self_attn.{part}_a_proj.weight"),
                    m,
                    128,
                    HIDDEN,
                )?;
                self.project(
                    &format!("{part}b"),
                    &format!("{part}a"),
                    &format!("self_attn.{part}_b_proj.weight"),
                    m,
                    8192,
                    128,
                )?;
            }
            self.project(
                "beta",
                "attn_norm",
                "self_attn.b_proj.weight",
                m,
                64,
                HIDDEN,
            )?;
        }
        self.mark(2)?;
        for row in 0..m {
            self.copy(
                self.p("attn_norm"),
                self.vr("attn_norm", row, HIDDEN * 2),
                HIDDEN * 2,
            )?;
            self.verification.as_mut().unwrap().row = Some(row);
            let result = if self.index % 4 == 3 || self.index == 45 {
                self.sparse_select()
            } else {
                self.kda()
            };
            self.verification.as_mut().unwrap().row = None;
            result?;
            if self.index % 4 == 3 || self.index == 45 {
                self.copy(
                    self.vr("selected", row, SPARSE_WIDTH * 4),
                    self.p("selected"),
                    SPARSE_WIDTH * 4,
                )?;
            }
            self.position += 1;
            self.snapshot(row + 1, false)?;
        }
        self.verification.as_mut().unwrap().rows = m;
        if self.index % 4 == 3 || self.index == 45 {
            self.verify_sparse_attention(m)?;
        }
        self.mark(3)?;
        let (core, width) = if self.index % 4 == 3 || self.index == 45 {
            ("attn_core", 16384)
        } else {
            ("gated", 8192)
        };
        self.project(
            "attention",
            core,
            "self_attn.o_proj.weight",
            m,
            HIDDEN,
            width,
        )?;
        for row in 0..m {
            for (name, bytes) in [("input", HC * HIDDEN * 2), ("attention", HIDDEN * 2)] {
                self.copy(self.p(name), self.vr(name, row, bytes), bytes)?;
            }
            self.copy(self.p("attn_map"), self.vr("ffn_map", row, 24 * 4), 24 * 4)?;
            self.expand("attention_residual", "input", "attention", "attn_map")?;
            self.hc_norm(
                "ffn",
                "ffn_map",
                "ffn_collapsed",
                "ffn_norm",
                "attention_residual",
                "post_attention_layernorm.weight",
            )?;
            for (name, bytes) in [
                ("attention_residual", HC * HIDDEN * 2),
                ("ffn_norm", HIDDEN * 2),
                ("ffn_map", 24 * 4),
            ] {
                self.copy(self.vr(name, row, bytes), self.p(name), bytes)?;
            }
        }
        self.mark(4)?;
        if self.index < 3 {
            self.spec_mlp(m, "mlp", None, false, false)?;
            self.copy(self.v("mlp"), self.v("expert_output"), m * HIDDEN * 2)?;
        } else {
            self.spec_experts(m, cache)?;
        }
        self.mark(5)?;
        for row in 0..m {
            for (name, bytes) in [
                ("attention_residual", HC * HIDDEN * 2),
                ("ffn_map", 24 * 4),
                ("mlp", HIDDEN * 2),
            ] {
                self.copy(self.p(name), self.vr(name, row, bytes), bytes)?;
            }
            self.expand("output", "attention_residual", "mlp", "ffn_map")?;
            self.copy(
                self.vr("output", row, HC * HIDDEN * 2),
                self.p("output"),
                HC * HIDDEN * 2,
            )?;
        }
        self.mark(6)?;
        self.done.record(&self.stream)?;
        Ok(self.v("output"))
    }
    fn verify_sparse_attention(&mut self, m: usize) -> Result<()> {
        if self.async_active {
            return self.verify_sparse_async(m);
        }
        let mut selected = vec![0; m * SPARSE_WIDTH * 4];
        let v = self.verification.as_ref().unwrap();
        if let Some(p) = &self.profile {
            p.download(&self.stream, &mut selected, &v.buffers["selected"])?;
        } else {
            self.stream
                .copy_d2h(&mut selected, &v.buffers["selected"])?;
        }
        // This single wait also retires every appended latent and all snapshots.
        let (union, indices) = sparse_union(&selected, v.start, m)?;
        let v = self.verification.as_mut().unwrap();
        v.sparse_counts = (indices.iter().map(Vec::len).sum(), union.len());
        let (latent, maps) = v.sparse_upload.as_mut().unwrap();
        for (row, &id) in union.iter().enumerate() {
            latent.as_mut_slice()[row * 1024..(row + 1) * 1024].copy_from_slice(
                &self.latent.as_ref().unwrap().as_slice()[id * 1024..(id + 1) * 1024],
            );
        }
        for (row, query) in indices.iter().enumerate() {
            for (dst, &index) in maps.as_mut_slice()[row * SPARSE_WIDTH * 4..]
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(query)
            {
                *dst = index.to_le_bytes();
            }
        }
        // Both pinned sources remain owned by the layer until its next verifier
        // wait/reset/drop. No source is rewritten while its H2D is in flight.
        unsafe {
            self.stream.copy_h2d_at_async(
                self.buffers.get_mut("latent_selected").unwrap(),
                0,
                latent.range(0..union.len() * 1024)?,
            )?;
            self.stream
                .copy_h2d_async(v.buffers.get_mut("selected_indices").unwrap(), maps)?;
        }
        self.linear_ptrs(
            self.p("kv"),
            self.p("latent_selected"),
            "self_attn.kv_b_proj.weight",
            union.len(),
            32768,
            512,
        )?;
        for (row, query) in indices.iter().enumerate() {
            check(
                unsafe {
                    g::hayate_glm_split_selected(
                        self.p("k"),
                        self.p("v"),
                        self.p("kv"),
                        self.vr("selected_indices", row, SPARSE_WIDTH * 4),
                        query.len() as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm verification KV union gather",
            )?;
            check(
                unsafe {
                    g::hayate_glm_attention_step(
                        self.vr("attn_core", row, 16384 * 2),
                        self.p("probabilities"),
                        self.vr("q", row, 16384 * 2),
                        self.p("k"),
                        self.p("v"),
                        query.len() as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm verification chronological attention",
            )?;
        }
        Ok(())
    }
    fn verify_sparse_async(&self, m: usize) -> Result<()> {
        let start = self.verification.as_ref().unwrap().start;
        let bound = (0..m)
            .map(|r| Self::sparse_count(start + r + 1))
            .sum::<usize>()
            .min(start + m);
        self.sparse_gather_async(self.v("selected"), m, bound)?;
        let name = "self_attn.kv_b_proj.weight";
        let matrix = if self.dense_formats.contains_key(name) || self.fp8.contains(name) {
            check(
                unsafe {
                    g::hayate_glm_dense_unpack(
                        self.p("dense.scratch"),
                        self.dense_weight(name),
                        32768,
                        512,
                        self.stream.as_raw(),
                    )
                },
                "glm async KV restore",
            )?;
            self.p("dense.scratch")
        } else {
            self.w(name)
        };
        check(
            unsafe {
                g::hayate_glm_sparse_project(
                    self.p("kv"),
                    self.p("latent_selected"),
                    matrix,
                    self.p("sparse.counts"),
                    bound as u32,
                    self.stream.as_raw(),
                )
            },
            "glm device-count KV projection",
        )?;
        for row in 0..m {
            let count = Self::sparse_count(start + row + 1);
            let maps = unsafe {
                self.p("sparse.maps")
                    .cast::<u8>()
                    .add(row * SPARSE_WIDTH * 4)
                    .cast()
            };
            check(
                unsafe {
                    g::hayate_glm_split_selected(
                        self.p("k"),
                        self.p("v"),
                        self.p("kv"),
                        maps,
                        count as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm async verifier KV split",
            )?;
            check(
                unsafe {
                    g::hayate_glm_attention_step(
                        self.vr("attn_core", row, 16384 * 2),
                        self.p("probabilities"),
                        self.vr("q", row, 16384 * 2),
                        self.p("k"),
                        self.p("v"),
                        count as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm async verifier attention",
            )?;
        }
        Ok(())
    }
    fn spec_experts_async(
        &mut self,
        m: usize,
        cache: &super::super::experts::Experts,
        input: &str,
    ) -> Result<()> {
        use super::super::async_decode::{IDS, INPUT, OUTPUT};
        let format = cache.format(self.index);
        let sequence = self.asynchronous.as_mut().unwrap().submit(
            cache.asynchronous.as_ref().unwrap(),
            self.index,
            m,
            format,
            self.hadamard_block,
        )?;
        let a = self.asynchronous.as_ref().unwrap();
        self.publish_host(a.mailbox.device(IDS), self.v("route_ids"), m * TOP_K * 4)?;
        self.publish_host(a.mailbox.device(INPUT), self.v(input), m * HIDDEN * 2)?;
        a.signal(&self.stream, sequence)?;
        check(
            unsafe {
                g::hayate_glm_async_groups(
                    self.v("async.groups"),
                    self.v("async.count"),
                    self.v("async.input"),
                    self.v(input),
                    self.v("route_ids"),
                    a.table.device_ptr(),
                    m as u32,
                    self.stream.as_raw(),
                )
            },
            "glm async verification groups",
        )?;
        self.spec_mlp(m, "mlp.shared_experts", None, false, false)?;
        for row in 0..m {
            self.copy(
                self.vr("expert_all", row * 9 + 8, HIDDEN * 2),
                self.vr("expert_output", row, HIDDEN * 2),
                HIDDEN * 2,
            )?;
        }
        check(
            unsafe {
                g::hayate_glm_async_gate(
                    self.v("async.product"),
                    self.v("async.input"),
                    self.v("async.groups"),
                    self.v("async.count"),
                    (m * TOP_K) as u32,
                    format.0 as i32,
                    format.1 as i32,
                    matrix_stride(format.0),
                    self.stream.as_raw(),
                )
            },
            "glm async verification gate/up",
        )?;
        if self.hadamard_block != 0 && format.0 && !format.1 {
            self.rotate(
                self.v("async.product"),
                self.v("async.product"),
                m * TOP_K * 4,
                2048,
            )?;
        }
        check(
            unsafe {
                g::hayate_glm_async_down(
                    self.v("expert_all"),
                    self.v("async.product"),
                    self.v("async.groups"),
                    self.v("async.count"),
                    (m * TOP_K) as u32,
                    format.0 as i32,
                    format.1 as i32,
                    matrix_stride(format.0),
                    self.stream.as_raw(),
                )
            },
            "glm async verification down",
        )?;
        self.mark(7)?;
        a.wait(&self.stream, sequence)?;
        self.mark(8)?;
        check(
            unsafe {
                g::hayate_glm_async_merge(
                    self.v("expert_all"),
                    a.mailbox.device(OUTPUT),
                    self.v("route_ids"),
                    a.table.device_ptr(),
                    m as u32,
                    self.stream.as_raw(),
                )
            },
            "glm async verification CPU merge",
        )?;
        self.mark(9)?;
        for row in 0..m {
            check(
                unsafe {
                    g::hayate_glm_moe_reduce(
                        self.vr("mlp", row, HIDDEN * 2),
                        self.vr("expert_all", row * 9, HIDDEN * 2),
                        self.vr("route_ids", row, TOP_K * 4),
                        self.vr("route_weights", row, TOP_K * 4),
                        self.stream.as_raw(),
                    )
                },
                "glm async verification reduction",
            )?;
        }
        Ok(())
    }
    fn spec_mlp(
        &self,
        m: usize,
        stem: &str,
        image: Option<Ptr>,
        nv: bool,
        down_fp8: bool,
    ) -> Result<()> {
        let dim = if self.index < 3 { 12288 } else { 2048 };
        for (part, target) in [(0, "gate"), (1, "up")] {
            let weight = if let Some(image) = image {
                self.routed_weight(image, part, nv, down_fp8)
            } else {
                self.dense_weight(&format!("{stem}.{target}_proj.weight"))
            };
            self.project_ptrs(
                self.v(&format!("expert_{target}")),
                self.v(if image.is_some() {
                    "expert_input"
                } else {
                    "ffn_norm"
                }),
                weight,
                m,
                dim,
                HIDDEN,
                image.is_none() || !nv,
            )?;
        }
        check(
            unsafe {
                g::hayate_glm_point(
                    self.v("expert_product"),
                    self.v("expert_gate"),
                    self.v("expert_up"),
                    m * dim,
                    1,
                    self.stream.as_raw(),
                )
            },
            "glm verification SwiGLU",
        )?;
        if image.is_some() && self.hadamard_block != 0 && nv && !down_fp8 {
            self.rotate(self.v("expert_product"), self.v("expert_product"), m, dim)?;
        }
        let weight = image.map_or_else(
            || self.dense_weight(&format!("{stem}.down_proj.weight")),
            |p| self.routed_weight(p, 2, nv, down_fp8),
        );
        self.project_ptrs(
            self.v("expert_output"),
            self.v("expert_product"),
            weight,
            m,
            HIDDEN,
            dim,
            image.is_none() || !nv || down_fp8,
        )
    }
    fn routed_weight(&self, image: Ptr, part: usize, nv: bool, down_fp8: bool) -> g::DenseWeight {
        let data = unsafe { image.cast::<u8>().add(part * matrix_stride(nv)).cast() };
        let packed = nv && !(part == 2 && down_fp8);
        g::DenseWeight {
            data,
            scale: if packed {
                std::ptr::null()
            } else {
                unsafe { data.cast::<u8>().add(4096 * 2048).cast() }
            },
            format: if packed { 3 } else { 1 },
        }
    }
    pub(in crate::glm53) fn report_spec_profile(&self, position: usize) -> Result<()> {
        if let Some(p) = &self.profile {
            let waits = p.waits.borrow();
            let sparse_counts = if self.async_active && self.index % 4 == 3 {
                // The executor reports only after the result read retires all work.
                unsafe {
                    self.asynchronous
                        .as_ref()
                        .unwrap()
                        .mailbox
                        .sparse_counts(self.verification.as_ref().unwrap().rows)
                }
            } else {
                self.verification.as_ref().unwrap().sparse_counts
            };
            eprintln!(
                "glm_spec_layer position={position} layer={} hc_attn_ms={:.6} projections_ms={:.6} recurrent_sparse_ms={:.6} output_hc_ms={:.6} moe_ms={:.6} expand_ms={:.6} cpu_tail_wait_ms={:.6} cpu_merge_ms={:.6} d2h_syncs={} d2h_wait_ms={:.6} kv_selected_rows={} kv_union_rows={}",
                self.index,
                p.ms(0, 1)?,
                p.ms(1, 2)?,
                p.ms(2, 3)?,
                p.ms(3, 4)?,
                p.ms(4, 5)?,
                p.ms(5, 6)?,
                if self.async_active && self.index >= 3 {
                    p.ms(7, 8)?
                } else {
                    0.
                },
                if self.async_active && self.index >= 3 {
                    p.ms(8, 9)?
                } else {
                    0.
                },
                waits.d2h_sync_count,
                waits.d2h_ms,
                sparse_counts.0,
                sparse_counts.1,
            );
        }
        Ok(())
    }
    fn spec_experts(&mut self, m: usize, cache: &mut super::super::experts::Experts) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_route_batch(
                    self.v("router_logits"),
                    self.v("route_weights"),
                    self.v("route_ids"),
                    self.v("ffn_norm"),
                    self.w("mlp.gate.weight"),
                    self.w("mlp.gate.e_score_correction_bias"),
                    m as u32,
                    self.stream.as_raw(),
                )
            },
            "glm verification router",
        )?;
        if self.hadamard_block != 0 {
            self.rotate(self.v("rotated_input"), self.v("ffn_norm"), m, HIDDEN)?;
        }
        let name = if self.hadamard_block != 0 {
            "rotated_input"
        } else {
            "ffn_norm"
        };
        if self.async_active {
            return self.spec_experts_async(m, cache, name);
        }
        let wait = std::time::Instant::now();
        let v = self.verification.as_mut().unwrap();
        unsafe {
            self.stream
                .copy_d2h_async(&mut v.route_host, &v.buffers["route_ids"])?;
        }
        if cache.cpu_enabled() {
            unsafe {
                self.stream
                    .copy_d2h_async(&mut v.cpu_input, &v.buffers[name])?;
            }
        }
        self.stream.synchronize()?;
        if let Some(p) = &self.profile {
            p.add_wait(wait);
            if cache.cpu_enabled() {
                p.enqueue_d2h();
            }
        }
        let ids: Vec<u32> = v.route_host.as_slice()[..m * TOP_K * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let groups = super::super::prefill::groups(&ids)?;
        for row in ids.as_chunks::<TOP_K>().0 {
            cache.route(
                self.index,
                &row.iter().map(|&e| e as u16).collect::<Vec<_>>(),
            )?;
        }
        let union: Vec<_> = groups
            .iter()
            .enumerate()
            .filter_map(|(e, rows)| (!rows.is_empty()).then_some(e as u16))
            .collect();
        let cpu = cache.cpu_images(self.index, &union);
        let (nv, down_fp8) = cache.format(self.index);
        // Shared is computed once for all rows before CPU misses are submitted.
        self.spec_mlp(m, "mlp.shared_experts", None, false, false)?;
        for row in 0..m {
            self.copy(
                self.vr("expert_all", row * 9 + 8, HIDDEN * 2),
                self.vr("expert_output", row, HIDDEN * 2),
                HIDDEN * 2,
            )?;
        }
        // Submit every GPU group first, then compute the host union concurrently.
        for (&expert, cpu) in union.iter().zip(&cpu) {
            if cpu.is_some() {
                continue;
            }
            let assignments = &groups[expert as usize];
            let image = cache
                .image(self.index, expert as usize, &self.stream)?
                .cast_mut();
            for (i, &a) in assignments.iter().enumerate() {
                let src = self.vr(name, a as usize / TOP_K, HIDDEN * 2);
                self.copy(self.vr("expert_input", i, HIDDEN * 2), src, HIDDEN * 2)?;
            }
            self.spec_mlp(assignments.len(), "", Some(image), nv, down_fp8)?;
            for (i, &a) in assignments.iter().enumerate() {
                let row = a as usize / TOP_K;
                let slot = a as usize % TOP_K;
                self.copy(
                    self.vr("expert_all", row * 9 + slot, HIDDEN * 2),
                    self.vr("expert_output", i, HIDDEN * 2),
                    HIDDEN * 2,
                )?;
            }
        }
        for (&expert, cpu) in union.iter().zip(cpu) {
            let Some(image) = cpu else {
                continue;
            };
            let assignments = &groups[expert as usize];
            let v = self.verification.as_mut().unwrap();
            let mut gathered = Vec::with_capacity(assignments.len() * HIDDEN * 2);
            for &a in assignments {
                let row = a as usize / TOP_K;
                gathered.extend_from_slice(
                    &v.cpu_input.as_slice()[row * HIDDEN * 2..(row + 1) * HIDDEN * 2],
                );
            }
            let mut output = vec![0; gathered.len()];
            cache.compute_cpu_group(
                self.index,
                image.as_slice(),
                &gathered,
                &mut output,
                self.hadamard_block,
            )?;
            for (i, &a) in assignments.iter().enumerate() {
                let row = a as usize / TOP_K;
                let slot = a as usize % TOP_K;
                let offset = (row * 9 + slot) * HIDDEN * 2;
                v.cpu_output.as_mut_slice()[offset..offset + HIDDEN * 2]
                    .copy_from_slice(&output[i * HIDDEN * 2..(i + 1) * HIDDEN * 2]);
                unsafe {
                    self.stream.copy_h2d_at_async(
                        v.buffers.get_mut("expert_all").unwrap(),
                        offset,
                        v.cpu_output.range(offset..offset + HIDDEN * 2)?,
                    )?;
                }
            }
        }
        for row in 0..m {
            check(
                unsafe {
                    g::hayate_glm_moe_reduce(
                        self.vr("mlp", row, HIDDEN * 2),
                        self.vr("expert_all", row * 9, HIDDEN * 2),
                        self.vr("route_ids", row, TOP_K * 4),
                        self.vr("route_weights", row, TOP_K * 4),
                        self.stream.as_raw(),
                    )
                },
                "glm verification ordered reduction",
            )?;
        }
        Ok(())
    }
    pub(in crate::glm53) fn mtp_checkpoint(&mut self, count: usize) -> Result<()> {
        self.verification.as_mut().unwrap().rows = count;
        self.snapshot(count, false)
    }
    pub(in crate::glm53) fn mtp_input(
        &mut self,
        hidden: Ptr,
        embedding: Ptr,
        ready: &Event,
    ) -> Result<Ptr> {
        self.stream.wait_event(ready)?;
        self.copy(self.p("input"), embedding, HIDDEN * 2)?;
        self.copy(self.p("attention_residual"), hidden, HIDDEN * 2)?;
        self.norm("gate", "input", "enorm.weight", HIDDEN)?;
        // Concatenation order is embedding, previous post-final-norm hidden.
        self.norm("up", "attention_residual", "hnorm.weight", HIDDEN)?;
        self.copy(self.p("product"), self.p("gate"), HIDDEN * 2)?;
        let tail = unsafe { self.p("product").cast::<u8>().add(HIDDEN * 2).cast() };
        self.copy(tail, self.p("up"), HIDDEN * 2)?;
        self.linear("attention", "product", "eh_proj.weight", HIDDEN, 2 * HIDDEN)?;
        for h in 0..HC {
            let dst = unsafe { self.p("input").cast::<u8>().add(h * HIDDEN * 2).cast() };
            self.copy(dst, self.p("attention"), HIDDEN * 2)?;
        }
        self.done.record(&self.stream)?;
        Ok(self.p("input"))
    }
    pub(in crate::glm53) fn mtp_norm(&self, out: Ptr) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_norm(
                    out,
                    self.p("output"),
                    self.w("shared_head.norm.weight"),
                    std::ptr::null(),
                    1,
                    HIDDEN as u32,
                    0,
                    self.stream.as_raw(),
                )
            },
            "glm MTP shared head norm",
        )?;
        self.done.record(&self.stream)
    }
}
