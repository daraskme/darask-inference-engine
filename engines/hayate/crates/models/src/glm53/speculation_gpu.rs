use super::super::gpu::Ptr;
use super::*;

impl Executor<'_> {
    pub(super) fn dump_spec_logits(&self, prefix: &Path, start: usize, rows: usize) -> Result<()> {
        let mut raw = vec![0; 4 * self.vocab * 2];
        self.stream.copy_d2h(&mut raw, &self.spec["logits"])?;
        for (row, logits) in raw.chunks_exact(self.vocab * 2).take(rows).enumerate() {
            let bytes: Vec<_> = logits
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|b| (u32::from(u16::from_le_bytes(*b)) << 16).to_le_bytes())
                .collect();
            std::fs::write(
                format!("{}.step{}.bin", prefix.display(), start + row),
                bytes,
            )?;
        }
        Ok(())
    }
    fn sp(&self, name: &str) -> Ptr {
        self.spec[name].as_ptr().cast_mut()
    }
    fn sr(&self, name: &str, row: usize, bytes: usize) -> Ptr {
        unsafe { self.sp(name).cast::<u8>().add(row * bytes).cast() }
    }
    fn copy_ptr(&self, dst: Ptr, src: Ptr, bytes: usize) -> Result<()> {
        check(
            unsafe { g::hayate_glm_copy(dst, src, bytes, self.stream.as_raw()) },
            "glm speculative D2D",
        )
    }
    pub(super) fn spec_copy(&self, name: &str, src: Ptr, bytes: usize) -> Result<()> {
        self.copy_ptr(self.sp(name), src, bytes)
    }
    fn draft_device(&mut self, id: Ptr, recursive: bool, prediction: Option<Ptr>) -> Result<()> {
        check(
            unsafe {
                g::hayate_glm_embed_id(
                    self.input.as_mut_ptr(),
                    self.globals["embed_tokens.weight"].as_ptr(),
                    id,
                    self.vocab as u32,
                    self.stream.as_raw(),
                )
            },
            "glm MTP device embedding",
        )?;
        self.ready.record(&self.stream)?;
        let hidden = self.sp(if recursive { "mtp_hidden" } else { "target" });
        let norm = self.sp("mtp_hidden");
        let embedding = self.input.as_mut_ptr();
        let layer = self.mtp.as_mut().ok_or_else(|| {
            Error::InvalidPlan("GLM missing MTP weights; convert with --mtp".into())
        })?;
        let input = layer.mtp_input(hidden, embedding, &self.ready)?;
        layer.step_device(input, &self.ready, &mut self.experts)?;
        layer.mtp_norm(norm)?;
        self.stream.wait_event(&layer.done)?;
        if let Some(prediction) = prediction {
            check(
                unsafe {
                    g::hayate_glm_dense(
                        self.sp("logits"),
                        norm,
                        g::DenseWeight {
                            data: self.globals["lm_head.weight"].as_ptr(),
                            scale: std::ptr::null(),
                            format: self.head_format,
                        },
                        self.vocab as u32,
                        HIDDEN as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm MTP shared lm_head",
            )?;
            check(
                unsafe {
                    g::hayate_glm_greedy(
                        prediction,
                        self.sp("logits"),
                        self.vocab as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm MTP greedy",
            )?;
        }
        Ok(())
    }
    pub(super) fn draft_prefill(&mut self, token: u32) -> Result<()> {
        self.spec_ids(token, None)?;
        self.draft_device(self.next.as_ptr().cast_mut(), false, None)
    }
    pub(super) fn draft_one(&mut self, token: u32, recursive: bool) -> Result<()> {
        self.prepare_async()?;
        self.spec_ids(token, None)?;
        self.draft_device(self.next.as_ptr().cast_mut(), recursive, None)
    }
    fn spec_ids(&self, anchor: u32, forced: Option<&[u32]>) -> Result<()> {
        let mut ids = [0; 3];
        if let Some(forced) = forced {
            ids[..forced.len()].copy_from_slice(forced);
        }
        check(
            unsafe {
                g::hayate_glm_spec_ids(
                    self.next.as_ptr().cast_mut(),
                    self.sp("drafts"),
                    anchor,
                    ids[0],
                    ids[1],
                    ids[2],
                    forced.is_some() as i32,
                    self.stream.as_raw(),
                )
            },
            "glm speculative IDs",
        )
    }
    fn repair_mtp(&mut self, n: usize) -> Result<()> {
        self.mtp.as_mut().unwrap().mtp_repair_start(&self.ready)?;
        for row in 0..=n {
            let id = if row == 0 {
                self.next.as_mut_ptr()
            } else {
                self.sr("drafts", row - 1, 4)
            };
            let hidden = if row == 0 {
                self.sp("target")
            } else {
                self.sr("hidden", row - 1, HIDDEN * 2)
            };
            check(
                unsafe {
                    g::hayate_glm_embed_id(
                        self.input.as_mut_ptr(),
                        self.globals["embed_tokens.weight"].as_ptr(),
                        id,
                        self.vocab as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm MTP cache repair embedding",
            )?;
            self.ready.record(&self.stream)?;
            let result = self.sp("result");
            let layer = self.mtp.as_mut().unwrap();
            layer.mtp_input(hidden, self.input.as_mut_ptr(), &self.ready)?;
            // Rejected tail latents/pools are invisible at the committed position.
            // Only the in-place partial pool is gated on GPU acceptance.
            layer.mtp_cache_append(result, row)?;
            self.stream.wait_event(&layer.done)?;
        }
        check(
            unsafe {
                g::hayate_glm_restore_prefix(
                    self.sp("target"),
                    self.sp("hidden"),
                    HIDDEN * 2,
                    self.sp("result"),
                    0,
                    self.stream.as_raw(),
                )
            },
            "glm accepted target hidden",
        )
    }
    /// The anchor was already emitted; returned IDs are accepted drafts and one
    /// correction/bonus. Forced drafts/cap are exclusively for correctness tests.
    pub fn spec_step(
        &mut self,
        anchor: u32,
        n: usize,
        forced: Option<&[u32]>,
        cap: usize,
        profile: bool,
    ) -> Result<Vec<u32>> {
        if !(1..=3).contains(&n)
            || cap > n
            || self.position + n + 1 > self.context
            || anchor as usize >= self.vocab
            || forced.is_some_and(|d| d.len() != n || d.iter().any(|&t| t as usize >= self.vocab))
            || self.reference
        {
            return Err(Error::InvalidShape(
                "GLM speculative length/context/tokens".into(),
            ));
        }
        self.prepare_async()?;
        let host_syncs = hayate_kernels::runtime::host_sync_count();
        let start = Instant::now();
        let cpu_before = self.experts.transfer_stats;
        self.mtp
            .as_mut()
            .ok_or_else(|| Error::InvalidPlan("GLM MTP is not loaded".into()))?
            .save_spec_start()?;
        self.spec_ids(anchor, forced)?;
        if forced.is_none() {
            for row in 0..n {
                let id = if row == 0 {
                    self.next.as_mut_ptr()
                } else {
                    self.sr("drafts", row - 1, 4)
                };
                self.draft_device(id, row != 0, Some(self.sr("drafts", row, 4)))?;
                self.mtp.as_mut().unwrap().mtp_checkpoint(row + 1)?;
            }
        }
        let draft_ms = start.elapsed().as_secs_f64() * 1000.;
        let verify = Instant::now();
        for row in 0..=n {
            let id = if row == 0 {
                self.next.as_mut_ptr()
            } else {
                self.sr("drafts", row - 1, 4)
            };
            check(
                unsafe {
                    g::hayate_glm_embed_id(
                        self.sr("input", row, HC * HIDDEN * 2),
                        self.globals["embed_tokens.weight"].as_ptr(),
                        id,
                        self.vocab as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm verifier embedding",
            )?;
        }
        self.ready.record(&self.stream)?;
        let mut input = self.sp("input");
        for index in 0..self.layers.len() {
            let (previous, current) = self.layers.split_at_mut(index);
            let ready = previous.last().map_or(&self.ready, |l| &l.done);
            input = current[0].verify_device(input, ready, n + 1, &mut self.experts, profile)?;
        }
        self.stream.wait_event(&self.layers.last().unwrap().done)?;
        for row in 0..=n {
            let ptr = unsafe { input.cast::<u8>().add(row * HC * HIDDEN * 2).cast() };
            check(
                unsafe {
                    g::hayate_glm_head_mean(
                        self.mean.as_mut_ptr(),
                        ptr,
                        HIDDEN as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm verifier head mean",
            )?;
            check(
                unsafe {
                    g::hayate_glm_norm(
                        self.sr("hidden", row, HIDDEN * 2),
                        self.mean.as_ptr(),
                        self.globals["norm.weight"].as_ptr(),
                        std::ptr::null(),
                        1,
                        HIDDEN as u32,
                        0,
                        self.stream.as_raw(),
                    )
                },
                "glm verifier final norm",
            )?;
        }
        check(
            unsafe {
                g::hayate_glm_dense_batch(
                    self.sp("logits"),
                    self.sp("hidden"),
                    g::DenseWeight {
                        data: self.globals["lm_head.weight"].as_ptr(),
                        scale: std::ptr::null(),
                        format: self.head_format,
                    },
                    (n + 1) as u32,
                    self.vocab as u32,
                    HIDDEN as u32,
                    1,
                    self.stream.as_raw(),
                )
            },
            "glm verifier shared head",
        )?;
        for row in 0..=n {
            check(
                unsafe {
                    g::hayate_glm_greedy(
                        self.sr("predictions", row, 4),
                        self.sr("logits", row, self.vocab * 2),
                        self.vocab as u32,
                        self.stream.as_raw(),
                    )
                },
                "glm verifier greedy",
            )?;
        }
        let verify_submit_ms = verify.elapsed().as_secs_f64() * 1000.;
        let accept = Instant::now();
        check(
            unsafe {
                g::hayate_glm_accept(
                    self.sp("result"),
                    self.sp("drafts"),
                    self.sp("predictions"),
                    n as u32,
                    cap as u32,
                    self.stream.as_raw(),
                )
            },
            "glm GPU acceptance",
        )?;
        self.ready.record(&self.stream)?;
        let accept_submit_ms = accept.elapsed().as_secs_f64() * 1000.;
        let repair = Instant::now();
        for layer in &self.layers {
            layer.rollback_device(self.sp("result"), &self.ready)?;
            self.stream.wait_event(&layer.done)?;
        }
        self.repair_mtp(n)?;
        let repair_submit_ms = repair.elapsed().as_secs_f64() * 1000.;
        let wait = Instant::now();
        // One result read retires acceptance, state restore and MTP cache repair.
        let mut raw = [0; 20];
        self.stream.copy_d2h(&mut raw, &self.spec["result"])?;
        let result_wait_ms = wait.elapsed().as_secs_f64() * 1000.;
        self.retire_async()?;
        let values: Vec<_> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let accepted = values[0] as usize;
        if accepted > n {
            return Err(Error::Weights("GLM invalid acceptance result".into()));
        }
        let output = values[1..accepted + 2].to_vec();
        for layer in &mut self.layers {
            layer.commit_spec_position(accepted + 1);
        }
        self.mtp
            .as_mut()
            .unwrap()
            .commit_spec_position(accepted + 1);
        self.position += accepted + 1;
        self.experts.finish_token(&self.stream)?;
        let (free, total) = self.device.memory_info()?;
        self.memory.vram_peak = self.memory.vram_peak.max(total - free);
        let step_ms = start.elapsed().as_secs_f64() * 1000.;
        if profile {
            let cpu_after = self.experts.transfer_stats;
            for layer in &self.layers {
                layer.report_spec_profile(self.position)?;
            }
            eprintln!(
                "glm_spec position={} n={n} accepted={accepted} draft_submit_ms={draft_ms:.6} verify_submit_ms={verify_submit_ms:.6} accept_submit_ms={accept_submit_ms:.6} repair_submit_ms={repair_submit_ms:.6} result_wait_ms={result_wait_ms:.6} step_ms={step_ms:.6} cpu_experts={} cpu_bytes={} cpu_ms={:.6} cpu_worker_ms={:.6} cpu_dispatch_ms={:.6}",
                self.position,
                cpu_after.cpu_experts - cpu_before.cpu_experts,
                cpu_after.cpu_bytes - cpu_before.cpu_bytes,
                cpu_after.cpu_ms - cpu_before.cpu_ms,
                cpu_after.cpu_worker_ms - cpu_before.cpu_worker_ms,
                cpu_after.cpu_dispatch_ms - cpu_before.cpu_dispatch_ms,
            );
        }
        self.host_syncs += hayate_kernels::runtime::host_sync_count() - host_syncs;
        self.decode_steps += 1;
        Ok(output)
    }
}

impl Executor<'_> {
    pub fn validate_speculation(&mut self, prompt: &[u32], tokens: usize) -> Result<()> {
        if tokens < 16 || prompt.len() + tokens + 4 > self.context || self.mtp.is_none() {
            return Err(Error::InvalidShape(
                "GLM spec validation needs >=16 tokens and prompt+tokens+4 context".into(),
            ));
        }
        self.reset()?;
        let mut reference = vec![self.prefill(prompt)?];
        let mut logits = vec![self.logits_raw()?];
        for _ in 1..tokens + 4 {
            reference.push(self.step(*reference.last().unwrap())?);
            logits.push(self.logits_raw()?);
        }
        for n in 1..=3 {
            for wanted in 0..=n + 1 {
                self.reset()?;
                let mut actual = vec![self.prefill(prompt)?];
                while actual.len() < tokens {
                    let offset = actual.len();
                    let mut drafts = reference[offset..offset + n].to_vec();
                    if wanted < n {
                        drafts[wanted] = (drafts[wanted] + 1) % self.vocab as u32;
                    }
                    let forced = (wanted <= n).then_some(drafts.as_slice());
                    let output = self.spec_step(*actual.last().unwrap(), n, forced, n, false)?;
                    let accepted = output.len() - 1;
                    if wanted <= n && accepted != wanted {
                        return Err(Error::Weights(format!(
                            "GLM forced acceptance n={n} wanted={wanted} got={accepted}"
                        )));
                    }
                    if output != reference[offset..offset + output.len()] {
                        return Err(Error::Weights(format!(
                            "GLM speculative token mismatch n={n} wanted={wanted} offset={offset}"
                        )));
                    }
                    let mut raw = vec![0; 4 * self.vocab * 2];
                    self.stream.copy_d2h(&mut raw, &self.spec["logits"])?;
                    for row in 0..=accepted {
                        if raw[row * self.vocab * 2..(row + 1) * self.vocab * 2]
                            != logits[offset + row]
                        {
                            return Err(Error::Weights(format!(
                                "GLM verification logits not bit identical n={n} wanted={wanted} offset={offset} row={row}"
                            )));
                        }
                    }
                    actual.extend(output);
                }
                eprintln!(
                    "glm_spec_validate n={n} acceptance={} tokens={} exact=true",
                    if wanted <= n {
                        wanted.to_string()
                    } else {
                        "natural".into()
                    },
                    actual.len()
                );
            }
        }
        self.reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires RTX PRO 6000 and GLM53_MTP_HYT; sync vs async exact decode/MTP"]
    fn glm53_async_same_placement_decode_and_mtp_bits() -> Result<()> {
        let root = std::env::var("GLM53_MTP_HYT")
            .map_err(|_| Error::InvalidPlan("set GLM53_MTP_HYT".into()))?;
        let device = Device::select()?;
        let options = |decode_async, reserve| Options {
            decode_async,
            mtp: true,
            mtp_reserve_only: reserve,
            maintenance_interval: 0,
            ..Options::default()
        };
        let prompt = [1, 17, 32, 9, 4];
        let mut reference = Executor::load(&device, Path::new(&root), options(false, true))?;
        let placement = reference.experts.main_placement();
        let mut ids = vec![reference.prefill(&prompt)?];
        let mut logits = vec![reference.logits_raw()?];
        for _ in 1..36 {
            ids.push(reference.step(*ids.last().unwrap())?);
            logits.push(reference.logits_raw()?);
        }
        drop(reference);
        // Simulate context/mailbox/other-process allocation differences across
        // loads without changing the model's fixed memory plan.
        let _allocation_jitter = device.alloc(64 << 20)?;
        let mut actual = Executor::load(&device, Path::new(&root), options(true, false))?;
        assert_eq!(actual.experts.main_placement(), placement);
        for n in 0..=3 {
            actual.reset()?;
            let mut generated = vec![actual.prefill(&prompt)?];
            assert_eq!(generated[0], ids[0]);
            assert_eq!(actual.logits_raw()?, logits[0]);
            while generated.len() < 32 {
                let offset = generated.len();
                let before = (actual.host_syncs, actual.decode_steps);
                if n == 0 {
                    generated.push(actual.step(*generated.last().unwrap())?);
                    assert_eq!(actual.logits_raw()?, logits[offset]);
                } else {
                    let output = actual.spec_step(*generated.last().unwrap(), n, None, n, false)?;
                    let mut raw = vec![0; 4 * actual.vocab * 2];
                    actual.stream.copy_d2h(&mut raw, &actual.spec["logits"])?;
                    for row in 0..output.len() {
                        assert_eq!(
                            raw[row * actual.vocab * 2..(row + 1) * actual.vocab * 2],
                            logits[offset + row]
                        );
                    }
                    generated.extend(output);
                }
                assert_eq!(actual.host_syncs - before.0, 1, "n={n}");
                assert_eq!(actual.decode_steps - before.1, 1);
                assert_eq!(generated, ids[..generated.len()]);
            }
            assert_eq!(actual.experts.main_placement(), placement);
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires RTX PRO 6000 and converted 45-layer + MTP HYT"]
    fn glm53_mtp_reserved_baseline_and_diagnostic_rows() -> Result<()> {
        let root = std::env::var("GLM53_MTP_HYT")
            .map_err(|_| Error::InvalidPlan("set GLM53_MTP_HYT".into()))?;
        let device = Device::select()?;
        let prompt = [1, 17, 32, 9, 4];
        for cpu in [false, true] {
            let options = |reserve| Options {
                mtp: true,
                mtp_reserve_only: reserve,
                mtp_cpu_experts: cpu,
                maintenance_interval: 0,
                ..Options::default()
            };
            let mut baseline = Executor::load(&device, Path::new(&root), options(true))?;
            let placement = baseline.experts.main_placement();
            let memory = (baseline.memory.mtp_added, baseline.memory.mtp_experts_gpu);
            if cpu {
                assert_eq!(memory.1, 0);
            }
            let mut ids = vec![baseline.prefill(&prompt)?];
            let mut logits = vec![baseline.logits_raw()?];
            for _ in 1..36 {
                ids.push(baseline.step(*ids.last().unwrap())?);
                logits.push(baseline.logits_raw()?);
            }
            // Release all weights before loading the other executor.
            drop(baseline);
            let mut executor = Executor::load(&device, Path::new(&root), options(false))?;
            assert_eq!(executor.experts.main_placement(), placement);
            assert_eq!(
                (executor.memory.mtp_added, executor.memory.mtp_experts_gpu),
                memory
            );
            for n in 1..=3 {
                executor.reset()?;
                let mut actual = vec![executor.prefill(&prompt)?];
                assert_eq!(actual[0], ids[0]);
                assert_eq!(executor.logits_raw()?, logits[0]);
                while actual.len() < 32 {
                    let offset = actual.len();
                    let output = executor.spec_step(*actual.last().unwrap(), n, None, n, false)?;
                    assert_eq!(output, ids[offset..offset + output.len()]);
                    let dir = std::env::temp_dir()
                        .join(format!("hayate-mtp-trace-{}", std::process::id()));
                    std::fs::create_dir_all(&dir)?;
                    let prefix = dir.join("logits");
                    executor.dump_spec_logits(&prefix, offset, output.len())?;
                    for row in 0..output.len() {
                        let bytes = std::fs::read(format!(
                            "{}.step{}.bin",
                            prefix.display(),
                            offset + row
                        ))?;
                        let expected: Vec<_> = logits[offset + row]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .flat_map(|b| (u32::from(u16::from_le_bytes(*b)) << 16).to_le_bytes())
                            .collect();
                        assert_eq!(bytes, expected, "cpu={cpu} n={n} row={row}");
                    }
                    std::fs::remove_dir_all(dir)?;
                    actual.extend(output);
                }
                assert_eq!(executor.experts.main_placement(), placement);
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires RTX PRO 6000 and converted 45-layer + MTP HYT"]
    fn glm53_mtp_cache_only_repair_bits() -> Result<()> {
        let root = std::env::var("GLM53_MTP_HYT")
            .map_err(|_| Error::InvalidPlan("set GLM53_MTP_HYT".into()))?;
        let device = Device::select()?;
        let mut executor = Executor::load(
            &device,
            Path::new(&root),
            Options {
                mtp: true,
                context: 4096,
                maintenance_interval: 0,
                ..Options::default()
            },
        )?;
        for prefix in 1..=8 {
            let prompt: Vec<_> = (0..prefix).map(|i| 1 + i * 7).collect();
            for n in 1..=3 {
                for accepted in 0..=n {
                    executor.reset()?;
                    executor.prefill(&prompt)?;
                    executor.mtp.as_mut().unwrap().save_spec_start()?;
                    for row in 0..=n {
                        executor.copy_ptr(
                            executor.sr("hidden", row, HIDDEN * 2),
                            executor.sp("target"),
                            HIDDEN * 2,
                        )?;
                    }
                    let ids = [37, 41, 43, 47];
                    for &id in &ids[..=accepted] {
                        executor.draft_one(id, false)?;
                    }
                    let expected = executor.mtp.as_ref().unwrap().mtp_cache_bytes()?;
                    executor.spec_ids(ids[0], Some(&ids[1..=n]))?;
                    executor.stream.copy_h2d(
                        executor.spec.get_mut("result").unwrap(),
                        &(accepted as u32).to_le_bytes(),
                    )?;
                    executor.ready.record(&executor.stream)?;
                    executor.repair_mtp(n)?;
                    executor.stream.synchronize()?;
                    executor
                        .mtp
                        .as_mut()
                        .unwrap()
                        .commit_spec_position(accepted + 1);
                    let actual = executor.mtp.as_ref().unwrap().mtp_cache_bytes()?;
                    assert_eq!(
                        actual, expected,
                        "MTP cache prefix={prefix} n={n} accepted={accepted}"
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires RTX PRO 6000 and converted 45-layer + MTP HYT"]
    fn glm53_mtp_real_greedy_all_acceptance_counts() -> Result<()> {
        let root = std::env::var("GLM53_MTP_HYT")
            .map_err(|_| Error::InvalidPlan("set GLM53_MTP_HYT".into()))?;
        let device = Device::select()?;
        let mut executor = Executor::load(
            &device,
            Path::new(&root),
            Options {
                mtp: true,
                context: 4096,
                // Fix placement while comparing GPU/CPU arithmetic at identical prefixes.
                maintenance_interval: usize::MAX,
                ..Options::default()
            },
        )?;
        for prompt in [
            vec![1, 17, 32, 9, 4],
            vec![1, 4, 7, 11, 17, 29, 35],
            vec![1, 32, 47, 3, 19, 42, 8, 6],
            (0..2051).map(|i| 1 + (i * 17 % 127)).collect(),
        ] {
            executor.validate_speculation(&prompt, 128)?;
        }
        Ok(())
    }
}
