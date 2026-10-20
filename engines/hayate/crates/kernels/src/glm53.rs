//! GLM stage-1 primitives. BF16 token-major activations; F32 KDA state, mHC
//! maps and router weights. Caller owns sizes, disjoint outputs and lifetimes.
//! All functions enqueue on the supplied stream, without allocation or sync.
#[path = "cpu/glm53.rs"]
pub mod cpu;
#[path = "cpu/glm53_exl3.rs"]
pub mod exl3;

/// EXL3 routed experts on the device (cu/glm53/exl3.cu, crates/models/src/glm53/EXL3.md): mul1 at k2 4..=8.
/// An expert image is gate | up | down (each trellis | suh | svh); `gate_k2` locates the down matrix.
/// Buffers and layouts are the LOWBIT kernels' (decode) and the packed grouped GEMM's (prefill). No allocation,
/// no synchronization: graph capture is valid.
#[cfg(feature = "cuda")]
pub mod exl3_ffi {
    use crate::{cuda::ffi::Stream, glm53::ffi::DenseWeight};
    use std::ffi::c_void;
    unsafe extern "C" {
        /// x [M,H], product [9,M,I] (SwiGLU); images: device array of 8 expert pointers (null: zeros).
        /// Row 8 is the shared expert (`gate`, `up`).
        pub fn hayate_glm_exl3_gate_up(
            product: *mut c_void,
            x: *const c_void,
            images: *const c_void,
            gate: DenseWeight,
            up: DenseWeight,
            m: u32,
            k2: u32,
            hidden: u32,
            intermediate: u32,
            st: Stream,
        ) -> i32;
        /// product [9,M,I] -> out [9,M,H]; a null image leaves its rows untouched (the CPU mailbox merge).
        pub fn hayate_glm_exl3_down(
            out: *mut c_void,
            product: *const c_void,
            images: *const c_void,
            shared: DenseWeight,
            m: u32,
            gate_k2: u32,
            k2: u32,
            hidden: u32,
            intermediate: u32,
            st: Stream,
        ) -> i32;
        /// Expert j's rows: ranges[2j] rows from row ranges[2j+1] of x [rows,H] and product [rows,I].
        pub fn hayate_glm_grouped_exl3_gate_up(
            product: *mut c_void,
            x: *const c_void,
            images: *const c_void,
            ranges: *const c_void,
            count: u32,
            k2: u32,
            hidden: u32,
            intermediate: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_grouped_exl3_down(
            out: *mut c_void,
            product: *const c_void,
            images: *const c_void,
            ranges: *const c_void,
            count: u32,
            gate_k2: u32,
            k2: u32,
            hidden: u32,
            intermediate: u32,
            st: Stream,
        ) -> i32;
        /// The PCIe lane: images src[j] (mapped host memory; null: none) copied into dst[j], `bytes` each
        /// (multiple of 16); dst and src are device arrays of 8 pointers. grid (blocks, 8).
        pub fn hayate_glm_exl3_fetch(
            dst: *const c_void,
            src: *const c_void,
            bytes: usize,
            blocks: u32,
            st: Stream,
        ) -> i32;
        /// One matrix [n,k] (n outputs) as BF16 rows of the original basis.
        pub fn hayate_glm_exl3_restore(
            out: *mut c_void,
            image: *const c_void,
            n: u32,
            k: u32,
            k2: u32,
            st: Stream,
        ) -> i32;
    }
}

#[cfg(feature = "cuda")]
pub mod ffi {
    use crate::cuda::ffi::Stream;
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct DenseWeight {
        pub data: *const c_void,
        pub scale: *const c_void,
        pub format: u32,
    }
    /// Deferred CPU experts of the previous MoE layer (`mailbox` null: none).
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct Deferred {
        pub mailbox: *const c_void,
        pub ids: *const c_void,
        pub weights: *const c_void,
        pub images: *const c_void,
        pub post: *const c_void,
        pub plan: *const c_void,
    }
    /// One GEMV of `hayate_glm_dev_gemv` (BF16 output, `rows` x k).
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct Segment {
        pub w: DenseWeight,
        pub out: *mut c_void,
        pub rows: u32,
    }
    // Device decode (cu/glm53/device.cu): graph-capturable one-token kernels.
    // `ctl` is the device control block {token, position, seq, error, next}.
    unsafe extern "C" {
        pub fn hayate_glm_dev_hc_norm(
            map: *mut c_void,
            collapsed: *mut c_void,
            out: *mut c_void,
            x: *const c_void,
            function: *const c_void,
            base: *const c_void,
            scale: *const c_void,
            w: *const c_void,
            logits: *mut c_void,
            counter: *mut c_void,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_dev_gemv(
            segments: *const Segment,
            count: u32,
            x: *const c_void,
            k: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_dev_kda(
            core: *mut c_void,
            gated: *mut c_void,
            state: *mut c_void,
            hq: *mut c_void,
            hk: *mut c_void,
            hv: *mut c_void,
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            wq: *const c_void,
            wk: *const c_void,
            wv: *const c_void,
            fa: *const c_void,
            fb: DenseWeight,
            ga: *const c_void,
            gb: DenseWeight,
            dt_bias: *const c_void,
            alog: *const c_void,
            beta_raw: *const c_void,
            norm: *const c_void,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_dev_sparse_prep(
            qr: *mut c_void,
            qa: *const c_void,
            qa_norm: *const c_void,
            kvr: *mut c_void,
            kva: *const c_void,
            kv_norm: *const c_void,
            latent: *mut c_void,
            ikn: *mut c_void,
            ik: *const c_void,
            kn_w: *const c_void,
            kn_b: *const c_void,
            ig: *const c_void,
            tail_keys: *mut c_void,
            tail_gates: *mut c_void,
            pools: *mut c_void,
            ape: *const c_void,
            ctl: *const c_void,
            st: Stream,
        ) -> i32;
        /// `plan`: the layer's {now, later, skip mask, defer mask} (u32 each).
        pub fn hayate_glm_dev_route(
            logits: *mut c_void,
            weights: *mut c_void,
            ids: *mut c_void,
            images: *mut c_void,
            plan: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            bias: *const c_void,
            table: *const c_void,
            mailbox: *mut c_void,
            ctl: *const c_void,
            counter: *mut c_void,
            skip_below: f32,
            defer_below: f32,
            format_table: *const c_void,
            formats: *mut c_void,
            // The PCIe lane (null host_table: none): this layer's [288] RAM-tier image pointers, the [4]
            // device scratch slots, the [4] fetch sources the router writes, and take[n] as nibble n.
            host_table: *const c_void,
            scratch: *const c_void,
            src: *mut c_void,
            take: u64,
            // Prefetch (null: none): [8] RAM-tier images fetched ahead for this layer, their [8] device slots and
            // [8] u64 completion tags (seq << 8 | layer); `progress` (u64) gets this layer's tag first.
            pre_src: *const c_void,
            pre_scratch: *const c_void,
            pre_flags: *const c_void,
            progress: *mut c_void,
            layer: u32,
            // Optional u32 [4] counters: prefetched, prefetch used, lane (serial), RAM-tier misses left to the CPU.
            stats: *mut c_void,
            st: Stream,
        ) -> i32;
        /// The prefetch copy for `layer`: slot s of dst [8] from src [8] (null: none) on a side stream; flags[s] gets
        /// the layer's tag when the whole slot is in, unless the router already passed it (progress >= tag).
        /// `done` / `bad`: u32 [8] each, zero between launches.
        pub fn hayate_glm_dev_prefetch(
            dst: *const c_void,
            src: *const c_void,
            bytes: usize,
            blocks: u32,
            done: *mut c_void,
            bad: *mut c_void,
            flags: *mut c_void,
            progress: *const c_void,
            ctl: *const c_void,
            layer: u32,
            st: Stream,
        ) -> i32;
        /// Prefetch pick for the next MoE layer: its router (`w`, `bias`) on this layer's `x`, top-8 in score
        /// order; of those neither in VRAM (`table`) nor without a RAM image (`host`), the first `slots` (<= 8)
        /// host images go to `src` [8] (null the rest). `stats[0]` counts them.
        pub fn hayate_glm_dev_predict(
            logits: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            bias: *const c_void,
            table: *const c_void,
            host: *const c_void,
            src: *mut c_void,
            slots: u32,
            counter: *mut c_void,
            stats: *mut c_void,
            st: Stream,
        ) -> i32;
        /// `flag`: 64 (DONE, this layer) or 96 (DONE2, deferred experts).
        pub fn hayate_glm_dev_moe_wait(
            count: *const c_void,
            mailbox: *const c_void,
            flag: u32,
            ctl: *mut c_void,
            limit_ns: u64,
            st: Stream,
        ) -> i32;
        /// `plan` masks leave dropped and deferred CPU experts out; `prev.mailbox`
        /// non-null adds the previous MoE layer's deferred CPU experts.
        pub fn hayate_glm_dev_moe_finish(
            out: *mut c_void,
            mlp: *mut c_void,
            expert: *const c_void,
            residual: *const c_void,
            map: *const c_void,
            attn_map: *const c_void,
            ids: *const c_void,
            weights: *const c_void,
            images: *const c_void,
            mailbox: *const c_void,
            plan: *const c_void,
            prev: Deferred,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_dev_head_norm(
            mean: *mut c_void,
            out: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            st: Stream,
        ) -> i32;
        /// Writes ctl.next and {next, error} to the mapped `host` pair.
        pub fn hayate_glm_dev_greedy(
            part_values: *mut c_void,
            part_ids: *mut c_void,
            logits: *const c_void,
            n: u32,
            counter: *mut c_void,
            ctl: *mut c_void,
            host: *mut c_void,
            st: Stream,
        ) -> i32;
        /// Loads {token, position, seq} from the mapped `host` block into `ctl`.
        pub fn hayate_glm_dev_embed(
            out: *mut c_void,
            table: *const c_void,
            ctl: *mut c_void,
            host: *const c_void,
            vocab: u32,
            st: Stream,
        ) -> i32;
        /// Bandwidth probe: reads `bytes` (device or mapped host memory).
        pub fn hayate_glm_dev_read_probe(
            src: *const c_void,
            bytes: usize,
            sink: *mut c_void,
            blocks: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_mla_select_device(
            ids: *mut c_void,
            scores: *mut c_void,
            q: *const c_void,
            pools: *const c_void,
            weights: *const c_void,
            position: *const c_void,
            stride: usize,
            st: Stream,
        ) -> i32;
    }
    unsafe extern "C" {
        pub fn hayate_glm_async_publish(
            out: *mut c_void,
            input: *const c_void,
            bytes: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_sparse_gather(
            out: *mut c_void,
            sorted: *mut c_void,
            ids: *mut c_void,
            maps: *mut c_void,
            counts: *mut c_void,
            selected: *const c_void,
            host: *const c_void,
            rows: u32,
            bound: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_sparse_project(
            out: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            counts: *const c_void,
            bound: u32,
            st: Stream,
        ) -> i32;

        pub fn hayate_glm_async_images(
            images: *mut c_void,
            ids: *const c_void,
            table: *const c_void,
            rows: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_async_merge(
            out: *mut c_void,
            host: *const c_void,
            ids: *const c_void,
            table: *const c_void,
            rows: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_async_groups(
            groups: *mut c_void,
            count: *mut c_void,
            gathered: *mut c_void,
            x: *const c_void,
            ids: *const c_void,
            table: *const c_void,
            rows: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_async_gate(
            out: *mut c_void,
            x: *const c_void,
            groups: *const c_void,
            count: *const c_void,
            bound: u32,
            nv: i32,
            down_fp8: i32,
            stride: usize,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_async_down(
            out: *mut c_void,
            x: *const c_void,
            groups: *const c_void,
            count: *const c_void,
            bound: u32,
            nv: i32,
            down_fp8: i32,
            stride: usize,
            st: Stream,
        ) -> i32;

        pub fn hayate_glm_embed_batch(
            out: *mut c_void,
            table: *const c_void,
            ids: *const c_void,
            rows: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_conv_chunk(
            out: *mut c_void,
            history: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            rows: u32,
            cols: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_pool_chunk(
            out: *mut c_void,
            oldk: *mut c_void,
            oldg: *mut c_void,
            keys: *const c_void,
            gates: *const c_void,
            ape: *const c_void,
            start: u32,
            rows: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_select_chunk(
            ids: *mut c_void,
            scores: *mut c_void,
            q: *const c_void,
            pools: *const c_void,
            weights: *const c_void,
            start: u32,
            rows: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_attention_indexed(
            out: *mut c_void,
            q: *const c_void,
            packed: *const c_void,
            indices: *const c_void,
            counts: *const c_void,
            rows: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_prefill_latent(
            host: *mut c_void,
            latent: *const c_void,
            bytes: usize,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_embed_id(
            out: *mut c_void,
            table: *const c_void,
            id: *const c_void,
            vocab: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_dense_batch(
            out: *mut c_void,
            x: *const c_void,
            w: DenseWeight,
            m: u32,
            n: u32,
            k: u32,
            rounded: i32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_route_batch(
            logits: *mut c_void,
            weights: *mut c_void,
            ids: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            bias: *const c_void,
            m: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_mtp_add(
            out: *mut c_void,
            x: *const c_void,
            branch: *const c_void,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_accept(
            out: *mut c_void,
            drafts: *const c_void,
            predictions: *const c_void,
            n: u32,
            cap: u32,
            st: Stream,
        ) -> i32;

        // Async 512-BF16 latent append to pinned RAM. Keep it alive and do not
        // read that row until a later synchronization of this stream.
        pub fn hayate_glm_latent_copy(host: *mut c_void, latent: *const c_void, st: Stream) -> i32;
        pub fn hayate_glm_hc_norm(
            map: *mut c_void,
            collapsed: *mut c_void,
            out: *mut c_void,
            x: *const c_void,
            function: *const c_void,
            base: *const c_void,
            scale: *const c_void,
            weight: *const c_void,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_kda_gated(
            core: *mut c_void,
            out: *mut c_void,
            state: *mut c_void,
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            decay: *const c_void,
            beta: *const c_void,
            gate: *const c_void,
            weight: *const c_void,
            st: Stream,
        ) -> i32;

        // Dense BF16 / block-FP8 / per-tensor FP8 / NVFP4, BF16 coefficients.
        pub fn hayate_glm_dense(
            out: *mut c_void,
            x: *const c_void,
            w: DenseWeight,
            n: u32,
            k: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_dense_unpack(
            out: *mut c_void,
            w: DenseWeight,
            n: u32,
            k: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_route_decode(
            logits: *mut c_void,
            weights: *mut c_void,
            ids: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            bias: *const c_void,
            st: Stream,
        ) -> i32;
        // images: device array of eight pointers, in top-k order. Shared is group 8.
        pub fn hayate_glm_moe_gate_up(
            out: *mut c_void,
            x: *const c_void,
            original: *const c_void,
            images: *const c_void,
            gate: DenseWeight,
            up: DenseWeight,
            nv: i32,
            down_fp8: i32,
            stride: usize,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_moe_down(
            out: *mut c_void,
            x: *const c_void,
            images: *const c_void,
            shared: DenseWeight,
            nv: i32,
            down_fp8: i32,
            stride: usize,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_moe_reduce(
            out: *mut c_void,
            x: *const c_void,
            ids: *const c_void,
            weights: *const c_void,
            st: Stream,
        ) -> i32;

        // In-place is allowed. Each CTA owns one disjoint rotation block.
        pub fn hayate_glm_hadamard(
            out: *mut c_void,
            x: *const c_void,
            rows: u32,
            cols: u32,
            block: u32,
            st: Stream,
        ) -> i32;
        // Grouped BF16 GEMM, up to 8 GLM expert matrices. Metadata is 608 bytes;
        // ranges is U32 [groups,2] (rows,offset), weights is BF16 [groups,n,k].
        pub fn hayate_glm_grouped(
            out: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            ranges: *const c_void,
            groups: u32,
            n: u32,
            k: u32,
            metadata: *mut c_void,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_nvfp4_restore_unscaled(
            out: *mut c_void,
            scale: *mut c_void,
            image: *const c_void,
            n: u32,
            k: u32,
            st: Stream,
        ) -> i32;
        // NVFP4 weights exclude the matrix scale; accumulator is FP32. scales
        // holds one FP32 multiplier per group. max_rows bounds each group.
        pub fn hayate_glm_grouped_nvfp4(
            out: *mut c_void,
            accumulator: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            ranges: *const c_void,
            groups: u32,
            n: u32,
            k: u32,
            metadata: *mut c_void,
            scales: *const c_void,
            max_rows: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_nvfp4_restore(
            out: *mut c_void,
            image: *const c_void,
            n: u32,
            k: u32,
            st: Stream,
        ) -> i32;
        // Assignments index [token,8] routes. Each scatter call contains at most
        // one row per token; callers enqueue experts in ascending ID order.
        pub fn hayate_glm_gather(
            out: *mut c_void,
            x: *const c_void,
            assignments: *const c_void,
            rows: u32,
            hidden: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_scatter(
            out: *mut c_void,
            x: *const c_void,
            weights: *const c_void,
            assignments: *const c_void,
            rows: u32,
            hidden: u32,
            st: Stream,
        ) -> i32;
        // Device copy; null x zeroes out. Buffers may not overlap.
        pub fn hayate_glm_copy(out: *mut c_void, x: *const c_void, bytes: usize, st: Stream)
        -> i32;
        pub fn hayate_glm_embed(
            out: *mut c_void,
            table: *const c_void,
            token: u32,
            hidden: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_fp8_gemv(
            out: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            scale: *const c_void,
            n: u32,
            k: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_conv_step(
            out: *mut c_void,
            history: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            cols: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_select_step(
            ids: *mut c_void,
            scores: *mut c_void,
            q: *const c_void,
            pools: *const c_void,
            weights: *const c_void,
            length: u32,
            st: Stream,
        ) -> i32;
        // Each index addresses one packed [64, 2, 256] KV union row. Outputs
        // contain count chronological rows; all four allocations are disjoint.
        pub fn hayate_glm_split_selected(
            k: *mut c_void,
            v: *mut c_void,
            packed: *const c_void,
            indices: *const c_void,
            count: u32,
            st: Stream,
        ) -> i32;
        // Scalar launch arguments own the IDs; no host copy/source lifetime.
        // With forced=0 drafts may be null; otherwise it holds three IDs.
        pub fn hayate_glm_spec_ids(
            next: *mut c_void,
            drafts: *mut c_void,
            anchor: u32,
            d0: u32,
            d1: u32,
            d2: u32,
            forced: i32,
            st: Stream,
        ) -> i32;
        // Copy snapshot[result[0]+add] to out; result[0] is accepted in 0..3.
        // add=1 restores target state, add=0 selects accepted hidden. No overlap.
        pub fn hayate_glm_restore_prefix(
            out: *mut c_void,
            snapshots: *const c_void,
            bytes: usize,
            result: *const c_void,
            add: u32,
            st: Stream,
        ) -> i32;
        // Copy only when row <= accepted, preserving the partial index pool.
        pub fn hayate_glm_copy_if(
            out: *mut c_void,
            x: *const c_void,
            bytes: usize,
            result: *const c_void,
            row: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_attention_step(
            out: *mut c_void,
            probs: *mut c_void,
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            count: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_head_mean(
            out: *mut c_void,
            x: *const c_void,
            hidden: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_greedy(id: *mut c_void, logits: *const c_void, n: u32, st: Stream)
        -> i32;

        pub fn hayate_glm_restore(
            out: *mut c_void,
            w: *const c_void,
            scale: *const c_void,
            n: u32,
            k: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_norm(
            out: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            bias: *const c_void,
            rows: u32,
            cols: u32,
            mode: i32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_point(
            out: *mut c_void,
            x: *const c_void,
            y: *const c_void,
            count: usize,
            mode: i32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_conv(
            out: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            tokens: u32,
            cols: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_hc(
            map: *mut c_void,
            out: *mut c_void,
            x: *const c_void,
            mapping: *const c_void,
            base: *const c_void,
            scale: *const c_void,
            tokens: u32,
            hidden: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_expand(
            out: *mut c_void,
            x: *const c_void,
            branch: *const c_void,
            map: *const c_void,
            tokens: u32,
            hidden: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_decay(
            out: *mut c_void,
            x: *const c_void,
            bias: *const c_void,
            alog: *const c_void,
            tokens: u32,
            heads: u32,
            dim: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_kda(
            out: *mut c_void,
            state: *mut c_void,
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const c_void,
            beta: *const c_void,
            tokens: u32,
            heads: u32,
            dim: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_kda_chunk(
            out: *mut c_void,
            state: *mut c_void,
            nq: *mut c_void,
            nk: *mut c_void,
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const c_void,
            beta: *const c_void,
            tokens: u32,
            heads: u32,
            dim: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_gated(
            out: *mut c_void,
            x: *const c_void,
            gate: *const c_void,
            w: *const c_void,
            rows: u32,
            dim: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_route(
            logits: *mut c_void,
            weights: *mut c_void,
            ids: *mut c_void,
            x: *const c_void,
            w: *const c_void,
            bias: *const c_void,
            tokens: u32,
            hidden: u32,
            experts: u32,
            topk: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_accumulate(
            out: *mut c_void,
            x: *const c_void,
            ids: *const c_void,
            weights: *const c_void,
            tokens: u32,
            hidden: u32,
            topk: u32,
            expert: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_pool(
            out: *mut c_void,
            keys: *const c_void,
            gates: *const c_void,
            ape: *const c_void,
            pools: u32,
            dim: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_select(
            ids: *mut c_void,
            scratch: *mut c_void,
            q: *const c_void,
            pools: *const c_void,
            weights: *const c_void,
            tokens: u32,
            heads: u32,
            dim: u32,
            budget: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_split(
            k: *mut c_void,
            v: *mut c_void,
            packed: *const c_void,
            tokens: u32,
            heads: u32,
            dim: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_mla_prepare(
            wkt: *mut c_void,
            wv: *mut c_void,
            kvb: *const c_void,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_mla_heads(
            out: *mut c_void,
            x: *const c_void,
            rows: u32,
            merge: i32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_mla_absorb_q(
            qabs: *mut c_void,
            q: *const c_void,
            wkt: *const c_void,
            rows: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_mla_absorb_out(
            out: *mut c_void,
            olat: *const c_void,
            wv: *const c_void,
            rows: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_mla_attention(
            out: *mut c_void,
            part: *mut c_void,
            stats: *mut c_void,
            qabs: *const c_void,
            latent: *const c_void,
            ids: *const c_void,
            rows: u32,
            splits: u32,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_mla_select(
            ids: *mut c_void,
            scores: *mut c_void,
            q: *const c_void,
            pools: *const c_void,
            weights: *const c_void,
            start: u32,
            rows: u32,
            stride: usize,
            st: Stream,
        ) -> i32;
        pub fn hayate_glm_attention(
            out: *mut c_void,
            probs: *mut c_void,
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            ids: *const c_void,
            tokens: u32,
            heads: u32,
            dim: u32,
            width: u32,
            st: Stream,
        ) -> i32;
    }
    /// One LoRA on one linear (cu/glm53/lora.cu): A [count][rank][k], B [count][n][rank],
    /// dtype 0 F32, 1 F16, 2 BF16; strides in elements between experts (count > 1).
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct GlmLora {
        pub a: *const c_void,
        pub b: *const c_void,
        pub a_stride: u64,
        pub b_stride: u64,
        pub rank: u32,
        pub k: u32,
        pub n: u32,
        pub dtype: u32,
        pub count: u32,
        pub scale: f32,
    }
    unsafe extern "C" {
        /// `out[row] = bf16(out[row] + scale * B_e (A_e x[row]))` for `rows` BF16 rows
        /// (x stride k, out stride n); `t` is rows x rank F32 scratch. `experts` (i32 per
        /// row) picks e when count > 1; rows whose `images` entry is null are skipped.
        pub fn hayate_glm_lora(
            out: *mut c_void,
            x: *const c_void,
            t: *mut c_void,
            lora: GlmLora,
            rows: u32,
            experts: *const c_void,
            images: *const c_void,
            st: Stream,
        ) -> i32;
    }
}

#[cfg(all(test, feature = "cuda"))]
mod lora_tests {
    use super::cpu::bf16;
    use super::ffi as g;
    use crate::runtime::{Device, check};
    use hayate_core::Result;

    /// F32 -> F16 bits, round to nearest even (normal range only, as the test values).
    fn half_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        if x == 0. {
            return sign;
        }
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        assert!((1..31).contains(&exp), "test value out of F16 normal range");
        let mant = bits & 0x7fffff;
        let mut out = ((exp as u32) << 10) | (mant >> 13);
        let rest = mant & 0x1fff;
        if rest > 0x1000 || (rest == 0x1000 && out & 1 == 1) {
            out += 1;
        }
        sign | out as u16
    }
    fn from_half(h: u16) -> f32 {
        let exp = ((h >> 10) & 31) as i32;
        let mant = (h & 1023) as f32;
        let v = if exp == 0 {
            mant / 1024. * 2f32.powi(-14)
        } else {
            (1. + mant / 1024.) * 2f32.powi(exp - 15)
        };
        if h & 0x8000 != 0 { -v } else { v }
    }
    fn bf(x: f32) -> u16 {
        (bf16(x).to_bits() >> 16) as u16
    }
    fn unbf(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }

    /// Routed rows (per-row expert, a skipped CPU row) and a dense call, against an F64
    /// reference of the same math; the only rounding is the final BF16 store.
    #[test]
    #[ignore = "requires supported CUDA GPU; checkpoint-free LoRA side path"]
    fn glm53_lora_matches_reference_gpu() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2. - 1.
        };
        for (dtype, rank, k, n, count) in [
            (1u32, 1usize, 2048usize, 4096usize, 5usize),
            (2, 3, 2048, 4096, 5),
            (0, 9, 8192, 4096, 1),
            (1, 16, 16384, 512, 1),
        ] {
            let rows = if count > 1 { 6 } else { 3 };
            let width = if dtype == 0 { 4 } else { 2 };
            let encode = |v: f32| -> (Vec<u8>, f32) {
                // Inside the F16 normal range.
                let v = if v.abs() < 1e-3 {
                    1e-3f32.copysign(v)
                } else {
                    v
                };
                match dtype {
                    0 => (v.to_le_bytes().to_vec(), v),
                    1 => {
                        let b = half_bits(v);
                        (b.to_le_bytes().to_vec(), from_half(b))
                    }
                    _ => {
                        let b = bf(v);
                        (b.to_le_bytes().to_vec(), unbf(b))
                    }
                }
            };
            let (mut a_raw, mut a) = (Vec::new(), Vec::new());
            for _ in 0..count * rank * k {
                let (raw, value) = encode(next() * 0.05);
                a_raw.extend(raw);
                a.push(value);
            }
            let (mut b_raw, mut b) = (Vec::new(), Vec::new());
            for _ in 0..count * n * rank {
                let (raw, value) = encode(next() * 0.05);
                b_raw.extend(raw);
                b.push(value);
            }
            let x: Vec<u16> = (0..rows * k).map(|_| bf(next())).collect();
            let base: Vec<u16> = (0..rows * n).map(|_| bf(next())).collect();
            let experts: Vec<i32> = (0..rows).map(|r| ((r * 3 + 1) % count) as i32).collect();
            // Row 2 is a CPU expert in the routed case (null image): left as is.
            let images: Vec<u64> = (0..rows).map(|r| u64::from(r != 2)).collect();
            let routed = count > 1;
            let scale = 0.75f32;
            let upload = |bytes: &[u8]| -> Result<_> {
                let mut buffer = device.alloc(bytes.len())?;
                stream.copy_h2d(&mut buffer, bytes)?;
                Ok(buffer)
            };
            let da = upload(&a_raw)?;
            let db = upload(&b_raw)?;
            let dx = upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
            let mut out = upload(
                &base
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let de = upload(
                &experts
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let di = upload(
                &images
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let mut t = device.alloc(rows * rank * 4)?;
            assert_eq!(a_raw.len(), count * rank * k * width);
            let lora = g::GlmLora {
                a: da.as_ptr(),
                b: db.as_ptr(),
                a_stride: (rank * k) as u64,
                b_stride: (n * rank) as u64,
                rank: rank as u32,
                k: k as u32,
                n: n as u32,
                dtype,
                count: count as u32,
                scale,
            };
            check(
                unsafe {
                    g::hayate_glm_lora(
                        out.as_mut_ptr(),
                        dx.as_ptr(),
                        t.as_mut_ptr(),
                        lora,
                        rows as u32,
                        if routed {
                            de.as_ptr()
                        } else {
                            std::ptr::null()
                        },
                        if routed {
                            di.as_ptr()
                        } else {
                            std::ptr::null()
                        },
                        stream.as_raw(),
                    )
                },
                "GLM LoRA test",
            )?;
            let mut raw = vec![0; rows * n * 2];
            stream.copy_d2h(&mut raw, &out)?;
            let actual: Vec<f32> = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| unbf(u16::from_le_bytes(*b)))
                .collect();
            let mut worst = 0f64;
            for row in 0..rows {
                let e = if routed { experts[row] as usize } else { 0 };
                let skip = routed && images[row] == 0;
                let tv: Vec<f64> = (0..rank)
                    .map(|j| {
                        (0..k)
                            .map(|c| a[(e * rank + j) * k + c] as f64 * unbf(x[row * k + c]) as f64)
                            .sum()
                    })
                    .collect();
                for c in 0..n {
                    let old = unbf(base[row * n + c]);
                    let got = actual[row * n + c];
                    if skip {
                        assert_eq!(got.to_bits(), old.to_bits(), "skipped row changed");
                        continue;
                    }
                    let delta: f64 = (0..rank)
                        .map(|j| b[(e * n + c) * rank + j] as f64 * tv[j])
                        .sum();
                    let want = old as f64 + scale as f64 * delta;
                    // One BF16 rounding of the sum (2^-8 relative) plus F32 sums.
                    let err = (got as f64 - want).abs() / want.abs().max(1e-3);
                    worst = worst.max(err);
                    assert!(err < 8e-3, "row {row} col {c}: {got} vs {want}");
                }
            }
            eprintln!(
                "lora dtype={dtype} rank={rank} k={k} n={n} count={count} worst_rel={worst:.3e}"
            );
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::ffi as g;
    use crate::runtime::{Device, check};
    use hayate_core::Result;

    #[test]
    #[ignore = "requires supported CUDA GPU; checkpoint-free ragged groups"]
    fn glm53_grouped_gemm_gpu() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        for (n, k) in [(2048usize, 4096usize), (4096, 2048)] {
            let ranges = [1u32, 0, 3, 1, 2, 4];
            let mut x = Vec::new();
            for row in 0..6 {
                let value = ((row + 1) as f32).to_bits() as u64 >> 16;
                x.extend((value as u16).to_le_bytes().repeat(k));
            }
            let mut weights = vec![0; 3 * n * k * 2];
            for group in 0..3 {
                for j in 0..n.min(k) {
                    let at = ((group * n + j) * k + j) * 2;
                    weights[at..at + 2].copy_from_slice(&0x3f80u16.to_le_bytes());
                }
            }
            let mut a = device.alloc(x.len())?;
            let mut w = device.alloc(weights.len())?;
            let mut r = device.alloc(ranges.len() * 4)?;
            let mut out = device.alloc(6 * n * 2)?;
            let mut meta = device.alloc(608)?;
            stream.copy_h2d(&mut a, &x)?;
            stream.copy_h2d(&mut w, &weights)?;
            stream.copy_h2d(
                &mut r,
                &ranges
                    .iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            check(
                unsafe {
                    g::hayate_glm_grouped(
                        out.as_mut_ptr(),
                        a.as_ptr(),
                        w.as_ptr(),
                        r.as_ptr(),
                        3,
                        n as u32,
                        k as u32,
                        meta.as_mut_ptr(),
                        stream.as_raw(),
                    )
                },
                "GLM ragged grouped test",
            )?;
            let mut raw = vec![0; out.bytes()];
            stream.copy_d2h(&mut raw, &out)?;
            for (i, bytes) in raw.as_chunks::<2>().0.iter().enumerate() {
                let expected = if i % n < k { (i / n + 1) as f32 } else { 0. };
                assert_eq!(
                    f32::from_bits((u16::from_le_bytes(*bytes) as u32) << 16),
                    expected
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires supported CUDA GPU; checkpoint-free NVFP4 restore/GEMV"]
    fn glm53_nvfp4_gemv_gpu() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let (n, k) = (129usize, 256usize);
        let bytes = hayate_core::QuantSpec::Nvfp4
            .matrix_bytes(n as u64, k as u64)
            .unwrap() as usize;
        let mut image = vec![0x38; bytes];
        image[..n * k / 2].fill(0x44); // E2M1(2) * E4M3(1) * .5 = 1
        image[bytes - 4..].copy_from_slice(&0.5f32.to_le_bytes());
        let mut weights = device.alloc(bytes)?;
        let mut x = device.alloc(k * 2)?;
        let mut restored = device.alloc(n * k * 2)?;
        let mut out = device.alloc(n * 2)?;
        stream.copy_h2d(&mut weights, &image)?;
        stream.copy_h2d(&mut x, &0x3f80u16.to_le_bytes().repeat(k))?;
        check(
            unsafe {
                g::hayate_glm_nvfp4_restore(
                    restored.as_mut_ptr(),
                    weights.as_ptr(),
                    n as u32,
                    k as u32,
                    stream.as_raw(),
                )
            },
            "GLM NVFP4 restore test",
        )?;
        let mut raw = vec![0; restored.bytes()];
        stream.copy_d2h(&mut raw, &restored)?;
        assert!(
            raw.as_chunks::<2>()
                .0
                .iter()
                .all(|b| u16::from_le_bytes(*b) == 0x3f80)
        );
        check(
            unsafe {
                crate::llm::ffi::hayate_llm_gemv(
                    out.as_mut_ptr(),
                    x.as_ptr(),
                    weights.as_ptr(),
                    n as u32,
                    k as u32,
                    1,
                    0,
                    std::ptr::null_mut(),
                    4,
                    1,
                    0,
                    stream.as_raw(),
                )
            },
            "GLM reused NVFP4 GEMV test",
        )?;
        let mut raw = vec![0; out.bytes()];
        stream.copy_d2h(&mut raw, &out)?;
        assert!(
            raw.as_chunks::<2>()
                .0
                .iter()
                .all(|b| u16::from_le_bytes(*b) == 0x4380)
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires supported CUDA GPU; checkpoint-free gather/scatter"]
    fn glm53_grouped_routes_gpu() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut input = device.alloc(2 * 16 * 2)?;
        let mut assignments = device.alloc(4 * 4)?;
        let mut weights = device.alloc(16 * 4)?;
        let mut gathered = device.alloc(4 * 16 * 2)?;
        let mut out = device.alloc(2 * 16 * 2)?;
        let mut x = 0x3f80u16.to_le_bytes().repeat(16);
        x.extend(0x4000u16.to_le_bytes().repeat(16));
        stream.copy_h2d(&mut input, &x)?;
        stream.copy_h2d(
            &mut assignments,
            &[7u32, 15, 0, 8]
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut w = [0f32; 16];
        w[7] = 0.5;
        w[15] = 0.25;
        w[0] = 1.;
        w[8] = 2.;
        stream.copy_h2d(
            &mut weights,
            &w.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>(),
        )?;
        check(
            unsafe {
                g::hayate_glm_gather(
                    gathered.as_mut_ptr(),
                    input.as_ptr(),
                    assignments.as_ptr(),
                    4,
                    16,
                    stream.as_raw(),
                )
            },
            "GLM gather test",
        )?;
        check(
            unsafe {
                g::hayate_glm_copy(
                    out.as_mut_ptr(),
                    std::ptr::null(),
                    out.bytes(),
                    stream.as_raw(),
                )
            },
            "GLM zero test",
        )?;
        for (offset, rows) in [(0, 2), (2, 1), (3, 1)] {
            check(
                unsafe {
                    g::hayate_glm_scatter(
                        out.as_mut_ptr(),
                        gathered.as_ptr().cast::<u8>().add(offset * 16 * 2).cast(),
                        weights.as_ptr(),
                        assignments.as_ptr().cast::<u8>().add(offset * 4).cast(),
                        rows,
                        16,
                        stream.as_raw(),
                    )
                },
                "GLM scatter test",
            )?;
        }
        let mut raw = vec![0; out.bytes()];
        stream.copy_d2h(&mut raw, &out)?;
        for (i, b) in raw.as_chunks::<2>().0.iter().enumerate() {
            assert_eq!(
                f32::from_bits((u16::from_le_bytes(*b) as u32) << 16),
                if i < 16 { 1.5 } else { 4.5 }
            );
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod decode_tests {
    use super::ffi as g;
    use crate::{
        graph::Graph,
        runtime::{Device, check},
    };
    use hayate_core::{QuantSpec, Result};

    fn bf(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    #[test]
    #[ignore = "requires sm_120 GPU; dense formats, partial tiles and changed-input graph replay"]
    fn glm53_dense_decode_gpu() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let (n, k) = (129usize, 80usize);
        let mut x = device.alloc(k * 2)?;
        let out = device.alloc(n * 2)?;
        let mut restored = device.alloc(n * k * 2)?;
        for format in 0..=3 {
            let mut raw = match format {
                0 => bf(0.375).to_le_bytes().repeat(n * k),
                1 => {
                    let mut bytes = vec![0x38; n * k];
                    bytes.extend(
                        0.375f32
                            .to_le_bytes()
                            .repeat(n.div_ceil(128) * k.div_ceil(128)),
                    );
                    bytes
                }
                2 => {
                    let mut bytes = vec![0x38; n * k];
                    bytes.extend(0.375f32.to_le_bytes());
                    bytes
                }
                _ => {
                    let bytes = QuantSpec::Nvfp4.matrix_bytes(n as u64, k as u64).unwrap() as usize;
                    let mut image = vec![0x38; bytes];
                    image[..n * k / 2].fill(0x22);
                    image[bytes - 4..].copy_from_slice(&0.375f32.to_le_bytes());
                    image
                }
            };
            let mut weight = device.alloc(raw.len())?;
            stream.copy_h2d(&mut weight, &raw)?;
            let w = g::DenseWeight {
                data: weight.as_ptr(),
                scale: if format == 1 {
                    unsafe { weight.as_ptr().cast::<u8>().add(n * k).cast() }
                } else {
                    std::ptr::null()
                },
                format,
            };
            stream.copy_h2d(&mut x, &bf(1.).to_le_bytes().repeat(k))?;
            let submit = || {
                check(
                    unsafe {
                        g::hayate_glm_dense(
                            out.as_ptr().cast_mut(),
                            x.as_ptr(),
                            w,
                            n as u32,
                            k as u32,
                            stream.as_raw(),
                        )
                    },
                    "glm dense test",
                )
            };
            submit()?;
            stream.synchronize()?;
            let graph = unsafe { Graph::capture(stream.as_raw(), submit) }?;
            for input in [1., -0.5] {
                stream.copy_h2d(&mut x, &bf(input).to_le_bytes().repeat(k))?;
                unsafe { graph.launch(stream.as_raw()) }?;
                let mut actual = vec![0; n * 2];
                stream.copy_d2h(&mut actual, &out)?;
                assert_eq!(actual, bf(input * 0.375 * k as f32).to_le_bytes().repeat(n));
            }
            if format != 0 {
                check(
                    unsafe {
                        g::hayate_glm_dense_unpack(
                            restored.as_mut_ptr(),
                            w,
                            n as u32,
                            k as u32,
                            stream.as_raw(),
                        )
                    },
                    "glm dense unpack test",
                )?;
                raw.resize(n * k * 2, 0);
                stream.copy_d2h(&mut raw, &restored)?;
                assert_eq!(raw, bf(0.375).to_le_bytes().repeat(n * k));
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires sm_120 GPU; parallel router compared with legacy GPU top-k"]
    fn glm53_router_decode_gpu() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut x = device.alloc(4096 * 2)?;
        stream.copy_h2d(&mut x, &bf(1.).to_le_bytes().repeat(4096))?;
        let mut weight = device.alloc(288 * 4096 * 2)?;
        let mut bytes = vec![0; weight.bytes()];
        for e in 0..288 {
            bytes[e * 4096 * 2..e * 4096 * 2 + 2]
                .copy_from_slice(&bf((e as f32 - 144.) / 32.).to_le_bytes());
        }
        stream.copy_h2d(&mut weight, &bytes)?;
        let mut bias = device.alloc(288 * 4)?;
        stream.copy_h2d(
            &mut bias,
            &(0..288)
                .flat_map(|e| ((e % 13) as f32 / 8.).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut logits = device.alloc(288 * 4)?;
        let mut weights = device.alloc(8 * 4)?;
        let mut ids = device.alloc(8 * 4)?;
        let mut expected = Vec::new();
        for reference in [true, false] {
            check(
                unsafe {
                    if reference {
                        g::hayate_glm_route(
                            logits.as_mut_ptr(),
                            weights.as_mut_ptr(),
                            ids.as_mut_ptr(),
                            x.as_ptr(),
                            weight.as_ptr(),
                            bias.as_ptr(),
                            1,
                            4096,
                            288,
                            8,
                            stream.as_raw(),
                        )
                    } else {
                        g::hayate_glm_route_decode(
                            logits.as_mut_ptr(),
                            weights.as_mut_ptr(),
                            ids.as_mut_ptr(),
                            x.as_ptr(),
                            weight.as_ptr(),
                            bias.as_ptr(),
                            stream.as_raw(),
                        )
                    }
                },
                "glm router test",
            )?;
            let mut actual = vec![0; 32];
            stream.copy_d2h(&mut actual, &ids)?;
            let mut probs = vec![0; 32];
            stream.copy_d2h(&mut probs, &weights)?;
            actual.extend(probs);
            if reference {
                expected = actual;
            } else {
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires sm_120 GPU; checkpoint-free nine-expert NVFP4 batch and shared BF16"]
    fn glm53_moe_decode_gpu() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let stride = 4_718_848usize;
        let bytes = 4_718_596usize;
        let mut host = vec![0; 3 * stride];
        for part in 0..3 {
            host[part * stride..part * stride + 4096 * 2048 / 2].fill(0x22);
            host[part * stride + 4096 * 2048 / 2..part * stride + bytes - 4].fill(0x38);
            host[part * stride + bytes - 4..part * stride + bytes].copy_from_slice(
                &(if part == 2 {
                    1f32 / 2048.
                } else {
                    1f32 / 4096.
                })
                .to_le_bytes(),
            );
        }
        let mut image = device.alloc(host.len())?;
        stream.copy_h2d(&mut image, &host)?;
        let mut images = device.alloc(8 * 8)?;
        stream.copy_h2d(
            &mut images,
            &(image.as_ptr() as u64).to_le_bytes().repeat(8),
        )?;
        let mut gate = device.alloc(4096 * 2048 * 2)?;
        let mut down = device.alloc(4096 * 2048 * 2)?;
        stream.copy_h2d(&mut gate, &bf(1. / 4096.).to_le_bytes().repeat(4096 * 2048))?;
        stream.copy_h2d(&mut down, &bf(1. / 2048.).to_le_bytes().repeat(4096 * 2048))?;
        let shared = |data| g::DenseWeight {
            data,
            scale: std::ptr::null(),
            format: 0,
        };
        let mut x = device.alloc(4096 * 2)?;
        stream.copy_h2d(&mut x, &bf(1.).to_le_bytes().repeat(4096))?;
        let mut product = device.alloc(9 * 2048 * 2)?;
        let mut expert = device.alloc(9 * 4096 * 2)?;
        let mut out = device.alloc(4096 * 2)?;
        let mut ids = device.alloc(8 * 4)?;
        stream.copy_h2d(
            &mut ids,
            &[287u32, 0, 17, 42, 8, 256, 3, 4]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut weights = device.alloc(8 * 4)?;
        let probs = [0.17f32, 0.13, 0.49, 0.53, 0.73, 0.11, 0.19, 0.23];
        stream.copy_h2d(
            &mut weights,
            &probs
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        check(
            unsafe {
                g::hayate_glm_moe_gate_up(
                    product.as_mut_ptr(),
                    x.as_ptr(),
                    x.as_ptr(),
                    images.as_ptr(),
                    shared(gate.as_ptr()),
                    shared(gate.as_ptr()),
                    1,
                    0,
                    stride,
                    stream.as_raw(),
                )
            },
            "glm batch gate test",
        )?;
        check(
            unsafe {
                g::hayate_glm_moe_down(
                    expert.as_mut_ptr(),
                    product.as_ptr(),
                    images.as_ptr(),
                    shared(down.as_ptr()),
                    1,
                    0,
                    stride,
                    stream.as_raw(),
                )
            },
            "glm batch down test",
        )?;
        check(
            unsafe {
                g::hayate_glm_moe_reduce(
                    out.as_mut_ptr(),
                    expert.as_ptr(),
                    ids.as_ptr(),
                    weights.as_ptr(),
                    stream.as_raw(),
                )
            },
            "glm batch reduce test",
        )?;
        let value = f32::from_bits((bf(1. / (1. + (-1f32).exp())) as u32) << 16);
        let mut sum = 0.;
        for i in [1, 6, 7, 4, 2, 3, 5, 0] {
            let term = f32::from_bits((bf(value * probs[i]) as u32) << 16);
            sum = f32::from_bits((bf(sum + term) as u32) << 16);
        }
        let mut actual = vec![0; 4096 * 2];
        stream.copy_d2h(&mut actual, &out)?;
        assert_eq!(actual, bf(sum + value).to_le_bytes().repeat(4096));
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod fusion_tests {
    use super::ffi as g;
    use crate::runtime::{Device, check};
    use hayate_core::Result;

    #[test]
    #[ignore = "requires sm_120 GPU; mHC collapse/norm and KDA/gated norm bit equality"]
    fn glm53_decode_fusions_gpu() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let make_bf = |count: usize, modulus: usize| -> Result<_> {
            let mut buffer = device.alloc(count * 2)?;
            let bytes = (0..count)
                .flat_map(|i| {
                    let v = (i % modulus) as f32 / 16. - 1.;
                    let bits = v.to_bits();
                    (((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16).to_le_bytes()
                })
                .collect::<Vec<_>>();
            stream.copy_h2d(&mut buffer, &bytes)?;
            Ok(buffer)
        };
        let input = make_bf(4 * 4096, 53)?;
        let function = make_bf(24 * 4 * 4096, 17)?;
        let weight = make_bf(4096, 23)?;
        let mut base = device.alloc(24 * 4)?;
        stream.copy_h2d(
            &mut base,
            &(0..24)
                .flat_map(|i| (i as f32 / 24.).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut scale = device.alloc(3 * 4)?;
        stream.copy_h2d(&mut scale, &0.001f32.to_le_bytes().repeat(3))?;
        let mut map = device.alloc(24 * 4)?;
        let mut collapsed = device.alloc(4096 * 2)?;
        let mut out = device.alloc(4096 * 2)?;
        check(
            unsafe {
                g::hayate_glm_hc(
                    map.as_mut_ptr(),
                    collapsed.as_mut_ptr(),
                    input.as_ptr(),
                    function.as_ptr(),
                    base.as_ptr(),
                    scale.as_ptr(),
                    1,
                    4096,
                    stream.as_raw(),
                )
            },
            "glm HC reference",
        )?;
        check(
            unsafe {
                g::hayate_glm_norm(
                    out.as_mut_ptr(),
                    collapsed.as_ptr(),
                    weight.as_ptr(),
                    std::ptr::null(),
                    1,
                    4096,
                    0,
                    stream.as_raw(),
                )
            },
            "glm HC norm reference",
        )?;
        let mut expected = vec![0; out.bytes()];
        stream.copy_d2h(&mut expected, &out)?;
        check(
            unsafe {
                g::hayate_glm_hc_norm(
                    map.as_mut_ptr(),
                    collapsed.as_mut_ptr(),
                    out.as_mut_ptr(),
                    input.as_ptr(),
                    function.as_ptr(),
                    base.as_ptr(),
                    scale.as_ptr(),
                    weight.as_ptr(),
                    stream.as_raw(),
                )
            },
            "glm HC fused",
        )?;
        let mut actual = vec![0; out.bytes()];
        stream.copy_d2h(&mut actual, &out)?;
        assert_eq!(actual, expected);
        let q = make_bf(8192, 17)?;
        let k = make_bf(8192, 19)?;
        let v = make_bf(8192, 23)?;
        let gate = make_bf(8192, 29)?;
        let beta = make_bf(64, 31)?;
        let norm = make_bf(128, 13)?;
        let mut decay = device.alloc(8192 * 4)?;
        stream.copy_h2d(&mut decay, &(-0.03f32).to_le_bytes().repeat(8192))?;
        let mut state = device.alloc(64 * 128 * 128 * 4)?;
        let mut second_state = device.alloc(state.bytes())?;
        check(
            unsafe {
                g::hayate_glm_copy(
                    state.as_mut_ptr(),
                    std::ptr::null(),
                    state.bytes(),
                    stream.as_raw(),
                )
            },
            "glm state zero",
        )?;
        check(
            unsafe {
                g::hayate_glm_copy(
                    second_state.as_mut_ptr(),
                    std::ptr::null(),
                    second_state.bytes(),
                    stream.as_raw(),
                )
            },
            "glm state zero",
        )?;
        let mut core = device.alloc(8192 * 2)?;
        let mut output = device.alloc(8192 * 2)?;
        for _ in 0..2 {
            check(
                unsafe {
                    g::hayate_glm_kda(
                        core.as_mut_ptr(),
                        state.as_mut_ptr(),
                        q.as_ptr(),
                        k.as_ptr(),
                        v.as_ptr(),
                        decay.as_ptr(),
                        beta.as_ptr(),
                        1,
                        64,
                        128,
                        stream.as_raw(),
                    )
                },
                "glm KDA reference",
            )?;
            check(
                unsafe {
                    g::hayate_glm_gated(
                        output.as_mut_ptr(),
                        core.as_ptr(),
                        gate.as_ptr(),
                        norm.as_ptr(),
                        64,
                        128,
                        stream.as_raw(),
                    )
                },
                "glm KDA norm reference",
            )?;
            let mut expected = vec![0; output.bytes()];
            stream.copy_d2h(&mut expected, &output)?;
            check(
                unsafe {
                    g::hayate_glm_kda_gated(
                        core.as_mut_ptr(),
                        output.as_mut_ptr(),
                        second_state.as_mut_ptr(),
                        q.as_ptr(),
                        k.as_ptr(),
                        v.as_ptr(),
                        decay.as_ptr(),
                        beta.as_ptr(),
                        gate.as_ptr(),
                        norm.as_ptr(),
                        stream.as_raw(),
                    )
                },
                "glm KDA fused",
            )?;
            let mut actual = vec![0; output.bytes()];
            stream.copy_d2h(&mut actual, &output)?;
            assert_eq!(actual, expected);
            let mut a = vec![0; state.bytes()];
            let mut b = vec![0; state.bytes()];
            stream.copy_d2h(&mut a, &state)?;
            stream.copy_d2h(&mut b, &second_state)?;
            assert_eq!(a, b);
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod speculation_tests {
    use super::ffi as g;
    use crate::runtime::{Device, check};
    use hayate_core::{QuantSpec, Result};
    fn bf(x: f32) -> [u8; 2] {
        ((super::cpu::bf16(x).to_bits() >> 16) as u16).to_le_bytes()
    }
    #[test]
    #[ignore = "requires sm_120 GPU; no model checkpoint"]
    fn glm53_gpu_prefix_restore_and_partial_pool() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let bytes = 299usize;
        let mut snapshots = device.alloc(5 * bytes)?;
        let raw: Vec<_> = (0..5 * bytes)
            .map(|i| ((i * 17 + i / bytes * 13) % 251) as u8)
            .collect();
        stream.copy_h2d(&mut snapshots, &raw)?;
        let restored = device.alloc(bytes)?;
        let hidden = device.alloc(bytes)?;
        let mut result = device.alloc(20)?;
        let mut pool = device.alloc(4 * bytes)?;
        for start in 0..8 {
            for n in 1..=3 {
                for accepted in 0..=n {
                    stream.copy_h2d(&mut result, &(accepted as u32).to_le_bytes())?;
                    stream.copy_h2d(&mut pool, &vec![0xa5; 4 * bytes])?;
                    for (out, add) in [(&restored, 1), (&hidden, 0)] {
                        check(
                            unsafe {
                                g::hayate_glm_restore_prefix(
                                    out.as_ptr().cast_mut(),
                                    snapshots.as_ptr(),
                                    bytes,
                                    result.as_ptr(),
                                    add,
                                    stream.as_raw(),
                                )
                            },
                            "glm device restore test",
                        )?;
                    }
                    for row in 0..=n {
                        let dst = unsafe {
                            pool.as_ptr()
                                .cast::<u8>()
                                .add((start + row) % 4 * bytes)
                                .cast_mut()
                                .cast()
                        };
                        let src =
                            unsafe { snapshots.as_ptr().cast::<u8>().add(row * bytes).cast() };
                        check(
                            unsafe {
                                g::hayate_glm_copy_if(
                                    dst,
                                    src,
                                    bytes,
                                    result.as_ptr(),
                                    row as u32,
                                    stream.as_raw(),
                                )
                            },
                            "glm conditional pool test",
                        )?;
                    }
                    let mut actual = vec![0; bytes];
                    for (out, slot) in [(&restored, accepted + 1), (&hidden, accepted)] {
                        stream.copy_d2h(&mut actual, out)?;
                        assert_eq!(actual, raw[slot * bytes..(slot + 1) * bytes]);
                    }
                    let mut expected = vec![0xa5; 4 * bytes];
                    for row in 0..=accepted {
                        let offset = (start + row) % 4 * bytes;
                        expected[offset..offset + bytes]
                            .copy_from_slice(&raw[row * bytes..(row + 1) * bytes]);
                    }
                    let mut actual = vec![0; 4 * bytes];
                    stream.copy_d2h(&mut actual, &pool)?;
                    assert_eq!(actual, expected, "start={start} n={n} accepted={accepted}");
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires sm_120 GPU; no model checkpoint"]
    fn glm53_verification_batch_dense_bits_and_gpu_acceptance() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        for (n, k) in [(129usize, 80usize), (17, 4096)] {
            for format in 0..=3 {
                let payload = n * k / if format == 3 { 2 } else { 1 };
                let mut raw = if format == 0 {
                    (0..n * k)
                        .flat_map(|i| bf(((i * 13 % 59) as f32 - 29.) / 71.))
                        .collect::<Vec<_>>()
                } else if format == 3 {
                    vec![0x38; QuantSpec::Nvfp4.matrix_bytes(n as u64, k as u64).unwrap() as usize]
                } else {
                    vec![0x38; payload]
                };
                if format == 3 {
                    for (i, b) in raw[..payload].iter_mut().enumerate() {
                        *b = (i * 17 % 256) as u8;
                    }
                    let end = raw.len();
                    raw[end - 4..].copy_from_slice(&0.0713f32.to_le_bytes());
                } else if format != 0 {
                    for (i, b) in raw[..payload].iter_mut().enumerate() {
                        *b = ((i * 13 % 119) as u8) | if i % 3 == 0 { 128 } else { 0 };
                    }
                    raw.extend(0.0713f32.to_le_bytes().repeat(if format == 1 {
                        n.div_ceil(128) * k.div_ceil(128)
                    } else {
                        1
                    }));
                }
                let mut weights = device.alloc(raw.len())?;
                stream.copy_h2d(&mut weights, &raw)?;
                let w = g::DenseWeight {
                    data: weights.as_ptr(),
                    scale: if format == 1 {
                        unsafe { weights.as_ptr().cast::<u8>().add(payload).cast() }
                    } else {
                        std::ptr::null()
                    },
                    format,
                };
                for rows in 1..=4 {
                    let mut input = device.alloc(rows * k * 2)?;
                    stream.copy_h2d(
                        &mut input,
                        &(0..rows * k)
                            .flat_map(|i| bf(((i * 31 % 127) as f32 - 63.) / 19.))
                            .collect::<Vec<_>>(),
                    )?;
                    for rounded in [false, true] {
                        let output = device.alloc(rows * n * 2)?;
                        let expected = device.alloc(rows * n * 2)?;
                        check(
                            unsafe {
                                g::hayate_glm_dense_batch(
                                    output.as_ptr().cast_mut(),
                                    input.as_ptr(),
                                    w,
                                    rows as u32,
                                    n as u32,
                                    k as u32,
                                    rounded as i32,
                                    stream.as_raw(),
                                )
                            },
                            "glm batch dense test",
                        )?;
                        for row in 0..rows {
                            let out = unsafe {
                                expected
                                    .as_ptr()
                                    .cast::<u8>()
                                    .add(row * n * 2)
                                    .cast_mut()
                                    .cast()
                            };
                            let x = unsafe { input.as_ptr().cast::<u8>().add(row * k * 2).cast() };
                            check(
                                unsafe {
                                    if rounded {
                                        g::hayate_glm_dense(
                                            out,
                                            x,
                                            w,
                                            n as u32,
                                            k as u32,
                                            stream.as_raw(),
                                        )
                                    } else {
                                        g::hayate_glm_dense_batch(
                                            out,
                                            x,
                                            w,
                                            1,
                                            n as u32,
                                            k as u32,
                                            0,
                                            stream.as_raw(),
                                        )
                                    }
                                },
                                "glm single dense test",
                            )?;
                        }
                        let mut a = vec![0; output.bytes()];
                        let mut b = vec![0; expected.bytes()];
                        stream.copy_d2h(&mut a, &output)?;
                        stream.copy_d2h(&mut b, &expected)?;
                        assert_eq!(
                            a, b,
                            "format={format}, rows={rows}, rounded={rounded}, n={n}, k={k}"
                        );
                    }
                }
            }
        }
        let drafts = device.alloc(12)?;
        let next = device.alloc(4)?;
        let mut predictions = device.alloc(16)?;
        let output = device.alloc(20)?;
        for n in 1..=3 {
            for a in 0..=n {
                let expected = [7u32, 11, 17, 29];
                let mut draft = expected[..3].to_vec();
                if a < n {
                    draft[a] += 1;
                }
                check(
                    unsafe {
                        g::hayate_glm_spec_ids(
                            next.as_ptr().cast_mut(),
                            drafts.as_ptr().cast_mut(),
                            101,
                            draft[0],
                            draft[1],
                            draft[2],
                            1,
                            stream.as_raw(),
                        )
                    },
                    "glm forced IDs test",
                )?;
                check(
                    unsafe {
                        g::hayate_glm_spec_ids(
                            next.as_ptr().cast_mut(),
                            std::ptr::null_mut(),
                            102,
                            0,
                            0,
                            0,
                            0,
                            stream.as_raw(),
                        )
                    },
                    "glm anchor IDs test",
                )?;
                stream.copy_h2d(
                    &mut predictions,
                    &expected
                        .iter()
                        .flat_map(|t| t.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                check(
                    unsafe {
                        g::hayate_glm_accept(
                            output.as_ptr().cast_mut(),
                            drafts.as_ptr(),
                            predictions.as_ptr(),
                            n as u32,
                            n as u32,
                            stream.as_raw(),
                        )
                    },
                    "glm acceptance test",
                )?;
                let mut raw = [0; 20];
                stream.copy_d2h(&mut raw, &output)?;
                let got: Vec<_> = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                assert_eq!(got[0], a as u32);
                assert_eq!(&got[1..a + 2], &expected[..a + 1]);
                let mut anchor = [0; 4];
                stream.copy_d2h(&mut anchor, &next)?;
                assert_eq!(u32::from_le_bytes(anchor), 102);
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "glm53_prefill_tests.rs"]
mod prefill_tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "glm53_decode_bench.rs"]
mod decode_bench;

#[cfg(all(test, feature = "cuda"))]
mod mla_tests {
    use super::cpu::bf16;
    use super::ffi as g;
    use crate::runtime::{Device, check};
    use hayate_core::Result;

    fn bits(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((bf16(*x).to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
    fn values(raw: &[u8]) -> Vec<f32> {
        raw.as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
            .collect()
    }
    fn noise(n: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                bf16(((s % 2001) as f32 / 1000. - 1.) * scale)
            })
            .collect()
    }
    /// Reference top-k selection: top-512 pools (ties lowest index), then tail.
    fn reference_select(scores: &[f32], length: usize) -> Vec<i32> {
        let np = length / 4;
        let mut order: Vec<usize> = (0..np).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        order.truncate(512);
        order.sort();
        let mut ids: Vec<i32> = order
            .iter()
            .flat_map(|&p| (0..4).map(move |c| (p * 4 + c) as i32))
            .collect();
        ids.extend((np * 4..length).map(|c| c as i32));
        ids.resize(2051, -1);
        ids
    }

    #[test]
    #[ignore = "requires supported CUDA GPU; MLA select/attention/absorb vs CPU"]
    fn glm53_mla_kernels_match_reference() -> Result<()> {
        let device = Device::select()?;
        let st = device.stream()?;
        let (start, rows, context) = (2997usize, 3usize, 3000usize);
        let stride = context.div_ceil(4);
        // Index inputs with deliberate ties.
        let q = noise(rows * 4096, 7, 1.);
        let pools = noise(stride * 128, 11, 1.);
        let mut pools_raw = bits(&pools);
        // Duplicate pool 5 into 9 and 700 for exact score ties.
        let copy: Vec<u8> = pools_raw[5 * 256..6 * 256].to_vec();
        pools_raw[9 * 256..10 * 256].copy_from_slice(&copy);
        pools_raw[700 * 256..701 * 256].copy_from_slice(&copy);
        let w = noise(rows * 32, 13, 1.);
        let mut dq = device.alloc(rows * 8192)?;
        let mut dp = device.alloc(pools_raw.len())?;
        let mut dw = device.alloc(rows * 64)?;
        st.copy_h2d(&mut dq, &bits(&q))?;
        st.copy_h2d(&mut dp, &pools_raw)?;
        st.copy_h2d(&mut dw, &bits(&w))?;
        let mut ids = device.alloc(rows * 2051 * 4)?;
        let mut scores = device.alloc(rows * stride * 4)?;
        check(
            unsafe {
                g::hayate_glm_mla_select(
                    ids.as_mut_ptr(),
                    scores.as_mut_ptr(),
                    dq.as_ptr(),
                    dp.as_ptr(),
                    dw.as_ptr(),
                    start as u32,
                    rows as u32,
                    stride,
                    st.as_raw(),
                )
            },
            "mla select",
        )?;
        let mut raw = vec![0; rows * 2051 * 4];
        st.copy_d2h(&mut raw, &ids)?;
        let got: Vec<i32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        let mut sraw = vec![0; rows * stride * 4];
        st.copy_d2h(&mut sraw, &scores)?;
        let s: Vec<f32> = sraw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let pv = values(&pools_raw);
        for t in 0..rows {
            let length = start + t + 1;
            for p in 0..length / 4 {
                let mut sum = 0f32;
                for h in 0..32 {
                    let mut dot = 0f32;
                    for c in 0..128 {
                        dot += q[t * 4096 + h * 128 + c] * pv[p * 128 + c];
                    }
                    sum += (dot / 128f32.sqrt()).max(0.) * w[t * 32 + h] / 32f32.sqrt();
                }
                assert!(
                    (sum - s[t * stride + p]).abs() <= 1e-3 * (1. + sum.abs()),
                    "score t={t} p={p}"
                );
            }
            assert_eq!(
                &got[t * 2051..(t + 1) * 2051],
                &reference_select(&s[t * stride..], length)[..],
                "selection row {t}"
            );
        }
        // Absorbed attention against an F64 reference on the same selection.
        let qabs = noise(64 * rows * 512, 17, 0.25);
        let latent = noise(context * 512, 19, 1.);
        let mut dqa = device.alloc(qabs.len() * 2)?;
        let mut dl = device.alloc(latent.len() * 2)?;
        st.copy_h2d(&mut dqa, &bits(&qabs))?;
        st.copy_h2d(&mut dl, &bits(&latent))?;
        let mut reference = vec![0f64; 64 * rows * 512];
        for h in 0..64 {
            for t in 0..rows {
                let sel: Vec<usize> = got[t * 2051..(t + 1) * 2051]
                    .iter()
                    .take_while(|&&i| i >= 0)
                    .map(|&i| i as usize)
                    .collect();
                let qa = &qabs[(h * rows + t) * 512..(h * rows + t + 1) * 512];
                let logits: Vec<f64> = sel
                    .iter()
                    .map(|&j| {
                        let dot: f64 = (0..512)
                            .map(|c| qa[c] as f64 * latent[j * 512 + c] as f64)
                            .sum();
                        dot / 16.
                    })
                    .collect();
                let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for (k, &j) in sel.iter().enumerate() {
                    for c in 0..512 {
                        reference[(h * rows + t) * 512 + c] +=
                            e[k] / z * latent[j * 512 + c] as f64;
                    }
                }
            }
        }
        let splits = 8;
        let mut out = device.alloc(64 * rows * 512 * 2)?;
        let mut part = device.alloc(splits * 64 * rows * 512 * 4)?;
        let mut stats = device.alloc(splits * 64 * rows * 8)?;
        for split in [1u32, splits as u32] {
            check(
                unsafe {
                    g::hayate_glm_mla_attention(
                        out.as_mut_ptr(),
                        part.as_mut_ptr(),
                        stats.as_mut_ptr(),
                        dqa.as_ptr(),
                        dl.as_ptr(),
                        ids.as_ptr(),
                        rows as u32,
                        split,
                        st.as_raw(),
                    )
                },
                "mla attention",
            )?;
            let mut o = vec![0; 64 * rows * 512 * 2];
            st.copy_d2h(&mut o, &out)?;
            let actual: Vec<f64> = values(&o).into_iter().map(f64::from).collect();
            let error: f64 = actual
                .iter()
                .zip(&reference)
                .map(|(a, r)| (a - r).powi(2))
                .sum();
            let energy: f64 = reference.iter().map(|r| r * r).sum();
            let rel = (error / energy).sqrt();
            println!("mla attention splits={split} relative_l2={rel:.6}");
            assert!(rel < 2e-2, "attention splits={split} rel={rel}");
        }
        // Absorb GEMVs (1..4 rows) against CPU.
        let wkt = noise(64 * 512 * 256, 23, 0.05);
        let wv = noise(64 * 256 * 512, 29, 0.05);
        let mut dwkt = device.alloc(wkt.len() * 2)?;
        let mut dwv = device.alloc(wv.len() * 2)?;
        st.copy_h2d(&mut dwkt, &bits(&wkt))?;
        st.copy_h2d(&mut dwv, &bits(&wv))?;
        for m in 1..=4usize {
            let qm = noise(m * 16384, 31 + m as u32, 1.);
            let mut dqm = device.alloc(m * 16384 * 2)?;
            st.copy_h2d(&mut dqm, &bits(&qm))?;
            let mut qa = device.alloc(64 * m * 512 * 2)?;
            check(
                unsafe {
                    g::hayate_glm_mla_absorb_q(
                        qa.as_mut_ptr(),
                        dqm.as_ptr(),
                        dwkt.as_ptr(),
                        m as u32,
                        st.as_raw(),
                    )
                },
                "absorb q",
            )?;
            let mut raw = vec![0; 64 * m * 512 * 2];
            st.copy_d2h(&mut raw, &qa)?;
            let got = values(&raw);
            for h in [0usize, 63] {
                for r in 0..m {
                    for c in [0usize, 511] {
                        let want: f32 = (0..256)
                            .map(|d| qm[r * 16384 + h * 256 + d] * wkt[(h * 512 + c) * 256 + d])
                            .sum();
                        let have = got[(h * m + r) * 512 + c];
                        assert!(
                            (have - want).abs() <= 1e-2 * (1. + want.abs()),
                            "absorb q m={m}"
                        );
                    }
                }
            }
            let mut o = device.alloc(m * 16384 * 2)?;
            check(
                unsafe {
                    g::hayate_glm_mla_absorb_out(
                        o.as_mut_ptr(),
                        qa.as_ptr(),
                        dwv.as_ptr(),
                        m as u32,
                        st.as_raw(),
                    )
                },
                "absorb out",
            )?;
            let mut raw_o = vec![0; m * 16384 * 2];
            st.copy_d2h(&mut raw_o, &o)?;
            let out_v = values(&raw_o);
            for h in [0usize, 63] {
                for r in 0..m {
                    for e in [0usize, 255] {
                        let want: f32 = (0..512)
                            .map(|c| got[(h * m + r) * 512 + c] * wv[(h * 256 + e) * 512 + c])
                            .sum();
                        let have = out_v[(r * 64 + h) * 256 + e];
                        assert!(
                            (have - want).abs() <= 1e-2 * (1. + want.abs()),
                            "absorb out m={m}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

/// LOWBIT mixed-expert decode and lossless prefill transcode. Payloads have no
/// header; metadata lives beside the eight device image pointers. Each matrix
/// starts at a 256-byte boundary, in gate/up/down order.
/// LOWBIT uses the NVFP4 lane/FMA order, including `scale * tensor` before the
/// weight multiply, so transcoding does not change decode rounding.
/// This deliberately uses the existing NVFP4 order rather than accumulating
/// unscaled group sums first. Activations remain BF16 throughout.
pub mod lowbit {
    use hayate_core::{Error, Result};

    /// `format`: 0 NVFP4, 1 three-bit, 2 two-bit, 3 EXL3. `grid`: GLM53_QUANT §3
    /// (EXL3: k2, twice the bits per weight). NVFP4 uses zero grid/group;
    /// LOWBIT uses group 16 or 32; EXL3 the codebook (2 = mul1). Reserved is zero.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MatrixFormat {
        pub format: u8,
        pub grid: u8,
        pub group: u8,
        pub reserved: u8,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct ExpertFormat {
        pub gate_up: MatrixFormat,
        pub down: MatrixFormat,
    }
    impl MatrixFormat {
        pub const NVFP4: Self = Self {
            format: 0,
            grid: 0,
            group: 0,
            reserved: 0,
        };
        pub fn new(grid: u8, group: u8) -> Result<Self> {
            let f = Self {
                format: if matches!(grid, 2 | 3) { 1 } else { 2 },
                grid,
                group,
                reserved: 0,
            };
            f.validate()?;
            Ok(f)
        }
        /// EXL3 trellis matrix (crates/models/src/glm53/EXL3.md).
        pub fn exl3(k2: u8, cb: u8) -> Result<Self> {
            let f = Self {
                format: 3,
                grid: k2,
                group: cb,
                reserved: 0,
            };
            f.validate()?;
            Ok(f)
        }
        pub fn is_exl3(self) -> bool {
            self.format == 3
        }
        fn validate(self) -> Result<()> {
            if self.format == 3 {
                return if self.reserved == 0
                    && hayate_core::QuantSpec::exl3_valid(self.grid, self.group)
                {
                    Ok(())
                } else {
                    Err(Error::InvalidShape("invalid GLM EXL3 format".into()))
                };
            }
            if self == Self::NVFP4
                || (self.reserved == 0
                    && self.grid <= 4
                    && matches!(self.group, 16 | 32)
                    && self.format == if matches!(self.grid, 2 | 3) { 1 } else { 2 })
            {
                Ok(())
            } else {
                Err(Error::InvalidShape("invalid GLM LOWBIT format".into()))
            }
        }
        /// Unpadded payload bytes, including scales and the FP32 tensor scale.
        pub fn storage_bytes(self, rows: u32, cols: u32) -> Result<usize> {
            self.validate()?;
            if self.format == 3 {
                return hayate_core::QuantSpec::Exl3 {
                    k2: self.grid,
                    cb: self.group,
                }
                .matrix_bytes(u64::from(rows), u64::from(cols))
                .and_then(|b| usize::try_from(b).ok())
                .ok_or_else(|| {
                    Error::InvalidShape("EXL3 needs 128-multiple rows and cols".into())
                });
            }
            if rows == 0 || cols == 0 || !cols.is_multiple_of(256) {
                return Err(Error::InvalidShape(
                    "LOWBIT needs R > 0, C % 256 == 0".into(),
                ));
            }
            let n = u64::from(rows);
            let k = u64::from(cols);
            let bytes = if self.format == 0 {
                n * k / 2 + n.div_ceil(128) * 128 * (k / 16) + 4
            } else {
                n * k / 8 * if self.format == 1 { 3 } else { 2 } + n * k / u64::from(self.group) + 4
            };
            usize::try_from(bytes)
                .ok()
                .filter(|&v| v <= isize::MAX as usize)
                .ok_or_else(|| Error::InvalidShape("LOWBIT image too large".into()))
        }
        /// Slot stride of one matrix. Expert bytes = 2*gate_stride + down_stride.
        pub fn stride(self, rows: u32, cols: u32) -> Result<usize> {
            let bytes = self.storage_bytes(rows, cols)?;
            Ok(bytes.div_ceil(256) * 256)
        }
    }

    #[cfg(feature = "cuda")]
    pub mod ffi {
        use super::{ExpertFormat, MatrixFormat};
        use crate::{cuda::ffi::Stream, glm53::ffi::DenseWeight};
        use std::ffi::c_void;
        // Output buffers are disjoint from all inputs and live through completion.
        // `images` is a device array of eight pointers (null = CPU miss);
        // `formats` is a device array of eight ExpertFormat entries (64 bytes),
        // gathered in the same route order. Entries must match actual image bytes.
        // These calls allocate nothing and do not synchronize; graph capture is valid.
        unsafe extern "C" {
            pub fn hayate_glm_lowbit_convert(
                out: *mut c_void,
                image: *const c_void,
                n: u32,
                k: u32,
                grid: u32,
                group: u32,
                st: Stream,
            ) -> i32;
            /// One launch for gate/up/down. Source and destination use 256-byte strides.
            pub fn hayate_glm_lowbit_convert_expert(
                out: *mut c_void,
                image: *const c_void,
                format: ExpertFormat,
                hidden: u32,
                intermediate: u32,
                st: Stream,
            ) -> i32;
            pub fn hayate_glm_lowbit_gemv(
                out: *mut c_void,
                x: *const c_void,
                image: *const c_void,
                format: MatrixFormat,
                m: u32,
                n: u32,
                k: u32,
                st: Stream,
            ) -> i32;
            /// x/original [M,H], out [9,M,I], H and I multiples of 256.
            /// The same eight selected experts process all M input rows.
            /// Null image: gate/up writes zero.
            /// Shared uses original (before optional routed input rotation).
            pub fn hayate_glm_lowbit_gate_up(
                out: *mut c_void,
                x: *const c_void,
                original: *const c_void,
                images: *const c_void,
                formats: *const ExpertFormat,
                gate: DenseWeight,
                up: DenseWeight,
                m: u32,
                hidden: u32,
                intermediate: u32,
                st: Stream,
            ) -> i32;
            /// x [9,M,I], out [9,M,H]. Null image: leaves output untouched
            /// for the CPU mailbox merge, as in the existing MoE decode.
            pub fn hayate_glm_lowbit_down(
                out: *mut c_void,
                x: *const c_void,
                images: *const c_void,
                formats: *const ExpertFormat,
                shared: DenseWeight,
                m: u32,
                hidden: u32,
                intermediate: u32,
                st: Stream,
            ) -> i32;
        }
    }

    /// Checked synchronous wrappers. The raw FFI above is the graph/decode path.
    #[cfg(feature = "cuda")]
    mod checked {
        use super::{ExpertFormat, MatrixFormat, ffi};
        use crate::runtime::{DeviceBuffer, Stream, check};
        use hayate_core::{Error, Result};

        fn size(buffer: &DeviceBuffer, bytes: u64) -> Result<()> {
            if buffer.bytes() as u64 >= bytes {
                Ok(())
            } else {
                Err(Error::InvalidShape("short LOWBIT device buffer".into()))
            }
        }
        /// Lossless conversion; output includes zero-filled NVFP4 scale padding.
        pub fn to_nvfp4(
            out: &mut DeviceBuffer,
            image: &DeviceBuffer,
            rows: u32,
            cols: u32,
            format: MatrixFormat,
            stream: &Stream,
        ) -> Result<()> {
            if format.format == 0 || format.is_exl3() {
                return Err(Error::InvalidShape("transcode needs LOWBIT input".into()));
            }
            size(image, format.storage_bytes(rows, cols)? as u64)?;
            size(out, MatrixFormat::NVFP4.storage_bytes(rows, cols)? as u64)?;
            let launch = check(
                unsafe {
                    ffi::hayate_glm_lowbit_convert(
                        out.as_mut_ptr(),
                        image.as_ptr(),
                        rows,
                        cols,
                        format.grid.into(),
                        format.group.into(),
                        stream.as_raw(),
                    )
                },
                "hayate_glm_lowbit_convert",
            );
            let sync = stream.synchronize();
            launch.and(sync)
        }
        /// One expert -> three NVFP4 matrix slots, including zeroed slot padding.
        pub fn expert_to_nvfp4(
            out: &mut DeviceBuffer,
            image: &DeviceBuffer,
            format: ExpertFormat,
            hidden: u32,
            intermediate: u32,
            stream: &Stream,
        ) -> Result<()> {
            let input_bytes = (format.gate_up.stride(intermediate, hidden)? as u64)
                .saturating_mul(2)
                .saturating_add(format.down.stride(hidden, intermediate)? as u64);
            let output_bytes =
                (MatrixFormat::NVFP4.stride(intermediate, hidden)? as u64).saturating_mul(3);
            size(image, input_bytes)?;
            size(out, output_bytes)?;
            let launch = check(
                unsafe {
                    ffi::hayate_glm_lowbit_convert_expert(
                        out.as_mut_ptr(),
                        image.as_ptr(),
                        format,
                        hidden,
                        intermediate,
                        stream.as_raw(),
                    )
                },
                "hayate_glm_lowbit_convert_expert",
            );
            let sync = stream.synchronize();
            launch.and(sync)
        }
        /// BF16 [M,K] -> [M,N], M=1..4. No BF16 rounding of routed weights.
        #[allow(clippy::too_many_arguments)]
        pub fn gemv(
            out: &mut DeviceBuffer,
            x: &DeviceBuffer,
            image: &DeviceBuffer,
            format: MatrixFormat,
            m: u32,
            n: u32,
            k: u32,
            stream: &Stream,
        ) -> Result<()> {
            if !(1..=4).contains(&m) {
                return Err(Error::InvalidShape("LOWBIT GEMV needs M=1..4".into()));
            }
            size(image, format.storage_bytes(n, k)? as u64)?;
            size(x, u64::from(m) * u64::from(k) * 2)?;
            size(out, u64::from(m) * u64::from(n) * 2)?;
            let launch = check(
                unsafe {
                    ffi::hayate_glm_lowbit_gemv(
                        out.as_mut_ptr(),
                        x.as_ptr(),
                        image.as_ptr(),
                        format,
                        m,
                        n,
                        k,
                        stream.as_raw(),
                    )
                },
                "hayate_glm_lowbit_gemv",
            );
            let sync = stream.synchronize();
            launch.and(sync)
        }
    }
    #[cfg(feature = "cuda")]
    pub use checked::{expert_to_nvfp4, gemv, to_nvfp4};
}

/// Packed prefill MoE: one launch per projection for all routed experts.
/// Weights stay packed; MMA uses unscaled BF16 weights and FP32 accumulators.
/// The matrix scale is applied before rounding to BF16, followed by the existing
/// clamped SwiGLU (including its intermediate BF16 rounding).
pub mod packed_grouped {
    #[cfg(any(test, feature = "cuda"))]
    use super::lowbit::ExpertFormat;
    #[cfg(any(test, feature = "cuda"))]
    use hayate_core::{Error, Result};

    #[cfg(any(test, feature = "cuda"))]
    pub(super) fn validate(
        formats: &[ExpertFormat],
        ranges: &[[u32; 2]],
        hidden: u32,
        intermediate: u32,
    ) -> Result<u32> {
        if formats.is_empty()
            || formats.len() > 288
            || formats.len() != ranges.len()
            || hidden == 0
            || hidden > 4096
            || !hidden.is_multiple_of(256)
            || intermediate == 0
            || intermediate > 4096
            || !intermediate.is_multiple_of(256)
        {
            return Err(Error::InvalidShape("invalid packed grouped shape".into()));
        }
        let mut rows = 0u32;
        for (f, &[count, offset]) in formats.iter().zip(ranges) {
            f.gate_up.storage_bytes(intermediate, hidden)?;
            f.down.storage_bytes(hidden, intermediate)?;
            if offset != rows || count > 65536 - rows {
                return Err(Error::InvalidShape(
                    "packed ranges must partition at most 65536 rows".into(),
                ));
            }
            rows += count;
        }
        Ok(rows)
    }

    #[cfg(feature = "cuda")]
    pub mod ffi {
        use super::ExpertFormat;
        use crate::cuda::ffi::Stream;
        use std::ffi::c_void;

        // Asynchronous, allocation-free, graph-capturable. No scratch needed.
        // Warm up each selected configuration before capture (shared-memory
        // attributes/module initialization); config 0 is the default.
        // Device metadata: images[count] pointers (non-null for nonempty experts),
        // formats[count] (8 bytes each), ranges[count][2] = {rows, first_row}.
        // count=1..288; H/I=256..4096, multiples of 256; total rows <= 65536.
        // Ranges partition contiguous rows in expert order; zero rows are valid.
        // Format descriptors must be valid and match the actual image storage:
        // gate/up/down at 0, gu_stride, 2*gu_stride; each stride rounded to 256 B.
        // All pointers, metadata and images must remain valid until completion;
        // outputs must be disjoint from every input, including other images.
        // x/product inputs require 16-byte alignment (cp.async); outputs 2 bytes,
        // image bases 16 bytes, pointer metadata 8 bytes and ranges 4 bytes.
        unsafe extern "C" {
            /// Selects the launch configuration on this host thread (default 0).
            /// 0: 64x128x128, 8 warps, 2 stages; 1: 32x128x128, 8 warps,
            /// 3 stages; 2: 32x64x128, 4 warps, 2 stages; 3: 32x64x256,
            /// 4 warps, 2 stages. Invalid IDs leave the selection unchanged.
            /// No allocation, synchronization or device work. Select before
            /// graph capture; replay uses the configuration captured in nodes.
            pub fn hayate_glm_grouped_packed_set_config(config: u32) -> i32;

            /// x [rows,H] BF16 -> product [rows,I] BF16. Gate/up share x loads.
            pub fn hayate_glm_grouped_packed_gate_up(
                product: *mut c_void,
                x: *const c_void,
                images: *const c_void,
                formats: *const ExpertFormat,
                ranges: *const c_void,
                count: u32,
                hidden: u32,
                intermediate: u32,
                st: Stream,
            ) -> i32;
            /// product [rows,I] BF16 -> out [rows,H] BF16.
            pub fn hayate_glm_grouped_packed_down(
                out: *mut c_void,
                product: *const c_void,
                images: *const c_void,
                formats: *const ExpertFormat,
                ranges: *const c_void,
                count: u32,
                hidden: u32,
                intermediate: u32,
                st: Stream,
            ) -> i32;
        }
    }

    #[cfg(feature = "cuda")]
    mod checked {
        use super::{ExpertFormat, ffi, validate};
        use crate::runtime::{Device, DeviceBuffer, Stream, check};
        use hayate_core::{Error, Result};

        #[allow(clippy::too_many_arguments)]
        fn run(
            down: bool,
            device: &Device,
            out: &mut DeviceBuffer,
            x: &DeviceBuffer,
            images: &[DeviceBuffer],
            formats: &[ExpertFormat],
            ranges: &[[u32; 2]],
            hidden: u32,
            intermediate: u32,
            stream: &Stream,
        ) -> Result<()> {
            let rows = u64::from(validate(formats, ranges, hidden, intermediate)?);
            let (n, k) = if down {
                (hidden, intermediate)
            } else {
                (intermediate, hidden)
            };
            if images.len() != formats.len()
                || !(x.as_ptr() as usize).is_multiple_of(16)
                || !(out.as_ptr() as usize).is_multiple_of(2)
                || (out.bytes() as u64) < rows * u64::from(n) * 2
                || (x.bytes() as u64) < rows * u64::from(k) * 2
            {
                return Err(Error::InvalidShape("short packed grouped buffer".into()));
            }
            for (image, f) in images.iter().zip(formats) {
                let bytes = 2 * f.gate_up.stride(intermediate, hidden)?
                    + f.down.stride(hidden, intermediate)?;
                if image.bytes() < bytes || !(image.as_ptr() as usize).is_multiple_of(16) {
                    return Err(Error::InvalidShape(
                        "short or unaligned packed expert image".into(),
                    ));
                }
            }
            let ptrs: Vec<_> = images
                .iter()
                .flat_map(|image| (image.as_ptr() as usize).to_ne_bytes())
                .collect();
            let descriptors: Vec<_> = formats
                .iter()
                .flat_map(|f| {
                    [f.gate_up, f.down]
                        .into_iter()
                        .flat_map(|m| [m.format, m.grid, m.group, m.reserved])
                })
                .collect();
            let ranges: Vec<_> = ranges
                .iter()
                .flatten()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            let mut pointers = device.alloc(ptrs.len())?;
            let mut device_formats = device.alloc(descriptors.len())?;
            let mut device_ranges = device.alloc(ranges.len())?;
            stream.copy_h2d(&mut pointers, &ptrs)?;
            stream.copy_h2d(&mut device_formats, &descriptors)?;
            stream.copy_h2d(&mut device_ranges, &ranges)?;
            let launch = if down {
                ffi::hayate_glm_grouped_packed_down
            } else {
                ffi::hayate_glm_grouped_packed_gate_up
            };
            let result = check(
                unsafe {
                    launch(
                        out.as_mut_ptr(),
                        x.as_ptr(),
                        pointers.as_ptr(),
                        device_formats.as_ptr().cast(),
                        device_ranges.as_ptr(),
                        formats.len() as u32,
                        hidden,
                        intermediate,
                        stream.as_raw(),
                    )
                },
                "GLM packed grouped",
            );
            let sync = stream.synchronize();
            result.and(sync)
        }

        /// Checked synchronous convenience wrapper; allocates/uploads the three
        /// small metadata arrays. The raw FFI is the prefill hot path.
        #[allow(clippy::too_many_arguments)]
        pub fn gate_up(
            device: &Device,
            product: &mut DeviceBuffer,
            x: &DeviceBuffer,
            images: &[DeviceBuffer],
            formats: &[ExpertFormat],
            ranges: &[[u32; 2]],
            hidden: u32,
            intermediate: u32,
            stream: &Stream,
        ) -> Result<()> {
            run(
                false,
                device,
                product,
                x,
                images,
                formats,
                ranges,
                hidden,
                intermediate,
                stream,
            )
        }

        /// Checked synchronous down projection; see `gate_up` for metadata cost.
        #[allow(clippy::too_many_arguments)]
        pub fn down(
            device: &Device,
            out: &mut DeviceBuffer,
            product: &DeviceBuffer,
            images: &[DeviceBuffer],
            formats: &[ExpertFormat],
            ranges: &[[u32; 2]],
            hidden: u32,
            intermediate: u32,
            stream: &Stream,
        ) -> Result<()> {
            run(
                true,
                device,
                out,
                product,
                images,
                formats,
                ranges,
                hidden,
                intermediate,
                stream,
            )
        }
    }
    #[cfg(feature = "cuda")]
    pub use checked::{down, gate_up};
}

#[cfg(test)]
#[path = "glm53_packed_grouped_tests.rs"]
mod packed_grouped_tests;
