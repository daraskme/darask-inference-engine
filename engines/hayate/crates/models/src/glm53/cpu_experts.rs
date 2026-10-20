#[cfg(test)]
use hayate_core::QuantSpec;
use hayate_core::{Error, Result};
#[cfg(test)]
use hayate_kernels::glm53::cpu::swiglu;
use hayate_kernels::glm53::cpu::{
    Format, Matrix, Pool, Segment, bf16, deinterleave, deinterleave_gpu,
};

/// Error diffusion keeps the requested PCIe fraction across layers/tokens,
/// including layers with only one miss. Resident hits never consume credit.
#[derive(Debug)]
pub(super) struct Split {
    fraction: f64,
    credit: f64,
}
impl Split {
    pub fn new(fraction: f64) -> Result<Self> {
        if !(0. ..=1.).contains(&fraction) {
            return Err(Error::InvalidPlan(
                "GLM pcie-frac must be finite and in 0..=1".into(),
            ));
        }
        Ok(Self {
            fraction,
            credit: 0.,
        })
    }
    pub fn pcie(&mut self) -> bool {
        self.credit += self.fraction;
        if self.credit >= 1. {
            self.credit -= 1.;
            true
        } else {
            false
        }
    }
}

/// One routed miss: a host expert image, its BF16 input rows and the BF16
/// output rows (same row count, at most four). `expert` (layer, expert)
/// selects the routed-expert LoRA; it is required while one is loaded.
#[derive(Debug)]
pub(super) struct Miss<'a> {
    pub image: &'a [u8],
    pub input: &'a [u8],
    pub output: &'a mut [u8],
    pub expert: Option<(usize, usize)>,
}

#[derive(Debug)]
pub(super) struct CpuExperts {
    pool: Pool,
    gpu_order: bool,
    x: Vec<f32>,
    xd: Vec<f32>,
    act: Vec<f32>,
    actd: Vec<f32>,
    output: Vec<f32>,
    /// EXL3 misses: the gate's and up's rotated inputs H(x * suh) and the
    /// down's rotated activation (`hayate_kernels::glm53::exl3`).
    xg: Vec<f32>,
    xu: Vec<f32>,
    ad: Vec<f32>,
    /// Routed-expert LoRA, added to the F32 down output before its BF16 rounding.
    pub lora: Option<std::sync::Arc<super::lora::HostExperts>>,
}
impl CpuExperts {
    #[cfg(test)]
    pub fn new() -> Result<Self> {
        Self::with_threads(8, false)
    }
    pub fn with_threads(threads: usize, spin: bool) -> Result<Self> {
        if !(1..=8).contains(&threads) {
            return Err(Error::InvalidPlan("GLM cpu-threads must be 1..8".into()));
        }
        Ok(Self {
            pool: Pool::with_wait(&hayate_core::machine::MAIN.cpu.compute[..threads], spin)?,
            gpu_order: std::env::var("GLM53_CPU_GPU_ORDER").is_ok_and(|v| v == "1"),
            x: Vec::new(),
            xd: Vec::new(),
            act: Vec::new(),
            actd: Vec::new(),
            output: Vec::new(),
            xg: Vec::new(),
            xu: Vec::new(),
            ad: Vec::new(),
            lora: None,
        })
    }
    pub fn timing(&self) -> (f64, f64) {
        (self.pool.worker_ms, self.pool.dispatch_ms)
    }
    /// Input has already received the same BF16 Hadamard as the GPU input.
    /// Only the down activation needs another rotation for all-NVFP4 weights.
    pub fn forward(
        &mut self,
        image: &[u8],
        input: &[u8],
        out: &mut [u8],
        nv: bool,
        down_fp8: bool,
        hadamard: u32,
    ) -> Result<()> {
        self.forward_layer(
            &mut [Miss {
                image,
                input,
                output: out,
                expert: None,
            }],
            nv,
            down_fp8,
            hadamard,
        )
    }
    /// Every miss of one layer in two pool dispatches: fused gate/up/SwiGLU
    /// for all experts, then all down projections. Results are bitwise equal
    /// to one `forward` per expert (same per-row kernel and order).
    pub fn forward_layer(
        &mut self,
        misses: &mut [Miss<'_>],
        nv: bool,
        down_fp8: bool,
        hadamard: u32,
    ) -> Result<()> {
        self.forward_layer_formats(
            misses,
            &[[
                if nv { Format::Nvfp4 } else { Format::Fp8Block },
                if nv && !down_fp8 {
                    Format::Nvfp4
                } else {
                    Format::Fp8Block
                },
            ]],
            hadamard,
        )
    }
    /// Each expert's [gate/up, down] formats; one pair can be broadcast to
    /// every miss. Images concatenate the three 256-byte-aligned payloads.
    pub fn forward_layer_formats(
        &mut self,
        misses: &mut [Miss<'_>],
        formats: &[[Format; 2]],
        hadamard: u32,
    ) -> Result<()> {
        if formats.len() != 1 && formats.len() != misses.len() {
            return Err(Error::InvalidShape("GLM CPU expert formats".into()));
        }
        let format = |i: usize| formats[if formats.len() == 1 { 0 } else { i }];
        let bytes = |f: Format| f.matrix_bytes(2048, 4096);
        let mut total = 0;
        for (i, m) in misses.iter().enumerate() {
            let rows = m.input.len() / 8192;
            if !(1..=4).contains(&rows)
                || m.input.len() != rows * 8192
                || m.output.len() != rows * 8192
            {
                return Err(Error::InvalidShape("GLM CPU expert activation".into()));
            }
            let [gate, down] = format(i);
            let size = bytes(gate).zip(bytes(down)).and_then(|(g, d)| {
                g.checked_next_multiple_of(256)?
                    .checked_mul(2)?
                    .checked_add(d)
            });
            if size.is_none_or(|size| m.image.len() < size) {
                return Err(Error::InvalidShape("GLM CPU expert image".into()));
            }
            total += rows;
        }
        if misses.is_empty() {
            return Ok(());
        }
        for buffer in [&mut self.x, &mut self.xd, &mut self.output] {
            buffer.resize(total * 4096, 0.);
        }
        for buffer in [&mut self.act, &mut self.actd] {
            buffer.resize(total * 2048, 0.);
        }
        // Decode misses share one input; convert and reorder it only once.
        let mut at = 0;
        let mut previous: Option<(*const u8, usize, usize)> = None;
        for m in misses.iter() {
            let len = m.input.len() / 2;
            match previous {
                Some((ptr, from, n)) if ptr == m.input.as_ptr() && n == len => {
                    self.x.copy_within(from..from + len, at);
                    self.xd.copy_within(from..from + len, at);
                }
                _ => {
                    for (v, b) in self.x[at..at + len]
                        .iter_mut()
                        .zip(m.input.as_chunks::<2>().0)
                    {
                        *v = f32::from_bits((u16::from_le_bytes(*b) as u32) << 16);
                    }
                    if self.gpu_order {
                        deinterleave_gpu(&self.x[at..at + len], &mut self.xd[at..at + len], 4096);
                    } else {
                        deinterleave(&self.x[at..at + len], &mut self.xd[at..at + len]);
                    }
                }
            }
            previous = Some((m.input.as_ptr(), at, len));
            at += len;
        }
        // EXL3 misses read their own rotated inputs (gate and up have their own suh).
        let exl3 = |f: Format| matches!(f, Format::Exl3 { .. });
        let mut sums: Vec<[[f32; 4]; 3]> = vec![[[0.; 4]; 3]; misses.len()];
        let exl3_gate = (0..misses.len()).any(|i| exl3(format(i)[0]));
        let exl3_down = (0..misses.len()).any(|i| exl3(format(i)[1]));
        if exl3_gate {
            self.xg.resize(total * 4096, 0.);
            self.xu.resize(total * 4096, 0.);
            let mut at = 0;
            for (i, m) in misses.iter().enumerate() {
                let rows = m.input.len() / 8192;
                if let Format::Exl3 { k2 } = format(i)[0] {
                    let stride = bytes(format(i)[0]).unwrap().next_multiple_of(256);
                    for (part, (dst, slot)) in [(&mut self.xg, 0), (&mut self.xu, 1)]
                        .into_iter()
                        .enumerate()
                    {
                        let view = hayate_kernels::glm53::exl3::Exl3::new(
                            &m.image[part * stride..][..bytes(format(i)[0]).unwrap()],
                            2048,
                            4096,
                            k2,
                            2,
                        )?;
                        for (r, sum) in sums[i][slot].iter_mut().enumerate().take(rows) {
                            let span = (at + r) * 4096..(at + r + 1) * 4096;
                            *sum = view.prepare(&self.x[span.clone()], &mut dst[span]);
                        }
                    }
                }
                at += rows;
            }
        }
        {
            let mut segments = Vec::with_capacity(misses.len());
            let (mut x, mut xd) = (&self.x[..], &self.xd[..]);
            let (mut xg, mut xu) = (&self.xg[..], &self.xu[..]);
            let mut act = &mut self.act[..];
            for (i, m) in misses.iter().enumerate() {
                let [gate, _] = format(i);
                let bytes = bytes(gate).unwrap();
                let stride = bytes.next_multiple_of(256);
                let rows = m.input.len() / 8192;
                let (xs, xr) = x.split_at(rows * 4096);
                let (ds, dr) = xd.split_at(rows * 4096);
                let (ys, yr) = std::mem::take(&mut act).split_at_mut(rows * 2048);
                let gate_m = Matrix::with_format(&m.image[..bytes], 2048, 4096, gate)?;
                let up_m = Matrix::with_format(&m.image[stride..stride + bytes], 2048, 4096, gate)?;
                if exl3(gate) {
                    let (gs, gr) = xg.split_at(rows * 4096);
                    let (us, ur) = xu.split_at(rows * 4096);
                    segments.push(Segment::exl3_gate_up(
                        gate_m, up_m, gs, sums[i][0], us, sums[i][1], ys, rows,
                    )?);
                    (xg, xu) = (gr, ur);
                } else {
                    segments.push(Segment::gate_up(
                        gate_m.with_gpu_order(self.gpu_order),
                        up_m.with_gpu_order(self.gpu_order),
                        xs,
                        ds,
                        ys,
                        rows,
                    )?);
                    if exl3_gate {
                        xg = &xg[rows * 4096..];
                        xu = &xu[rows * 4096..];
                    }
                }
                (x, xd, act) = (xr, dr, yr);
            }
            self.pool.run(&segments)?;
        }
        if hadamard != 0 {
            let mut at = 0;
            for (i, m) in misses.iter().enumerate() {
                let len = m.input.len() / 8192 * 2048;
                if format(i)[1] != Format::Fp8Block && !exl3(format(i)[1]) {
                    rotate(&mut self.act[at..at + len], hadamard as usize)?;
                }
                at += len;
            }
        }
        if self.gpu_order {
            deinterleave_gpu(
                &self.act[..total * 2048],
                &mut self.actd[..total * 2048],
                2048,
            );
        } else {
            deinterleave(&self.act[..total * 2048], &mut self.actd[..total * 2048]);
        }
        if exl3_down {
            self.ad.resize(total * 2048, 0.);
            let mut at = 0;
            for (i, m) in misses.iter().enumerate() {
                let rows = m.input.len() / 8192;
                let [gate, down] = format(i);
                if let Format::Exl3 { k2 } = down {
                    let stride = bytes(gate).unwrap().next_multiple_of(256);
                    let view = hayate_kernels::glm53::exl3::Exl3::new(
                        &m.image[2 * stride..][..bytes(down).unwrap()],
                        4096,
                        2048,
                        k2,
                        2,
                    )?;
                    for (r, sum) in sums[i][2].iter_mut().enumerate().take(rows) {
                        let span = (at + r) * 2048..(at + r + 1) * 2048;
                        *sum = view.prepare(&self.act[span.clone()], &mut self.ad[span]);
                    }
                }
                at += rows;
            }
        }
        {
            let mut segments = Vec::with_capacity(misses.len());
            let (mut x, mut xd) = (&self.act[..], &self.actd[..]);
            let mut ad = &self.ad[..];
            let mut out = &mut self.output[..];
            for (i, m) in misses.iter().enumerate() {
                let [gate, down] = format(i);
                let stride = bytes(gate).unwrap().next_multiple_of(256);
                let bytes = bytes(down).unwrap();
                let rows = m.input.len() / 8192;
                let (xs, xr) = x.split_at(rows * 2048);
                let (ds, dr) = xd.split_at(rows * 2048);
                let (ys, yr) = std::mem::take(&mut out).split_at_mut(rows * 4096);
                let down_m = Matrix::with_format(
                    &m.image[2 * stride..2 * stride + bytes],
                    4096,
                    2048,
                    down,
                )?;
                if exl3(down) {
                    let (a, rest) = ad.split_at(rows * 2048);
                    segments.push(Segment::exl3_plain(down_m, a, sums[i][2], ys, rows)?);
                    ad = rest;
                } else {
                    segments.push(Segment::plain(
                        down_m.with_gpu_order(self.gpu_order),
                        xs,
                        ds,
                        ys,
                        rows,
                    )?);
                    if exl3_down {
                        ad = &ad[rows * 2048..];
                    }
                }
                (x, xd, out) = (xr, dr, yr);
            }
            self.pool.run(&segments)?;
        }
        if let Some(lora) = &self.lora {
            let (mut x, mut y) = (0, 0);
            for m in misses.iter() {
                let rows = m.input.len() / 8192;
                let (layer, expert) = m.expert.ok_or_else(|| {
                    Error::InvalidPlan("GLM CPU expert LoRA needs the expert id".into())
                })?;
                lora.apply(
                    layer,
                    expert,
                    &self.act[x..x + rows * 2048],
                    &mut self.output[y..y + rows * 4096],
                );
                (x, y) = (x + rows * 2048, y + rows * 4096);
            }
        }
        let mut at = 0;
        for m in misses.iter_mut() {
            for (b, &v) in m
                .output
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(&self.output[at..])
            {
                *b = ((bf16(v).to_bits() >> 16) as u16).to_le_bytes();
            }
            at += m.output.len() / 2;
        }
        Ok(())
    }
}
fn rotate(x: &mut [f32], block: usize) -> Result<()> {
    if ![32, 64, 128, 256].contains(&block) || !x.len().is_multiple_of(block) {
        return Err(Error::InvalidShape("GLM CPU Hadamard block".into()));
    }
    let scale = 1. / (block as f32).sqrt();
    for chunk in x.chunks_exact_mut(block) {
        let mut stride = 1;
        while stride < block {
            for pair in chunk.chunks_exact_mut(stride * 2) {
                for j in 0..stride {
                    let (a, b) = (pair[j], pair[j + stride]);
                    pair[j] = a + b;
                    pair[j + stride] = a - b;
                }
            }
            stride *= 2;
        }
        for v in chunk {
            *v = bf16(*v * scale);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn glm53_cpu_fraction_tracks_single_misses() {
        for fraction in [0., 0.1, 0.25, 0.5, 0.9, 1.] {
            let mut s = Split::new(fraction).unwrap();
            let count = (0..1000).filter(|_| s.pcie()).count();
            assert!((count as f64 - 1000. * fraction).abs() <= 1.);
        }
        for v in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
            assert!(Split::new(v).is_err());
        }
    }
    #[test]
    fn glm53_cpu_clamp_rounding_and_hadamard() {
        assert_eq!(swiglu(100., 100.), swiglu(10., 10.));
        assert_eq!(swiglu(-100., -100.), swiglu(-100., -10.));
        assert_eq!(swiglu(0., 10.), 0.);
        for block in [32, 64, 128, 256] {
            let mut x = vec![1.; block * 2];
            rotate(&mut x, block).unwrap();
            assert_eq!(x[0], bf16((block as f32).sqrt()));
            assert!(x[1..block].iter().all(|&v| v == 0.));
        }
        assert!(rotate(&mut [1.; 32], 17).is_err());
    }
    #[test]
    fn glm53_cpu_expert_constant_matrices() -> Result<()> {
        // 0.5 * 2**-10 weights, exactly representable on both devices.
        let stride = super::super::experts::matrix_stride(true);
        let mut image = vec![0; stride * 3];
        for (part, (n, k)) in [(2048, 4096), (2048, 4096), (4096, 2048)]
            .into_iter()
            .enumerate()
        {
            let bytes = QuantSpec::Nvfp4.matrix_bytes(n, k).unwrap() as usize;
            let raw = &mut image[part * stride..part * stride + bytes];
            raw[..(n * k / 2) as usize].fill(0x11);
            raw[(n * k / 2) as usize..bytes - 4].fill(0x38);
            raw[bytes - 4..].copy_from_slice(&(1f32 / 1024.).to_le_bytes());
        }
        let input = 0x3f00u16.to_le_bytes().repeat(4096);
        let mut output = vec![0; 8192];
        let mut cpu = CpuExperts::new()?;
        cpu.forward(&image, &input, &mut output, true, false, 0)?;
        let expected = (bf16(swiglu(1., 1.)).to_bits() >> 16) as u16;
        assert!(
            output
                .as_chunks::<2>()
                .0
                .iter()
                .all(|b| u16::from_le_bytes(*b) == expected)
        );
        for rows in 2..=4 {
            let input: Vec<_> = (0..rows * 4096)
                .flat_map(|i| {
                    ((bf16(((i * 13 % 37) as f32 - 18.) / 32.).to_bits() >> 16) as u16)
                        .to_le_bytes()
                })
                .collect();
            let mut actual = vec![0; rows * 8192];
            cpu.forward(&image, &input, &mut actual, true, false, 0)?;
            let mut single = vec![0; rows * 8192];
            for row in 0..rows {
                cpu.forward(
                    &image,
                    &input[row * 8192..(row + 1) * 8192],
                    &mut single[row * 8192..(row + 1) * 8192],
                    true,
                    false,
                    0,
                )?;
            }
            assert_eq!(actual, single);
        }
        image.resize(2 * stride + super::super::experts::IMAGE_BYTES / 3, 0);
        let down = &mut image[2 * stride..];
        down[..4096 * 2048].fill(0x30);
        for scale in down[4096 * 2048..].as_chunks_mut::<4>().0 {
            scale.copy_from_slice(&(1f32 / 1024.).to_le_bytes());
        }
        cpu.forward(&image, &input, &mut output, true, true, 0)?;
        assert!(
            output
                .as_chunks::<2>()
                .0
                .iter()
                .all(|b| u16::from_le_bytes(*b) == expected)
        );
        Ok(())
    }
    /// Routed-expert LoRA on the CPU: the constant expert's SwiGLU output a0 feeds
    /// A (expert 3: 1/1024 everywhere, t = 2 a0), B row r = (r % 4) / 4, scale 0.5,
    /// so output r = bf16(a0 + (r % 4) a0 / 4); other experts and layers are untouched.
    #[test]
    fn glm53_cpu_expert_lora() -> Result<()> {
        use super::super::lora::{Dtype, HostExperts, Layer, Pair, ROUTED};
        let stride = super::super::experts::matrix_stride(true);
        let mut image = vec![0; stride * 3];
        for (part, (n, k)) in [(2048, 4096), (2048, 4096), (4096, 2048)]
            .into_iter()
            .enumerate()
        {
            let bytes = QuantSpec::Nvfp4.matrix_bytes(n, k).unwrap() as usize;
            let raw = &mut image[part * stride..part * stride + bytes];
            raw[..(n * k / 2) as usize].fill(0x11);
            raw[(n * k / 2) as usize..bytes - 4].fill(0x38);
            raw[bytes - 4..].copy_from_slice(&(1f32 / 1024.).to_le_bytes());
        }
        let input = 0x3f00u16.to_le_bytes().repeat(4096);
        let a0 = bf16(swiglu(1., 1.));
        let mut a = vec![0f32; 288 * 2048];
        a[3 * 2048..4 * 2048].fill(1. / 1024.);
        let b: Vec<f32> = (0..288 * 4096)
            .map(|i| (i % 4096 % 4) as f32 / 4.)
            .collect();
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut layers = vec![Layer::new(); 6];
        layers[5].insert(
            ROUTED.into(),
            vec![Pair {
                rank: 1,
                k: 2048,
                n: 4096,
                count: 288,
                dtype: Dtype::F32,
                a: bytes(&a),
                b: bytes(&b),
                scale: 0.5,
            }],
        );
        let mut cpu = CpuExperts::new()?;
        cpu.lora = Some(std::sync::Arc::new(HostExperts::new(&layers)));
        let mut outputs = vec![vec![0u8; 8192]; 3];
        let mut misses: Vec<_> = outputs
            .iter_mut()
            .zip([(5, 3), (5, 0), (4, 3)])
            .map(|(output, expert)| Miss {
                image: &image,
                input: &input,
                output,
                expert: Some(expert),
            })
            .collect();
        cpu.forward_layer(&mut misses, true, false, 0)?;
        let value = |out: &[u8], r: usize| {
            f32::from_bits((u16::from_le_bytes([out[2 * r], out[2 * r + 1]]) as u32) << 16)
        };
        for r in 0..4096 {
            assert_eq!(value(&outputs[0], r), bf16(a0 + (r % 4) as f32 * a0 / 4.));
            assert_eq!(value(&outputs[1], r), a0);
            assert_eq!(value(&outputs[2], r), a0);
        }
        // A loaded adapter needs every miss's expert id.
        let mut output = vec![0; 8192];
        assert!(
            cpu.forward(&image, &input, &mut output, true, false, 0)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn glm53_cpu_lowbit_layer_formats_and_rounding() -> Result<()> {
        let mut cpu = CpuExperts::new()?;
        let formats: Vec<_> = (0..5)
            .flat_map(|grid| [16, 32].map(|group| Format::Lowbit { grid, group }))
            .collect();
        let images: Vec<_> = formats
            .iter()
            .map(|&format| {
                let Format::Lowbit { grid, group } = format else {
                    unreachable!()
                };
                let size = format.matrix_bytes(2048, 4096).unwrap();
                let stride = size.next_multiple_of(256);
                let mut image = vec![0; stride * 3];
                let code = [2u8, 3, 5, 4, 2][grid as usize];
                for part in 0..3 {
                    let raw = &mut image[part * stride..part * stride + size];
                    let low = 2048 * 4096 / 4;
                    raw[..low].fill((code & 3) * 0x55);
                    let planes = if matches!(grid, 2 | 3) {
                        raw[low..low + low / 2].fill(if code & 4 != 0 { 255 } else { 0 });
                        low + low / 2
                    } else {
                        low
                    };
                    raw[planes..planes + 2048 * 4096 / group].fill(0x38);
                    raw[size - 4..].copy_from_slice(
                        &(if grid == 3 {
                            1f32 / 1024.
                        } else {
                            1f32 / 2048.
                        })
                        .to_le_bytes(),
                    );
                }
                image
            })
            .collect();
        let input: Vec<_> = [0.5f32, 0.25, -0.5, 0.]
            .into_iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes().repeat(4096))
            .collect();
        // Different gate/up and down tiers in the same layer exercise each
        // expert's own offsets; all matrices represent exactly 1/2048.
        let pairs: Vec<_> = formats
            .iter()
            .enumerate()
            .map(|(i, &gate)| [gate, formats[(i + 3) % formats.len()]])
            .collect();
        let mixed: Vec<_> = pairs
            .iter()
            .enumerate()
            .map(|(i, &[gate, down])| {
                let gate_stride = gate.matrix_bytes(2048, 4096).unwrap().next_multiple_of(256);
                let down_index = (i + 3) % formats.len();
                let down_stride = down.matrix_bytes(4096, 2048).unwrap().next_multiple_of(256);
                let mut image = images[i][..gate_stride * 2].to_vec();
                image.extend_from_slice(&images[down_index][down_stride * 2..]);
                image
            })
            .collect();
        let mut batched = vec![vec![0; input.len()]; formats.len()];
        let mut misses: Vec<_> = mixed
            .iter()
            .zip(&mut batched)
            .map(|(image, output)| Miss {
                image,
                input: &input,
                output,
                expert: None,
            })
            .collect();
        cpu.forward_layer_formats(&mut misses, &pairs, 0)?;
        for (i, output) in batched.iter().enumerate() {
            for (row, &v) in [0.5f32, 0.25, -0.5, 0.].iter().enumerate() {
                let expected = (bf16(swiglu(v * 2., v * 2.)).to_bits() >> 16) as u16;
                assert!(
                    output[row * 8192..(row + 1) * 8192]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .all(|b| u16::from_le_bytes(*b) == expected),
                    "expert {i} row {row}"
                );
            }
            let mut single = vec![0; input.len()];
            cpu.forward_layer_formats(
                &mut [Miss {
                    image: &mixed[i],
                    input: &input,
                    output: &mut single,
                    expert: None,
                }],
                &pairs[i..i + 1],
                0,
            )?;
            assert_eq!(&single, output);
        }
        let mut misses: Vec<_> = mixed
            .iter()
            .zip(&mut batched)
            .map(|(image, output)| Miss {
                image,
                input: &input,
                output,
                expert: None,
            })
            .collect();
        cpu.forward_layer_formats(&mut misses, &pairs, 32)?;
        for (i, output) in batched.iter().enumerate() {
            let mut single = vec![0; input.len()];
            cpu.forward_layer_formats(
                &mut [Miss {
                    image: &mixed[i],
                    input: &input,
                    output: &mut single,
                    expert: None,
                }],
                &pairs[i..i + 1],
                32,
            )?;
            assert_eq!(&single, output);
        }
        assert!(
            cpu.forward_layer_formats(&mut [], &[[formats[0]; 2]; 2], 0)
                .is_err()
        );
        let mut bad = [0; 8192];
        assert!(
            cpu.forward_layer_formats(
                &mut [Miss {
                    image: &[],
                    input: &input[..8192],
                    output: &mut bad,
                    expert: None,
                }],
                &[[formats[0]; 2]],
                0
            )
            .is_err()
        );
        Ok(())
    }
    /// EXL3 misses (K2.5 gate/up, K3.5 down) through forward_layer_formats against
    /// the exactly decoded weights in F64, with a LOWBIT-free NVFP4 miss beside them.
    #[test]
    fn glm53_cpu_exl3_layer_formats() -> Result<()> {
        use hayate_kernels::glm53::exl3::{Exl3, had128};
        let matrix = |n: usize, k: usize, k2: usize, seed: usize| -> Vec<u8> {
            let mut v: Vec<u8> = (0..n * k * k2 / 16)
                .map(|i| ((i + seed).wrapping_mul(2_654_435_761) >> 13) as u8)
                .collect();
            for i in 0..n + k {
                let h: u16 = (0x3400 + ((i * 41 + seed) % 0x400) as u16)
                    | if (i + seed).is_multiple_of(3) {
                        0x8000
                    } else {
                        0
                    };
                v.extend(h.to_le_bytes());
            }
            v
        };
        let (g, u, d) = (
            matrix(2048, 4096, 5, 1),
            matrix(2048, 4096, 5, 2),
            matrix(4096, 2048, 7, 3),
        );
        let mut image = g.clone();
        image.extend(&u);
        image.extend(&d);
        let formats = [[Format::Exl3 { k2: 5 }, Format::Exl3 { k2: 7 }]];
        let mut cpu = CpuExperts::new()?;
        let rows = 2;
        let input: Vec<u8> = (0..rows * 4096)
            .flat_map(|i| {
                ((bf16(((i * 37) % 101) as f32 / 50. - 1.).to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let mut output = vec![0u8; rows * 8192];
        cpu.forward_layer_formats(
            &mut [Miss {
                image: &image,
                input: &input,
                output: &mut output,
                expert: None,
            }],
            &formats,
            0,
        )?;
        // reference: y = H(H(x * suh) W') * svh in F64 from the decoded W'
        let apply = |raw: &[u8], n: usize, k: usize, k2: u8, x: &[f32]| -> Vec<f32> {
            let m = Exl3::new(raw, n, k, k2, 2).unwrap();
            let mut xh = vec![0.; k];
            m.prepare(x, &mut xh);
            let w = m.inner();
            let mut y: Vec<f32> = (0..n)
                .map(|c| {
                    (0..k)
                        .map(|r| f64::from(xh[r]) * f64::from(w[r * n + c]))
                        .sum::<f64>() as f32
                })
                .collect();
            had128(&mut y);
            y.iter().enumerate().map(|(c, v)| v * m.svh(c)).collect()
        };
        for r in 0..rows {
            let x: Vec<f32> = input[r * 8192..][..8192]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
                .collect();
            let (a, b) = (apply(&g, 2048, 4096, 5, &x), apply(&u, 2048, 4096, 5, &x));
            let act: Vec<f32> = a
                .iter()
                .zip(&b)
                .map(|(&a, &b)| swiglu(bf16(a), bf16(b)))
                .collect();
            let y = apply(&d, 4096, 2048, 7, &act);
            let scale = y.iter().map(|v| v.abs()).fold(0f32, f32::max);
            for (c, &want) in y.iter().enumerate() {
                let got = f32::from_bits(
                    (u16::from_le_bytes([output[r * 8192 + 2 * c], output[r * 8192 + 2 * c + 1]])
                        as u32)
                        << 16,
                );
                assert!(
                    (got - want).abs() <= 2e-2 * scale,
                    "row {r} col {c}: {got} vs {want}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn glm53_cpu_layer_batch_matches_single_expert_bits() -> Result<()> {
        let stride = super::super::experts::matrix_stride(true);
        let mut s = 0x9e3779b9u32;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let images: Vec<Vec<u8>> = (0..3)
            .map(|_| {
                let mut image = vec![0; stride * 3];
                for (part, (n, k)) in [(2048u64, 4096u64), (2048, 4096), (4096, 2048)]
                    .into_iter()
                    .enumerate()
                {
                    let bytes = QuantSpec::Nvfp4.matrix_bytes(n, k).unwrap() as usize;
                    let raw = &mut image[part * stride..part * stride + bytes];
                    let half = (n * k / 2) as usize;
                    raw[..half].iter_mut().for_each(|b| *b = next() as u8);
                    raw[half..bytes - 4]
                        .iter_mut()
                        .for_each(|b| *b = 0x20 + (next() % 48) as u8);
                    raw[bytes - 4..].copy_from_slice(&(1f32 / 64.).to_le_bytes());
                }
                image
            })
            .collect();
        let mut cpu = CpuExperts::new()?;
        for rows in [1usize, 3] {
            let input: Vec<_> = (0..rows * 4096)
                .flat_map(|i| {
                    ((bf16(((i * 29 % 61) as f32 - 30.) / 40.).to_bits() >> 16) as u16)
                        .to_le_bytes()
                })
                .collect();
            let mut single = vec![vec![0; rows * 8192]; 3];
            for (image, out) in images.iter().zip(&mut single) {
                cpu.forward(image, &input, out, true, false, 0)?;
            }
            let mut batched = vec![vec![0; rows * 8192]; 3];
            let mut misses: Vec<_> = images
                .iter()
                .zip(&mut batched)
                .map(|(image, output)| Miss {
                    image,
                    input: &input,
                    output,
                    expert: None,
                })
                .collect();
            cpu.forward_layer(&mut misses, true, false, 0)?;
            assert_eq!(batched, single);
            assert!(single[0].iter().any(|&b| b != 0));
        }
        Ok(())
    }
}

#[cfg(test)]
mod measurements {
    use super::*;
    use hayate_core::{MachineId, hyt::HytFile};

    // Own the complete mapping, including the alignment prefix, so repeated
    // huge-page benchmark runs release their pages too.
    struct Mapping {
        base: *mut u8,
        len: usize,
    }
    impl Drop for Mapping {
        fn drop(&mut self) {
            unsafe extern "C" {
                fn munmap(base: *mut u8, len: usize) -> i32;
            }
            unsafe {
                munmap(self.base, self.len);
            }
        }
    }
    fn load_image(file: &HytFile, expert: usize) -> Result<(Vec<u8>, bool, bool)> {
        let find = |stem: &str, suffix: &str| -> Result<usize> {
            let name = format!("mlp.experts.{expert}.{stem}_proj.{suffix}");
            file.entries()
                .iter()
                .position(|e| e.name == name)
                .ok_or(Error::Weights(name))
        };
        let nv = file.entries()[find("gate", "weight")?].quant == QuantSpec::Nvfp4;
        let down_fp8 = nv && file.entries()[find("down", "weight")?].quant != QuantSpec::Nvfp4;
        let stride = super::super::experts::matrix_stride(nv);
        let mut raw = vec![0; 2 * stride + super::super::experts::matrix_stride(nv && !down_fp8)];
        for (part, stem) in ["gate", "up", "down"].into_iter().enumerate() {
            let mut at = part * stride;
            for suffix in ["weight", "weight_scale_inv"] {
                if suffix != "weight" && nv && !(part == 2 && down_fp8) {
                    continue;
                }
                let index = find(stem, suffix)?;
                let bytes = file.entries()[index].bytes as usize;
                file.read_tensor(index, &mut raw[at..at + bytes])?;
                at += bytes.next_multiple_of(256);
            }
        }
        Ok((raw, nv, down_fp8))
    }
    fn uptime() -> Result<String> {
        let out = std::process::Command::new("uptime").output()?;
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
    }
    fn extra_cores(cpu: &mut CpuExperts, threads: usize, spin: bool) -> Result<usize> {
        let extra = std::env::var("GLM53_CPU_ECORES")
            .unwrap_or_else(|_| "0".into())
            .parse::<usize>()
            .map_err(|_| Error::InvalidPlan("GLM53_CPU_ECORES must be 0..6".into()))?;
        if extra > 6 {
            return Err(Error::InvalidPlan("GLM53_CPU_ECORES must be 0..6".into()));
        }
        if extra != 0 {
            let mut cores = hayate_core::machine::MAIN.cpu.compute[..threads].to_vec();
            cores.extend(16..16 + extra as u16);
            cpu.pool = Pool::with_wait(&cores, spin)?;
        }
        Ok(threads + extra)
    }
    fn file() -> Result<HytFile> {
        let root = std::env::var("GLM53_HYT").unwrap_or_else(|_| {
            "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53-stage6-fp8-dense".into()
        });
        HytFile::open(
            std::path::Path::new(&root).join("layer3/weights.hyt"),
            MachineId::Main,
        )
    }
    /// The converted EXL3 checkpoint through the loader's per-layer and per-expert checks and the image reader,
    /// and one real expert through the CPU lane against its exactly decoded weights.
    #[test]
    #[ignore = "CPU only: the converted EXL3 checkpoint (GLM53_EXL3_HYT, default glm53x3-sc260)"]
    fn glm53_cpu_exl3_checkpoint_layers() -> Result<()> {
        use hayate_kernels::glm53::exl3::{Exl3, had128};
        let root = std::env::var("GLM53_EXL3_HYT").unwrap_or_else(|_| {
            "/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53x3-sc260".into()
        });
        for (layer, want) in [(3usize, (7u8, 7u8)), (4, (5, 5)), (44, (5, 5))] {
            let file = HytFile::open(
                std::path::Path::new(&root).join(format!("layer{layer}/weights.hyt")),
                MachineId::Main,
            )?;
            assert!(super::super::experts::is_mixed(&file));
            assert_eq!(super::super::experts::exl3_bitrates(&file)?, Some(want));
            let (entries, format, bytes) = super::super::experts::mixed_expert(&file, 287)?;
            assert!(format.gate_up.is_exl3() && format.down.is_exl3());
            // the reader wants a 256-byte aligned destination, as the pinned images are
            let mut buffer = vec![0u8; bytes + 256];
            let at = buffer.as_ptr().align_offset(256);
            let image = &mut buffer[at..at + bytes];
            super::super::experts::read_image(&file, &entries, image)?;
            let image = &*image;
            if layer != 4 {
                continue;
            }
            // expert 287 of layer 4 on the CPU lane against y = H(H(x suh) W') svh from the decoded W'
            let mut cpu = CpuExperts::new()?;
            let input: Vec<u8> = (0..4096)
                .flat_map(|i| {
                    ((bf16(((i * 37) % 101) as f32 / 50. - 1.).to_bits() >> 16) as u16)
                        .to_le_bytes()
                })
                .collect();
            let mut output = vec![0u8; 8192];
            let f = Format::Exl3 { k2: 5 };
            cpu.forward_layer_formats(
                &mut [Miss {
                    image,
                    input: &input,
                    output: &mut output,
                    expert: None,
                }],
                &[[f, f]],
                0,
            )?;
            let gb = f.matrix_bytes(2048, 4096).unwrap();
            let apply = |raw: &[u8], n: usize, k: usize, x: &[f32]| -> Vec<f32> {
                let m = Exl3::new(raw, n, k, 5, 2).unwrap();
                let mut xh = vec![0.; k];
                m.prepare(x, &mut xh);
                let w = m.inner();
                let mut y: Vec<f32> = (0..n)
                    .map(|c| {
                        (0..k)
                            .map(|r| f64::from(xh[r]) * f64::from(w[r * n + c]))
                            .sum::<f64>() as f32
                    })
                    .collect();
                had128(&mut y);
                y.iter().enumerate().map(|(c, v)| v * m.svh(c)).collect()
            };
            let x: Vec<f32> = input
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
                .collect();
            let (a, b) = (
                apply(&image[..gb], 2048, 4096, &x),
                apply(&image[gb..2 * gb], 2048, 4096, &x),
            );
            let act: Vec<f32> = a
                .iter()
                .zip(&b)
                .map(|(&a, &b)| swiglu(bf16(a), bf16(b)))
                .collect();
            let y = apply(
                &image[2 * gb..2 * gb + f.matrix_bytes(4096, 2048).unwrap()],
                4096,
                2048,
                &act,
            );
            let scale = y.iter().map(|v| v.abs()).fold(0f32, f32::max);
            for (c, &want) in y.iter().enumerate() {
                let got = f32::from_bits(
                    (u16::from_le_bytes([output[2 * c], output[2 * c + 1]]) as u32) << 16,
                );
                assert!(
                    (got - want).abs() <= 2e-2 * scale,
                    "col {c}: {got} vs {want}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "CPU only: real HYT rows, CUDA lane-order F32 oracle"]
    fn glm53_cpu_nvfp4_gpu_order_real_rows() -> Result<()> {
        let file = file()?;
        let (image, nv, down_fp8) = load_image(&file, 0)?;
        assert!(nv && !down_fp8);
        let stride = super::super::experts::matrix_stride(true);
        let mut pool = Pool::new(hayate_core::machine::MAIN.cpu.compute)?;
        let mut checked = 0;
        let (mut error, mut energy) = (0f64, 0f64);
        for (part, (n, k)) in [(2048, 4096), (2048, 4096), (4096, 2048)]
            .into_iter()
            .enumerate()
        {
            let size = Format::Nvfp4.matrix_bytes(n, k).unwrap();
            let raw = &image[part * stride..part * stride + size];
            let matrix = Matrix::new(raw, n, k, true)?;
            let global = f32::from_le_bytes(raw[size - 4..].try_into().unwrap());
            for rows in 1..=4 {
                let x: Vec<_> = (0..rows * k)
                    .map(|i| bf16(((i * 83 % 251) as f32 - 125.) / 47.))
                    .collect();
                let mut actual = vec![0.; rows * n];
                let mut old = vec![0.; rows * n];
                pool.batch(matrix.with_gpu_order(true), &x, &mut actual, rows)?;
                pool.batch(matrix, &x, &mut old, rows)?;
                for (&a, &b) in actual.iter().zip(&old) {
                    error += (a as f64 - b as f64).powi(2);
                    energy += (b as f64).powi(2);
                }
                for m in 0..rows {
                    for r in [0, 31, 32, 127, 128, n - 1] {
                        let mut lanes = [0f32; 32];
                        for (lane, sum) in lanes.iter_mut().enumerate() {
                            for c in (lane * 8..k).step_by(256) {
                                let col = c / 16;
                                let at = ((r / 128) * (k / 64) + col / 4) * 512
                                    + r % 32 * 16
                                    + (r % 128 / 32) * 4
                                    + col % 4;
                                let b = raw[n * k / 2 + at];
                                let (e, v) = ((b >> 3) & 15, b & 7);
                                let scale = if e == 0 {
                                    v as f32 / 512.
                                } else {
                                    f32::from_bits(((e as u32 + 120) << 23) | ((v as u32) << 20))
                                } * if b & 128 != 0 { -1. } else { 1. };
                                let multiplier = scale * global;
                                for j in 0..8 {
                                    let code =
                                        (raw[(r * k + c + j) / 2] >> (4 * ((c + j) % 2))) & 15;
                                    let level = [0f32, 0.5, 1., 1.5, 2., 3., 4., 6.]
                                        [(code & 7) as usize]
                                        * if code & 8 != 0 { -1. } else { 1. };
                                    *sum = (level * multiplier).mul_add(x[m * k + c + j], *sum);
                                }
                            }
                        }
                        for offset in [16, 8, 4, 2, 1] {
                            for lane in 0..32 - offset {
                                lanes[lane] += lanes[lane + offset];
                            }
                        }
                        assert_eq!(
                            actual[m * n + r].to_bits(),
                            lanes[0].to_bits(),
                            "part={part} m={m} r={r}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        println!(
            "GPU-order real rows: {checked} bit matches; F32 relative_l2 vs old={:.9}; {}",
            (error / energy).sqrt(),
            uptime()?
        );
        Ok(())
    }
    #[test]
    #[ignore = "CPU bandwidth: real HYT, release mode, 2 warmup + 5 measured sweeps"]
    fn glm53_cpu_expert_bandwidth() -> Result<()> {
        let rows = std::env::var("GLM53_CPU_ROWS")
            .unwrap_or_else(|_| "1".into())
            .parse::<usize>()
            .map_err(|_| Error::InvalidPlan("GLM53_CPU_ROWS must be 1..4".into()))?;
        if !(1..=4).contains(&rows) {
            return Err(Error::InvalidPlan("GLM53_CPU_ROWS must be 1..4".into()));
        }
        let file = file()?;
        let owned = (0..32)
            .map(|e| load_image(&file, e))
            .collect::<Result<Vec<_>>>()?;
        // GLM53_CPU_HUGE=1: copy the images into one 2 MiB-page region.
        let huge = std::env::var("GLM53_CPU_HUGE").is_ok_and(|v| v == "1");
        let size = owned[0].0.len().next_multiple_of(2 << 20);
        let mut mapping = None;
        let region: &mut [u8] = if huge {
            unsafe extern "C" {
                fn mmap(a: *mut u8, l: usize, p: i32, f: i32, fd: i32, o: i64) -> *mut u8;
                fn madvise(a: *mut u8, l: usize, advice: i32) -> i32;
            }
            let len = size * owned.len() + (2 << 20);
            let base = unsafe { mmap(std::ptr::null_mut(), len, 3, 0x22, -1, 0) };
            if base as isize == -1 {
                return Err(std::io::Error::last_os_error().into());
            }
            mapping = Some(Mapping { base, len });
            let aligned = unsafe { base.add(base.align_offset(2 << 20)) };
            if unsafe { madvise(aligned, size * owned.len(), 14) } != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            unsafe { std::slice::from_raw_parts_mut(aligned, size * owned.len()) }
        } else {
            &mut []
        };
        let images: Vec<(&[u8], bool, bool)> = if huge {
            for (i, (raw, _, _)) in owned.iter().enumerate() {
                region[i * size..i * size + raw.len()].copy_from_slice(raw);
            }
            owned
                .iter()
                .enumerate()
                .map(|(i, (raw, nv, d))| (&region[i * size..i * size + raw.len()], *nv, *d))
                .collect()
        } else {
            owned
                .iter()
                .map(|(raw, nv, d)| (&raw[..], *nv, *d))
                .collect()
        };
        if huge {
            let status = std::fs::read_to_string("/proc/self/smaps_rollup")?;
            println!(
                "{}",
                status
                    .lines()
                    .find(|s| s.starts_with("AnonHugePages:"))
                    .unwrap_or("AnonHugePages: unavailable")
            );
        }
        let hadamard = super::super::experts::hadamard_block(&file)?;
        let input: Vec<_> = (0..rows * 4096)
            .flat_map(|c| ((bf16((c as f32 * 0.031).sin()).to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let batch = std::env::var("GLM53_CPU_BATCH")
            .unwrap_or_else(|_| "8".into())
            .parse::<usize>()
            .map_err(|_| Error::InvalidPlan("GLM53_CPU_BATCH".into()))?
            .clamp(1, 32);
        let threads = std::env::var("GLM53_CPU_THREADS")
            .unwrap_or_else(|_| "8".into())
            .parse::<usize>()
            .map_err(|_| Error::InvalidPlan("GLM53_CPU_THREADS".into()))?;
        let spin = std::env::var("GLM53_CPU_SPIN").is_ok_and(|v| v == "1");
        let host_core = std::env::var("GLM53_CPU_HOST_CORE")
            .unwrap_or_else(|_| "23".into())
            .parse::<u16>()
            .map_err(|_| Error::InvalidPlan("GLM53_CPU_HOST_CORE must be 0 or 23".into()))?;
        if ![0, 23].contains(&host_core) {
            return Err(Error::InvalidPlan(
                "GLM53_CPU_HOST_CORE must be 0 or 23".into(),
            ));
        }
        hayate_kernels::cpu::pin_thread(host_core)?;
        let mut cpu = CpuExperts::with_threads(threads, spin)?;
        let threads = extra_cores(&mut cpu, threads, spin)?;
        let mut samples = Vec::new();
        let bytes: usize = images.iter().map(|(raw, _, _)| raw.len()).sum();
        println!("{}", uptime()?);
        for run in 0..7 {
            let before = cpu.timing();
            let start = std::time::Instant::now();
            for group in images.chunks(batch) {
                let mut outputs = vec![vec![0u8; rows * 8192]; group.len()];
                let mut misses: Vec<_> = group
                    .iter()
                    .zip(&mut outputs)
                    .map(|((raw, _, _), out)| Miss {
                        image: raw,
                        input: &input,
                        output: out,
                        expert: None,
                    })
                    .collect();
                let (_, nv, down_fp8) = group[0];
                cpu.forward_layer(&mut misses, nv, down_fp8, hadamard)?;
                std::hint::black_box(&outputs);
            }
            let seconds = start.elapsed().as_secs_f64();
            if run >= 2 {
                samples.push(seconds);
                println!(
                    "glm53_cpu_expert_bandwidth run={} cores={threads} batch={batch} huge={huge} spin={spin} gpu_order={} host_core={host_core} experts=32 rows={rows} bytes={bytes} seconds={seconds:.6} gb_s={:.6} ms_expert_token={:.6} worker_ms={:.6} dispatch_ms={:.6}",
                    run - 2,
                    cpu.gpu_order,
                    bytes as f64 / seconds / 1e9,
                    seconds * 1000. / (32 * rows) as f64,
                    cpu.timing().0 - before.0,
                    cpu.timing().1 - before.1,
                );
            }
        }
        println!("{}", uptime()?);
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        drop(mapping);
        println!(
            "glm53_cpu_expert_bandwidth median gb_s={:.6} us_expert={:.3}",
            bytes as f64 / median / 1e9,
            median * 1e6 / 32.
        );
        Ok(())
    }
    #[test]
    #[ignore = "CPU bandwidth: synthetic LOWBIT experts larger than L3, release mode"]
    fn glm53_cpu_lowbit_bandwidth() -> Result<()> {
        let threads = std::env::var("GLM53_CPU_THREADS")
            .unwrap_or_else(|_| "8".into())
            .parse::<usize>()
            .map_err(|_| Error::InvalidPlan("GLM53_CPU_THREADS".into()))?;
        let batch = std::env::var("GLM53_CPU_BATCH")
            .unwrap_or_else(|_| "8".into())
            .parse::<usize>()
            .map_err(|_| Error::InvalidPlan("GLM53_CPU_BATCH".into()))?
            .clamp(1, 32);
        hayate_kernels::cpu::pin_thread(23)?;
        let mut cpu = CpuExperts::with_threads(threads, false)?;
        let threads = extra_cores(&mut cpu, threads, false)?;
        let input: Vec<_> = (0..4096)
            .flat_map(|c| ((bf16((c as f32 * 0.031).sin()).to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        for grid in 0..5 {
            for group in [16, 32] {
                let format = Format::Lowbit { grid, group };
                let size = format.matrix_bytes(2048, 4096).unwrap();
                let stride = size.next_multiple_of(256);
                let mut seed = 0x9e3779b9u32;
                // 32 distinct experts, 252..340 MB: more than the 30 MB L3.
                let images: Vec<_> = (0..32)
                    .map(|_| {
                        let mut image = vec![0; stride * 3];
                        let planes = 2048 * 4096 / 8 * if matches!(grid, 2 | 3) { 3 } else { 2 };
                        for part in 0..3 {
                            let raw = &mut image[part * stride..part * stride + size];
                            for b in &mut raw[..planes] {
                                seed ^= seed << 13;
                                seed ^= seed >> 17;
                                seed ^= seed << 5;
                                *b = seed as u8;
                            }
                            raw[planes..size - 4].fill(0x38);
                            raw[size - 4..].copy_from_slice(&(1f32 / 1024.).to_le_bytes());
                        }
                        image
                    })
                    .collect();
                let bytes = 32 * 3 * size;
                let mut outputs = vec![vec![0; 8192]; 32];
                let mut samples = Vec::new();
                println!("{}", uptime()?);
                for run in 0..7 {
                    let start = std::time::Instant::now();
                    for (images, outputs) in images.chunks(batch).zip(outputs.chunks_mut(batch)) {
                        let mut misses: Vec<_> = images
                            .iter()
                            .zip(outputs)
                            .map(|(image, output)| Miss {
                                image,
                                input: &input,
                                output,
                                expert: None,
                            })
                            .collect();
                        cpu.forward_layer_formats(&mut misses, &[[format; 2]], 0)?;
                    }
                    let seconds = start.elapsed().as_secs_f64();
                    std::hint::black_box(&outputs);
                    if run >= 2 {
                        samples.push(seconds);
                        println!(
                            "glm53_cpu_lowbit_bandwidth grid={grid} group={group} cores={threads} batch={batch} experts=32 run={} gb_s={:.6} us_expert={:.3}",
                            run - 2,
                            bytes as f64 / seconds / 1e9,
                            seconds * 1e6 / 32.
                        );
                    }
                }
                println!("{}", uptime()?);
                samples.sort_by(f64::total_cmp);
                let median = samples[samples.len() / 2];
                println!(
                    "glm53_cpu_lowbit_bandwidth median grid={grid} group={group} cores={threads} batch={batch} bytes={bytes} gb_s={:.6} us_expert={:.3}",
                    bytes as f64 / median / 1e9,
                    median * 1e6 / 32.
                );
            }
        }
        Ok(())
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires main CUDA GPU and real GLM53_HYT; direct CPU/GPU expert comparison"]
    fn glm53_cpu_expert_matches_gpu() -> Result<()> {
        use hayate_kernels::{
            glm53::ffi as g,
            runtime::{Device, check},
        };
        let file = file()?;
        let (image, nv, down_fp8) = load_image(&file, 0)?;
        let hadamard = super::super::experts::hadamard_block(&file)?;
        let stride = super::super::experts::matrix_stride(nv);
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut weights = device.alloc(image.len())?;
        stream.copy_h2d(&mut weights, &image)?;
        let input: Vec<_> = (0..4096).map(|c| bf16((c as f32 * 0.031).sin())).collect();
        let raw_input: Vec<_> = input
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let mut x = device.alloc(8192)?;
        stream.copy_h2d(&mut x, &raw_input)?;
        let mut product = device.alloc(9 * 2048 * 2)?;
        let mut output = device.alloc(9 * 8192)?;
        stream.copy_h2d(&mut output, &0x4120u16.to_le_bytes().repeat(9 * 4096))?;
        let pointers: Vec<_> = (0..8)
            .flat_map(|i| {
                if i == 0 {
                    (weights.as_ptr() as u64).to_le_bytes()
                } else {
                    0u64.to_le_bytes()
                }
            })
            .collect();
        let mut images = device.alloc(64)?;
        stream.copy_h2d(&mut images, &pointers)?;
        let dense = |part: usize| {
            let ptr = unsafe { weights.as_ptr().cast::<u8>().add(part * stride) };
            let packed = nv && !(part == 2 && down_fp8);
            g::DenseWeight {
                data: ptr.cast(),
                scale: if packed {
                    std::ptr::null()
                } else {
                    unsafe { ptr.add(4096 * 2048).cast() }
                },
                format: if packed { 3 } else { 1 },
            }
        };
        check(
            unsafe {
                g::hayate_glm_moe_gate_up(
                    product.as_mut_ptr(),
                    x.as_ptr(),
                    x.as_ptr(),
                    images.as_ptr(),
                    dense(0),
                    dense(1),
                    nv as i32,
                    down_fp8 as i32,
                    stride,
                    stream.as_raw(),
                )
            },
            "CPU comparison gate/up",
        )?;
        if hadamard != 0 && nv && !down_fp8 {
            check(
                unsafe {
                    g::hayate_glm_hadamard(
                        product.as_mut_ptr(),
                        product.as_ptr(),
                        8,
                        2048,
                        hadamard,
                        stream.as_raw(),
                    )
                },
                "CPU comparison rotation",
            )?;
        }
        check(
            unsafe {
                g::hayate_glm_moe_down(
                    output.as_mut_ptr(),
                    product.as_ptr(),
                    images.as_ptr(),
                    dense(2),
                    nv as i32,
                    down_fp8 as i32,
                    stride,
                    stream.as_raw(),
                )
            },
            "CPU comparison down",
        )?;
        let mut gpu = vec![0; 9 * 8192];
        stream.copy_d2h(&mut gpu, &output)?;
        // Null images represent CPU slots. The GPU must leave them untouched.
        assert!(
            gpu[8192..8 * 8192]
                .as_chunks::<2>()
                .0
                .iter()
                .all(|b| u16::from_le_bytes(*b) == 0x4120)
        );
        let mut cpu = CpuExperts::new()?;
        let mut actual = vec![0; 8192];
        cpu.forward(&image, &raw_input, &mut actual, nv, down_fp8, hadamard)?;
        let values = |raw: &[u8]| -> Vec<f64> {
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16) as f64)
                .collect()
        };
        let error = super::super::relative_l2(&values(&actual), &values(&gpu[..8192]))?;
        println!("GLM CPU/GPU expert relative_l2={error:.9}");
        if cpu.gpu_order {
            // Check SwiGLU separately: CUDA --use_fast_math may differ from
            // the host exp/reciprocal even when all GEMV sums are identical.
            let mut gpu_act = vec![0; 9 * 2048 * 2];
            stream.copy_d2h(&mut gpu_act, &product)?;
            let cpu_act: Vec<_> = cpu
                .act
                .iter()
                .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            assert_eq!(cpu_act, gpu_act[..2048 * 2], "SwiGLU/rotation boundary");
            assert_eq!(actual, gpu[..8192], "down boundary");
        } else {
            assert!(error < 1e-3);
        }
        Ok(())
    }
}
