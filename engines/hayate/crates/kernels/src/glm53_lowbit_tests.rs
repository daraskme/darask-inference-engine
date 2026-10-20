//! Scalar LOWBIT oracle and checkpoint-free GPU checks / DRAM benchmarks.
use crate::glm53::lowbit::MatrixFormat;

const LEVELS: [&[f32]; 5] = [
    &[-3., -1., 1., 3.],
    &[-2., -1., 0., 1.],
    &[-4., -3., -2., -1., 0., 1., 2., 3.],
    &[-4., -3., -1.5, -0.5, 0.5, 1.5, 3., 4.],
    &[-4., -1., 1., 4.],
];
const FP4: [f32; 8] = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
fn bf(x: f32) -> u16 {
    let v = x.to_bits();
    ((v + 0x7fff + ((v >> 16) & 1)) >> 16) as u16
}
fn unbf(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}
fn fp8(x: u8) -> f32 {
    let e = (x >> 3) & 15;
    let m = x & 7;
    let v = if e == 0 {
        f32::from(m) / 512.
    } else {
        (1. + f32::from(m) / 8.) * 2f32.powi(i32::from(e) - 7)
    };
    if x & 128 == 0 { v } else { -v }
}
fn random(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}
#[derive(Debug)]
struct Matrix {
    n: usize,
    k: usize,
    f: MatrixFormat,
    codes: Vec<u8>,
    scales: Vec<u8>,
    tensor: f32,
    image: Vec<u8>,
}
impl Matrix {
    fn new(n: usize, k: usize, grid: u8, group: u8, seed: &mut u32) -> Self {
        let f = MatrixFormat::new(grid, group).unwrap();
        let codes: Vec<u8> = (0..n * k)
            .map(|_| (random(seed) as usize % LEVELS[grid as usize].len()) as u8)
            .collect();
        let scales: Vec<u8> = (0..n * k / usize::from(group))
            .map(|_| [0, 1, 7, 8, 0x19, 0x31, 0x3f, 0x68, 0x7e][random(seed) as usize % 9])
            .collect();
        let tensor = 0.00731f32;
        let mut image = vec![0; f.storage_bytes(n as u32, k as u32).unwrap()];
        for r in 0..n {
            for c in 0..k {
                let code = codes[r * k + c];
                image[r * k / 4 + c / 128 * 32 + c % 32] |= (code & 3) << (2 * (c % 128 / 32));
                if f.format == 1 {
                    image[n * k / 4 + r * k / 8 + c / 256 * 32 + c % 32] |=
                        (code >> 2) << (c % 256 / 32);
                }
            }
        }
        let at = n * k / 8 * if f.format == 1 { 3 } else { 2 };
        image[at..at + scales.len()].copy_from_slice(&scales);
        image[at + scales.len()..].copy_from_slice(&tensor.to_le_bytes());
        Self {
            n,
            k,
            f,
            codes,
            scales,
            tensor,
            image,
        }
    }
    fn level(&self, r: usize, c: usize) -> f32 {
        LEVELS[self.f.grid as usize][self.codes[r * self.k + c] as usize]
    }
    fn scale(&self, r: usize, c: usize) -> f32 {
        fp8(self.scales[(r * self.k + c) / usize::from(self.f.group)])
    }
    fn nibble(&self, i: usize) -> u8 {
        let v = LEVELS[self.f.grid as usize][self.codes[i] as usize];
        FP4.iter().position(|&x| x == v.abs()).unwrap() as u8 | if v < 0. { 8 } else { 0 }
    }
    // Independent scalar equivalent of glm53_pack.to_nvfp4 / quant.tile_scales.
    fn to_nvfp4(&self) -> Vec<u8> {
        let mut out = vec![
            0;
            MatrixFormat::NVFP4
                .storage_bytes(self.n as u32, self.k as u32)
                .unwrap()
        ];
        for (i, byte) in out.iter_mut().take(self.n * self.k / 2).enumerate() {
            *byte = self.nibble(2 * i) | (self.nibble(2 * i + 1) << 4);
        }
        for r in 0..self.n {
            for c in 0..self.k / 16 {
                let at = ((r / 128) * (self.k / 64) + c / 4) * 512
                    + r % 32 * 16
                    + r % 128 / 32 * 4
                    + c % 4;
                out[self.n * self.k / 2 + at] = self.scales
                    [r * self.k / usize::from(self.f.group) + c / (usize::from(self.f.group) / 16)];
            }
        }
        let len = out.len();
        out[len - 4..].copy_from_slice(&self.tensor.to_le_bytes());
        out
    }
    // Identical lane ownership, scaled weights and FMA order for LOWBIT/NVFP4.
    fn dot(&self, x: &[u16], r: usize) -> f32 {
        let mut sums = [0f32; 32];
        for (lane, sum) in sums.iter_mut().enumerate() {
            for c in (lane * 8..self.k).step_by(256) {
                for j in 0..8 {
                    let weight = self.level(r, c + j) * (self.scale(r, c + j) * self.tensor);
                    *sum = weight.mul_add(unbf(x[c + j]), *sum);
                }
            }
        }
        for step in [16, 8, 4, 2, 1] {
            let prev = sums;
            for lane in 0..32 - step {
                sums[lane] += prev[lane + step];
            }
        }
        sums[0]
    }
    // Specification's group-wise sum, using f64 to separate rounding from layout.
    fn reference(&self, x: &[u16], r: usize) -> f64 {
        let group = usize::from(self.f.group);
        let mut sum = 0.;
        for c in (0..self.k).step_by(group) {
            let dot: f64 = (c..c + group)
                .map(|j| f64::from(self.level(r, j)) * f64::from(unbf(x[j])))
                .sum();
            sum += dot * f64::from(self.scale(r, c));
        }
        sum * f64::from(self.tensor)
    }
}
fn activation(count: usize, seed: &mut u32) -> Vec<u16> {
    (0..count)
        .map(|_| bf((random(seed) % 257) as f32 / 128. - 1.))
        .collect()
}

#[test]
fn glm53_lowbit_layout_and_scalar_gemv() {
    let mut seed = 13;
    let mut max_error = 0f64;
    let mut bf16_differences = 0;
    let mut outputs = 0;
    assert_eq!(std::mem::size_of::<MatrixFormat>(), 4);
    assert_eq!(std::mem::size_of::<crate::glm53::lowbit::ExpertFormat>(), 8);
    for grid in 0..5 {
        for group in [16, 32] {
            for (n, k) in [(3, 256), (129, 768)] {
                let w = Matrix::new(n, k, grid, group, &mut seed);
                let nv = w.to_nvfp4();
                for r in 0..n {
                    for c in 0..k {
                        let code = (nv[(r * k + c) / 2] >> (c % 2 * 4)) & 15;
                        let value = FP4[(code & 7) as usize] * if code & 8 == 0 { 1. } else { -1. };
                        let scol = c / 16;
                        let offset = ((r / 128) * (k / 64) + scol / 4) * 512
                            + r % 32 * 16
                            + r % 128 / 32 * 4
                            + scol % 4;
                        let have = value * fp8(nv[n * k / 2 + offset]) * w.tensor;
                        let want = w.level(r, c) * w.scale(r, c) * w.tensor;
                        assert_eq!(
                            have.to_bits(),
                            want.to_bits(),
                            "grid={grid} group={group} r={r} c={c}"
                        );
                    }
                }
                for m in 1..=4 {
                    let x = activation(m * k, &mut seed);
                    for t in 0..m {
                        for r in 0..n.min(7) {
                            let have = w.dot(&x[t * k..][..k], r);
                            let want = w.reference(&x[t * k..][..k], r);
                            max_error = max_error.max((f64::from(have) - want).abs());
                            bf16_differences += usize::from(bf(have) != bf(want as f32));
                            outputs += 1;
                            assert!(
                                (f64::from(have) - want).abs() <= 2e-5 * (1. + want.abs()),
                                "grid={grid} group={group} m={m}: {have} vs {want}"
                            );
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "LOWBIT scalar outputs={outputs} max_abs_f32_vs_f64={max_error:e} bf16_vs_f64_differences={bf16_differences}"
    );
}

#[test]
fn glm53_lowbit_sizes_and_validation() {
    for (grid, group, gu, down) in [
        (0, 16, 5_243_392, 2_621_696),
        (3, 32, 6_816_256, 3_408_128),
        (4, 32, 4_719_104, 2_359_552),
    ] {
        let f = MatrixFormat::new(grid, group).unwrap();
        assert_eq!(2 * f.stride(2048, 4096).unwrap(), gu);
        assert_eq!(f.stride(4096, 2048).unwrap(), down);
    }
    assert_eq!(
        2 * MatrixFormat::NVFP4.stride(2048, 4096).unwrap(),
        9_437_696
    );
    assert!(MatrixFormat::new(5, 16).is_err());
    assert!(MatrixFormat::new(0, 8).is_err());
    let f = MatrixFormat::new(0, 16).unwrap();
    assert!(f.storage_bytes(0, 256).is_err());
    assert!(f.storage_bytes(1, 255).is_err());
    assert!(
        MatrixFormat { format: 1, ..f }
            .storage_bytes(1, 256)
            .is_err()
    );
    assert!(
        MatrixFormat { reserved: 1, ..f }
            .storage_bytes(1, 256)
            .is_err()
    );
}

#[test]
fn glm53_lowbit_cuda_helpers_on_cpu() {
    use std::{fs, process::Command};
    let dir = std::env::temp_dir().join(format!("hayate-lowbit-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let source = dir.join("check.cpp");
    fs::write(&source, r#"
#include "lowbit.cuh"
#include <cstdio>
#include <cstring>
#include <vector>
float from_bf(unsigned short x) {
  unsigned bits=unsigned(x)<<16;float v;memcpy(&v,&bits,4);return v;
}
int main() {
  unsigned h[4];
  while(fread(h,sizeof(h),1,stdin)==1) {
    if(!h[0])break;
    unsigned n=h[0],k=h[1],grid=h[2],group=h[3];
    glm53::lowbit::MatrixFormat f={(unsigned char)(glm53::lowbit::bits(grid)==3?1:2),(unsigned char)grid,(unsigned char)group,0};
    std::vector<unsigned char> in(glm53::lowbit::storage(n,k,f));
    if(fread(in.data(),in.size(),1,stdin)!=1)return 2;
    std::vector<unsigned short> x(4*k);
    if(fread(x.data(),x.size()*2,1,stdin)!=1)return 2;
    size_t bytes=glm53::lowbit::storage(n,k,{0,0,0,0});
    for(size_t i=0;i<bytes;++i)putchar(glm53::lowbit::convert_byte(in.data(),i,n,k,grid,group));
    std::vector<unsigned> weights(n*k/8);
    for(unsigned r=0;r<n;++r)for(unsigned c=0;c<k;c+=8) {
      unsigned lo[2],hi[2]={};
      memcpy(lo,in.data()+size_t(r)*k/4+c/128*32+c%32,8);
      if(glm53::lowbit::bits(grid)==3)memcpy(hi,in.data()+size_t(n)*k/4+size_t(r)*k/8+c/256*32+c%32,8);
      unsigned packed=glm53::lowbit::packed8(lo[0],lo[1],hi[0],hi[1],c%128/32,grid,c%256/128*4);
      weights[(r*k+c)/8]=packed;
      for(unsigned j=0;j<8;++j)putchar((packed>>(4*j))&15);
    }
    const float fp4[]={0,.5,1,1.5,2,3,4,6};
    auto scales=in.data()+glm53::lowbit::scale_start(n,k,grid);
    float tensor;memcpy(&tensor,scales+size_t(n)*k/group,4);
    for(unsigned t=0;t<4;++t)for(unsigned r=0;r<n;++r) {
      float sums[32]={};
      for(unsigned lane=0;lane<32;++lane)
        for(unsigned c=lane*8;c<k;c+=256) {
          unsigned packed=weights[(r*k+c)/8];
          float multiplier=glm53::fp8(scales[(r*k+c)/group])*tensor;
          for(unsigned j=0;j<8;++j) {
            unsigned q=(packed>>(4*j))&15;
            float weight=(q&8?-fp4[q&7]:fp4[q&7])*multiplier;
            sums[lane]=std::fma(weight,from_bf(x[t*k+c+j]),sums[lane]);
          }
        }
      for(unsigned step=16;step;step/=2) {
        float prev[32];memcpy(prev,sums,sizeof(sums));
        for(unsigned lane=0;lane<32-step;++lane)sums[lane]+=prev[lane+step];
      }
      if(fwrite(sums,4,1,stdout)!=1)return 3;
    }
  }
  while(fread(h,sizeof(h),1,stdin)==1) {
    unsigned hidden=h[0],intermediate=h[1];
    glm53::lowbit::ExpertFormat f;
    memcpy(&f.gate_up,h+2,4);memcpy(&f.down,h+3,4);
    size_t input_bytes=2*glm53::lowbit::stride(intermediate,hidden,f.gate_up)
        +glm53::lowbit::stride(hidden,intermediate,f.down);
    std::vector<unsigned char> image(input_bytes);
    if(fread(image.data(),input_bytes,1,stdin)!=1)return 2;
    size_t step=glm53::lowbit::stride(intermediate,hidden,{0,0,0,0});
    for(unsigned part=0;part<3;++part)for(size_t i=0;i<step;++i)
      putchar(glm53::lowbit::convert_expert_byte(image.data(),i,part,f,hidden,intermediate));
  }
}
"#).unwrap();
    let build = Command::new("c++")
        .args(["-O2", "-std=c++20", "-I"])
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/cu/glm53"))
        .arg(&source)
        .arg("-o")
        .arg(dir.join("check"))
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let mut seed = 4242;
    let mut input = Vec::new();
    let mut expected = Vec::new();
    for grid in 0..5 {
        for group in [16, 32] {
            for (n, k) in [(1, 256), (33, 512), (129, 768), (9, 2048), (3, 4096)] {
                let w = Matrix::new(n, k, grid, group, &mut seed);
                for h in [n as u32, k as u32, u32::from(grid), u32::from(group)] {
                    input.extend(h.to_le_bytes());
                }
                input.extend(&w.image);
                let x = activation(4 * k, &mut seed);
                input.extend(x.iter().flat_map(|v| v.to_le_bytes()));
                expected.extend(w.to_nvfp4());
                expected.extend((0..n * k).map(|i| w.nibble(i)));
                for t in 0..4 {
                    for r in 0..n {
                        expected.extend(w.dot(&x[t * k..][..k], r).to_le_bytes());
                    }
                }
            }
        }
    }
    input.extend([0; 16]);
    let mut formats = vec![MatrixFormat::NVFP4];
    for grid in 0..5 {
        for group in [16, 32] {
            formats.push(MatrixFormat::new(grid, group).unwrap());
        }
    }
    let (h, i) = (512, 256);
    let nv_step = MatrixFormat::NVFP4.stride(i, h).unwrap();
    for idx in 0..formats.len() {
        for (gate_up, down) in [
            (formats[idx], formats[(idx + 4) % formats.len()]),
            (formats[idx], MatrixFormat::NVFP4),
            (MatrixFormat::NVFP4, formats[idx]),
        ] {
            for value in [h, i] {
                input.extend(value.to_le_bytes());
            }
            for f in [gate_up, down] {
                input.extend([f.format, f.grid, f.group, f.reserved]);
            }
            let step = gate_up.stride(i, h).unwrap();
            let mut image = vec![0xa5; 2 * step + down.stride(h, i).unwrap()];
            let mut nv = vec![0; 3 * nv_step];
            for (part, n, k, f) in [(0, i, h, gate_up), (1, i, h, gate_up), (2, h, i, down)] {
                let low = if f.format == 0 {
                    MatrixFormat::new(3, 16).unwrap()
                } else {
                    f
                };
                let w = Matrix::new(n as usize, k as usize, low.grid, low.group, &mut seed);
                let converted = w.to_nvfp4();
                let source = if f.format == 0 { &converted } else { &w.image };
                image[part * step..part * step + source.len()].copy_from_slice(source);
                nv[part * nv_step..part * nv_step + converted.len()].copy_from_slice(&converted);
            }
            input.extend(image);
            expected.extend(nv);
        }
    }
    fs::write(dir.join("input"), input).unwrap();
    let run = Command::new(dir.join("check"))
        .stdin(fs::File::open(dir.join("input")).unwrap())
        .output()
        .unwrap();
    assert!(run.status.success());
    assert_eq!(run.stdout.len(), expected.len());
    for (i, (&have, &want)) in run.stdout.iter().zip(&expected).enumerate() {
        assert_eq!(have, want, "byte {i}");
    }
    eprintln!(
        "LOWBIT CUDA helpers on CPU: {} bytes matched (conversion, nibbles, F32 GEMV)",
        expected.len()
    );
    fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use crate::glm53::{
        ffi as g,
        lowbit::{self, ExpertFormat, ffi as l},
    };
    use crate::runtime::{Device, DeviceBuffer, Stream, check};
    use hayate_core::Result;
    use std::ffi::c_void;

    fn upload(device: &Device, stream: &Stream, bytes: &[u8]) -> Result<DeviceBuffer> {
        let mut out = device.alloc(bytes.len())?;
        stream.copy_h2d(&mut out, bytes)?;
        Ok(out)
    }
    fn bytes(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn download(stream: &Stream, buffer: &DeviceBuffer) -> Result<Vec<u16>> {
        let mut out = vec![0; buffer.bytes()];
        stream.copy_d2h(&mut out, buffer)?;
        Ok(out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect())
    }
    fn formats_bytes(formats: &[ExpertFormat]) -> Vec<u8> {
        formats
            .iter()
            .flat_map(|e| [e.gate_up, e.down])
            .flat_map(|f| [f.format, f.grid, f.group, f.reserved])
            .collect()
    }
    fn weight(data: *const c_void, format: u32) -> g::DenseWeight {
        g::DenseWeight {
            data,
            scale: std::ptr::null(),
            format,
        }
    }

    #[test]
    #[ignore = "requires sm_120 GPU; all grids/groups, M=1..4, ragged rows and scale padding"]
    fn glm53_lowbit_gpu_reference() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let mut seed = 31;
        for grid in 0..5 {
            for group in [16, 32] {
                for (n, k) in [(9, 256), (129, 768), (9, 2048), (3, 4096)] {
                    let w = Matrix::new(n, k, grid, group, &mut seed);
                    let image = upload(&device, &stream, &w.image)?;
                    let expected_nv = w.to_nvfp4();
                    // Nonzero initial bytes catch unwritten padding/tensor bytes.
                    let mut nv = upload(&device, &stream, &vec![0xa5; expected_nv.len()])?;
                    lowbit::to_nvfp4(&mut nv, &image, n as u32, k as u32, w.f, &stream)?;
                    let mut actual_nv = vec![0; nv.bytes()];
                    stream.copy_d2h(&mut actual_nv, &nv)?;
                    assert_eq!(actual_nv, expected_nv, "grid={grid} group={group}");
                    for m in 1..=4 {
                        let x = activation(m * k, &mut seed);
                        let input = upload(&device, &stream, &bytes(&x))?;
                        let mut out = device.alloc(m * n * 2)?;
                        lowbit::gemv(
                            &mut out, &input, &image, w.f, m as u32, n as u32, k as u32, &stream,
                        )?;
                        let got = download(&stream, &out)?;
                        for t in 0..m {
                            for r in 0..n {
                                assert_eq!(
                                    got[t * n + r],
                                    bf(w.dot(&x[t * k..][..k], r)),
                                    "grid={grid} group={group} m={m} t={t} r={r}"
                                );
                            }
                        }
                        // NVFP4 dispatch calls the existing dot implementation with its exact order.
                        let mut old = device.alloc(m * n * 2)?;
                        lowbit::gemv(
                            &mut out,
                            &input,
                            &nv,
                            MatrixFormat::NVFP4,
                            m as u32,
                            n as u32,
                            k as u32,
                            &stream,
                        )?;
                        check(
                            unsafe {
                                g::hayate_glm_dense_batch(
                                    old.as_mut_ptr(),
                                    input.as_ptr(),
                                    weight(nv.as_ptr(), 3),
                                    m as u32,
                                    n as u32,
                                    k as u32,
                                    0,
                                    stream.as_raw(),
                                )
                            },
                            "old NVFP4 GEMV",
                        )?;
                        assert_eq!(got, download(&stream, &out)?, "LOWBIT vs transcoded NVFP4");
                        assert_eq!(download(&stream, &out)?, download(&stream, &old)?);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires sm_120 GPU; mixed nine-expert gate/up/down, rotation input and misses"]
    fn glm53_lowbit_gpu_moe() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let (h, i) = (512usize, 256usize);
        let mut seed = 777;
        let mut images = Vec::new();
        let mut formats = Vec::new();
        let mut matrices = Vec::new();
        for e in 0..8 {
            let a = Matrix::new(
                i,
                h,
                (e % 5) as u8,
                if e % 2 == 0 { 16 } else { 32 },
                &mut seed,
            );
            let b = Matrix::new(i, h, (e % 5) as u8, a.f.group, &mut seed);
            let d = Matrix::new(
                h,
                i,
                ((e + 2) % 5) as u8,
                if e % 2 == 0 { 32 } else { 16 },
                &mut seed,
            );
            let f = ExpertFormat {
                gate_up: if e == 0 { MatrixFormat::NVFP4 } else { a.f },
                down: if e == 1 { MatrixFormat::NVFP4 } else { d.f },
            };
            let step = f.gate_up.stride(i as u32, h as u32)?;
            let mut raw = vec![0; 2 * step + f.down.stride(h as u32, i as u32)?];
            for (part, matrix, format) in [(0, &a, f.gate_up), (1, &b, f.gate_up), (2, &d, f.down)]
            {
                let payload = if format.format == 0 {
                    matrix.to_nvfp4()
                } else {
                    matrix.image.clone()
                };
                raw[part * step..part * step + payload.len()].copy_from_slice(&payload);
            }
            images.push(upload(&device, &stream, &raw)?);
            let nv_step = MatrixFormat::NVFP4.stride(i as u32, h as u32)?;
            let mut scratch = upload(&device, &stream, &vec![0xa5; 3 * nv_step])?;
            lowbit::expert_to_nvfp4(
                &mut scratch,
                images.last().unwrap(),
                f,
                h as u32,
                i as u32,
                &stream,
            )?;
            let mut expected = vec![0; 3 * nv_step];
            for (part, matrix) in [&a, &b, &d].into_iter().enumerate() {
                let nv = matrix.to_nvfp4();
                expected[part * nv_step..part * nv_step + nv.len()].copy_from_slice(&nv);
            }
            let mut actual = vec![0; scratch.bytes()];
            stream.copy_d2h(&mut actual, &scratch)?;
            assert_eq!(actual, expected, "expert transcode {e}");

            formats.push(f);
            matrices.push((a, b, d));
        }
        let dformats = upload(&device, &stream, &formats_bytes(&formats))?;
        // BF16 shared weights: exact reductions, and original != routed input.
        let shared_gate = upload(&device, &stream, &bf(0.125).to_le_bytes().repeat(h * i))?;
        let shared_up = upload(&device, &stream, &bf(-0.25).to_le_bytes().repeat(h * i))?;
        let shared_down = upload(&device, &stream, &bf(0.0625).to_le_bytes().repeat(h * i))?;
        for m in 1..=4 {
            let x = activation(m * h, &mut seed);
            let original: Vec<u16> = x.iter().map(|&v| bf(-unbf(v) * 2.)).collect();
            let dx = upload(&device, &stream, &bytes(&x))?;
            let doriginal = upload(&device, &stream, &bytes(&original))?;
            for missed in [false, true] {
                let table: Vec<u8> = images
                    .iter()
                    .enumerate()
                    .flat_map(|(e, b)| {
                        (if missed && e == 7 {
                            0
                        } else {
                            b.as_ptr() as u64
                        })
                        .to_le_bytes()
                    })
                    .collect();
                let table = upload(&device, &stream, &table)?;
                let mut product = device.alloc(9 * m * i * 2)?;
                let mut out = upload(&device, &stream, &bf(-123.).to_le_bytes().repeat(9 * m * h))?;
                check(
                    unsafe {
                        l::hayate_glm_lowbit_gate_up(
                            product.as_mut_ptr(),
                            dx.as_ptr(),
                            doriginal.as_ptr(),
                            table.as_ptr(),
                            dformats.as_ptr().cast(),
                            weight(shared_gate.as_ptr(), 0),
                            weight(shared_up.as_ptr(), 0),
                            m as u32,
                            h as u32,
                            i as u32,
                            stream.as_raw(),
                        )
                    },
                    "mixed gate/up",
                )?;
                let got = download(&stream, &product)?;
                for e in 0..9 {
                    for t in 0..m {
                        for r in 0..i {
                            if missed && e == 7 {
                                assert_eq!(got[(e * m + t) * i + r], 0);
                                continue;
                            }
                            let (a, b) = if e == 8 {
                                let sum: f32 =
                                    original[t * h..][..h].iter().map(|&v| unbf(v)).sum();
                                (unbf(bf(sum * 0.125)), unbf(bf(sum * -0.25)))
                            } else {
                                (
                                    unbf(bf(matrices[e].0.dot(&x[t * h..][..h], r))),
                                    unbf(bf(matrices[e].1.dot(&x[t * h..][..h], r))),
                                )
                            };
                            let a = a.min(10.);
                            let want =
                                unbf(bf(unbf(bf(a / (1. + (-a).exp()))) * b.clamp(-10., 10.)));
                            let have = unbf(got[(e * m + t) * i + r]);
                            assert!(
                                (have - want).abs() <= 0.008 * want.abs().max(1e-10),
                                "gate/up e={e} m={m} r={r}: {have} vs {want}"
                            );
                        }
                    }
                }
                check(
                    unsafe {
                        l::hayate_glm_lowbit_down(
                            out.as_mut_ptr(),
                            product.as_ptr(),
                            table.as_ptr(),
                            dformats.as_ptr().cast(),
                            weight(shared_down.as_ptr(), 0),
                            m as u32,
                            h as u32,
                            i as u32,
                            stream.as_raw(),
                        )
                    },
                    "mixed down",
                )?;
                let result = download(&stream, &out)?;
                for e in 0..9 {
                    for t in 0..m {
                        let input = &got[(e * m + t) * i..][..i];
                        for r in 0..h {
                            let want = if missed && e == 7 {
                                bf(-123.)
                            } else if e == 8 {
                                let sum: f32 = input.iter().map(|&v| unbf(v) * 0.0625).sum();
                                bf(sum)
                            } else {
                                bf(matrices[e].2.dot(input, r))
                            };
                            // Shared reduction order can differ from the scalar sum.
                            if e == 8 {
                                assert!(
                                    (unbf(result[(e * m + t) * h + r]) - unbf(want)).abs()
                                        <= 0.01 * unbf(want).abs().max(1e-8)
                                );
                            } else {
                                assert_eq!(
                                    result[(e * m + t) * h + r],
                                    want,
                                    "down e={e} m={m} r={r}"
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // A full-sized all-NVFP4 comparison against both old fused launches, including
    // real mixed signs/scales, shared FP8, clipped gate/up and a missing expert.
    #[test]
    #[ignore = "requires sm_120 GPU; bit-exact existing fused NVFP4 decode"]
    fn glm53_lowbit_gpu_nvfp4_bitwise() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let (h, i) = (4096usize, 2048usize);
        let f = MatrixFormat::NVFP4;
        let step = f.stride(i as u32, h as u32)?;
        let mut seed = 23;
        let mut host = vec![0; 3 * step];
        for part in 0..3 {
            let (n, k) = if part == 2 { (h, i) } else { (i, h) };
            let len = f.storage_bytes(n as u32, k as u32)?;
            let raw = &mut host[part * step..part * step + len];
            for byte in &mut raw[..n * k / 2] {
                *byte = random(&mut seed) as u8;
            }
            for byte in &mut raw[n * k / 2..len - 4] {
                *byte = [0, 1, 7, 8, 0x19, 0x31, 0x3f, 0x68, 0x7e][random(&mut seed) as usize % 9];
            }
            raw[len - 4..].copy_from_slice(&0.00731f32.to_le_bytes());
        }
        let image = upload(&device, &stream, &host)?;
        let formats = upload(
            &device,
            &stream,
            &formats_bytes(
                &[ExpertFormat {
                    gate_up: f,
                    down: f,
                }; 8],
            ),
        )?;
        let x = upload(&device, &stream, &bytes(&activation(4 * h, &mut seed)))?;
        let original = upload(&device, &stream, &bytes(&activation(4 * h, &mut seed)))?;
        let mut raw_shared = vec![0x31; h * i];
        raw_shared.extend(0.019f32.to_le_bytes());
        let shared = upload(&device, &stream, &raw_shared)?;
        for missed in [false, true] {
            let table: Vec<u8> = (0..8)
                .flat_map(|e| {
                    (if missed && e == 4 {
                        0
                    } else {
                        image.as_ptr() as u64
                    })
                    .to_le_bytes()
                })
                .collect();
            let table = upload(&device, &stream, &table)?;
            for m in 1..=4 {
                let mut b = device.alloc(9 * m * i * 2)?;
                let mut db = upload(&device, &stream, &vec![0x5a; 9 * m * h * 2])?;
                let mut expected_gate = vec![0; 9 * m * i];
                let mut expected_down = vec![0; 9 * m * h];
                for t in 0..m {
                    let mut a = device.alloc(9 * i * 2)?;
                    let mut da = upload(&device, &stream, &vec![0x5a; 9 * h * 2])?;
                    check(
                        unsafe {
                            g::hayate_glm_moe_gate_up(
                                a.as_mut_ptr(),
                                x.as_ptr().byte_add(t * h * 2),
                                original.as_ptr().byte_add(t * h * 2),
                                table.as_ptr(),
                                weight(shared.as_ptr(), 2),
                                weight(shared.as_ptr(), 2),
                                1,
                                0,
                                step,
                                stream.as_raw(),
                            )
                        },
                        "old fused gate/up",
                    )?;
                    check(
                        unsafe {
                            g::hayate_glm_moe_down(
                                da.as_mut_ptr(),
                                a.as_ptr(),
                                table.as_ptr(),
                                weight(shared.as_ptr(), 2),
                                1,
                                0,
                                step,
                                stream.as_raw(),
                            )
                        },
                        "old fused down",
                    )?;
                    let old_gate = download(&stream, &a)?;
                    let old_down = download(&stream, &da)?;
                    for e in 0..9 {
                        expected_gate[(e * m + t) * i..][..i]
                            .copy_from_slice(&old_gate[e * i..][..i]);
                        expected_down[(e * m + t) * h..][..h]
                            .copy_from_slice(&old_down[e * h..][..h]);
                    }
                }
                check(
                    unsafe {
                        l::hayate_glm_lowbit_gate_up(
                            b.as_mut_ptr(),
                            x.as_ptr(),
                            original.as_ptr(),
                            table.as_ptr(),
                            formats.as_ptr().cast(),
                            weight(shared.as_ptr(), 2),
                            weight(shared.as_ptr(), 2),
                            m as u32,
                            h as u32,
                            i as u32,
                            stream.as_raw(),
                        )
                    },
                    "new fused gate/up",
                )?;
                assert_eq!(expected_gate, download(&stream, &b)?, "gate/up m={m}");
                check(
                    unsafe {
                        l::hayate_glm_lowbit_down(
                            db.as_mut_ptr(),
                            b.as_ptr(),
                            table.as_ptr(),
                            formats.as_ptr().cast(),
                            weight(shared.as_ptr(), 2),
                            m as u32,
                            h as u32,
                            i as u32,
                            stream.as_raw(),
                        )
                    },
                    "new fused down",
                )?;
                assert_eq!(expected_down, download(&stream, &db)?, "down m={m}");
            }
        }
        Ok(())
    }

    fn timed(stream: &Stream, mut submit: impl FnMut(usize) -> Result<()>) -> Result<f64> {
        for r in 0..4 {
            submit(r)?;
        }
        let mut trials = Vec::new();
        for _ in 0..3 {
            let start = stream.event()?;
            let end = stream.event()?;
            start.record(stream)?;
            for r in 0..64 {
                submit(r)?;
            }
            end.record(stream)?;
            end.synchronize()?;
            trials.push(f64::from(start.elapsed_ms(&end)?) * 1000. / 64.);
        }
        trials.sort_by(f64::total_cmp);
        eprintln!("trials_us={trials:?}");
        Ok(trials[1])
    }
    fn report(f: MatrixFormat, m: u32, kernel: &str, us: f64, bytes: usize) {
        eprintln!(
            "glm_lowbit_bench format={} grid={} group={} m={m} kernel={kernel} median_us={us:.3} bytes={bytes} tb_s={:.3}",
            f.format,
            f.grid,
            f.group,
            bytes as f64 / us / 1e6
        );
    }
    fn filled_image(n: usize, k: usize, f: MatrixFormat) -> Vec<u8> {
        let len = f.storage_bytes(n as u32, k as u32).unwrap();
        let mut raw = vec![0x55; len];
        let at = if f.format == 0 {
            n * k / 2
        } else {
            n * k / 8 * if f.format == 1 { 3 } else { 2 }
        };
        raw[at..len - 4].fill(0x30);
        raw[len - 4..].copy_from_slice(&0.01f32.to_le_bytes());
        raw
    }
    #[test]
    #[ignore = "requires sm_120 GPU; real shapes, >512 MiB weight rotation, three timing trials"]
    fn glm53_lowbit_gpu_bench() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        let (h, i) = (4096usize, 2048usize);
        let x = upload(&device, &stream, &bf(0.25).to_le_bytes().repeat(4 * h))?;
        let mut product = device.alloc(9 * 4 * i * 2)?;
        let mut out = device.alloc(9 * 4 * h * 2)?;
        let mut all_formats = vec![MatrixFormat::NVFP4];
        for grid in 0..5 {
            for group in [16, 32] {
                all_formats.push(MatrixFormat::new(grid, group)?);
            }
        }
        for f in all_formats {
            let step = f.stride(i as u32, h as u32)?;
            let down_step = f.stride(h as u32, i as u32)?;
            let slot = 2 * step + down_step;
            let sets = (512usize << 20).div_ceil(8 * slot).max(2);
            let mut raw = vec![0; slot];
            for (offset, n, k) in [(0, i, h), (step, i, h), (2 * step, h, i)] {
                let image = filled_image(n, k, f);
                raw[offset..offset + image.len()].copy_from_slice(&image);
            }
            let mut images = Vec::new();
            let mut tables = Vec::new();
            let mut shared = Vec::new();
            let mut shared_raw = vec![0x30; h * i];
            shared_raw.extend(0.01f32.to_le_bytes());
            for _ in 0..sets {
                let mut pointers = Vec::new();
                for _ in 0..8 {
                    images.push(upload(&device, &stream, &raw)?);
                    pointers.extend((images.last().unwrap().as_ptr() as u64).to_le_bytes());
                }
                tables.push(upload(&device, &stream, &pointers)?);
                for _ in 0..3 {
                    shared.push(upload(&device, &stream, &shared_raw)?);
                }
            }
            let formats = upload(
                &device,
                &stream,
                &formats_bytes(
                    &[ExpertFormat {
                        gate_up: f,
                        down: f,
                    }; 8],
                ),
            )?;
            if let Ok(load) = std::process::Command::new("uptime").output() {
                eprintln!("{}", String::from_utf8_lossy(&load.stdout).trim());
            }
            for m in 1..=4 {
                let us = timed(&stream, |r| {
                    let s = r % sets;
                    check(
                        unsafe {
                            l::hayate_glm_lowbit_gate_up(
                                product.as_mut_ptr(),
                                x.as_ptr(),
                                x.as_ptr(),
                                tables[s].as_ptr(),
                                formats.as_ptr().cast(),
                                weight(shared[3 * s].as_ptr(), 2),
                                weight(shared[3 * s + 1].as_ptr(), 2),
                                m,
                                h as u32,
                                i as u32,
                                stream.as_raw(),
                            )
                        },
                        "bench LOWBIT gate/up",
                    )
                })?;
                report(
                    f,
                    m,
                    "gate_up",
                    us,
                    16 * f.storage_bytes(i as u32, h as u32)? + 2 * (h * i + 4),
                );
                let us = timed(&stream, |r| {
                    let s = r % sets;
                    check(
                        unsafe {
                            l::hayate_glm_lowbit_down(
                                out.as_mut_ptr(),
                                product.as_ptr(),
                                tables[s].as_ptr(),
                                formats.as_ptr().cast(),
                                weight(shared[3 * s + 2].as_ptr(), 2),
                                m,
                                h as u32,
                                i as u32,
                                stream.as_raw(),
                            )
                        },
                        "bench LOWBIT down",
                    )
                })?;
                report(
                    f,
                    m,
                    "down",
                    us,
                    8 * f.storage_bytes(h as u32, i as u32)? + h * i + 4,
                );
            }
            if f.format != 0 {
                // Rotate destinations too: a single 14 MB scratch would stay in
                // L2 and overstate DRAM write bandwidth on the conversion path.
                let scratch_bytes = 3 * MatrixFormat::NVFP4.stride(i as u32, h as u32)?;
                let mut scratch: Vec<_> = (0..(512usize << 20).div_ceil(scratch_bytes))
                    .map(|_| device.alloc(scratch_bytes))
                    .collect::<Result<_>>()?;
                let us = timed(&stream, |r| {
                    let dst = r % scratch.len();
                    check(
                        unsafe {
                            l::hayate_glm_lowbit_convert_expert(
                                scratch[dst].as_mut_ptr(),
                                images[r % images.len()].as_ptr(),
                                ExpertFormat {
                                    gate_up: f,
                                    down: f,
                                },
                                h as u32,
                                i as u32,
                                stream.as_raw(),
                            )
                        },
                        "bench LOWBIT expert transcode",
                    )
                })?;
                report(
                    f,
                    0,
                    "expert_to_nvfp4",
                    us,
                    2 * f.storage_bytes(i as u32, h as u32)?
                        + f.storage_bytes(h as u32, i as u32)?
                        + scratch_bytes,
                );
            }
        }
        Ok(())
    }
}
