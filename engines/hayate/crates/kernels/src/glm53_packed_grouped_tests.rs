#[cfg(feature = "cuda")]
use super::packed_grouped;
use super::{
    lowbit::{ExpertFormat, MatrixFormat},
    packed_grouped::validate,
};
#[cfg(feature = "cuda")]
use hayate_core::Result;

const FP4: [f32; 8] = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
const NIBBLES: [&[u8]; 5] = [
    &[13, 10, 2, 5],
    &[12, 10, 0, 2],
    &[14, 13, 12, 10, 0, 2, 4, 5],
    &[14, 13, 11, 9, 1, 3, 5, 6],
    &[14, 10, 2, 6],
];
fn bf(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
fn unbf(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}
fn fp8(x: u8) -> f32 {
    let e = (x >> 3) & 15;
    let m = x & 7;
    let value = if e == 0 {
        f32::from(m) / 512.
    } else {
        (1. + f32::from(m) / 8.) * 2f32.powi(i32::from(e) - 7)
    };
    if x & 128 == 0 { value } else { -value }
}
fn random(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}
fn formats() -> Vec<MatrixFormat> {
    let mut fs = vec![MatrixFormat::NVFP4];
    for grid in 0..5 {
        for group in [16, 32] {
            fs.push(MatrixFormat::new(grid, group).unwrap());
        }
    }
    fs
}
// Construct independent codes and logical scales before packing. The NVFP4
// version is the legacy convert -> restore_unscaled CPU reference.
fn matrix(n: u32, k: u32, f: MatrixFormat, seed: &mut u32) -> (Vec<u8>, Vec<u8>) {
    let (n, k) = (n as usize, k as usize);
    let mut packed = vec![0; f.storage_bytes(n as u32, k as u32).unwrap()];
    let mut nv = vec![
        0;
        MatrixFormat::NVFP4
            .storage_bytes(n as u32, k as u32)
            .unwrap()
    ];
    for r in 0..n {
        for c in 0..k {
            let code = random(seed) as usize;
            let q = if f.format == 0 {
                (code & 15) as u8
            } else {
                NIBBLES[f.grid as usize][code % NIBBLES[f.grid as usize].len()]
            };
            nv[(r * k + c) / 2] |= q << (4 * (c % 2));
            if f.format != 0 {
                let code = (code % NIBBLES[f.grid as usize].len()) as u8;
                packed[r * (k / 4) + c / 128 * 32 + c % 32] |= (code & 3) << (2 * (c % 128 / 32));
                if f.format == 1 {
                    packed[n * k / 4 + r * (k / 8) + c / 256 * 32 + c % 32] |=
                        (code >> 2) << (c % 256 / 32);
                }
            }
        }
    }
    let group = if f.format == 0 {
        16
    } else {
        usize::from(f.group)
    };
    let start = n * k / 8 * if f.format == 1 { 3 } else { 2 };
    for r in 0..n {
        for c in 0..k / group {
            let scale =
                [0, 1, 7, 8, 0x19, 0x31, 0x3f, 0x50, 0x68, 0x7e][random(seed) as usize % 10];
            if f.format != 0 {
                packed[start + r * k / group + c] = scale;
            }
            for s in c * (group / 16)..(c + 1) * (group / 16) {
                let at =
                    ((r / 128) * (k / 64) + s / 4) * 512 + r % 32 * 16 + r % 128 / 32 * 4 + s % 4;
                nv[n * k / 2 + at] = scale;
            }
        }
    }
    // Deliberately not BF16: folding this into weights would change rounding.
    let tensor = (0.0007312346 + (random(seed) % 7) as f32 * 0.0001).to_le_bytes();
    let len = nv.len();
    nv[len - 4..].copy_from_slice(&tensor);
    if f.format == 0 {
        packed.copy_from_slice(&nv);
    } else {
        let len = packed.len();
        packed[len - 4..].copy_from_slice(&tensor);
    }
    (packed, nv)
}
fn expert(h: u32, i: u32, f: ExpertFormat, seed: &mut u32) -> (Vec<u8>, [Vec<u8>; 3]) {
    let step = f.gate_up.stride(i, h).unwrap();
    let mut image = vec![0xa5; 2 * step + f.down.stride(h, i).unwrap()];
    let mut reference = [Vec::new(), Vec::new(), Vec::new()];
    for (part, n, k, f) in [
        (0, i, h, f.gate_up),
        (1, i, h, f.gate_up),
        (2, h, i, f.down),
    ] {
        let (packed, nv) = matrix(n, k, f, seed);
        image[part * step..part * step + packed.len()].copy_from_slice(&packed);
        reference[part] = nv;
    }
    (image, reference)
}
fn weight(nv: &[u8], r: usize, c: usize, n: usize, k: usize) -> f32 {
    let q = (nv[(r * k + c) / 2] >> (4 * (c % 2))) & 15;
    let s = c / 16;
    let at = ((r / 128) * (k / 64) + s / 4) * 512 + r % 32 * 16 + r % 128 / 32 * 4 + s % 4;
    let value = FP4[(q & 7) as usize] * fp8(nv[n * k / 2 + at]);
    if q & 8 == 0 { value } else { -value }
}
fn gemm(x: &[u16], nv: &[u8], rows: usize, n: usize, k: usize) -> Vec<u16> {
    let tensor = f32::from_le_bytes(nv[nv.len() - 4..].try_into().unwrap());
    let mut out = vec![0; rows * n];
    for t in 0..rows {
        for r in 0..n {
            let mut sum = 0f32;
            for c in 0..k {
                sum = unbf(x[t * k + c]).mul_add(weight(nv, r, c, n, k), sum);
            }
            out[t * n + r] = bf(sum * tensor);
        }
    }
    out
}
fn activation(count: usize, seed: &mut u32) -> Vec<u16> {
    (0..count)
        .map(|_| bf((random(seed) % 257) as f32 / 129. - 1.))
        .collect()
}
fn bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn point(gate: &[u16], up: &[u16]) -> Vec<u16> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            let g = unbf(g).min(10.);
            let u = unbf(u).clamp(-10., 10.);
            bf(unbf(bf(g * (1. / (1. + (-g).exp())))) * u)
        })
        .collect()
}

#[test]
fn glm53_packed_grouped_validation() {
    let f = ExpertFormat {
        gate_up: MatrixFormat::NVFP4,
        down: MatrixFormat::NVFP4,
    };
    assert_eq!(validate(&[f; 288], &[[0, 0]; 288], 4096, 2048).unwrap(), 0);
    assert_eq!(
        validate(&[f; 2], &[[65536, 0], [0, 65536]], 4096, 2048).unwrap(),
        65536
    );
    for (fs, rs, h, i) in [
        (vec![], vec![], 4096, 2048),
        (vec![f; 289], vec![[0, 0]; 289], 4096, 2048),
        (vec![f], vec![[1, 1]], 4096, 2048),
        (vec![f; 2], vec![[1, 0]; 2], 4096, 2048),
        (vec![f], vec![[65537, 0]], 4096, 2048),
        (vec![f], vec![[1, 0]], 4095, 2048),
        (vec![f], vec![[1, 0]], 4096, 0),
        (vec![f], vec![[1, 0]], 4352, 2048),
        (vec![f], vec![], 4096, 2048),
    ] {
        assert!(validate(&fs, &rs, h, i).is_err());
    }
    let invalid = ExpertFormat {
        down: MatrixFormat {
            reserved: 1,
            ..f.down
        },
        ..f
    };
    assert!(validate(&[invalid], &[[1, 0]], 256, 512).is_err());
}

#[test]
fn glm53_packed_grouped_cpu_reference() {
    use std::{fs, process::Command};
    let dir = std::env::temp_dir().join(format!("hayate-packed-grouped-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("check.cpp"),
        r#"
#include "packed_grouped.cu"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cassert>
using namespace glm53;
using namespace glm53::lowbit;
using namespace glm53::packed_grouped;
float value(unsigned pair,unsigned half) {
  const float levels[]={0,.5,1,1.5,2,3,4,6};
  unsigned q=(pair>>(4*half))&15;
  return (q&8?-levels[q&7]:levels[q&7])*fp8(pair>>8);
}
void emit_bf(float x) {
  float v=bf16(x);unsigned bits;memcpy(&bits,&v,4);unsigned short b=bits>>16;
  assert(fwrite(&b,2,1,stdout)==1);
}
std::vector<float> project(const std::vector<float>& x,const unsigned char* image,MatrixFormat f,
    unsigned rows,unsigned n,unsigned k) {
  std::vector<float> out(rows*n);
  float scale;memcpy(&scale,image+storage(n,k,f)-4,4);
  for(unsigned t=0;t<rows;++t)for(unsigned r=0;r<n;++r) {
    float sum=0;
    for(unsigned c=0;c<k;c+=2) {
      unsigned pair=weight_pair(image,r,c,n,k,f);
      sum=std::fma(x[t*k+c],value(pair,0),sum);
      sum=std::fma(x[t*k+c+1],value(pair,1),sum);
    }
    out[t*n+r]=bf16(sum*scale);
  }
  return out;
}
template<class C> void check_scheduler() {
  for(unsigned n: {256u,2048u,4096u}) {
    unsigned ends[288],sum=0,rows[288];
    for(unsigned j=0;j<288;++j) {
      rows[j]=j==7?1500:j%5==0?0:j%131;sum+=tile_count(rows[j],n,C::m,C::n);ends[j]=sum;
    }
    std::vector<unsigned> seen(sum,0);
    for(unsigned block=0;block<BLOCKS;++block)for(unsigned tile=block;tile<sum;tile+=BLOCKS) {
      unsigned j=find_expert(ends,288,tile);assert(j<288&&rows[j]);
      unsigned local=tile-(j?ends[j-1]:0);
      assert(local/(n/C::n)*C::m<rows[j]);assert(local%(n/C::n)*C::n<n);++seen[tile];
    }
    for(auto v:seen)assert(v==1);
  }
}
template<class C> void check_staging(const unsigned char* image,unsigned n,unsigned k,MatrixFormat f) {
  assert(C::shared(false)+288*4+3*4<=99*1024);
  std::vector<unsigned char> stage(C::weight);
  for(unsigned n0=0;n0<n;n0+=C::n)for(unsigned base=0;base<k;base+=C::k) {
    std::fill(stage.begin(),stage.end(),0xa5);
    for(unsigned r=0;r<C::n;++r) {
      for(unsigned byte=0;byte<payload_bytes(C::k,f);byte+=16) {
        auto src=payload_source(n0+r,byte,base,n,k,C::k,f);
        assert(src%16==0&&src+16<=storage(n,k,f)-4);
        memcpy(stage.data()+r*C::payload+byte,image+src,16);
      }
      for(unsigned byte=0;byte<C::k/(f.format?f.group:16);byte+=4) {
        auto src=scale_source(n0+r,byte,base,n,k,f);
        assert(src%4==0&&src+4<=storage(n,k,f)-4);
        memcpy(stage.data()+C::n*C::payload+r*C::scales+byte,image+src,4);
      }
      for(unsigned c=0;c<C::k;c+=2)
        assert(staged_pair(stage.data()+r*C::payload,stage.data()+C::n*C::payload+r*C::scales,
            c,base,C::k,f)==weight_pair(image,n0+r,base+c,n,k,f));
    }
  }
}
int main() {
  check_scheduler<C0>();check_scheduler<C1>();check_scheduler<C2>();check_scheduler<C3>();
  unsigned dims[3];
  while(fread(dims,sizeof(dims),1,stdin)==1) {
    unsigned h=dims[0],i=dims[1],rows=dims[2];ExpertFormat f;
    assert(fread(&f,sizeof(f),1,stdin)==1);
    size_t step=stride(i,h,f.gate_up);
    std::vector<unsigned char> image(2*step+stride(h,i,f.down));
    assert(fread(image.data(),image.size(),1,stdin)==1);
    // Exhaustive weight-pair comparison covers zeros, subnormal scales, signs,
    // every grid/group and the tiled NVFP4 scales, independently of GEMM.
    for(unsigned part=0;part<3;++part) {
      unsigned n=part==2?h:i,k=part==2?i:h;
      MatrixFormat mf=part==2?f.down:f.gate_up;
      check_staging<C0>(image.data()+part*step,n,k,mf);
      check_staging<C1>(image.data()+part*step,n,k,mf);
      check_staging<C2>(image.data()+part*step,n,k,mf);
      check_staging<C3>(image.data()+part*step,n,k,mf);
      for(unsigned r=0;r<n;++r)for(unsigned c=0;c<k;c+=2) {
        unsigned pair=weight_pair(image.data()+part*step,r,c,n,k,mf);
        for(unsigned half=0;half<2;++half)emit_bf(value(pair,half));
      }
    }
    std::vector<unsigned short> raw(rows*h);assert(fread(raw.data(),raw.size()*2,1,stdin)==1);
    std::vector<float> x(rows*h);
    for(unsigned j=0;j<raw.size();++j) {unsigned b=unsigned(raw[j])<<16;memcpy(&x[j],&b,4);}
    auto a=project(x,image.data(),f.gate_up,rows,i,h);
    auto b=project(x,image.data()+step,f.gate_up,rows,i,h);
    for(unsigned j=0;j<a.size();++j) {a[j]=swiglu(a[j],b[j]);emit_bf(a[j]);}
    auto out=project(a,image.data()+2*step,f.down,rows,h,i);
    for(auto v:out)emit_bf(v);
  }
}
"#,
    )
    .unwrap();
    let build = Command::new("c++")
        .args(["-O2", "-std=c++20", "-I"])
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/cu/glm53"))
        .arg(dir.join("check.cpp"))
        .arg("-o")
        .arg(dir.join("check"))
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let fs = formats();
    let mut seed = 7919;
    let mut input = Vec::new();
    let mut expected = Vec::new();
    for (index, &gu) in fs.iter().enumerate() {
        let f = ExpertFormat {
            gate_up: gu,
            down: fs[(index + 4) % fs.len()],
        };
        let (h, i, rows) = (512u32, 256u32, 3usize);
        let (image, nv) = expert(h, i, f, &mut seed);
        input.extend([h, i, rows as u32].iter().flat_map(|v| v.to_le_bytes()));
        for m in [f.gate_up, f.down] {
            input.extend([m.format, m.grid, m.group, m.reserved]);
        }
        input.extend(image);
        for (part, n, k) in [(0, i, h), (1, i, h), (2, h, i)] {
            for r in 0..n as usize {
                for c in 0..k as usize {
                    let v = weight(&nv[part], r, c, n as usize, k as usize);
                    // No information lost by the register BF16 expansion.
                    assert_eq!(unbf(bf(v)).to_bits(), v.to_bits());
                    expected.extend(bf(v).to_le_bytes());
                }
            }
        }
        let x = activation(rows * h as usize, &mut seed);
        input.extend(bytes(&x));
        let gate = gemm(&x, &nv[0], rows, i as usize, h as usize);
        let up = gemm(&x, &nv[1], rows, i as usize, h as usize);
        let product = point(&gate, &up);
        let out = gemm(&product, &nv[2], rows, h as usize, i as usize);
        expected.extend(bytes(&product));
        expected.extend(bytes(&out));
    }
    fs::write(dir.join("input"), input).unwrap();
    let run = Command::new(dir.join("check"))
        .stdin(fs::File::open(dir.join("input")).unwrap())
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(run.stdout.len(), expected.len());
    for (index, (&actual, &want)) in run.stdout.iter().zip(&expected).enumerate() {
        assert_eq!(actual, want, "CPU reference byte {index}");
    }
    eprintln!(
        "packed CPU reference: 4 staging/scheduler configurations, 11 mixed formats, {} bytes bit exact; max_relative_error=0",
        expected.len()
    );
    fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use crate::glm53::{ffi as g, lowbit::ffi as l, packed_grouped::ffi};
    use crate::runtime::{Device, DeviceBuffer, Stream, check};
    use std::ffi::c_void;

    fn upload(device: &Device, stream: &Stream, raw: &[u8]) -> Result<DeviceBuffer> {
        let mut out = device.alloc(raw.len())?;
        stream.copy_h2d(&mut out, raw)?;
        Ok(out)
    }
    fn download(stream: &Stream, buffer: &DeviceBuffer) -> Result<Vec<u16>> {
        let mut raw = vec![0; buffer.bytes()];
        stream.copy_d2h(&mut raw, buffer)?;
        Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes(*v))
            .collect())
    }
    fn at(buffer: &DeviceBuffer, offset: usize) -> *mut c_void {
        assert!(offset < buffer.bytes());
        unsafe { buffer.as_ptr().cast::<u8>().add(offset).cast_mut().cast() }
    }
    fn ranges(counts: &[u32]) -> Vec<[u32; 2]> {
        let mut offset = 0;
        counts
            .iter()
            .map(|&rows| {
                let pair = [rows, offset];
                offset += rows;
                pair
            })
            .collect()
    }
    fn load() -> String {
        std::process::Command::new("uptime")
            .output()
            .map(|r| String::from_utf8_lossy(&r.stdout).trim().to_owned())
            .unwrap()
    }
    fn compare(label: &str, actual: &[u16], expected: &[u16]) {
        assert_eq!(actual.len(), expected.len());
        let (mut error, mut norm, mut max_relative, mut max_absolute, mut different) =
            (0f64, 0f64, 0f64, 0f64, 0usize);
        let (mut max_relative_nonzero, mut max_absolute_at_zero) = (0f64, 0f64);
        for (&a, &b) in actual.iter().zip(expected) {
            let (a, b) = (f64::from(unbf(a)), f64::from(unbf(b)));
            assert!(a.is_finite() && b.is_finite());
            let diff = (a - b).abs();
            max_absolute = max_absolute.max(diff);
            max_relative = max_relative.max(diff / b.abs().max(1e-6));
            if b == 0. {
                max_absolute_at_zero = max_absolute_at_zero.max(diff);
            } else {
                max_relative_nonzero = max_relative_nonzero.max(diff / b.abs());
            }
            error += diff * diff;
            norm += b * b;
            different += usize::from(a.to_bits() != b.to_bits());
        }
        let l2 = (error / norm.max(1e-30)).sqrt();
        eprintln!(
            "{label}: values={} differing={different} max_absolute={max_absolute:e} max_relative_nonzero={max_relative_nonzero:e} max_absolute_at_zero={max_absolute_at_zero:e} max_relative_floor_1e-6={max_relative:e} relative_l2={l2:e}",
            actual.len()
        );
        assert_eq!(different, 0, "{label}: packed GEMM must remain bit exact");
    }

    #[derive(Debug)]
    struct Case {
        h: u32,
        i: u32,
        formats: Vec<ExpertFormat>,
        ranges: Vec<[u32; 2]>,
        images: Vec<DeviceBuffer>,
        pointers: DeviceBuffer,
        descriptors: DeviceBuffer,
        device_ranges: DeviceBuffer,
        x: DeviceBuffer,
        product: DeviceBuffer,
        out: DeviceBuffer,
    }
    impl Case {
        fn new(
            device: &Device,
            stream: &Stream,
            formats: Vec<ExpertFormat>,
            counts: &[u32],
            exact: bool,
            (h, i): (u32, u32),
        ) -> Result<Self> {
            let ranges = ranges(counts);
            let rows = validate(&formats, &ranges, h, i)? as usize;
            let mut seed = 104729;
            // Separate device allocations exceed L2 in the 288-expert bench.
            // Templates reduce host generation cost; no image pointers alias.
            let mut templates: Vec<(ExpertFormat, Vec<u8>)> = Vec::new();
            let mut images = Vec::new();
            for &f in &formats {
                let found = templates
                    .iter()
                    .position(|(tf, _)| tf.gate_up == f.gate_up && tf.down == f.down);
                let index = if let Some(index) = found {
                    index
                } else {
                    let image = if exact {
                        let step = MatrixFormat::NVFP4.stride(i, h)?;
                        let mut image = vec![0; 3 * step];
                        for (part, n, k) in [(0, i, h), (1, i, h), (2, h, i)] {
                            let start = part * step;
                            let payload = n as usize * k as usize / 2;
                            let len = MatrixFormat::NVFP4.storage_bytes(n, k)?;
                            image[start..start + payload].fill(0x22);
                            image[start + payload..start + len - 4].fill(0x38);
                            image[start + len - 4..start + len]
                                .copy_from_slice(&0.007312346f32.to_le_bytes());
                        }
                        image
                    } else {
                        expert(h, i, f, &mut seed).0
                    };
                    templates.push((f, image));
                    templates.len() - 1
                };
                images.push(upload(device, stream, &templates[index].1)?);
            }
            let pointers = upload(
                device,
                stream,
                &images
                    .iter()
                    .flat_map(|im| (im.as_ptr() as usize).to_ne_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let descriptors = upload(
                device,
                stream,
                &formats
                    .iter()
                    .flat_map(|f| [f.gate_up, f.down])
                    .flat_map(|m| [m.format, m.grid, m.group, m.reserved])
                    .collect::<Vec<_>>(),
            )?;
            let device_ranges = upload(
                device,
                stream,
                &ranges
                    .iter()
                    .flatten()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let x = if exact {
                (0..rows * h as usize)
                    .map(|j| bf((j % 17) as f32 / 8. - 1.))
                    .collect()
            } else {
                activation(rows * h as usize, &mut seed)
            };
            Ok(Self {
                h,
                i,
                formats,
                ranges,
                images,
                pointers,
                descriptors,
                device_ranges,
                x: upload(device, stream, &bytes(&x))?,
                product: device.alloc(rows * i as usize * 2)?,
                out: device.alloc(rows * h as usize * 2)?,
            })
        }
        fn packed(&self, stream: &Stream) -> Result<()> {
            check(
                unsafe {
                    ffi::hayate_glm_grouped_packed_gate_up(
                        self.product.as_ptr().cast_mut(),
                        self.x.as_ptr(),
                        self.pointers.as_ptr(),
                        self.descriptors.as_ptr().cast(),
                        self.device_ranges.as_ptr(),
                        self.formats.len() as u32,
                        self.h,
                        self.i,
                        stream.as_raw(),
                    )
                },
                "packed gate/up test",
            )?;
            self.down(&self.product, &self.out, stream)
        }
        fn down(&self, product: &DeviceBuffer, out: &DeviceBuffer, stream: &Stream) -> Result<()> {
            check(
                unsafe {
                    ffi::hayate_glm_grouped_packed_down(
                        out.as_ptr().cast_mut(),
                        product.as_ptr(),
                        self.pointers.as_ptr(),
                        self.descriptors.as_ptr().cast(),
                        self.device_ranges.as_ptr(),
                        self.formats.len() as u32,
                        self.h,
                        self.i,
                        stream.as_raw(),
                    )
                },
                "packed down test",
            )
        }
    }
    // Matches prefill_experts: eight images -> convert -> three unscaled
    // restores -> FP32 grouped -> rescale/BF16 -> point -> grouped down.
    #[derive(Debug)]
    struct Legacy {
        w: [DeviceBuffer; 3],
        scales: [DeviceBuffer; 3],
        converted: DeviceBuffer,
        metadata: DeviceBuffer,
        accumulator: DeviceBuffer,
        gate: DeviceBuffer,
        up: DeviceBuffer,
        product: DeviceBuffer,
        out: DeviceBuffer,
    }
    impl Legacy {
        fn new(device: &Device, c: &Case) -> Result<Self> {
            let matrices = c.formats.len().min(8);
            let matrix_bytes = c.h as usize * c.i as usize * 2 * matrices;
            let rows: usize = c.ranges.iter().map(|r| r[0] as usize).sum();
            Ok(Self {
                w: [
                    device.alloc(matrix_bytes)?,
                    device.alloc(matrix_bytes)?,
                    device.alloc(matrix_bytes)?,
                ],
                scales: [device.alloc(32)?, device.alloc(32)?, device.alloc(32)?],
                converted: device.alloc(3 * MatrixFormat::NVFP4.stride(c.i, c.h)?)?,
                metadata: device.alloc(608)?,
                accumulator: device.alloc(rows * c.h.max(c.i) as usize * 4)?,
                gate: device.alloc(rows * c.i as usize * 2)?,
                up: device.alloc(rows * c.i as usize * 2)?,
                product: device.alloc(rows * c.i as usize * 2)?,
                out: device.alloc(rows * c.h as usize * 2)?,
            })
        }
        fn run(&self, c: &Case, stream: &Stream) -> Result<()> {
            let step = MatrixFormat::NVFP4.stride(c.i, c.h)?;
            let matrix_bytes = c.h as usize * c.i as usize * 2;
            for start in (0..c.formats.len()).step_by(8) {
                let end = (start + 8).min(c.formats.len());
                let group = &c.ranges[start..end];
                let max_rows = group.iter().map(|r| r[0]).max().unwrap();
                if max_rows == 0 {
                    continue;
                }
                for (j, index) in (start..end).enumerate() {
                    let f = c.formats[index];
                    let image = if f.gate_up.format == 0 && f.down.format == 0 {
                        c.images[index].as_ptr()
                    } else {
                        check(
                            unsafe {
                                l::hayate_glm_lowbit_convert_expert(
                                    self.converted.as_ptr().cast_mut(),
                                    c.images[index].as_ptr(),
                                    f,
                                    c.h,
                                    c.i,
                                    stream.as_raw(),
                                )
                            },
                            "legacy convert",
                        )?;
                        self.converted.as_ptr()
                    };
                    for (part, n, k) in [(0, c.i, c.h), (1, c.i, c.h), (2, c.h, c.i)] {
                        check(
                            unsafe {
                                g::hayate_glm_nvfp4_restore_unscaled(
                                    at(&self.w[part], j * matrix_bytes),
                                    at(&self.scales[part], j * 4),
                                    image.cast::<u8>().add(part * step).cast(),
                                    n,
                                    k,
                                    stream.as_raw(),
                                )
                            },
                            "legacy restore",
                        )?;
                    }
                }
                for (part, out, input, n, k) in [
                    (0, &self.gate, &c.x, c.i, c.h),
                    (1, &self.up, &c.x, c.i, c.h),
                    (2, &self.out, &self.product, c.h, c.i),
                ] {
                    if part == 2 {
                        let offset = group[0][1] as usize * c.i as usize * 2;
                        let count: usize = group.iter().map(|r| r[0] as usize).sum();
                        check(
                            unsafe {
                                g::hayate_glm_point(
                                    at(&self.product, offset),
                                    at(&self.gate, offset),
                                    at(&self.up, offset),
                                    count * c.i as usize,
                                    1,
                                    stream.as_raw(),
                                )
                            },
                            "legacy point",
                        )?;
                    }
                    check(
                        unsafe {
                            g::hayate_glm_grouped_nvfp4(
                                out.as_ptr().cast_mut(),
                                self.accumulator.as_ptr().cast_mut(),
                                input.as_ptr(),
                                self.w[part].as_ptr(),
                                at(&c.device_ranges, start * 8),
                                (end - start) as u32,
                                n,
                                k,
                                self.metadata.as_ptr().cast_mut(),
                                self.scales[part].as_ptr(),
                                max_rows,
                                stream.as_raw(),
                            )
                        },
                        "legacy grouped",
                    )?;
                }
            }
            Ok(())
        }
    }

    const CONFIGS: [&str; 4] = [
        "m64n128k128_w8_s2",
        "m32n128k128_w8_s3",
        "m32n64k128_w4_s2",
        "m32n64k256_w4_s2",
    ];
    fn select_config(config: usize) -> Result<()> {
        check(
            unsafe { ffi::hayate_glm_grouped_packed_set_config(config as u32) },
            "packed configuration",
        )
    }

    #[test]
    #[ignore = "requires sm_120 GPU; all configs bit exact versus legacy convert/restore/grouped"]
    fn glm53_packed_grouped_gpu_compare() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        eprintln!("{}", load());
        assert_ne!(unsafe { ffi::hayate_glm_grouped_packed_set_config(4) }, 0);
        let nv = MatrixFormat::NVFP4;
        // Small K covers a one-stage pipeline and draining fewer K tiles than
        // pipeline stages; model dimensions wrap/reuse the ring many times.
        // The legacy CUTLASS path rejects the small (256, 512) shape (CUDA error 1);
        // it is covered by the CPU reference test.
        for dims in [(4096, 2048)] {
            for exact in [true, false] {
                let fs = formats();
                let fs = if exact {
                    vec![
                        ExpertFormat {
                            gate_up: nv,
                            down: nv
                        };
                        3
                    ]
                } else {
                    fs.iter()
                        .enumerate()
                        .map(|(j, &gu)| ExpertFormat {
                            gate_up: gu,
                            down: fs[(j + 4) % fs.len()],
                        })
                        .collect()
                };
                let counts = if exact {
                    vec![1, 64, 65]
                } else {
                    // Force persistent blocks to reuse the pipeline across jobs.
                    vec![0, 1, 15, 16, 17, 31, 32, 33, 64, 65, 1500]
                };
                let c = Case::new(&device, &stream, fs, &counts, exact, dims)?;
                let old = Legacy::new(&device, &c)?;
                old.run(&c, &stream)?;
                stream.synchronize()?;
                let old_product = download(&stream, &old.product)?;
                let old_out = download(&stream, &old.out)?;
                for (config, name) in CONFIGS.iter().enumerate() {
                    select_config(config)?;
                    eprintln!("gpu_compare config={config}:{name} dims={dims:?} uniform={exact}");
                    c.packed(&stream)?;
                    stream.synchronize()?;
                    compare("gate_up", &download(&stream, &c.product)?, &old_product);
                    compare("end_to_end", &download(&stream, &c.out)?, &old_out);
                    if dims == (256, 512) {
                        let graph = unsafe {
                            crate::graph::Graph::capture(stream.as_raw(), || c.packed(&stream))
                        }?;
                        // Changing the host selection must not alter a captured graph.
                        select_config((config + 1) % CONFIGS.len())?;
                        for _ in 0..2 {
                            unsafe { graph.launch(stream.as_raw()) }?;
                            stream.synchronize()?;
                            compare(
                                "graph_gate_up",
                                &download(&stream, &c.product)?,
                                &old_product,
                            );
                            compare("graph_end_to_end", &download(&stream, &c.out)?, &old_out);
                        }
                        select_config(config)?;
                    }
                    c.down(&old.product, &c.out, &stream)?;
                    stream.synchronize()?;
                    compare("down_same_product", &download(&stream, &c.out)?, &old_out);
                    let mut checked_product = device.alloc(c.product.bytes())?;
                    let mut checked_out = device.alloc(c.out.bytes())?;
                    packed_grouped::gate_up(
                        &device,
                        &mut checked_product,
                        &c.x,
                        &c.images,
                        &c.formats,
                        &c.ranges,
                        c.h,
                        c.i,
                        &stream,
                    )?;
                    packed_grouped::down(
                        &device,
                        &mut checked_out,
                        &checked_product,
                        &c.images,
                        &c.formats,
                        &c.ranges,
                        c.h,
                        c.i,
                        &stream,
                    )?;
                    compare(
                        "checked_vs_raw",
                        &download(&stream, &checked_product)?,
                        &old_product,
                    );
                    compare(
                        "checked_down_vs_raw",
                        &download(&stream, &checked_out)?,
                        &old_out,
                    );
                }
            }
        }
        select_config(0)?;
        Ok(())
    }

    fn route_counts(tokens: u32, hot: bool) -> Vec<u32> {
        let mut counts = vec![0; 288];
        let mut seed = 65537;
        let mut ends = Vec::new();
        let mut sum = 0;
        for j in 0..288 {
            sum += 288000 / (j + 1);
            ends.push(sum);
        }
        for token in 0..tokens {
            let mut selected = Vec::with_capacity(8);
            if hot && token < tokens.min(1500) {
                selected.push(0);
            }
            while selected.len() < 8 {
                let target = random(&mut seed) % sum;
                let expert = ends.partition_point(|&end| end <= target);
                if !selected.contains(&expert) && !(hot && expert == 0) {
                    selected.push(expert);
                }
            }
            for expert in selected {
                counts[expert] += 1;
            }
        }
        counts
    }
    fn median_ms(stream: &Stream, mut run: impl FnMut() -> Result<()>) -> Result<f32> {
        let begin = stream.event()?;
        let end = stream.event()?;
        let mut times = Vec::new();
        for repeat in 0..7 {
            begin.record(stream)?;
            run()?;
            end.record(stream)?;
            end.synchronize()?;
            if repeat >= 2 {
                times.push(begin.elapsed_ms(&end)?);
            }
        }
        times.sort_by(f32::total_cmp);
        Ok(times[2])
    }
    #[test]
    #[ignore = "requires sm_120 GPU; 288 real-sized images (~3.3 GB), T=512/2190/8192, legacy and packed median CUDA events"]
    fn glm53_packed_grouped_bench() -> Result<()> {
        let device = Device::select()?;
        let stream = device.stream()?;
        unsafe extern "C" {
            fn cudaDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> i32;
        }
        let (mut sms, mut clock_khz) = (0, 0);
        check(
            unsafe { cudaDeviceGetAttribute(&mut sms, 16, 0) },
            "SM count",
        )?;
        check(
            unsafe { cudaDeviceGetAttribute(&mut clock_khz, 13, 0) },
            "max SM clock",
        )?;
        // Dense BF16 mma.sync: 1024 FLOP/SM/clock; exclude 2:4 sparsity.
        let peak = f64::from(sms) * f64::from(clock_khz) * 1024. / 1e9;
        eprintln!(
            "gpu={:?} sms={sms} max_clock_khz={clock_khz} dense_bf16_peak_tflops={peak:.3}",
            device.name
        );
        let low = MatrixFormat::new(3, 32)?;
        let nv = MatrixFormat::NVFP4;
        // b150.json global table: (nf3,nf3)=6523, (nf3,nv)=2051,
        // (nv,nf3)=9, (nv,nv)=3513 / 12096. 288-expert approximation:
        let fs: Vec<_> = (0..288)
            .map(|j| {
                if j < 155 {
                    ExpertFormat {
                        gate_up: low,
                        down: low,
                    }
                } else if j < 204 {
                    ExpertFormat {
                        gate_up: low,
                        down: nv,
                    }
                } else {
                    ExpertFormat {
                        gate_up: nv,
                        down: nv,
                    }
                }
            })
            .collect();
        // Interleave formats so the hottest experts don't all use LOWBIT.
        let fs: Vec<_> = (0..288).map(|j| fs[j * 53 % 288]).collect();
        for (tokens, hot) in [(512, false), (2190, false), (8192, false), (2190, true)] {
            let counts = route_counts(tokens, hot);
            let c = Case::new(&device, &stream, fs.clone(), &counts, false, (4096, 2048))?;
            let old = Legacy::new(&device, &c)?;
            let legacy_load_before = load();
            let legacy_ms = median_ms(&stream, || old.run(&c, &stream))?;
            eprintln!(
                "legacy_load_before={legacy_load_before} legacy_load_after={}",
                load()
            );
            let old_product = download(&stream, &old.product)?;
            let old_out = download(&stream, &old.out)?;
            for (config, name) in CONFIGS.iter().enumerate() {
                select_config(config)?;
                let before = load();
                let packed_ms = median_ms(&stream, || c.packed(&stream))?;
                let rows = 8 * tokens;
                let tflops = 6. * f64::from(rows) * f64::from(c.h) * f64::from(c.i)
                    / f64::from(packed_ms)
                    / 1e9;
                let target = match tokens {
                    512 => 4.,
                    8192 => 25.,
                    _ => 8.,
                };
                eprintln!(
                    "packed_bench config={config}:{name} tokens={tokens} rows={rows} experts=288 mix=155/49/84 hot={hot} max_rows={} legacy_ms={legacy_ms:.6} packed_gate_up_down_ms={packed_ms:.6} speedup={:.3} effective_tflops={tflops:.3} dense_peak_percent={:.3} target_ms={target} target_met={} warmups=2 repeats=5\nload_before={before}\nload_after={}",
                    counts.iter().max().unwrap(),
                    legacy_ms / packed_ms,
                    100. * tflops / peak,
                    packed_ms <= target,
                    load()
                );
                compare(
                    "bench_gate_up",
                    &download(&stream, &c.product)?,
                    &old_product,
                );
                compare("bench_end_to_end", &download(&stream, &c.out)?, &old_out);
            }
        }
        select_config(0)?;
        Ok(())
    }
}
