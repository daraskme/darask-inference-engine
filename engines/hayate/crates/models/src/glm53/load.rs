#[cfg(feature = "cuda")]
use hayate_core::hyt::HytFile;
use hayate_core::{DType, Error, QuantSpec, Result, hyt::TensorEntry};
#[cfg(feature = "cuda")]
use hayate_kernels::runtime::{Device, DeviceBuffer, Event, PinnedHost, Stream};
#[cfg(feature = "cuda")]
use std::{collections::BTreeMap, time::Instant};

// Two bounded buffers per worker: at most 64 MiB pinned across all four workers,
// independent of embedding/head size. Read/validate the next chunk during H2D.
#[cfg(feature = "cuda")]
const STAGING_BYTES: usize = 8 << 20;

#[derive(Clone, Copy)]
enum Check {
    Raw(QuantSpec),
    Packed,
    PositiveScale,
}

impl Check {
    fn validate(self, raw: &[u8]) -> Result<()> {
        match self {
            Self::Raw(quant) => super::validate_raw(raw, quant),
            Self::Packed => Ok(()), // Every E2M1 nibble encodes a finite value.
            Self::PositiveScale => {
                let scale = raw.try_into().ok().map(f32::from_le_bytes);
                if scale.is_some_and(|s| s.is_finite() && s > 0.) {
                    Ok(())
                } else {
                    Err(Error::Weights("GLM invalid tensor scale".into()))
                }
            }
        }
    }
}

/// Separate the payload and scale regions so a chunk never splits an FP32
/// tensor scale or treats packed nibbles as FP8. The original validation rules
/// still apply to every byte, including padded NVFP4 block scales.
fn regions(entry: &TensorEntry) -> Result<Vec<(usize, usize, Check)>> {
    let bytes =
        usize::try_from(entry.bytes).map_err(|_| Error::Weights("GLM tensor too large".into()))?;
    let fp8 = Check::Raw(QuantSpec::Plain(DType::F8E4M3));
    match entry.quant {
        QuantSpec::Plain(DType::F32 | DType::Bf16 | DType::F8E4M3) => {
            Ok(vec![(0, bytes, Check::Raw(entry.quant))])
        }
        QuantSpec::PerTensor(DType::F8E4M3) if bytes > 4 => Ok(vec![
            (0, bytes - 4, fp8),
            (bytes - 4, 4, Check::PositiveScale),
        ]),
        QuantSpec::Nvfp4 => {
            let dims = entry.shape.dims();
            if dims.len() != 2 || entry.quant.matrix_bytes(dims[0], dims[1]) != Some(entry.bytes) {
                return Err(Error::Weights("GLM NVFP4 shape/size".into()));
            }
            let payload = (entry.shape.numel() / 2) as usize;
            Ok(vec![
                (0, payload, Check::Packed),
                (payload, bytes - payload - 4, fp8),
                (bytes - 4, 4, Check::PositiveScale),
            ])
        }
        _ => Err(Error::Weights(
            "GLM unsupported resident weight format".into(),
        )),
    }
}

#[cfg(feature = "cuda")]
struct Staging {
    stream: Stream,
    buffers: [PinnedHost; 2],
    ready: [Event; 2],
    busy: [bool; 2],
    next: usize,
    pending: bool,
}

#[cfg(feature = "cuda")]
impl Staging {
    fn new(device: &Device, bytes: usize) -> Result<Self> {
        let stream = device.stream()?;
        let buffers = [device.alloc_pinned(bytes)?, device.alloc_pinned(bytes)?];
        let ready = [stream.event()?, stream.event()?];
        Ok(Self {
            stream,
            buffers,
            ready,
            busy: [false; 2],
            next: 0,
            pending: false,
        })
    }

    fn upload(&mut self, file: &HytFile, index: usize, dst: &mut DeviceBuffer) -> Result<()> {
        for (start, bytes, check) in regions(&file.entries()[index])? {
            let capacity = self.buffers[0].as_slice().len();
            for offset in (start..start + bytes).step_by(capacity) {
                let count = capacity.min(start + bytes - offset);
                let slot = self.next;
                if self.busy[slot] {
                    self.ready[slot].synchronize()?;
                }
                let raw = &mut self.buffers[slot].as_mut_slice()[..count];
                file.read_tensor_range(index, offset as u64, raw)?;
                check.validate(raw)?;
                // Set before submission: failures and panics must also drain DMA
                // before either staging or the caller's destination can be freed.
                self.pending = true;
                unsafe {
                    self.stream.copy_h2d_at_async(
                        dst,
                        offset,
                        self.buffers[slot].range(0..count)?,
                    )?;
                }
                self.ready[slot].record(&self.stream)?;
                self.busy[slot] = true;
                self.next ^= 1;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.pending {
            self.stream.synchronize()?;
            self.pending = false;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for Staging {
    fn drop(&mut self) {
        // Runs before fields are dropped, including after validation/I/O errors.
        // The destination buffers belong to upload's enclosing thread scope.
        let _ = self.finish();
    }
}

#[cfg(feature = "cuda")]
pub(super) fn upload(
    device: &Device,
    file: &HytFile,
    indices: &[usize],
) -> Result<BTreeMap<String, DeviceBuffer>> {
    let start = Instant::now();
    let mut buffers = indices
        .iter()
        .map(|&i| device.alloc(file.entries()[i].bytes as usize))
        .collect::<Result<Vec<_>>>()?;
    let chunk = buffers.len().div_ceil(4).max(1);
    std::thread::scope(|scope| -> Result<()> {
        let workers: Vec<_> = buffers
            .chunks_mut(chunk)
            .zip(indices.chunks(chunk))
            .enumerate()
            .map(|(worker, (buffers, indices))| {
                scope.spawn(move || -> Result<()> {
                    hayate_kernels::cpu::pin_thread(
                        hayate_core::machine::MAIN.cpu.service[worker],
                    )?;
                    let capacity = buffers
                        .iter()
                        .map(DeviceBuffer::bytes)
                        .max()
                        .unwrap()
                        .min(STAGING_BYTES)
                        .next_multiple_of(4);
                    let mut staging = Staging::new(device, capacity)?;
                    for (buffer, &i) in buffers.iter_mut().zip(indices) {
                        staging.upload(file, i, buffer)?;
                    }
                    staging.finish()
                })
            })
            .collect();
        for worker in workers {
            worker
                .join()
                .map_err(|_| Error::Weights("GLM weight load worker panicked".into()))??;
        }
        Ok(())
    })?;
    eprintln!(
        "GLM resident upload: bytes={} seconds={:.3}",
        buffers.iter().map(DeviceBuffer::bytes).sum::<usize>(),
        start.elapsed().as_secs_f64()
    );
    Ok(indices
        .iter()
        .zip(buffers)
        .map(|(&i, b)| (file.entries()[i].name.clone(), b))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hayate_core::Shape;

    fn chunked(raw: &[u8], quant: QuantSpec, shape: &[u64], chunk: usize) -> bool {
        let entry = TensorEntry {
            name: "test".into(),
            quant,
            shape: Shape::new(shape).unwrap(),
            offset: 0,
            bytes: raw.len() as u64,
        };
        let Ok(regions) = regions(&entry) else {
            return false;
        };
        regions.into_iter().all(|(start, bytes, check)| {
            raw[start..start + bytes]
                .chunks(chunk)
                .all(|part| check.validate(part).is_ok())
        })
    }

    #[test]
    fn streamed_validation_matches_whole_tensor_at_boundaries() {
        for (quant, raw) in [
            (
                QuantSpec::Plain(DType::F32),
                [1f32.to_le_bytes(), (-3f32).to_le_bytes()].concat(),
            ),
            (
                QuantSpec::Plain(DType::Bf16),
                vec![0x80, 0x3f, 0, 0, 0, 0x80],
            ),
            (
                QuantSpec::Plain(DType::F8E4M3),
                vec![0, 0x38, 0xb8, 0x7e, 0xfe],
            ),
            (
                QuantSpec::PerTensor(DType::F8E4M3),
                [vec![0x38; 7], 0.75f32.to_le_bytes().to_vec()].concat(),
            ),
        ] {
            for chunk in [4, 8, 16] {
                assert!(chunked(&raw, quant, &[1], chunk));
                for at in 0..raw.len() {
                    for bad in [0x7f, 0xff, 0x80, 0] {
                        let mut changed = raw.clone();
                        changed[at] = bad;
                        assert_eq!(
                            chunked(&changed, quant, &[1], chunk),
                            super::super::validate_raw(&changed, quant).is_ok()
                        );
                    }
                }
            }
        }
        assert!(!chunked(&[0; 3], QuantSpec::Plain(DType::F32), &[1], 4));
        assert!(!chunked(
            &[0; 4],
            QuantSpec::PerTensor(DType::F8E4M3),
            &[1],
            4
        ));
        assert!(!chunked(&[0; 2], QuantSpec::Plain(DType::F16), &[1], 4));
    }

    #[test]
    fn streamed_nvfp4_checks_scales_including_padding_but_not_payload() {
        let (n, k) = (3, 32);
        let mut raw = vec![0xff; QuantSpec::Nvfp4.matrix_bytes(n, k).unwrap() as usize];
        let len = raw.len();
        raw[(n * k / 2) as usize..len - 4].fill(0x38);
        raw[len - 4..].copy_from_slice(&0.5f32.to_le_bytes());
        for chunk in [4, 32, 64, 512] {
            assert!(chunked(&raw, QuantSpec::Nvfp4, &[n, k], chunk));
            for at in [0, 47, 48, 63, 511, len - 5, len - 1] {
                let mut changed = raw.clone();
                changed[at] = 0xff;
                assert_eq!(
                    chunked(&changed, QuantSpec::Nvfp4, &[n, k], chunk),
                    super::super::validate_nvfp4(&changed, n as usize, k as usize).is_ok()
                );
            }
        }
        assert!(!chunked(&raw[..len - 1], QuantSpec::Nvfp4, &[n, k], 32));
        assert!(!chunked(&raw, QuantSpec::Nvfp4, &[1, n, k], 32));
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "load_gpu_tests.rs"]
mod gpu_tests;
