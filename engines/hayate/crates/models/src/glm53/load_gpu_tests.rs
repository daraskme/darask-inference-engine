//! Run under gpu_slot; no model execution or server restart is needed.
use super::*;
use hayate_core::{
    MachineId, Shape,
    hyt::{FORMAT_VERSION, Header, encode_table},
};
use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn fixture() -> Result<(Fixture, HytFile)> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("hayate-upload-{}-{id}.hyt", std::process::id()));
    let mut file = File::options().write(true).create_new(true).open(&path)?;
    let guard = Fixture(path);
    let mut entries = Vec::new();
    let mut payloads = Vec::new();
    let mut offset = 0u64;
    for i in 0..11 {
        let (quant, shape, raw) = match i % 4 {
            0 => (
                QuantSpec::Plain(DType::Bf16),
                vec![1, (STAGING_BYTES + 12) as u64],
                [0x80, 0x3f].repeat(STAGING_BYTES + 12),
            ),
            1 => (
                QuantSpec::PerTensor(DType::F8E4M3),
                vec![1, (STAGING_BYTES + 3) as u64],
                [vec![0xb8; STAGING_BYTES + 3], 0.3f32.to_le_bytes().to_vec()].concat(),
            ),
            2 => {
                let mut raw = vec![0xfe; QuantSpec::Nvfp4.matrix_bytes(3, 256).unwrap() as usize];
                let end = raw.len();
                raw[384..end - 4].fill(0x38);
                raw[end - 4..].copy_from_slice(&0.125f32.to_le_bytes());
                (QuantSpec::Nvfp4, vec![3, 256], raw)
            }
            _ => (
                QuantSpec::Plain(DType::F32),
                vec![7],
                (-0.25f32).to_le_bytes().repeat(7),
            ),
        };
        entries.push(TensorEntry {
            name: format!("weight{i}"),
            quant,
            shape: Shape::new(&shape)?,
            offset,
            bytes: raw.len() as u64,
        });
        offset += (raw.len() as u64).next_multiple_of(4096);
        payloads.push(raw);
    }
    let table = encode_table(&entries)?;
    let data_offset = (64 + table.len() as u64).next_multiple_of(4096);
    let header = Header {
        format_version: FORMAT_VERSION,
        machine: 0,
        tensor_count: entries.len() as u32,
        table_offset: 64,
        data_offset,
        source_hash: [0; 16],
    };
    file.write_all(&header.encode())?;
    file.write_all(&table)?;
    for (entry, raw) in entries.iter().zip(payloads) {
        file.seek(SeekFrom::Start(data_offset + entry.offset))?;
        file.write_all(&raw)?;
    }
    let hyt = HytFile::open(&guard.0, MachineId::Main)?;
    Ok((guard, hyt))
}

#[test]
#[ignore = "requires CUDA and gpu_slot; uses less than 1 GiB"]
fn glm53_streamed_upload_matches_file_and_drains_errors() -> Result<()> {
    let device = Device::select()?;
    let stream = device.stream()?;
    let (_guard, file) = fixture()?;
    assert!(upload(&device, &file, &[])?.is_empty());
    for indices in [vec![0], (0..file.entries().len()).collect()] {
        let buffers = upload(&device, &file, &indices)?;
        for i in indices {
            let entry = &file.entries()[i];
            let mut expected = vec![0; entry.bytes as usize];
            let mut actual = expected.clone();
            file.read_tensor(i, &mut expected)?;
            stream.copy_d2h(&mut actual, &buffers[&entry.name])?;
            assert_eq!(actual, expected, "{}", entry.name);
        }
    }
    // Fail on a later tensor after DMA was submitted; its destinations outlive
    // the guard. A second transfer must still work after cleanup.
    let mut good = device.alloc(file.entries()[0].bytes as usize)?;
    let mut short = device.alloc(1)?;
    {
        let mut staging = Staging::new(&device, 4096)?;
        staging.upload(&file, 0, &mut good)?;
        assert!(staging.upload(&file, 1, &mut short).is_err());
    }
    let mut actual = vec![0; good.bytes()];
    stream.copy_d2h(&mut actual, &good)?;
    let mut expected = actual.clone();
    file.read_tensor(0, &mut expected)?;
    assert_eq!(actual, expected);

    // Reject a nonfinite scale after earlier payload chunks have been queued.
    // Only the disposable fixture is mutated; the reader retains its open fd.
    let mut writer = File::options().write(true).open(&_guard.0)?;
    let bad = &file.entries()[1];
    writer.seek(SeekFrom::Start(
        file.header().data_offset + bad.offset + bad.bytes - 4,
    ))?;
    writer.write_all(&f32::NAN.to_le_bytes())?;
    assert!(upload(&device, &file, &[0, 1]).is_err());

    // A file truncated after open must also fail and drain the earlier writes.
    let last = file.entries().last().unwrap();
    writer.set_len(file.header().data_offset + last.offset + 1)?;
    assert!(upload(&device, &file, &[0, file.entries().len() - 1]).is_err());
    assert_eq!(upload(&device, &file, &[3])?.len(), 1);
    Ok(())
}

// Exact old resident-loader strategy: four workers, one full-tensor pinned
// buffer each, validation followed by a synchronous copy for every tensor.
fn serial_upload(device: &Device, file: &HytFile, indices: &[usize]) -> Result<Vec<DeviceBuffer>> {
    let mut buffers = indices
        .iter()
        .map(|&i| device.alloc(file.entries()[i].bytes as usize))
        .collect::<Result<Vec<_>>>()?;
    let chunk = indices.len().div_ceil(4).max(1);
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
                    let stream = device.stream()?;
                    let mut pinned = device
                        .alloc_pinned(buffers.iter().map(DeviceBuffer::bytes).max().unwrap())?;
                    for (buffer, &i) in buffers.iter_mut().zip(indices) {
                        let entry = &file.entries()[i];
                        let raw = &mut pinned.as_mut_slice()[..entry.bytes as usize];
                        file.read_tensor(i, raw)?;
                        if entry.quant == QuantSpec::Nvfp4 {
                            let dims = entry.shape.dims();
                            super::super::validate_nvfp4(raw, dims[0] as usize, dims[1] as usize)?;
                        } else {
                            super::super::validate_raw(raw, entry.quant)?;
                        }
                        stream.copy_h2d(buffer, raw)?;
                    }
                    Ok(())
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap()?;
        }
        Ok(())
    })?;
    Ok(buffers)
}

#[test]
#[ignore = "CUDA upload benchmark; gpu_slot need depends on HAYATE_UPLOAD_BENCH_HYT"]
fn glm53_streamed_upload_benchmark() -> Result<()> {
    let device = Device::select()?;
    let (guard, fixture_file) = fixture()?;
    let file = match std::env::var_os("HAYATE_UPLOAD_BENCH_HYT") {
        Some(path) => HytFile::open(path, MachineId::Main)?,
        None => fixture_file,
    };
    // Match Layer::load_inner: routed experts use their own cache/uploader.
    let indices: Vec<_> = file
        .entries()
        .iter()
        .enumerate()
        .filter(|(_, entry)| !entry.name.starts_with("mlp.experts."))
        .map(|(i, _)| i)
        .collect();
    let bytes: u64 = indices.iter().map(|&i| file.entries()[i].bytes).sum();
    // Verify every byte of the selected real file as well as the synthetic
    // boundary fixture, outside the timed region and with bounded host memory.
    let stream = device.stream()?;
    let verified = upload(&device, &file, &indices)?;
    let mut actual = vec![0; STAGING_BYTES];
    let mut expected = vec![0; STAGING_BYTES];
    for &i in &indices {
        let entry = &file.entries()[i];
        for offset in (0..entry.bytes as usize).step_by(STAGING_BYTES) {
            let n = STAGING_BYTES.min(entry.bytes as usize - offset);
            stream.copy_d2h_at(&mut actual[..n], &verified[&entry.name], offset)?;
            file.read_tensor_range(i, offset as u64, &mut expected[..n])?;
            assert_eq!(actual[..n], expected[..n], "{} offset {offset}", entry.name);
        }
    }
    drop(verified);
    eprintln!("upload verified bytes={bytes} tensors={}", indices.len());
    let mut old = Vec::new();
    let mut new = Vec::new();
    for round in 0..6 {
        // Alternate order, discard warmup, include allocation, reads, validation,
        // synchronization and pinned-buffer cleanup; exclude device-buffer free.
        for pipeline in if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let start = Instant::now();
            let elapsed = if pipeline {
                let out = upload(&device, &file, &indices)?;
                let elapsed = start.elapsed().as_secs_f64();
                drop(out);
                elapsed
            } else {
                let out = serial_upload(&device, &file, &indices)?;
                let elapsed = start.elapsed().as_secs_f64();
                drop(out);
                elapsed
            };
            if round > 0 {
                if pipeline {
                    new.push(elapsed);
                } else {
                    old.push(elapsed);
                }
            }
        }
    }
    old.sort_by(f64::total_cmp);
    new.sort_by(f64::total_cmp);
    let chunk = indices.len().div_ceil(4).max(1);
    let capacities: Vec<_> = indices
        .chunks(chunk)
        .map(|is| is.iter().map(|&i| file.entries()[i].bytes).max().unwrap())
        .collect();
    eprintln!(
        "upload benchmark bytes={bytes} old_s={old:?} new_s={new:?} speedup={:.3} old_pinned_bytes={} new_pinned_bytes={}",
        old[2] / new[2],
        capacities.iter().sum::<u64>(),
        capacities
            .iter()
            .map(|&n| 2 * n.min(STAGING_BYTES as u64).next_multiple_of(4))
            .sum::<u64>()
    );
    drop(guard);
    Ok(())
}
