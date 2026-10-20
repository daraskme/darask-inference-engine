//! Host bandwidth measurements for ARCHITECTURE.md §4.3 and §9.3.

use hayate_core::io::RowTable;
use hayate_core::{Error, Machine, QuantSpec, Result};
use hayate_kernels::cpu::{ComputePool, pin_thread};
use std::fs::{File, OpenOptions};
use std::hint::black_box;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::sync::mpsc;
use std::time::Instant;

const WARMUP: usize = 2;
const SAMPLES: usize = 9;
const RAM_BYTES: usize = 512 << 20;
const RAM_PASSES: usize = 4;
const EXPERTS: usize = 64;
const ROW_BATCHES: usize = 1024;
const ROWS_PER_BATCH: usize = 4;
const ROW_SEED: u64 = 0x4e6772616d;

#[derive(Debug, PartialEq, Eq)]
struct Options {
    json: bool,
    nvme: Option<String>,
    nvme_threads: usize,
    rows: Option<String>,
    row_bytes: Option<usize>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self> {
        let mut options = Self {
            json: false,
            nvme: None,
            nvme_threads: 8,
            rows: None,
            row_bytes: None,
        };
        let mut args = args.iter();
        let mut nvme_threads_set = false;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--json" if !options.json => options.json = true,
                "--nvme" if options.nvme.is_none() => {
                    options.nvme = Some(
                        args.next()
                            .filter(|s| !s.starts_with("--"))
                            .ok_or_else(|| Error::InvalidPlan("--nvme requires a file".into()))?
                            .clone(),
                    );
                }
                "--rows" if options.rows.is_none() => {
                    options.rows = Some(
                        args.next()
                            .filter(|s| !s.starts_with("--"))
                            .ok_or_else(|| Error::InvalidPlan("--rows requires a file".into()))?
                            .clone(),
                    );
                }
                "--nvme-threads" if !nvme_threads_set => {
                    options.nvme_threads = args
                        .next()
                        .and_then(|s| s.parse::<usize>().ok())
                        .filter(|&n| n > 0)
                        .ok_or_else(|| {
                            Error::InvalidPlan("--nvme-threads requires a positive integer".into())
                        })?;
                    nvme_threads_set = true;
                }
                "--row-bytes" if options.row_bytes.is_none() => {
                    options.row_bytes = Some(
                        args.next()
                            .and_then(|s| s.parse::<usize>().ok())
                            .filter(|&n| n > 0)
                            .ok_or_else(|| {
                                Error::InvalidPlan("--row-bytes requires a positive integer".into())
                            })?,
                    );
                }
                _ => {
                    return Err(Error::InvalidPlan(format!(
                        "unknown or repeated bench host option: {arg}"
                    )));
                }
            }
        }
        if options.rows.is_some() != options.row_bytes.is_some() {
            return Err(Error::InvalidPlan(
                "--rows and --row-bytes must be specified together".into(),
            ));
        }
        if nvme_threads_set && options.nvme.is_none() {
            return Err(Error::InvalidPlan("--nvme-threads requires --nvme".into()));
        }
        Ok(options)
    }
}

#[derive(Debug)]
struct Measurement {
    bytes: u64,
    seconds: f64,
}

impl Measurement {
    fn gb_per_s(&self) -> f64 {
        self.bytes as f64 / self.seconds / 1e9
    }
}

fn measure(bytes: u64, mut operation: impl FnMut() -> Result<()>) -> Result<Measurement> {
    for _ in 0..WARMUP {
        operation()?;
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        operation()?;
        samples.push(start.elapsed().as_secs_f64());
    }
    samples.sort_unstable_by(f64::total_cmp);
    Ok(Measurement {
        bytes,
        seconds: samples[SAMPLES / 2],
    })
}

fn machine() -> &'static Machine {
    &hayate_core::machine::MAIN
}

fn isa() -> &'static str {
    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    {
        "avx512"
    }
    #[cfg(all(
        target_feature = "avx2",
        target_feature = "fma",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    {
        "avx2-fma"
    }
    #[cfg(not(any(
        all(target_feature = "avx512f", target_feature = "avx512bw"),
        all(target_feature = "avx2", target_feature = "fma")
    )))]
    {
        "scalar-reference"
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let options = Options::parse(args)?;
    let machine = machine();
    let (name, cpu_name) = match machine.id {
        hayate_core::MachineId::Main => ("main", "i9-12900KS"),
        hayate_core::MachineId::Sub => ("sub", "Ryzen 7 8700F"),
    };
    if !std::fs::read_to_string("/proc/cpuinfo")?.contains(cpu_name) {
        return Err(Error::UnsupportedMachine(format!(
            "bench host requires {cpu_name}"
        )));
    }
    // The coordinator and optional disk reads leave compute cores to workers.
    pin_thread(machine.cpu.service[0])?;
    let mut pool = ComputePool::new(machine)?;
    let ram = {
        let words = vec![0x123456789abcdef0u64; RAM_BYTES / 8];
        measure((RAM_BYTES * RAM_PASSES) as u64, || {
            for _ in 0..RAM_PASSES {
                black_box(pool.read_ram(black_box(&words))?);
            }
            Ok(())
        })?
    };
    let up = gemv(&mut pool, 4096, 2048)?;
    let down = gemv(&mut pool, 2048, 4096)?;
    let nvme = options
        .nvme
        .as_deref()
        .map(|path| nvme(path, options.nvme_threads, machine))
        .transpose()?;
    let rows = options
        .rows
        .as_deref()
        .map(|path| {
            pin_thread(machine.cpu.compute[0])?;
            let result = rows(path, options.row_bytes.unwrap(), machine);
            pin_thread(machine.cpu.service[0])?;
            result
        })
        .transpose()?;
    if options.json {
        let rows_json = rows
            .as_ref()
            .map_or_else(|| "null".into(), RowMeasurements::json);
        let nvme_json = match &nvme {
            Some(modes) => {
                let measurements: Vec<String> = modes
                    .iter()
                    .map(|(threads, m)| {
                        format!(
                            "{{\"threads\":{threads},\"gb_per_s\":{:.6},\"seconds\":{:.6}}}",
                            m.gb_per_s(),
                            m.seconds
                        )
                    })
                    .collect();
                let sequential = &modes[0].1;
                format!(
                    "{{\"mode\":\"O_DIRECT\",\"chunk_bytes\":4194304,\"bytes_per_sample\":{},\"gb_per_s\":{:.6},\"seconds\":{:.6},\"concurrency\":[{}]}}",
                    sequential.bytes,
                    sequential.gb_per_s(),
                    sequential.seconds,
                    measurements.join(",")
                )
            }
            None => "null".into(),
        };
        println!(
            "{{\"machine\":\"{name}\",\"isa\":\"{}\",\"compute_cores\":{:?},\"warmup\":{WARMUP},\"samples\":{SAMPLES},\"statistic\":\"median\",\"ram\":{{\"buffer_bytes\":{RAM_BYTES},\"passes\":{RAM_PASSES},\"gb_per_s\":{:.6}}},\"gemv\":{{\"group\":32,\"experts\":{EXPERTS},\"weight_bytes_per_expert\":{},\"4096x2048\":{{\"gb_per_s\":{:.6},\"ms_per_expert\":{:.6}}},\"2048x4096\":{{\"gb_per_s\":{:.6},\"ms_per_expert\":{:.6}}}}},\"nvme\":{nvme_json},\"rows\":{rows_json}}}",
            isa(),
            machine.cpu.compute,
            ram.gb_per_s(),
            up.bytes / EXPERTS as u64,
            up.gb_per_s(),
            up.seconds * 1000.0 / EXPERTS as f64,
            down.gb_per_s(),
            down.seconds * 1000.0 / EXPERTS as f64
        );
    } else {
        println!(
            "host: {name} / {} / compute {:?}",
            isa(),
            machine.cpu.compute
        );
        println!("warmup={WARMUP}, samples={SAMPLES}, median; GB=10^9 bytes");
        println!(
            "RAM: {RAM_BYTES} bytes, {RAM_PASSES} passes/sample; GEMV: {EXPERTS} distinct experts/shape, group=32"
        );
        println!("{:<28} {:>12} {:>14}", "measurement", "GB/s", "ms/expert");
        println!(
            "{:<28} {:>12.3} {:>14}",
            "RAM sequential read",
            ram.gb_per_s(),
            "-"
        );
        for (shape, m) in [("Q4Bf16 4096x2048", &up), ("Q4Bf16 2048x4096", &down)] {
            println!(
                "{shape:<28} {:>12.3} {:>14.3}",
                m.gb_per_s(),
                m.seconds * 1000.0 / EXPERTS as f64
            );
        }
        if let Some(modes) = nvme {
            println!(
                "NVMe: {} aligned bytes/sample, 4 MiB/request, one outstanding pread/thread; dispatch and completion are timed",
                modes[0].1.bytes
            );
            for (threads, m) in modes {
                println!(
                    "{:<28} {:>12.3} {:>14}",
                    format!("NVMe O_DIRECT x{threads}"),
                    m.gb_per_s(),
                    "-"
                );
            }
        }
        println!("GEMV counts packed weights + BF16 scales; dispatch and completion are timed.");
        println!("ms/expert measures one rows x cols projection of an expert.");
        println!("Use GEMV effective bandwidth for §9.3 B; RAM read is the memory-only ceiling.");
        if let Some(rows) = rows {
            rows.print();
        }
    }
    Ok(())
}

#[derive(Debug)]
struct RowMeasurements {
    file_bytes: u64,
    row_bytes: usize,
    gather_core: u16,
    prefetch_core: u16,
    // cold/plain, cold/prefetch, warm/plain, warm/prefetch.
    modes: [Measurement; 4],
}

impl RowMeasurements {
    fn json(&self) -> String {
        let modes: Vec<String> = self.modes.iter().enumerate().map(|(i, m)| {
            format!(
                "{{\"cache\":\"{}\",\"prefetch\":{},\"bytes_per_sample\":{},\"gb_per_s\":{:.6},\"seconds\":{:.9},\"us_per_batch\":{:.6}}}",
                if i < 2 { "cold" } else { "warm" },
                i % 2 == 1,
                m.bytes,
                m.gb_per_s(),
                m.seconds,
                m.seconds * 1e6 / ROW_BATCHES as f64,
            )
        }).collect();
        format!(
            "{{\"file_bytes\":{},\"row_bytes\":{},\"batches\":{ROW_BATCHES},\"rows_per_batch\":{ROWS_PER_BATCH},\"seed\":{ROW_SEED},\"gather_core\":{},\"prefetch_core\":{},\"timing\":\"gather plus prefetch submission and waits\",\"cold_prepare\":\"MADV_DONTNEED + POSIX_FADV_DONTNEED\",\"warm_prepare\":\"gather selected rows\",\"modes\":[{}],\"cold_prefetch_speedup\":{:.6},\"warm_prefetch_speedup\":{:.6}}}",
            self.file_bytes,
            self.row_bytes,
            self.gather_core,
            self.prefetch_core,
            modes.join(","),
            self.modes[0].seconds / self.modes[1].seconds,
            self.modes[2].seconds / self.modes[3].seconds,
        )
    }

    fn print(&self) {
        println!(
            "Rows: {} file bytes, {} bytes/row, {ROW_BATCHES} batches x {ROWS_PER_BATCH} rows/sample, seed={ROW_SEED}; gather CPU {}, prefetch CPU {}",
            self.file_bytes, self.row_bytes, self.gather_core, self.prefetch_core,
        );
        println!("{:<28} {:>12} {:>14}", "row gather", "GB/s", "us/batch");
        for (name, m) in ["cold", "cold + prefetch", "warm", "warm + prefetch"]
            .iter()
            .zip(&self.modes)
        {
            println!(
                "{name:<28} {:>12.3} {:>14.3}",
                m.gb_per_s(),
                m.seconds * 1e6 / ROW_BATCHES as f64
            );
        }
        println!(
            "Prefetch speedup: cold {:.3}x, warm {:.3}x",
            self.modes[0].seconds / self.modes[1].seconds,
            self.modes[2].seconds / self.modes[3].seconds
        );
        println!(
            "Cold: MADV_DONTNEED + POSIX_FADV_DONTNEED before each sample (advisory). Warm: gather selected rows before each sample."
        );
        println!(
            "Submission and prefetch waits are timed; cache preparation is excluded. GB/s counts gathered row bytes only."
        );
    }
}

fn rows(path: &str, row_bytes: usize, machine: &Machine) -> Result<RowMeasurements> {
    let mut table = RowTable::open(path, row_bytes, machine)?;
    let buffer_bytes = row_bytes
        .checked_mul(ROWS_PER_BATCH)
        .ok_or_else(|| Error::InvalidPlan("row benchmark buffer size overflow".into()))?;
    let sample_bytes = buffer_bytes
        .checked_mul(ROW_BATCHES)
        .ok_or_else(|| Error::InvalidPlan("row benchmark sample size overflow".into()))?
        as u64;
    let mut output = vec![0u8; buffer_bytes];
    let mut random = ROW_SEED;
    let ids: Vec<usize> = (0..ROW_BATCHES * ROWS_PER_BATCH)
        .map(|_| {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            (random % table.rows() as u64) as usize
        })
        .collect();
    let mut modes = Vec::with_capacity(4);
    for (cold, prefetch) in [(true, false), (true, true), (false, false), (false, true)] {
        let mut samples = Vec::with_capacity(SAMPLES);
        for iteration in 0..WARMUP + SAMPLES {
            if cold {
                table.discard_cache()?;
            } else {
                gather_rows(&table, &ids, &mut output, false)?;
            }
            let start = Instant::now();
            gather_rows(&table, &ids, &mut output, prefetch)?;
            let seconds = start.elapsed().as_secs_f64();
            if iteration >= WARMUP {
                samples.push(seconds);
            }
        }
        samples.sort_unstable_by(f64::total_cmp);
        modes.push(Measurement {
            bytes: sample_bytes,
            seconds: samples[SAMPLES / 2],
        });
    }
    Ok(RowMeasurements {
        file_bytes: (table.rows() * row_bytes) as u64,
        row_bytes,
        gather_core: machine.cpu.compute[0],
        prefetch_core: machine.cpu.service[0],
        modes: modes.try_into().unwrap(),
    })
}

fn gather_rows(table: &RowTable, ids: &[usize], output: &mut [u8], prefetch: bool) -> Result<()> {
    let mut batches = ids.as_chunks::<ROWS_PER_BATCH>().0.iter().peekable();
    if prefetch && let Some(first) = batches.peek() {
        table.prefetch(first.as_slice())?.wait()?;
    }
    while let Some(batch) = batches.next() {
        // Read the next batch on the service core while gathering this batch.
        let next = if prefetch {
            batches
                .peek()
                .map(|rows| table.prefetch(rows.as_slice()))
                .transpose()?
        } else {
            None
        };
        table.gather(black_box(batch), output)?;
        black_box(&*output);
        if let Some(next) = next {
            next.wait()?;
        }
    }
    Ok(())
}

fn gemv(pool: &mut ComputePool, rows: usize, cols: usize) -> Result<Measurement> {
    let bytes = QuantSpec::Q4Bf16 { group: 32 }
        .matrix_bytes(rows as u64, cols as u64)
        .ok_or_else(|| Error::InvalidShape("invalid benchmark expert shape".into()))?;
    let weight_len = rows * cols / 2;
    let scale_len = rows * cols / 32;
    // 302 MB of distinct weights exceeds either host's LLC. No repeated single
    // expert that would inflate §9.3 B with cache bandwidth.
    let w: Vec<u8> = (0..weight_len * EXPERTS)
        .map(|i| i.wrapping_mul(101).wrapping_add(i / 16) as u8)
        .collect();
    let scales: Vec<u16> = (0..scale_len * EXPERTS)
        .map(|i| 0x3c00 + (i % 512) as u16)
        .collect();
    let x: Vec<f32> = (0..cols).map(|i| (i % 127) as f32 / 127.0 - 0.5).collect();
    let mut y = vec![0.0; rows];
    measure(bytes * EXPERTS as u64, || {
        for e in 0..EXPERTS {
            pool.gemv_q4_bf16(
                black_box(&w[e * weight_len..(e + 1) * weight_len]),
                black_box(&scales[e * scale_len..(e + 1) * scale_len]),
                black_box(&x),
                &mut y,
            )?;
            black_box(&y);
        }
        Ok(())
    })
}

fn nvme(path: &str, max_threads: usize, machine: &Machine) -> Result<Vec<(usize, Measurement)>> {
    const ALIGN: usize = 4096;
    const CHUNK: usize = 4 << 20;
    const O_DIRECT: i32 = 0o40000;
    let file: File = OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECT)
        .open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() < ALIGN as u64 {
        return Err(Error::InvalidPlan(
            "--nvme requires a regular file of at least 4096 bytes".into(),
        ));
    }
    let bytes = metadata.len() / ALIGN as u64 * ALIGN as u64;
    let pages = bytes / ALIGN as u64;
    if max_threads == 0 || max_threads as u64 > pages {
        return Err(Error::InvalidPlan(
            "NVMe thread count must be nonzero and not exceed aligned file pages".into(),
        ));
    }
    let mut modes = Vec::new();
    let mut threads = 1;
    loop {
        let measurement = std::thread::scope(|scope| -> Result<Measurement> {
            let mut workers = Vec::with_capacity(threads);
            for i in 0..threads {
                let core = machine.cpu.service[i % machine.cpu.service.len()];
                // Disjoint page-aligned contiguous regions cover the file once.
                let start = (pages / threads as u64 * i as u64
                    + (i as u64).min(pages % threads as u64))
                    * ALIGN as u64;
                let end = start
                    + (pages / threads as u64 + u64::from((i as u64) < pages % threads as u64))
                        * ALIGN as u64;
                let file = &file;
                let (jobs, requests) = mpsc::sync_channel(1);
                let (complete, done) = mpsc::sync_channel(1);
                let (ready_tx, ready) = mpsc::sync_channel(1);
                scope.spawn(move || {
                    let result = pin_thread(core);
                    let pinned = result.is_ok();
                    let mut storage = vec![0u8; CHUNK + ALIGN - 1];
                    let offset = storage.as_ptr().align_offset(ALIGN);
                    let buffer = &mut storage[offset..offset + CHUNK];
                    if ready_tx.send(result).is_err() || !pinned {
                        return;
                    }
                    while requests.recv().is_ok() {
                        let result = (|| -> Result<()> {
                            let mut position = start;
                            while position < end {
                                let count = (end - position).min(CHUNK as u64) as usize;
                                file.read_exact_at(&mut buffer[..count], position)?;
                                black_box(&buffer[..count]);
                                position += count as u64;
                            }
                            Ok(())
                        })();
                        if complete.send(result).is_err() {
                            return;
                        }
                    }
                });
                ready
                    .recv()
                    .map_err(|_| std::io::Error::other("NVMe worker stopped"))??;
                workers.push((jobs, done));
            }
            measure(bytes, || {
                for (jobs, _) in &workers {
                    jobs.send(())
                        .map_err(|_| std::io::Error::other("NVMe worker stopped"))?;
                }
                for (_, done) in &workers {
                    done.recv()
                        .map_err(|_| std::io::Error::other("NVMe worker stopped"))??;
                }
                Ok(())
            })
        })?;
        modes.push((threads, measurement));
        if threads == max_threads {
            break;
        }
        threads = threads.saturating_mul(2).min(max_threads);
    }
    Ok(modes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| (*s).into()).collect()
    }

    #[test]
    fn parses_host_options() {
        assert_eq!(
            Options::parse(&args(&["--nvme", "weights.hyt", "--json"])).unwrap(),
            Options {
                json: true,
                nvme: Some("weights.hyt".into()),
                nvme_threads: 8,
                rows: None,
                row_bytes: None,
            }
        );
        for values in [
            vec!["--nvme"],
            vec!["--nvme", "--json"],
            vec!["--json", "--json"],
            vec!["--bad"],
        ] {
            assert!(Options::parse(&args(&values)).is_err());
        }
    }

    #[test]
    fn parses_nvme_threads_and_rejects_invalid_values() {
        assert_eq!(
            Options::parse(&args(&["--nvme-threads", "3", "--nvme", "weights.hyt"]))
                .unwrap()
                .nvme_threads,
            3
        );
        for values in [
            vec!["--nvme-threads", "8"],
            vec!["--nvme", "file", "--nvme-threads"],
            vec!["--nvme", "file", "--nvme-threads", "0"],
            vec!["--nvme", "file", "--nvme-threads", "-1"],
            vec!["--nvme", "file", "--nvme-threads", "bad"],
            vec![
                "--nvme",
                "file",
                "--nvme-threads",
                "2",
                "--nvme-threads",
                "4",
            ],
        ] {
            assert!(Options::parse(&args(&values)).is_err(), "{values:?}");
        }
    }

    #[test]
    fn direct_reads_measure_concurrency_and_aligned_file_tail() {
        let path = std::env::temp_dir().join(format!("hayate-nvme-bench-{}", std::process::id()));
        // More than one 4 MiB request plus an incomplete final page.
        let bytes = (4 << 20) + 4096 * 7;
        std::fs::write(&path, vec![42u8; bytes + 17]).unwrap();
        File::open(&path).unwrap().sync_all().unwrap();
        let mut machine = *machine();
        machine.cpu.service = &[0];
        for threads in [1, 3, 8] {
            let modes = nvme(path.to_str().unwrap(), threads, &machine).unwrap();
            let expected = match threads {
                1 => vec![1],
                3 => vec![1, 2, 3],
                _ => vec![1, 2, 4, 8],
            };
            assert_eq!(modes.iter().map(|(n, _)| *n).collect::<Vec<_>>(), expected);
            for (_, m) in modes {
                assert_eq!(m.bytes, bytes as u64);
                assert!(m.seconds > 0.0 && m.gb_per_s().is_finite());
            }
        }
        assert!(nvme(path.to_str().unwrap(), 0, &machine).is_err());
        assert!(nvme(path.to_str().unwrap(), bytes / 4096 + 1, &machine).is_err());
        machine.cpu.service = &[1023];
        assert!(matches!(
            nvme(path.to_str().unwrap(), 2, &machine),
            Err(Error::Io(_))
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn parses_row_options_and_rejects_incomplete_or_repeated_values() {
        assert_eq!(
            Options::parse(&args(&[
                "--rows",
                "ngrams.bin",
                "--row-bytes",
                "1536",
                "--json"
            ]))
            .unwrap(),
            Options {
                json: true,
                nvme: None,
                nvme_threads: 8,
                rows: Some("ngrams.bin".into()),
                row_bytes: Some(1536),
            }
        );
        for values in [
            vec!["--rows"],
            vec!["--rows", "--json"],
            vec!["--rows", "table"],
            vec!["--row-bytes", "4"],
            vec!["--rows", "table", "--row-bytes"],
            vec!["--rows", "table", "--row-bytes", "0"],
            vec!["--rows", "table", "--row-bytes", "-1"],
            vec!["--rows", "table", "--row-bytes", "bad"],
            vec!["--rows", "table", "--row-bytes", "4", "--rows", "other"],
            vec!["--rows", "table", "--row-bytes", "4", "--row-bytes", "4"],
        ] {
            assert!(Options::parse(&args(&values)).is_err(), "{values:?}");
        }
    }

    #[test]
    fn row_benchmark_runs_all_cache_and_prefetch_modes() {
        let path = std::env::temp_dir().join(format!("hayate-row-bench-{}", std::process::id()));
        std::fs::write(&path, vec![42u8; 4096]).unwrap();
        File::open(&path).unwrap().sync_all().unwrap();
        let result = rows(path.to_str().unwrap(), 16, machine());
        std::fs::remove_file(path).unwrap();
        let measurements = result.unwrap();
        assert_eq!(measurements.file_bytes, 4096);
        for m in &measurements.modes {
            assert_eq!(m.bytes, (16 * ROW_BATCHES * ROWS_PER_BATCH) as u64);
            assert!(m.seconds > 0.0 && m.gb_per_s().is_finite());
        }
        assert!(measurements.json().contains("\"cold_prefetch_speedup\""));
    }
}
