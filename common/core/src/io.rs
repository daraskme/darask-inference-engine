//! NVMe mappings and page-cache prefetch for the N-gram table (§4.3).

use crate::{Error, Machine, Result};
use std::ffi::c_void;
use std::fs::File;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

const PAGE_BYTES: usize = 4096;
const PREFETCH_BYTES: usize = 128 << 10;
const MADV_WILLNEED: i32 = 3;
const MADV_DONTNEED: i32 = 4;
const POSIX_FADV_DONTNEED: i32 = 4;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> i32;
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
}

/// Read-only mapping of a non-empty regular file. The file must remain
/// unchanged (including its length) until the mapping is dropped.
#[derive(Debug)]
pub struct MappedFile {
    file: File,
    ptr: NonNull<c_void>,
    len: usize,
}

impl MappedFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > isize::MAX as u64 {
            return Err(Error::Weights(
                "mapping requires a non-empty regular file fitting isize".into(),
            ));
        }
        let len = metadata.len() as usize;
        // SAFETY: the descriptor is open; PROT_READ=1, MAP_PRIVATE=2, offset=0.
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, 1, 2, file.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            return Err(std::io::Error::last_os_error().into());
        }
        let ptr = NonNull::new(ptr).expect("Linux mmap cannot map address zero without MAP_FIXED");
        Ok(Self { file, ptr, len })
    }

    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: the mapping is readable for len bytes and lives until Drop.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast(), self.len) }
    }

    fn is_resident(&self, range: Range<usize>) -> Result<bool> {
        let mut offset = range.start / PAGE_BYTES * PAGE_BYTES;
        let mut resident = [0u8; PREFETCH_BYTES / PAGE_BYTES];
        while offset < range.end {
            let count = PREFETCH_BYTES.min(range.end - offset);
            // SAFETY: the range is within the mapping, the address is page aligned,
            // and resident has one byte per page (including the partial last page).
            let addr = unsafe { self.ptr.as_ptr().cast::<u8>().add(offset).cast() };
            if unsafe { mincore(addr, count, resident.as_mut_ptr()) } != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            if resident[..count.div_ceil(PAGE_BYTES)]
                .iter()
                .any(|&page| page & 1 == 0)
            {
                return Ok(false);
            }
            offset += count;
        }
        Ok(true)
    }

    /// Advise a byte range in 128 KiB chunks, rounding its start down to a page.
    pub fn prefetch(&self, range: Range<usize>) -> Result<()> {
        if range.start > range.end || range.end > self.len {
            return Err(Error::Weights(
                "mapping prefetch range is out of bounds".into(),
            ));
        }
        if range.is_empty() {
            return Ok(());
        }
        let mut offset = range.start / PAGE_BYTES * PAGE_BYTES;
        while offset < range.end {
            let count = PREFETCH_BYTES.min(range.end - offset);
            // SAFETY: offset is page aligned and this range lies in the mapping.
            let addr = unsafe { self.ptr.as_ptr().cast::<u8>().add(offset).cast() };
            if unsafe { madvise(addr, count, MADV_WILLNEED) } != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            offset += count;
        }
        Ok(())
    }

    /// Cold-benchmark preparation: release mapped pages before asking Linux to
    /// evict the file's page cache. Advisory only; other readers can retain it.
    pub fn discard_cache(&mut self) -> Result<()> {
        // SAFETY: this is the entire live mapping; no slices are borrowed.
        if unsafe { madvise(self.ptr.as_ptr(), self.len, MADV_DONTNEED) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        // SAFETY: fd remains open. A zero length means through end of file.
        let code = unsafe { posix_fadvise(self.file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED) };
        if code != 0 {
            return Err(std::io::Error::from_raw_os_error(code).into());
        }
        Ok(())
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        // SAFETY: this mapping is owned exclusively and has no remaining borrows.
        unsafe { munmap(self.ptr.as_ptr(), self.len) };
    }
}

type PrefetchJob = (Vec<usize>, SyncSender<Result<()>>);

/// Completion of a page-cache read. Dropping it does not cancel the read.
#[derive(Debug)]
pub struct Prefetch {
    done: Option<Receiver<Result<()>>>,
}

impl Prefetch {
    pub fn wait(self) -> Result<()> {
        match self.done {
            Some(done) => done.recv().map_err(|_| worker_stopped())?,
            None => Ok(()),
        }
    }
}

/// Raw fixed-size rows, copied without conversion into caller-owned storage.
/// One persistent pread worker is pinned to Machine::cpu.service[0].
#[derive(Debug)]
pub struct RowTable {
    mapped: MappedFile,
    row_bytes: usize,
    regions: Vec<(usize, usize)>,
    rows: usize,
    tx: Option<SyncSender<PrefetchJob>>,
    worker: Option<JoinHandle<()>>,
}

impl RowTable {
    pub fn open(path: impl AsRef<Path>, row_bytes: usize, machine: &Machine) -> Result<Self> {
        let len = usize::try_from(File::open(&path)?.metadata()?.len())
            .map_err(|_| Error::Weights("row file too large".into()))?;
        if row_bytes == 0 || !len.is_multiple_of(row_bytes) {
            return Err(Error::Weights(
                "row bytes must be nonzero and divide file length".into(),
            ));
        }
        Self::open_regions(path, row_bytes, &[(0, len / row_bytes)], machine)
    }

    /// Logical consecutive rows in disjoint payload regions of one HYT file.
    /// Each pair is (absolute byte offset, row count), in logical row order.
    pub fn open_regions(
        path: impl AsRef<Path>,
        row_bytes: usize,
        regions: &[(usize, usize)],
        machine: &Machine,
    ) -> Result<Self> {
        let mapped = MappedFile::open(&path)?;
        if row_bytes == 0 || regions.is_empty() {
            return Err(Error::Weights("empty row regions".into()));
        }
        let mut ranges = Vec::new();
        let mut rows = 0usize;
        for &(offset, count) in regions {
            let end = count
                .checked_mul(row_bytes)
                .and_then(|bytes| offset.checked_add(bytes))
                .filter(|&end| end <= mapped.len)
                .ok_or_else(|| Error::Weights("row region out of bounds".into()))?;
            if count == 0 {
                return Err(Error::Weights("empty row region".into()));
            }
            ranges.push(offset..end);
            rows = rows
                .checked_add(count)
                .ok_or_else(|| Error::Weights("row count overflow".into()))?;
        }
        ranges.sort_by_key(|range| range.start);
        if ranges.windows(2).any(|w| w[0].end > w[1].start) {
            return Err(Error::Weights("overlapping row regions".into()));
        }
        let core =
            *machine.cpu.service.first().ok_or_else(|| {
                Error::UnsupportedMachine("no service cores for row prefetch".into())
            })?;
        let file = mapped.file.try_clone()?;
        let (tx, jobs) = mpsc::sync_channel::<PrefetchJob>(1);
        let worker_regions = regions.to_vec();
        let (ready_tx, ready) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name(format!("hayate-io-{core}"))
            .spawn(move || {
                let result = pin_service(core);
                let pinned = result.is_ok();
                if ready_tx.send(result).is_err() || !pinned {
                    return;
                }
                let mut buffer = vec![0u8; row_bytes.min(PREFETCH_BYTES)];
                while let Ok((rows, done)) = jobs.recv() {
                    let result = read_rows(&file, row_bytes, &worker_regions, &rows, &mut buffer);
                    let _ = done.send(result);
                }
            })?;
        let table = Self {
            mapped,
            row_bytes,
            regions: regions.to_vec(),
            rows,
            tx: Some(tx),
            worker: Some(worker),
        };
        // Errors on startup are reported rather than silently running unpinned.
        ready.recv().map_err(|_| worker_stopped())??;
        Ok(table)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    fn validate_rows(&self, rows: &[usize]) -> Result<()> {
        if let Some(row) = rows.iter().find(|&&row| row >= self.rows()) {
            return Err(Error::Weights(format!(
                "row {row} is out of bounds for {} rows",
                self.rows()
            )));
        }
        Ok(())
    }

    /// Preserve input order and duplicates. Reject invalid input before writing.
    pub fn gather(&self, rows: &[usize], output: &mut [u8]) -> Result<()> {
        if rows.len().checked_mul(self.row_bytes) != Some(output.len()) {
            return Err(Error::Weights(
                "gather buffer must fit the requested rows exactly".into(),
            ));
        }
        self.validate_rows(rows)?;
        let bytes = self.mapped.as_bytes();
        for (&row, dst) in rows.iter().zip(output.chunks_exact_mut(self.row_bytes)) {
            let offset = row_offset(&self.regions, self.row_bytes, row);
            dst.copy_from_slice(&bytes[offset..offset + self.row_bytes]);
        }
        Ok(())
    }

    /// Queue only rows with missing pages for pread. Fully resident requests do
    /// not allocate, submit a job or wait on the worker. Residency is a snapshot;
    /// the OS can evict pages afterward, in which case gather faults them back in.
    /// The queue holds one pending job; further submissions apply backpressure.
    pub fn prefetch(&self, rows: &[usize]) -> Result<Prefetch> {
        self.validate_rows(rows)?;
        let mut missing = Vec::new();
        for &row in rows {
            let start = row_offset(&self.regions, self.row_bytes, row);
            if !self.mapped.is_resident(start..start + self.row_bytes)? {
                missing.push(row);
            }
        }
        if missing.is_empty() {
            return Ok(Prefetch { done: None });
        }
        let (complete, done) = mpsc::sync_channel(1);
        self.tx
            .as_ref()
            .unwrap()
            .send((missing, complete))
            .map_err(|_| worker_stopped())?;
        Ok(Prefetch { done: Some(done) })
    }

    /// Wait for outstanding prefetches before calling this benchmark helper.
    pub fn discard_cache(&mut self) -> Result<()> {
        self.mapped.discard_cache()
    }
}

impl Drop for RowTable {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn worker_stopped() -> Error {
    std::io::Error::other("row prefetch worker stopped").into()
}

fn pin_service(core: u16) -> Result<()> {
    let mut mask = [0u64; 16]; // Linux cpu_set_t (1024 bits).
    if usize::from(core) >= mask.len() * 64 {
        return Err(Error::UnsupportedMachine(format!(
            "CPU {core} exceeds affinity mask"
        )));
    }
    mask[usize::from(core) / 64] = 1 << (core % 64);
    // SAFETY: pid 0 names this thread; mask holds cpusetsize readable bytes.
    if unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

fn row_offset(regions: &[(usize, usize)], row_bytes: usize, mut row: usize) -> usize {
    for &(offset, count) in regions {
        if row < count {
            return offset + row * row_bytes;
        }
        row -= count;
    }
    unreachable!("validated logical row")
}

fn read_rows(
    file: &File,
    row_bytes: usize,
    regions: &[(usize, usize)],
    rows: &[usize],
    buffer: &mut [u8],
) -> Result<()> {
    for &row in rows {
        let start = row_offset(regions, row_bytes, row);
        let mut offset = 0;
        while offset < row_bytes {
            let count = buffer.len().min(row_bytes - offset);
            // FileExt uses pread, leaving the shared file offset untouched.
            file.read_exact_at(&mut buffer[..count], (start + offset) as u64)?;
            offset += count;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_machine() -> Machine {
        let mut machine = crate::machine::MAIN;
        // CPU 0 exists on both targets; these tests exercise I/O, not E-core speed.
        machine.cpu.service = &[0];
        machine
    }

    fn temp_file(bytes: &[u8]) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "hayate-io-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn table(bytes: &[u8], row_bytes: usize) -> RowTable {
        let path = temp_file(bytes);
        let result = RowTable::open(&path, row_bytes, &test_machine());
        std::fs::remove_file(path).unwrap();
        result.unwrap()
    }

    #[test]
    fn mapping_contents_and_prefetch_ranges() {
        let bytes: Vec<u8> = (0..PREFETCH_BYTES * 2 + 17).map(|i| i as u8).collect();
        let path = temp_file(&bytes);
        let mut mapped = MappedFile::open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(mapped.as_bytes(), bytes);
        mapped.prefetch(3..bytes.len()).unwrap();
        mapped.prefetch(bytes.len()..bytes.len()).unwrap();
        assert!(mapped.prefetch(0..bytes.len() + 1).is_err());
        assert!(mapped.prefetch(Range { start: 2, end: 1 }).is_err());
        mapped.discard_cache().unwrap();
        assert_eq!(mapped.as_bytes(), bytes);
    }

    #[test]
    fn rejects_empty_files_and_invalid_row_sizes() {
        let path = temp_file(&[]);
        assert!(MappedFile::open(&path).is_err());
        std::fs::write(&path, [1, 2, 3, 4]).unwrap();
        for size in [0, 3, 5] {
            assert!(RowTable::open(&path, size, &test_machine()).is_err());
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn gather_preserves_order_duplicates_and_row_boundaries() {
        let table = table(b"abcdefghijkl", 3);
        assert_eq!(table.rows(), 4);
        assert_eq!(table.row_bytes(), 3);
        let mut output = [0u8; 12];
        table.gather(&[3, 0, 2, 3], &mut output).unwrap();
        assert_eq!(&output, b"jklabcghijkl");
        table.gather(&[], &mut []).unwrap();
    }

    #[test]
    fn hyt_regions_preserve_logical_order_and_skip_padding() {
        let path = temp_file(b"HEADabcdefPADghijklTAIL");
        let table = RowTable::open_regions(&path, 3, &[(13, 2), (4, 2)], &test_machine()).unwrap();
        assert_eq!(table.rows(), 4);
        let mut out = [0u8; 12];
        table.prefetch(&[0, 2, 3, 0]).unwrap().wait().unwrap();
        table.gather(&[0, 2, 3, 0], &mut out).unwrap();
        assert_eq!(&out, b"ghiabcdefghi");
        for regions in [&[(4, 2), (5, 1)][..], &[(usize::MAX, 1)][..], &[(4, 0)][..]] {
            assert!(RowTable::open_regions(&path, 3, regions, &test_machine()).is_err());
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn gather_and_prefetch_reject_out_of_bounds_rows() {
        let table = table(b"abcdefgh", 4);
        let mut output = [99u8; 8];
        for rows in [[0, 2], [usize::MAX, 0]] {
            assert!(table.gather(&rows, &mut output).is_err());
            assert_eq!(output, [99u8; 8]);
            assert!(table.prefetch(&rows).is_err());
        }
        assert!(table.gather(&[0], &mut output).is_err());
        assert!(table.gather(&[0, 1], &mut output[..4]).is_err());
    }

    #[test]
    fn prefetch_completes_multiple_jobs_and_large_rows() {
        let row_bytes = PREFETCH_BYTES + 17;
        let bytes: Vec<u8> = (0..row_bytes * 3).map(|i| i as u8).collect();
        let mut table = table(&bytes, row_bytes);
        table.discard_cache().unwrap();
        let first = table.prefetch(&[2, 0, 2]).unwrap();
        let second = table.prefetch(&[1]).unwrap();
        first.wait().unwrap();
        second.wait().unwrap();
        table.prefetch(&[]).unwrap().wait().unwrap();
        let mut output = vec![0u8; row_bytes * 2];
        table.gather(&[2, 0], &mut output).unwrap();
        assert_eq!(&output[..row_bytes], &bytes[row_bytes * 2..]);
        assert_eq!(&output[row_bytes..], &bytes[..row_bytes]);
    }

    #[test]
    fn resident_prefetch_needs_no_worker_and_checks_partial_pages() {
        // Misaligned rows cross a page boundary, including a partial final page.
        let row_bytes = PAGE_BYTES + 1;
        let bytes = vec![42u8; row_bytes * 3];
        let mut table = table(&bytes, row_bytes);
        let mut output = vec![0u8; bytes.len()];
        table.gather(&[2, 0, 1], &mut output).unwrap();
        // A warm prefetch must succeed even after the worker is disconnected.
        table.tx.take();
        table.worker.take().unwrap().join().unwrap();
        table.prefetch(&[2, 0, 1, 2]).unwrap().wait().unwrap();
        table.prefetch(&[]).unwrap().wait().unwrap();
        assert_eq!(output, bytes);
    }

    #[test]
    fn prefetch_rechecks_cache_after_eviction_and_handles_mixed_rows() {
        let row_bytes = PAGE_BYTES;
        let bytes: Vec<u8> = (0..row_bytes * 3).map(|i| (i / row_bytes) as u8).collect();
        let path = temp_file(&bytes);
        File::open(&path).unwrap().sync_all().unwrap();
        let mut table = RowTable::open(&path, row_bytes, &test_machine()).unwrap();
        let mut output = vec![0u8; bytes.len()];
        table.gather(&[0, 1, 2], &mut output).unwrap();
        assert!(table.prefetch(&[0, 1, 2]).unwrap().done.is_none());
        table.discard_cache().unwrap();
        assert!(!table.mapped.is_resident(0..bytes.len()).unwrap());
        // pread just the middle page without faulting adjacent mapped pages.
        let mut page = vec![0u8; PAGE_BYTES];
        table
            .mapped
            .file
            .read_exact_at(&mut page, PAGE_BYTES as u64)
            .unwrap();
        assert!(
            table
                .mapped
                .is_resident(PAGE_BYTES..PAGE_BYTES * 2)
                .unwrap()
        );
        let prefetch = table.prefetch(&[1, 0, 2, 1]).unwrap();
        assert!(prefetch.done.is_some());
        prefetch.wait().unwrap();
        assert!(table.prefetch(&[0, 1, 2]).unwrap().done.is_none());
        table.gather(&[2, 1, 0], &mut output).unwrap();
        assert_eq!(&output[..row_bytes], &bytes[row_bytes * 2..]);
        assert_eq!(
            &output[row_bytes..row_bytes * 2],
            &bytes[row_bytes..row_bytes * 2]
        );
        assert_eq!(&output[row_bytes * 2..], &bytes[..row_bytes]);
        drop(table);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn prefetch_reports_read_errors() {
        let path = temp_file(b"abcdefgh");
        let table = RowTable::open(&path, 4, &test_machine()).unwrap();
        // Exercise pread failure without accessing the now-truncated mapping.
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();
        let error = table.prefetch(&[1]).unwrap().wait().unwrap_err();
        assert!(matches!(error, Error::Io(e) if e.kind() == std::io::ErrorKind::UnexpectedEof));
        drop(table);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_invalid_service_placement() {
        let path = temp_file(b"abcd");
        let mut machine = crate::machine::MAIN;
        machine.cpu.service = &[];
        assert!(RowTable::open(&path, 4, &machine).is_err());
        machine.cpu.service = &[1024];
        assert!(RowTable::open(&path, 4, &machine).is_err());
        machine.cpu.service = &[1023];
        assert!(matches!(
            RowTable::open(&path, 4, &machine),
            Err(Error::Io(_))
        ));
        std::fs::remove_file(path).unwrap();
    }
}
