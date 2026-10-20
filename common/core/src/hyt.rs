//! `.hyt` weight files: checkpoints pre-converted by `hayate_tools.convert` into the
//! exact format, layout and tensor order the kernels consume, so loading is
//! one positional file read per tensor with no transformation at startup.
//!
//! One `.hyt` file is built per (model, machine) pair — the main and sub PCs
//! get different quantization and expert placement. The byte-level format is
//! specified in ARCHITECTURE.md §5.3; this module mirrors it.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom, Take};
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::tensor::MAX_RANK;
use crate::{DType, Error, MachineId, QuantSpec, Result, Shape, TensorDesc};

pub const MAGIC: [u8; 4] = *b"HYT0";
/// Bumped whenever the header, entry table or any kernel-visible layout changes.
/// The loader rejects files with a different version instead of guessing.
pub const FORMAT_VERSION: u32 = 2;
pub const DATA_ALIGN: u64 = 4096;

/// Fixed 64-byte little-endian header at offset 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub format_version: u32,
    /// `MachineId` as u32: 0 = main, 1 = sub.
    pub machine: u32,
    pub tensor_count: u32,
    /// Byte offset of the entry table (follows the header).
    pub table_offset: u64,
    /// Byte offset of the 4 KiB-aligned data region.
    pub data_offset: u64,
    /// First 16 bytes of SHA-256 over the source safetensors files, so a
    /// stale conversion of an updated checkpoint is detected.
    pub source_hash: [u8; 16],
}

pub const HEADER_LEN: usize = 64;

impl Header {
    pub fn decode(b: &[u8; HEADER_LEN]) -> Result<Self> {
        if b[0..4] != MAGIC {
            return Err(Error::Weights("not a .hyt file".into()));
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let h = Header {
            format_version: u32_at(4),
            machine: u32_at(8),
            tensor_count: u32_at(12),
            table_offset: u64_at(16),
            data_offset: u64_at(24),
            source_hash: b[32..48].try_into().unwrap(),
        };
        if h.format_version != FORMAT_VERSION {
            return Err(Error::Weights(format!(
                "format version {} != {FORMAT_VERSION}",
                h.format_version
            )));
        }
        if h.machine > 1 {
            return Err(Error::Weights(format!("unknown machine id {}", h.machine)));
        }
        // Layout: header, then table, then the 4 KiB-aligned data region.
        // A non-empty table must have room before the data region; its exact
        // length and the file end are checked by the reader, not here.
        let table_fits = if h.tensor_count == 0 {
            h.table_offset <= h.data_offset
        } else {
            h.table_offset < h.data_offset
        };
        if !h.data_offset.is_multiple_of(DATA_ALIGN)
            || h.table_offset < HEADER_LEN as u64
            || !table_fits
        {
            return Err(Error::Weights("misaligned or overlapping regions".into()));
        }
        Ok(h)
    }

    pub fn encode(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..4].copy_from_slice(&MAGIC);
        b[4..8].copy_from_slice(&self.format_version.to_le_bytes());
        b[8..12].copy_from_slice(&self.machine.to_le_bytes());
        b[12..16].copy_from_slice(&self.tensor_count.to_le_bytes());
        b[16..24].copy_from_slice(&self.table_offset.to_le_bytes());
        b[24..32].copy_from_slice(&self.data_offset.to_le_bytes());
        b[32..48].copy_from_slice(&self.source_hash);
        b
    }
}

/// One table entry; `offset` is relative to the file's data region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    pub quant: QuantSpec,
    pub shape: Shape,
    pub offset: u64,
    pub bytes: u64,
}

impl TensorEntry {
    fn validate(&self) -> Result<()> {
        if self.name.is_empty() || u32::try_from(self.name.len()).is_err() {
            return Err(Error::Weights("invalid tensor name length".into()));
        }
        let expected = TensorDesc {
            quant: self.quant,
            shape: self.shape,
            offset: self.offset,
        }
        .storage_bytes()
        .map_err(|e| Error::Weights(e.to_string()))?;
        if self.bytes != expected || !self.offset.is_multiple_of(DATA_ALIGN) {
            return Err(Error::Weights(format!(
                "{}: invalid size or alignment",
                self.name
            )));
        }
        Ok(())
    }
}

// Wire IDs are explicit; Rust enum discriminants are never written to disk.
fn dtype_tag(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => 0,
        DType::Bf16 => 1,
        DType::F16 => 2,
        DType::F8E4M3 => 3,
        DType::F8E5M2 => 4,
        DType::F4E2M1 => 5,
        DType::I8 => 6,
        DType::I32 => 7,
        DType::I4 => 8,
    }
}

fn quant_words(quant: QuantSpec) -> [u32; 3] {
    match quant {
        QuantSpec::Plain(d) => [0, dtype_tag(d), 0],
        QuantSpec::PerTensor(d) => [1, dtype_tag(d), 0],
        QuantSpec::Nvfp4 => [2, 0, 0],
        QuantSpec::Mxfp8 => [3, 0, 0],
        QuantSpec::Q4Bf16 { group } => [4, 0, group],
        QuantSpec::Int8Row { convrot_group } => [5, 0, convrot_group],
        QuantSpec::Lowbit { grid, group } => [6, grid as u32, group as u32],
        QuantSpec::Exl3 { k2, cb } => [7, k2 as u32, cb as u32],
    }
}

fn decode_quant([tag, dtype, group]: [u32; 3]) -> Result<QuantSpec> {
    match (tag, dtype, group) {
        (0 | 1, 0..=8, 0) => {
            let d = match dtype {
                0 => DType::F32,
                1 => DType::Bf16,
                2 => DType::F16,
                3 => DType::F8E4M3,
                4 => DType::F8E5M2,
                5 => DType::F4E2M1,
                6 => DType::I8,
                7 => DType::I32,
                8 => DType::I4,
                _ => unreachable!(),
            };
            Ok(if tag == 0 {
                QuantSpec::Plain(d)
            } else {
                QuantSpec::PerTensor(d)
            })
        }
        (2, 0, 0) => Ok(QuantSpec::Nvfp4),
        (3, 0, 0) => Ok(QuantSpec::Mxfp8),
        (4, 0, 1..) => Ok(QuantSpec::Q4Bf16 { group }),
        (5, 0, convrot_group) => Ok(QuantSpec::Int8Row { convrot_group }),
        (6, 0..=4, 16 | 32) => Ok(QuantSpec::Lowbit {
            grid: dtype as u8,
            group: group as u8,
        }),
        (7, 2..=16, 0..=2) if QuantSpec::exl3_valid(dtype as u8, group as u8) => {
            Ok(QuantSpec::Exl3 {
                k2: dtype as u8,
                cb: group as u8,
            })
        }
        _ => Err(Error::Weights(
            "unknown QuantSpec or nonzero unused parameter".into(),
        )),
    }
}

/// Encode entries without header or alignment padding.
pub fn encode_table(entries: &[TensorEntry]) -> Result<Vec<u8>> {
    u32::try_from(entries.len()).map_err(|_| Error::Weights("too many tensors".into()))?;
    validate_entries(entries, u64::MAX)?;
    let mut out = Vec::new();
    for entry in entries {
        out.extend_from_slice(&(entry.name.len() as u32).to_le_bytes());
        out.extend_from_slice(entry.name.as_bytes());
        for word in quant_words(entry.quant) {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.extend_from_slice(&(entry.shape.dims().len() as u32).to_le_bytes());
        for dim in entry.shape.dims() {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&entry.offset.to_le_bytes());
        out.extend_from_slice(&entry.bytes.to_le_bytes());
    }
    Ok(out)
}

/// Decode exactly `count` entries, returning bytes consumed before table padding.
pub fn decode_table(bytes: &[u8], count: u32) -> Result<(Vec<TensorEntry>, u64)> {
    let mut reader = Cursor::new(bytes).take(bytes.len() as u64);
    let entries = read_table(&mut reader, count)?;
    validate_entries(&entries, u64::MAX)?;
    Ok((entries, bytes.len() as u64 - reader.limit()))
}

fn read_word<const N: usize>(reader: &mut impl Read) -> Result<[u8; N]> {
    let mut bytes = [0; N];
    reader.read_exact(&mut bytes).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::Weights("truncated tensor table".into())
        } else {
            Error::Io(e)
        }
    })?;
    Ok(bytes)
}

fn read_table(reader: &mut Take<impl Read>, count: u32) -> Result<Vec<TensorEntry>> {
    // The smallest entry is a scalar with a one-byte name (37 bytes).
    if u64::from(count) > reader.limit() / 37 {
        return Err(Error::Weights("tensor count exceeds table region".into()));
    }
    let mut entries = Vec::new();
    for _ in 0..count {
        let name_len = u32::from_le_bytes(read_word(reader)?) as usize;
        if name_len == 0 || name_len as u64 > reader.limit().saturating_sub(32) {
            return Err(Error::Weights("invalid tensor name length".into()));
        }
        let mut name = vec![0; name_len];
        reader.read_exact(&mut name)?;
        let name = String::from_utf8(name)
            .map_err(|_| Error::Weights("tensor name is not UTF-8".into()))?;
        let quant = decode_quant([
            u32::from_le_bytes(read_word(reader)?),
            u32::from_le_bytes(read_word(reader)?),
            u32::from_le_bytes(read_word(reader)?),
        ])?;
        let rank = u32::from_le_bytes(read_word(reader)?) as usize;
        if rank > MAX_RANK {
            return Err(Error::Weights(format!("rank {rank} > {MAX_RANK}")));
        }
        let mut dims = [0; MAX_RANK];
        for dim in &mut dims[..rank] {
            *dim = u64::from_le_bytes(read_word(reader)?);
        }
        let shape = Shape::new(&dims[..rank]).map_err(|e| Error::Weights(e.to_string()))?;
        entries.push(TensorEntry {
            name,
            quant,
            shape,
            offset: u64::from_le_bytes(read_word(reader)?),
            bytes: u64::from_le_bytes(read_word(reader)?),
        });
    }
    Ok(entries)
}

fn validate_entries(entries: &[TensorEntry], data_bytes: u64) -> Result<()> {
    let mut names = HashSet::new();
    let mut ranges = Vec::new();
    for entry in entries {
        entry.validate()?;
        if !names.insert(entry.name.as_str()) {
            return Err(Error::Weights(format!("duplicate tensor {}", entry.name)));
        }
        let end = entry
            .offset
            .checked_add(entry.bytes)
            .filter(|&end| end <= data_bytes)
            .ok_or_else(|| Error::Weights(format!("{}: tensor outside data region", entry.name)))?;
        ranges.push((entry.offset, end));
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(Error::Weights("overlapping tensors".into()));
    }
    Ok(())
}

/// Validated metadata and an open file; tensor reads neither allocate nor seek.
#[derive(Debug)]
pub struct HytFile {
    file: File,
    header: Header,
    entries: Vec<TensorEntry>,
}

impl HytFile {
    pub fn open(path: impl AsRef<Path>, machine: MachineId) -> Result<Self> {
        let mut file = File::open(path)?;
        let file_bytes = file.metadata()?.len();
        let mut bytes = [0; HEADER_LEN];
        file.read_exact_at(&mut bytes, 0)?;
        let header = Header::decode(&bytes)?;
        let expected_machine = match machine {
            MachineId::Main => 0,
            MachineId::Sub => 1,
        };
        if header.machine != expected_machine {
            return Err(Error::Weights(
                "checkpoint targets a different machine".into(),
            ));
        }
        if header.data_offset > file_bytes {
            return Err(Error::Weights("data region outside file".into()));
        }
        file.seek(SeekFrom::Start(header.table_offset))?;
        let entries = read_table(
            &mut BufReader::new(&file).take(header.data_offset - header.table_offset),
            header.tensor_count,
        )?;
        validate_entries(&entries, file_bytes - header.data_offset)?;
        Ok(Self {
            file,
            header,
            entries,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn entries(&self) -> &[TensorEntry] {
        &self.entries
    }

    /// Read a tensor into a caller-owned staging buffer of exactly its size.
    pub fn read_tensor(&self, index: usize, dst: &mut [u8]) -> Result<()> {
        let entry = self
            .entries
            .get(index)
            .ok_or_else(|| Error::Weights(format!("tensor index {index} out of range")))?;
        if dst.len() as u64 != entry.bytes {
            return Err(Error::Weights(format!(
                "{}: wrong destination size",
                entry.name
            )));
        }
        self.read_tensor_range(index, 0, dst)
    }

    /// Read a bounded byte range without staging the entire tensor in RAM.
    pub fn read_tensor_range(&self, index: usize, offset: u64, dst: &mut [u8]) -> Result<()> {
        let entry = self
            .entries
            .get(index)
            .ok_or_else(|| Error::Weights(format!("tensor index {index} out of range")))?;
        if offset > entry.bytes || dst.len() as u64 > entry.bytes - offset {
            return Err(Error::Weights(format!(
                "{}: tensor read out of range",
                entry.name
            )));
        }
        // The complete entry's file range was checked by HytFile::open.
        self.file
            .read_exact_at(dst, self.header.data_offset + entry.offset + offset)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowbit_quant_words_round_trip() {
        for grid in 0..=4u8 {
            for group in [16u8, 32] {
                let q = QuantSpec::Lowbit { grid, group };
                assert_eq!(decode_quant(quant_words(q)).unwrap(), q);
            }
        }
        assert!(decode_quant([6, 5, 16]).is_err());
        assert!(decode_quant([6, 3, 8]).is_err());
    }

    #[test]
    fn exl3_quant_words_round_trip() {
        for (k2, cb) in [(5, 2), (7, 2), (3, 2), (4, 0), (6, 1), (16, 2)] {
            let q = QuantSpec::Exl3 { k2, cb };
            assert_eq!(decode_quant(quant_words(q)).unwrap(), q);
        }
        for words in [[7, 5, 1], [7, 1, 2], [7, 17, 2], [7, 4, 3], [7, 9, 2]] {
            assert!(decode_quant(words).is_err(), "{words:?}");
        }
    }

    #[test]
    fn header_round_trips() {
        let h = Header {
            format_version: FORMAT_VERSION,
            machine: 1,
            tensor_count: 42,
            table_offset: 64,
            data_offset: 8192,
            source_hash: [7; 16],
        };
        assert_eq!(Header::decode(&h.encode()).unwrap(), h);
    }

    fn header(machine: u32, tensor_count: u32, table_offset: u64, data_offset: u64) -> Header {
        Header {
            format_version: FORMAT_VERSION,
            machine,
            tensor_count,
            table_offset,
            data_offset,
            source_hash: [0; 16],
        }
    }

    #[test]
    fn rejects_bad_regions_and_machine() {
        let bad = [
            header(0, 1, 64, 0),      // data inside the header
            header(0, 1, 8192, 4096), // table after data
            header(0, 1, 4096, 4096), // no room for a non-empty table
            header(0, 0, 64, 100),    // data not 4 KiB aligned
            header(9, 0, 64, 4096),   // unknown machine
        ];
        for h in bad {
            assert!(Header::decode(&h.encode()).is_err(), "{h:?}");
        }
        assert!(Header::decode(&header(1, 0, 4096, 4096).encode()).is_ok());
    }

    #[test]
    fn rejects_other_versions() {
        let mut b = Header {
            format_version: FORMAT_VERSION,
            machine: 0,
            tensor_count: 0,
            table_offset: 64,
            data_offset: 4096,
            source_hash: [0; 16],
        }
        .encode();
        b[4] = 99;
        assert!(Header::decode(&b).is_err());
    }
    const GOLDEN: &[u8] = include_bytes!("../../../engines/hayate/python-tools/tests/golden.hyt");

    fn golden_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../engines/hayate/python-tools/tests/golden.hyt")
    }

    fn entry(name: &str, quant: QuantSpec, dims: &[u64], offset: u64) -> TensorEntry {
        let shape = Shape::new(dims).unwrap();
        TensorEntry {
            name: name.into(),
            quant,
            shape,
            offset,
            bytes: TensorDesc {
                quant,
                shape,
                offset,
            }
            .storage_bytes()
            .unwrap(),
        }
    }

    #[test]
    fn table_round_trips_all_quant_specs_and_dtypes() {
        let dtypes = [
            DType::F32,
            DType::Bf16,
            DType::F16,
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F4E2M1,
            DType::I8,
            DType::I32,
            DType::I4,
        ];
        let mut specs = vec![
            QuantSpec::Nvfp4,
            QuantSpec::Mxfp8,
            QuantSpec::Q4Bf16 { group: 32 },
            QuantSpec::Int8Row { convrot_group: 256 },
            QuantSpec::Int8Row { convrot_group: 0 },
        ];
        for dtype in dtypes {
            specs.extend([QuantSpec::Plain(dtype), QuantSpec::PerTensor(dtype)]);
        }
        let entries: Vec<_> = specs
            .into_iter()
            .enumerate()
            .map(|(i, quant)| entry(&format!("重み{i}"), quant, &[2, 64], i as u64 * DATA_ALIGN))
            .collect();
        let encoded = encode_table(&entries).unwrap();
        let (decoded, used) = decode_table(&encoded, entries.len() as u32).unwrap();
        assert_eq!(decoded, entries);
        assert_eq!(used, encoded.len() as u64);
        let scalar = entry("scalar", QuantSpec::Plain(DType::F32), &[], 0);
        assert_eq!(
            decode_table(&encode_table(std::slice::from_ref(&scalar)).unwrap(), 1)
                .unwrap()
                .0,
            vec![scalar]
        );
    }

    #[test]
    fn shared_python_golden_loads_and_reads() {
        let weights = HytFile::open(golden_path(), MachineId::Main).unwrap();
        assert_eq!(
            weights.header().source_hash,
            [
                0xc5, 0xe8, 0xba, 0x63, 0x15, 0x7f, 0xbf, 0xf9, 0x98, 0x4d, 0x1f, 0x11, 0xbc, 0x7a,
                0xc5, 0x6a
            ]
        );
        let expected = [
            entry("重み", QuantSpec::PerTensor(DType::F8E4M3), &[2, 2], 0),
            entry("half", QuantSpec::Plain(DType::F16), &[], 4096),
            entry("float", QuantSpec::Plain(DType::F32), &[1], 8192),
        ];
        assert_eq!(weights.entries(), expected);
        let encoded = encode_table(&expected).unwrap();
        assert_eq!(&GOLDEN[64..64 + encoded.len()], encoded);
        let mut fp8 = [0; 8];
        let mut half = [0; 2];
        let mut float = [0; 4];
        weights.read_tensor(2, &mut float).unwrap();
        weights.read_tensor(0, &mut fp8).unwrap();
        weights.read_tensor(1, &mut half).unwrap();
        assert_eq!(fp8, [0x00, 0x38, 0xb8, 0x7e, 0, 0, 0x80, 0x3f]);
        assert_eq!(half, [0x00, 0x38]);
        assert_eq!(float, 1f32.to_le_bytes());
        assert!(weights.read_tensor(3, &mut fp8).is_err());
        assert!(weights.read_tensor(0, &mut half).is_err());
        assert!(HytFile::open(golden_path(), MachineId::Sub).is_err());
    }

    #[test]
    fn tensor_ranges_match_full_read_and_stay_inside_entry() {
        let weights = HytFile::open(golden_path(), MachineId::Main).unwrap();
        let mut whole = [0; 8];
        weights.read_tensor(0, &mut whole).unwrap();
        for start in 0..=whole.len() {
            for end in start..=whole.len() {
                let mut part = vec![0; end - start];
                weights
                    .read_tensor_range(0, start as u64, &mut part)
                    .unwrap();
                assert_eq!(part, whole[start..end]);
            }
        }
        for offset in [8, 9, u64::MAX] {
            assert!(weights.read_tensor_range(0, offset, &mut [0]).is_err());
        }
        assert!(weights.read_tensor_range(0, 9, &mut []).is_err());
        assert!(weights.read_tensor_range(3, 0, &mut []).is_err());
        // Bytes after an entry (padding or the next tensor) are never readable through it.
        assert!(weights.read_tensor_range(1, 1, &mut [0; 2]).is_err());
    }

    #[test]
    fn table_rejects_truncation_and_malformed_fields() {
        let encoded = encode_table(&[entry("a", QuantSpec::Plain(DType::Bf16), &[2], 0)]).unwrap();
        for len in 0..encoded.len() {
            assert!(decode_table(&encoded[..len], 1).is_err(), "{len}");
        }
        // name length, name UTF-8, quant tag/dtype/unused param, rank, dimension,
        // tensor offset alignment and byte count.
        let mutations: &[(usize, &[u8])] = &[
            (0, &0u32.to_le_bytes()),
            (0, &u32::MAX.to_le_bytes()),
            (4, &[0xff]),
            (5, &9u32.to_le_bytes()),
            (9, &99u32.to_le_bytes()),
            (13, &1u32.to_le_bytes()),
            (17, &7u32.to_le_bytes()),
            (21, &0u64.to_le_bytes()),
            (29, &1u64.to_le_bytes()),
            (37, &5u64.to_le_bytes()),
        ];
        for &(offset, replacement) in mutations {
            let mut bad = encoded.clone();
            bad[offset..offset + replacement.len()].copy_from_slice(replacement);
            assert!(decode_table(&bad, 1).is_err(), "offset {offset}");
        }
        assert!(decode_table(&encoded, u32::MAX).is_err());
        assert!(decode_table(&encoded, 2).is_err());
        let mut padded = encoded.clone();
        padded.extend_from_slice(&[0; 100]);
        assert_eq!(decode_table(&padded, 1).unwrap().1, encoded.len() as u64);
    }

    #[test]
    fn table_rejects_overlap_duplicates_and_overflow() {
        let first = entry("a", QuantSpec::Plain(DType::Bf16), &[4096], 0);
        let overlapping = entry("b", QuantSpec::Plain(DType::F32), &[1], 4096);
        assert!(encode_table(&[first, overlapping]).is_err());
        let a = entry("a", QuantSpec::Plain(DType::F32), &[1], 0);
        let b = entry("a", QuantSpec::Plain(DType::F32), &[1], 4096);
        assert!(encode_table(&[a, b]).is_err());
        let overflowing = entry("a", QuantSpec::Plain(DType::Bf16), &[4096], u64::MAX - 4095);
        assert!(encode_table(&[overflowing]).is_err());
        let invalid_block = TensorEntry {
            name: "bad".into(),
            quant: QuantSpec::Nvfp4,
            shape: Shape::new(&[17]).unwrap(),
            offset: 0,
            bytes: 1,
        };
        assert!(encode_table(&[invalid_block]).is_err());
        let huge = TensorEntry {
            name: "huge".into(),
            quant: QuantSpec::Plain(DType::F32),
            shape: Shape::new(&[1 << 59]).unwrap(),
            offset: 0,
            bytes: 0,
        };
        assert!(encode_table(&[huge]).is_err());
    }

    fn load_bytes(bytes: &[u8]) -> Result<HytFile> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "hayate-hyt-{}-{}.hyt",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        let result = HytFile::open(&path, MachineId::Main);
        std::fs::remove_file(path).unwrap();
        result
    }

    #[test]
    fn file_rejects_out_of_bounds_table_and_tensors() {
        // First entry ends at byte 122, second at 162, third at 211.
        let mutations: &[(usize, &[u8])] = &[
            (0, b"BAD!"),
            (4, &1u32.to_le_bytes()),
            (12, &u32::MAX.to_le_bytes()),
            (16, &4080u64.to_le_bytes()),   // table crosses into data
            (24, &16384u64.to_le_bytes()),  // data outside file
            (106, &12288u64.to_le_bytes()), // tensor outside data
            (106, &(u64::MAX - 4095).to_le_bytes()), // range overflow
            (146, &0u64.to_le_bytes()),     // overlap with first tensor
            (114, &9u64.to_le_bytes()),     // wrong storage size
        ];
        for &(offset, replacement) in mutations {
            let mut bad = GOLDEN.to_vec();
            bad[offset..offset + replacement.len()].copy_from_slice(replacement);
            assert!(load_bytes(&bad).is_err(), "offset {offset}");
        }
        for len in [0, 63, 4095, GOLDEN.len() - 1] {
            assert!(load_bytes(&GOLDEN[..len]).is_err(), "length {len}");
        }
        // Enough total file space, but the table itself must stop at data_offset.
        let large = entry("wide", QuantSpec::Plain(DType::Bf16), &[1; 6], 0);
        let mut table = encode_table(&[large]).unwrap();
        table.resize(4096, 0);
        let header = header(0, 1, 4060, 4096);
        let mut bytes = header.encode().to_vec();
        bytes.resize(4060, 0);
        bytes.extend(table);
        assert!(load_bytes(&bytes).is_err());
    }

    #[test]
    fn empty_file_and_batched_nvfp4_sizes() {
        let mut bytes = header(0, 0, 64, 4096).encode().to_vec();
        bytes.resize(4096, 0);
        assert!(load_bytes(&bytes).unwrap().entries().is_empty());
        let tensor = entry("experts", QuantSpec::Nvfp4, &[2, 1, 16], 0);
        assert_eq!(tensor.bytes, 1048);
        let table = encode_table(std::slice::from_ref(&tensor)).unwrap();
        let mut bytes = header(0, 1, 64, 4096).encode().to_vec();
        bytes.extend(table);
        bytes.resize(4096 + tensor.bytes as usize, 0);
        assert_eq!(load_bytes(&bytes).unwrap().entries(), [tensor]);
    }
}
