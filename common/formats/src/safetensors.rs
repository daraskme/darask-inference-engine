//! safetensors headers: tensor name -> dtype, shape, absolute byte range in the file.

use crate::json::Json;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub file: PathBuf,
    /// Absolute offset of the data in `file`.
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    pub fn read(&self) -> std::io::Result<Vec<u8>> {
        let mut f = File::open(&self.file)?;
        f.seek(SeekFrom::Start(self.offset))?;
        let mut buf = vec![0; self.bytes as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    pub fn read_i64(&self) -> std::io::Result<Vec<i64>> {
        assert_eq!(self.dtype, "I64", "{}", self.name);
        Ok(self
            .read()?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect())
    }

    pub fn read_f32(&self) -> std::io::Result<Vec<f32>> {
        assert_eq!(self.dtype, "F32", "{}", self.name);
        Ok(self
            .read()?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
}

pub fn dtype_bytes(dtype: &str) -> Option<f64> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 8.0,
        "F32" | "I32" | "U32" => 4.0,
        "BF16" | "F16" | "I16" | "U16" => 2.0,
        "F8_E4M3" | "F8_E5M2" | "U8" | "I8" | "BOOL" | "F8_E8M0" => 1.0,
        "F4" => 0.5,
        _ => return None,
    })
}

/// Reads one file's header.
pub fn read_header(path: &Path) -> Result<Vec<TensorInfo>, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let len = u64::from_le_bytes(len);
    if len > 512 << 20 {
        return Err(format!(
            "{}: header length {len} is not plausible",
            path.display()
        ));
    }
    let mut text = vec![0; len as usize];
    f.read_exact(&mut text)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let text = String::from_utf8(text).map_err(|e| e.to_string())?;
    let json = Json::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = 8 + len;
    let mut out = Vec::new();
    for (name, v) in json.as_obj().ok_or("header is not an object")? {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v
            .get("dtype")
            .and_then(Json::as_str)
            .ok_or(format!("{name}: no dtype"))?
            .to_string();
        let shape = v
            .get("shape")
            .and_then(Json::as_arr)
            .ok_or(format!("{name}: no shape"))?;
        let shape = shape
            .iter()
            .map(|d| d.as_u64().ok_or(format!("{name}: bad dim")))
            .collect::<Result<Vec<_>, _>>()?;
        let off = v
            .get("data_offsets")
            .and_then(Json::as_arr)
            .ok_or(format!("{name}: no data_offsets"))?;
        let (a, b) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
        out.push(TensorInfo {
            name: name.clone(),
            dtype,
            shape,
            file: path.to_owned(),
            offset: base + a,
            bytes: b - a,
        });
    }
    Ok(out)
}

/// Reads every `*.safetensors` header in `dir`, sorted by name.
pub fn read_dir(dir: &Path) -> Result<Vec<TensorInfo>, String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    // Shards may live in a subdirectory (nvidia's `tensors/`): take the files the index names.
    if let Ok(text) = std::fs::read_to_string(dir.join("model.safetensors.index.json")) {
        let index = crate::json::Json::parse(&text)?;
        if let Some(map) = index.get("weight_map").and_then(|m| m.as_obj()) {
            for (_, file) in map {
                if let Some(f) = file.as_str() {
                    let p = dir.join(f);
                    // A shard left out on purpose (e.g. a table taken from another checkpoint,
                    // see the engine's tsumuji.json) is skipped; missing tensors fail where used.
                    if !files.contains(&p) && p.exists() {
                        files.push(p);
                    }
                }
            }
        }
    }
    files.sort();
    let mut all = Vec::new();
    for f in files {
        all.extend(read_header(&f)?);
    }
    all.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(all)
}

impl TensorInfo {
    /// Reads the tensor's bytes into `dst` (exactly `bytes` long) with a positioned read, so
    /// many threads can fill disjoint parts of one buffer.
    pub fn read_into(&self, dst: &mut [u8]) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        assert_eq!(dst.len() as u64, self.bytes, "{}", self.name);
        File::open(&self.file)?.read_exact_at(dst, self.offset)
    }

    /// Reads `dst.len()` bytes of the tensor starting at byte `start` of its data.
    pub fn read_range_into(&self, start: u64, dst: &mut [u8]) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        assert!(start + dst.len() as u64 <= self.bytes, "{}", self.name);
        File::open(&self.file)?.read_exact_at(dst, self.offset + start)
    }

    /// The values as f32, from F32 or BF16 storage.
    pub fn read_as_f32(&self) -> std::io::Result<Vec<f32>> {
        let raw = self.read()?;
        Ok(match self.dtype.as_str() {
            "F32" => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            "BF16" => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
                .collect(),
            other => panic!("{}: cannot read {other} as f32", self.name),
        })
    }
}
