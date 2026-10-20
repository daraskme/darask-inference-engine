//! Minimal GGUF (v2/v3) reader: metadata and F32 / F16 / BF16 tensors, enough for LoRA adapters
//! (llama.cpp's `general.type = adapter`, as heretic writes them). Dimensions are given in
//! ggml order (ne[0] = the innermost, contiguous one).

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    /// ggml order: ne[0] innermost.
    pub ne: Vec<u64>,
    /// ggml type id (0 F32, 1 F16, 30 BF16).
    pub ty: u32,
    /// Absolute file offset of the data.
    pub offset: u64,
}

#[derive(Debug)]
pub struct Gguf {
    pub meta: HashMap<String, Value>,
    pub tensors: Vec<Tensor>,
    path: std::path::PathBuf,
}

struct R<T: Read> {
    r: T,
}

impl<T: Read> R<T> {
    fn bytes<const N: usize>(&mut self) -> std::io::Result<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> std::io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> std::io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> std::io::Result<String> {
        let n = self.u64()? as usize;
        let mut b = vec![0u8; n];
        self.r.read_exact(&mut b)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
    fn value(&mut self, ty: u32) -> std::io::Result<Value> {
        Ok(match ty {
            0 => Value::Int(i64::from(u8::from_le_bytes(self.bytes()?))),
            1 => Value::Int(i64::from(i8::from_le_bytes(self.bytes()?))),
            2 => Value::Int(i64::from(u16::from_le_bytes(self.bytes()?))),
            3 => Value::Int(i64::from(i16::from_le_bytes(self.bytes()?))),
            4 => Value::Int(i64::from(self.u32()?)),
            5 => Value::Int(i64::from(i32::from_le_bytes(self.bytes()?))),
            6 => Value::Float(f64::from(f32::from_le_bytes(self.bytes()?))),
            7 => Value::Bool(self.bytes::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let (ety, n) = (self.u32()?, self.u64()?);
                Value::Array(
                    (0..n)
                        .map(|_| self.value(ety))
                        .collect::<std::io::Result<_>>()?,
                )
            }
            10 => Value::Int(self.u64()? as i64),
            11 => Value::Int(i64::from_le_bytes(self.bytes()?)),
            12 => Value::Float(f64::from_le_bytes(self.bytes()?)),
            _ => return Err(std::io::Error::other(format!("GGUF value type {ty}"))),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self, String> {
        let err = |e: std::io::Error| format!("{}: {e}", path.display());
        let f = std::fs::File::open(path).map_err(err)?;
        let mut r = R {
            r: std::io::BufReader::new(f),
        };
        if &r.bytes::<4>().map_err(err)? != b"GGUF" {
            return Err(format!("{}: not a GGUF file", path.display()));
        }
        let version = r.u32().map_err(err)?;
        if !(2..=3).contains(&version) {
            return Err(format!("{}: GGUF version {version}", path.display()));
        }
        let (n_tensors, n_kv) = (r.u64().map_err(err)?, r.u64().map_err(err)?);
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let key = r.string().map_err(err)?;
            let ty = r.u32().map_err(err)?;
            meta.insert(key, r.value(ty).map_err(err)?);
        }
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string().map_err(err)?;
            let nd = r.u32().map_err(err)?;
            let ne = (0..nd)
                .map(|_| r.u64())
                .collect::<std::io::Result<Vec<_>>>()
                .map_err(err)?;
            let ty = r.u32().map_err(err)?;
            let offset = r.u64().map_err(err)?;
            tensors.push(Tensor {
                name,
                ne,
                ty,
                offset,
            });
        }
        let align = match meta.get("general.alignment") {
            Some(Value::Int(a)) => *a as u64,
            _ => 32,
        };
        let pos = r.r.stream_position().map_err(err)?;
        let data = pos.div_ceil(align) * align;
        for t in &mut tensors {
            t.offset += data;
        }
        Ok(Self {
            meta,
            tensors,
            path: path.to_path_buf(),
        })
    }

    pub fn f64(&self, key: &str) -> Option<f64> {
        match self.meta.get(key)? {
            Value::Int(v) => Some(*v as f64),
            Value::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        match self.meta.get(key)? {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// A tensor's values as f32 (F32, F16 or BF16 storage).
    pub fn read_f32(&self, t: &Tensor) -> Result<Vec<f32>, String> {
        let n: u64 = t.ne.iter().product();
        let size = match t.ty {
            0 => 4,
            1 | 30 => 2,
            ty => return Err(format!("{}: ggml type {ty} not supported", t.name)),
        };
        let mut f = std::fs::File::open(&self.path).map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(t.offset))
            .map_err(|e| e.to_string())?;
        let mut b = vec![0u8; (n * size) as usize];
        f.read_exact(&mut b)
            .map_err(|e| format!("{}: {e}", t.name))?;
        Ok(match t.ty {
            0 => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            1 => b
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            _ => b
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect(),
        })
    }
}

/// IEEE half -> f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = (
        u32::from(h >> 15),
        u32::from((h >> 10) & 0x1f),
        u32::from(h & 0x3ff),
    );
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn half() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }
}
