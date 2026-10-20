use crate::{Error, QuantSpec, Result};

pub const MAX_RANK: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: [u64; MAX_RANK],
    rank: u8,
}

impl Shape {
    /// Rejects rank > `MAX_RANK`, zero-sized dimensions and element counts
    /// that overflow `u64`.
    pub fn new(dims: &[u64]) -> Result<Self> {
        if dims.len() > MAX_RANK {
            return Err(Error::InvalidShape(format!(
                "rank {} > {MAX_RANK}",
                dims.len()
            )));
        }
        if dims.contains(&0) {
            return Err(Error::InvalidShape(format!("zero dimension in {dims:?}")));
        }
        dims.iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| Error::InvalidShape(format!("element count overflows: {dims:?}")))?;
        let mut d = [1; MAX_RANK];
        d[..dims.len()].copy_from_slice(dims);
        Ok(Shape {
            dims: d,
            rank: dims.len() as u8,
        })
    }

    pub fn dims(&self) -> &[u64] {
        &self.dims[..self.rank as usize]
    }

    /// Cannot overflow: checked in `new`.
    pub fn numel(&self) -> u64 {
        self.dims().iter().product()
    }

    /// Flatten to `[rows, last_dim]`.
    pub fn as_matrix(&self) -> (u64, u64) {
        let cols = *self.dims().last().unwrap_or(&1);
        (self.numel() / cols, cols)
    }
}

/// A view into arena memory. Tensors never own memory; the arena does.
#[derive(Clone, Copy, Debug)]
pub struct TensorDesc {
    pub quant: QuantSpec,
    pub shape: Shape,
    /// Byte offset of the payload inside the GPU's arena.
    pub offset: u64,
}

impl TensorDesc {
    /// Total bytes including scales; errors if the format does not fit the shape.
    ///
    /// Block-scaled formats (NVFP4, MXFP8) treat every dim before the last two
    /// as a batch of independent matrices (e.g. `[experts, N, K]`): each matrix
    /// gets its own scale-tile padding and tensor scale, matching the per-expert
    /// layout grouped GEMMs read. Other formats are sized as one flat matrix.
    pub fn storage_bytes(&self) -> Result<u64> {
        let dims = self.shape.dims();
        let batched = matches!(self.quant, QuantSpec::Nvfp4 | QuantSpec::Mxfp8) && dims.len() > 2;
        let bytes = if batched {
            let (batch, mat) = dims.split_at(dims.len() - 2);
            let per_matrix = self.quant.matrix_bytes(mat[0], mat[1]);
            per_matrix.and_then(|b| b.checked_mul(batch.iter().product()))
        } else {
            let (rows, cols) = self.shape.as_matrix();
            self.quant.matrix_bytes(rows, cols)
        };
        bytes
            .ok_or_else(|| Error::InvalidShape(format!("{:?} cannot store {:?}", self.quant, dims)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_overflowing_shape() {
        assert!(Shape::new(&[1 << 32, 1 << 32]).is_err());
        assert!(Shape::new(&[3, 0]).is_err());
        assert!(Shape::new(&[1; 7]).is_err());
    }

    #[test]
    fn batched_nvfp4_pads_each_matrix() {
        // 128 Gemma 26B experts of [704, 2816]: padding applies per expert.
        let t = TensorDesc {
            quant: QuantSpec::Nvfp4,
            shape: Shape::new(&[128, 704, 2816]).unwrap(),
            offset: 0,
        };
        let one = QuantSpec::Nvfp4.matrix_bytes(704, 2816).unwrap();
        assert_eq!(t.storage_bytes().unwrap(), 128 * one);
    }
}
