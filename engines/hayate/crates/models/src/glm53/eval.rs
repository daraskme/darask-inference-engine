//! Teacher-forced rows follow Qwen: row 0 predicts after consuming ids[prefix].
use hayate_core::{Error, Result};
use std::{io::Write, path::Path};

pub(super) fn read_ids(path: &Path, prefix: usize, context: usize) -> Result<Vec<u32>> {
    let raw = std::fs::read(path)?;
    if raw.is_empty() || !raw.len().is_multiple_of(4) {
        return Err(Error::InvalidShape("GLM eval requires I32 LE IDs".into()));
    }
    let ids: Vec<_> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    if prefix == 0 || prefix >= ids.len().saturating_sub(1) || ids.len() > context {
        return Err(Error::InvalidShape(
            "GLM eval needs prefix > 0 and at least two scored inputs within context".into(),
        ));
    }
    Ok(ids)
}

pub(super) fn write_logits(out: &mut impl Write, raw: &[u8]) -> Result<()> {
    if raw.is_empty()
        || !raw.len().is_multiple_of(2)
        || raw
            .as_chunks::<2>()
            .0
            .iter()
            .any(|b| u16::from_le_bytes(*b) & 0x7f80 == 0x7f80)
    {
        return Err(Error::Weights(
            "GLM eval nonfinite/invalid BF16 logits".into(),
        ));
    }
    out.write_all(raw)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ids_and_native_logits_validation() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("hayate-glm-eval-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("ids.bin");
        std::fs::write(
            &path,
            [1u32, 2, 3, 4]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        assert_eq!(read_ids(&path, 1, 4)?, [1, 2, 3, 4]);
        for (prefix, context) in [(0, 4), (3, 4), (4, 4), (1, 3)] {
            assert!(read_ids(&path, prefix, context).is_err());
        }
        std::fs::write(&path, [1, 2, 3])?;
        assert!(read_ids(&path, 1, 4).is_err());
        std::fs::remove_dir_all(dir)?;
        let mut out = Vec::new();
        write_logits(&mut out, &[0x80, 0x3f, 0x00, 0xc0])?;
        assert_eq!(out, [0x80, 0x3f, 0x00, 0xc0]);
        for bad in [&[][..], &[0][..], &[0x80, 0x7f][..], &[0xc0, 0x7f][..]] {
            assert!(write_logits(&mut out, bad).is_err());
        }
        Ok(())
    }
}
