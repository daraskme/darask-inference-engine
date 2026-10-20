//! Static memory planning.
//!
//! Each pipeline declares its buffers up front; the planner assigns offsets
//! inside the GPU's single arena before the first kernel runs. Nothing calls
//! `cudaMalloc` on the hot path, which is also what lets whole denoising
//! loops be captured into a single CUDA Graph.
//!
//! Reuse rule (ARCHITECTURE.md §5.2): stage indices are *stream-ordered*.
//! Two buffers may alias only if their lifetimes are disjoint **and** they
//! are used on the same stream, so the later writer is ordered after the
//! earlier reader by the stream itself. Buffers on different streams never
//! alias — cross-stream reuse would need event edges the planner cannot see.

use crate::{Error, Result};

/// When a buffer is live, in stage indices on its stream (inclusive).
#[derive(Clone, Copy, Debug)]
pub struct Lifetime {
    pub first: u32,
    pub last: u32,
}

impl Lifetime {
    fn overlaps(&self, other: &Lifetime) -> bool {
        self.first <= other.last && other.first <= self.last
    }
}

/// Logical stream a buffer is produced and consumed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamId(pub u8);

#[derive(Clone, Debug)]
pub struct BufferRequest {
    pub name: &'static str,
    pub bytes: u64,
    pub stream: StreamId,
    pub lifetime: Lifetime,
}

/// Every buffer starts on a 256-byte boundary (TMA / vectorized loads).
pub const ALIGN: u64 = 256;

#[derive(Clone, Debug)]
pub struct Plan {
    /// One offset per request, in request order.
    pub offsets: Vec<u64>,
    pub peak: u64,
}

/// Greedy first-fit by descending size.
pub fn plan(requests: &[BufferRequest], capacity: u64) -> Result<Plan> {
    let mut sizes = Vec::with_capacity(requests.len());
    for r in requests {
        if r.lifetime.first > r.lifetime.last {
            return Err(Error::InvalidPlan(format!(
                "{}: lifetime ends before it starts",
                r.name
            )));
        }
        let size = r
            .bytes
            .checked_next_multiple_of(ALIGN)
            .ok_or_else(|| Error::InvalidPlan(format!("{}: size overflows", r.name)))?;
        sizes.push(size);
    }

    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]));

    let mut placed: Vec<(usize, u64)> = Vec::with_capacity(requests.len());
    let mut offsets = vec![0; requests.len()];
    let mut peak = 0;

    for i in order {
        let req = &requests[i];
        let size = sizes[i];
        let conflicts = |j: usize| {
            requests[j].stream != req.stream || requests[j].lifetime.overlaps(&req.lifetime)
        };
        let mut busy: Vec<(u64, u64)> = placed
            .iter()
            .filter(|&&(j, _)| conflicts(j))
            .map(|&(j, off)| (off, off + sizes[j]))
            .collect();
        busy.sort_unstable();

        let mut offset = 0u64;
        for (start, end) in busy {
            if offset.checked_add(size).is_some_and(|e| e <= start) {
                break;
            }
            offset = offset.max(end);
        }
        let end =
            offset
                .checked_add(size)
                .filter(|&e| e <= capacity)
                .ok_or(Error::OutOfMemory {
                    buffer: req.name,
                    requested: size,
                    available: capacity.saturating_sub(offset),
                })?;
        peak = peak.max(end);
        placed.push((i, offset));
        offsets[i] = offset;
    }
    Ok(Plan { offsets, peak })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(name: &'static str, bytes: u64, stream: u8, first: u32, last: u32) -> BufferRequest {
        BufferRequest {
            name,
            bytes,
            stream: StreamId(stream),
            lifetime: Lifetime { first, last },
        }
    }

    #[test]
    fn disjoint_lifetimes_share_memory() {
        let p = plan(&[req("a", 1024, 0, 0, 1), req("b", 1024, 0, 2, 3)], 4096).unwrap();
        assert_eq!(p.peak, 1024);
    }

    #[test]
    fn touching_lifetimes_do_not_share() {
        let p = plan(&[req("a", 1024, 0, 0, 1), req("b", 1024, 0, 1, 2)], 4096).unwrap();
        assert_eq!(p.peak, 2048);
    }

    #[test]
    fn different_streams_never_share() {
        let p = plan(&[req("a", 1024, 0, 0, 1), req("b", 1024, 1, 2, 3)], 4096).unwrap();
        assert_eq!(p.peak, 2048);
        assert_ne!(p.offsets[0], p.offsets[1]);
    }

    #[test]
    fn alignment_is_applied() {
        let p = plan(&[req("a", 1, 0, 0, 0), req("b", 1, 0, 0, 0)], 4096).unwrap();
        assert_eq!(p.peak, 2 * ALIGN);
    }

    #[test]
    fn reports_oom() {
        assert!(plan(&[req("a", 8192, 0, 0, 0)], 4096).is_err());
    }

    #[test]
    fn huge_sizes_do_not_wrap() {
        assert!(plan(&[req("a", u64::MAX, 0, 0, 0)], 0).is_err());
        // Rounds up past u64::MAX.
        assert!(plan(&[req("a", u64::MAX - 10, 0, 0, 0)], u64::MAX).is_err());
        // Two buffers whose end offset would overflow.
        let big = u64::MAX / 2 + 1;
        assert!(plan(&[req("a", big, 0, 0, 0), req("b", big, 0, 0, 0)], u64::MAX).is_err());
    }

    #[test]
    fn inverted_lifetime_is_rejected() {
        assert!(plan(&[req("a", 1, 0, 3, 1)], 4096).is_err());
    }
}
