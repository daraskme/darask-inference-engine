use hayate_core::{Error, Result};

pub(super) const SPARSE_WIDTH: usize = 2051;

/// Chronological union and per-query indices; reject future/duplicate cache rows.
pub(super) fn sparse_union(
    selected: &[u8],
    start: usize,
    rows: usize,
) -> Result<(Vec<usize>, Vec<Vec<u32>>)> {
    if !(1..=4).contains(&rows) || selected.len() != rows * SPARSE_WIDTH * 4 {
        return Err(Error::InvalidShape(
            "GLM sparse verification selections".into(),
        ));
    }
    let mut queries = Vec::with_capacity(rows);
    for (row, raw) in selected
        .as_chunks::<{ SPARSE_WIDTH * 4 }>()
        .0
        .iter()
        .enumerate()
    {
        let mut ids = Vec::new();
        for bytes in raw.as_chunks::<4>().0 {
            let id = i32::from_le_bytes(*bytes);
            if id == -1 {
                continue;
            }
            if id < 0 || id as usize > start + row {
                return Err(Error::Weights("GLM sparse future/invalid selection".into()));
            }
            ids.push(id as usize);
        }
        ids.sort_unstable();
        if ids.is_empty() || ids.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::Weights(
                "GLM sparse empty/duplicate selection".into(),
            ));
        }
        queries.push(ids);
    }
    let mut union: Vec<_> = queries.iter().flatten().copied().collect();
    union.sort_unstable();
    union.dedup();
    let indices = queries
        .iter()
        .map(|ids| {
            ids.iter()
                .map(|id| union.binary_search(id).unwrap() as u32)
                .collect()
        })
        .collect();
    Ok((union, indices))
}

#[cfg(test)]
fn accepted_prefix(drafts: &[u32], predictions: &[u32], cap: usize) -> Result<usize> {
    if !(1..=3).contains(&drafts.len())
        || predictions.len() != drafts.len() + 1
        || cap > drafts.len()
    {
        return Err(Error::InvalidShape("GLM speculative predictions".into()));
    }
    Ok(drafts
        .iter()
        .zip(predictions)
        .take(cap)
        .take_while(|(a, b)| a == b)
        .count())
}

/// Compare complete steps, including rejection repair, against measured decode.
#[derive(Debug, Default)]
pub(super) struct Metrics {
    pub steps: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub emitted: usize,
    pub seconds: f64,
    pub disabled: bool,
    pub baseline_seconds: f64,
    pub baseline_tokens: usize,
}
impl Metrics {
    pub fn record(
        &mut self,
        proposed: usize,
        accepted: usize,
        emitted: usize,
        seconds: f64,
    ) -> Result<()> {
        if !(1..=3).contains(&proposed)
            || accepted > proposed
            || emitted == 0
            || emitted > accepted + 1
            || !seconds.is_finite()
            || seconds <= 0.
        {
            return Err(Error::InvalidShape("GLM speculation measurement".into()));
        }
        self.steps += 1;
        self.proposed += proposed;
        self.accepted += accepted;
        self.emitted += emitted;
        self.seconds += seconds;
        Ok(())
    }
    pub fn should_disable(&self) -> bool {
        self.steps >= 8
            && self.baseline_tokens >= 8
            && self.seconds / self.emitted as f64
                >= self.baseline_seconds / self.baseline_tokens as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn selections(queries: &[Vec<i32>]) -> Vec<u8> {
        queries
            .iter()
            .flat_map(|ids| {
                ids.iter()
                    .copied()
                    .chain(std::iter::repeat(-1))
                    .take(SPARSE_WIDTH)
                    .flat_map(i32::to_le_bytes)
            })
            .collect()
    }
    #[test]
    fn glm53_sparse_union_keeps_causal_order_across_pool_boundaries() -> Result<()> {
        for start in [0, 1, 2, 3, 4, 2047, 2048, 2049, 8191] {
            for rows in 1..=4 {
                let queries: Vec<Vec<i32>> = (0..rows)
                    .map(|row| {
                        let length = start + row + 1;
                        let pools = length / 4;
                        // Query-dependent pool subsets, deliberately out of order.
                        let mut ids = Vec::new();
                        for p in (0..pools).rev().filter(|p| (p + row) % 3 != 0).take(512) {
                            ids.extend((p * 4..p * 4 + 4).map(|id| id as i32));
                        }
                        ids.extend((pools * 4..length).map(|id| id as i32));
                        if ids.is_empty() {
                            ids.push(0);
                        }
                        ids
                    })
                    .collect();
                let (union, indices) = sparse_union(&selections(&queries), start, rows)?;
                assert!(union.windows(2).all(|w| w[0] < w[1]));
                assert!(union.len() <= (start + rows).min(rows * SPARSE_WIDTH));
                for (row, query) in queries.iter().enumerate() {
                    let mut expected = query.clone();
                    expected.sort_unstable();
                    let restored: Vec<_> = indices[row]
                        .iter()
                        .map(|&i| union[i as usize] as i32)
                        .collect();
                    assert_eq!(restored, expected);
                    assert!(restored.iter().all(|&id| id as usize <= start + row));
                }
                if start == 8191 && rows > 1 {
                    assert!(union.len() > SPARSE_WIDTH);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn glm53_sparse_union_rejects_corrupt_or_future_ids() {
        for queries in [
            vec![vec![]],
            vec![vec![0, 0]],
            vec![vec![-2]],
            vec![vec![1]],
            vec![vec![1], vec![0, 1]],
        ] {
            assert!(sparse_union(&selections(&queries), 0, queries.len()).is_err());
        }
        assert!(sparse_union(&[], 0, 0).is_err());
        assert!(sparse_union(&[], 0, 5).is_err());
        assert!(sparse_union(&[0; 4], 0, 1).is_err());
    }
    #[test]
    fn glm53_fixed_length_cache_repair_preserves_every_accepted_prefix() {
        for start in 0..8 {
            for n in 1..=3 {
                for accepted in 0..=n {
                    let mut expected = [11, 13, 17, 19];
                    let mut actual = expected;
                    let mut expected_pools = Vec::new();
                    let mut actual_pools = Vec::new();
                    let mut latents = Vec::new();
                    for row in 0..=accepted {
                        let position = start + row;
                        expected[position % 4] = 31 + row as u32;
                        if position % 4 == 3 {
                            expected_pools.push(expected);
                        }
                    }
                    for row in 0..=n {
                        let position = start + row;
                        let value = 31 + row as u32;
                        latents.push(value); // Rejected future rows are invisible.
                        if row <= accepted {
                            actual[position % 4] = value;
                        }
                        if position % 4 == 3 {
                            actual_pools.push(actual);
                        }
                    }
                    assert_eq!(actual, expected);
                    assert_eq!(&actual_pools[..expected_pools.len()], expected_pools);
                    assert_eq!(
                        &latents[..=accepted],
                        &(31..=31 + accepted as u32).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    #[derive(Clone, Debug, PartialEq)]
    struct State {
        kda: f32,
        conv: [u32; 3],
        kv: Vec<u32>,
        pool_tail: [u32; 4],
    }
    impl State {
        fn forward(&mut self, id: u32) -> u32 {
            self.kda = self.kda * 0.75 + id as f32 / 32.;
            self.conv.rotate_left(1);
            self.conv[2] = id;
            self.pool_tail[self.kv.len() % 4] = id;
            self.kv.push(id);
            (self.kda.to_bits().wrapping_add(self.conv[0]) ^ self.pool_tail.iter().sum::<u32>())
                % 31
        }
    }
    #[test]
    fn glm53_speculation_all_prefixes_match_greedy_state_and_pool_boundaries() -> Result<()> {
        for n in 1..=3 {
            for wanted in 0..=n {
                for prefix in 1..=8 {
                    let mut normal = State {
                        kda: 0.,
                        conv: [0; 3],
                        kv: vec![],
                        pool_tail: [0; 4],
                    };
                    let mut anchor = 7;
                    for _ in 0..prefix {
                        anchor = normal.forward(anchor);
                    }
                    let mut spec = normal.clone();
                    let mut output = vec![anchor];
                    let mut reference = output.clone();
                    while output.len() < 160 {
                        let mut oracle = normal.clone();
                        let mut token = anchor;
                        let mut drafts = vec![];
                        for i in 0..n {
                            token = oracle.forward(token);
                            drafts.push(if i == wanted { (token + 1) % 31 } else { token });
                        }
                        let mut snapshots = vec![spec.clone()];
                        let mut predictions = vec![];
                        for id in std::iter::once(anchor).chain(drafts.iter().copied()) {
                            predictions.push(spec.forward(id));
                            snapshots.push(spec.clone());
                        }
                        let a = accepted_prefix(&drafts, &predictions, n)?;
                        assert_eq!(a, wanted);
                        spec = snapshots[a + 1].clone();
                        for &prediction in predictions.iter().take(a + 1) {
                            assert_eq!(normal.forward(*reference.last().unwrap()), prediction);
                            reference.push(prediction);
                        }
                        output.extend_from_slice(&drafts[..a]);
                        output.push(predictions[a]);
                        anchor = predictions[a];
                        assert_eq!(output, reference);
                        assert_eq!(spec, normal);
                    }
                }
            }
        }
        assert!(accepted_prefix(&[], &[0], 0).is_err());
        assert!(accepted_prefix(&[1], &[1], 1).is_err());
        Ok(())
    }
    #[test]
    fn glm53_adaptive_includes_repair_and_rejects_invalid_metrics() -> Result<()> {
        for faster in [true, false] {
            let mut m = Metrics {
                baseline_tokens: 8,
                baseline_seconds: 0.8,
                ..Metrics::default()
            };
            for _ in 0..8 {
                m.record(3, 2, 3, if faster { 0.2 } else { 0.4 })?;
            }
            assert_eq!(m.should_disable(), !faster);
            assert!(!m.disabled);
        }
        assert!(Metrics::default().record(3, 4, 1, 1.).is_err());
        assert!(Metrics::default().record(3, 0, 1, f64::NAN).is_err());
        Ok(())
    }
}
