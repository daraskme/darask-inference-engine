//! Per-op GPU time of decode steps from CUDA events recorded between ops on the engine stream.
//! Recording adds a little time of its own; use it for the breakdown, not for tok/s.

use std::collections::BTreeMap;
use tsumuji_cuda::{Event, Stream};

#[derive(Debug)]
pub struct Profiler {
    events: Vec<Event>,
    labels: Vec<&'static str>,
    used: usize,
    /// label -> (total ms, count per step summed over steps)
    pub totals: BTreeMap<&'static str, (f64, usize)>,
    pub steps: usize,
}

impl Profiler {
    pub fn new(capacity: usize) -> tsumuji_cuda::Result<Self> {
        Ok(Self {
            events: (0..capacity)
                .map(|_| Event::new())
                .collect::<tsumuji_cuda::Result<_>>()?,
            labels: vec![""; capacity],
            used: 0,
            totals: BTreeMap::new(),
            steps: 0,
        })
    }

    /// Records the end of the op `label` (the time since the previous mark is charged to it).
    pub fn mark(&mut self, stream: &Stream, label: &'static str) -> tsumuji_cuda::Result<()> {
        assert!(self.used < self.events.len(), "profiler capacity");
        self.events[self.used].record(stream)?;
        self.labels[self.used] = label;
        self.used += 1;
        Ok(())
    }

    /// Waits for the step and adds its intervals to the totals.
    pub fn end_step(&mut self) -> tsumuji_cuda::Result<()> {
        for i in 1..self.used {
            let ms = f64::from(self.events[i - 1].elapsed_ms(&self.events[i])?);
            let e = self.totals.entry(self.labels[i]).or_default();
            e.0 += ms;
            e.1 += 1;
        }
        self.used = 0;
        self.steps += 1;
        Ok(())
    }

    /// Table: label, ms per step, share, calls per step.
    pub fn report(&self) -> String {
        let total: f64 = self.totals.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = self.totals.iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let n = self.steps.max(1) as f64;
        let mut s = format!(
            "{:<10} {:>9} {:>7} {:>7}\n",
            "op", "ms/step", "share", "calls"
        );
        for (label, (ms, count)) in rows {
            s += &format!(
                "{label:<10} {:>9.3} {:>6.1}% {:>7.0}\n",
                ms / n,
                100.0 * ms / total,
                *count as f64 / n
            );
        }
        s += &format!(
            "{:<10} {:>9.3}  ({:.1} tok/s with profiling)\n",
            "total",
            total / n,
            1000.0 * n / total
        );
        s
    }
}
