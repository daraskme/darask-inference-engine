//! Native engine adapter for the shared OpenAI frontend. No subprocess or foreign backend.
use crate::engine::Engine;
use darask_api::{
    generate::{Flow, GenRequest, Generator, Stats},
    sample::Sampler,
};
use std::{path::Path, time::Instant};
pub struct NativeGen {
    eng: Engine,
}
impl NativeGen {
    pub fn open(path: &Path, ctx: usize) -> Result<Self, String> {
        Ok(Self {
            eng: Engine::load(path, ctx)?,
        })
    }
}
impl Generator for NativeGen {
    fn ctx(&self) -> usize {
        self.eng.context
    }
    fn generate(
        &mut self,
        req: GenRequest<'_>,
        emit: &mut dyn FnMut(u32) -> Flow,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Stats, String> {
        if req.variant != 0 {
            return Err("native Ornith has no runtime variants".into());
        }
        if req.prompt.is_empty() || req.prompt.len() + req.max_new > self.eng.context {
            return Err("prompt/context limit".into());
        }
        self.eng.reset()?;
        let start = Instant::now();
        let mut next = 0;
        for &id in req.prompt {
            if cancelled() {
                return Ok(Stats::default());
            }
            next = self.eng.step(id)?;
        }
        let mut stats = Stats {
            prefill_s: start.elapsed().as_secs_f64(),
            sampling: if req.params.greedy() {
                "greedy"
            } else {
                "host"
            },
            host_sampling: !req.params.greedy(),
            ..Default::default()
        };
        let start = Instant::now();
        let mut sampler = Sampler::new(req.params, req.prompt);
        for i in 0..req.max_new {
            if cancelled() {
                break;
            }
            if !req.params.greedy() {
                next = sampler.sample(&mut self.eng.logits()?);
            }
            sampler.accept(next);
            stats.generated += 1;
            if emit(next) == Flow::Stop {
                break;
            }
            if i + 1 < req.max_new {
                next = self.eng.step(next)?;
            }
        }
        stats.decode_s = start.elapsed().as_secs_f64();
        Ok(stats)
    }
}
