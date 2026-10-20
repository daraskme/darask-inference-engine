//! Bounded FIFO shared by HTTP producers and the single GPU worker.
use super::{Job, gpu};
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

#[derive(Debug)]
struct State {
    jobs: VecDeque<Job>,
    senders: usize,
    open: bool,
}
#[derive(Debug)]
struct Queue {
    state: Mutex<State>,
    models: Mutex<Option<String>>,
    gpu: Mutex<Option<gpu::Snapshot>>,
    changed: Condvar,
    capacity: usize,
}
#[derive(Debug)]
pub struct Sender(Arc<Queue>);
#[derive(Debug)]
pub struct Receiver(Arc<Queue>);

pub fn channel(capacity: usize) -> (Sender, Receiver) {
    assert!(capacity > 0);
    let queue = Arc::new(Queue {
        state: Mutex::new(State {
            jobs: VecDeque::new(),
            senders: 1,
            open: true,
        }),
        changed: Condvar::new(),
        models: Mutex::new(None),
        gpu: Mutex::new(None),
        capacity,
    });
    (Sender(queue.clone()), Receiver(queue))
}
impl Clone for Sender {
    fn clone(&self) -> Self {
        self.0.state.lock().unwrap().senders += 1;
        Self(self.0.clone())
    }
}
impl Drop for Sender {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().senders -= 1;
        self.0.changed.notify_all();
    }
}
impl Sender {
    pub(crate) fn models(&self) -> Option<String> {
        self.0.models.lock().unwrap().clone()
    }
    pub(crate) fn gpu(&self) -> Option<gpu::Snapshot> {
        self.0.gpu.lock().unwrap().clone()
    }
    pub fn send(&self, job: Job) -> Result<(), Box<Job>> {
        let mut state = self.0.state.lock().unwrap();
        while state.open && state.jobs.len() == self.0.capacity {
            state = self.0.changed.wait(state).unwrap();
        }
        if !state.open {
            return Err(Box::new(job));
        }
        state.jobs.push_back(job);
        self.0.changed.notify_all();
        Ok(())
    }
    pub fn try_send(&self, job: Job) -> Result<(), Box<Job>> {
        let mut state = self.0.state.lock().unwrap();
        if !state.open || state.jobs.len() == self.0.capacity {
            return Err(Box::new(job));
        }
        state.jobs.push_back(job);
        self.0.changed.notify_all();
        Ok(())
    }
}
pub(crate) enum Next {
    Job(Box<Job>),
    Idle,
    Closed,
}
impl Receiver {
    pub(crate) fn publish_models(&self, models: String) {
        *self.0.models.lock().unwrap() = Some(models);
    }
    pub(crate) fn publish_gpu(&self, snapshot: gpu::Snapshot) {
        *self.0.gpu.lock().unwrap() = Some(snapshot);
    }
    /// Next job; `Idle` once `idle` passes without one.
    pub(crate) fn wait(&mut self, idle: Option<Duration>) -> Next {
        let deadline = idle.map(|d| Instant::now() + d);
        let mut state = self.0.state.lock().unwrap();
        loop {
            if let Some(job) = state.jobs.pop_front() {
                self.0.changed.notify_all();
                return Next::Job(Box::new(job));
            }
            if state.senders == 0 {
                return Next::Closed;
            }
            state = match deadline {
                None => self.0.changed.wait(state).unwrap(),
                Some(deadline) => {
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Next::Idle;
                    }
                    self.0.changed.wait_timeout(state, left).unwrap().0
                }
            };
        }
    }
}
impl Iterator for Receiver {
    type Item = Job;
    fn next(&mut self) -> Option<Job> {
        match self.wait(None) {
            Next::Job(job) => Some(*job),
            Next::Idle | Next::Closed => None,
        }
    }
}
impl Drop for Receiver {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap();
        state.open = false;
        state.jobs.clear();
        self.0.changed.notify_all();
    }
}
