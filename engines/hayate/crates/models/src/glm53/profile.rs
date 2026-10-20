use hayate_core::Result;
use hayate_kernels::runtime::{DeviceBuffer, Event, Stream};
use std::{cell::RefCell, time::Instant};

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Waits {
    pub d2h_count: u64,
    pub d2h_sync_count: u64,
    pub d2h_ms: f64,
}
#[derive(Debug)]
pub(super) struct Profile {
    events: Vec<Event>,
    pub waits: RefCell<Waits>,
}
impl Profile {
    pub fn new(stream: &Stream, count: usize) -> Result<Self> {
        Ok(Self {
            events: (0..count).map(|_| stream.event()).collect::<Result<_>>()?,
            waits: RefCell::new(Waits::default()),
        })
    }
    pub fn mark(&self, index: usize, stream: &Stream) -> Result<()> {
        self.events[index].record(stream)
    }
    pub fn ms(&self, from: usize, to: usize) -> Result<f32> {
        self.events[from].elapsed_ms(&self.events[to])
    }
    pub fn reset(&self) {
        *self.waits.borrow_mut() = Waits::default();
    }
    pub fn download(&self, stream: &Stream, dst: &mut [u8], src: &DeviceBuffer) -> Result<()> {
        let start = Instant::now();
        let result = stream.copy_d2h(dst, src);
        self.add_wait(start);
        result
    }
    pub fn enqueue_d2h(&self) {
        self.waits.borrow_mut().d2h_count += 1;
    }
    pub fn add_wait(&self, start: Instant) {
        let mut waits = self.waits.borrow_mut();
        waits.d2h_count += 1;
        waits.d2h_sync_count += 1;
        waits.d2h_ms += start.elapsed().as_secs_f64() * 1000.;
    }
}
