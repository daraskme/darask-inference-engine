use std::fmt;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    /// A non-persistent CUDA error (launch config, invalid args, allocation).
    /// `code` is the raw `cudaError_t`.
    Cuda {
        code: i32,
        call: &'static str,
    },
    /// A persistent CUDA error, classified by code even at submission time
    /// (`sync_point` names where it was observed). Illegal address or device-side
    /// assert are sticky: the CUDA context is dead, so the process stops
    /// submitting work and exits; the supervisor restarts it.
    CudaAsync {
        code: i32,
        sync_point: &'static str,
    },
    /// An allocation did not fit in the planned arena.
    OutOfMemory {
        buffer: &'static str,
        requested: u64,
        available: u64,
    },
    /// A shape is out of range or does not fit its storage format.
    InvalidShape(String),
    /// A memory plan request is malformed (inverted lifetime, size overflow).
    InvalidPlan(String),
    /// The machine does not match what Hayate was built for.
    UnsupportedMachine(String),
    /// A weight file is malformed or does not match the model definition.
    Weights(String),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cuda { code, call } => write!(f, "CUDA error {code} in {call}"),
            Error::CudaAsync { code, sync_point } => {
                write!(f, "CUDA async error {code} observed at {sync_point}")
            }
            Error::OutOfMemory {
                buffer,
                requested,
                available,
            } => write!(
                f,
                "{buffer}: requested {requested} bytes, {available} bytes left in arena"
            ),
            Error::InvalidShape(msg) => write!(f, "invalid shape: {msg}"),
            Error::InvalidPlan(msg) => write!(f, "invalid memory plan: {msg}"),
            Error::UnsupportedMachine(msg) => write!(f, "unsupported machine: {msg}"),
            Error::Weights(msg) => write!(f, "weights: {msg}"),
            Error::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
