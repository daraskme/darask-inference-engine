//! Hayate core: the pieces every other crate shares.
//!
//! Hayate targets exactly two machines (see `docs/ARCHITECTURE.md` §1):
//! the main PC (RTX PRO 6000 Blackwell 96 GB, i9-12900KS) and the sub PC
//! `ai-workstation` (RTX 5060 Ti 16 GB), both `sm_120a`. There is no
//! portability layer: anything that would only exist for other hardware is
//! deliberately left out.

pub mod dtype;
pub mod error;
pub mod hyt;
pub mod io;
pub mod machine;
pub mod memory;
pub mod tensor;

pub use dtype::{DType, QuantSpec};
pub use error::{Error, Result};
pub use hyt::{HytFile, TensorEntry};
pub use machine::{CpuCores, CpuIsa, GpuSpec, Machine, MachineId};
pub use memory::{BufferRequest, Lifetime, Plan, StreamId};
pub use tensor::{Shape, TensorDesc};
