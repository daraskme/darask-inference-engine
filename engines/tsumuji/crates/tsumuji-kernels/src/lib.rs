//! sm_120a cubins built from `kernels/`, one constant per source file (`<DIR>_<FILE>`).
//! Kernels are `extern "C"`, so their names are the plain C identifiers.

include!(concat!(env!("OUT_DIR"), "/cubins.rs"));
