"""Offline Hayate tools (convert, quantize, reference pipelines)."""

import importlib.util


def _fix_nixpkgs_triton():
    # nixpkgs patches triton's libcuda_dirs() to return the libcuda.so *file*
    # as a str; callers iterate it as a list of directories, so every
    # character becomes a `-L` flag and Triton's JIT build fails. Return the
    # directory list it is supposed to.
    if importlib.util.find_spec("triton") is None:
        return
    import triton.backends.nvidia.driver as driver

    if isinstance(driver.libcuda_dirs(), str):
        driver.libcuda_dirs = lambda: ["/run/opengl-driver/lib"]


_fix_nixpkgs_triton()
