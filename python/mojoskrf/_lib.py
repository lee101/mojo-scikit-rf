"""ctypes access to the Mojo network-math kernels."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOSKRF_LIB") or os.path.join(
    ROOT, "dist", "libmojo-scikit-rf.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "msrf_inverse": ([I, I, I, I, I], I),
    "msrf_s2z": ([I, I, I, I, I, I, I], I),
    "msrf_z2s": ([I, I, I, I, I, I, I], I),
    "msrf_s2t": ([I, I, I, I, I], I),
    "msrf_t2s": ([I, I, I, I, I], I),
    "msrf_s2a": ([I, I, I, I], I),
    "msrf_a2s": ([I, I, I, I], I),
    "msrf_passivity": ([I, I, I, I], None),
    "msrf_reciprocity": ([I, I, I, I], None),
    "msrf_innerconnect": ([I, I, I, I, I, I], I),
}

_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        if not os.path.exists(LIB):
            raise RuntimeError(
                f"Mojo library not found at {LIB}; run `pixi run build` first"
            )
        _loaded = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _loaded


def c128(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if not np.can_cast(source.dtype, np.dtype(np.complex128), casting="safe"):
        raise TypeError(
            f"cannot safely convert {source.dtype} to complex128"
        )
    if copy:
        return np.array(source, dtype=np.complex128, order="C", copy=True)
    return np.ascontiguousarray(source, dtype=np.complex128)


def addr(array: np.ndarray) -> int:
    if array.dtype != np.complex128 or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous complex128 arrays")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null address for a non-empty buffer")
    return address
