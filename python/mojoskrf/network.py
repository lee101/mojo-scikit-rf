"""S-parameter network math with scikit-rf-compatible function signatures."""

from __future__ import annotations

import warnings

import numpy as np

from ._lib import addr, c128, lib

S_DEFINITIONS = ("power", "pseudo", "traveling")
S_DEF_DEFAULT = "power"
ZERO = 1e-4


def fix_z0_shape(z0, nfreqs: int, nports: int) -> np.ndarray:
    if np.shape(z0) == (nfreqs, nports):
        return np.asarray(z0).copy()
    if np.ndim(z0) == 0:
        return np.full((nfreqs, nports), z0)
    if len(z0) == nports:
        return np.tile(np.asarray(z0), (nfreqs, 1))
    if len(z0) == nfreqs:
        return np.tile(np.asarray(z0)[:, None], (1, nports))
    raise IndexError("z0 is not an acceptable shape")


def _matrix(value, name: str) -> np.ndarray:
    matrix = c128(value)
    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError(f"{name} must have shape (nfreqs, nports, nports)")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one port")
    return matrix


def _z0(value, nf: int, n: int) -> np.ndarray:
    z0 = c128(fix_z0_shape(value, nf, n), copy=True)
    z0.real[z0.real == 0] += ZERO
    return z0


def _definition(s_def: str) -> int:
    try:
        return S_DEFINITIONS.index(s_def)
    except ValueError:
        raise ValueError(f"Unknown s_def: {s_def}") from None


def _inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = _matrix(matrix, "matrix")
    nf, n, _ = matrix.shape
    result = np.empty_like(matrix)
    if nf == 0:
        return result
    scratch = np.empty((n, n), dtype=np.complex128)
    ok = lib().msrf_inverse(
        addr(matrix), addr(result), addr(scratch), nf, n
    )
    if not ok:
        raise np.linalg.LinAlgError("Singular matrix")
    return result


def s2z(
    s: np.ndarray, z0=50, s_def: str = S_DEF_DEFAULT
) -> np.ndarray:
    s = _matrix(s, "s")
    nf, n, _ = s.shape
    impedance = _z0(z0, nf, n)
    result = np.empty_like(s)
    if nf == 0:
        return result
    scratch = np.empty((n, n), dtype=np.complex128)
    ok = lib().msrf_s2z(
        addr(s), addr(impedance), addr(result), addr(scratch),
        nf, n, _definition(s_def),
    )
    if not ok:
        raise np.linalg.LinAlgError("S to Z conversion is singular")
    return result


def z2s(
    z, z0=50, s_def: str = S_DEF_DEFAULT
) -> np.ndarray:
    z = _matrix(z, "z")
    nf, n, _ = z.shape
    impedance = _z0(z0, nf, n)
    result = np.empty_like(z)
    if nf == 0:
        return result
    scratch = np.empty((n, n), dtype=np.complex128)
    ok = lib().msrf_z2s(
        addr(z), addr(impedance), addr(result), addr(scratch),
        nf, n, _definition(s_def),
    )
    if not ok:
        raise np.linalg.LinAlgError("Z to S conversion is singular")
    return result


def s2y(
    s: np.ndarray, z0=50, s_def: str = S_DEF_DEFAULT
) -> np.ndarray:
    return _inverse(s2z(s, z0=z0, s_def=s_def))


def y2s(
    y, z0=50, s_def: str = S_DEF_DEFAULT
) -> np.ndarray:
    return z2s(_inverse(_matrix(y, "y")), z0=z0, s_def=s_def)


def z2y(z: np.ndarray) -> np.ndarray:
    return _inverse(_matrix(z, "z"))


def y2z(y: np.ndarray) -> np.ndarray:
    return _inverse(_matrix(y, "y"))


def s2t(s: np.ndarray) -> np.ndarray:
    s = _matrix(s, "s")
    nf, n, _ = s.shape
    if n % 2:
        raise IndexError("Network does not have an even number of ports")
    result = np.empty_like(s)
    if nf == 0:
        return result
    scratch = np.empty(3 * (n // 2) ** 2, dtype=np.complex128)
    if not lib().msrf_s2t(addr(s), addr(result), addr(scratch), nf, n):
        raise np.linalg.LinAlgError("S transfer block is singular")
    return result


def t2s(t: np.ndarray) -> np.ndarray:
    t = _matrix(t, "t")
    nf, n, _ = t.shape
    if n % 2:
        raise IndexError("Network does not have an even number of ports")
    result = np.empty_like(t)
    if nf == 0:
        return result
    scratch = np.empty(3 * (n // 2) ** 2, dtype=np.complex128)
    if not lib().msrf_t2s(addr(t), addr(result), addr(scratch), nf, n):
        raise np.linalg.LinAlgError("T transfer block is singular")
    return result


def s2a(s: np.ndarray, z0=50) -> np.ndarray:
    s = _matrix(s, "s")
    nf, n, _ = s.shape
    if n != 2:
        raise IndexError("abcd parameters are defined for 2-ports networks only")
    impedance = _z0(z0, nf, n)
    result = np.empty_like(s)
    if nf == 0:
        return result
    if not lib().msrf_s2a(addr(s), addr(impedance), addr(result), nf):
        raise RuntimeError("S to ABCD conversion failed")
    return result


def a2s(a: np.ndarray, z0=50) -> np.ndarray:
    a = _matrix(a, "a")
    nf, n, _ = a.shape
    if n != 2:
        raise IndexError("abcd parameters are defined for 2-ports networks only")
    impedance = _z0(z0, nf, n)
    result = np.empty_like(a)
    if nf == 0:
        return result
    if not lib().msrf_a2s(addr(a), addr(impedance), addr(result), nf):
        raise RuntimeError("ABCD to S conversion failed")
    return result


def renormalize_s(
    s: np.ndarray,
    z_old,
    z_new,
    s_def: str = S_DEF_DEFAULT,
    s_def_old: str | None = None,
) -> np.ndarray:
    if s_def_old not in S_DEFINITIONS and s_def_old is not None:
        raise ValueError("s_def_old parameter should be one of:", S_DEFINITIONS)
    if s_def_old is None:
        s_def_old = s_def
    _definition(s_def)
    return z2s(s2z(s, z0=z_old, s_def=s_def_old), z0=z_new, s_def=s_def)


def passivity(s: np.ndarray) -> np.ndarray:
    s = _matrix(s, "s")
    nf, n, _ = s.shape
    if n == 1:
        raise ValueError("Doesn't exist for one ports")
    result = np.empty_like(s)
    if nf == 0:
        return result
    lib().msrf_passivity(addr(s), addr(result), nf, n)
    return result


def reciprocity(s: np.ndarray) -> np.ndarray:
    s = _matrix(s, "s")
    nf, n, _ = s.shape
    if n == 1:
        raise ValueError("Doesn't exist for one ports")
    result = np.empty_like(s)
    if nf == 0:
        return result
    lib().msrf_reciprocity(addr(s), addr(result), nf, n)
    return result


def _innerconnect_lstsq(a: np.ndarray, k: int, l: int) -> np.ndarray:
    ext = [i for i in range(a.shape[1]) if i not in (k, l)]
    result = np.array(a[:, ext][:, :, ext], order="C")
    for f in range(a.shape[0]):
        aii = np.eye(2) - a[f][np.ix_([l, k], [k, l])]
        aie = a[f][np.ix_([l, k], ext)]
        aei = a[f][np.ix_(ext, [k, l])]
        result[f] += aei @ np.linalg.lstsq(aii, aie, rcond=None)[0]
    return result


def innerconnect_s(a: np.ndarray, k: int, l: int) -> np.ndarray:
    a = _matrix(a, "A")
    nf, n, _ = a.shape
    if k > n - 1 or l > n - 1 or k < 0 or l < 0:
        raise ValueError("port indices are out of range")
    if k == l:
        raise ValueError("cannot connect a port to itself")
    result = np.empty((nf, n - 2, n - 2), dtype=np.complex128)
    if nf == 0 or result.size == 0:
        return result
    ok = lib().msrf_innerconnect(addr(a), addr(result), nf, n, k, l)
    if not ok:
        warnings.warn(
            "Singular matrix detected, using numpy.linalg.lstsq instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _innerconnect_lstsq(a, k, l)
    return result


def connect_s(
    A: np.ndarray, k: int, B: np.ndarray, l: int, num: int = 1
) -> np.ndarray:
    A = _matrix(A, "A")
    B = _matrix(B, "B")
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of frequency points")
    if k > A.shape[-1] - 1 or l > B.shape[-1] - 1 or k < 0 or l < 0:
        raise ValueError("port indices are out of range")
    nf, n_a, _ = A.shape
    n_b = B.shape[1]
    composite = np.zeros((nf, n_a + n_b, n_a + n_b), dtype=np.complex128)
    if n_b == 2 and n_a > 2 and num == 1:
        composite[:, :k, :k] = A[:, :k, :k]
        composite[:, :k, k + n_b :] = A[:, :k, k:]
        composite[:, k + n_b :, :k] = A[:, k:, :k]
        composite[:, k + n_b :, k + n_b :] = A[:, k:, k:]
        composite[:, k : k + n_b, k : k + n_b] = B
        return innerconnect_s(composite, k + n_b, k + l)
    composite[:, :n_a, :n_a] = A
    composite[:, n_a:, n_a:] = B
    return innerconnect_s(composite, k, n_a + l)


def inv(s: np.ndarray) -> np.ndarray:
    return t2s(_inverse(s2t(s)))
