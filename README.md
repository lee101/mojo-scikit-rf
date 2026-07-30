# mojo-scikit-rf

`mojo-scikit-rf` is a standalone Mojo port of the compute-heavy network math in
[scikit-rf](https://scikit-rf.org/). It operates on the same batched
`complex128` arrays and exposes scikit-rf's function names and signatures for
the covered subset. It does not require scikit-rf at runtime; scikit-rf is a
development dependency used for parity tests and benchmarks.

## Coverage

The port currently covers:

- S/Z/Y conversion: `s2z`, `z2s`, `s2y`, `y2s`, `z2y`, and `y2z`
- power, pseudo, and traveling wave definitions, including complex,
  frequency-dependent per-port impedances
- balanced 2N-port scattering-transfer conversion: `s2t` and `t2s`
- two-port ABCD conversion: `s2a` and `a2s`
- impedance renormalization and de-embedding inverse: `renormalize_s` and `inv`
- multiport metrics: `passivity` and `reciprocity`
- network connection math: `innerconnect_s` and `connect_s`
- scikit-rf-compatible `fix_z0_shape`

The port does not include scikit-rf's `Network` class, Touchstone I/O,
calibration, media models, frequency interpolation, noise analysis, vector
fitting, plotting, or instrument support. Parameter conversions that upstream
scikit-rf itself leaves unimplemented (`z2t`, `t2z`, `y2t`, and `t2y`) are also
out of scope.

## Install and build

Install the pinned Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-scikit-rf.so`. Run the parity suite with:

```bash
pixi run test
```

## Usage

The API accepts arrays shaped `(nfreqs, nports, nports)`, matching scikit-rf:

```python
import numpy as np
import mojoskrf as rf

s = np.array(
    [
        [[0.10 + 0.02j, 0.80 - 0.05j],
         [0.79 - 0.04j, 0.08 + 0.01j]],
        [[0.12 + 0.03j, 0.76 - 0.08j],
         [0.75 - 0.07j, 0.09 + 0.02j]],
    ],
    dtype=np.complex128,
)

z = rf.s2z(s, z0=[50, 75], s_def="power")
s_at_60_ohms = rf.renormalize_s(s, z_old=[50, 75], z_new=60)
assert np.allclose(rf.z2s(z, z0=[50, 75]), s)
```

Save this as a script and run it with `pixi run python script.py`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, Python 3.13.14, NumPy 2.5.1, and scikit-rf 2.0.1.
Times are the best of three measured runs after a warm-up, on identical
`complex128` inputs. Speedup is scikit-rf time divided by mojo-scikit-rf time.

| Kernel | Mojo | scikit-rf | Speedup |
|---|---:|---:|---:|
| `s2z`, 300k x 2-port | 42.42 ms | 2205.91 ms | 52.01x |
| `s2t`, 10k x 2-port | 0.56 ms | 162.57 ms | 288.38x |
| `s2a`, 300k x 2-port | 20.69 ms | 272.30 ms | 13.16x |
| `s2z`, 20k x 4-port | 10.15 ms | 408.45 ms | 40.23x |
| `passivity`, 10k x 4-port | 2.32 ms | 49.28 ms | 21.27x |
| `innerconnect_s`, 20k x 6-port | 3.39 ms | 8.31 ms | 2.45x |
| `innerconnect_s`, 80k x 6-port | 18.20 ms | 97.62 ms | 5.36x |
| `innerconnect_s`, 150k x 6-port | 39.47 ms | 269.58 ms | 6.83x |

These results are specific to the machine and versions above. In particular,
scikit-rf 2.0.1 performs some transforms frequency-by-frequency in Python;
that overhead is reflected in the `s2t` and `passivity` rows.

No GPU path is included. These kernels operate directly on CPU-resident NumPy
buffers; this project has not benchmarked or validated a GPU implementation.

## How it works

All numerical kernels are compiled together from `src/kernels.mojo` into one
shared library. The Python layer validates and broadcasts inputs like
scikit-rf, makes them C-contiguous when needed, and calls the library through
`ctypes`. Buffers cross the C ABI as integer addresses.

NumPy `complex128` already uses the memory layout expected by the kernels:
interleaved 64-bit real and imaginary values in row-major order. Mojo performs
batched complex matrix conversion, pivoted Gauss-Jordan solves, block transfer
transforms, and connection updates directly in caller-owned output and scratch
buffers. The connection update uses SIMD for contiguous complex row segments,
including a scalar remainder, and splits batches of at least 131,072 frequency
points between two CPU workers. Smaller batches stay serial to avoid thread
launch overhead. No allocation or ownership crosses the FFI boundary. A
singular two-port connection uses the same NumPy least-squares fallback as
scikit-rf.
