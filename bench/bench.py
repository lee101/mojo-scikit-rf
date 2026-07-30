"""Benchmarks against scikit-rf on identical complex128 arrays."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
import skrf
import skrf.network as reference

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojoskrf as rf  # noqa: E402


def best_time(function, repeats=3):
    function()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def measure(name, mojo_fn, reference_fn, rows):
    mojo_value = mojo_fn()
    reference_value = reference_fn()
    np.testing.assert_allclose(mojo_value, reference_value, rtol=2e-10, atol=2e-10)
    mojo_s = best_time(mojo_fn)
    reference_s = best_time(reference_fn)
    rows.append((name, mojo_s, reference_s, reference_s / mojo_s))
    print(f"measured: {name}", flush=True)


def main():
    rng = np.random.default_rng(7)
    rows = []

    s2 = 0.1 * (
        rng.normal(size=(300_000, 2, 2))
        + 1j * rng.normal(size=(300_000, 2, 2))
    )
    measure(
        "s2z, 300k x 2-port",
        lambda: rf.s2z(s2),
        lambda: reference.s2z(s2),
        rows,
    )
    measure(
        "s2t, 10k x 2-port",
        lambda: rf.s2t(s2[:10_000]),
        lambda: reference.s2t(s2[:10_000]),
        rows,
    )
    measure(
        "s2a, 300k x 2-port",
        lambda: rf.s2a(s2),
        lambda: reference.s2a(s2),
        rows,
    )

    s4 = 0.07 * (
        rng.normal(size=(80_000, 4, 4))
        + 1j * rng.normal(size=(80_000, 4, 4))
    )
    measure(
        "s2z, 20k x 4-port",
        lambda: rf.s2z(s4[:20_000]),
        lambda: reference.s2z(s4[:20_000]),
        rows,
    )
    measure(
        "passivity, 10k x 4-port",
        lambda: rf.passivity(s4[:10_000]),
        lambda: reference.passivity(s4[:10_000]),
        rows,
    )

    del s2, s4
    s6 = 0.05 * (
        rng.normal(size=(150_000, 6, 6))
        + 1j * rng.normal(size=(150_000, 6, 6))
    )
    measure(
        "innerconnect_s, 20k x 6-port",
        lambda: rf.innerconnect_s(s6[:20_000], 1, 4),
        lambda: reference.innerconnect_s(s6[:20_000], 1, 4),
        rows,
    )
    measure(
        "innerconnect_s, 80k x 6-port",
        lambda: rf.innerconnect_s(s6[:80_000], 1, 4),
        lambda: reference.innerconnect_s(s6[:80_000], 1, 4),
        rows,
    )
    measure(
        "innerconnect_s, 150k x 6-port",
        lambda: rf.innerconnect_s(s6, 1, 4),
        lambda: reference.innerconnect_s(s6, 1, 4),
        rows,
    )

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print(
        f"Python {platform.python_version()}; NumPy {np.__version__}; "
        f"scikit-rf {skrf.__version__}"
    )
    print()
    print("| Kernel | Mojo | scikit-rf | Speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo_s, reference_s, speedup in rows:
        print(
            f"| {name} | {mojo_s * 1e3:.2f} ms | "
            f"{reference_s * 1e3:.2f} ms | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
