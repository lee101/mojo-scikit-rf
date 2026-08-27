"""Batched complex128 network-parameter kernels.

Complex buffers use NumPy's interleaved [real, imag] layout. Callers own every
buffer, including scratch space.
"""

from std.math import sqrt
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of as simdwidthof
from max.algorithm import parallelize

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


@fieldwise_init
struct C(Copyable, ImplicitlyCopyable):
    var re: Float64
    var im: Float64


def load(p: Ptr, i: Int) -> C:
    return C(p[2 * i], p[2 * i + 1])


def store(p: Ptr, i: Int, z: C):
    p[2 * i] = z.re
    p[2 * i + 1] = z.im


def add(a: C, b: C) -> C:
    return C(a.re + b.re, a.im + b.im)


def sub(a: C, b: C) -> C:
    return C(a.re - b.re, a.im - b.im)


def neg(a: C) -> C:
    return C(-a.re, -a.im)


def conj(a: C) -> C:
    return C(a.re, -a.im)


def mul(a: C, b: C) -> C:
    return C(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)


def scale(a: C, x: Float64) -> C:
    return C(a.re * x, a.im * x)


def div(a: C, b: C) -> C:
    var d = b.re * b.re + b.im * b.im
    return C((a.re * b.re + a.im * b.im) / d,
             (a.im * b.re - a.re * b.im) / d)


def abs2(a: C) -> Float64:
    return a.re * a.re + a.im * a.im


def innerconnect_segment(
    a: Ptr, dst: Ptr, b: Int, ob: Int, n: Int, m: Int, i: Int,
    oi: Int, j_start: Int, oj_start: Int, count: Int, pk: Int, pl: Int,
    qpk: C, qpl: C,
):
    comptime W = simdwidthof[DType.float64]()
    comptime CW = W // 2
    var j = 0
    while j + CW <= count:
        var va = a.load[width=W](2 * (b + i * n + j_start + j))
        var vpk = a.load[width=W](2 * (b + pk * n + j_start + j))
        var vpl = a.load[width=W](2 * (b + pl * n + j_start + j))
        var are, aim = va.deinterleave()
        var pkr, pki = vpk.deinterleave()
        var plr, pli = vpl.deinterleave()
        var vre = are + pkr * qpk.re - pki * qpk.im
        vre = vre + plr * qpl.re - pli * qpl.im
        var vim = aim + pkr * qpk.im + pki * qpk.re
        vim = vim + plr * qpl.im + pli * qpl.re
        dst.store(2 * (ob + oi * m + oj_start + j), vre.interleave(vim))
        j += CW
    while j < count:
        var aij = load(a, b + i * n + j_start + j)
        var corr = add(
            mul(load(a, b + pk * n + j_start + j), qpk),
            mul(load(a, b + pl * n + j_start + j), qpl),
        )
        store(dst, ob + oi * m + oj_start + j, add(aij, corr))
        j += 1


def innerconnect_frequency(
    a: Ptr, dst: Ptr, f: Int, n: Int, pk: Int, pl: Int,
):
    var m = n - 2
    var b = f * n * n
    var ob = f * m * m
    var akl = sub(C(1.0, 0.0), load(a, b + pk * n + pl))
    var alk = sub(C(1.0, 0.0), load(a, b + pl * n + pk))
    var akk = load(a, b + pk * n + pk)
    var allv = load(a, b + pl * n + pl)
    var det = sub(mul(akl, alk), mul(akk, allv))
    var inv_det = div(C(1.0, 0.0), det)
    var lo = min(pk, pl)
    var hi = max(pk, pl)
    var oi = 0
    for i in range(n):
        if i == pk or i == pl:
            continue
        var aik = load(a, b + i * n + pk)
        var ail = load(a, b + i * n + pl)
        var qpk = mul(add(mul(ail, alk), mul(allv, aik)), inv_det)
        var qpl = mul(add(mul(aik, akl), mul(akk, ail)), inv_det)
        innerconnect_segment(
            a, dst, b, ob, n, m, i, oi, 0, 0, lo, pk, pl, qpk, qpl,
        )
        innerconnect_segment(
            a, dst, b, ob, n, m, i, oi, lo + 1, lo, hi - lo - 1,
            pk, pl, qpk, qpl,
        )
        innerconnect_segment(
            a, dst, b, ob, n, m, i, oi, hi + 1, hi - 1, n - hi - 1,
            pk, pl, qpk, qpl,
        )
        oi += 1


def csqrt(a: C) -> C:
    var mag = sqrt(abs2(a))
    var re = sqrt(max(0.0, (mag + a.re) * 0.5))
    var im = sqrt(max(0.0, (mag - a.re) * 0.5))
    if a.im < 0.0:
        im = -im
    return C(re, im)


def identity(p: Ptr, n: Int):
    for i in range(n):
        for j in range(n):
            store(p, i * n + j, C(1.0 if i == j else 0.0, 0.0))


def solve_left(a: Ptr, b: Ptr, n: Int) -> Bool:
    """Solve A X = B in place by pivoted Gauss-Jordan elimination."""
    for col in range(n):
        var pivot = col
        var best = abs2(load(a, col * n + col))
        for row in range(col + 1, n):
            var cand = abs2(load(a, row * n + col))
            if cand > best:
                best = cand
                pivot = row
        if best <= 1e-300:
            return False
        if pivot != col:
            for j in range(n):
                var t = load(a, col * n + j)
                store(a, col * n + j, load(a, pivot * n + j))
                store(a, pivot * n + j, t)
                t = load(b, col * n + j)
                store(b, col * n + j, load(b, pivot * n + j))
                store(b, pivot * n + j, t)
        var piv = load(a, col * n + col)
        for j in range(n):
            store(a, col * n + j, div(load(a, col * n + j), piv))
            store(b, col * n + j, div(load(b, col * n + j), piv))
        for row in range(n):
            if row == col:
                continue
            var factor = load(a, row * n + col)
            if abs2(factor) == 0.0:
                continue
            for j in range(n):
                store(a, row * n + j,
                      sub(load(a, row * n + j), mul(factor, load(a, col * n + j))))
                store(b, row * n + j,
                      sub(load(b, row * n + j), mul(factor, load(b, col * n + j))))
    return True


def transpose_in_place(a: Ptr, n: Int):
    for i in range(n):
        for j in range(i + 1, n):
            var t = load(a, i * n + j)
            store(a, i * n + j, load(a, j * n + i))
            store(a, j * n + i, t)


def matmul_value(a: Ptr, b: Ptr, i: Int, j: Int, n: Int) -> C:
    var acc = C(0.0, 0.0)
    for k in range(n):
        acc = add(acc, mul(load(a, i * n + k), load(b, k * n + j)))
    return acc


@export("msrf_inverse")
def msrf_inverse(src_addr: Int, dst_addr: Int, scratch_addr: Int,
                 nf: Int, n: Int) abi("C") -> Int:
    var src = Ptr(unsafe_from_address=src_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    var scratch = Ptr(unsafe_from_address=scratch_addr)
    for f in range(nf):
        var base = f * n * n
        for i in range(n * n):
            store(scratch, i, load(src, base + i))
        identity(dst + 2 * base, n)
        if not solve_left(scratch, dst + 2 * base, n):
            return 0
    return 1


@export("msrf_s2z")
def msrf_s2z(s_addr: Int, z0_addr: Int, dst_addr: Int, scratch_addr: Int,
             nf: Int, n: Int, definition: Int) abi("C") -> Int:
    var s = Ptr(unsafe_from_address=s_addr)
    var z0 = Ptr(unsafe_from_address=z0_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    var a = Ptr(unsafe_from_address=scratch_addr)
    for f in range(nf):
        var mb = f * n * n
        var zb = f * n
        for i in range(n):
            var zi = load(z0, zb + i)
            var ui = sqrt(zi.re) / sqrt(abs2(zi))
            for j in range(n):
                var zj = load(z0, zb + j)
                var fj = 1.0 / (2.0 * sqrt(zj.re))
                var uj = sqrt(zj.re) / sqrt(abs2(zj))
                var sqj = csqrt(zj)
                var sij = load(s, mb + i * n + j)
                var one = C(1.0 if i == j else 0.0, 0.0)
                if definition == 0:
                    store(a, i * n + j, scale(sub(one, sij), fj))
                    store(dst, mb + i * n + j,
                          scale(add(mul(sij, zj), conj(zi) if i == j else C(0.0, 0.0)), fj))
                elif definition == 1:
                    var usu = scale(sij, uj / ui)
                    store(a, i * n + j, sub(one, usu))
                    store(dst, mb + i * n + j, mul(add(one, usu), zj))
                else:
                    store(a, i * n + j, sub(one, sij))
                    store(dst, mb + i * n + j, mul(add(one, sij), sqj))
        if not solve_left(a, dst + 2 * mb, n):
            return 0
        if definition == 2:
            for i in range(n):
                var sqi = csqrt(load(z0, zb + i))
                for j in range(n):
                    store(dst, mb + i * n + j,
                          mul(sqi, load(dst, mb + i * n + j)))
    return 1


@export("msrf_z2s")
def msrf_z2s(z_addr: Int, z0_addr: Int, dst_addr: Int, scratch_addr: Int,
             nf: Int, n: Int, definition: Int) abi("C") -> Int:
    var z = Ptr(unsafe_from_address=z_addr)
    var z0 = Ptr(unsafe_from_address=z0_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    var at = Ptr(unsafe_from_address=scratch_addr)
    for f in range(nf):
        var mb = f * n * n
        var zb = f * n
        for i in range(n):
            var zi = load(z0, zb + i)
            var fi = 1.0 / (2.0 * sqrt(zi.re))
            var ui = sqrt(zi.re) / sqrt(abs2(zi))
            var yi = div(C(1.0, 0.0), csqrt(zi))
            for j in range(n):
                var zj = load(z0, zb + j)
                var yj = div(C(1.0, 0.0), csqrt(zj))
                var zij = load(z, mb + i * n + j)
                var diag = i == j
                var aa: C
                var bb: C
                if definition == 0:
                    aa = scale(add(zij, zi if diag else C(0.0, 0.0)), fi)
                    bb = scale(sub(zij, conj(zi) if diag else C(0.0, 0.0)), fi)
                elif definition == 1:
                    aa = scale(add(zij, zi if diag else C(0.0, 0.0)), ui)
                    bb = scale(sub(zij, zi if diag else C(0.0, 0.0)), ui)
                else:
                    var zn = mul(mul(yi, zij), yj)
                    aa = add(zn, C(1.0 if diag else 0.0, 0.0))
                    bb = sub(zn, C(1.0 if diag else 0.0, 0.0))
                store(at, j * n + i, aa)
                store(dst, mb + j * n + i, bb)
        if not solve_left(at, dst + 2 * mb, n):
            return 0
        transpose_in_place(dst + 2 * mb, n)
    return 1


@export("msrf_s2t")
def msrf_s2t(s_addr: Int, dst_addr: Int, scratch_addr: Int,
             nf: Int, n: Int) abi("C") -> Int:
    if n % 2 != 0:
        return 0
    var s = Ptr(unsafe_from_address=s_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    var a = Ptr(unsafe_from_address=scratch_addr)
    var si = a + 2 * (n // 2) * (n // 2)
    var w = si + 2 * (n // 2) * (n // 2)
    var h = n // 2
    for f in range(nf):
        var mb = f * n * n
        for i in range(h):
            for j in range(h):
                store(a, i * h + j, load(s, mb + (i + h) * n + j))
        identity(si, h)
        if not solve_left(a, si, h):
            return 0
        for i in range(h):
            for j in range(h):
                var acc = C(0.0, 0.0)
                for k in range(h):
                    acc = add(acc, mul(load(si, i * h + k),
                                       load(s, mb + (k + h) * n + j + h)))
                store(w, i * h + j, acc)
        for i in range(h):
            for j in range(h):
                var tl = load(s, mb + i * n + j + h)
                var tr = C(0.0, 0.0)
                var corr = C(0.0, 0.0)
                for k in range(h):
                    tr = add(tr, mul(load(s, mb + i * n + k), load(si, k * h + j)))
                    corr = add(corr, mul(load(s, mb + i * n + k), load(w, k * h + j)))
                store(dst, mb + i * n + j, sub(tl, corr))
                store(dst, mb + i * n + j + h, tr)
                store(dst, mb + (i + h) * n + j, neg(load(w, i * h + j)))
                store(dst, mb + (i + h) * n + j + h, load(si, i * h + j))
    return 1


@export("msrf_t2s")
def msrf_t2s(t_addr: Int, dst_addr: Int, scratch_addr: Int,
             nf: Int, n: Int) abi("C") -> Int:
    if n % 2 != 0:
        return 0
    var t = Ptr(unsafe_from_address=t_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    var a = Ptr(unsafe_from_address=scratch_addr)
    var ti = a + 2 * (n // 2) * (n // 2)
    var w = ti + 2 * (n // 2) * (n // 2)
    var h = n // 2
    for f in range(nf):
        var mb = f * n * n
        for i in range(h):
            for j in range(h):
                store(a, i * h + j, load(t, mb + (i + h) * n + j + h))
        identity(ti, h)
        if not solve_left(a, ti, h):
            return 0
        for i in range(h):
            for j in range(h):
                var acc = C(0.0, 0.0)
                for k in range(h):
                    acc = add(acc, mul(load(ti, i * h + k),
                                       load(t, mb + (k + h) * n + j)))
                store(w, i * h + j, acc)
        for i in range(h):
            for j in range(h):
                var s11 = C(0.0, 0.0)
                var corr = C(0.0, 0.0)
                for k in range(h):
                    s11 = add(s11, mul(load(t, mb + i * n + k + h),
                                       load(ti, k * h + j)))
                    corr = add(corr, mul(load(t, mb + i * n + k + h),
                                        load(w, k * h + j)))
                store(dst, mb + i * n + j, s11)
                store(dst, mb + i * n + j + h,
                      sub(load(t, mb + i * n + j), corr))
                store(dst, mb + (i + h) * n + j, load(ti, i * h + j))
                store(dst, mb + (i + h) * n + j + h, neg(load(w, i * h + j)))
    return 1


@export("msrf_s2a")
def msrf_s2a(s_addr: Int, z0_addr: Int, dst_addr: Int, nf: Int) abi("C") -> Int:
    var s = Ptr(unsafe_from_address=s_addr)
    var z0 = Ptr(unsafe_from_address=z0_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    for f in range(nf):
        var b = f * 4
        var z1 = load(z0, f * 2)
        var z2 = load(z0, f * 2 + 1)
        var s11 = load(s, b)
        var s12 = load(s, b + 1)
        var s21 = load(s, b + 2)
        var s22 = load(s, b + 3)
        var denom = scale(s21, 2.0 * sqrt(z1.re * z2.re))
        var one = C(1.0, 0.0)
        var cross = mul(s12, s21)
        var az = add(conj(z1), mul(s11, z1))
        var dz = add(conj(z2), mul(s22, z2))
        var av = div(add(mul(az, sub(one, s22)), mul(cross, z1)), denom)
        var cv = div(sub(mul(sub(one, s11), sub(one, s22)), cross), denom)
        var bv = div(sub(mul(az, dz), mul(cross, mul(z1, z2))), denom)
        var dv = div(add(mul(sub(one, s11), dz), mul(cross, z2)), denom)
        store(dst, b, av)
        store(dst, b + 1, bv)
        store(dst, b + 2, cv)
        store(dst, b + 3, dv)
    return 1


@export("msrf_a2s")
def msrf_a2s(a_addr: Int, z0_addr: Int, dst_addr: Int, nf: Int) abi("C") -> Int:
    var a = Ptr(unsafe_from_address=a_addr)
    var z0 = Ptr(unsafe_from_address=z0_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    for f in range(nf):
        var b = f * 4
        var z1 = load(z0, f * 2)
        var z2 = load(z0, f * 2 + 1)
        var av = load(a, b)
        var bv = load(a, b + 1)
        var cv = load(a, b + 2)
        var dv = load(a, b + 3)
        var denom = add(add(mul(av, z2), bv), add(mul(cv, mul(z1, z2)), mul(dv, z1)))
        var root2 = 2.0 * sqrt(z1.re * z2.re)
        store(dst, b, div(sub(add(mul(av, z2), bv),
                              add(mul(cv, mul(conj(z1), z2)), mul(dv, conj(z1)))), denom))
        store(dst, b + 2, div(C(root2, 0.0), denom))
        store(dst, b + 1, div(scale(sub(mul(av, dv), mul(bv, cv)), root2), denom))
        store(dst, b + 3, div(add(sub(bv, mul(av, conj(z2))),
                                  sub(mul(dv, z1), mul(cv, mul(z1, conj(z2))))), denom))
    return 1


@export("msrf_passivity")
def msrf_passivity(s_addr: Int, dst_addr: Int, nf: Int, n: Int) abi("C"):
    var s = Ptr(unsafe_from_address=s_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    for f in range(nf):
        var b = f * n * n
        for i in range(n):
            for j in range(n):
                var acc = C(0.0, 0.0)
                for k in range(n):
                    acc = add(acc, mul(conj(load(s, b + k * n + i)),
                                       load(s, b + k * n + j)))
                store(dst, b + i * n + j, csqrt(acc))


@export("msrf_reciprocity")
def msrf_reciprocity(s_addr: Int, dst_addr: Int, nf: Int, n: Int) abi("C"):
    var s = Ptr(unsafe_from_address=s_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    for f in range(nf):
        var b = f * n * n
        for i in range(n):
            for j in range(n):
                var d = sub(load(s, b + i * n + j), load(s, b + j * n + i))
                store(dst, b + i * n + j, C(sqrt(abs2(d)), 0.0))


@export("msrf_innerconnect")
def msrf_innerconnect(a_addr: Int, dst_addr: Int, nf: Int, n: Int,
                      pk: Int, pl: Int) abi("C") -> Int:
    var a = Ptr(unsafe_from_address=a_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    for f in range(nf):
        var b = f * n * n
        var akl = sub(C(1.0, 0.0), load(a, b + pk * n + pl))
        var alk = sub(C(1.0, 0.0), load(a, b + pl * n + pk))
        var akk = load(a, b + pk * n + pk)
        var allv = load(a, b + pl * n + pl)
        var det = sub(mul(akl, alk), mul(akk, allv))
        if abs2(det) < 1e-24:
            return 0
    if nf >= 131072:
        def work(task: Int) {
            imm a, imm dst, imm nf, imm n, imm pk, imm pl
        }:
            var start = task * nf // 2
            var end = (task + 1) * nf // 2
            for f in range(start, end):
                innerconnect_frequency(a, dst, f, n, pk, pl)
        initialize_runtime()
        parallelize(work, 2, 2)
    else:
        for f in range(nf):
            innerconnect_frequency(a, dst, f, n, pk, pl)
    return 1
