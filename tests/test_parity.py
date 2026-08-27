import numpy as np
import pytest
import skrf.network as reference

import mojoskrf as rf
import mojoskrf.network as implementation

rng = np.random.default_rng(2026)


def random_s(nfreqs=31, nports=2, scale=0.12):
    shape = (nfreqs, nports, nports)
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def assert_parity(got, expected, rtol=2e-11, atol=2e-11):
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


def test_fix_z0_shape_matches_upstream_broadcasting():
    for z0 in (50, [45, 55], np.linspace(40, 60, 7), np.full((7, 2), 50 + 2j)):
        assert_parity(rf.fix_z0_shape(z0, 7, 2), reference.fix_z0_shape(z0, 7, 2))


@pytest.mark.parametrize("s_def", rf.S_DEFINITIONS)
def test_s2z_matches_upstream_for_all_wave_definitions(s_def):
    s = random_s(nfreqs=37, nports=4)
    z0 = np.column_stack([
        np.linspace(45, 50, 37) + 1j,
        np.linspace(50, 55, 37) - 2j,
        np.linspace(55, 60, 37) + 3j,
        np.linspace(60, 65, 37) - 4j,
    ])
    assert_parity(rf.s2z(s, z0, s_def), reference.s2z(s, z0, s_def))


@pytest.mark.parametrize("s_def", rf.S_DEFINITIONS)
def test_z2s_matches_upstream_for_all_wave_definitions(s_def):
    s = random_s(nfreqs=29, nports=3)
    z0 = np.array([45 + 2j, 50 - 1j, 72 + 4j])
    z = reference.s2z(s, z0, s_def)
    assert_parity(rf.z2s(z, z0, s_def), reference.z2s(z, z0, s_def))


@pytest.mark.parametrize("s_def", rf.S_DEFINITIONS)
def test_s_z_roundtrip(s_def):
    s = random_s(nfreqs=23, nports=5)
    z0 = np.linspace(43, 67, 5) + 1j * np.linspace(-3, 3, 5)
    assert_parity(rf.z2s(rf.s2z(s, z0, s_def), z0, s_def), s)


def test_s2y_and_y2s_match_upstream():
    s = random_s(nfreqs=41, nports=3)
    z0 = [45 + 1j, 50 - 2j, 70 + 3j]
    y = rf.s2y(s, z0)
    assert_parity(y, reference.s2y(s, z0))
    assert_parity(rf.y2s(y, z0), reference.y2s(y, z0))


def test_z2y_y2z_match_upstream_and_roundtrip():
    s = random_s(nfreqs=17, nports=4)
    z = reference.s2z(s)
    y = rf.z2y(z)
    assert_parity(y, reference.z2y(z))
    assert_parity(rf.y2z(y), reference.y2z(y))
    assert_parity(rf.y2z(y), z)


@pytest.mark.parametrize("nports", [2, 4, 6])
def test_scattering_transfer_conversion_matches_upstream(nports):
    s = random_s(nfreqs=19, nports=nports)
    t = rf.s2t(s)
    assert_parity(t, reference.s2t(s))
    assert_parity(rf.t2s(t), reference.t2s(t))
    assert_parity(rf.t2s(t), s)


def test_transfer_conversion_rejects_odd_port_count():
    with pytest.raises(IndexError, match="even number"):
        rf.s2t(random_s(nports=3))
    with pytest.raises(IndexError, match="even number"):
        rf.t2s(random_s(nports=3))


def test_s2a_a2s_match_upstream_with_complex_impedances():
    s = random_s(nfreqs=53, nports=2, scale=0.2)
    z0 = np.column_stack([np.full(53, 48 + 2j), np.full(53, 73 - 4j)])
    a = rf.s2a(s, z0)
    assert_parity(a, reference.s2a(s, z0))
    assert_parity(rf.a2s(a, z0), reference.a2s(a, z0))
    assert_parity(rf.a2s(a, z0), s)


def test_abcd_ideal_through_vector():
    through = np.array([[[0, 1], [1, 0]]], dtype=np.complex128)
    assert_parity(rf.s2a(through), np.array([[[1, 0], [0, 1]]]))


def test_abcd_rejects_non_two_port():
    with pytest.raises(IndexError, match="2-ports"):
        rf.s2a(random_s(nports=3))
    with pytest.raises(IndexError, match="2-ports"):
        rf.a2s(random_s(nports=3))


@pytest.mark.parametrize(
    "new_def,old_def", [("power", None), ("pseudo", "power"), ("traveling", "pseudo")]
)
def test_renormalize_s_matches_upstream(new_def, old_def):
    s = random_s(nfreqs=47, nports=3)
    old = np.array([45 + 1j, 50 - 2j, 60 + 3j])
    new = np.array([55 - 2j, 65 + 1j, 75 - 4j])
    got = rf.renormalize_s(s, old, new, new_def, old_def)
    expected = reference.renormalize_s(s, old, new, new_def, old_def)
    assert_parity(got, expected)


def test_passivity_matches_upstream():
    s = random_s(nfreqs=67, nports=5)
    assert_parity(rf.passivity(s), reference.passivity(s))


def test_passivity_of_lossless_through_is_identity():
    through = np.repeat(
        np.array([[[0, 1], [1, 0]]], dtype=np.complex128), 5, axis=0
    )
    assert_parity(rf.passivity(through), np.repeat(np.eye(2)[None], 5, axis=0))


def test_reciprocity_matches_upstream():
    s = random_s(nfreqs=71, nports=6)
    assert_parity(rf.reciprocity(s), reference.reciprocity(s))


def test_one_port_metrics_raise_like_upstream():
    one = random_s(nports=1)
    with pytest.raises(ValueError):
        rf.passivity(one)
    with pytest.raises(ValueError):
        rf.reciprocity(one)


@pytest.mark.parametrize("ports", [(0, 1), (1, 4)])
def test_innerconnect_s_matches_upstream(ports):
    s = random_s(nfreqs=43, nports=5)
    k, l = ports
    assert_parity(
        rf.innerconnect_s(s, k, l), reference.innerconnect_s(s, k, l)
    )


@pytest.mark.parametrize(
    "nfreqs,nports,ports",
    [
        (16, 6, (0, 5)),
        (17, 7, (1, 5)),
    ],
)
def test_innerconnect_s_simd_blocks_and_scalar_tails(nfreqs, nports, ports):
    s = random_s(nfreqs=nfreqs, nports=nports)
    k, l = ports
    assert_parity(rf.innerconnect_s(s, k, l), reference.innerconnect_s(s, k, l))


@pytest.mark.parametrize("nfreqs", [16_383, 16_384])
def test_innerconnect_s_parallel_threshold(nfreqs):
    s = random_s(nfreqs=nfreqs, nports=3, scale=0.05)
    assert_parity(
        rf.innerconnect_s(s, 0, 2),
        reference.innerconnect_s(s, 0, 2),
    )


def test_innerconnect_s_singular_fallback_matches_upstream_lstsq(monkeypatch):
    monkeypatch.setattr(implementation, "INNERCONNECT_PARALLEL_THRESHOLD", 1)
    s = np.zeros((3, 4, 4), dtype=np.complex128)
    s[:, 0, 1] = 1
    s[:, 1, 0] = 1
    with pytest.warns(RuntimeWarning, match="Singular"):
        got = rf.innerconnect_s(s, 0, 1)
    assert_parity(got, reference.innerconnect_s_lstsq(s, 0, 1))


def test_connect_s_matches_upstream_for_two_two_ports():
    a = random_s(nfreqs=37, nports=2)
    b = random_s(nfreqs=37, nports=2)
    assert_parity(rf.connect_s(a, 1, b, 0), reference.connect_s(a, 1, b, 0))


def test_connect_s_matches_upstream_special_multiport_ordering():
    a = random_s(nfreqs=31, nports=5)
    b = random_s(nfreqs=31, nports=2)
    assert_parity(rf.connect_s(a, 2, b, 0), reference.connect_s(a, 2, b, 0))


def test_deembedding_inverse_matches_upstream():
    s = random_s(nfreqs=59, nports=2, scale=0.2)
    assert_parity(rf.inv(s), reference.inv(s), rtol=1e-10, atol=1e-10)


def test_unknown_wave_definition_raises():
    with pytest.raises(ValueError, match="Unknown s_def"):
        rf.s2z(random_s(), s_def="voltage")


def test_noncontiguous_and_lower_precision_inputs_are_safely_copied():
    base = random_s(nfreqs=13, nports=3).astype(np.complex64)
    noncontiguous = base[:, ::-1, ::-1]
    assert not noncontiguous.flags.c_contiguous
    assert_parity(rf.s2z(noncontiguous), reference.s2z(noncontiguous))


def test_unsafe_dtype_narrowing_is_rejected():
    if hasattr(np, "complex256"):
        with pytest.raises(TypeError, match="safely convert"):
            rf.s2z(np.zeros((1, 2, 2), dtype=np.complex256))
    with pytest.raises(TypeError, match="safely convert"):
        rf.s2z(np.empty((1, 2, 2), dtype=object))


@pytest.mark.parametrize(
    "function,args",
    [
        (rf.s2z, ()),
        (rf.z2s, ()),
        (rf.s2y, ()),
        (rf.y2s, ()),
        (rf.z2y, ()),
        (rf.y2z, ()),
        (rf.s2t, ()),
        (rf.t2s, ()),
        (rf.s2a, ()),
        (rf.a2s, ()),
        (rf.passivity, ()),
        (rf.reciprocity, ()),
        (rf.inv, ()),
    ],
)
def test_empty_frequency_batch_never_crosses_ffi_with_null_buffers(function, args):
    value = np.empty((0, 2, 2), dtype=np.complex128)
    assert function(value, *args).shape == value.shape


def test_zero_port_matrices_are_rejected_before_ffi():
    with pytest.raises(ValueError, match="at least one port"):
        rf.s2z(np.empty((1, 0, 0), dtype=np.complex128))
