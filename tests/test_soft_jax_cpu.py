"""CPU tests for the differentiable soft-s jax frontend.

The jax frontend is an extension of :mod:`cucount.jax`; its FFI targets exist for
both the CUDA and the CPU platform. These tests pin the CPU path against the
validated :mod:`cucount.numpy.soft` reference (counts and analytic adjoint), so
they run anywhere the CPU extension is built.
"""

import numpy as np
import pytest

from cucount.numpy import BinAttrs, MeshAttrs, Particles
from cucount.numpy import soft as nsoft

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)

from cucount import jax as cjax  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(cjax, 'ffi_cpulib') or not hasattr(cjax.ffi_cpulib, 'count2_soft'),
    reason='the soft jax frontend needs the CPU FFI module')

ACC = 'float64'  # exact reference comparisons; the CPU accumulator is double anyway

BOX = 300.0
EDGES = np.linspace(1.0, 100.0, 21)
SIGMA = 2.0
CUTOFF = 1e-13
N = 64


def _setup(seed=7, n=N, weighted=True, edges=EDGES):
    rng = np.random.default_rng(seed)
    pos1 = rng.uniform(0., BOX, (n, 3))
    pos2 = rng.uniform(0., BOX, (n, 3))
    w1 = rng.uniform(0.5, 1.5, n) if weighted else None
    w2 = rng.uniform(0.5, 1.5, n) if weighted else None
    battrs = BinAttrs(s=edges)
    p1, p2 = Particles(pos1, w1), Particles(pos2, w2)
    mattrs = MeshAttrs(p1, p2, boxsize=BOX, battrs=battrs, periodic=True)
    return pos1, pos2, w1, w2, battrs, mattrs


def _reference(pos1, pos2, w1, w2, battrs, mattrs, sigma=SIGMA, **kw):
    p1 = Particles(np.asarray(pos1), None if w1 is None else np.asarray(w1))
    p2 = Particles(np.asarray(pos2), None if w2 is None else np.asarray(w2))
    return nsoft.count2_soft(p1, p2, battrs=battrs, soft_sigma=sigma,
                             soft_cutoff=CUTOFF, mattrs=mattrs, backend='cpu',
                             soft_accum=ACC, **kw)


def _reference_vjp(pos1, pos2, w1, w2, battrs, mattrs, cot, sigma=SIGMA, **kw):
    p1 = Particles(np.asarray(pos1), None if w1 is None else np.asarray(w1))
    p2 = Particles(np.asarray(pos2), None if w2 is None else np.asarray(w2))
    return nsoft.count2_soft_vjp(p1, p2, battrs=battrs, cotangent=cot,
                                 soft_sigma=sigma, soft_cutoff=CUTOFF,
                                 mattrs=mattrs, backend='cpu', soft_accum=ACC, **kw)


def _jax(pos1, w1, pos2, w2, battrs, mattrs, sigma=SIGMA, **kw):
    p1, p2 = cjax.Particles(pos1, w1), cjax.Particles(pos2, w2)
    return cjax.count2_soft(p1, p2, battrs=battrs, soft_sigma=sigma,
                            soft_cutoff=CUTOFF, mattrs=mattrs, backend='cpu',
                            soft_accum=ACC, **kw)['weight']


def test_counts_match_numpy_frontend():
    pos1, pos2, w1, w2, battrs, mattrs = _setup()
    got = np.asarray(_jax(pos1, w1, pos2, w2, battrs, mattrs))
    expected = _reference(pos1, pos2, w1, w2, battrs, mattrs)['weight']
    assert got.shape == expected.shape == (EDGES.size - 1,)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_position_and_weight_gradients_match_the_analytic_adjoint():
    pos1, pos2, w1, w2, battrs, mattrs = _setup(seed=11)
    cot = np.random.default_rng(3).normal(size=EDGES.size - 1)
    ref = _reference_vjp(pos1, pos2, w1, w2, battrs, mattrs, cot)

    def loss(a, b, u, v):
        return (jax.numpy.sum(_jax(a, u, b, v, battrs, mattrs) * cot))

    gp1, gp2, gw1, gw2 = jax.grad(loss, argnums=(0, 1, 2, 3))(pos1, pos2, w1, w2)
    np.testing.assert_allclose(np.asarray(gp1), ref['grad_position1'], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.asarray(gp2), ref['grad_position2'], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.asarray(gw1).ravel(), ref['grad_weight1'], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.asarray(gw2).ravel(), ref['grad_weight2'], rtol=1e-11, atol=1e-11)


def test_position_gradient_matches_finite_difference():
    pos1, pos2, w1, w2, battrs, mattrs = _setup(seed=13)
    cot = np.random.default_rng(5).normal(size=EDGES.size - 1)

    def ref_loss(a):
        return np.dot(_reference(a, pos2, w1, w2, battrs, mattrs)['weight'], cot)

    grad = np.asarray(jax.grad(lambda a: jax.numpy.sum(_jax(a, w1, pos2, w2, battrs, mattrs) * cot))(pos1))
    h = 1e-5
    for i in (0, 7, 21):
        for d in range(3):
            plus, minus = pos1.copy(), pos1.copy()
            plus[i, d] += h
            minus[i, d] -= h
            fd = (ref_loss(plus) - ref_loss(minus)) / (2 * h)
            assert fd == pytest.approx(grad[i, d], rel=2e-5, abs=1e-8)


def test_autocorrelation_gradient_is_symmetric_and_translation_invariant():
    pos, _, w, _, battrs, _ = _setup(seed=17)
    part = Particles(pos, w)
    mattrs = MeshAttrs(part, part, boxsize=BOX, battrs=battrs, periodic=True)
    cot = np.random.default_rng(9).normal(size=EDGES.size - 1)

    def loss(p):
        q = cjax.Particles(p, w)
        return jax.numpy.sum(cjax.count2_soft(q, q, battrs=battrs, soft_sigma=SIGMA,
                                               soft_cutoff=CUTOFF, mattrs=mattrs,
                                               backend='cpu', soft_accum=ACC)['weight'] * cot)

    grad = np.asarray(jax.grad(loss)(pos))
    np.testing.assert_allclose(grad, jax.grad(loss)(pos), rtol=0., atol=0.)
    # Every pair is visited twice, so the sum over particles of dL/dp cancels in a
    # periodic box (translation invariance).
    scale = np.abs(grad).max()
    assert np.abs(grad.sum(axis=0)).max() <= 1e-9 * scale


def test_counts_and_gradient_work_under_jit():
    pos1, pos2, w1, w2, battrs, mattrs = _setup(seed=19)
    cot = np.random.default_rng(11).normal(size=EDGES.size - 1)

    def counts(a, b):
        return cjax.count2_soft(cjax.Particles(a, w1), cjax.Particles(b, w2),
                                battrs=battrs, soft_sigma=SIGMA, soft_cutoff=CUTOFF,
                                mattrs=mattrs, backend='cpu', soft_accum=ACC)['weight']

    got = np.asarray(jax.jit(counts)(pos1, pos2))
    expected = _reference(pos1, pos2, w1, w2, battrs, mattrs)['weight']
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    g1, g2 = jax.jit(jax.grad(lambda a, b: jax.numpy.sum(counts(a, b) * cot),
                              argnums=(0, 1)))(pos1, pos2)
    ref = _reference_vjp(pos1, pos2, w1, w2, battrs, mattrs, cot)
    np.testing.assert_allclose(np.asarray(g1), ref['grad_position1'], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.asarray(g2), ref['grad_position2'], rtol=1e-11, atol=1e-11)


def test_non_uniform_edges_match_numpy_and_gradient():
    edges = np.geomspace(1.0, 100.0, 9)
    pos1, pos2, w1, w2, battrs, mattrs = _setup(seed=23, n=32, edges=edges)
    expected = _reference(pos1, pos2, w1, w2, battrs, mattrs, sigma=None)['weight']
    got = np.asarray(_jax(pos1, w1, pos2, w2, battrs, mattrs, sigma=None))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    cot = np.random.default_rng(29).normal(size=edges.size - 1)
    grad = jax.grad(lambda a: jax.numpy.sum(
        _jax(a, w1, pos2, w2, battrs, mattrs, sigma=None) * cot))(pos1)
    ref = _reference_vjp(pos1, pos2, w1, w2, battrs, mattrs, cot, sigma=None)
    np.testing.assert_allclose(np.asarray(grad), ref['grad_position1'], rtol=1e-11, atol=1e-11)


def test_soft_pad_matches_numpy_and_keeps_real_bins():
    pos1, pos2, w1, w2, battrs, mattrs = _setup(seed=31, n=48)
    expected = _reference(pos1, pos2, w1, w2, battrs, mattrs, soft_pad=2.0)
    got = _jax(pos1, w1, pos2, w2, battrs, mattrs, soft_pad=2.0)
    np.testing.assert_allclose(np.asarray(got), expected['weight'], rtol=1e-12, atol=1e-12)
    assert expected['soft']['pad_lo'] + expected['soft']['pad_hi'] > 0

    cot = np.random.default_rng(37).normal(size=EDGES.size - 1)
    grad = jax.grad(lambda a: jax.numpy.sum(
        _jax(a, w1, pos2, w2, battrs, mattrs, soft_pad=2.0) * cot))(pos1)
    ref = _reference_vjp(pos1, pos2, w1, w2, battrs, mattrs, cot, soft_pad=2.0)
    np.testing.assert_allclose(np.asarray(grad), ref['grad_position1'], rtol=1e-11, atol=1e-11)


def test_soft_mu_multipoles_match_numpy():
    edges = np.linspace(1.0, 100.0, 11)
    rng = np.random.default_rng(41)
    n = 48
    pos1 = rng.uniform(0., BOX, (n, 3))
    pos2 = rng.uniform(0., BOX, (n, 3))
    w1 = rng.uniform(0.5, 1.5, n)
    w2 = rng.uniform(0.5, 1.5, n)
    battrs = BinAttrs(s=edges, pole=(np.array([0, 2, 4]), 'x'))
    p1, p2 = Particles(pos1, w1), Particles(pos2, w2)
    mattrs = MeshAttrs(p1, p2, boxsize=BOX, battrs=battrs, periodic=True)

    kw = dict(soft_mu_sigma=0.05)
    expected = _reference(pos1, pos2, w1, w2, battrs, mattrs, **kw)['weight']
    got = np.asarray(_jax(pos1, w1, pos2, w2, battrs, mattrs, **kw))
    assert got.shape == expected.shape == (edges.size - 1, 3)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    cot = rng.normal(size=expected.shape)
    grad = jax.grad(lambda a: jax.numpy.sum(_jax(a, w1, pos2, w2, battrs, mattrs, **kw) * cot))(pos1)
    ref = _reference_vjp(pos1, pos2, w1, w2, battrs, mattrs, cot, **kw)
    np.testing.assert_allclose(np.asarray(grad), ref['grad_position1'], rtol=1e-11, atol=1e-11)


def test_float32_samples_use_the_f32_ffi_and_match_float64():
    # With x64 enabled, individual weights are formatted as float64 (so the f64
    # target would be picked); unweighted Particles get ``ones_like(positions)``,
    # which keeps the float32 dtype and selects the f32 FFI target.
    pos1, pos2, _, _, battrs, mattrs = _setup(seed=43, n=48, weighted=False)
    p1 = pos1.astype(np.float32)
    p2 = pos2.astype(np.float32)

    def counts(a, b):
        # positions_type='xyz' takes (x, y, z) and preserves float32; the
        # default 'pos' path casts positions to float64 (picking the f64 target).
        return cjax.count2_soft(cjax.Particles(tuple(a.T), positions_type='xyz'),
                                cjax.Particles(tuple(b.T), positions_type='xyz'),
                                battrs=battrs, soft_sigma=SIGMA, soft_cutoff=CUTOFF,
                                mattrs=mattrs, backend='cpu', soft_accum=ACC)['weight']

    got = counts(p1, p2)
    assert got.dtype == jax.numpy.float32
    expected = _reference(p1, p2, None, None, battrs, mattrs)['weight']
    np.testing.assert_allclose(np.asarray(got), expected, rtol=2e-5, atol=2e-5)

    cot = np.random.default_rng(47).normal(size=EDGES.size - 1).astype(np.float32)
    grad = jax.grad(lambda a: jax.numpy.sum(counts(a, p2) * cot))(p1)
    assert grad.dtype == jax.numpy.float32
    ref = _reference_vjp(p1, p2, None, None, battrs, mattrs, cot.astype(np.float64))
    np.testing.assert_allclose(np.asarray(grad), ref['grad_position1'], rtol=5e-5, atol=5e-5)
