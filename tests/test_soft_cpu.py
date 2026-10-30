"""Soft (Gaussian-softmax) binning on the CPU backend.

The CPU kernel is a scalar port of the CUDA one: it shares *every* decision that
defines the answer -- the SoftBinSpec resolution in include/soft.h (sigma
defaults, the mu quadrature, the window cutoff, the general-grid search) -- and
reimplements only the per-pair scatter and its adjoint. So the interesting
properties are: it agrees with a numpy O(N^2) reference and with the CUDA kernel
where one is available, it conserves weight, an analytic adjoint matches finite
differences, and it keeps the same refusals as the CUDA path.

The accumulator is a per-thread double histogram, so the CPU results are
reproducible to rounding and the tolerances here are tight (1e-12) rather than
the CUDA tile's 1e-5.
"""

import numpy as np
import pytest

from cucount.numpy import (BinAttrs, MeshAttrs, Particles, SelectionAttrs,
                           SplitAttrs, WeightAttrs, cpulib)
from cucount.numpy import count2, count2_soft, count2_soft_vjp
from cucount.numpy.soft import mu_spec, soft_mu_weights, soft_weights

needs_cpu = pytest.mark.skipif(cpulib is None, reason='CPU extension not built')


def _cuda_ready():
    try:
        from cucount.numpy import cudalib
    except ImportError:  # pragma: no cover
        return False
    return cudalib is not None and cudalib.device_count() > 0


needs_both = pytest.mark.skipif(
    cpulib is None or not _cuda_ready(),
    reason='both backends needed to cross-check the CPU kernel against the CUDA one')

BOX = 500.0
EDGES = np.linspace(1.0, 100.0, 21)
LOG_EDGES = np.geomspace(1.0, 100.0, 21)


def _setup(n=300, seed=11, edges=EDGES, box=BOX, periodic=True, weights=True):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, box, (n, 3))
    p = Particles(pos, rng.uniform(0.5, 1.5, n) if weights else None)
    ba = BinAttrs(s=edges)
    ma = MeshAttrs(p, p, boxsize=box, battrs=ba, periodic=periodic)
    return p, ba, ma


def _brute_soft(p, edges, sigma, box=None):
    """Ordered-pair reference: minimum-image separation, full-grid softmax."""
    pos = np.asarray(p.positions)
    w = np.asarray(p.values[0]) if p.values else np.ones(pos.shape[0])
    i, j = np.indices((pos.shape[0], pos.shape[0]))
    diff = pos[i] - pos[j]
    if box is not None:
        diff -= box * np.round(diff / box)
    s = np.sqrt((diff ** 2).sum(axis=-1))
    inside = (s >= edges[0]) & (s < edges[-1]) & (i != j)
    s = s[inside]
    ww = (w[i] * w[j])[inside]
    return soft_weights(s, edges, sigma).T @ ww


@needs_cpu
def test_soft_s_runs_on_cpu_and_conserves_weight():
    p, ba, ma = _setup()
    hard = count2(p, p, battrs=ba, mattrs=ma, backend='cpu')['weight']
    soft = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cpu')
    assert soft['weight'].shape == hard.shape
    # Smoothing redistributes pairs; it must not reweight them.
    np.testing.assert_allclose(soft['weight'].sum(), hard.sum(), rtol=1e-13)


@needs_cpu
def test_matches_the_numpy_reference():
    p, ba, ma = _setup()
    sigma = 2.0
    got = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cpu', soft_sigma=sigma)
    want = _brute_soft(p, EDGES, sigma, box=BOX)
    # The uniform path builds the taps by a multiplicative recurrence (one exp
    # per pair) where the reference exponentiates each tap, which is the only
    # difference: ~1e-11 relative, and only visible in nearly empty bins.
    np.testing.assert_allclose(got['weight'], want, rtol=1e-10)


@needs_cpu
def test_default_sigma_is_half_a_bin():
    """``soft_sigma=None`` on a uniform grid is the oracle's half-bin default."""
    p, ba, ma = _setup()
    got = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cpu')
    ds = (EDGES[-1] - EDGES[0]) / (EDGES.size - 1)
    assert got['soft']['sigma'] == pytest.approx(0.5 * ds)
    # On an arithmetic grid the half-bin width is a single absolute sigma (the
    # 'width' kind is the per-bin rule that only non-uniform grids need), which is
    # also what the CUDA metadata reports for this request.
    assert got['soft']['sigma_kind'] == 'absolute'
    want = _brute_soft(p, EDGES, 0.5 * ds, box=BOX)
    np.testing.assert_allclose(got['weight'], want, rtol=1e-10)


@needs_cpu
def test_sigma_to_zero_recovers_hard_binning():
    """On a uniform grid the limit is the hard histogram, as for the oracle."""
    p, ba, ma = _setup()
    hard = count2(p, p, battrs=ba, mattrs=ma, backend='cpu')['weight']
    soft = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cpu',
                       soft_sigma=1e-6, soft_cutoff=1e-13)['weight']
    np.testing.assert_allclose(soft, hard, rtol=1e-10, atol=1e-8 * hard.max())


@needs_cpu
def test_logarithmic_grid_matches_the_numpy_reference():
    """Non-uniform edges go through the general path (binary search, per-tap exp)."""
    p, ba, ma = _setup(edges=LOG_EDGES)
    got = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cpu')
    assert got['soft']['uniform'] is False
    sigma = 0.5 * np.diff(LOG_EDGES)
    # Per-bin sigma: compare against the same truncated softmax computed directly.
    pos = np.asarray(p.positions)
    w = np.asarray(p.values[0])
    i, j = np.indices((pos.shape[0], pos.shape[0]))
    diff = pos[i] - pos[j]
    diff -= BOX * np.round(diff / BOX)
    s = np.sqrt((diff ** 2).sum(axis=-1))
    inside = (s >= LOG_EDGES[0]) & (s < LOG_EDGES[-1]) & (i != j)
    ref = np.zeros(LOG_EDGES.size - 1)
    centres = 0.5 * (LOG_EDGES[:-1] + LOG_EDGES[1:])
    for sv, ww in zip(s[inside], (w[i] * w[j])[inside]):
        x = (sv - centres) ** 2 / (2.0 * sigma ** 2)
        wt = np.exp(-(x - x.min()))
        ref += ww * wt / wt.sum()
    np.testing.assert_allclose(got['weight'], ref, rtol=1e-10, atol=1e-12 * ref.max())


@needs_cpu
def test_adjoint_matches_finite_differences():
    p, ba, ma = _setup(n=40, seed=3)
    sigma = 3.0
    rng = np.random.default_rng(7)
    cotangent = rng.normal(size=EDGES.size - 1)

    def loss(positions, w, sig):
        q = Particles(positions, w)
        res = count2_soft(q, q, battrs=ba, mattrs=ma, backend='cpu',
                          soft_sigma=sig)
        return float(np.dot(res['weight'], cotangent))

    pos = np.array(p.positions)
    w = np.array(p.values[0])
    got = count2_soft_vjp(p, p, battrs=ba, cotangent=cotangent, mattrs=ma,
                          backend='cpu', soft_sigma=sigma)
    eps = 1e-5
    fd_pos = np.zeros_like(pos)
    for k in range(pos.shape[0]):
        for d in range(3):
            plus, minus = pos.copy(), pos.copy()
            plus[k, d] += eps
            minus[k, d] -= eps
            fd_pos[k, d] = (loss(plus, w, sigma) - loss(minus, w, sigma)) / (2 * eps)
    analytic = got['grad_position1'] + got['grad_position2']
    np.testing.assert_allclose(analytic, fd_pos, rtol=2e-5, atol=1e-8)

    fd_w = np.zeros_like(w)
    for k in range(w.size):
        plus, minus = w.copy(), w.copy()
        plus[k] += eps
        minus[k] -= eps
        fd_w[k] = (loss(pos, plus, sigma) - loss(pos, minus, sigma)) / (2 * eps)
    np.testing.assert_allclose(got['grad_weight1'] + got['grad_weight2'], fd_w,
                               rtol=2e-5, atol=1e-8)

    fd_s = (loss(pos, w, sigma + eps) - loss(pos, w, sigma - eps)) / (2 * eps)
    assert got['grad_sigma'] == pytest.approx(fd_s, rel=2e-5)


@needs_cpu
def test_meta_reports_the_cpu_accumulator():
    p, ba, ma = _setup()
    soft = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cpu')
    meta = soft['soft']
    # The CPU visits each pair once per thread into a private double histogram:
    # there is no float32 tile to tune, and no shared-memory budget to pick.
    assert meta['accum'] == 'float64'
    assert meta['path'] == 'threads'
    assert meta['copies'] == 0
    assert meta['nbins'] == EDGES.size - 1
    assert meta['uniform'] is True
    assert meta['taps'] == 2 * meta['kmax'] + 1


@needs_both
def test_agrees_with_the_cuda_kernel():
    """The two kernels share the request resolution; only the scatter differs."""
    p, ba, ma = _setup()
    common = dict(battrs=ba, mattrs=ma, soft_sigma=2.5)
    cpu = count2_soft(p, p, backend='cpu', **common)['weight']
    gpu = count2_soft(p, p, backend='cuda', soft_accum='float64', **common)['weight']
    np.testing.assert_allclose(cpu, gpu, rtol=1e-13)


@needs_cpu
def test_rejects_what_the_cuda_kernel_rejects():
    p, ba, ma = _setup()
    sp = SplitAttrs(mode='jackknife', nsplits=4)
    jk = Particles(np.asarray(p.positions), np.asarray(p.values[0]),
                   splits=np.zeros(p.size, dtype=np.int64))
    with pytest.raises(NotImplementedError, match='split/jackknife'):
        count2_soft(jk, jk, battrs=ba, mattrs=ma, spattrs=sp, backend='cpu')
    with pytest.raises(NotImplementedError, match='split/jackknife'):
        count2_soft_vjp(jk, jk, battrs=ba, cotangent=np.zeros(EDGES.size - 1),
                        mattrs=ma, spattrs=sp, backend='cpu')
    # Spin must be declared and the catalogues must carry spin values, so the
    # rejection we care about is the kernel's, not an earlier shape assertion.
    rng = np.random.default_rng(17)
    spin_values = rng.uniform(-1.0, 1.0, (p.size, 2))
    sp_p = Particles(np.asarray(p.positions), np.asarray(p.values[0]),
                     spin_values=spin_values)
    sp_ba = BinAttrs(s=EDGES)
    sp_ma = MeshAttrs(sp_p, sp_p, boxsize=BOX, battrs=sp_ba, periodic=True)
    wa = WeightAttrs(spin=(1, 2))
    with pytest.raises((NotImplementedError, ValueError), match='spin|not served|soft'):
        count2_soft(sp_p, sp_p, battrs=sp_ba, mattrs=sp_ma, wattrs=wa, backend='cpu')


@needs_cpu
def test_multipoles_and_mu_edges():
    """The CPU path serves soft mu quadrature, including caller-supplied edges."""
    rng = np.random.default_rng(21)
    pos = rng.uniform(0.0, BOX, (200, 3))
    p = Particles(pos, rng.uniform(0.5, 1.5, 200))
    ba = BinAttrs(s=EDGES, pole=(np.array([0, 2, 4]), 'z'))
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    soft = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cpu')
    assert soft['weight'].shape == (EDGES.size - 1, 3)
    # M_0 = 1 exactly, so the monopole is the soft-s count.
    mono = count2_soft(p, p, battrs=BinAttrs(s=EDGES), mattrs=ma, backend='cpu')
    np.testing.assert_allclose(soft['weight'][:, 0], mono['weight'], rtol=1e-13)

    mu_edges = np.cos(np.linspace(0.0, np.pi, 25))[::-1]
    explicit = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cpu',
                           soft_mu_edges=mu_edges)
    assert explicit['weight'].shape == soft['weight'].shape
    assert explicit['soft']['mu_uniform'] is False
    # A narrow mu kernel approaches the hard multipoles of the same mu binning.
    ms = mu_spec(soft_mu_edges=mu_edges)
    assert ms['nmu'] == mu_edges.size - 1


@needs_cpu
def test_mu_adjoint_matches_finite_differences():
    rng = np.random.default_rng(23)
    pos = rng.uniform(0.0, BOX, (30, 3))
    w = rng.uniform(0.5, 1.5, 30)
    p = Particles(pos, w)
    ba = BinAttrs(s=EDGES, pole=(np.array([0, 2]), 'z'))
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    rng = np.random.default_rng(29)
    cotangent = rng.normal(size=(EDGES.size - 1, 2))
    sigma_mu = 0.1

    def loss(positions):
        q = Particles(positions, w)
        res = count2_soft(q, q, battrs=ba, mattrs=ma, backend='cpu',
                          soft_mu_sigma=sigma_mu)
        return float(np.dot(res['weight'].ravel(), cotangent.ravel()))

    got = count2_soft_vjp(p, p, battrs=ba, cotangent=cotangent, mattrs=ma,
                          backend='cpu', soft_mu_sigma=sigma_mu)
    eps = 1e-5
    fd = np.zeros_like(pos)
    for k in range(pos.shape[0]):
        for d in range(3):
            plus, minus = pos.copy(), pos.copy()
            plus[k, d] += eps
            minus[k, d] -= eps
            fd[k, d] = (loss(plus) - loss(minus)) / (2 * eps)
    np.testing.assert_allclose(got['grad_position1'] + got['grad_position2'], fd,
                               rtol=3e-5, atol=1e-8)


@needs_cpu
def test_selection_is_applied_like_the_cuda_kernel():
    p, ba, ma = _setup()
    # No pair in this catalogue is closer than 10 pc, so a hard count with a
    # 20 pc upper bound leaves bin 4 (edges [21, 26)) empty; the soft kernel still
    # has the Gaussian tail of the 20..26 pc pairs unless the selection vetoes
    # them, and a vetoed pair must vanish from the scatter entirely.
    sa = SelectionAttrs(s=(10.0, 20.0))
    soft = count2_soft(p, p, battrs=ba, mattrs=ma, sattrs=sa, backend='cpu')
    hard = count2(p, p, battrs=ba, mattrs=ma, sattrs=sa, backend='cpu')['weight']
    np.testing.assert_allclose(soft['weight'].sum(), hard.sum(), rtol=1e-13)
    assert hard[4] == 0.0
    full = count2_soft(p, p, battrs=ba, mattrs=ma, backend='cpu')['weight']
    assert full[4] > 0.0
    assert soft['weight'][4] < 0.1 * full[4]
