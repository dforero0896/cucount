"""Edge padding (buffer bins) for soft binning.

The soft kernel spreads every pair with a Gaussian softmax over the bin centres,
but only pairs inside ``[edges[0], edges[-1])`` are traversed.  The infinite
Gaussian tail therefore leaves the grid unaccounted for, which biases the last
band high (and the first low).  ``soft_pad=z`` extends the counting range by
``z`` kernel widths with *buffer bins* and drops them from the result, exactly
as ``pyfcfc`` does.

Most of these tests compare the padded call against the same call on a manually
extended grid, which is the definition of the feature.  The unpadded estimator
is pinned by ``soft_pad=0`` everywhere it is used as a reference; the rest of
the soft suite does the same.
"""

import numpy as np
import pytest

from cucount.numpy import (BinAttrs, MeshAttrs, Particles, cpulib,
                           count2, count2_soft, count2_soft_vjp)
from cucount.numpy.soft import soft_mu_weights, soft_weights
from cucount.softpad import (SOFT_PAD_DEFAULT, pad_cotangent, plan_padding,
                             validate_soft_pad)
from cucount.types import count2_soft as wrap_soft

needs_cpu = pytest.mark.skipif(cpulib is None, reason='CPU extension not built')

BOX = 500.0
EDGES = np.linspace(1.0, 100.0, 21)
LOG_EDGES = np.geomspace(1.0, 100.0, 21)
ZERO_EDGES = np.linspace(0.0, 50.0, 16)
ARB_EDGES = np.array([0.5, 0.8, 1.3, 2.0, 3.3, 5.5, 9.0, 15.0, 25.0, 40.0, 70.0])


def _setup(n=120, seed=7, edges=EDGES, weights=True):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, BOX, (n, 3))
    p = Particles(pos, rng.uniform(0.5, 1.5, n) if weights else None)
    ba = BinAttrs(s=edges)
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    return p, ba, ma


def _extended(p, edges, nlo, nhi):
    """A mesh/binning pair covering the manually extended grid."""
    ba = BinAttrs(s=edges)
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    return ba, ma


def _padded(p, battrs, mattrs, requested, **kwargs):
    return count2_soft(p, p, battrs=battrs, mattrs=mattrs, soft_pad=requested,
                       soft_accum='float64', backend='cpu', **kwargs)


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------
def test_soft_pad_default_is_two_kernel_widths():
    assert SOFT_PAD_DEFAULT == 2.0
    assert validate_soft_pad(None) == 2.0
    assert validate_soft_pad(0) == 0.0
    assert validate_soft_pad(np.float64(1.5)) == 1.5


def test_plan_padding_zero_is_identity():
    edges, nlo, nhi, sigma = plan_padding(EDGES, 0.0)
    np.testing.assert_array_equal(edges, EDGES)
    assert (nlo, nhi, sigma) == (0, 0, None)


def test_plan_padding_linear_continues_arithmetically():
    # A lower edge below one bin from zero cannot grow a buffer bin without
    # crossing s = 0, so only the upper edge extends here.
    edges, nlo, nhi, sigma = plan_padding(EDGES, 2.0, None, False)
    # The default sigma is pinned to the *original* grid (it is the same on
    # the arithmetic extension, but the plan reports it explicitly).
    assert sigma == pytest.approx(0.5 * np.diff(EDGES).mean())
    assert nlo == 0 and nhi == 1
    ext = np.asarray(edges, dtype=np.float64)
    np.testing.assert_allclose(np.diff(ext), np.diff(EDGES)[0], rtol=1e-14)
    assert ext[-1] == pytest.approx(EDGES[-1] + np.diff(EDGES)[-1])

    # Starting further from zero leaves room for a lower buffer too.
    lower = np.linspace(10.0, 20.0, 11)
    ext, nlo, nhi, sigma = plan_padding(lower, 2.0, None, False)
    assert (nlo, nhi) == (1, 1)
    assert sigma == pytest.approx(0.5)
    np.testing.assert_allclose(np.diff(ext), 1.0, rtol=1e-14)


def test_plan_padding_log_continues_geometrically():
    edges, nlo, nhi, sigma = plan_padding(LOG_EDGES, 2.0, None, False)
    # A log grid defaults to a per-bin width, so the extended grid reproduces
    # the default by itself and needs no explicit sigma.
    assert sigma is None
    assert nlo == 2 and nhi == 1
    ext = np.asarray(edges, dtype=np.float64)
    np.testing.assert_allclose(np.diff(np.log(ext)), np.log(LOG_EDGES[1] / LOG_EDGES[0]),
                               rtol=1e-14)
    assert ext[0] < LOG_EDGES[0] and ext[-1] > LOG_EDGES[-1]


def test_plan_padding_stays_above_zero():
    edges = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    ext, nlo, nhi, _ = plan_padding(edges, 10.0, None, False)
    assert ext[0] >= 0.0
    # The lower extension can never reach below zero.
    assert nlo == int(np.floor(edges[0] / (edges[1] - edges[0])))


def test_plan_padding_respects_max_edge():
    # The periodic minimum-image limit truncates (or removes) the upper buffer.
    _, _, nhi, _ = plan_padding(EDGES, 2.0, None, False, max_edge=102.0)
    assert nhi == 0
    _, _, nhi, _ = plan_padding(EDGES, 2.0, None, False, max_edge=250.0)
    assert nhi == 1


def test_plan_padding_scale_free_grid():
    # soft_sigma_log: the width is a fraction of s, so the plan is scale free.
    ext, nlo, nhi, sigma = plan_padding(np.geomspace(1.0, 2.0, 5), 2.0, None, True)
    # f = 0.5 dln s / 4, so two sigma is exactly one log bin on each side.
    assert nlo == 1 and nhi == 1
    assert sigma is not None
    with pytest.raises(ValueError, match='positive'):
        plan_padding(np.array([0.0, 1.0, 2.0]), 2.0, None, True)


def test_plan_padding_validates_request():
    with pytest.raises(TypeError):
        validate_soft_pad(True)
    with pytest.raises(TypeError):
        validate_soft_pad('wide')
    with pytest.raises(ValueError):
        validate_soft_pad(-1.0)
    with pytest.raises(ValueError):
        validate_soft_pad(np.inf)


# --------------------------------------------------------------------------
# counts
# --------------------------------------------------------------------------
@needs_cpu
@pytest.mark.parametrize('edges', [EDGES, LOG_EDGES, ARB_EDGES])
@pytest.mark.parametrize('sigma', [None, 1.5])
def test_padded_counts_match_the_manually_extended_grid(edges, sigma):
    p, ba, ma = _setup(edges=edges)
    got = _padded(p, ba, ma, 2.0, soft_sigma=sigma)
    meta = got['soft']
    nlo, nhi = meta['pad_lo'], meta['pad_hi']
    ext = np.asarray(meta['s_edges_ext'], dtype=np.float64)
    ns = edges.size - 1
    assert meta['nbins'] == ns and meta['nbins_padded'] == ext.size - 1
    assert ext.size - 1 == ns + nlo + nhi

    bae, mae = _extended(p, ext, nlo, nhi)
    want = count2_soft(p, p, battrs=bae, mattrs=mae, soft_pad=0.0,
                       soft_sigma=sigma, soft_accum='float64', backend='cpu')['weight']
    np.testing.assert_allclose(got['weight'], want[nlo:nlo + ns], rtol=1e-12, atol=0.0)


@needs_cpu
def test_padded_metadata_reports_the_buffer_layout():
    edges = np.linspace(10.0, 20.0, 11)
    p, ba, ma = _setup(edges=edges)
    got = _padded(p, ba, ma, None)
    meta = got['soft']
    assert meta['soft_pad'] == SOFT_PAD_DEFAULT
    assert meta['pad_lo'] == 1 and meta['pad_hi'] == 1
    assert got['weight'].shape == (edges.size - 1,)
    # The extended edges are exposed for the lsstypes self-pair profile.
    assert meta['s_edges_ext'][0] < edges[0]
    assert meta['s_edges_ext'][-1] > edges[-1]

    unpadded = count2_soft(p, p, battrs=ba, mattrs=ma, soft_pad=0.0,
                           soft_accum='float64', backend='cpu')
    assert unpadded['soft']['soft_pad'] == 0.0
    assert 's_edges_ext' not in unpadded['soft']


@needs_cpu
def test_padding_counts_pairs_beyond_the_original_edges():
    """The buffer shell enters the histogram even when no real bin can hold it."""
    p1 = Particles(np.array([[0.0, 0.0, 0.0]]), np.array([1.0]))
    p2 = Particles(np.array([[103.0, 0.0, 0.0]]), np.array([1.0]))
    ba = BinAttrs(s=np.array([1.0, 50.0, 100.0]))
    ma = MeshAttrs(p1, p2, boxsize=BOX, battrs=ba, periodic=True)

    hard = count2(p1, p2, battrs=ba, mattrs=ma, backend='cpu')['weight']
    np.testing.assert_array_equal(hard, 0.0)
    unpadded = count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_pad=0.0,
                           soft_sigma=1.0, soft_accum='float64', backend='cpu')['weight']
    np.testing.assert_array_equal(unpadded, 0.0)

    padded = count2_soft(p1, p2, battrs=ba, mattrs=ma, soft_pad=5.0,
                         soft_sigma=20.0, soft_accum='float64', backend='cpu')['weight']
    # s=103 is outside the original grid but inside the padded window, so the
    # buffer bins see it and the outermost real bin receives part of its weight.
    assert 0.0 < padded[-1] < 1.0
    ext = np.asarray(count2_soft(
        p1, p2, battrs=ba, mattrs=ma, soft_pad=5.0, soft_sigma=20.0,
        soft_accum='float64', backend='cpu')['soft']['s_edges_ext'])
    ns = ba.edges('s').shape[0]
    want = soft_weights(103.0, ext, sigma=20.0)[0][ns - 1]
    assert padded[-1] == pytest.approx(want, rel=1e-12)


@needs_cpu
def test_padded_soft_counts_track_hard_binning_at_the_upper_edge():
    """Hard binning is exact; padding should remove the soft edge deficit.

    A uniform catalogue makes the hard counts a smooth reference.  With a
    kernel close to one bin wide, the unpadded soft estimator loses a
    noticeable fraction of the last bin; padding restores the hard trend.
    """
    rng = np.random.default_rng(11)
    p = Particles(rng.uniform(0.0, BOX, (2000, 3)))
    edges = np.linspace(50.0, 150.0, 21)
    ba = BinAttrs(s=edges)
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)

    hard = count2(p, p, battrs=ba, mattrs=ma, backend='cpu')['weight']
    kwargs = dict(soft_sigma=4.0, soft_accum='float64', backend='cpu')
    unpadded = count2_soft(p, p, battrs=ba, mattrs=ma, soft_pad=0.0,
                           **kwargs)['weight']
    padded = count2_soft(p, p, battrs=ba, mattrs=ma, **kwargs)['weight']

    dev_unpadded = unpadded[-1] / hard[-1] - 1.0
    dev_padded = padded[-1] / hard[-1] - 1.0
    assert dev_unpadded < -0.05
    assert abs(dev_padded) < 0.03
    assert abs(dev_padded) < 0.3 * abs(dev_unpadded)


@needs_cpu
def test_padded_autocorrelation_removes_the_self_pair_on_the_padded_profile():
    """The self pair at s=0 deposits on the extended grid, then is sliced."""
    p, ba, ma = _setup(edges=ZERO_EDGES, weights=True)
    got = _padded(p, ba, ma, 2.0)
    meta = got['soft']
    nlo = meta['pad_lo']
    ext = np.asarray(meta['s_edges_ext'], dtype=np.float64)
    ns = ZERO_EDGES.size - 1

    bae, mae = _extended(p, ext, nlo, meta['pad_hi'])
    raw = count2_soft(p, p, battrs=bae, mattrs=mae, soft_pad=0.0,
                      soft_accum='float64', backend='cpu')['weight']
    np.testing.assert_allclose(got['weight'], raw[nlo:nlo + ns],
                               rtol=1e-12, atol=0.0)
    # The raw frontend keeps the self pairs; the lsstypes wrapper removes them
    # on the padded profile (covered by the wrapper test below).
    assert got['weight'][0] > 0.0


@needs_cpu
def test_padded_poles_match_the_manually_extended_grid():
    p, _, _ = _setup(edges=ZERO_EDGES)
    ells = [0, 2, 4]
    ba = BinAttrs(s=ZERO_EDGES, pole=(ells, 'z'))
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    got = _padded(p, ba, ma, 2.0, soft_mu_sigma=0.05, soft_mu_bins=24)
    meta = got['soft']
    nlo, nhi = meta['pad_lo'], meta['pad_hi']
    ext = np.asarray(meta['s_edges_ext'], dtype=np.float64)
    ns = ZERO_EDGES.size - 1

    bae = BinAttrs(s=ext, pole=(ells, 'z'))
    mae = MeshAttrs(p, p, boxsize=BOX, battrs=bae, periodic=True)
    raw = count2_soft(p, p, battrs=bae, mattrs=mae, soft_pad=0.0,
                      soft_mu_sigma=0.05, soft_mu_bins=24, soft_accum='float64',
                      backend='cpu')['weight']
    np.testing.assert_allclose(got['weight'], raw[nlo:nlo + ns],
                               rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------------
# adjoint
# --------------------------------------------------------------------------
@needs_cpu
@pytest.mark.parametrize('edges', [EDGES, LOG_EDGES, ARB_EDGES])
def test_padded_adjoint_matches_the_manually_extended_grid(edges):
    p, ba, ma = _setup(edges=edges)
    rng = np.random.default_rng(3)
    cot = rng.normal(size=edges.size - 1)

    got = count2_soft_vjp(p, p, battrs=ba, mattrs=ma, cotangent=cot,
                          soft_pad=2.0, soft_accum='float64', backend='cpu')
    meta = got['soft']
    nlo, nhi = meta['pad_lo'], meta['pad_hi']
    ext = np.asarray(meta['s_edges_ext'], dtype=np.float64)
    ns = edges.size - 1

    bae, mae = _extended(p, ext, nlo, nhi)
    want = count2_soft_vjp(p, p, battrs=bae, mattrs=mae,
                           cotangent=pad_cotangent(cot, nlo, ext.size - 1),
                           soft_pad=0.0, soft_accum='float64', backend='cpu')
    for key in ('grad_position1', 'grad_position2', 'grad_weight1'):
        np.testing.assert_allclose(got[key], want[key], rtol=1e-12, atol=0.0)


@needs_cpu
def test_padded_adjoint_rejects_a_padded_cotangent():
    p, ba, ma = _setup()
    got = _padded(p, ba, ma, 2.0)
    n_ext = got['soft']['nbins_padded']
    with pytest.raises(ValueError, match='real separation bin'):
        count2_soft_vjp(p, p, battrs=ba, mattrs=ma, cotangent=np.ones(n_ext),
                        soft_pad=2.0, soft_accum='float64', backend='cpu')


# --------------------------------------------------------------------------
# lsstypes wrapper
# --------------------------------------------------------------------------
@needs_cpu
def test_wrapper_slices_the_padded_self_pair_profile():
    p, _, _ = _setup(edges=ZERO_EDGES, weights=True)
    ba = BinAttrs(s=ZERO_EDGES)
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    DD = wrap_soft(p, battrs=ba, mattrs=ma, soft_pad=2.0,
                   soft_accum='float64')['weight']
    meta = DD.attrs
    assert meta['soft_pad'] == 2.0
    assert meta['soft_pad'] > 0
    assert meta['soft_nbins_padded'] > ZERO_EDGES.size - 1
    assert meta['soft_pad_lo'] == 0 and meta['soft_pad_hi'] > 0
    assert DD.values('counts').shape == (ZERO_EDGES.size - 1,)

    raw = _padded(p, ba, ma, 2.0)
    nlo = raw['soft']['pad_lo']
    ns = ZERO_EDGES.size - 1
    w = np.asarray(p.get('weights')[0])
    ext = np.asarray(raw['soft']['s_edges_ext'], dtype=np.float64)
    zero = soft_weights(0.0, ext)[0][nlo:nlo + ns]
    bae, mae = _extended(p, ext, nlo, raw['soft']['pad_hi'])
    want = count2_soft(p, p, battrs=bae, mattrs=mae, soft_pad=0.0,
                       soft_accum='float64', backend='cpu')['weight'][nlo:nlo + ns]
    want = want - (w ** 2).sum() * zero
    np.testing.assert_allclose(np.asarray(DD.values('counts')), want,
                               rtol=1e-12, atol=0.0)


@needs_cpu
def test_wrapper_padding_is_opt_out():
    p, _, _ = _setup()
    ba = BinAttrs(s=EDGES)
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    DD = wrap_soft(p, battrs=ba, mattrs=ma, soft_pad=0.0, soft_accum='float64')['weight']
    assert DD.attrs['soft_pad'] == 0.0
    assert DD.attrs['soft_pad_lo'] == 0
    assert DD.attrs['soft_pad_hi'] == 0
