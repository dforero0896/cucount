"""Soft (Gaussian) two-point pair counts.

This is an *extension* to :mod:`cucount.numpy`, not a replacement: imports the
hard-binning frontend and adds one entry point beside it. The hard ``count2``
is untouched -- no option, flag or branch was added to it -- so an existing
call cannot change meaning or speed because this module exists.

Every pair inside the separation range is spread over the separation bins with a
Gaussian kernel centred on the bin centres,

    x_i     = (s - c_i)^2 / (2 sigma_i^2),   c_i = (edges[i] + edges[i+1]) / 2
    W_i     = A_i exp(-x_i)                  soft_norm='gaussian' (default)
    W_i     = exp(-x_i) / sum_j exp(-x_j)    soft_norm='softmax' (legacy)
    counts += w_pair * W_i

with the discrete-quadrature prefactor ``A_i = dmui / (sqrt(2 pi) sigma_i)``.
The gaussian mode is a single pass and keeps each bin's weight independent of
its neighbours (no competing normalisation), at the cost of a per-pair row sum
of one only up to quadrature error (about 0.5% at the default
``sigma = dmu/2``), plus whatever kernel tail falls outside the counted range
(``soft_pad`` recovers that part). The softmax mode sums the window and renormalises, so a
pair's weight is exactly conserved. In both modes ``sigma -> 0`` recovers hard
binning.

``sigma=None`` means "half a bin", the oracle's default, and on a non-uniform
grid it means half of *each* bin, ``sigma_i = 0.5 * (edges[i+1] - edges[i])``.
That keeps the smoothing proportional to the local resolution (a log grid gets
constant relative smoothing instead of over-smoothing its narrow bins) and makes
sigma a property of the binning rather than a free parameter. Passing a number
instead uses that absolute width in separation units for every bin.

Both uniform and arbitrary monotone edges are served: the uniform grid keeps a
multiplicative recurrence (two ``exp`` per pair), any other grid uses a binary
search for the peak plus one ``exp`` per tap.

With a ``pole`` axis the ``mu`` axis is softened as well and the result is
soft multipoles: a Gaussian kernel is applied to a ``mu`` grid, and each pair
deposits ``w_pair * Ws_i * M_l`` where

    M_l(mu) = sum_j Wmu_j (2l+1) P_l(mu_j),
    Wmu_j   = exp(-(mu - mu_j)^2 / (2 sigma_mu^2)) / sum_k exp(...)

so the mu weights never touch the histogram (that is what would make them
expensive): the multipole profile is computed once per pair and multiplied by
the separation weights, which keeps the cost at ``taps_s * nell``.

The mu axis is renormalised whatever ``soft_norm`` selects for the separation
axis. ``mu = s_parallel / s`` is confined to ``[-1, 1]``, so a kernel that
straddles a pole would otherwise lose up to half of its mass there (``M_0``
near 0.5 for a pair within a ``sigma_mu`` of ``|mu| = 1``), and padding is not
an option on a bounded domain. Renormalising is what makes smoothing a
distribution on that domain well defined: ``M_0 = 1`` exactly, so the monopole
*is* the isotropic soft-s count.

Differences from the oracle, both because the kernel is new rather than a port:

* the ``pi`` axis is not softened (soft ``s`` and ``mu`` are);
* the CUDA kernel scatters through a shared-memory histogram tile rather than
  the oracle's scalar loop, and the CPU kernel visits every pair once per
  OpenMP thread into a thread-private histogram (double, reduced at the end);
* weights outside ``soft_cutoff`` of the peak are dropped. In softmax mode the
  window is then renormalised, as the oracle does; in gaussian mode the dropped
  tail is a small quadrature error. Its default cutoff is 1e-13, which is what
  makes the tap count (and so the cost) ~9 taps per pair at the default sigma.
"""
import numpy as np

from ..softpad import (pad_cotangent, plan_padding,
                       slice_soft_result, validate_soft_pad)
from . import (
    BinAttrs,
    MeshAttrs,
    Particles,
    SelectionAttrs,
    SplitAttrs,
    WeightAttrs,
    _check_count2,
    _dispatch,
    _lib_and_nthreads,
    _log_timings,
    _resolve_backend,
    _resolve_tuning,
    _setup_cucount_logging,
    _to_c_particles,
)


def _bin_centres(edges):
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError('soft binning needs a 1D array of at least two edges')
    return 0.5 * (edges[1:] + edges[:-1])


def default_sigma(edges):
    """Half the mean bin width, which is the oracle's default (``sigma = 0``).

    Only meaningful for a uniform grid; on a non-uniform one the kernel uses a
    per-bin sigma (see :func:`sigmas`).
    """
    edges = np.asarray(edges, dtype=np.float64)
    return 0.5 * float(edges[-1] - edges[0]) / (edges.size - 1)


def is_uniform(edges):
    """True when the edges are an arithmetic progression (to 1e-9 relative)."""
    edges = np.asarray(edges, dtype=np.float64)
    ds = edges[1] - edges[0]
    want = edges[0] + np.arange(edges.size) * ds
    return bool(np.all(np.abs(edges - want) <= 1e-9 * abs(ds)))


def sigmas(edges, sigma=None):
    """Per-bin kernel widths the kernel will use: ``sigma_i``, shape ``(nbins,)``.

    ``None`` gives half of each bin's width; a number is broadcast to every bin.
    """
    edges = np.asarray(edges, dtype=np.float64)
    widths = np.diff(edges)
    if sigma is None:
        return 0.5 * widths
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f'sigma must be positive and finite, got {sigma!r}')
    return np.full(widths.shape, sigma, dtype=np.float64)


def _s_edges(battrs):
    """The separation axis of a BinAttrs as one monotone edge array."""
    intervals = np.asarray(battrs.edges('s'), dtype=np.float64)
    return np.concatenate([intervals[:, 0], intervals[-1:, 1]])


def _battrs_with_edges(battrs, edges):
    """A copy of ``battrs`` with the separation edges replaced."""
    names = list(battrs.varnames)
    if names == ['s']:
        return BinAttrs(s=np.asarray(edges, dtype=np.float64))
    if names == ['s', 'pole']:
        ells = np.asarray(battrs.coords('pole'), dtype=np.int64)
        los = str(battrs.losnames[names.index('pole')])
        return BinAttrs(s=np.asarray(edges, dtype=np.float64), pole=(ells, los))
    raise NotImplementedError(
        'soft binning needs BinAttrs(s=edges) or BinAttrs(s=edges, pole=(ells, los))')


def _pad_setup(particles, battrs, mattrs, sattrs, mode, soft_pad, soft_sigma):
    """Plan the buffer bins and make the mesh reach them.

    Returns ``(battrs_ext, edges_ext, edges_real, nlo, nhi, pad, mattrs,
    soft_sigma)``.  The mesh is rebuilt only when the caller's mesh cannot see
    the padded range, so an explicit mesh is otherwise left alone.
    """
    edges_real = _s_edges(battrs)
    pad = validate_soft_pad(soft_pad)
    if mattrs is None:
        mattrs = MeshAttrs(*particles, sattrs=sattrs, battrs=battrs, backend=mode)
    max_edge = None
    if getattr(mattrs, 'periodic', False):
        max_edge = 0.5 * float(np.min(np.asarray(mattrs.boxsize))) * (1.0 - 1e-9)
    edges_ext, nlo, nhi, sig_pin = plan_padding(
        edges_real, pad, soft_sigma, False, max_edge)
    if nlo == 0 and nhi == 0:
        return battrs, edges_real, edges_real, 0, 0, pad, mattrs, soft_sigma

    battrs_ext = _battrs_with_edges(battrs, edges_ext)
    want = float(edges_ext[-1]) * (1.0 - 1e-9)
    if float(np.asarray(mattrs.smax)) < want:
        mattrs = MeshAttrs(*particles, boxsize=mattrs.boxsize,
                           boxcenter=mattrs.boxcenter,
                           periodic=mattrs.periodic, sattrs=sattrs,
                           battrs=battrs_ext, backend=mode)
    if sig_pin is not None and soft_sigma is None:
        soft_sigma = float(sig_pin)
    return battrs_ext, edges_ext, edges_real, nlo, nhi, pad, mattrs, soft_sigma


def mu_spec(soft_mu_sigma=None, soft_mu_bins=0, soft_mu_edges=None,
            soft_cutoff=1e-13):
    """Resolve the mu quadrature the kernel will use.

    Returns a dict with ``centres`` (midpoints), ``edges`` (``nmu + 1``),
    ``sigmas`` (per-cell width), ``sigma`` (the one absolute width when a single
    value applies, else 0), ``uniform`` (arithmetic grid, i.e. the fast
    recurrence path), ``dmu`` (cell width for a uniform grid, else ``None``),
    ``nmu`` and ``kmax``.

    ``soft_mu_edges`` are the caller's own cell edges. They must be strictly
    increasing and lie in ``[-1, 1]``; mu is **not** folded, so a grid that only
    covers part of the interval leaves the pairs beyond it to the boundary cell,
    which is what the kernel does too. Arithmetic edges are equivalent to the
    internal grid and still take the recurrence path.
    """
    if not np.isfinite(soft_cutoff) or not 0.0 < float(soft_cutoff) < 1.0:
        raise ValueError(f'soft_cutoff must be in (0, 1), got {soft_cutoff!r}')
    cutoff = float(soft_cutoff)
    sigma = 0.0 if soft_mu_sigma is None else float(soft_mu_sigma)

    if not np.isfinite(sigma):
        raise ValueError(f'soft_mu_sigma must be finite, got {soft_mu_sigma!r}')

    if soft_mu_edges is not None:
        if soft_mu_bins and int(soft_mu_bins) > 0:
            raise ValueError('give soft_mu_edges or soft_mu_bins, not both')
        edges = np.asarray(soft_mu_edges, dtype=np.float64)
        if edges.ndim != 1:
            raise ValueError('soft_mu_edges must be a 1D array of cell edges')
        if edges.size < 2:
            raise ValueError('soft_binning needs at least two mu edges')
        if edges.size - 1 > 1024:
            raise ValueError(f'at most 1024 mu bins are served, got {edges.size - 1}')
        if not np.all(np.isfinite(edges)):
            raise ValueError('mu edges must be finite')
        if edges[0] < -1.0 - 1e-9 or edges[-1] > 1.0 + 1e-9:
            raise ValueError(
                f'mu edges must lie in [-1, 1] (mu is not folded); got '
                f'[{edges[0]}, {edges[-1]}]')
        if not np.all(np.diff(edges) > 0):
            raise ValueError('mu edges must be strictly increasing')
        uniform = is_uniform(edges)
        dmu = float(edges[1] - edges[0]) if uniform else None
    else:
        if soft_mu_bins and int(soft_mu_bins) > 0:
            nmu = int(soft_mu_bins)
        elif sigma > 0.0:
            nmu = int(np.ceil(4.0 / sigma))
        else:
            nmu = 40
        nmu = min(max(nmu, 4), 1024)
        dmu = 2.0 / nmu
        edges = np.linspace(-1.0, 1.0, nmu + 1)
        uniform = True

    centres = 0.5 * (edges[1:] + edges[:-1])
    widths = np.diff(edges)

    if sigma > 0.0:
        sigmas = np.full(centres.shape, sigma, dtype=np.float64)
        sigma_out = sigma
    else:
        # Half of each cell, the mu analogue of the s default.
        sigmas = 0.5 * widths
        sigma_out = float(sigmas[0]) if uniform else 0.0

    if uniform:
        dmu = float(widths[0])
        kmax = int(np.ceil(np.sqrt(2.0 * np.log(1.0 / cutoff)) * sigma_out / dmu))
        kmax = min(max(kmax, 1), centres.size - 1)
    else:
        kmax = centres.size - 1  # adaptive: the kernel stops on the cutoff

    return dict(centres=centres, edges=edges, sigmas=sigmas, sigma=sigma_out,
                uniform=bool(uniform), dmu=dmu, nmu=int(centres.size),
                kmax=int(kmax))


def mu_grid(soft_mu_sigma=None, soft_mu_bins=0, soft_mu_edges=None):
    """The mu quadrature the kernel uses: ``(centres, sigma_mu, dmu)``.

    ``sigma_mu`` is one scalar when an absolute width applies (or the grid is
    arithmetic, where every cell has the same width), and a per-cell array on a
    non-arithmetic grid, in which case ``dmu`` is ``None``. See :func:`mu_spec`
    for the full description.
    """
    spec = mu_spec(soft_mu_sigma, soft_mu_bins, soft_mu_edges)
    sigma = spec['sigma'] if spec['uniform'] else spec['sigmas']
    return spec['centres'], sigma, spec['dmu']


def mu_kmax(soft_mu_sigma, soft_mu_bins, soft_cutoff=1e-13, soft_mu_edges=None):
    """Half-width of the truncated mu window, in mu cells (``nmu - 1`` when the
    grid is not arithmetic, because then the window adapts to the cutoff)."""
    return mu_spec(soft_mu_sigma, soft_mu_bins, soft_mu_edges, soft_cutoff)['kmax']


def soft_mu_weights(mu, ells, soft_mu_sigma=None, soft_mu_bins=0,
                    soft_cutoff=1e-13, soft_mu_edges=None):
    """Reference multipole profile, shape ``(len(mu), len(ells))``.

    ``M_l(mu)`` for the requested ``ells`` on the kernel's truncated mu window.
    The cell weights are renormalised per mu, so ``M_0 == 1`` exactly and the
    monopole is the isotropic soft count.

    The mu axis is renormalised whatever ``soft_norm`` says about the
    separation axis: ``mu = s_parallel / s`` is confined to ``[-1, 1]``, so an
    un-renormalised kernel straddling a pole would silently lose up to half of
    its mass there. This is the definition the kernel implements; it exists so
    tests and plots do not have to re-derive it. mu is used unfolded: a
    negative mu stays negative.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    ells = np.atleast_1d(np.asarray(ells, dtype=np.int64))
    spec = mu_spec(soft_mu_sigma, soft_mu_bins, soft_mu_edges, soft_cutoff)
    centres = spec['centres']
    sigmas = spec['sigmas']
    widths = np.diff(spec['edges'])
    nmu = centres.size

    # P_l at every centre, (nmu, nell), for the orders that were asked for.
    leg = np.polynomial.legendre.legvander(centres, int(max(ells.max(), 1)))
    leg = leg[:, ells]
    pf = (2 * ells + 1).astype(np.float64)

    out = np.zeros((mu.size, ells.size), dtype=np.float64)

    if spec['uniform']:
        kmax = spec['kmax']
        c0, dmu = centres[0], spec['dmu']

        for i, m_ in enumerate(mu):
            peak = int(np.floor((m_ - c0) / dmu + 0.5))
            peak = min(max(peak, 0), nmu - 1)
            lo = max(peak - kmax, 0)
            hi = min(peak + kmax, nmu - 1)
            x = (m_ - centres[lo:hi + 1]) ** 2 / (2.0 * sigmas[lo:hi + 1] ** 2)
            w = _kernel_weights(x, widths[lo:hi + 1], sigmas[lo:hi + 1],
                                'softmax')
            out[i] = w @ (pf * leg[lo:hi + 1])
    else:
        # Full-grid stable softmax. The kernel truncates the same weights at the
        # cutoff, which drops less than `cutoff` of the normalisation.
        for i, m_ in enumerate(mu):
            x = (m_ - centres) ** 2 / (2.0 * sigmas ** 2)
            w = _kernel_weights(x, widths, sigmas, 'softmax')
            out[i] = w @ (pf * leg)

    return out


def _kernel_weights(x, widths, sigmas, soft_norm):
    """Cell weights for logits ``x``, mirroring the kernel's normalisation.

    ``'gaussian'`` is the discrete-quadrature mass ``width / (sqrt(2 pi) sigma)
    exp(-x)``; ``'softmax'`` is the legacy per-window renormalisation, which is
    what the kernel used before the option existed.
    """
    e = np.exp(-x)

    if soft_norm == 'softmax':
        # Normalise each row: one row is one pair, so the sum has to run over
        # the bins of that pair and not over the whole (pairs x bins) array.
        e /= e.sum(axis=-1, keepdims=True)

        return e

    if soft_norm != 'gaussian':
        raise ValueError(
            f"soft_norm must be 'gaussian' or 'softmax', got {soft_norm!r}")

    return e * widths / (np.sqrt(2.0 * np.pi) * sigmas)


def soft_weights(s, edges, sigma=None, soft_norm='gaussian'):
    """Reference weights, shape ``(len(s), len(edges) - 1)``.

    Full-grid weights over the bin centres -- no window truncation, so this is
    the definition rather than the kernel's truncated version; the two agree to
    the cutoff tail. Follows the kernel's sigma rule: ``None`` means half of
    each bin's width, a number means one absolute width.

    ``soft_norm='gaussian'`` (the default) is the discrete-quadrature mass
    ``w_i = dmui / (sqrt(2 pi) sigma_i) exp(-x_i)``, whose per-pair sum is 1 up
    to quadrature error; ``'softmax'`` renormalises each row exactly.
    """
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    edges = np.asarray(edges, dtype=np.float64)
    centres = _bin_centres(edges)
    sg = sigmas(edges, sigma)
    x = (s[:, None] - centres[None, :]) ** 2 / (2.0 * sg[None, :] ** 2)
    widths = np.diff(edges)

    if soft_norm == 'softmax':
        x -= x.min(axis=1, keepdims=True)

    return _kernel_weights(x, widths, sg, soft_norm)


def count2_soft(*particles: Particles, battrs: BinAttrs, soft_sigma: float=None,
                soft_cutoff: float=1e-13, soft_taps: str='exact',
                soft_accum: str='float32', soft_pad: float=None,
                soft_norm: str='gaussian',
                soft_mu_sigma: float=None, soft_mu_bins: int=0,
                soft_mu_edges=None,
                wattrs: WeightAttrs=None,
                sattrs: SelectionAttrs=None, spattrs: SplitAttrs=None,
                mattrs: MeshAttrs=None, nthreads: int=1, backend: str=None,
                tuning: dict=None):
    """Two-point pair counts with Gaussian soft binning on ``s``.

    The kernel assigns each pair a Gaussian weight on the separation axis. Two
    normalisations are available; ``soft_norm='gaussian'`` (the default) is the
    discrete-quadrature mass ``w_i = dmui / (sqrt(2 pi) sigma_i) exp(-x_i)``,
    which needs a single pass and treats every bin independently, so the
    gradients stay well conditioned; ``soft_norm='softmax'`` renormalises the
    window so the weights sum to exactly 1, which costs a second pass.

    Parameters
    ----------
    *particles : Particles
        Exactly two catalogs, as for :func:`cucount.numpy.count2`.
    battrs : BinAttrs
        Must be the single separation axis: ``BinAttrs(s=edges)``. Any strictly
        increasing edges are served. A uniform grid (``np.arange``, a
        ``(min, max, step)`` tuple, or ``BIN_LIN``) hits the fast recurrence
        kernel; anything else uses the general path, which is exact but pays one
        ``exp`` per tap instead of two per pair.
    soft_sigma : float, optional
        Kernel width in separation units. ``None`` (or 0) means half a bin; on a
        non-uniform grid that is half of each bin, i.e. ``sigma_i`` tracks the
        local width.
    soft_cutoff : float, optional
        Relative weight below which the Gaussian tail is dropped. With
        ``soft_norm='softmax'`` the window is renormalised, so weight stays
        conserved exactly; with ``'gaussian'`` the dropped tail is simply a
        small quadrature error. This is the speed knob: the tap count grows as
        ``sqrt(2 ln(1 / cutoff))``. Default 1e-13 matches the oracle.
    soft_taps : {'exact', 'fast'}
        Only affects the general (non-uniform) path. ``'exact'`` (default) uses
        double ``exp``; ``'fast'`` uses ``__expf``, trading ~1e-6 relative
        accuracy in the weights for roughly half the transcendental cost.
    soft_accum : {'float32', 'float64'}
        Precision of the shared-memory per-block histograms. ``'float32'``
        (default) is 1.5-3x faster than ``'float64'`` on Ampere and later,
        because ``atomicAdd(double*)`` in shared memory is a CAS retry loop;
        the tap math, the window search and the pair selection stay double, so
        the only cost is rounding when each block adds its partial sum to the
        output histogram (a few 1e-7 relative). Use ``'float64'`` to recover the
        previous bit-comparable path, e.g. for conservation tests at 1e-15.
    soft_norm : {'gaussian', 'softmax'}
        How the Gaussian weights are normalised. ``'gaussian'`` (default) uses
        the discrete-quadrature prefactor ``dmui / (sqrt(2 pi) sigma_i)``: one
        pass per pair, no per-pair sum, per-bin weights that do not compete, so
        gradients are less prone to collapsing onto one bin. The per-pair sum
        is then 1 only up to quadrature error (about 0.5% for the default
        ``sigma = dmu/2``), and the kernel tail outside the binning range is
        simply dropped (see ``soft_pad``). ``'softmax'`` is the legacy
        behaviour: weights are divided by the window sum (an extra pass), which
        conserves weight exactly. This selects the separation-axis
        normalisation only; the mu axis is always renormalised, because
        ``mu`` in ``[-1, 1]`` is a bounded domain where an un-renormalised
        kernel would lose mass at the poles.
    soft_pad : float, optional
        Edge padding in kernel widths. The traversal only counts pairs inside
        the binning range, so without padding the outermost bins average a
        *truncated* kernel and pick up an edge bias. With ``soft_pad = z`` the
        counting range is extended by ``z`` kernel widths with buffer bins
        (geometric continuation on a log grid, arithmetic otherwise); the kernel
        deposits over the real and buffer centres (the softmax also normalises
        over them), and the buffer bins are dropped from the returned counts. ``None`` selects
        the pyfcfc default, ``z = 2``; ``0`` restores the previous estimator
        bit for bit. In a periodic box the extension is clamped below ``L/2``.
        Padding costs the extra pairs in the buffer shells (roughly
        ``exp(3 z f) - 1`` at the upper edge for log bins).
    soft_mu_sigma : float, optional
        Width of the mu kernel, in mu units, for a request whose second axis is
        ``pole=(ells, los)``. ``None`` (or 0) means half a mu bin. The mu axis is
        a uniform grid on [-1, 1] unless ``soft_mu_edges`` is given;
        ``soft_mu_bins`` sets its resolution (default ``ceil(4 / sigma_mu)``, or
        40 for the default sigma).
    soft_mu_bins : int, optional
        Number of mu bins in the quadrature; 0 means "derive it".
    soft_mu_edges : array_like, optional
        The caller's own mu cell edges instead of the internal grid. Must be
        strictly increasing and lie in ``[-1, 1]`` (mu is not folded, so odd
        multipoles keep their sign). Arithmetic edges still take the fast
        recurrence path; any other grid gets one ``exp`` per tap and per-cell
        widths (half the cell unless ``soft_mu_sigma`` is given). A grid that
        covers part of the interval leaves the pairs beyond it to the boundary
        cell.
    wattrs, sattrs, spattrs, mattrs, nthreads, backend, tuning
        As for :func:`cucount.numpy.count2`. Both backends serve soft binning;
        both reject spin, angular upweighting, split/jackknife and any second
        binning axis other than ``pole`` rather than quietly ignoring them.
        ``nthreads`` (through ``tuning``) is GPUs on CUDA and threads on the CPU.

    Returns
    -------
    result : dict
        ``{'weight': array(nbins)}``, or ``array(nbins, nell)`` for soft
        multipoles, plus ``result['soft']`` describing what ran
        (``sigma`` after the default, ``sigma_kind`` = 'absolute' or 'width',
        ``uniform``, ``fast_taps``, ``accum`` = 'float32' or 'float64',
        ``cutoff``, ``kmax``, ``norm`` = 'gaussian' or 'softmax',
        ``taps`` (``-1`` when
        the window is adaptive), ``path`` = 'shared' or 'global', ``nbins``,
        and for multipoles ``multipoles``, ``mu_sigma``, ``mu_bins``,
        ``mu_kmax``, ``los``, ``mu_uniform``, ``mu_sigma_kind``, ``mu_taps``
        and, on a caller-supplied non-arithmetic grid, ``mu_edges``).
    """
    _setup_cucount_logging()
    assert len(particles) == 2
    if wattrs is None: wattrs = WeightAttrs()
    if sattrs is None: sattrs = SelectionAttrs()
    if spattrs is None: spattrs = SplitAttrs()
    wattrs.check(*particles)
    spattrs.check(*particles)
    # The kernel writes one histogram, not one per split; raising here keeps a
    # jackknife request from coming back as plausible-looking plain counts.
    if getattr(spattrs, 'nsplits', 0):
        raise NotImplementedError('split/jackknife counts are not served by the soft kernel yet')
    _check_count2(particles, battrs, wattrs, spattrs)

    mode = _resolve_backend(backend)
    cuda_tuning, cpu_tuning = _resolve_tuning(mode, tuning, nthreads=nthreads)

    (battrs_ext, edges_ext, edges_real, nlo, nhi, pad, mattrs,
     soft_sigma) = _pad_setup(particles, battrs, mattrs, sattrs, mode,
                              soft_pad, soft_sigma)
    ns_real = int(edges_real.size - 1)

    cparticles = [_to_c_particles(p) for p in particles]

    def run(which, tuning):
        lib, lib_nthreads = _lib_and_nthreads(which, tuning)
        result, timings = lib.count2_soft(
            *cparticles, mattrs._to_c(), battrs_ext,
            wattrs=wattrs._to_c(), sattrs=sattrs, spattrs=spattrs,
            soft_sigma=soft_sigma, soft_cutoff=float(soft_cutoff),
            soft_taps=str(soft_taps), soft_accum=str(soft_accum),
            soft_norm=str(soft_norm),
            soft_mu_sigma=soft_mu_sigma, soft_mu_bins=int(soft_mu_bins),
            soft_mu_edges=soft_mu_edges,
            nthreads=lib_nthreads, return_timings=True)
        _log_timings(which, *timings)
        return result

    result = _dispatch(mode, run, cuda_tuning, cpu_tuning)
    return slice_soft_result(result, nlo, nhi, ns_real, edges_ext, pad)


def count2_soft_vjp(*particles: Particles, battrs: BinAttrs, cotangent,
                    soft_sigma: float=None, soft_cutoff: float=1e-13,
                    soft_taps: str='exact', soft_accum: str='float32',
                    soft_pad: float=None,
                    soft_norm: str='gaussian',
                    soft_mu_sigma: float=None,
                    soft_mu_bins: int=0, soft_mu_edges=None,
                    wattrs: WeightAttrs=None, sattrs: SelectionAttrs=None,
                    spattrs: SplitAttrs=None, mattrs: MeshAttrs=None,
                    nthreads: int=1, backend: str=None, tuning: dict=None):
    """Vector-Jacobian product of :func:`count2_soft`.

    Given ``g = dL/dcounts`` for the soft histogram, returns the gradients of
    ``L`` with respect to the two catalogues' particle positions and individual
    weights, and with respect to ``soft_sigma``. This is what makes the soft
    counts differentiable end to end: the jax layer wraps the forward call in a
    ``custom_vjp`` and hands the incoming cotangent to this kernel, so no
    scatter is differentiated by autodiff and the whole backward pass is one
    more kernel.

    The kernel is the analytic adjoint of the forward (the same window, the same
    normalisation, differentiated exactly); for the derivation and the
    conventions --
    window membership and the ``[smin, smax)`` range check held constant -- see
    ``cuda/include/count2_soft.h``. This is the same adjoint pyfcfc exposes as
    ``soft_hist_vjp``.

    Parameters
    ----------
    *particles : Particles
        The two catalogs, exactly as passed to :func:`count2_soft` (same
        ``battrs``, ``soft_sigma``, weighting, selections and mesh, or the
        adjoint is of a different function than the forward you ran).
    battrs, soft_sigma, soft_cutoff, soft_taps, soft_norm, wattrs, sattrs, spattrs, mattrs
        As for :func:`count2_soft`; they must reproduce the forward's setup.
    soft_mu_sigma, soft_mu_bins
        As for :func:`count2_soft`. With soft mu binning the mu weights also
        contribute to the position gradients (through ``dmu/dposition``), so the
        adjoint is the derivative of the *multipoles*, not of a frozen mu.
    cotangent : array_like
        One entry per *real* histogram slot, ``dL/dcounts``: ``(nbins,)`` for
        soft ``s`` or ``(nbins, nell)``.  With padding the buffer-bin
        cotangents are zero internally: the buffer bins are discarded from the
        forward counts, so the caller never sees them.

    Returns
    -------
    result : dict
        ``grad_position1`` (n1, 3), ``grad_position2`` (n2, 3),
        ``grad_weight1`` (n1,), ``grad_weight2`` (n2,) (``None`` when the
        catalogue carries no individual weights), ``grad_sigma`` (float) and
        ``result['soft']`` as in the forward.

    Notes
    -----
    For an autocorrelation -- the same catalog passed twice -- the derivative
    with respect to that one catalog is ``grad_position1 + grad_position2``:
    each is the derivative of the ordered-pair sum with respect to one of its
    two slots. The forward enumerates both directions, so the sum is the total.
    """
    _setup_cucount_logging()
    assert len(particles) == 2
    if wattrs is None: wattrs = WeightAttrs()
    if sattrs is None: sattrs = SelectionAttrs()
    if spattrs is None: spattrs = SplitAttrs()
    wattrs.check(*particles)
    spattrs.check(*particles)
    # The kernel writes one histogram, not one per split; raising here keeps a
    # jackknife request from coming back as plausible-looking plain counts.
    if getattr(spattrs, 'nsplits', 0):
        raise NotImplementedError('split/jackknife counts are not served by the soft kernel yet')
    _check_count2(particles, battrs, wattrs, spattrs)

    mode = _resolve_backend(backend)
    cuda_tuning, cpu_tuning = _resolve_tuning(mode, tuning, nthreads=nthreads)

    (battrs_ext, edges_ext, edges_real, nlo, nhi, pad, mattrs,
     soft_sigma) = _pad_setup(particles, battrs, mattrs, sattrs, mode,
                              soft_pad, soft_sigma)
    ns_real = int(edges_real.size - 1)

    cparticles = [_to_c_particles(p) for p in particles]

    cotangent = np.asarray(cotangent, dtype=np.float64)
    if cotangent.ndim == 1:
        # A flattened (nbins, nell) cotangent is accepted as before; only the
        # separation axis is resized by the buffer bins.
        if cotangent.size != ns_real and ns_real and cotangent.size % ns_real == 0:
            cotangent = cotangent.reshape(ns_real, -1)
    if cotangent.ndim not in (1, 2):
        raise ValueError(f'cotangent must be 1D or 2D, got shape {cotangent.shape}')
    if cotangent.shape[0] != ns_real:
        raise ValueError(
            'the cotangent must have one entry per real separation bin, got '
            f'{cotangent.shape[0]} rows for {ns_real} bins')
    if nlo or nhi:
        cotangent = pad_cotangent(cotangent, nlo, int(edges_ext.size - 1))
    else:
        cotangent = np.ascontiguousarray(cotangent)

    def run(which, tuning):
        lib, lib_nthreads = _lib_and_nthreads(which, tuning)
        result, timings = lib.count2_soft_vjp(
            *cparticles, mattrs._to_c(), battrs_ext, cotangent,
            wattrs=wattrs._to_c(), sattrs=sattrs, spattrs=spattrs,
            soft_sigma=soft_sigma, soft_cutoff=float(soft_cutoff),
            soft_taps=str(soft_taps), soft_accum=str(soft_accum),
            soft_norm=str(soft_norm),
            soft_mu_sigma=soft_mu_sigma, soft_mu_bins=int(soft_mu_bins),
            soft_mu_edges=soft_mu_edges,
            nthreads=lib_nthreads, return_timings=True)
        _log_timings(which, *timings)
        return result

    result = _dispatch(mode, run, cuda_tuning, cpu_tuning)
    return slice_soft_result(result, nlo, nhi, ns_real, edges_ext, pad)
