"""The two soft-binning kernels, side by side: shapes, gradients and counts.

``count2_soft`` sends every counted pair through a Gaussian kernel over the bin
centres, and ``soft_norm`` chooses how that kernel is normalised:

``'gaussian'`` (the default)
    the discrete-quadrature mass ``W_b = A_b exp(-x_b)`` with
    ``A_b = Delta_b / (sqrt(2 pi) sigma_b)``.  Each pair is deposited in one
    pass; the bins do not compete for the pair's weight, so the Jacobian of the
    deposit with respect to ``s`` is a single-signed bump per bin and the
    gradient cannot collapse onto one bin.  The price is that the per-pair row
    sum is only one to quadrature accuracy (0.4% at the default half-bin
    ``sigma``, worse as the kernel narrows), and the kernel tail that falls
    outside the counted range is simply lost -- which is exactly the edge
    effect ``soft_pad`` addresses.

``'softmax'`` (the legacy behaviour)
    ``W_b = exp(-x_b) / sum_j exp(-x_j)`` over the truncated window.  The row
    sum is exactly one for every pair, at the price of a second pass and of
    competition between the bins: the window is renormalised per pair, so
    ``dW_b/ds`` has negative lobes elsewhere and ``sum_b dW_b/ds = 0`` exactly.

Two figures come out of one run:

``soft_norm_comparison.png``
    the kernel algebra, on a uniform grid in units of the bin width: weights,
    per-pair row sums, the ``s``-derivative of the pair total, the ratio between
    the two paths, and how the width compares with the local bin width.

``soft_norm_2pcf.png``
    the correlation function itself.  ``xi(s)`` on log-log axes (small scales),
    ``s^2 xi(s)`` on linear axes (large scales), and the fractional difference
    from hard binning with and without ``soft_pad``.

The measurements are a real catalogue: an Abacus halo sample at z=0.5, cut to a
periodic sub-box so the pair counts stay honest, with several kernel widths
against the hard ``count2`` binning.

On a **log-spaced** grid a fixed absolute ``sigma`` is not the same kernel at
every scale: the local bin width goes from ~0.35 to ~7.4 Mpc/h here, so
``sigma = 1 x median width`` (1.6 Mpc/h) is under half a bin at large ``s`` and
nine bins wide at small ``s``.  That over-smoothing, not the normalisation, is
what dominates the small-scale deviation in the absolute-width series.  The
scale-free choice is ``soft_sigma=None`` (the API default), i.e.
``sigma_i = Delta_i / 2`` per bin: the kernel is always half a *local* bin, which
on a log grid means ``sigma(s) proportional to s``.  That is the same kernel as
pyfcfc's ``soft_sigma_log=True`` mode (``sigma(s) = f * s``, a linear-space
Gaussian of constant relative width) with ``f = Delta ln s / 2``; pyfcfc's own
default (``soft_sigma=0``) is instead half the *mean* bin width, an absolute
width, so on a log grid the two codes' defaults are different kernels.

Run on any machine (CPU backend, a few minutes)::

    python examples/soft_norm_comparison.py

On a cluster whose GPU partition can see the catalogue::

    srun -p gpu --gres=gpu:1 python examples/soft_norm_comparison.py --backend cuda

It writes both PNGs and ``examples/figures/soft_norm_comparison.json``.

The catalogue path resolution is :mod:`examples._catalogues`, which follows the
``holi-refactor`` Abacus loader.  Pass ``--catalogue PATH`` to use a specific
file, or ``--synthetic`` when no real catalogue is reachable.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(REPO)

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', 'cucount-mpl'))


def _bootstrap():
    """Use the development build (and the local .pydeps), not site-packages."""
    sys.meta_path[:] = [finder for finder in sys.meta_path
                        if type(finder).__name__ != 'ScikitBuildRedirectingFinder']
    for path in (os.path.join(WORKSPACE, '.devlib'), REPO,
                 os.path.join(WORKSPACE, '.pydeps')):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_bootstrap()

import numpy as np  # noqa: E402

from cucount.numpy import BinAttrs, MeshAttrs, Particles, count2  # noqa: E402
from cucount.numpy.soft import count2_soft, soft_weights  # noqa: E402

try:  # run as ``python examples/soft_norm_comparison.py`` or as a module
    from examples._catalogues import load_abacus, synthetic_clustered  # noqa: E402
except ImportError:
    from _catalogues import load_abacus, synthetic_clustered  # noqa: E402

#: Relative half-width of the finite-difference step, in bin widths.
FINITE_DIFFERENCE_STEP = 1e-4

#: Kernel widths for the kernel-algebra panels, in units of the bin width.
SIGMA_FRACTIONS = (0.25, 0.5, 1.0, 2.0)

#: Ratios are only plotted where hard binning has this many pairs.
MINIMUM_HARD_COUNTS = 50

NORMS = ('gaussian', 'softmax')

COLOURS = {'gaussian': '#1f77b4', 'softmax': '#c0392b'}
STYLES = {'gaussian': '-', 'softmax': '--'}
LABELS = {'gaussian': "'gaussian' (default)", 'softmax': "'softmax' (legacy)"}

SIGMA_COLOURS = {0.25: '#4c72b0', 0.5: '#55a868', 1.0: '#dd8452', 2.0: '#8172b3'}


def kernel_grid(nbins=20, width=1.0):
    """Edges of a uniform grid in units of ``width``, used for the kernel panels."""
    return np.arange(nbins + 1, dtype=np.float64) * width


def row_sum_across_the_range(edges, sigma, soft_norm):
    """Per-pair row sum versus ``s``, from the lower edge to the upper one."""
    s_values = np.linspace(edges[0] + 1e-6, edges[-1] - 1e-6, 1201)
    weights = soft_weights(s_values, edges, sigma=sigma, soft_norm=soft_norm)
    return s_values, weights.sum(axis=1)


def row_sum_versus_width(edges, soft_norm, fractions=None):
    """Per-pair row sum at an interior bin centre, versus ``sigma / Delta``.

    The discrete sum of a Gaussian sampled at the bin centres is the integral
    plus the Euler-Maclaurin tail; the tail dies as ``exp(-2 pi^2 sigma^2 /
    Delta^2)``, so a *wide* kernel integrates accurately and it is the narrow
    one that loses the weight between the centres.  (The real-catalogue
    deviations at wide kernels are the counting edges, not this.)
    """
    if fractions is None:
        fractions = np.geomspace(0.05, 4.0, 61)

    centre = 0.5 * (edges[len(edges) // 2] + edges[len(edges) // 2 + 1])
    sums = [float(soft_weights([centre], edges, sigma=fraction,
                               soft_norm=soft_norm)[0].sum())
            for fraction in fractions]
    return np.asarray(fractions), np.asarray(sums)


def total_weight_gradient(edges, sigma, soft_norm, npoints=801):
    """``|sum_b dW_b/ds|``: how much a pair's total weight drifts with ``s``."""
    s_values = np.linspace(edges[0] + 1e-4, edges[-1] - 1e-4, npoints)
    step = FINITE_DIFFERENCE_STEP * (edges[1] - edges[0])
    plus = soft_weights(s_values + step, edges, sigma=sigma,
                        soft_norm=soft_norm).sum(axis=1)
    minus = soft_weights(s_values - step, edges, sigma=sigma,
                         soft_norm=soft_norm).sum(axis=1)
    return s_values, np.abs(plus - minus) / (2.0 * step)


def _deviations(counts, reference):
    with np.errstate(divide='ignore', invalid='ignore'):
        return counts / reference - 1.0


def random_pair_counts(edges, box_size, n_particles):
    """Unordered random-pair counts per shell of a periodic box."""
    shell_volume = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    return 0.5 * n_particles * (n_particles - 1) * shell_volume / box_size ** 3


def hard_counts(positions, box_size, edges, backend):
    """``count2`` with hard binning, the reference the soft paths are judged by."""
    particles = Particles(positions)
    battrs = BinAttrs(s=edges)
    mattrs = MeshAttrs(particles, particles, boxsize=box_size, battrs=battrs,
                       periodic=True, backend=backend)
    return count2(particles, particles, battrs=battrs, mattrs=mattrs,
                  backend=backend)['weight']


def soft_counts(positions, box_size, edges, sigma, soft_norm, backend, soft_pad):
    """One soft count.  ``soft_pad=0`` is the raw estimator, ``None`` the padded one."""
    particles = Particles(positions)
    battrs = BinAttrs(s=edges)
    mattrs = MeshAttrs(particles, particles, boxsize=box_size, battrs=battrs,
                       periodic=True, backend=backend)
    result = count2_soft(particles, particles, battrs=battrs, mattrs=mattrs,
                         soft_sigma=sigma, soft_pad=soft_pad,
                         soft_accum='float64', soft_norm=soft_norm,
                         backend=backend)
    return result['weight'], result['soft']


def measurement(positions, box_size, edges, sigma, soft_norm, backend, soft_pad,
                hard, random_counts):
    """Soft counts plus the two ``xi`` estimates, ready for plotting."""
    soft, meta = soft_counts(positions, box_size, edges, sigma, soft_norm,
                             backend, soft_pad)
    return dict(soft=soft, hard=hard, meta=meta,
                xi_hard=hard / random_counts - 1.0,
                xi_soft=soft / random_counts - 1.0,
                random_counts=random_counts,
                deviation=_deviations(soft, hard))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sub-box', type=float, default=400.0,
                        help='side of the periodic sub-box cut from the catalogue, '
                             'in Mpc/h (default 400: ~57k LRG at the true density)')
    parser.add_argument('--n-particles', type=int, default=None,
                        help='random subsample instead of a sub-box cut')
    parser.add_argument('--tracer', default='LRG', help='Abacus tracer (default LRG)')
    parser.add_argument('--redshift', type=float, default=0.5,
                        help='target redshift (default 0.5)')
    parser.add_argument('--catalogue', default=None,
                        help='explicit Abacus FITS catalogue, skipping the search')
    parser.add_argument('--synthetic', action='store_true',
                        help='use a clustered synthetic catalogue instead of Abacus')
    parser.add_argument('--backend', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--s-min', type=float, default=2.0)
    parser.add_argument('--s-max', type=float, default=60.0)
    parser.add_argument('--nbins', type=int, default=24)
    parser.add_argument('--sigma-bins', type=float, nargs='+', default=[0.5, 1.0, 2.0],
                        help='absolute kernel widths to sweep, in units of the median '
                             'bin width of the separation grid (default 0.5 1 2); the '
                             'API default sigma=None is always measured as well')
    parser.add_argument('--pad', default='auto',
                        help="soft_pad for the padded series: 'auto' (the default "
                             '2 kernel widths), a number, or \'none\' to skip')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    if args.synthetic:
        box_size = 400.0
        positions = synthetic_clustered(box_size=box_size, n_particles=50000,
                                        seed=args.seed)
        info = dict(path='synthetic', tracer='synthetic', box_size=box_size,
                    redshift=float('nan'), resolved_by='--synthetic',
                    n_total=int(positions.shape[0]))
    else:
        try:
            positions, box_size, info = load_abacus(
                tracer=args.tracer, redshift=args.redshift,
                n_particles=args.n_particles, sub_box=args.sub_box,
                seed=args.seed, path=args.catalogue)
        except FileNotFoundError as error:
            print(f'warning: {error}\nfalling back to --synthetic')
            box_size = 400.0
            positions = synthetic_clustered(box_size=box_size, n_particles=50000,
                                            seed=args.seed)
            info = dict(path='synthetic', tracer='synthetic', box_size=box_size,
                        redshift=float('nan'), resolved_by='fallback',
                        n_total=int(positions.shape[0]))

    edges = (np.geomspace(args.s_min, args.s_max, args.nbins + 1)
             if args.s_min > 0 else np.linspace(args.s_min, args.s_max, args.nbins + 1))
    centres = 0.5 * (edges[1:] + edges[:-1])
    local_widths = np.diff(edges)
    median_width = float(np.median(local_widths))

    print(f'catalogue: {info["resolved_by"]}, tracer {info["tracer"]} '
          f'z={info["redshift"]:.3f}, {info["n_total"]} points, '
          f'box {box_size:g} Mpc/h')
    print(f'grid: {args.nbins} bins in [{args.s_min:g}, {args.s_max:g}] Mpc/h; '
          f'bin width {local_widths.min():g} to {local_widths.max():g} '
          f'(median {median_width:g})')

    # ------------------------------------------------------------------
    # kernel algebra (uniform grid in units of the bin width)
    # ------------------------------------------------------------------
    unit_edges = kernel_grid(nbins=20, width=1.0)
    unit_centres = 0.5 * (unit_edges[1:] + unit_edges[:-1])
    interior_s = 10.5

    kernel_weights = {}
    for fraction in SIGMA_FRACTIONS:
        for soft_norm in NORMS:
            kernel_weights[fraction, soft_norm] = soft_weights(
                [interior_s], unit_edges, sigma=fraction,
                soft_norm=soft_norm)[0]

    row_sums = {}
    for fraction in (0.5, 1.0, 2.0):
        for soft_norm in NORMS:
            row_sums[fraction, soft_norm] = row_sum_across_the_range(
                unit_edges, fraction, soft_norm)

    total_gradients = {}
    for fraction in (0.5, 1.0):
        for soft_norm in NORMS:
            total_gradients[fraction, soft_norm] = total_weight_gradient(
                unit_edges, fraction, soft_norm)

    quadrature_curves = {soft_norm: row_sum_versus_width(unit_edges, soft_norm)
                         for soft_norm in NORMS}

    # ------------------------------------------------------------------
    # the real catalogue
    # ------------------------------------------------------------------
    def configurations():
        """(label, sigma) pairs: the API default first, then the absolute widths."""
        yield 'default ($\\sigma=\\Delta/2$ per bin)', None
        for fraction in args.sigma_bins:
            yield f'$\\sigma={fraction:g}\\,\\Delta_{{\\rm med}}$', fraction * median_width

    # ``soft_pad=None`` means "the front-end default" (two kernel widths), which
    # is the padded estimator; ``soft_pad=0.0`` is the raw one.  Note that those
    # two are different values, not different spellings of the same one.
    if args.pad == 'none':
        pad_values = (0.0,)
        padded_value = 0.0
    elif args.pad == 'auto':
        pad_values = (0.0, None)
        padded_value = None
    else:
        padded_value = float(args.pad)
        pad_values = (0.0, padded_value)

    reference_hard = hard_counts(positions, box_size, edges, args.backend)
    random_counts = random_pair_counts(edges, box_size, positions.shape[0])

    print()
    header = (f'{"sigma":>26}  {"norm":>8}  {"pad":>5}  {"taps":>8}  '
              f'{"kmax":>4}  {"total":>7}  {"max|dev|":>8}  {"first bin":>9}')
    print(header)
    print('-' * len(header))

    measurements = {}

    def report(label, sigma, soft_norm, soft_pad, result):
        meta = result['meta']
        # On a non-uniform grid the tap count is chosen per pair, and the backend
        # reports -1; kmax is then the whole window.
        taps = 'adaptive' if int(meta['taps']) < 0 else f'{int(meta["taps"]):d}'
        populated = np.isfinite(result['deviation']) & (result['hard']
                                                        >= MINIMUM_HARD_COUNTS)
        worst = 100 * np.nanmax(np.abs(result['deviation'][populated]))
        first = 100 * result['deviation'][populated][0]
        pad_text = 'off' if soft_pad == 0.0 else 'on'
        total = 100 * (result['soft'].sum() / result['hard'].sum() - 1)
        print(f'{label:>26}  {soft_norm:>8}  {pad_text:>5}  {taps:>8}  '
              f'{int(meta["kmax"]):4d}  {total:+6.2f}%  {worst:7.2f}%  {first:+8.2f}%')

    for label, sigma in configurations():
        for soft_norm in NORMS:
            for soft_pad in pad_values:
                result = measurement(positions, box_size, edges, sigma, soft_norm,
                                     args.backend, soft_pad, reference_hard,
                                     random_counts)
                measurements[label, soft_norm, soft_pad] = result
                report(label, sigma, soft_norm, soft_pad, result)

    # A sanity check on the estimator rather than on the kernel: the largest
    # separations of a finite box must average to xi ~ 0.
    print('\nxi_hard at the four largest separations: '
          + ', '.join(f'{value:+.3f}' for value in
                      (reference_hard / random_counts - 1.0)[-4:]))

    # ------------------------------------------------------------------
    # figure 1: the kernel algebra
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), layout='constrained')
    fig.suptitle(
        f'Soft-binning kernels: {info["tracer"]} z={info["redshift"]:.3f}, '
        f'{info["n_total"]} points, box {box_size:g} Mpc/h', fontsize=11.5)

    # Panel 1: the kernel itself.  At fixed s the two normalisations differ by
    # exactly one factor, 1 / sum_b W_b, so the shapes carry no information
    # about which path ran.
    ax = axes[0, 0]
    for fraction in SIGMA_FRACTIONS:
        colour = SIGMA_COLOURS[fraction]
        for soft_norm in NORMS:
            ax.semilogy(unit_centres - interior_s,
                        kernel_weights[fraction, soft_norm],
                        STYLES[soft_norm], color=colour, lw=1.4,
                        label=rf'$\sigma={fraction:g}\,\Delta s$, {soft_norm}')
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(1e-6, 1.6)
    ax.set_xlabel(r'$(c_b - s)\ [\Delta s]$')
    ax.set_ylabel(r'deposited weight $W_b$ for one pair')
    ax.set_title('Kernel shape (log, clipped below $10^{-6}$)\n'
                 'the two paths differ only by a factor', fontsize=9.5)
    ax.legend(fontsize=6.8, ncol=2)
    ax.grid(alpha=0.25, which='both')

    # Panel 2: per-pair row sum -- the one real algebraic difference.
    ax = axes[0, 1]
    for fraction in (0.5, 1.0, 2.0):
        colour = SIGMA_COLOURS[fraction]
        for soft_norm in NORMS:
            s_axis, sums = row_sums[fraction, soft_norm]
            ax.plot(s_axis, sums, STYLES[soft_norm], color=colour, lw=1.5,
                    label=rf'$\sigma={fraction:g}\,\Delta s$, {soft_norm}')
    ax.axhline(1.0, color='0.4', lw=0.9, ls=':')
    ax.set_xlabel(r'$s\ [\Delta s]$')
    ax.set_ylabel(r'$\sum_b W_b$ for one pair')
    ax.set_title('Total weight a pair deposits\n'
                 'softmax: exactly 1, gaussian: sags at the edges', fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # Panel 3: the discrete-quadrature error, which is what "quadrature accuracy"
    # means and why a narrow kernel is the inaccurate one.
    ax = axes[1, 0]
    for soft_norm in NORMS:
        fractions, sums = quadrature_curves[soft_norm]
        ax.semilogx(fractions, 100 * np.abs(sums - 1.0), STYLES[soft_norm],
                    color=COLOURS[soft_norm], lw=1.6,
                    label=LABELS[soft_norm])
    ax.set_xlabel(r'$\sigma / \Delta s$')
    ax.set_ylabel(r'$|\sum_b W_b - 1|\ [\%]$ (interior bin)')
    ax.set_title('Quadrature error of the row sum\n'
                 'exact for wide kernels, off for narrow ones', fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which='both')

    # Panel 4: why one absolute sigma cannot be "one bin width" everywhere on a
    # log grid -- the reason the absolute-width series in figure 2 deviate.
    ax = axes[1, 1]
    ax.semilogx(centres, local_widths, 'k-', lw=1.6, label=r'$\Delta s$ (local)')
    for fraction in args.sigma_bins:
        ax.semilogx(centres, np.full_like(centres, fraction * median_width),
                    '--', lw=1.3,
                    label=rf'$\sigma={fraction:g}\,\Delta_{{\rm med}}'
                          rf'={fraction * median_width:.2f}$')
    ax.semilogx(centres, 0.5 * local_widths, ':', lw=1.6,
                label=r'$\sigma=\Delta s/2$ (default)')
    ax.set_xlabel(r'$s\ [\mathrm{Mpc}/h]$')
    ax.set_ylabel(r'kernel width $[\mathrm{Mpc}/h]$')
    ax.set_title(f'Log bins: local width spans {local_widths.min():.2f}-'
                 f'{local_widths.max():.2f} Mpc/h\n'
                 'a fixed absolute $\\sigma$ cannot be "one bin"', fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which='both')

    figure_dir = os.path.join(HERE, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    png = os.path.join(figure_dir, 'soft_norm_comparison.png')
    fig.savefig(png, dpi=150)
    print(f'\nwrote {png}')

    # ------------------------------------------------------------------
    # figure 2: the two-point function itself
    # ------------------------------------------------------------------
    default_label = next(iter(configurations()))[0]
    narrow_label = f'$\\sigma={args.sigma_bins[0]:g}\\,\\Delta_{{\\rm med}}$'

    def curve(label, soft_norm, soft_pad):
        return measurements[label, soft_norm, soft_pad]

    fig2, axes2 = plt.subplots(2, 2, figsize=(12.5, 8.6), layout='constrained')
    fig2.suptitle(rf'$\xi(s)$ from soft binning: {info["tracer"]} '
                  f'z={info["redshift"]:.3f}, {info["n_total"]} points, '
                  f'box {box_size:g} Mpc/h', fontsize=11.5)

    # Panel A: xi(s) log-log -- small scales.  The scale-free default (sigma =
    # half the *local* bin width) is the one that can track hard binning at
    # every scale; an absolute width is wide against the small-s bins.
    ax = axes2[0, 0]
    hard_xi = curve(default_label, 'gaussian', 0.0)['xi_hard']
    ax.plot(centres, hard_xi, 'o-', color='0.2', lw=1.8, ms=4.0,
            label='hard binning')
    for soft_norm in NORMS:
        ax.plot(centres, curve(default_label, soft_norm, padded_value)['xi_soft'],
                's' + STYLES[soft_norm], ms=3.6, lw=1.4, color=COLOURS[soft_norm],
                label=f'{LABELS[soft_norm]}, ' + r'$\sigma=\Delta s/2$ (default)')
    for soft_norm in NORMS:
        ax.plot(centres, curve(narrow_label, soft_norm, padded_value)['xi_soft'],
                '^' + STYLES[soft_norm], ms=3.2, lw=1.1, color=COLOURS[soft_norm],
                alpha=0.55, label=f'{LABELS[soft_norm]}, {narrow_label}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$s\ [\mathrm{Mpc}/h]$')
    ax.set_ylabel(r'$\xi(s)$')
    ax.set_title(r'The correlation function on log-log axes (padded series)'
                 '\nthe scale-free default tracks hard binning at every scale')
    ax.legend(fontsize=6.6)
    ax.grid(alpha=0.25, which='both')

    # Panels B and C: the fractional difference from hard binning, wide and
    # narrow kernels, with and without the buffer bins.
    for ax, label in ((axes2[0, 1], default_label), (axes2[1, 0], narrow_label)):
        well_measured = (curve(label, 'gaussian', 0.0)['hard']
                         >= MINIMUM_HARD_COUNTS)
        for soft_norm in NORMS:
            for soft_pad, pad_alpha, pad_style in ((0.0, 0.5, '-'),
                                                   (padded_value, 1.0, '-')):
                if pad_style == '-' and soft_pad == 0.0 and padded_value == 0.0:
                    continue
                pad_text = ('unpadded' if soft_pad == 0.0
                            else f'soft_pad={soft_pad}')
                ax.plot(centres[well_measured],
                        100 * curve(label, soft_norm, soft_pad)['deviation'][well_measured],
                        pad_style, color=COLOURS[soft_norm], lw=1.5, alpha=pad_alpha,
                        label=f'{LABELS[soft_norm]}, {pad_text}')
        ax.axhline(0.0, color='0.4', lw=0.9)
        ax.set_xscale('log')
        ax.set_xlabel(r'$s\ [\mathrm{Mpc}/h]$')
        ax.set_ylabel(r'$(\mathrm{soft}/\mathrm{hard} - 1)\ [\%]$')
        ax.set_title(f'Deviation from hard binning, {label}\n'
                     rf'(bins with $\geq {MINIMUM_HARD_COUNTS}$ pairs; '
                     'thin = unpadded)')
        ax.legend(fontsize=6.8)
        ax.grid(alpha=0.25, which='both')

    # Panel D: large scales, y-axis weighted by s^2.
    ax = axes2[1, 1]
    ax.plot(centres, centres ** 2 * hard_xi, 'o-', color='0.2', lw=1.8, ms=4.0,
            label='hard binning')
    for soft_norm in NORMS:
        ax.plot(centres, centres ** 2 * curve(default_label, soft_norm,
                                             padded_value)['xi_soft'],
                's' + STYLES[soft_norm], ms=3.6, color=COLOURS[soft_norm], lw=1.4,
                label=f'{LABELS[soft_norm]}, ' + r'$\sigma=\Delta s/2$ (default)')
    ax.axhline(0.0, color='0.4', lw=0.9)
    ax.set_xlabel(r'$s\ [\mathrm{Mpc}/h]$')
    ax.set_ylabel(r'$s^2\,\xi(s)$')
    ax.set_title(r'Large scales: $s^2\,\xi(s)$ (padding keeps the edge bins honest)')
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)

    png2 = os.path.join(figure_dir, 'soft_norm_2pcf.png')
    fig2.savefig(png2, dpi=150)
    print(f'wrote {png2}')

    # ------------------------------------------------------------------
    # machine-readable measurements
    # ------------------------------------------------------------------
    payload = dict(
        catalogue=info, box_size=box_size,
        n_particles=int(positions.shape[0]),
        edges=edges.tolist(), local_widths=local_widths.tolist(),
        median_width=median_width,
        kernel_grid=unit_edges.tolist(),
        sigma_fractions=list(SIGMA_FRACTIONS),
        sigma_bins=list(args.sigma_bins),
        row_sums={f'{fraction}|{soft_norm}': row_sums[fraction, soft_norm][1].tolist()
                  for fraction in (0.5, 1.0, 2.0) for soft_norm in NORMS},
        quadrature={soft_norm: dict(
            fractions=quadrature_curves[soft_norm][0].tolist(),
            row_sums=quadrature_curves[soft_norm][1].tolist())
            for soft_norm in NORMS},
        # |d(sum_b W_b)/ds| per pair: identically zero for softmax, non-zero at
        # the counting edges for the plain Gaussian.  Kept in the payload rather
        # than drawn, since a finite difference of an exactly-zero quantity is
        # mostly rounding noise.
        total_weight_gradient={
            f'{fraction}|{soft_norm}': total.tolist()
            for (fraction, soft_norm), (s_axis, total) in total_gradients.items()},
        total_weight_gradient_s={
            f'{fraction}|{soft_norm}': s_axis.tolist()
            for (fraction, soft_norm), (s_axis, total) in total_gradients.items()},
        xi_hard=(reference_hard / random_counts - 1.0).tolist(),
        hard_counts=reference_hard.tolist(),
        measurements={
            f'{soft_norm}|{label}|pad={soft_pad}': dict(
                sigma=(None if label == default_label
                       else float(label.split('=')[1].split('\\')[0]) * median_width),
                taps=int(result['meta']['taps']),
                kmax=int(result['meta']['kmax']),
                norm=result['meta'].get('norm'),
                soft_pad=result['meta'].get('soft_pad'),
                counts=result['soft'].tolist(),
                xi=result['xi_soft'].tolist(),
                deviation=result['deviation'].tolist(),
                total_weight_deviation=float(
                    abs(result['soft'].sum() - reference_hard.sum())
                    / reference_hard.sum()))
            for (label, soft_norm, soft_pad), result in measurements.items()})

    js = os.path.join(figure_dir, 'soft_norm_comparison.json')
    with open(js, 'w') as handle:
        json.dump(payload, handle, indent=1)
    print(f'measurements saved to {js}')


if __name__ == '__main__':
    main()
