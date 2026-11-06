"""What the two soft-binning normalisations cost: forward and backward.

    python examples/bench_soft_norm.py                 # real Abacus LRG sub-box
    srun -p gpu --gres=gpu:1 python examples/bench_soft_norm.py --backends cpu,cuda

``soft_norm='gaussian'`` (the default) makes a single pass over the window and
deposits the discrete-quadrature mass ``A_b exp(-x_b)``; ``soft_norm='softmax'``
makes a second pass to renormalise the window so that each pair's weights sum to
exactly one.  The second pass is the whole reason the option exists: this script
measures what it buys and what it costs.

Both doors into the kernel are timed:

* **cold** -- the first call for a given (backend, normalisation, width).  It
  pays the one-off work: scratch allocation, the first launch of a
  configuration (CUDA module load / plan, OpenMP pool spin-up).
* **warm** -- the median of ``--repeats`` further calls, i.e. the steady-state
  cost that a sampler or a fit actually sees.

The *device* warm-up is excluded from both: a throwaway hard ``count2`` runs
first (it is reported as ``device init``), so that CUDA context creation does not
get charged to whichever normalisation happens to be measured first.

Each row also reports the kernel-only time that the bindings return from
``return_timings=True``, plus the implied throughput ``N^2/2 / t`` -- an upper
bound on the pair rate, since pairs outside the counting range are skipped.

The catalogue is a periodic sub-box cut out of a real AbacusSummit halo
catalogue (see ``examples/_catalogues.py``): a sub-box keeps the *true* number
density, which is what makes the small-separation bins well populated.  With
``--synthetic`` a halo-profile mock is used instead, so the script also runs
where the mocks are not mounted.

Outputs ``examples/figures/bench_soft_norm.png`` and the same measurements as
``examples/figures/bench_soft_norm.json``.
"""

import argparse
import json
import os
import sys
import time

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

from cucount.numpy import (BinAttrs, MeshAttrs, Particles, SelectionAttrs,  # noqa: E402
                           SplitAttrs, WeightAttrs, _to_c_particles, cpulib,
                           cudalib)

try:  # run as ``python examples/bench_soft_norm.py`` or as a module
    from examples._catalogues import load_abacus, synthetic_clustered  # noqa: E402
except ImportError:
    from _catalogues import load_abacus, synthetic_clustered  # noqa: E402

NORMS = ('gaussian', 'softmax')
FIGDIR = os.path.join(HERE, 'figures')


# --------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------

def _kernel_seconds(timings):
    """Kernel time only, from the bindings' ``return_timings`` payload."""
    out = timings[1] if isinstance(timings, tuple) else timings
    return float(out[0]) if isinstance(out, (tuple, list)) else float(out)


def _time_call(call, repeats):
    """Return ``(cold, warm_median, warm_best, kernel_seconds, result)``.

    ``cold`` is the first call; the warm statistics are over ``repeats`` more.
    Wall-clock is what a user sees, so that is what is reported; the kernel-only
    time comes from the binding's own timings.
    """
    start = time.perf_counter()
    result = call()
    cold = time.perf_counter() - start

    warm, kernel = [], []
    for _ in range(max(int(repeats), 1)):
        start = time.perf_counter()
        result = call()
        warm.append(time.perf_counter() - start)
        timings = result[1] if isinstance(result, tuple) else None
        if timings is not None:
            kernel.append(_kernel_seconds(timings))
    return (cold, float(np.median(warm)), float(np.min(warm)),
            float(np.median(kernel)) if kernel else float('nan'), result)


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def benchmark_backend(name, lib, parts, cparts, box_size, edges, sigma_bins,
                      repeats, nthreads):
    """Time forward and backward for both normalisations on one backend.

    The widths are fractions of the *median* bin width.  On a log grid the local
    width varies across the range, so a fraction of the narrowest bin would
    silently be a very different kernel at the two ends of the grid.
    """
    wattrs = WeightAttrs()._to_c()
    sattrs, spattrs = SelectionAttrs(), SplitAttrs()
    ba = BinAttrs(s=edges)
    mesh = MeshAttrs(parts, parts, boxsize=box_size, battrs=ba, periodic=True)
    ma = mesh._to_c()
    n_pairs = 0.5 * cparts.size * (cparts.size - 1)
    cotangent = np.ones(edges.size - 1)

    def hard():
        return lib.count2(cparts, cparts, ma, ba, wattrs=wattrs, sattrs=sattrs,
                          spattrs=spattrs, nthreads=nthreads,
                          return_timings=True)

    def forward(sigma, soft_norm):
        return lib.count2_soft(cparts, cparts, ma, ba, wattrs=wattrs,
                               sattrs=sattrs, spattrs=spattrs,
                               soft_sigma=sigma, soft_norm=soft_norm,
                               nthreads=nthreads, return_timings=True)

    def backward(sigma, soft_norm):
        return lib.count2_soft_vjp(cparts, cparts, ma, ba, cotangent,
                                   wattrs=wattrs, sattrs=sattrs,
                                   spattrs=spattrs,
                                   soft_sigma=sigma, soft_norm=soft_norm,
                                   nthreads=nthreads, return_timings=True)

    # Device warm-up, reported separately so it is not charged to a norm.
    start = time.perf_counter()
    hard()
    init = time.perf_counter() - start
    hard_cold, hard_warm, hard_best, hard_kernel, _ = _time_call(hard, repeats)

    rows = []
    median_width = float(np.median(np.diff(edges)))
    for fraction in sigma_bins:
        sigma = float(fraction * median_width)
        for soft_norm in NORMS:
            cold_f, warm_f, best_f, kern_f, out = _time_call(
                lambda: forward(sigma, soft_norm), repeats)
            meta = out[0]['soft']
            cold_b, warm_b, best_b, kern_b, _ = _time_call(
                lambda: backward(sigma, soft_norm), repeats)
            rows.append(dict(
                backend=name, fraction=float(fraction), sigma=sigma,
                median_width=median_width,
                norm=soft_norm, taps=int(meta['taps']), kmax=int(meta['kmax']),
                uniform=bool(meta['uniform']),
                forward=dict(cold=cold_f, warm=warm_f, best=best_f, kernel=kern_f),
                backward=dict(cold=cold_b, warm=warm_b, best=best_b, kernel=kern_b),
                pair_rate=n_pairs / warm_f,
            ))
            print(f'{name:>5} {soft_norm:>8} sigma/ds {fraction:4.2f} '
                  f'sigma {sigma:6.3f}  taps {int(meta["taps"]):>3}  '
                  f'fwd cold {1e3 * cold_f:8.1f} warm {1e3 * warm_f:7.1f} '
                  f'(kernel {1e3 * kern_f:7.1f}) ms   '
                  f'bwd cold {1e3 * cold_b:8.1f} warm {1e3 * warm_b:7.1f} '
                  f'(kernel {1e3 * kern_b:7.1f}) ms   '
                  f'{n_pairs / warm_f / 1e9:5.2f} Gpair/s')
    print(f'{name:>5} {"hard":>8} {"-":>14}              '
          f'fwd cold {1e3 * hard_cold:8.1f} warm {1e3 * hard_warm:7.1f} '
          f'(kernel {1e3 * hard_kernel:7.1f}) ms   (reference)')
    return dict(backend=name, init=init, n_particles=int(cparts.size),
                box_size=float(box_size), n_pairs=n_pairs, nthreads=nthreads,
                hard=dict(cold=hard_cold, warm=hard_warm, best=hard_best,
                          kernel=hard_kernel),
                rows=rows)


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

def make_figure(measurements, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    norms = dict(gaussian=dict(colour='#1f77b4', marker='o', ls='-'),
                 softmax=dict(colour='#d62728', marker='s', ls='--'))
    panels = ('forward', 'backward')
    fig, axes = plt.subplots(len(measurements), 3,
                             figsize=(13.8, 4.4 * len(measurements)),
                             squeeze=False)

    for irow, measurement in enumerate(measurements):
        rows = measurement['rows']
        for icol, what in enumerate(panels):
            ax = axes[irow, icol]
            for soft_norm in NORMS:
                selected = [row for row in rows if row['norm'] == soft_norm]
                fractions = [row['fraction'] for row in selected]
                warm = [1e3 * row[what]['warm'] for row in selected]
                cold = [1e3 * row[what]['cold'] for row in selected]
                style = norms[soft_norm]
                ax.plot(fractions, warm, style['ls'], marker=style['marker'],
                        color=style['colour'], lw=1.6, ms=4,
                        label=f"{soft_norm} (warm)")
                ax.plot(fractions, cold, ':', marker=style['marker'], mfc='none',
                        color=style['colour'], lw=1.3, ms=5,
                        label=f"{soft_norm} (cold)")
            if what == 'forward':
                ax.axhline(1e3 * measurement['hard']['warm'], color='0.35',
                           lw=1.1, ls='-.',
                           label='hard count2 (warm)')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\sigma / \Delta s$')
            ax.set_ylabel(f'{what} wall time [ms]')
            ax.set_title(f"{what}: {measurement['backend']} backend, "
                         f"{measurement['n_particles']:,} points")
            ax.grid(alpha=0.25, which='both')
            ax.legend(fontsize=8)

        ax = axes[irow, 2]
        for soft_norm in NORMS:
            style = norms[soft_norm]
            for what, ls in (('forward', '-'), ('backward', '--')):
                selected = [row for row in rows if row['norm'] == soft_norm]
                fractions = [row['fraction'] for row in selected]
                ratio = [row[what]['cold'] / row[what]['warm'] for row in selected]
                ax.plot(fractions, ratio, ls, marker=style['marker'],
                        color=style['colour'], lw=1.5, ms=4,
                        label=f'{soft_norm}, {what}')
        ax.axhline(1.0, color='0.4', lw=0.9, ls=':')
        ax.set_xlabel(r'$\sigma / \Delta s$')
        ax.set_ylabel('cold / warm')
        ax.set_yscale('log')
        ax.set_title(f"first call over steady state "
                     f"({measurement['backend']} backend)")
        ax.grid(alpha=0.25, which='both')
        ax.legend(fontsize=8)

    fig.suptitle('soft binning cost: one pass (gaussian) versus two (softmax), '
                 'forward and adjoint')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sub-box', type=float, default=400.0,
                        help='periodic sub-box cut from the catalogue, Mpc/h '
                             '(default 400: ~57k LRG at the true density)')
    parser.add_argument('--n-particles', type=int, default=None,
                        help='random subsample instead of a sub-box cut')
    parser.add_argument('--tracer', default='LRG')
    parser.add_argument('--redshift', type=float, default=0.5)
    parser.add_argument('--catalogue', default=None,
                        help='explicit catalogue path instead of Abacus lookup')
    parser.add_argument('--synthetic', action='store_true',
                        help='use a generated halo-profile mock')
    parser.add_argument('--edges', default='geomspace:2.5:60:25',
                        help='bin grid: linspace:lo:hi:n or geomspace:lo:hi:n')
    parser.add_argument('--sigma-bins', type=float, nargs='+',
                        default=[0.25, 0.5, 1.0, 2.0],
                        help='kernel widths in units of the bin width')
    parser.add_argument('--backends', default='auto',
                        help="'auto', 'cpu', 'cuda' or 'cpu,cuda'")
    parser.add_argument('--repeats', type=int, default=5,
                        help='warm calls per configuration (default 5)')
    parser.add_argument('--nthreads', type=int, default=None,
                        help='CPU threads (default: min(32, cpu_count); CUDA '
                             'always uses 1)')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', default=None,
                        help='figure path (default examples/figures/bench_soft_norm.png)')
    args = parser.parse_args(argv)

    kind, lo, hi, count = args.edges.split(':')
    if kind == 'linspace':
        edges = np.linspace(float(lo), float(hi), int(count))
    elif kind == 'geomspace':
        edges = np.geomspace(float(lo), float(hi), int(count))
    else:
        raise SystemExit(f'unsupported --edges kind {kind!r}')

    if args.synthetic:
        positions = synthetic_clustered(box_size=args.sub_box or 400.0,
                                        n_particles=50000, seed=args.seed)
        box_size = float(args.sub_box or 400.0)
        info = dict(tracer='synthetic', redshift=float('nan'), path='synthetic')
    else:
        positions, box_size, info = load_abacus(
            tracer=args.tracer, redshift=args.redshift,
            n_particles=args.n_particles, sub_box=args.sub_box,
            seed=args.seed, path=args.catalogue)

    parts = Particles(positions)
    cparts = _to_c_particles(parts)
    print(f"catalogue: {info['tracer']} z={info['redshift']}, "
          f"{positions.shape[0]:,} points, box {box_size:g} Mpc/h, "
          f"{edges.size - 1} bins over [{edges[0]:g}, {edges[-1]:g}]")
    print(f"upper bound on pairs: N^2/2 = {0.5 * cparts.size * (cparts.size - 1):.3e}\n")

    available = dict(cpu=cpulib, cuda=cudalib)
    wanted = list(available) if args.backends == 'auto' else \
        [name.strip() for name in args.backends.split(',') if name.strip()]
    measurements = []
    for name in wanted:
        lib = available.get(name)
        if lib is None or not hasattr(lib, 'count2_soft'):
            print(f'skipping {name}: no count2_soft in this build')
            continue
        nthreads = args.nthreads if args.nthreads is not None else (
            min(32, os.cpu_count() or 1) if name == 'cpu' else 1)
        try:
            measurement = benchmark_backend(
                name, lib, parts, cparts, box_size, edges, args.sigma_bins,
                args.repeats, nthreads)
        except Exception as error:      # a missing GPU must not kill the CPU run
            print(f'skipping {name}: {type(error).__name__}: {error}')
            continue
        measurements.append(measurement)
        print(f"{name}: device warm-up (first hard count2) "
              f"{1e3 * measurement['init']:.1f} ms")

    if not measurements:
        raise SystemExit('no backend could be benchmarked')

    print('\n| backend | norm | sigma/ds | taps | fwd warm [ms] | fwd cold [ms] '
          '| bwd warm [ms] | bwd cold [ms] | cold/warm (fwd) |')
    print('|---|---|---|---|---|---|---|---|---|')
    for measurement in measurements:
        print(f"| {measurement['backend']} | hard | - | - | "
              f"{1e3 * measurement['hard']['warm']:.1f} | "
              f"{1e3 * measurement['hard']['cold']:.1f} | - | - | "
              f"{measurement['hard']['cold'] / measurement['hard']['warm']:.2f} |")
        for row in measurement['rows']:
            print(f"| {row['backend']} | {row['norm']} | {row['fraction']:g} | "
                  f"{row['taps']} | {1e3 * row['forward']['warm']:.1f} | "
                  f"{1e3 * row['forward']['cold']:.1f} | "
                  f"{1e3 * row['backward']['warm']:.1f} | "
                  f"{1e3 * row['backward']['cold']:.1f} | "
                  f"{row['forward']['cold'] / row['forward']['warm']:.2f} |")

    def speedup(what, fraction):
        values = {}
        for measurement in measurements:
            for row in measurement['rows']:
                if row['fraction'] == fraction:
                    values[measurement['backend'], row['norm']] = row[what]['warm']
        out = {}
        for backend in {key[0] for key in values}:
            try:
                out[backend] = values[backend, 'softmax'] / values[backend, 'gaussian']
            except KeyError:
                pass
        return out

    print('\nwarm forward speed-up, softmax / gaussian:')
    for fraction in args.sigma_bins:
        factors = speedup('forward', fraction)
        print(f'  sigma = {fraction:4.2f} ds : ' + '  '.join(
            f'{backend} x{factor:.2f}' for backend, factor in factors.items()))
    print('warm backward speed-up, softmax / gaussian:')
    for fraction in args.sigma_bins:
        factors = speedup('backward', fraction)
        print(f'  sigma = {fraction:4.2f} ds : ' + '  '.join(
            f'{backend} x{factor:.2f}' for backend, factor in factors.items()))

    figure_path = args.output or os.path.join(FIGDIR, 'bench_soft_norm.png')
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    make_figure(measurements, figure_path)
    payload = dict(catalogue=dict(info), edges=edges.tolist(),
                   width=float(np.median(np.diff(edges))),
                   sigma_bins=[float(f) for f in args.sigma_bins],
                   repeats=int(args.repeats),
                   measurements=measurements)
    with open(os.path.splitext(figure_path)[0] + '.json', 'w') as handle:
        json.dump(payload, handle, indent=2)
    print(f'\nwrote {figure_path}')
    print(f'measurements saved to {os.path.splitext(figure_path)[0] + ".json"}')


if __name__ == '__main__':
    main()
