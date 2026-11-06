"""Show that ``soft_pad`` removes the truncated-kernel bias at the bin edges.

Hard binning is exact: every pair inside ``[edges[0], edges[-1])`` is counted
once, and pairs outside are simply not in the histogram.  The soft histogram
instead gives each counted pair a Gaussian kernel over the bin centres.  That
kernel has infinite support, but the pair traversal stops at the outermost
edges, so without padding the boundary bins see a truncated kernel: the legacy
``soft_norm='softmax'`` inflates them by renormalising over the inward-only
window, while the default ``soft_norm='gaussian'`` simply misses the tail that
fell outside.  ``soft_pad`` extends the counting range with buffer bins, lets
the kernel deposit over the extended centres (the softmax also normalises over
them), and then drops the buffer bins from the result.  The example runs the
``soft_norm='softmax'`` mode, where that bias has a definite sign; the default
``soft_norm='gaussian'`` kernel is biased in the opposite direction but by a
comparable amount, and ``soft_pad`` fixes both.

This example uses a uniform random catalogue, so the hard counts are a smooth
reference and any soft-to-hard discrepancy is easy to read.  The separation grid
is linear and the kernel is close to one bin wide: wide enough for the edge
truncation to matter, narrow enough that the padded soft counts follow the hard
counts.

Four panels show

1. the hard counts and the soft counts before/after padding,
2. the soft counts relative to the hard counts (the edge leakage is the spike
   at the last bin),
3. a zoom on the upper edge, and
4. the last-bin deviation from hard binning as a function of ``soft_pad``.

Run with::

    python examples/soft_pad_leakage.py

It writes ``examples/figures/soft_pad_leakage.png``.  Everything runs on the
CPU backend, so no GPU is required.
"""

import os

import numpy as np

from cucount.numpy import BinAttrs, MeshAttrs, Particles, count2
from cucount.numpy.soft import count2_soft
from cucount.softpad import SOFT_PAD_DEFAULT

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BOX = 500.0
N = 4000
EDGES = np.linspace(50.0, 150.0, 21)
NS = EDGES.size - 1
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])
SIGMA = 4.0  # close to one bin width: visible edge bias, calm interior
BACKEND = 'cpu'


def catalogue(n=N, box=BOX, seed=11):
    """A uniform catalogue: the hard counts are the exact reference."""
    rng = np.random.default_rng(seed)
    return Particles(rng.uniform(0.0, box, (n, 3)))


def _counts(p, battrs, mattrs, pad):
    # Pinned to the softmax normalisation: it gives the edge bias a definite
    # sign (the truncated window is renormalised inward), which is what the
    # panels show.  The default gaussian kernel is biased the same way in
    # magnitude but in the opposite direction, since it drops the tail instead
    # of redistributing it, and soft_pad fixes both.
    return count2_soft(p, p, battrs=battrs, mattrs=mattrs,
                       soft_sigma=SIGMA, soft_pad=pad, soft_norm='softmax',
                       soft_accum='float64', backend=BACKEND)


def main():
    p = catalogue()
    battrs = BinAttrs(s=EDGES)
    mattrs = MeshAttrs(p, p, boxsize=BOX, battrs=battrs, periodic=True,
                       backend=BACKEND)

    # Hard binning: exact, and unaffected by soft_pad.
    hard = count2(p, p, battrs=battrs, mattrs=mattrs, backend=BACKEND)['weight']

    unpadded = _counts(p, battrs, mattrs, 0.0)
    padded = _counts(p, battrs, mattrs, None)
    meta = padded['soft']

    un = unpadded['weight']
    pa = padded['weight']
    dev_un = un / hard - 1.0
    dev_pa = pa / hard - 1.0

    # The last-bin deviation as the padding grows.  soft_pad=0 is the old
    # estimator; the default is SOFT_PAD_DEFAULT.  The padded curve converges
    # to a small residual because a finite-width kernel still smooths counts,
    # but the edge leakage itself is gone.
    z_values = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
    last_dev = []
    for z in z_values:
        last_dev.append(_counts(p, battrs, mattrs, z)['weight'][-1] / hard[-1] - 1.0)
    last_dev = np.asarray(last_dev)

    print(f'{N} particles, {NS} linear bins in '
          f'[{EDGES[0]:g}, {EDGES[-1]:g}], soft_sigma={SIGMA:g}')
    print(f"default soft_pad: {meta['soft_pad']:g} "
          f"({meta['pad_lo']} lower + {meta['pad_hi']} upper buffer bins)")
    print(f'hard total: {hard.sum():.1f}')
    print(f'last bin hard count: {hard[-1]:.1f}')
    print(f'last bin deviation from hard: unpadded {100 * dev_un[-1]:+.2f}%, '
          f'padded {100 * dev_pa[-1]:+.2f}%')
    print(f'first bin deviation from hard: unpadded {100 * dev_un[0]:+.2f}%, '
          f'padded {100 * dev_pa[0]:+.2f}%')
    print('last-bin deviation vs soft_pad:')
    for z, value in zip(z_values, last_dev):
        print(f'  soft_pad={z:>3g}: {100 * value:+.2f}%')

    # ------------------------------------------------------------------
    # figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), layout='constrained')
    fig.suptitle(r'Soft-binning edge leakage: hard counts versus '
                 r'$\mathtt{soft\_pad}$', fontsize=11.5)
    colours = {'hard': '0.2', 'unpadded': '#c0392b', 'padded': '#1f77b4'}

    ax = axes[0, 0]
    ax.plot(CENTRES, hard, '-', color=colours['hard'], lw=2.2,
            label='hard binning (exact)')
    ax.plot(CENTRES, un, 'o--', ms=3.5, color=colours['unpadded'], lw=1.3,
            label=r'$\mathtt{soft\_pad}=0$ (old estimator)')
    ax.plot(CENTRES, pa, 's-', ms=3.2, color=colours['padded'], lw=1.3,
            label=r'$\mathtt{soft\_pad}=2$ (default)')
    ax.set_xlabel(r'$s$')
    ax.set_ylabel('pair counts')
    ax.set_title('Counts: hard, unpadded soft, padded soft')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.axhline(0.0, color=colours['hard'], lw=1.4, label='hard binning')
    ax.plot(CENTRES, 100 * dev_un, 'o--', ms=3.5, color=colours['unpadded'],
            lw=1.3, label=r'$\mathtt{soft\_pad}=0$')
    ax.plot(CENTRES, 100 * dev_pa, 's-', ms=3.2, color=colours['padded'],
            lw=1.3, label=r'$\mathtt{soft\_pad}=2$')
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$(\mathrm{soft}/\mathrm{hard} - 1)$  [%]')
    ax.set_title('Deviation from the exact hard counts')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    sl = slice(NS - 5, NS)
    ax.plot(CENTRES[sl], hard[sl], 'o-', color=colours['hard'], lw=2.0,
            label='hard binning')
    ax.plot(CENTRES[sl], un[sl], 'o--', color=colours['unpadded'], lw=1.4,
            label=r'$\mathtt{soft\_pad}=0$')
    ax.plot(CENTRES[sl], pa[sl], 's-', color=colours['padded'], lw=1.4,
            label=r'$\mathtt{soft\_pad}=2$')
    ax.axvline(EDGES[-1], color='0.6', lw=0.9, ls=':')
    ax.annotate('counting edge', xy=(EDGES[-1], hard[-1]),
                xytext=(-0.35, 0.55), textcoords='axes fraction',
                fontsize=8.5, color='0.35',
                arrowprops=dict(arrowstyle='->', color='0.45', lw=0.9))
    ax.set_xlabel(r'$s$')
    ax.set_ylabel('pair counts')
    ax.set_title('Upper-edge zoom: the unpadded last bin falls short')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.axhline(0.0, color=colours['hard'], lw=1.4, label='hard binning')
    ax.plot(z_values, 100 * last_dev, 'o-', ms=5, color=colours['padded'],
            lw=1.5, label='cuCount soft last bin')
    ax.axvline(SOFT_PAD_DEFAULT, color='0.75', lw=0.9, ls='--')
    ax.annotate('default', xy=(SOFT_PAD_DEFAULT, 100 * last_dev.max()),
                xytext=(4, -4), textcoords='offset points',
                fontsize=8.5, color='0.4')
    ax.set_xlabel(r'$\mathtt{soft\_pad}$  (kernel widths)')
    ax.set_ylabel(r'$(\mathrm{soft}_{\rm last}/\mathrm{hard}_{\rm last} - 1)$  [%]')
    ax.set_title('The edge bias disappears once buffer bins are added')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, 'soft_pad_leakage.png')
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
