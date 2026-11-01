"""Why a normalized correlation-function loss can look frozen in float32.

The soft adjoint is tested against finite differences in float64 and against the
float64 reference in float32 (``tests/test_soft_f32_cuda.py``), so the gradient
*computation* is not the problem.  The problem can be the *update*: a
correlation function is normalized by ``norm ~ N^2``, so the gradient of
``xi = counts / norm`` is ``~1/N^2`` of the raw counts gradient.  Positions live
around the box size, where a float32 ulp is already ``~1e-5``.  A gradient of
``1e-8`` times a learning rate of ``1e-2`` is ``1e-10``: adding it to a float32
position rounds back to the same bits, the positions never move, and the loss is
literally constant.  The power spectrum does not have this ``1/N^2``
normalization, which is why gradients through it can flow while xi stalls.

This example measures that scale separation with the CPU soft adjoint and plots
when a float32 update starts to change positions.  Run it from the repository
root as ``python scripts/_boot.py cucount/examples/soft_grad_precision.py``.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cucount.numpy import BinAttrs, MeshAttrs, Particles, count2_soft, count2_soft_vjp

BOX = 500.0
N = 2500
EDGES = np.geomspace(1.0, 100.0, 21)
HERE = os.path.dirname(os.path.abspath(__file__))


def catalogue(seed=11):
    """A clustered sample: uniform pairs have almost none of the small scales."""
    rng = np.random.default_rng(seed)
    nclusters = 40
    centres = rng.uniform(0.0, BOX, (nclusters, 3))
    pos = centres[rng.integers(0, nclusters, N)]
    pos = (pos + rng.normal(0.0, 4.0, (N, 3))) % BOX
    return pos, rng.uniform(0.5, 1.5, N)


def gradient(pos, w, ba, ma, cotangent):
    p = Particles(pos, w)
    out = count2_soft_vjp(p, p, battrs=ba, cotangent=cotangent, mattrs=ma,
                          soft_accum='float64', backend='cpu')
    return np.asarray(out['grad_position1']) + np.asarray(out['grad_position2'])


def main():
    pos, w = catalogue()
    ba = BinAttrs(s=EDGES)
    p = Particles(pos, w)
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    counts = count2_soft(p, p, battrs=ba, mattrs=ma, soft_accum='float64',
                         backend='cpu')['weight']
    norm = w.sum() ** 2 - (w ** 2).sum()
    xi = counts / norm

    # The gradient of sum(xi) -- what a simple xi loss hands the adjoint.
    g_xi = gradient(pos, w, ba, ma, np.ones_like(counts) / norm)
    g_counts = gradient(pos, w, ba, ma, np.ones_like(counts))
    rms_xi = float(np.sqrt(np.mean(g_xi ** 2)))
    rms_counts = float(np.sqrt(np.mean(g_counts ** 2)))
    print(f'norm = {norm:.3e}')
    print(f'rms d(sum counts)/dposition = {rms_counts:.3e}')
    print(f'rms d(sum xi)/dposition     = {rms_xi:.3e}  '
          f'(ratio {rms_xi / rms_counts:.3e}, expected ~1/norm)')

    # Split the loss by separation: a fit at small s only weights the first bins.
    nbin = counts.size
    masks = {'s < 4': np.arange(nbin) < 5,
             '4 < s < 25': (np.arange(nbin) >= 5) & (np.arange(nbin) < 10),
             's > 25': np.arange(nbin) >= 10}
    rms_by_band = {}
    for label, mask in masks.items():
        cot = mask.astype(float) / norm
        g = gradient(pos, w, ba, ma, cot)
        rms_by_band[label] = float(np.sqrt(np.mean(g ** 2)))
        print(f'  band {label:12s}: rms gradient {rms_by_band[label]:.3e}')

    # Float32 positions: an update is lost unless lr * gradient exceeds the ulp.
    pos32 = pos.astype(np.float32)
    ulp = np.abs(np.spacing(pos32))
    lrs = np.geomspace(1e-4, 1e2, 25)
    moved32, moved64 = [], []
    for lr in lrs:
        p32 = (pos32 + (lr * g_xi).astype(np.float32)).astype(np.float32)
        moved32.append(float(np.mean(np.any(p32 != pos32, axis=1))))
        p64 = pos + lr * g_xi
        moved64.append(float(np.mean(np.any(p64 != pos, axis=1))))
    moved32 = np.asarray(moved32)
    moved64 = np.asarray(moved64)
    # Learning rate at which a typical component of lr * g_xi reaches the ulp.
    lr_cross = float(np.median(ulp) / max(np.sqrt(np.mean(g_xi ** 2)), 1e-300))
    print(f'median float32 position ulp = {np.median(ulp):.3e}')
    print(f'lr at which a typical update reaches that ulp = {lr_cross:.3e}')
    print(f'at lr=1e-2, fraction of particles whose float32 position changes = '
          f'{moved32[np.argmin(abs(lrs - 1e-2))]:.3f}')

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    ax = axes[0, 0]
    centres = np.sqrt(EDGES[:-1] * EDGES[1:])
    ax.plot(centres, xi, 'o-', color='C0')
    ax.axhline(0.0, color='0.7', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('separation s')
    ax.set_ylabel(r'$\xi(s)$ (raw counts / norm)')
    ax.set_title('A normalized correlation function')

    ax = axes[0, 1]
    labels = list(rms_by_band)
    vals = [rms_by_band[k] for k in labels]
    ax.bar(labels, vals, color=['C0', 'C1', 'C2'])
    ax.set_yscale('log')
    ax.set_ylabel(r'rms $|\partial \sum \xi / \partial \mathbf{x}|$')
    ax.set_title('Gradient carried by each separation band')

    ax = axes[1, 0]
    typical = lrs * rms_xi
    ax.loglog(lrs, typical, color='C0', label=r'$lr \times$ rms gradient')
    ax.axhline(np.median(ulp), color='C3', ls='--',
               label='median float32 position ulp')
    ax.axvline(lr_cross, color='0.5', ls=':')
    ax.set_xlabel('learning rate')
    ax.set_ylabel('position change / resolution')
    ax.set_title('The update is below float32 resolution')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.semilogx(lrs, moved32, 'o-', color='C3', label='float32 positions')
    ax.semilogx(lrs, moved64, 'o-', color='C0', label='float64 positions')
    ax.axvline(lr_cross, color='0.5', ls=':')
    ax.set_xlabel('learning rate')
    ax.set_ylabel('fraction of particles that move')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('Why the loss repeats: no bit changes')
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle('cuCount soft xi gradients: computation is right, float32 '
                 'updates can be too small to land', fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, 'figures', 'soft_grad_precision.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
