"""Why a 2pcf loss can look frozen while a P(k) loss trains.

The cuCount soft adjoint is correct -- it matches finite differences in float64
and the float64 adjoint in float32 (``tests/test_soft_vjp_cuda.py`` and
``tests/test_soft_f32_cuda.py``).  What can go wrong is the *scale* of the loss
you hand it, and the precision of the positions you update.

A correlation function is normalized by ``norm ~ N^2``::

    xi = counts / norm

so ``d xi / d x`` is ``~1 / N^2`` times ``d counts / d x``.  A power-spectrum
loss has no such normalization, so in a joint objective the P(k) gradient can be
four to five orders of magnitude larger than the xi gradient.  With equal
weights the optimizer follows P(k), the positions move, P(k) improves -- and the
xi term is a negligible perturbation that never trains.  The xi *loss* does
change, but only because P(k) dragged the positions around, not because xi is
being fit.

The fix is to weight the two terms comparably.  The cleanest choice is to fit
counts instead of normalized xi: multiply the xi cotangent by ``norm`` (that is
exactly ``d mean((counts - target)^2) / d counts``).  Multiplying both terms by
a constant does not help; the *relative* weight is what matters.

Separately, if xi *is* the only term, float32 positions can still starve the
update: the gradient is tiny, positions sit at the box scale where a float32 ulp
is ``~1e-5``, and a default Adam ``eps=1e-8`` can dominate a ``1e-9`` gradient.
Use float64 positions (``JAX_ENABLE_X64=1``), a smaller Adam ``eps``, or the
counts loss above.

This example uses the exact float64 CPU adjoint and simulates float32/float64
position storage, so it isolates the optimization arithmetic from the kernel.
Run it from the repository root as::

    python scripts/_boot.py cucount/examples/soft_grad_training.py
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cucount.numpy import BinAttrs, MeshAttrs, Particles
from cucount.numpy import soft as nsoft

BOX = 500.0
N = 1500
EDGES = np.geomspace(1.0, 100.0, 21)
TUNING = {'nthreads': 4}
HERE = os.path.dirname(os.path.abspath(__file__))
K = 2.0 * np.pi * np.array(
    [[1, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [0, 2, 0], [1, 1, 0]],
    dtype=float) / BOX


def catalogue(n=N, width=4.0, seed=11):
    """Clustered sample: uniform pairs have almost no small-scale power."""
    rng = np.random.default_rng(seed)
    nclusters = 40
    centres = rng.uniform(0.0, BOX, (nclusters, 3))
    pos = centres[rng.integers(0, nclusters, n)]
    pos = (pos + rng.normal(0.0, width, (n, 3))) % BOX
    return pos, rng.uniform(0.5, 1.5, n)


def pk_loss_and_grad(pos, w, ptarget):
    """A cheap P(k)-like loss: a few Fourier modes of the density field."""
    phase = pos @ K.T
    F = np.sum(w[:, None] * np.exp(1j * phase), axis=0)
    norm = w.sum() ** 2 - w @ w
    P = np.abs(F) ** 2 / norm
    loss = np.mean((np.log10(P + 1e-300) - np.log10(ptarget + 1e-300)) ** 2)
    coeff = (2.0 * (np.log10(P + 1e-300) - np.log10(ptarget + 1e-300))
             / (len(K) * P * np.log(10.0)))
    grad = np.zeros_like(pos)
    for a, k in enumerate(K):
        ph = pos @ k
        # dP/dx_i = (2 w_i / norm) Re(conj(F) i k e^{i k.x})
        dP = -(2.0 * w / norm)[:, None] * (
            np.conj(F[a]) * k[None, :] * np.exp(1j * ph)[:, None]).imag
        grad += coeff[a] * dP
    return float(loss), grad


def xi_loss_and_grad(pos, w, ba, ma, txi, norm):
    """xi = counts / norm, and its exact soft adjoint."""
    p = Particles(pos, w)
    counts = nsoft.count2_soft(p, p, battrs=ba, mattrs=ma,
                               soft_accum='float64', backend='cpu',
                               tuning=TUNING)['weight']
    xi = counts / norm
    loss = np.mean((xi - txi) ** 2)
    # cotangent of mean((xi - txi)^2) with respect to counts
    cot = 2.0 * (xi - txi) / (xi.size * norm)
    out = nsoft.count2_soft_vjp(p, p, battrs=ba, mattrs=ma, cotangent=cot,
                                soft_accum='float64', backend='cpu',
                                tuning=TUNING)
    grad = (np.asarray(out['grad_position1'])
            + np.asarray(out['grad_position2']))
    return float(loss), grad, xi


def train(dtype, weight, steps=40, lr=3e-3, eps=1e-8):
    """Adam on L = L_xi + L_pk, storing positions as float32 or float64.

    ``weight='norm'`` fits counts instead of normalized xi, i.e. multiplies the
    xi cotangent by ``norm``.  ``weight='one'`` uses the raw normalized loss.
    """
    pos, w = catalogue()
    tpos, _ = catalogue(width=12.0, seed=12)
    upos, _ = catalogue(width=25.0, seed=13)
    ba = BinAttrs(s=EDGES)
    p = Particles(pos, w)
    ma = MeshAttrs(p, p, boxsize=BOX, battrs=ba, periodic=True)
    norm = w.sum() ** 2 - w @ w

    txi = nsoft.count2_soft(Particles(tpos, w), Particles(tpos, w),
                            battrs=ba, mattrs=ma, soft_accum='float64',
                            backend='cpu', tuning=TUNING)['weight'] / norm
    phase = upos @ K.T
    ft = np.sum(w[:, None] * np.exp(1j * phase), axis=0)
    ptarget = np.abs(ft) ** 2 / norm

    q = pos.astype(dtype)
    m = np.zeros_like(q)
    v = np.zeros_like(q)
    hist = []
    xi_init = g_xi_init = g_pk_init = None
    for t in range(1, steps + 1):
        x = q.astype(float)
        l_xi, g_xi, xi = xi_loss_and_grad(x, w, ba, ma, txi, norm)
        l_pk, g_pk = pk_loss_and_grad(x, w, ptarget)
        if t == 1:
            xi_init, g_xi_init, g_pk_init = xi.copy(), g_xi.copy(), g_pk.copy()
        g = (norm if weight == 'norm' else 1.0) * g_xi + g_pk
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        q = (q - lr * (m / (1 - 0.9 ** t))
             / (np.sqrt(v / (1 - 0.999 ** t)) + eps)).astype(dtype)
        hist.append((l_xi, l_pk, xi.copy(), float(np.max(np.abs(q - pos)))))
    return hist, pos, xi_init, g_xi_init, g_pk_init, norm


def main():
    runs = {}
    for label, dtype, weight in [
            ('f32 joint, equal weights', np.float32, 'one'),
            ('f64 joint, equal weights', np.float64, 'one'),
            ('f32 counts loss (xi x norm)', np.float32, 'norm')]:
        print(f'running {label} ...')
        runs[label] = train(dtype, weight)

    hist, pos, xi0, g_xi, g_pk, norm = runs['f32 joint, equal weights']
    ratio = np.max(np.abs(g_xi)) / np.max(np.abs(g_pk))
    print(f'norm = {norm:.3e}')
    print(f'max |d L_xi / d x| = {np.max(np.abs(g_xi)):.3e}')
    print(f'max |d L_pk / d x| = {np.max(np.abs(g_pk)):.3e}')
    print(f'xi / P(k) gradient ratio = {ratio:.3e}')

    for label, res in runs.items():
        h = res[0]
        print(f'{label:32s}: xi loss {h[0][0]:.6e} -> {h[-1][0]:.6e}, '
              f'P(k) loss {h[0][1]:.6e} -> {h[-1][1]:.6e}, '
              f'max move {h[-1][3]:.3e}')

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    ax = axes[0, 0]
    ax.bar(['xi term', 'P(k) term'],
           [np.max(np.abs(g_xi)), np.max(np.abs(g_pk))],
           color=['C0', 'C1'])
    ax.set_yscale('log')
    ax.set_ylabel(r'max $|\partial L / \partial \mathbf{x}|$')
    ax.set_title(f'Equal weights: xi gradient is {ratio:.1e} of P(k)')
    for i, val in enumerate([np.max(np.abs(g_xi)), np.max(np.abs(g_pk))]):
        ax.text(i, val, f'{val:.1e}', ha='center', va='bottom', fontsize=9)

    ax = axes[0, 1]
    for label, res in runs.items():
        ax.semilogy([x[0] for x in res[0]], label=label)
    ax.set_xlabel('Adam step')
    ax.set_ylabel(r'$\langle (\xi-\xi_{target})^2 \rangle$')
    ax.set_title('The xi loss with different weights / precision')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    centres = np.sqrt(EDGES[:-1] * EDGES[1:])
    ax.plot(centres, xi0, 'o-', color='0.5', label='initial')
    for label, res in runs.items():
        ax.plot(centres, res[0][-1][2], 'o-', ms=3, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('separation s')
    ax.set_ylabel(r'$\xi(s)$')
    ax.set_title('The 2pcf barely moves under the joint loss')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    pos32 = pos.astype(np.float32)
    ulp = float(np.median(np.abs(np.spacing(pos32))))
    lrs = np.geomspace(1e-5, 1e-1, 25)
    one_step = lrs * np.max(np.abs(g_xi))
    frac = []
    for lr in lrs:
        q = (pos32 + (lr * g_xi).astype(np.float32)).astype(np.float32)
        frac.append(float(np.mean(np.any(q != pos32, axis=1))))
    ax.loglog(lrs, one_step / ulp, color='C0',
              label=r'$lr \times$ max xi gradient')
    ax.axhline(1.0, color='C3', ls='--', label='one f32 ulp')
    ax.axhline(1e-8 / ulp, color='0.6', ls=':', label='typical Adam eps / ulp')
    ax.set_xlabel('learning rate (xi term only)')
    ax.set_ylabel('position change / f32 ulp')
    ax.set_title('Why f32 updates can also round away')
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle('cuCount soft gradients are correct: check the loss weight '
                 'and the position precision', fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, 'figures', 'soft_grad_training.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
