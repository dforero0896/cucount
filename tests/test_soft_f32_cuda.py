"""float32 samples through the soft frontend, with JAX_ENABLE_X64 unset.

The soft kernel always computes in float64: separations, logits, window search
and accumulation are double. "float32 support" only means the FFI boundary
accepts and returns float32 buffers, so a JAX user does not have to set
``JAX_ENABLE_X64`` (which is what the hard count2 path still requires). The
float32 twin widens the samples on the way in and rounds the results once on the
way out, so the counts should agree with the float64 path to a few 1e-7.

``jax_enable_x64`` is a process-wide configuration flag, so the check runs in a
subprocess started without the variable; the parent process keeps float64 (other
tests in this directory depend on it).
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from cucount.numpy import cudalib

needs_cuda = pytest.mark.skipif(
    cudalib is None or cudalib.device_count() == 0,
    reason='CUDA extension not built, or no GPU visible')

BOX = 300.0
N = 96
EDGES = np.linspace(1.0, 100.0, 11)
ELLS = np.array([0, 2, 4])
CUT = 1e-13

# Runs in the subprocess. Kept short and self-contained: it must not import the
# test module (pytest is not on the subprocess path) and must fail loudly with a
# non-zero exit code if any check breaks.
SCRIPT = r'''
import sys
sys.meta_path[:] = [f for f in sys.meta_path
                    if 'ScikitBuildRedirectingFinder' not in type(f).__name__]

import numpy as np
import jax
import jax.numpy as jnp

import cucount.jax as cjax
import cucount.numpy as cn
import cucount.numpy.soft as nsoft

assert not jax.config.read('jax_enable_x64')

BOX, N, CUT = 300.0, 96, 1e-13
EDGES = np.linspace(1.0, 100.0, 11)
ELLS = np.array([0, 2, 4])

rng = np.random.default_rng(7)
pos1, pos2 = rng.random((N, 3)) * BOX, rng.random((N, 3)) * BOX
w1, w2 = rng.random(N) + 0.5, rng.random(N) + 0.5

battrs = cn.BinAttrs(s=EDGES)
battrs_pole = cn.BinAttrs(s=EDGES, pole=(ELLS, 'z'))
p1, p2 = cn.Particles(pos1, w1), cn.Particles(pos2, w2)
# The reference must see the same periodic mesh, otherwise the pair sets differ.
mnp = cn.MeshAttrs(pos1, pos2, battrs=battrs, periodic=True, boxsize=BOX)
mnp_pole = cn.MeshAttrs(pos1, pos2, battrs=battrs_pole, periodic=True, boxsize=BOX)
ref = nsoft.count2_soft(p1, p2, battrs=battrs, soft_cutoff=CUT, mattrs=mnp)['weight']

mattrs = cjax.MeshAttrs(pos1, pos2, battrs=battrs, periodic=True, boxsize=BOX)
# Build the particles from *jax* arrays: _format_positions casts a numpy array to
# float64, so a numpy float32 sample would silently become float64.
j1 = cjax.Particles(jnp.asarray(pos1, jnp.float32), jnp.asarray(w1, jnp.float32))
j2 = cjax.Particles(jnp.asarray(pos2, jnp.float32), jnp.asarray(w2, jnp.float32))
assert j1.positions.dtype == np.float32

got = cjax.count2_soft(j1, j2, battrs=battrs, soft_cutoff=CUT, mattrs=mattrs)['weight']
assert got.dtype == np.float32, got.dtype
rel = np.max(np.abs(np.asarray(got) - ref)) / np.max(np.abs(ref))
assert rel < 2e-5, rel

# Gradients: float32 samples, analytic float64 adjoint as the reference.
g = rng.normal(size=ref.shape)

def loss(a1, b1, a2, b2):
    h = cjax.count2_soft(cjax.Particles(a1, b1), cjax.Particles(a2, b2),
                         battrs=battrs, soft_cutoff=CUT, mattrs=mattrs)['weight']
    return jnp.sum(h * jnp.asarray(g, dtype=h.dtype))

grads = jax.grad(loss, argnums=(0, 1, 2, 3))(
    jnp.asarray(pos1, jnp.float32), jnp.asarray(w1, jnp.float32),
    jnp.asarray(pos2, jnp.float32), jnp.asarray(w2, jnp.float32))
assert all(np.asarray(x).dtype == np.float32 for x in grads), [x.dtype for x in grads]

adj = nsoft.count2_soft_vjp(p1, p2, battrs=battrs, cotangent=g,
                            soft_cutoff=CUT, mattrs=mnp)
for name, a, b in zip(('pos1', 'w1', 'pos2', 'w2'), grads,
                      (adj['grad_position1'], adj['grad_weight1'],
                       adj['grad_position2'], adj['grad_weight2'])):
    r = np.max(np.abs(np.asarray(a, dtype=np.float64) - b)) / np.max(np.abs(b))
    assert r < 5e-5, (name, r)

# The flat (bin, multipole) slot layout must reshape in float32 too.
refp = nsoft.count2_soft(p1, p2, battrs=battrs_pole, soft_cutoff=CUT,
                         mattrs=mnp_pole)['weight']
mpole = cjax.MeshAttrs(pos1, pos2, battrs=battrs_pole, periodic=True, boxsize=BOX)
gotp = cjax.count2_soft(j1, j2, battrs=battrs_pole, soft_cutoff=CUT,
                        mattrs=mpole)['weight']
assert gotp.shape == refp.shape and gotp.dtype == np.float32
r = np.max(np.abs(np.asarray(gotp) - refp)) / np.max(np.abs(refp))
assert r < 2e-5, r

print('F32 OK')
'''


def _run_without_x64():
    env = dict(os.environ)
    for key in ('JAX_ENABLE_X64', 'JAX_ENABLE_X64 '):
        env.pop(key, None)
    # The subprocess cannot rely on the pytest entry point having fixed sys.path.
    existing = env.get('PYTHONPATH')
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + ([existing] if existing else []))
    proc = subprocess.run([sys.executable, '-c', SCRIPT], env=env,
                          capture_output=True, text=True)
    return proc


@needs_cuda
def test_float32_samples_work_without_enabling_x64():
    """No JAX_ENABLE_X64: float32 in, float32 out, float64 accuracy."""
    proc = _run_without_x64()
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert 'F32 OK' in proc.stdout
