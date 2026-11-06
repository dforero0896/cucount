# cucount: CUDA-Powered Pair Counts

**cucount** is a high-performance CUDA implementation for computing pair counts (positions - spins), and triplet counts, optimized for GPUs. It provides both NumPy and JAX interfaces depending on your workflow.

> ⚠️ A CUDA-capable GPU is required for the default backend. Building with `-DCUCOUNT_BUILD_CUDA=OFF` skips CUDA entirely: `cucount.numpy` still imports (the attribute classes live in a CUDA-free extension) and serves the experimental `backend='cpu'`; any CUDA call then raises with a message naming the flag.

---

## 📦 Installation

You can install `cucount` directly via pip:

```bash
pip install git+https://github.com/adematti/cucount.git
```

The JAX API (through FFI) will be built automatically if the `jax.ffi` library is found during installation.

---

## 🧮 NumPy API

Use the NumPy API if you're not using JAX. All data stays on the **host (CPU)** and is internally transferred to the GPU.

### Example

```python
import numpy as np
from cucount.numpy import count2, Particles, BinAttrs

# Prepare catalogs
size = int(1e6)
boxsize = np.array((3000.,) * 3)
rng = np.random.RandomState(seed=42)

def generate_catalog(rng, size):
    offset = boxsize
    positions = rng.uniform(0., 1., (size, 3)) * boxsize + offset
    weights = rng.uniform(0., 1., size)
    return positions, weights

positions1, weights1 = generate_catalog(rng, size)
positions2, weights2 = generate_catalog(rng, size)

# Define binning and line-of-sight
edges = (np.linspace(1., 201, 201), np.linspace(-1., 1., 201))
los = 'midpoint'

# Compute pair counts, on 4 GPUs
# If you want to go multi-node, MPI is a good option
particles1 = Particles(positions1, weights1)
particles2 = Particles(positions2, weights2)
battrs = BinAttrs(s=edges[0], mu=(edges[1], los))
counts = count2(particles1, particles2, battrs=battrs, tuning={'nthreads': 4})
# counts is a dictionary with key "weight"
```

### Tuning

Backend tuning options go through the `tuning=` keyword as a flat dict addressed
to the selected backend; unknown keys are rejected by name. The CUDA backend
accepts `nthreads` (number of GPUs on the node; this replaces the deprecated
`nthreads=` keyword). The CPU backend accepts `nthreads` (CPU threads), `isa`
(pin a Highway target, e.g. `'AVX2'`) and `scatter` (`'scalar'` or `'binmajor'`):

```python
counts = count2(particles1, particles2, battrs=battrs, backend='cpu',
                tuning={'nthreads': 32, 'isa': 'AVX2'})
```

`backend='compare'` runs both backends, so there the dict nests per backend:
`tuning={'cpu': {...}, 'cuda': {...}}`.

### CPU Backend
The `count2()` function in the NumPy API supports an experimental CPU backend:

```python
counts = count2(particles1, particles2, backend='cpu')
```

`count2`, `count3` and `count3close` are all supported, through the numpy API
and the JAX one alike. For `count2`: every binning variable (s, mu, rp, pi,
theta, pole, k) in any combination, every line of sight, the cartesian and
angular meshes, s and theta selections, jackknife splits, individual weights,
spin/shear (galaxy-shear and shear-shear channels), and bitwise (PIP),
negative and 1D angular weights. For the triplet counts: legs binned in s or
theta with multipoles to ell = 5, the optional (2, 3) axis, selections and
vetoes, and the 3-dimensional angular upweight.

The CPU backend declines nothing the CUDA one serves. What neither serves --
multipoles above the kernels' cache, N-dimensional angular weights in `count2`,
spin with other than two components -- is a limit of both kernels, checked once
in the frontend before a backend is chosen, so every backend refuses the same
request the same way.

Internally two paths sit behind the one entry point: a vectorised Highway
kernel for s, (s, mu) and (s, pole) on a cartesian mesh, and a scalar path for
everything else. Which one runs is an implementation detail; the result is the
same either way, and the `tuning` knobs below (`isa`, `scatter`) only reach
the vectorised one.
`backend='compare'` runs both backends and raises if they disagree. More features are
expected to be added over time; please open an issue if a particular feature is
important to you. See [cpu/README.md](cpu/README.md) for details.

The CPU kernels are written in an ISA-portable way using Google Highway. One should be able to build the code once and run it on a wide variety of CPU architectures with SIMD acceleration. Many optimization opportunities remain, however, so performance is not expected to match Corrfunc, for example.

### Repository layout

Each backend owns a subdirectory with the same shape — [cuda/](cuda/) (the default
`cucountlib.cuda` extension and the JAX FFI module) and [cpu/](cpu/) (the portable
Highway backend) — while backend-neutral code lives at the root: `include/attrs.h`
(the attribute classes, compiled into the CUDA-free `cucountlib.attrs` extension),
`include/cmath.h` (scalar per-pair math shared by both backends) and
`include/common.h` (type definitions). `-DCUCOUNT_BUILD_CUDA=OFF` /
`-DCUCOUNT_BUILD_CPU=OFF` skip either backend.

The Python frontend does not mirror that split, because after the two
backends converged there was little left to separate: `cucount/numpy/` and
`cucount/jax/` each hold both backends in one module, side by side. What is
per backend is the extension handle, its tuning keys and the call itself;
everything else — the attribute classes, `Particles`, what a request may ask
for, the dispatch, how a disagreement is reported — is backend-neutral.

---

## ⚡ JAX API

Use the JAX interface if JAX is already part of your codebase.

### Why a separate JAX API?

1. **JAX preallocates GPU memory**, which can cause `cudaMalloc` to fail if using the NumPy backend.
2. **Passing device arrays** (from JAX) directly avoids host-device transfers.
3. **JAX's distributed capabilities** (e.g., `shard_map`) are well-suited for scaling.

### Backend selection

The JAX API takes the same `backend=` keyword as the NumPy one, except that
there is no `'compare'`: a traced computation runs one backend throughout.

```python
counts = cucount.jax.count2(*particles, battrs=battrs, mattrs=mattrs, backend='cpu')
```

Multi-device works the same way on either backend — catalogue 1 is split
across the devices of a `sharding_mesh` and the counts are summed — so the
only difference is what the devices are. For several CPU devices, tell XLA
before jax starts:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python run.py
```

`cucount.jax.set_cpu_nthreads(n)` sets the threads each CPU FFI call uses.
It is per device, so N devices at M threads occupy N * M cores.

---

### 🚀 Single-Device JAX Example

```python
import jax
from jax import config
config.update("jax_enable_x64", True)  # Currently only double precision is supported

from cucount.jax import count2, Particles, BinAttrs

# Assume positions1, positions2, weights1, weights2 are already defined
edges = (np.linspace(1., 201, 201), np.linspace(-1., 1., 201))
los = 'midpoint'
battrs = BinAttrs(s=edges[0], mu=(edges[1], los))

particles1 = Particles(positions1, weights1)
particles2 = Particles(positions2, weights2)

counts = count2(particles1, particles2, battrs=battrs)
# counts is a dictionary with key "weight"
```

---

### 🧩 Multi-Device (Distributed) JAX Example

Using [shard\_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html), you can parallelize over multiple devices:

```python
import jax
jax.config.update("jax_enable_x64", True)
# Initialize distributed environment (if needed)
jax.distributed.initialize()
from cucount.jax import count2, Particles, BinAttrs, create_sharding_mesh

battrs = BinAttrs(s=edges[0], mu=(edges[1], los))

# Run distributed pair counts
with create_sharding_mesh():
    # Pass exchange=True if input is distributed over multiple processes
    particles1 = Particles(positions1, weights1)
    particles2 = Particles(positions2, weights2)
    counts = count2(particles1, particles2, battrs=battrs)
```

---

## Angular Upweights

Angular (PIP) upweights are passed through `WeightAttrs(angular=...)`, as a function of the angular
separation θ **in degrees**. Tabulate it either at sample points (`sep`, linearly interpolated,
1 outside the range) or in bins (`edges`, piecewise-constant, 1 outside the range).
`weight` may be N-dimensional, in which case provide one `sep`/`edges` array per dimension.

```python
import numpy as np
from cucount.numpy import count2, Particles, BinAttrs, WeightAttrs

# Angular upweights, tabulated at sep (in degrees)
sep = np.linspace(0., 0.1, 100)
weight = 1. + np.linspace(0., 1., sep.size)

battrs = BinAttrs(s=np.linspace(1., 201., 201), mu=(np.linspace(-1., 1., 201), 'midpoint'))
wattrs = WeightAttrs(angular=dict(sep=sep, weight=weight))

particles1 = Particles(positions1, weights1)
particles2 = Particles(positions2, weights2)
counts = count2(particles1, particles2, battrs=battrs, wattrs=wattrs)
```

Angular upweights typically come with bitwise (PIP) weights: pass integer arrays as weights to
`Particles` (they are automatically recognized as bitwise weights) and add `bitwise` to `WeightAttrs`.

```python
# weights1 = [individual_weights, bitwise_weight_0, bitwise_weight_1] (integer dtype => bitwise)
particles1 = Particles(positions1, weights1)
wattrs = WeightAttrs(bitwise=dict(weights=particles1.get('bitwise_weight')),
                     angular=dict(sep=sep, weight=weight))
counts = count2(particles1, particles2, battrs=battrs, wattrs=wattrs)
```

Same syntax with `cucount.jax`.

---

## Triplet Counts

`count3` computes *factorized* triplet counts: for each particle of catalog 1, catalog 2 is binned as
a function of the (1, 2) separation and catalog 3 as a function of the (1, 3) separation, accumulating
`w1 * w2(r12) * w3(r13)`. There is no binning nor selection on the (2, 3) separation.
`count3close` is the *close*-triplet version, which additionally accepts `battrs23` and (2, 3) selections.

```python
import numpy as np
from cucount.numpy import count3, count3close, Particles, BinAttrs, SelectionAttrs

particles = Particles(positions, weights)

sedges = np.linspace(0., 40., 11)
battrs12 = BinAttrs(s=sedges, pole=([0], 'firstpoint'))
battrs13 = BinAttrs(s=sedges, pole=([0, 2], 'firstpoint'))
# Restrict to close triplets: angular separation < 20 deg
sattrs = SelectionAttrs(theta=(0., 20.))

counts = count3(particles, particles, particles,
                battrs12=battrs12, battrs13=battrs13,
                sattrs12=sattrs, sattrs13=sattrs)
# counts is a dictionary with key "weight"

# Explicit triplets, optionally with binning in the (2, 3) separation
counts = count3close(particles, particles, particles,
                     battrs12=battrs12, battrs13=battrs13,
                     sattrs12=sattrs, sattrs13=sattrs)
```

For periodic boxes, `count3_analytic(battrs12, battrs13, mattrs)` gives the analytic (random) counts.
As above, `cucount.jax` provides the same functions for device arrays.

---

## 📦 lsstypes Wrapper

`cucount.types` wraps the low-level counters and returns
[lsstypes](https://github.com/cosmodesi/lsstypes) containers instead of raw arrays:
it handles the normalization (sum of weights, self-pair removal, jackknife splits) for you,
and the result carries the coordinates, edges and metadata --- ready to be combined into a
correlation function estimator, written to disk and plotted.

Because normalization is computed internally, pass **one** `Particles` instance for an autocorrelation
(instead of repeating it twice).

```python
import numpy as np
from cucount.numpy import Particles, BinAttrs, WeightAttrs, MeshAttrs, SplitAttrs
from cucount.types import count2, count2_analytic
import lsstypes as types

battrs = BinAttrs(s=np.linspace(0., 200., 21), mu=(np.linspace(-1., 1., 21), 'midpoint'))
wattrs = WeightAttrs()

data = Particles(data_positions, data_weights)
randoms = Particles(random_positions, random_weights)

DD = count2(data, battrs=battrs, wattrs=wattrs)['weight']              # autocorrelation
DR = count2(data, randoms, battrs=battrs, wattrs=wattrs)['weight']     # cross-correlation
RR = count2(randoms, battrs=battrs, wattrs=wattrs)['weight']
RD = DR.clone(value=DR.value()[:, ::-1])  # reverse mu for RD

correlation = types.Count2Correlation(estimator='landyszalay', DD=DD, DR=DR, RD=RD, RR=RR)
correlation.write('correlation.h5')
correlation.project(ells=[0, 2, 4]).plot()
```

For a periodic box, replace `RR` by the analytic counts:

```python
mattrs = MeshAttrs(data, boxsize=boxsize, battrs=battrs, periodic=True)
DD = count2(data, battrs=battrs, wattrs=wattrs, mattrs=mattrs)['weight']
RR = count2_analytic(battrs=battrs, mattrs=mattrs)
correlation = types.Count2Correlation(estimator='natural', DD=DD, RR=RR)
```

Passing `splits` to `Particles` together with `SplitAttrs` returns a `Count2Jackknife`,
which carries the per-split counts required for jackknife covariance estimates:

```python
data = data.clone(splits=splits)  # splits: integer array in [0, nsplits)
spattrs = SplitAttrs(mode='jackknife', nsplits=nsplits)
DD = count2(data, battrs=battrs, wattrs=wattrs, spattrs=spattrs, mattrs=mattrs)['weight']
```

`cucount.types` also provides `count3`, `count3close` and `count3_analytic`, returning
`Count3` / `Count3Poles` containers:

```python
from cucount.types import count3

counts = count3(data, battrs12=battrs12, battrs13=battrs13,
                sattrs12=sattrs, sattrs13=sattrs)['weight']
```

All these functions accept `cucount.jax` `Particles` as well, and dispatch to the JAX backend
automatically.

### 🌫️ Soft binning

`count2` places every pair in exactly one bin. `count2_soft` shares it between
neighbouring bins with a Gaussian kernel around each bin centre instead; a kernel
narrow against the bin recovers `count2` (exactly in the `'softmax'` mode).
The default `soft_norm='gaussian'` uses the
discrete-quadrature weights `A_i exp(-x_i)` with
`A_i = dmu_i / (sqrt(2 pi) sigma_i)`: each pair's weights sum to one up to
quadrature error (about 0.5% at the default half-bin sigma; kernel mass landing
outside the counted range is dropped, which `soft_pad` recovers), the kernel is a
single pass, and no bin competes with its neighbours for a pair's weight, so
gradients stay well conditioned. `soft_norm='softmax'` restores the old
behaviour on the separation axis, renormalising each pair's window exactly at
the cost of a second pass. The `mu` axis of a multipole request is always
renormalised, whatever `soft_norm` says: `mu = s_parallel / s` lives in
`[-1, 1]`, a bounded domain where an un-renormalised kernel would lose up to
half its mass at the poles (and padding is not an option), so `M_0 == 1` and the
monopole is exactly the isotropic soft count.

**Coming from an earlier release:** the default used to be the softmax
normalisation, so pass `soft_norm='softmax'` to reproduce older numbers. The two
differ by about 0.4% in the bulk and by several percent at the outermost bins,
where the kernel tail leaves the counted range; `CHANGELOG.md` has the details.
Note also that `soft_sigma` is a *fraction of each bin's width* by default, so on
a wide log grid an absolute width is not the same kernel at both ends.

It returns the same `Count2` / `Count2Poles` containers, and `attrs` records the soft
configuration (`soft_sigma`, `soft_norm`, `soft_cutoff`, `soft_multipoles`, ...).

```python
from cucount.types import count2_soft

sedges = np.logspace(np.log10(1.), np.log10(100.), 21)  # any strictly increasing grid
battrs = BinAttrs(s=sedges)
DD = count2_soft(data, battrs=battrs, wattrs=wattrs, mattrs=mattrs)['weight']
print(DD.attrs['soft_sigma'], DD.attrs['soft_sigma_kind'])  # 0.5, 'width'

# Soft multipoles: the mu axis is smoothed too, so the poles stay differentiable
battrs = BinAttrs(s=sedges, pole=([0, 2, 4], 'midpoint'))
DD = count2_soft(data, battrs=battrs, wattrs=wattrs, mattrs=mattrs,
                 soft_mu_sigma=0.05)['weight']  # Count2Poles
```

`count2_soft` is JAX-traceable: it composes with `jax.jit`, `jax.grad` and
`jax.value_and_grad`, and the gradient runs through the analytic CUDA adjoint (one extra
kernel, not finite differences). The lsstypes containers are not pytrees, so a jit-ed
function must return arrays --- take `values('counts')`, stacking the poles for
`Count2Poles`:

```python
@jax.jit
def counts(p, w):
    DD = count2_soft(Particles(p, w), battrs=battrs, mattrs=mattrs)['weight']
    return jnp.stack([pole.values('counts') for pole in DD], axis=-1)

jax.grad(lambda w: jnp.sum(counts(positions, w)))(weights)  # -> positions/weights
```

The mesh has to be passed explicitly (`mattrs=...`), exactly as for the hard `count2`: it
cannot infer the box from traced positions. The hard `count2` has no adjoint, so it is not
differentiable --- gradients need the soft kernel.

`soft_sigma` defaults to half a bin: on a non-uniform grid it is half of *each* bin, so
`attrs['soft_sigma']` is the multiplier of the local width (`soft_sigma_kind='width'`),
while on an arithmetic grid it is an absolute width (`'absolute'`). Pass a number for a
constant width in separation units.

`soft_pad` controls the edge truncation. The kernel spreads each pair with a Gaussian
of infinite support, but only traverses pairs inside `[s_edges[0], s_edges[-1])`, so the
outermost bins average a *truncated* kernel and pick up an edge bias that grows with
the steepness of the correlation function and the kernel width. Its sign depends on
the normalisation: the softmax redistributes the missing tail within the window it
keeps, while the gaussian default simply drops it. `soft_pad=z` extends the counting range by `z` kernel widths with
**buffer bins** (geometric continuation on a log grid, arithmetic otherwise), lets the
kernel deposit over the real and buffer centres, and drops the buffer bins from the
result. `None` selects the pyfcfc default `z=2`; `soft_pad=0` restores the previous
estimator bit for bit. In a periodic box the extended maximum edge is clamped below
`L/2` (the minimum-image limit), and the lower edge never goes below zero. The buffer
bins make the traversal enumerate the extra pairs in those shells --- roughly
`exp(3 z f) - 1` more at the upper edge for log bins --- and `attrs` records the layout
as `soft_pad`, `soft_pad_lo`, `soft_pad_hi` and `soft_nbins_padded`.

`soft_cutoff` drops the Gaussian tail (`soft_norm='softmax'` renormalizes the remaining
window, so weight stays conserved; under `'gaussian'` the dropped tail is a small
quadrature error) and `soft_taps='fast'` trades ~1e-6 weight accuracy for speed on
non-uniform grids. For multipoles the mu quadrature is a uniform grid over the full `[-1, 1]` (mu is
not folded, so odd multipoles keep their sign): `soft_mu_bins` sets its resolution and
`soft_mu_edges` takes caller-supplied cells --- arithmetic ones still take the fast
recurrence path. The small `soft_sigma -> 0` limit is the honest one for multipoles:
softening mu shifts the poles at `O(sigma_mu^2)`, so it is opt-in.

Samples may be `float32` or `float64`; `cucount.jax.soft` picks the matching FFI entry point
from the particle dtype, so soft counts work without setting `JAX_ENABLE_X64` (the float32
samples are widened once at the boundary and the result narrowed once on the way out --- the
arithmetic stays `float64`). Independently, `soft_accum` selects the precision of the
per-block shared-memory histogram: `'float32'` (the default) is faster because Ampere's
`atomicAdd(double*)` is a compare-and-swap loop, and costs a few `1e-7` relative on the
counts; `'float64'` restores the bit-comparable path for conservation tests.

A float32 sample dtype does not weaken the adjoint: the kernel computes it in `float64`
and narrows only the returned gradient, and the float32 test compares it against the
`float64` adjoint to `5e-5`. Two things *can* make a fit look frozen even though the
gradient is correct. First, the weighting: `xi = counts / norm` carries a
`1/norm ~ 1/N^2` factor, so the gradient of a normalized-xi loss is `~1/N^2` of the raw
counts gradient. A power-spectrum loss has no such factor, so in a joint objective the
P(k) term can dominate the update by four or five orders of magnitude: P(k) trains and
moves the particles while the xi term never gets a comparable step. Fit counts instead
(multiply the xi cotangent by `norm`), or otherwise weight the two terms to comparable
gradient scale; rescaling the whole loss does not change the relative weight. Second,
the update precision: even when xi is the only term, a tiny gradient added to a float32
position near the box size can round back to the same bits, and a default Adam `eps`
can dominate a `~1e-9` gradient. Try `JAX_ENABLE_X64=1` (float64 positions), a smaller
Adam `eps`, or the counts loss above. `examples/soft_grad_training.py` reproduces the
joint P(k)/xi case with both position precisions and shows the counts-loss fix;
`examples/soft_grad_precision.py` measures the float32 ulp threshold.

Soft binning runs on both backends. `backend='cuda'` scatters into per-block shared-memory
histograms; `backend='cpu'` scatters into one `float64` histogram per thread and reduces
them at the end, so the CPU kernel has no `soft_accum`/`soft_taps` occupancy trade-off --- it
accumulates exactly, and reports `soft['path'] == 'threads'` with `soft['accum'] ==
'float64'`. Both backends resolve the request through the same shared header
(`include/soft.h`), so the window, the per-bin sigmas and the caller's mu cells cannot
drift between them; only the per-pair scatter and the adjoint are written twice. The CPU
adjoint is the same analytic one (positions, individual weights and `soft_sigma`), so
`count2_soft(...)['weight']` stays differentiable on a machine without a GPU.

On CPU the soft count costs about 4.7x the hard one per core, and 2.4x at the best wall time
achievable (the soft kernel keeps scaling one thread doubling longer). Quote those numbers
only with the thread count pinned: `backend='cpu'` defaults to the *affinity mask*, so a
timing taken on a login node is an oversubscription measurement.

JAX soft counts (`cucount.jax.soft`) run on both backends. The CPU FFI handler computes
the histogram with one private `float64` histogram per OpenMP thread, exactly like the
NumPy frontend, and exposes the same four FFI targets (`count2_soft[_f32]` and
`count2_soft_vjp[_f32]`); `backend='cpu'` therefore supports `jax.jit`, `jax.grad` and the
float32 sample path without a GPU. On CPU the reported `soft['accum']` is `'float64'` and
`soft['path']` is `'threads'`, matching `cucount.numpy.soft`.
Both backends reject spin, angular upweights and split/jackknife requests instead of
silently ignoring them. Runnable examples: `examples/soft_counts_lsstypes.py` (this
frontend, isotropic and multipoles), `examples/soft_hist_grad_jax.py` (gradients through
`cucount.jax.soft`), `examples/soft_pad_leakage.py` (unpadded vs padded soft counts
against hard binning at the last edge), `examples/soft_grad_training.py` (why a joint
P(k)+xi loss can leave xi untrained, and how fitting counts fixes it) and
`examples/soft_grad_precision.py` (why a normalized xi gradient can be too small for a
float32 position update).

---

## 📎 References

- [JAX FFI documentation](https://docs.jax.dev/en/latest/ffi.html)
- [shard\_map usage guide](https://docs.jax.dev/en/latest/notebooks/shard_map.html)

---

## 🙏 Acknowledgment

Special thanks to **François Lanusse** for valuable advice on Python-JAX bindings.

---

## ❓ Questions or Feedback

Feel free to open an issue or discussion on the repository if you encounter problems or have suggestions.
