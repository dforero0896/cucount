# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
at [semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- `soft_norm` keyword on the soft pair-count API (`cucount.numpy.soft`,
  `cucount.jax.soft`, `cucount.types.count2_soft`, and the four pybind11/FFI
  entry points). It selects how a pair's kernel weights are normalised on the
  separation axis:
  - `'gaussian'` (new default): discrete-quadrature mass per bin,
    `W_i = A_i exp(-x_i)` with `A_i = delta_i / (sqrt(2 pi) sigma_i)`. One pass
    over the window, no per-pair sum, so neighbouring bins no longer compete for
    the same pair weight and the gradients are better conditioned. The row sum
    is one only to quadrature accuracy (about 0.4% at the default half-bin
    width) and kernel mass outside the counting range is dropped.
  - `'softmax'` (previous behaviour): `W_i = exp(-x_i) / sum_j exp(-x_j)` over
    the truncated window, which is exactly conserved but needs a second pass.
- The mode is reported back in `result['soft']['norm']`.
- `soft_sigma=None` (the default) is scale-free on a logarithmic grid: it means
  half of each bin's own width, so the kernel is `sigma ∝ s`. This matches
  pyfcfc's `soft_sigma_log` mode. An absolute `soft_sigma` is *not* the same
  kernel on a wide log grid and is over-smoothing at small separations.
- The mu axis of the multipoles is always renormalised per pair, whatever
  `soft_norm` selects for the separation axis. `mu ∈ [-1, 1]` is a bounded
  domain, so an unnormalised kernel would lose about half its mass for pairs
  near a pole; padding cannot recover a domain truncation the way it recovers
  the separation range. `M_0 = 1` therefore holds in both modes and the
  monopole equals the isotropic soft pair count.
- Host-side per-bin prefactor table (`SoftBinSpec::quad_norm`), filled once per
  call, instead of recomputing a square root per kernel tap on a non-uniform
  grid.
- Examples: `examples/soft_norm_comparison.py` (kernel algebra and the effect on
  the 2PCF, on an AbacusSummit LRG sub-box or synthetic data),
  `examples/bench_soft_norm.py` (cold and warm forward/adjoint cost of both
  modes, CPU and CUDA), and `examples/_catalogues.py` (the Abacus loader both
  use).

### Changed

- **Behaviour change: the default is now `soft_norm='gaussian'`.** Counts move
  relative to previous releases. The shift is about 0.4% in the bulk from the
  quadrature sum rule, and much larger at the outermost bins, where the kernel
  tail falls outside the counting range: on a 24-bin log grid from 2.5 to 60
  Mpc/h with the default width, the total weight drops 5.9% and the first bin
  15% without padding, and `soft_pad` brings the total back to 0.3%. If you need
  the old numbers, pass `soft_norm='softmax'`.
- The mu axis is renormalised in every mode, so multipoles no longer depend on
  `soft_norm` (previously the only behaviour).

### Fixed

- `count2_soft_vjp` on the CPU rejected `soft_norm`, because the binding's
  argument list and its `py::arg` registrations disagreed.
- The NumPy reference implementation (`soft_weights`) normalised a softmax over
  the whole `(pairs, bins)` array instead of per row, so multi-row calls did not
  sum to one per pair.
- The separation derivative of the plain-Gaussian multipole adjoint had the
  wrong sign, disagreeing with the softmax branch and with finite differences.
- Restored the non-arithmetic `mu` profile kernel on CUDA, which had been
  dropped while reworking the profilers.

### Known issues

- The CUDA adjoint is about 15% slower in `gaussian` mode than in `softmax`
  mode at the narrowest kernel widths tested (sigma = 0.25 bin), although the
  gaussian branch executes strictly fewer instructions. Flat in sigma, so it
  looks like the shared reverse-mode driver rather than the tap loop, but it is
  not explained yet. The forward pass and the CPU adjoint are faster in
  `gaussian` mode at every width tested.
- Figures under `examples/figures/` are generated artefacts and are not tracked;
  several were produced before this change and do not match what the scripts
  print now. See the notes in each example.
