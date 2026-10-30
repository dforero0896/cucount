#ifndef _CUCOUNT_COUNT2_SOFT_
#define _CUCOUNT_COUNT2_SOFT_

// Soft (Gaussian-softmax) pair counts on the separation axis.
//
// This is an extension, not a mode of count2: its own header, its own
// translation unit, its own binding. The hard kernel's inner loop is the
// reference for both correctness and speed, and it must not grow a branch or a
// register for soft binning, so nothing here is included by it.
//
// v1 serves one ``s`` axis -- isotropic clustering. A request for anything else
// (a second axis, multipoles, spin, angular upweights, splits) is refused by the
// binding with a Python exception; these kernels assume the request has already
// been narrowed.
//
// Semantics (the same as pyfcfc's soft_hist.c, which is the oracle):
//
//   c_i = (edges[i] + edges[i+1]) / 2      bin centres
//   x_i = (s - c_i)^2 / (2 sigma_i^2)      Gaussian logits
//   W_i = exp(-x_i) / sum_j exp(-x_j)      softmax over the bins
//   counts[i] += w_pair * W_i              per pair, w_pair = product of weights
//
// The sum runs over the taps within ``cutoff`` of the peak, then everything is
// renormalized, so a pair's weight is exactly conserved and sigma -> 0
// degenerates to hard binning (nearest centre). Pairs outside [smin, smax) are
// dropped whole, exactly as the hard kernel drops them, which is what makes the
// soft and hard totals agree at any sigma.
//
// Two grids, two code paths:
//
//  * uniform edges keep the multiplicative-recurrence kernel: two exp() calls
//    per pair and one multiply per tap;
//  * arbitrary monotone edges use the general kernel: a binary search for the
//    peak, a window expanded until the logit distance passes ln(1/cutoff), and
//    one exp() per tap. Because the general path has no recurrence, the
//    sigma_i^2 and centre arrays are uploaded once per call (a few kB).
//
// sigma follows the width of the bin unless asked otherwise:
//
//   sigma <= 0 (None) + non-uniform edges -> sigma_i = 0.5 * (edges[i+1] - edges[i])
//   sigma <= 0 (None) + uniform edges     -> sigma = 0.5 * ds   (the same thing)
//   sigma > 0                             -> one absolute sigma for every bin
//
// The width-scaled default is what makes a log-spaced grid mean "constant
// relative resolution" instead of "wide bins are under-smoothed and narrow ones
// are over-smoothed", and because sigma_i is then a function of the edges it is
// a constant of the binning: there is no dL/dsigma to compute for it.
//
// Multipoles (a second axis, VAR_POLE) are softened on the *mu* axis with the
// same truncated softmax, evaluated on a uniform grid of nmu centres over
// [-1, 1]. The mu weights are folded into the multipoles before the s scatter,
//
//   M_l = sum_j Wmu_j (2 l + 1) P_l(mu_j),        sum_j Wmu_j = 1
//   counts[s bin i, l] += w_pair * Ws_i * M_l
//
// so a pair costs taps_s * nell histogram updates instead of the
// taps_s * taps_mu it would cost to soften both axes on the histogram grid.
// P_0 = 1, so M_0 = 1 and the monopole is exactly the soft-s count.
//
// mu is *not* folded: it runs over the full [-1, 1] interval that cucount's
// line of sight defines, so odd multipoles are meaningful and no |mu| symmetry is
// assumed (fcfc folds mu; cucount does not, and the soft kernel follows cucount).
// The grid over that interval is uniform by default (``mu_bins``/``mu_sigma``
// below), which keeps the two-exp recurrence, but the caller may hand in its own
// strictly increasing ``mu_edges`` covering (part of) [-1, 1]. Arithmetic edges
// keep the recurrence path; anything else gets a per-cell sigma and one exp per
// tap, the same split the s axis has. A pair whose mu falls outside a partial
// grid is carried by the nearest boundary cell.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "common.h"
#include "args.h"

// The request resolution (SoftBinSpec, make_soft_bin_spec) is shared with the
// CPU backend; it lives in include/soft.h so the two cannot diverge.
#include "soft.h"

// Writes spec.nslots entries: nbins for a plain soft-s request, nbins * nell
// for soft mu binning (s-major: slot = s bin * nell + multipole position).
void count2_soft(FLOAT *counts, const Mesh *list_mesh, const Count2Attrs &attrs,
                 SoftBinSpec &spec, DeviceMemoryBuffer *buffer,
                 cudaStream_t stream);


// Vector-Jacobian product of the forward above: given the cotangent of the
// histogram (dL/dcounts), accumulate dL/dpositions, dL/dindividual weights and
// dL/dsigma. This is what lets the jax layer declare a custom_vjp and get
// differentiable pair counts without differentiating a scatter by hand.
//
// The gradients are *zeroed* by this call and then accumulated with atomics, so
// grad_position1 holds NDIM * mesh[0].size entries (catalog 1), grad_position2
// NDIM * mesh[1].size (catalog 2), grad_weight* one per particle, and
// grad_sigma a single entry. For an autocorrelation pass the same arrays as
// catalog 1 and 2: the ordered pairs the kernel walks then contribute to both
// endpoints, which is exactly dL/dr_i for the symmetric sum.
//
// Why this is a second kernel rather than reverse-mode AD: the forward is a
// scatter into a histogram, so its input cotangent is a gather from the same
// window, and the per-pair reduction collapses to a handful of sums. Written
// out (c_b = centres, x_b = logits, W_b = softmax, cbar/xbar their W-means):
//
//   dL/dw_pair = sum_b g_b W_b                          (a scalar per pair)
//   dL/ds      = (w_pair / sigma^2) sum_b g_b W_b (c_b - cbar)
//   dL/dsigma  = (2 w_pair / sigma) sum_b g_b W_b (x_b - xbar)
//   dL/dr_i    = dL/ds * (r_i - r_j) / s ,  dL/dr_j = -dL/dr_i
//
// With soft mu binning there is one more chain: the multipole profile
// M_l = sum_j Wmu_j (2l+1) P_l(mu_j) depends on the pair's mu,
//
//   dL/dmu = (w_pair / sum_j Wmu_j) sum_l G_l dM_l/dmu,
//   dM_l/dmu = (dN_l - M_l dS) / sum_j Wmu_j,
//   dN_l = sum_j dWmu_j/dmu (2l+1) P_l(mu_j),   dS = sum_j dWmu_j/dmu,
//   dWmu_j/dmu = Wmu_j (mu_j - mubar) / sigma_j^2,
//
// and mu is a function of the pair difference (and, for the midpoint and
// first/end-point lines of sight, of the endpoints themselves), so dL/dmu
// enters dL/dpositions through the same (r_i - r_j) / s geometry plus the
// line-of-sight term. The mu window and grid are constants, so they are not
// differentiated. dL/dsigma_mu is deliberately zero: the mu width is a fixed
// property of the request, like the s width of a per-bin sigma.
//
// which is pyfcfc's soft_hist_vjp. Window membership and the range check are
// piecewise constant in s, so they are treated as constants (the standard
// convention) and the window's own renormalisation is differentiated exactly.
//
// With a per-bin sigma the same reduction holds tap by tap,
//
//   dL/ds = 2 w_pair [ sum_b g_b W_b (c_b - s) / sigma_b^2
//                      + (sum_b g_b W_b) * sum_j W_j (s - c_j) / sigma_j^2 ]
//
// which reduces to the constant-sigma line above when every sigma_b is equal.
// In that case there is no sigma cotangent (sigma_i comes from the edges, so it
// is a constant of the binning) and grad_sigma is left alone.
void count2_soft_vjp(const FLOAT *cotangent, FLOAT *grad_position1,
                     FLOAT *grad_position2, FLOAT *grad_weight1,
                     FLOAT *grad_weight2, FLOAT *grad_sigma,
                     const Mesh *list_mesh, const Count2Attrs &attrs,
                     const SoftBinSpec &spec, DeviceMemoryBuffer *buffer,
                     cudaStream_t stream);

// Narrow-storage boundary support.
//
// The kernel computes in FLOAT (float64) regardless of how the caller stores its
// positions and weights, so the jax FFI entry points can accept float32 arrays
// without giving up the double-precision geometry: separations, logits and the
// per-pair maths stay wide, only the *sample* is narrow. These two helpers copy
// between the caller's float32 buffers and a float64 staging area. Widening a
// float32 sample is exact; narrowing the result rounds once, at the boundary.
void soft_cast_f32_to_f64(FLOAT *dst, const float *src, size_t n,
                          cudaStream_t stream);
void soft_cast_f64_to_f32(float *dst, const FLOAT *src, size_t n,
                          cudaStream_t stream);


#endif  // _CUCOUNT_COUNT2_SOFT_
