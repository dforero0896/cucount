// Soft pair counts on the CPU: forward and adjoint.
//
// A port of cuda/src/count2_soft.cu, kept close to it for the same reason
// generic.cpp is kept close to cuda/src/count2.cu: the two backends must bucket
// the same pair into the same taps with the same weights, and reading them side
// by side is how that stays true. The request itself is not resolved here at
// all -- SoftBinSpec and make_soft_bin_spec come from the shared
// include/soft.h, so a CPU call and a CUDA call cannot disagree about the
// window, the sigma or the mu quadrature.
//
// What does not survive the port, and why the CPU version looks simpler:
//
//  * no shared-memory tile and no atomics. Every thread scatters into a private
//    histogram and the rows are reduced at the end (as generic.cpp does), so
//    the accumulator is plain double: `soft_accum='float32'`, i.e. the narrower
//    per-block tile, is a GPU occupancy decision with nothing to buy here.
//  * no `path`/`copies` bookkeeping, no register-tuned window, no f32 staging:
//    those are all answers to GPU memory-hierarchy costs.
//  * no slot->input map for the *forward*: the CPU mesh keeps the same
//    cell-ordered slots, but the adjoint needs the inverse (ScalarMesh::origin,
//    built on request) because gradients are per input particle.

#include "count2_soft.h"
// The scalar mesh, candidate walk and per-pair geometry, shared with the hard
// count2 and the triplet counts.
#include "mesh.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cucount {
namespace cpu {
namespace {

using pairmath::add3;
using pairmath::compute_pair_mu;
using pairmath::dot3;
using pairmath::pair_bitwise_weight;


// ---------------------------------------------------------------------------
// s axis, forward
// ---------------------------------------------------------------------------

// `stride` is the histogram stride between neighbouring s bins: 1 for a plain
// soft-s request, nell when the multipoles are folded into `hist` (see
// soft_add_pole, which calls this once per multipole on hist + ill).
inline void soft_add_uniform(double* hist, const SoftBinSpec& sp, double s, double w,
                             const int stride) {
    // Same rule as the hard kernel's range check: a pair outside the binning is
    // dropped whole, not clipped into the first/last bin. NaN falls out here.
    if (!(s >= sp.lo) || !(s < sp.hi)) return;

    int m = static_cast<int>(std::floor((s - sp.c0) * sp.inv_ds + 0.5));

    if (m < 0)
        m = 0;
    else if (m >= static_cast<int>(sp.nbins))
        m = static_cast<int>(sp.nbins) - 1;

    const double u = s - (sp.c0 + m * sp.ds);

    // log of w_{m+1}/w_m and of w_{m-1}/w_m. Only one exponential is needed:
    // rdn = iq / rup, since exp(-md) = 1 / exp(md). The fallback only triggers
    // for sigma far below the bin width, where exp(md) underflows.
    const double md = (2 * u - sp.ds) * sp.ds * sp.inv2sig2;
    const double rup = std::exp(md);
    const double rdn = (rup > 0) ? sp.iq / rup : std::exp(-md - sp.gg);
    const int kmax = sp.kmax;
    const int nbins = static_cast<int>(sp.nbins);

    // Fast path: the whole window is interior, so no tap can leave [0, nbins)
    // and every tap inside kmax is above the cutoff by construction.
    if (m >= kmax && m + kmax < nbins) {
        double sum = 1;
        double wu = 1, ru = rup, wd = 1, rd = rdn;

        for (int t = 1; t <= kmax; ++t) {
            wu *= ru;
            ru *= sp.iq;
            wd *= rd;
            rd *= sp.iq;
            sum += wu + wd;
        }

        const double scale = w / sum;
        hist[m * stride] += scale;

        wu = 1; ru = rup;
        wd = 1; rd = rdn;

        for (int t = 1; t <= kmax; ++t) {
            wu *= ru;
            ru *= sp.iq;
            wd *= rd;
            rd *= sp.iq;
            hist[(m + t) * stride] += wu * scale;
            hist[(m - t) * stride] += wd * scale;
        }
        return;
    }

    double sum = 1;
    double wu = 1, ru = rup;
    double wd = 1, rd = rdn;
    bool oku = true, okd = true;

    for (int t = 1; t <= kmax; ++t) {
        wu *= ru;
        ru *= sp.iq;
        wd *= rd;
        rd *= sp.iq;

        oku = (m + t < nbins) && (wu > sp.cutoff);
        okd = (m - t >= 0) && (wd > sp.cutoff);

        if (oku) sum += wu;
        if (okd) sum += wd;
        if (!oku && !okd) break;
    }

    const double scale = w / sum;
    hist[m * stride] += scale;

    wu = 1; ru = rup;
    wd = 1; rd = rdn;

    for (int t = 1; t <= kmax; ++t) {
        wu *= ru;
        ru *= sp.iq;
        wd *= rd;
        rd *= sp.iq;

        oku = (m + t < nbins) && (wu > sp.cutoff);
        okd = (m - t >= 0) && (wd > sp.cutoff);

        if (oku) hist[(m + t) * stride] += wu * scale;
        if (okd) hist[(m - t) * stride] += wd * scale;
        if (!oku && !okd) break;
    }
}


// General (non-uniform) grid: arbitrary monotone edges have no arithmetic
// progression, so neither the nearest-centre formula nor the tap recurrence
// survives. The truncation rule does: a tap is kept while its logit stays within
// ln(1/cutoff) of the peak's. The window is therefore found by expanding
// outwards from the peak, and each tap pays one exp per pass (this is what
// soft_taps='fast' exists for).
struct SoftWindow {
    int m;       // peak bin
    int up;      // taps above m
    int dn;      // taps below m
    double lmin;  // logit at the peak
};

inline SoftWindow soft_general_window(const SoftBinSpec& sp, double s) {
    const int nbins = static_cast<int>(sp.nbins);
    const double* cen = sp.centres;
    const double* i2s2 = sp.inv2s2;
    const double* edg = sp.edges;

    // Bin containing s: the last edge at or below it. Searching the centres
    // instead would answer "previous bin" for the lower half of every bin, which
    // a wide softmax hides but a narrow one does not.
    int lo = 0, hi = nbins - 1;

    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;

        if (edg[mid] <= s)
            lo = mid;
        else
            hi = mid - 1;
    }

    int m = lo;
    double d = s - cen[m];

    SoftWindow win;
    win.lmin = d * d * i2s2[m];

    // With per-bin sigma a *wide* neighbouring bin can have a lower logit than
    // the bin that actually contains s, so "the bin containing s" is not always
    // the peak. Two steps of descent suffice for sigma_i proportional to the
    // local width.
    for (int it = 0; it < 2; ++it) {
        if (m > 0) {
            const double dd = s - cen[m - 1];
            const double l = dd * dd * i2s2[m - 1];

            if (l < win.lmin) {
                win.lmin = l;
                m -= 1;
                continue;
            }
        }

        if (m + 1 < nbins) {
            const double dd = s - cen[m + 1];
            const double l = dd * dd * i2s2[m + 1];

            if (l < win.lmin) {
                win.lmin = l;
                m += 1;
            }
        }
    }

    win.m = m;
    win.up = 0;
    win.dn = 0;

    const double thresh = sp.thresh;
    const int kmax = sp.kmax;

    while (win.up < kmax && m + win.up + 1 < nbins) {
        const double dd = s - cen[m + win.up + 1];

        if (dd * dd * i2s2[m + win.up + 1] - win.lmin > thresh) break;

        win.up += 1;
    }

    while (win.dn < kmax && m - win.dn - 1 >= 0) {
        const double dd = s - cen[m - win.dn - 1];

        if (dd * dd * i2s2[m - win.dn - 1] - win.lmin > thresh) break;

        win.dn += 1;
    }

    return win;
}


inline void soft_add_general(double* hist, const SoftBinSpec& sp, double s, double w,
                             const int stride) {
    if (!(s >= sp.lo) || !(s < sp.hi)) return;

    const double* cen = sp.centres;
    const double* i2s2 = sp.inv2s2;
    const bool fast = sp.fast_taps != 0;
    const SoftWindow win = soft_general_window(sp, s);
    const int m = win.m;

    // Pass 1: the normaliser. The peak's weight is exp(0) = 1.
    double sum = 1;

    for (int k = 1; k <= win.up; ++k) {
        const double dd = s - cen[m + k];
        sum += soft_tap_weight(dd * dd * i2s2[m + k] - win.lmin, fast);
    }

    for (int k = 1; k <= win.dn; ++k) {
        const double dd = s - cen[m - k];
        sum += soft_tap_weight(dd * dd * i2s2[m - k] - win.lmin, fast);
    }

    // Pass 2: scatter. A second exp per tap, because the general path has no
    // recurrence to replay.
    const double scale = w / sum;
    hist[m * stride] += scale;

    for (int k = 1; k <= win.up; ++k) {
        const double dd = s - cen[m + k];
        hist[(m + k) * stride] +=
            scale * soft_tap_weight(dd * dd * i2s2[m + k] - win.lmin, fast);
    }

    for (int k = 1; k <= win.dn; ++k) {
        const double dd = s - cen[m - k];
        hist[(m - k) * stride] +=
            scale * soft_tap_weight(dd * dd * i2s2[m - k] - win.lmin, fast);
    }
}


// The one entry the pair loop calls: one predictable branch per pair picks the
// path.
inline void soft_add(double* hist, const SoftBinSpec& sp, double s, double w,
                     const int stride) {
    if (sp.uniform)
        soft_add_uniform(hist, sp, s, w, stride);
    else
        soft_add_general(hist, sp, s, w, stride);
}


// ---------------------------------------------------------------------------
// Multipole axis: soft mu binning
// ---------------------------------------------------------------------------
//
// mu is softened with the same truncated Gaussian softmax, on a grid over
// [-1, 1]. What the pair accumulates is not a histogram in mu but the
// multipoles,
//
//   M_l = sum_j Wmu_j (2l + 1) P_l(mu_j),       sum_j Wmu_j = 1
//
// and the s scatter multiplies by M_l instead of by 1. P_0 = 1 makes the
// monopole bit-for-bit the soft-s count, and sigma_mu -> 0 reproduces the hard
// multipoles up to the quadrature -- the same limit as pyfcfc.

// P_l(centre) for every l <= ellmax, by Bonnet's recursion. The hard kernel's
// set_legendre() fills only the requested orders on its even-ell fast path,
// which is fine there (its layout is contiguous from 0) but not here: a soft
// request may list any multipoles in any order.
inline void soft_legendre_all(double* leg, const int ellmax, const double mu) {
    leg[0] = 1;

    if (ellmax >= 1) leg[1] = mu;

    for (int ell = 2; ell <= ellmax; ++ell) {
        const double inv = 1 / static_cast<double>(ell);
        leg[ell] = ((2 * ell - 1) * mu * leg[ell - 1] - (ell - 1) * leg[ell - 2]) * inv;
    }
}


// The truncated softmax over the mu grid folded onto the multipoles. DERIV also
// returns the derivative of the profile with respect to mu, which is what the
// adjoint needs:
//
//   M_l(mu) = sum_j Wmu_j(mu) (2l+1) P_l(c_j) / sum_j Wmu_j(mu)
//   dM_l/dmu = (dN_l - M_l dS) / S,   dWmu_j/dmu = Wmu_j (c_j - mu)/sigma^2
//
// There is no dP_l/dmu term: the taps sit on the *fixed* mu grid, so the only
// thing that moves with mu is the Gaussian weight.
template <bool DERIV>
inline void soft_mu_profile_uniform(const SoftBinSpec& sp, const double mu, double* M,
                                    double* dM) {
    int m = static_cast<int>(std::floor((mu - sp.mu_c0) * sp.mu_inv_dmu + 0.5));

    if (m < 0)
        m = 0;
    else if (m >= sp.nmu)
        m = sp.nmu - 1;

    const double u = mu - (sp.mu_c0 + m * sp.mu_dmu);

    // Same recurrence as a uniform s window: one exp for the upper ratio.
    const double md = (2 * u - sp.mu_dmu) * sp.mu_dmu * sp.mu_inv2sig2;
    const double rup = std::exp(md);
    const double rdn = (rup > 0) ? sp.mu_iq / rup : std::exp(-md - sp.mu_gg);
    const int kmax = sp.mu_kmax;
    const int nmu = sp.nmu;
    const int nell = sp.nell;

    double num[CUCOUNT_SOFT_MAXELL], dnum[CUCOUNT_SOFT_MAXELL], leg[CUCOUNT_SOFT_MAXELL];

    for (int ill = 0; ill < nell; ++ill) {
        num[ill] = 0;
        dnum[ill] = 0;
    }

    double sum = 1, dsum = 0;

    // `W` is the (unnormalised) weight of the tap, `B` its mu cell.
    const auto tap = [&](int B, double W) {
        const double m_ = sp.mu_c0 + B * sp.mu_dmu;
        soft_legendre_all(leg, sp.ell_max, m_);
        const double dw_ = DERIV ? W * (m_ - mu) * 2 * sp.mu_inv2sig2 : 0.;

        for (int ill = 0; ill < nell; ++ill) {
            const int ell = sp.ells[ill];
            const double pf = static_cast<double>(2 * ell + 1);
            num[ill] += W * pf * leg[ell];
            if (DERIV) dnum[ill] += dw_ * pf * leg[ell];
        }

        if (DERIV) dsum += dw_;
    };

    tap(m, 1);

    double wu = 1, ru = rup;
    double wd = 1, rd = rdn;

    for (int t = 1; t <= kmax; ++t) {
        wu *= ru;
        ru *= sp.mu_iq;
        wd *= rd;
        rd *= sp.mu_iq;

        const bool oku = (m + t < nmu) && (wu > sp.cutoff);
        const bool okd = (m - t >= 0) && (wd > sp.cutoff);

        if (oku) {
            sum += wu;
            tap(m + t, wu);
        }

        if (okd) {
            sum += wd;
            tap(m - t, wd);
        }

        if (!oku && !okd) break;
    }

    const double inv = 1 / sum;

    for (int ill = 0; ill < nell; ++ill) {
        M[ill] = num[ill] * inv;
        if (DERIV) dM[ill] = (dnum[ill] - M[ill] * dsum) * inv;
    }
}


// The same profile on a caller-supplied (non-arithmetic) mu grid: no recurrence
// across unequal cells, so every tap pays one exp. Arbitrary edges can make
// neighbouring cells very different widths, so the logit is not monotone in the
// cell index and a local peak search would stop on the wrong side of a wide
// cell; the whole (capped-at-1024) grid is scanned instead.
template <bool DERIV>
inline void soft_mu_profile_general(const SoftBinSpec& sp, const double mu, double* M,
                                    double* dM) {
    const int nmu = sp.nmu;
    const double* cen = sp.mu_centres;
    const double* i2s2 = sp.mu_inv2s2;
    const int nell = sp.nell;
    const bool fast = sp.fast_taps != 0;

    double lmin = (mu - cen[0]) * (mu - cen[0]) * i2s2[0];

    for (int j = 1; j < nmu; ++j) {
        const double d = mu - cen[j];
        const double l = d * d * i2s2[j];

        if (l < lmin) lmin = l;
    }

    double num[CUCOUNT_SOFT_MAXELL], dnum[CUCOUNT_SOFT_MAXELL], leg[CUCOUNT_SOFT_MAXELL];

    for (int ill = 0; ill < nell; ++ill) {
        num[ill] = 0;
        dnum[ill] = 0;
    }

    double sum = 0, dsum = 0;

    for (int j = 0; j < nmu; ++j) {
        const double d = mu - cen[j];
        const double l = d * d * i2s2[j] - lmin;

        if (!(l <= sp.thresh)) continue;

        const double w = soft_tap_weight(l, fast);
        sum += w;

        soft_legendre_all(leg, sp.ell_max, cen[j]);
        const double dw_ = DERIV ? w * (cen[j] - mu) * 2 * i2s2[j] : 0.;

        for (int ill = 0; ill < nell; ++ill) {
            const int ell = sp.ells[ill];
            const double pf = static_cast<double>(2 * ell + 1);
            num[ill] += w * pf * leg[ell];
            if (DERIV) dnum[ill] += dw_ * pf * leg[ell];
        }

        if (DERIV) dsum += dw_;
    }

    const double inv = 1 / sum;

    for (int ill = 0; ill < nell; ++ill) {
        M[ill] = num[ill] * inv;
        if (DERIV) dM[ill] = (dnum[ill] - M[ill] * dsum) * inv;
    }
}


// Dispatch: the arithmetic grid keeps the two-exp recurrence, anything else the
// per-tap window. Both fill the same M/dM contract.
template <bool DERIV>
inline void soft_mu_profile(const SoftBinSpec& sp, const double mu, double* M, double* dM) {
    if (sp.mu_uniform)
        soft_mu_profile_uniform<DERIV>(sp, mu, M, dM);
    else
        soft_mu_profile_general<DERIV>(sp, mu, M, dM);
}


// Soft-s scatter of a pair whose multipole profile is already known: the s
// window is walked once per multipole (the sum that normalises it does not
// depend on l).
inline void soft_add_pole(double* hist, const SoftBinSpec& sp, double s, double mu,
                          double w) {
    if (!(s >= sp.lo) || !(s < sp.hi)) return;

    double M[CUCOUNT_SOFT_MAXELL];

    soft_mu_profile<false>(sp, mu, M, M);

    for (int ill = 0; ill < sp.nell; ++ill)
        soft_add(hist + ill, sp, s, w * M[ill], sp.nell);
}


// ---------------------------------------------------------------------------
// Adjoint (vector-Jacobian product)
// ---------------------------------------------------------------------------
//
// The forward is a scatter of one pair weight into a Gaussian window, so the
// adjoint is a gather of the histogram cotangent over the same window. With the
// softmax written out (W_b = e_b / sum, cbar/xbar the W-means of the centres and
// logits):
//
//   dL/dw_pair = sum_b g_b W_b
//   dL/ds      = (w_pair / sigma^2) sum_b g_b W_b (c_b - cbar)
//   dL/dsigma  = (2 w_pair / sigma) sum_b g_b W_b (x_b - xbar)
//
// which is pyfcfc's soft_hist_vjp. Membership of the window and the
// [smin, smax) range check are piecewise constant in s, so they are held fixed,
// while the window's own renormalisation is differentiated exactly.
struct SoftVjpPair {
    double dw;
    double ds;
    double dsigma;
    double dmu;  // dL/dmu: only nonzero with soft mu binning
};


inline void soft_vjp_pair_uniform(const double* g, const SoftBinSpec& sp, double s,
                                  double w, SoftVjpPair* out) {
    out->dw = 0;
    out->ds = 0;
    out->dsigma = 0;
    out->dmu = 0;

    if (!(s >= sp.lo) || !(s < sp.hi)) return;

    int m = static_cast<int>(std::floor((s - sp.c0) * sp.inv_ds + 0.5));

    if (m < 0)
        m = 0;
    else if (m >= static_cast<int>(sp.nbins))
        m = static_cast<int>(sp.nbins) - 1;

    const double u = s - (sp.c0 + m * sp.ds);
    const double md = (2 * u - sp.ds) * sp.ds * sp.inv2sig2;
    const double rup = std::exp(md);
    const double rdn = (rup > 0) ? sp.iq / rup : std::exp(-md - sp.gg);
    const int kmax = sp.kmax;
    const int nbins = static_cast<int>(sp.nbins);

    // Pass 1: the normaliser, exactly the sum the forward divides by.
    double sum = 1;
    double wu = 1, ru = rup, wd = 1, rd = rdn;

    for (int t = 1; t <= kmax; ++t) {
        wu *= ru;
        ru *= sp.iq;
        wd *= rd;
        rd *= sp.iq;

        const bool oku = (m + t < nbins) && (wu > sp.cutoff);
        const bool okd = (m - t >= 0) && (wd > sp.cutoff);

        if (oku) sum += wu;
        if (okd) sum += wd;
        if (!oku && !okd) break;
    }

    const double scale = 1 / sum;

    // Pass 2: the cotangent sums. x_b = (s - c_b)^2 / (2 sigma^2).
    double A = 0, cbar = 0, xbar = 0, tcg = 0, txg = 0;

    const auto tap = [&](int B, double W) {
        const double c_ = sp.c0 + B * sp.ds;
        const double d_ = s - c_;
        const double x_ = d_ * d_ * sp.inv2sig2;
        const double w_ = W * scale;
        const double g_ = g[B];
        A += w_ * g_;
        cbar += w_ * c_;
        xbar += w_ * x_;
        tcg += w_ * g_ * c_;
        txg += w_ * g_ * x_;
    };

    tap(m, 1);

    wu = 1; ru = rup;
    wd = 1; rd = rdn;

    for (int t = 1; t <= kmax; ++t) {
        wu *= ru;
        ru *= sp.iq;
        wd *= rd;
        rd *= sp.iq;

        const bool oku = (m + t < nbins) && (wu > sp.cutoff);
        const bool okd = (m - t >= 0) && (wd > sp.cutoff);

        if (oku) tap(m + t, wu);
        if (okd) tap(m - t, wd);
        if (!oku && !okd) break;
    }

    // 1 / sigma^2 = 2 * inv2sig2, so no second division is needed.
    out->dw = A;
    out->ds = w * (tcg - cbar * A) * 2 * sp.inv2sig2;
    out->dsigma = w * (txg - xbar * A) * 2 * sp.inv_sigma;
    out->dmu = 0;
}


// General (non-uniform) adjoint: the same window as the forward -- it comes from
// the same helper, which is what keeps finite differences honest -- and the
// per-bin-sigma reduction derived in the header.
inline void soft_vjp_pair_general(const double* g, const SoftBinSpec& sp, double s,
                                  double w, SoftVjpPair* out) {
    out->dw = 0;
    out->ds = 0;
    out->dsigma = 0;
    out->dmu = 0;

    if (!(s >= sp.lo) || !(s < sp.hi)) return;

    const double* cen = sp.centres;
    const double* i2s2 = sp.inv2s2;
    const bool fast = sp.fast_taps != 0;
    const SoftWindow win = soft_general_window(sp, s);
    const int m = win.m;

    // Pass 1: the normaliser, exactly the sum the forward divides by.
    double sum = 1;

    for (int k = 1; k <= win.up; ++k) {
        const double dd = s - cen[m + k];
        sum += soft_tap_weight(dd * dd * i2s2[m + k] - win.lmin, fast);
    }

    for (int k = 1; k <= win.dn; ++k) {
        const double dd = s - cen[m - k];
        sum += soft_tap_weight(dd * dd * i2s2[m - k] - win.lmin, fast);
    }

    const double scale = 1 / sum;

    double A = 0;     // sum_b g_b W_b
    double cbar = 0;  // sum_b W_b c_b
    double xbar = 0;  // sum_b W_b x_b
    double tcg = 0;   // sum_b g_b W_b c_b
    double txg = 0;   // sum_b g_b W_b x_b
    double T = 0;     // sum_b g_b W_b (c_b - s) / sigma_b^2
    double U = 0;     // sum_b W_b (s - c_b) / sigma_b^2

    const auto tap = [&](int B, double W) {
        const double c_ = cen[B];
        const double d_ = s - c_;
        const double i2_ = i2s2[B];
        const double x_ = d_ * d_ * i2_;
        const double w_ = W * scale;
        const double g_ = g[B];
        A += w_ * g_;
        cbar += w_ * c_;
        xbar += w_ * x_;
        tcg += w_ * g_ * c_;
        txg += w_ * g_ * x_;
        T += w_ * g_ * (c_ - s) * i2_;
        U += w_ * (s - c_) * i2_;
    };

    tap(m, 1);

    for (int k = 1; k <= win.up; ++k) {
        const double dd = s - cen[m + k];
        tap(m + k, soft_tap_weight(dd * dd * i2s2[m + k] - win.lmin, fast));
    }

    for (int k = 1; k <= win.dn; ++k) {
        const double dd = s - cen[m - k];
        tap(m - k, soft_tap_weight(dd * dd * i2s2[m - k] - win.lmin, fast));
    }

    out->dw = A;
    out->ds = 2 * w * (T + A * U);
    out->dmu = 0;

    // With sigma_i taken from the bin widths there is no sigma to differentiate.
    if (!sp.per_bin_sigma) out->dsigma = w * (txg - xbar * A) * 2 * sp.inv_sigma;
}


// The line of sight as a unit vector plus the two coefficients that carry its
// own dependence on the endpoints. mu = (diff . lhat) / s, so
//
//   d mu / d diff_b   = (lhat_b - mu rhat_b) / s,     rhat = diff / s
//   d mu / d lhat_b   = rhat_b
//
// and lhat depends on a position only for the pair-dependent lines of sight
// (first point, end point, midpoint), giving the `k1`/`k2` coefficient below
// (zero for the fixed x/y/z axes).
inline void soft_mu_line_of_sight(const LOS_TYPE los, const double* sposition1,
                                  const double* sposition2, const double* position1,
                                  const double* position2, double* lhat, double* k1,
                                  double* k2) {
    *k1 = 0;
    *k2 = 0;

    for (int axis = 0; axis < NDIM; ++axis) lhat[axis] = 0;

    if (los == LOS_FIRSTPOINT) {
        // The mesh already stores the unit position of each particle.
        for (int axis = 0; axis < NDIM; ++axis) lhat[axis] = sposition1[axis];

        const double r = std::sqrt(dot3(position1, position1));

        if (r > 0) *k1 = 1 / r;
    }
    else if (los == LOS_ENDPOINT) {
        for (int axis = 0; axis < NDIM; ++axis) lhat[axis] = sposition2[axis];

        const double r = std::sqrt(dot3(position2, position2));

        if (r > 0) *k2 = 1 / r;
    }
    else if (los == LOS_MIDPOINT) {
        double v[NDIM];

        add3(v, position1, position2);

        const double r = std::sqrt(dot3(v, v));

        if (r > 0) {
            for (int axis = 0; axis < NDIM; ++axis) lhat[axis] = v[axis] / r;

            *k1 = 1 / r;
            *k2 = 1 / r;
        }
    }
    else if (los == LOS_X) {
        lhat[0] = 1;
    }
    else if (los == LOS_Y) {
        lhat[1] = 1;
    }
    else {
        // LOS_Z, and the default for a request that reached here without one.
        lhat[NDIM - 1] = 1;
    }
}


// Adjoint of a pair counted with soft mu binning. The multipole profile is
// gathered through the same truncated mu window the forward used, and the
// cotangent is contracted with it in two directions at once:
//
//   * against M_ill, which is what the s adjoint sees (`g` becomes
//     `geff_b = sum_ill g[b,ill] M_ill`), so the s formulas are unchanged;
//   * against dM_ill/dmu, which is the mu cotangent the position update needs,
//     dL/dmu = w sum_ill (sum_b g[b,ill] Ws_b) dM_ill/dmu.
inline void soft_vjp_pair_pole(const double* g, const SoftBinSpec& sp, double s,
                               double mu, double w, SoftVjpPair* out) {
    out->dw = 0;
    out->ds = 0;
    out->dsigma = 0;
    out->dmu = 0;

    if (!(s >= sp.lo) || !(s < sp.hi)) return;

    double M[CUCOUNT_SOFT_MAXELL], dM[CUCOUNT_SOFT_MAXELL];

    soft_mu_profile<true>(sp, mu, M, dM);

    const int nell = sp.nell;

    // Per multipole: sum_b W_b g[b,ill], and the sums the s formula needs.
    double Bl[CUCOUNT_SOFT_MAXELL];
    double tcg_l[CUCOUNT_SOFT_MAXELL];
    double txg_l[CUCOUNT_SOFT_MAXELL];
    double T_l[CUCOUNT_SOFT_MAXELL];

    for (int ill = 0; ill < nell; ++ill) {
        Bl[ill] = 0;
        tcg_l[ill] = 0;
        txg_l[ill] = 0;
        T_l[ill] = 0;
    }

    double cbar = 0;  // sum_b W_b c_b
    double xbar = 0;  // sum_b W_b x_b
    double U = 0;     // general path only: sum_b W_b (s - c_b) / sigma_b^2

    if (sp.uniform) {
        int m = static_cast<int>(std::floor((s - sp.c0) * sp.inv_ds + 0.5));

        if (m < 0)
            m = 0;
        else if (m >= static_cast<int>(sp.nbins))
            m = static_cast<int>(sp.nbins) - 1;

        const double u = s - (sp.c0 + m * sp.ds);
        const double md = (2 * u - sp.ds) * sp.ds * sp.inv2sig2;
        const double rup = std::exp(md);
        const double rdn = (rup > 0) ? sp.iq / rup : std::exp(-md - sp.gg);
        const int kmax = sp.kmax;
        const int nbins = static_cast<int>(sp.nbins);

        double sum = 1;
        double wu = 1, ru = rup;
        double wd = 1, rd = rdn;

        for (int t = 1; t <= kmax; ++t) {
            wu *= ru;
            ru *= sp.iq;
            wd *= rd;
            rd *= sp.iq;

            const bool oku = (m + t < nbins) && (wu > sp.cutoff);
            const bool okd = (m - t >= 0) && (wd > sp.cutoff);

            if (oku) sum += wu;
            if (okd) sum += wd;
            if (!oku && !okd) break;
        }

        const double scale = 1 / sum;

        const auto tap = [&](int B, double W) {
            const double c_ = sp.c0 + B * sp.ds;
            const double d_ = s - c_;
            const double x_ = d_ * d_ * sp.inv2sig2;
            const double w_ = W * scale;
            const double* g_ = &g[static_cast<size_t>(B) * nell];
            cbar += w_ * c_;
            xbar += w_ * x_;

            for (int ill = 0; ill < nell; ++ill) {
                Bl[ill] += w_ * g_[ill];
                tcg_l[ill] += w_ * g_[ill] * c_;
                txg_l[ill] += w_ * g_[ill] * x_;
            }
        };

        tap(m, 1);

        wu = 1; ru = rup;
        wd = 1; rd = rdn;

        for (int t = 1; t <= kmax; ++t) {
            wu *= ru;
            ru *= sp.iq;
            wd *= rd;
            rd *= sp.iq;

            const bool oku = (m + t < nbins) && (wu > sp.cutoff);
            const bool okd = (m - t >= 0) && (wd > sp.cutoff);

            if (oku) tap(m + t, wu);
            if (okd) tap(m - t, wd);
            if (!oku && !okd) break;
        }
    }
    else {
        // Same window as the forward's soft_add_general, from the same helper.
        const double* cen = sp.centres;
        const double* i2s2 = sp.inv2s2;
        const bool fast = sp.fast_taps != 0;
        const SoftWindow win = soft_general_window(sp, s);
        const int m = win.m;

        double sum = 1;

        for (int k = 1; k <= win.up; ++k) {
            const double dd = s - cen[m + k];
            sum += soft_tap_weight(dd * dd * i2s2[m + k] - win.lmin, fast);
        }

        for (int k = 1; k <= win.dn; ++k) {
            const double dd = s - cen[m - k];
            sum += soft_tap_weight(dd * dd * i2s2[m - k] - win.lmin, fast);
        }

        const double scale = 1 / sum;

        const auto tap = [&](int B, double W) {
            const double c_ = cen[B];
            const double d_ = s - c_;
            const double i2_ = i2s2[B];
            const double x_ = d_ * d_ * i2_;
            const double w_ = W * scale;
            const double* g_ = &g[static_cast<size_t>(B) * nell];
            cbar += w_ * c_;
            xbar += w_ * x_;
            U += w_ * (s - c_) * i2_;

            for (int ill = 0; ill < nell; ++ill) {
                Bl[ill] += w_ * g_[ill];
                tcg_l[ill] += w_ * g_[ill] * c_;
                txg_l[ill] += w_ * g_[ill] * x_;
                T_l[ill] += w_ * g_[ill] * (c_ - s) * i2_;
            }
        };

        tap(m, 1);

        for (int k = 1; k <= win.up; ++k) {
            const double dd = s - cen[m + k];
            tap(m + k, soft_tap_weight(dd * dd * i2s2[m + k] - win.lmin, fast));
        }

        for (int k = 1; k <= win.dn; ++k) {
            const double dd = s - cen[m - k];
            tap(m - k, soft_tap_weight(dd * dd * i2s2[m - k] - win.lmin, fast));
        }
    }

    // Contract the per-multipole sums with the profile: this is where the mu
    // weights disappear from the s adjoint and reappear as dL/dmu.
    double A = 0;
    double tcg = 0;
    double txg = 0;
    double T = 0;

    for (int ill = 0; ill < nell; ++ill) {
        A += M[ill] * Bl[ill];
        tcg += M[ill] * tcg_l[ill];
        txg += M[ill] * txg_l[ill];
        T += M[ill] * T_l[ill];
        out->dmu += w * Bl[ill] * dM[ill];
    }

    out->dw = A;

    if (sp.uniform)
        out->ds = w * (tcg - cbar * A) * 2 * sp.inv2sig2;
    else
        out->ds = 2 * w * (T + A * U);
}


inline void soft_vjp_pair(const double* g, const SoftBinSpec& sp, double s, double mu,
                          double w, SoftVjpPair* out) {
    if (sp.nell > 0)
        soft_vjp_pair_pole(g, sp, s, mu, w, out);
    else if (sp.uniform)
        soft_vjp_pair_uniform(g, sp, s, w, out);
    else
        soft_vjp_pair_general(g, sp, s, w, out);
}


// ---------------------------------------------------------------------------
// Per-pair weight, exactly as the hard scalar path forms it
// ---------------------------------------------------------------------------

// The soft kernel serves individual weights, bitwise weights and the negative
// weight correction -- the same subset the CUDA soft kernel accepts (no spin, no
// angular upweight, no splits: make_soft_bin_spec refuses them by name).
struct PairWeight {
    double total;
    double w1;  // individual weight of side 1 (1 without one)
    double w2;  // individual weight of side 2 (1 without one)
    double bw;  // bitwise factor (1 without one)
};


inline PairWeight soft_pair_weight(const double* value1, const double* value2,
                                   const IndexValue& index_value1,
                                   const IndexValue& index_value2,
                                   const WeightAttrs& wattrs) {
    PairWeight pw{1., 1., 1., 1.};

    if (index_value1.size_individual_weight)
        pw.w1 = value1[index_value1.start_individual_weight];

    if (index_value2.size_individual_weight)
        pw.w2 = value2[index_value2.start_individual_weight];

    if (index_value1.size_bitwise_weight && index_value2.size_bitwise_weight)
        pw.bw = pair_bitwise_weight(value1, value2, index_value1, index_value2,
                                   wattrs.bitwise);

    pw.total = pw.w1 * pw.w2 * pw.bw;

    // Linear in the pair weight, so the negative-weight correction is one more
    // subtraction before the scatter, not a second pass: the soft weights sum to
    // one, so w * 1 is conserved either way.
    if (index_value1.size_negative_weight && index_value2.size_negative_weight)
        pw.total -= value1[index_value1.start_negative_weight] *
                    value2[index_value2.start_negative_weight];

    return pw;
}

}  // namespace


// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

void Count2Soft(const Count2SoftArgs& args) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    const MeshAttrs& mattrs = args.attrs.mattrs;
    SoftBinSpec& spec = *args.spec;
    // The caller's grid, when it is not arithmetic, lives in these host tables for
    // the duration of the count; the CUDA backend copies the very same tables to
    // the device instead (both derive them in soft_build_host_tables()).
    // (FLOAT is a common.h macro that cmath.h undefines in the CPU translation
    // units, and on the host it is double.)
    std::vector<double> s_centres, s_inv2s2, s_edges;
    std::vector<double> mu_centres, mu_inv2s2, mu_edges;
    soft_build_host_tables(spec, s_centres, s_inv2s2, s_edges,
                           mu_centres, mu_inv2s2, mu_edges);

    const ScalarMesh m1 = build_mesh(args.p1, mattrs);
    const ScalarMesh m2 = build_mesh(args.p2, mattrs);

    const auto t1 = Clock::now();

    const size_t csize = spec.nslots;
    const long total1 = static_cast<long>(m1.total);

#ifdef _OPENMP
#pragma omp parallel num_threads(args.nthreads)
#endif
    {
        // A private histogram per thread, reduced at the end: the CPU has no
        // shared-memory atomics to make a tile worth it (see the file comment).
        std::vector<double> local(csize, 0.);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
        for (long ii = 0; ii < total1; ii++) {
            const double* position1 = m1.position(ii);
            const double* sposition1 = m1.sposition(ii);
            const double* value1 = m1.value(ii);

            for_each_candidate(mattrs, m2, position1, sposition1, [&](size_t jj) {
                const double* position2 = m2.position(jj);
                const double* sposition2 = m2.sposition(jj);
                const double* value2 = m2.value(jj);

                if (!is_selected_pair(sposition1, sposition2, position1, position2,
                                      args.attrs.sattrs, mattrs))
                    return;

                double diff[NDIM];
                // diff = position2 - position1, the CUDA kernel's convention.
                difference(diff, position2, position1, mattrs);

                const double s = std::sqrt(dot3(diff, diff));
                const PairWeight pw =
                    soft_pair_weight(value1, value2, m1.iv, m2.iv, args.attrs.wattrs);

                if (spec.nell > 0) {
                    // The pair's mu, measured exactly as the hard kernel measures
                    // it, so the soft multipoles and the hard ones see the same
                    // line of sight.
                    double mu = 0., mu2 = 0.;

                    compute_pair_mu(diff, sposition1, sposition2, position1, position2,
                                    spec.los, s, s * s, true, &mu, &mu2);

                    soft_add_pole(local.data(), spec, s, mu, pw.total);
                }
                else {
                    soft_add(local.data(), spec, s, pw.total, 1);
                }
            });
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        for (size_t b = 0; b < csize; b++) args.out[b] += local[b];
    }

    if (args.timings) {
        using Sec = std::chrono::duration<double>;
        args.timings[0] = Sec(t1 - t0).count();
        args.timings[1] = Sec(Clock::now() - t1).count();
    }
}


// ---------------------------------------------------------------------------
// Adjoint
// ---------------------------------------------------------------------------
//
// Parallel over side 1, like the forward: a particle of side 1 is visited by
// exactly one thread, so its position and weight gradients need no atomics. Side
// 2 is the other way round -- a particle of side 2 pairs with particles handled
// by every thread -- so each thread accumulates into a private copy that is
// reduced once, at the end. The cost is 4 doubles per side-2 particle per
// thread; the alternative (a critical section per pair) is far worse.
void Count2SoftVjp(const Count2SoftVjpArgs& args) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    const MeshAttrs& mattrs = args.attrs.mattrs;
    SoftBinSpec& spec = *args.spec;
    // The caller's grid, when it is not arithmetic, lives in these host tables for
    // the duration of the count; the CUDA backend copies the very same tables to
    // the device instead (both derive them in soft_build_host_tables()).
    // (FLOAT is a common.h macro that cmath.h undefines in the CPU translation
    // units, and on the host it is double.)
    std::vector<double> s_centres, s_inv2s2, s_edges;
    std::vector<double> mu_centres, mu_inv2s2, mu_edges;
    soft_build_host_tables(spec, s_centres, s_inv2s2, s_edges,
                           mu_centres, mu_inv2s2, mu_edges);

    const IndexValue& index_value1 = args.p1.index_value;
    const IndexValue& index_value2 = args.p2.index_value;
    const bool have_w1 = index_value1.size_individual_weight != 0;
    const bool have_w2 = index_value2.size_individual_weight != 0;

    // Slot -> input particle: the mesh reorders by cell and the gradients are
    // per input particle.
    const ScalarMesh m1 = build_mesh(args.p1, mattrs, /*keep_original_index=*/true);
    const ScalarMesh m2 = build_mesh(args.p2, mattrs, /*keep_original_index=*/true);

    const auto t1 = Clock::now();

    const size_t n1 = args.p1.size;
    const size_t n2 = args.p2.size;
    const long total1 = static_cast<long>(m1.total);

    std::memset(args.grad_position1, 0, NDIM * n1 * sizeof(double));
    std::memset(args.grad_position2, 0, NDIM * n2 * sizeof(double));

    if (have_w1) std::memset(args.grad_weight1, 0, n1 * sizeof(double));
    if (have_w2) std::memset(args.grad_weight2, 0, n2 * sizeof(double));

    args.grad_sigma[0] = 0.;

    double gsigma = 0.;

#ifdef _OPENMP
#pragma omp parallel num_threads(args.nthreads) reduction(+ : gsigma)
#endif
    {
        // Per-thread partials; they are reduced into args at the end. (Not
        // named gp2/gw2: the per-pair pointers below would shadow them and
        // initialize from themselves.)
        std::vector<double> local_gp(NDIM * n2, 0.);
        std::vector<double> local_gw(have_w2 ? n2 : 0, 0.);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
        for (long ii = 0; ii < total1; ii++) {
            const double* position1 = m1.position(ii);
            const double* sposition1 = m1.sposition(ii);
            const double* value1 = m1.value(ii);
            const size_t i1 = m1.origin[ii];

            for_each_candidate(mattrs, m2, position1, sposition1, [&](size_t jj) {
                const double* position2 = m2.position(jj);
                const double* sposition2 = m2.sposition(jj);
                const double* value2 = m2.value(jj);
                const size_t i2 = m2.origin[jj];

                if (!is_selected_pair(sposition1, sposition2, position1, position2,
                                      args.attrs.sattrs, mattrs))
                    return;

                double diff[NDIM];
                difference(diff, position2, position1, mattrs);

                const double s = std::sqrt(dot3(diff, diff));

                // The same pair weight as the forward, kept as factors because
                // the weight cotangents are its partial derivatives.
                const PairWeight pw =
                    soft_pair_weight(value1, value2, index_value1, index_value2,
                                     args.attrs.wattrs);

                // The pair's mu, measured exactly as the forward measures it.
                double mu = 0., mu2 = 0.;

                if (spec.nell > 0)
                    compute_pair_mu(diff, sposition1, sposition2, position1, position2,
                                    spec.los, s, s * s, true, &mu, &mu2);

                SoftVjpPair acc;
                soft_vjp_pair(args.cotangent, spec, s, mu, pw.total, &acc);

                // d pair_weight / d w1 = w2 * bw, and the negative-weight term
                // does not depend on the individual weights.
                if (have_w1) args.grad_weight1[i1] += acc.dw * pw.w2 * pw.bw;
                if (have_w2) local_gw[i2] += acc.dw * pw.w1 * pw.bw;

                // s = |r1 - r2|, so ds/dr1 = (r1 - r2) / s = -diff / s. With soft
                // mu binning there is a second, mu-only direction:
                // mu = (diff . lhat) / s, so with rhat = diff / s and k the line
                // of sight's own endpoint dependence (zero for x/y/z),
                //
                //   dL/dr1_b = -ds rhat_b + dL/dmu [-(lhat_b - mu rhat_b)/s
                //                                    + k1 (rhat_b - mu lhat_b)]
                //   dL/dr2_b = +ds rhat_b + dL/dmu [ +(lhat_b - mu rhat_b)/s
                //                                    + k2 (rhat_b - mu lhat_b) ]
                //
                // (see soft_mu_line_of_sight for where k comes from).
                if (s > 0) {
                    const double e = acc.ds / s;
                    double* gp1 = &args.grad_position1[NDIM * i1];
                    double* gpos2 = &local_gp[NDIM * i2];

                    if (acc.dmu != 0) {
                        double lhat[NDIM];
                        double k1 = 0, k2 = 0;

                        soft_mu_line_of_sight(spec.los, sposition1, sposition2, position1,
                                              position2, lhat, &k1, &k2);

                        const double gm = acc.dmu;

                        for (size_t axis = 0; axis < NDIM; ++axis) {
                            const double rhat = diff[axis] / s;
                            const double dl = (lhat[axis] - mu * rhat) / s;
                            const double dk = rhat - mu * lhat[axis];

                            gp1[axis] += -e * diff[axis] + gm * (k1 * dk - dl);
                            gpos2[axis] += e * diff[axis] + gm * (k2 * dk + dl);
                        }
                    }
                    else {
                        for (size_t axis = 0; axis < NDIM; ++axis) {
                            gp1[axis] += -e * diff[axis];
                            gpos2[axis] += e * diff[axis];
                        }
                    }
                }

                if (acc.dsigma != 0) gsigma += acc.dsigma;
            });
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for (size_t k = 0; k < NDIM * n2; k++) args.grad_position2[k] += local_gp[k];
            for (size_t k = 0; k < (have_w2 ? n2 : 0); k++) args.grad_weight2[k] += local_gw[k];
        }
    }

    args.grad_sigma[0] = gsigma;

    if (args.timings) {
        using Sec = std::chrono::duration<double>;
        args.timings[0] = Sec(t1 - t0).count();
        args.timings[1] = Sec(Clock::now() - t1).count();
    }
}

}  // namespace cpu
}  // namespace cucount
