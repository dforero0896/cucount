// Soft (Gaussian-softmax) pair counts on the CPU: forward and adjoint.
//
// This mirrors cuda/include/count2_soft.h entry for entry, so the numpy
// frontend can run either backend through the same signature. The request
// resolution (SoftBinSpec, make_soft_bin_spec) is shared from include/soft.h,
// which is what keeps the two backends from deciding different windows, sigmas
// or mu quadratures for the same Python call; the per-pair math is an
// independent scalar port in src/count2_soft.cpp, in the same spirit as
// src/generic.cpp being a port of cuda/src/count2.cu (its comment calls the
// duplication deliberate: same structure, shared per-pair geometry, no shared
// object code so each backend keeps its own inner loop).
//
// No CUDA header is involved: the spec's table pointers, which the CUDA entry
// points at device memory, are host arrays here.
#pragma once

#include <cstddef>

// BIN_TYPE/LOS_TYPE/... come from the shared vocabulary, exactly as for the
// hard CPU path (see cpu/include/count2.h for why common.h is included with
// CUCOUNT_NO_CUDA defined).
#ifndef CUCOUNT_NO_CUDA
#define CUCOUNT_NO_CUDA
#endif
#include "common.h"
#include "args.h"
#include "soft.h"

namespace cucount {
namespace cpu {

// Writes spec.nslots entries: nbins for a plain soft-s request, nbins * nell
// for soft mu binning (s-major: slot = s bin * nell + multipole position).
struct Count2SoftArgs {
    Particles p1;
    Particles p2;
    Count2Attrs attrs;
    SoftBinSpec *spec;
    double *out;       // spec->nslots entries, zeroed by the caller
    int nthreads;
    double *timings;   // [mesh_seconds, count_seconds], or null
};

void Count2Soft(const Count2SoftArgs &args);


// The analytic adjoint of Count2Soft: given dL/dcounts, accumulate dL/dpositions,
// dL/dindividual weights and dL/dsigma. Same five outputs as the CUDA adjoint,
// because the jax layer's custom_vjp consumes them by name.
struct Count2SoftVjpArgs {
    Particles p1;
    Particles p2;
    Count2Attrs attrs;
    SoftBinSpec *spec;
    const double *cotangent;  // spec->nslots entries
    double *grad_position1;   // 3 * p1.size
    double *grad_position2;   // 3 * p2.size
    double *grad_weight1;     // p1.size, or null when it has no individual weights
    double *grad_weight2;     // p2.size, or null
    double *grad_sigma;       // one entry
    int nthreads;
    double *timings;
};

void Count2SoftVjp(const Count2SoftVjpArgs &args);

}  // namespace cpu
}  // namespace cucount
