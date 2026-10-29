#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <type_traits>
#include <memory>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "mesh.h"
#include "count2.h"
#include "count2_soft.h"
#include "count3close.h"
#include "count3.h"
#include "common.h"
#include "cucount.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

#define MAX_NBLOCKS 256
#define MAX_NTHREADS_PER_BLOCK 512

// -----------------------------------------------------------------------------
// Static state for count2
// -----------------------------------------------------------------------------

static MeshAttrs mattrs2;
static BinAttrs battrs2;
static SelectionAttrs sattrs2;
static WeightAttrs wattrs2;
static SplitAttrs spattrs2;
static IndexValue index_value2[2] = {0};
// Soft binning rides the same attrs; only the tap window is extra state.
static SoftBinSpec soft_spec2;

// Host copy of the optional mu edges the setter was given; soft_spec2 points
// into it (see set_count2_soft_attrs_py).
static std::vector<FLOAT> soft_mu_edges2;

// -----------------------------------------------------------------------------
// Static state for count3close
// -----------------------------------------------------------------------------

static MeshAttrs mattrs3_1;
static MeshAttrs mattrs3_2;
static MeshAttrs mattrs3_3;
static BinAttrs battrs3_12;
static BinAttrs battrs3_13;
static BinAttrs battrs3_23;
static SelectionAttrs sattrs3_12;
static SelectionAttrs sattrs3_13;
static SelectionAttrs sattrs3_23;
static SelectionAttrs veto3_12;
static SelectionAttrs veto3_13;
static SelectionAttrs veto3_23;
static WeightAttrs wattrs3;
static CLOSE_PAIR close_pair_3 = CLOSE_PAIR_12;
static IndexValue index_value3[3] = {0};

// -----------------------------------------------------------------------------
// Owned host-copy helpers
// -----------------------------------------------------------------------------

// The staged attrs point into numpy buffers the caller may drop as soon as the
// setter returns, so every array is copied here and kept alive until the next
// staging of the same entry point.
//
// One owner per entry point, not one list for the module: staging count2's
// attrs must not free the arrays count3 is still pointing at. Identical to
// OwnedArrays in cpu/src/ffi_bind.cpp.
struct OwnedArrays {
    std::vector<void*> ptrs;

    ~OwnedArrays() { clear(); }

    void clear() {
        for (void* p : ptrs) std::free(p);
        ptrs.clear();
    }

    FLOAT* copy(const FLOAT* src, size_t n) {
        FLOAT* buf = (FLOAT*) std::malloc(n * sizeof(FLOAT));
        if (!buf) throw std::bad_alloc();
        std::memcpy(buf, src, n * sizeof(FLOAT));
        ptrs.push_back(buf);
        return buf;
    }

    void own(BinAttrs* battrs) {
        for (size_t i = 0; i < battrs->ndim; i++) {
            if (battrs->asize[i] != 0 && battrs->array[i] != nullptr)
                battrs->array[i] = copy(battrs->array[i], battrs->asize[i]);
        }
    }

    void own(WeightAttrs* wattrs) {
        if (wattrs->bitwise.p_nbits > 0 && wattrs->bitwise.p_correction_nbits != nullptr) {
            const size_t size = wattrs->bitwise.p_nbits * wattrs->bitwise.p_nbits;
            wattrs->bitwise.p_correction_nbits =
                copy(wattrs->bitwise.p_correction_nbits, size);
        }
        for (size_t idim = 0; idim < wattrs->angular.ndim; ++idim) {
            if (wattrs->angular.sep[idim] == nullptr) continue;
            // sep_is_edges is per axis: indexed, not taken whole. Read as a
            // scalar it is an array decaying to a pointer, so always true,
            // and every axis is copied one element long.
            const size_t n = wattrs->angular.sep_is_edges[idim]
                ? wattrs->angular.shape[idim] + 1 : wattrs->angular.shape[idim];
            wattrs->angular.sep[idim] = copy(wattrs->angular.sep[idim], n);
        }
        if (wattrs->angular.size > 0 && wattrs->angular.weight != nullptr)
            wattrs->angular.weight = copy(wattrs->angular.weight, wattrs->angular.size);
    }
};

static OwnedArrays owned2;
static OwnedArrays owned3;

// -----------------------------------------------------------------------------
// Python setters for count2 attrs
// -----------------------------------------------------------------------------

void set_count2_attrs_py(
    MeshAttrs_py mattrs_py,
    BinAttrs_py battrs_py,
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs_py = SelectionAttrs_py(),
    const SplitAttrs_py spattrs_py = SplitAttrs_py())
{
    owned2.clear();

    mattrs2 = mattrs_py.data();
    battrs2 = battrs_py.data();
    wattrs2 = wattrs_py.data();
    sattrs2 = sattrs_py.data();
    spattrs2 = spattrs_py.data();

    owned2.own(&battrs2);
    owned2.own(&wattrs2);
}

// -----------------------------------------------------------------------------
// Python setter for the soft-binning request
// -----------------------------------------------------------------------------

// The soft kernel is an extension, so its request is resolved into a
// SoftBinSpec here and kept beside the count2 attrs rather than inside them:
// nothing the hard kernel reads grows a field for a feature it does not serve.
void set_count2_soft_attrs_py(
    MeshAttrs_py mattrs_py,
    BinAttrs_py battrs_py,
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs_py = SelectionAttrs_py(),
    const SplitAttrs_py spattrs_py = SplitAttrs_py(),
    double soft_sigma = 0.,
    double soft_cutoff = 1e-13,
    const std::string &soft_taps = "exact",
    const std::string &soft_accum = "float32",
    double soft_mu_sigma = 0.,
    int soft_mu_bins = 0,
    py::object soft_mu_edges = py::none())
{
    owned2.clear();

    mattrs2 = mattrs_py.data();
    battrs2 = battrs_py.data();
    wattrs2 = wattrs_py.data();
    sattrs2 = sattrs_py.data();
    spattrs2 = spattrs_py.data();

    owned2.own(&battrs2);
    owned2.own(&wattrs2);

    // Optional caller-supplied mu edges. The spec keeps a pointer into this
    // static vector, which stays stable until the next set (when the spec is
    // rebuilt), so the FFI handlers can copy the spec and launch later.
    soft_mu_edges2.clear();

    if (!soft_mu_edges.is_none()) {
        py::array_t<FLOAT, py::array::c_style | py::array::forcecast> arr =
            py::cast<py::array_t<FLOAT, py::array::c_style | py::array::forcecast>>(
                soft_mu_edges);
        const auto *data = arr.data();
        soft_mu_edges2.assign(data, data + arr.size());
    }

    size_t csize = 0;
    soft_spec2 = make_soft_bin_spec(
        battrs2, wattrs2, spattrs2, soft_sigma, soft_cutoff, &csize, soft_mu_sigma,
        soft_mu_bins, soft_mu_edges2.empty() ? nullptr : soft_mu_edges2.data(),
        (int) soft_mu_edges2.size());

    if (soft_taps == "fast") {
        soft_spec2.fast_taps = 1;
    }
    else if (soft_taps != "exact") {
        throw std::invalid_argument(
            "cuda backend, soft binning: soft_taps must be 'exact' or 'fast'");
    }

    // f32 tile accumulation is the default (see SoftBinSpec::acc_float); 'float64'
    // restores the bit-comparable path for callers who need the extra digits.
    if (soft_accum == "float64") {
        soft_spec2.acc_float = 0;
    }
    else if (soft_accum != "float32") {
        throw std::invalid_argument(
            "cuda backend, soft binning: soft_accum must be 'float32' or 'float64'");
    }
}

// -----------------------------------------------------------------------------
// Python setters for count3close attrs
// -----------------------------------------------------------------------------

void set_count3close_attrs_py(
    MeshAttrs_py mattrs1_py,
    MeshAttrs_py mattrs2_py,
    MeshAttrs_py mattrs3_py,
    BinAttrs_py battrs12_py,
    BinAttrs_py battrs13_py,
    py::object battrs23_obj = py::none(),
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs12_py = SelectionAttrs_py(),
    const SelectionAttrs_py sattrs13_py = SelectionAttrs_py(),
    const SelectionAttrs_py sattrs23_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto12_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto13_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto23_py = SelectionAttrs_py(),
    py::tuple close_pair = py::make_tuple(1, 2))
{
    owned3.clear();

    mattrs3_1 = mattrs1_py.data();
    mattrs3_2 = mattrs2_py.data();
    mattrs3_3 = mattrs3_py.data();

    battrs3_12 = battrs12_py.data();
    battrs3_13 = battrs13_py.data();

    if (battrs23_obj.is_none()) {
        std::memset(&battrs3_23, 0, sizeof(BinAttrs));
    }
    else {
        battrs3_23 = py::cast<BinAttrs_py>(battrs23_obj).data();
    }

    wattrs3 = wattrs_py.data();

    sattrs3_12 = sattrs12_py.data();
    sattrs3_13 = sattrs13_py.data();
    sattrs3_23 = sattrs23_py.data();

    veto3_12 = veto12_py.data();
    veto3_13 = veto13_py.data();
    veto3_23 = veto23_py.data();

    close_pair_3 = parse_close_pair(close_pair);

    owned3.own(&battrs3_12);
    owned3.own(&battrs3_13);
    if (battrs3_23.ndim) owned3.own(&battrs3_23);
    owned3.own(&wattrs3);
}


void set_count3_attrs_py(
    MeshAttrs_py mattrs1_py,
    MeshAttrs_py mattrs2_py,
    MeshAttrs_py mattrs3_py,
    BinAttrs_py battrs12_py,
    BinAttrs_py battrs13_py,
    WeightAttrs_py wattrs_py = WeightAttrs_py(),
    const SelectionAttrs_py sattrs12_py = SelectionAttrs_py(),
    const SelectionAttrs_py sattrs13_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto12_py = SelectionAttrs_py(),
    const SelectionAttrs_py veto13_py = SelectionAttrs_py())
{
    set_count3close_attrs_py(
        mattrs1_py,
        mattrs2_py,
        mattrs3_py,
        battrs12_py,
        battrs13_py,
        py::none(),
        wattrs_py,
        sattrs12_py,
        sattrs13_py,
        SelectionAttrs_py(),
        veto12_py,
        veto13_py,
        SelectionAttrs_py(),
        py::make_tuple(1, 2));
}

// -----------------------------------------------------------------------------
// Index-value setters
// -----------------------------------------------------------------------------

void set_count2_index_value_py(
    const size_t iparticle,
    const int size_split = 0,
    const int size_spin = 0,
    const int size_individual_weight = 0,
    const int size_bitwise_weight = 0,
    const int size_negative_weight = 0)
{
    index_value2[iparticle] = get_index_value(
        size_split, size_spin, size_individual_weight, size_bitwise_weight, size_negative_weight);
}

void set_count3close_index_value_py(
    const size_t iparticle,
    const int size_split = 0,
    const int size_spin = 0,
    const int size_individual_weight = 0,
    const int size_bitwise_weight = 0,
    const int size_negative_weight = 0)
{
    index_value3[iparticle] = get_index_value(
        size_split, size_spin, size_individual_weight, size_bitwise_weight, size_negative_weight);
}

// -----------------------------------------------------------------------------
// Layout helpers
// -----------------------------------------------------------------------------

py::tuple get_count2_layout_py()
{
    Count2Layout layout = get_count2_layout(
        index_value2[0],
        index_value2[1],
        battrs2,
        spattrs2);

    py::tuple shape(layout.shape.size());
    for (size_t i = 0; i < layout.shape.size(); ++i) {
        shape[i] = py::int_(layout.shape[i]);
    }

    return py::make_tuple(layout.names, shape);
}

py::tuple get_count3close_layout_py()
{
    Count3Layout layout = get_count3_layout(
        battrs3_12,
        battrs3_13,
        battrs3_23);

    py::tuple shape(layout.shape.size());
    for (size_t i = 0; i < layout.shape.size(); ++i) {
        shape[i] = py::int_(layout.shape[i]);
    }

    return py::make_tuple(layout.names, shape);
}


py::tuple get_count3_layout_py()
{
    BinAttrs battrs23{};
    Count3Layout layout = get_count3_layout(
        battrs3_12,
        battrs3_13,
        battrs23);

    py::tuple shape(layout.shape.size());
    for (size_t i = 0; i < layout.shape.size(); ++i) {
        shape[i] = py::int_(layout.shape[i]);
    }

    return py::make_tuple(layout.names, shape);
}


// -----------------------------------------------------------------------------
// FFI helpers
// -----------------------------------------------------------------------------

// The scratch buffer is just bytes: it is declared with the same element type as
// the call's float type so that a float32 call does not need a float64 (and
// therefore JAX_ENABLE_X64) array to carry its scratch.
template <ffi::DataType DT>
void set_mem_buffer(DeviceMemoryBuffer *membuffer, ffi::ResultBuffer<DT> buffer) {
    constexpr size_t width = (DT == ffi::F32) ? 4 : 8;

    membuffer->ptr = (void *) buffer->typed_data();
    membuffer->size = buffer->dimensions().front() * width / sizeof(char);
    membuffer->offset = 0;
}

Particles get_ffi_particles(
    ffi::Buffer<ffi::F64> positions,
    ffi::Buffer<ffi::F64> values,
    IndexValue index_value)
{
    Particles particles;
    particles.positions = positions.typed_data();
    particles.values = values.typed_data();
    particles.size = positions.dimensions().front();
    particles.index_value = index_value;
    return particles;
}

// -----------------------------------------------------------------------------
// count2 FFI impl
// -----------------------------------------------------------------------------

ffi::Error count2Impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> positions1,
    ffi::Buffer<ffi::F64> values1,
    ffi::Buffer<ffi::F64> positions2,
    ffi::Buffer<ffi::F64> values2,
    ffi::ResultBuffer<ffi::F64> counts,
    ffi::ResultBuffer<ffi::F64> buffer)
{
    Particles list_particles[MAX_NMESH];
    Mesh list_mesh[MAX_NMESH];

    for (size_t imesh = 0; imesh < MAX_NMESH; imesh++) {
        list_particles[imesh].size = 0;
        list_mesh[imesh].total_nparticles = 0;
        // set_mesh skips empty catalogues, so leave no dangling slot -> input map
        // behind for the free path to follow.
        list_mesh[imesh].original_index = nullptr;
    }

    list_particles[0] = get_ffi_particles(positions1, values1, index_value2[0]);
    list_particles[1] = get_ffi_particles(positions2, values2, index_value2[1]);

    DeviceMemoryBuffer membuffer;
    set_mem_buffer(&membuffer, buffer);
    membuffer.nblocks = MAX_NBLOCKS;
    membuffer.nthreads_per_block = MAX_NTHREADS_PER_BLOCK * 16;  // not used in memory allocation

    set_mesh(list_particles, list_mesh, mattrs2, &membuffer, stream);
    const Count2Attrs attrs{mattrs2, battrs2, wattrs2, sattrs2, spattrs2};
    count2(counts->typed_data(), list_mesh, attrs, &membuffer, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    count2ffi, count2Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);

// -----------------------------------------------------------------------------
// soft count2 FFI impl
// -----------------------------------------------------------------------------

// The two entry points below are float64 *and* float32, selected by the caller's
// dtype: with JAX_ENABLE_X64 unset no float64 array can even be constructed on
// the jax side, so "cast in the frontend" is not an option there. The kernel is
// float64 internally either way; a float32 call widens its samples into
// `soft_scratch` and narrows the results back once, at the boundary.

template <ffi::DataType DT>
static size_t ffi_elements(const ffi::Buffer<DT> &buffer)
{
    size_t n = 1;

    for (int64_t dim : buffer.dimensions()) {
        n *= (size_t) dim;
    }

    return n;
}

// XLA owns only the declared result buffers, so a float32 call has nowhere to
// put its float64 working set: stage it here and grow on demand. Note the two
// consequences: the memory is per process (a larger call reallocates, and
// cudaFree/cudaMalloc synchronise), and two calls in flight on different streams
// would share it -- the attrs staging (`soft_spec2`) already assumes one soft
// call at a time per process.
struct SoftScratch {
    std::vector<void *> ptrs;
    std::vector<size_t> bytes;

    FLOAT *f64(size_t slot, size_t n)
    {
        if (slot >= ptrs.size()) {
            ptrs.resize(slot + 1, nullptr);
            bytes.resize(slot + 1, 0);
        }

        const size_t want = n * sizeof(FLOAT);

        if (bytes[slot] < want) {
            if (ptrs[slot]) {
                cudaFree(ptrs[slot]);
            }

            void *p = nullptr;
            cudaMalloc(&p, want);
            ptrs[slot] = p;
            bytes[slot] = p ? want : 0;
        }

        return (FLOAT *) ptrs[slot];
    }
};

static SoftScratch soft_scratch;

enum {
    SOFT_SCRATCH_POS1 = 0,
    SOFT_SCRATCH_VAL1,
    SOFT_SCRATCH_POS2,
    SOFT_SCRATCH_VAL2,
    SOFT_SCRATCH_COUNTS,
    SOFT_SCRATCH_COTANGENT,
    SOFT_SCRATCH_GPOS1,
    SOFT_SCRATCH_GPOS2,
    SOFT_SCRATCH_GW1,
    SOFT_SCRATCH_GW2,
    SOFT_SCRATCH_GSIGMA,
};

static Particles soft_particles(const FLOAT *positions, const FLOAT *values,
                                size_t n, IndexValue index_value)
{
    Particles particles;
    particles.positions = const_cast<FLOAT *>(positions);
    particles.values = const_cast<FLOAT *>(values);
    particles.size = n;
    particles.index_value = index_value;
    return particles;
}

template <ffi::DataType DT>
ffi::Error count2SoftImpl(
    cudaStream_t stream,
    ffi::Buffer<DT> positions1,
    ffi::Buffer<DT> values1,
    ffi::Buffer<DT> positions2,
    ffi::Buffer<DT> values2,
    ffi::ResultBuffer<DT> counts,
    ffi::ResultBuffer<DT> buffer)
{
    Particles list_particles[MAX_NMESH];
    Mesh list_mesh[MAX_NMESH];

    for (size_t imesh = 0; imesh < MAX_NMESH; imesh++) {
        list_particles[imesh].size = 0;
        list_mesh[imesh].total_nparticles = 0;
        // set_mesh skips empty catalogues, so leave no dangling slot -> input map
        // behind for the free path to follow.
        list_mesh[imesh].original_index = nullptr;
    }

    const size_t n1 = (size_t) positions1.dimensions().front();
    const size_t n2 = (size_t) positions2.dimensions().front();

    const FLOAT *p1 = nullptr;
    const FLOAT *v1 = nullptr;
    const FLOAT *p2 = nullptr;
    const FLOAT *v2 = nullptr;
    FLOAT *hist = nullptr;

    if constexpr (DT == ffi::F64) {
        p1 = positions1.typed_data();
        v1 = values1.typed_data();
        p2 = positions2.typed_data();
        v2 = values2.typed_data();
        hist = counts->typed_data();
    }
    else {
        const size_t npos1 = ffi_elements(positions1);
        const size_t nval1 = ffi_elements(values1);
        const size_t npos2 = ffi_elements(positions2);
        const size_t nval2 = ffi_elements(values2);

        p1 = soft_scratch.f64(SOFT_SCRATCH_POS1, npos1);
        soft_cast_f32_to_f64(const_cast<FLOAT *>(p1), positions1.typed_data(),
                             npos1, stream);
        v1 = soft_scratch.f64(SOFT_SCRATCH_VAL1, nval1);
        soft_cast_f32_to_f64(const_cast<FLOAT *>(v1), values1.typed_data(),
                             nval1, stream);
        p2 = soft_scratch.f64(SOFT_SCRATCH_POS2, npos2);
        soft_cast_f32_to_f64(const_cast<FLOAT *>(p2), positions2.typed_data(),
                             npos2, stream);
        v2 = soft_scratch.f64(SOFT_SCRATCH_VAL2, nval2);
        soft_cast_f32_to_f64(const_cast<FLOAT *>(v2), values2.typed_data(),
                             nval2, stream);

        // count2_soft zeroes the histogram itself.
        hist = soft_scratch.f64(SOFT_SCRATCH_COUNTS, (size_t) soft_spec2.nslots);
    }

    list_particles[0] = soft_particles(p1, v1, n1, index_value2[0]);
    list_particles[1] = soft_particles(p2, v2, n2, index_value2[1]);

    DeviceMemoryBuffer membuffer;
    set_mem_buffer(&membuffer, buffer);
    membuffer.nblocks = MAX_NBLOCKS;
    membuffer.nthreads_per_block = MAX_NTHREADS_PER_BLOCK * 16;  // not used in memory allocation

    set_mesh(list_particles, list_mesh, mattrs2, &membuffer, stream);
    const Count2Attrs attrs{mattrs2, battrs2, wattrs2, sattrs2, spattrs2};
    SoftBinSpec spec = soft_spec2;  // the kernel writes the staging path back into it
    count2_soft(hist, list_mesh, attrs, spec, &membuffer, stream);

    if constexpr (DT != ffi::F64) {
        soft_cast_f64_to_f32(counts->typed_data(), hist,
                             (size_t) soft_spec2.nslots, stream);
    }

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

#define CUCOUNT_DEFINE_SOFT_FFI(NAME, IMPL, TYPE)                            \
XLA_FFI_DEFINE_HANDLER_SYMBOL(                                               \
    NAME, IMPL<TYPE>,                                                        \
    ffi::Ffi::Bind()                                                         \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()                            \
        .Arg<ffi::Buffer<TYPE>>()                                            \
        .Arg<ffi::Buffer<TYPE>>()                                            \
        .Arg<ffi::Buffer<TYPE>>()                                            \
        .Arg<ffi::Buffer<TYPE>>()                                            \
        .Ret<ffi::Buffer<TYPE>>()                                            \
        .Ret<ffi::Buffer<TYPE>>()                                            \
);

CUCOUNT_DEFINE_SOFT_FFI(count2Softffi, count2SoftImpl, ffi::F64)
CUCOUNT_DEFINE_SOFT_FFI(count2SoftF32ffi, count2SoftImpl, ffi::F32)

#undef CUCOUNT_DEFINE_SOFT_FFI

// -----------------------------------------------------------------------------
// soft count2 adjoint FFI impl
// -----------------------------------------------------------------------------

// Signature: two catalogues (positions, values), the histogram cotangent, and
// five gradients. Every gradient operand is zeroed here because the kernel
// accumulates with atomics; the mesh keeps the slot -> input-index map the
// scatter/reorder in set_mesh would otherwise destroy.
//
// A float32 call stages the cotangent and the five gradients in float64: the
// adjoint cancels terms of order sigma^2 against each other, so narrowing it
// would throw away much more than it throws away in the forward.
template <ffi::DataType DT>
ffi::Error count2SoftVjpImpl(
    cudaStream_t stream,
    ffi::Buffer<DT> positions1,
    ffi::Buffer<DT> values1,
    ffi::Buffer<DT> positions2,
    ffi::Buffer<DT> values2,
    ffi::Buffer<DT> cotangent,
    ffi::ResultBuffer<DT> grad_positions1,
    ffi::ResultBuffer<DT> grad_positions2,
    ffi::ResultBuffer<DT> grad_values1,
    ffi::ResultBuffer<DT> grad_values2,
    ffi::ResultBuffer<DT> grad_sigma,
    ffi::ResultBuffer<DT> buffer)
{
    const size_t n1 = (size_t) positions1.dimensions().front();
    const size_t n2 = (size_t) positions2.dimensions().front();

    Particles list_particles[MAX_NMESH];
    Mesh list_mesh[MAX_NMESH];

    for (size_t imesh = 0; imesh < MAX_NMESH; imesh++) {
        list_particles[imesh].size = 0;
        list_mesh[imesh].total_nparticles = 0;
        // set_mesh skips empty catalogues, so leave no dangling slot -> input map
        // behind for the free path to follow.
        list_mesh[imesh].original_index = nullptr;
    }

    const FLOAT *p1 = nullptr;
    const FLOAT *v1 = nullptr;
    const FLOAT *p2 = nullptr;
    const FLOAT *v2 = nullptr;
    const FLOAT *g = nullptr;
    FLOAT *gpos1 = nullptr;
    FLOAT *gpos2 = nullptr;
    FLOAT *gw1 = nullptr;
    FLOAT *gw2 = nullptr;
    FLOAT *gsigma = nullptr;

    if constexpr (DT == ffi::F64) {
        p1 = positions1.typed_data();
        v1 = values1.typed_data();
        p2 = positions2.typed_data();
        v2 = values2.typed_data();
        g = cotangent.typed_data();
        gpos1 = grad_positions1->typed_data();
        gpos2 = grad_positions2->typed_data();
        gw1 = grad_values1->typed_data();
        gw2 = grad_values2->typed_data();
        gsigma = grad_sigma->typed_data();
    }
    else {
        const size_t npos1 = ffi_elements(positions1);
        const size_t nval1 = ffi_elements(values1);
        const size_t npos2 = ffi_elements(positions2);
        const size_t nval2 = ffi_elements(values2);
        const size_t nc = (size_t) soft_spec2.nslots;

        p1 = soft_scratch.f64(SOFT_SCRATCH_POS1, npos1);
        soft_cast_f32_to_f64(const_cast<FLOAT *>(p1), positions1.typed_data(),
                             npos1, stream);
        v1 = soft_scratch.f64(SOFT_SCRATCH_VAL1, nval1);
        soft_cast_f32_to_f64(const_cast<FLOAT *>(v1), values1.typed_data(),
                             nval1, stream);
        p2 = soft_scratch.f64(SOFT_SCRATCH_POS2, npos2);
        soft_cast_f32_to_f64(const_cast<FLOAT *>(p2), positions2.typed_data(),
                             npos2, stream);
        v2 = soft_scratch.f64(SOFT_SCRATCH_VAL2, nval2);
        soft_cast_f32_to_f64(const_cast<FLOAT *>(v2), values2.typed_data(),
                             nval2, stream);

        FLOAT *gc = soft_scratch.f64(SOFT_SCRATCH_COTANGENT, nc);
        soft_cast_f32_to_f64(gc, cotangent.typed_data(), nc, stream);
        g = gc;

        gpos1 = soft_scratch.f64(SOFT_SCRATCH_GPOS1, NDIM * n1);
        gpos2 = soft_scratch.f64(SOFT_SCRATCH_GPOS2, NDIM * n2);
        gw1 = soft_scratch.f64(SOFT_SCRATCH_GW1, n1);
        gw2 = soft_scratch.f64(SOFT_SCRATCH_GW2, n2);
        gsigma = soft_scratch.f64(SOFT_SCRATCH_GSIGMA, 1);
    }

    list_particles[0] = soft_particles(p1, v1, n1, index_value2[0]);
    list_particles[1] = soft_particles(p2, v2, n2, index_value2[1]);

    DeviceMemoryBuffer membuffer;
    set_mem_buffer(&membuffer, buffer);
    membuffer.nblocks = MAX_NBLOCKS;
    membuffer.nthreads_per_block = MAX_NTHREADS_PER_BLOCK * 16;

    CUDA_CHECK(cudaMemsetAsync(gpos1, 0, NDIM * n1 * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemsetAsync(gpos2, 0, NDIM * n2 * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemsetAsync(gw1, 0, n1 * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemsetAsync(gw2, 0, n2 * sizeof(FLOAT), stream));
    CUDA_CHECK(cudaMemsetAsync(gsigma, 0, sizeof(FLOAT), stream));

    // true: keep the slot -> input index map, the adjoint needs it to write each
    // pair's contribution back to the particles it came from.
    set_mesh(list_particles, list_mesh, mattrs2, &membuffer, stream, true);

    const Count2Attrs attrs{mattrs2, battrs2, wattrs2, sattrs2, spattrs2};
    count2_soft_vjp(g, gpos1, gpos2, gw1, gw2, gsigma,
                    list_mesh, attrs, soft_spec2, &membuffer, stream);

    if constexpr (DT != ffi::F64) {
        soft_cast_f64_to_f32(grad_positions1->typed_data(), gpos1,
                             NDIM * n1, stream);
        soft_cast_f64_to_f32(grad_positions2->typed_data(), gpos2,
                             NDIM * n2, stream);
        soft_cast_f64_to_f32(grad_values1->typed_data(), gw1, n1, stream);
        soft_cast_f64_to_f32(grad_values2->typed_data(), gw2, n2, stream);
        soft_cast_f64_to_f32(grad_sigma->typed_data(), gsigma, 1, stream);
    }

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

#define CUCOUNT_DEFINE_SOFT_VJP_FFI(NAME, IMPL, TYPE)                        \
XLA_FFI_DEFINE_HANDLER_SYMBOL(                                               \
    NAME, IMPL<TYPE>,                                                        \
    ffi::Ffi::Bind()                                                         \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()                            \
        .Arg<ffi::Buffer<TYPE>>()                                            \
        .Arg<ffi::Buffer<TYPE>>()                                            \
        .Arg<ffi::Buffer<TYPE>>()                                            \
        .Arg<ffi::Buffer<TYPE>>()                                            \
        .Arg<ffi::Buffer<TYPE>>()                                            \
        .Ret<ffi::Buffer<TYPE>>()                                            \
        .Ret<ffi::Buffer<TYPE>>()                                            \
        .Ret<ffi::Buffer<TYPE>>()                                            \
        .Ret<ffi::Buffer<TYPE>>()                                            \
        .Ret<ffi::Buffer<TYPE>>()                                            \
        .Ret<ffi::Buffer<TYPE>>()                                            \
);

CUCOUNT_DEFINE_SOFT_VJP_FFI(count2SoftVjpffi, count2SoftVjpImpl, ffi::F64)
CUCOUNT_DEFINE_SOFT_VJP_FFI(count2SoftVjpF32ffi, count2SoftVjpImpl, ffi::F32)

#undef CUCOUNT_DEFINE_SOFT_VJP_FFI

// -----------------------------------------------------------------------------
// count3close FFI impl
// -----------------------------------------------------------------------------

ffi::Error count3closeImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> positions1,
    ffi::Buffer<ffi::F64> values1,
    ffi::Buffer<ffi::F64> positions2,
    ffi::Buffer<ffi::F64> values2,
    ffi::Buffer<ffi::F64> positions3,
    ffi::Buffer<ffi::F64> values3,
    ffi::ResultBuffer<ffi::F64> counts,
    ffi::ResultBuffer<ffi::F64> buffer)
{
    Particles list_particles[MAX_NMESH];

    for (size_t imesh = 0; imesh < MAX_NMESH; imesh++) {
        list_particles[imesh].size = 0;
    }

    list_particles[0] = get_ffi_particles(positions1, values1, index_value3[0]);
    list_particles[1] = get_ffi_particles(positions2, values2, index_value3[1]);
    list_particles[2] = get_ffi_particles(positions3, values3, index_value3[2]);

    DeviceMemoryBuffer membuffer;
    set_mem_buffer(&membuffer, buffer);
    membuffer.nblocks = MAX_NBLOCKS;
    membuffer.nthreads_per_block = MAX_NTHREADS_PER_BLOCK * 16;  // not used in memory allocation

    Mesh mesh1{};
    Mesh mesh2{};
    Mesh mesh3{};

    Particles plist[MAX_NMESH];
    Mesh mlist[MAX_NMESH];

    for (size_t i = 0; i < MAX_NMESH; ++i) {
        plist[i].size = 0;
        mlist[i].total_nparticles = 0;
    }

    plist[0] = list_particles[0];
    set_mesh(plist, mlist, mattrs3_1, &membuffer, stream);
    mesh1 = mlist[0];

    plist[0] = list_particles[1];
    set_mesh(plist, mlist, mattrs3_2, &membuffer, stream);
    mesh2 = mlist[0];

    plist[0] = list_particles[2];
    set_mesh(plist, mlist, mattrs3_3, &membuffer, stream);
    mesh3 = mlist[0];

    const Count3Attrs attrs{
        mattrs3_1, mattrs3_2, mattrs3_3,
        battrs3_12, battrs3_13, battrs3_23,
        wattrs3,
        sattrs3_12, sattrs3_13, sattrs3_23,
        veto3_12, veto3_13, veto3_23};

    count3close(counts->typed_data(), mesh1, mesh2, mesh3,
                attrs, close_pair_3, &membuffer, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }

    return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    count3closeffi, count3closeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);



ffi::Error count3Impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> positions1,
    ffi::Buffer<ffi::F64> values1,
    ffi::Buffer<ffi::F64> positions2,
    ffi::Buffer<ffi::F64> values2,
    ffi::Buffer<ffi::F64> positions3,
    ffi::Buffer<ffi::F64> values3,
    ffi::ResultBuffer<ffi::F64> counts,
    ffi::ResultBuffer<ffi::F64> buffer)
{
    Particles list_particles[MAX_NMESH];

    for (size_t imesh = 0; imesh < MAX_NMESH; imesh++) {
        list_particles[imesh].size = 0;
    }

    list_particles[0] = get_ffi_particles(positions1, values1, index_value3[0]);
    list_particles[1] = get_ffi_particles(positions2, values2, index_value3[1]);
    list_particles[2] = get_ffi_particles(positions3, values3, index_value3[2]);

    DeviceMemoryBuffer membuffer;
    set_mem_buffer(&membuffer, buffer);
    membuffer.nblocks = MAX_NBLOCKS;
    membuffer.nthreads_per_block = MAX_NTHREADS_PER_BLOCK;

    Mesh mesh1{};
    Mesh mesh2{};
    Mesh mesh3{};

    Particles plist[MAX_NMESH];
    Mesh mlist[MAX_NMESH];

    for (size_t i = 0; i < MAX_NMESH; ++i) {
        plist[i].size = 0;
        mlist[i].total_nparticles = 0;
    }

    plist[0] = list_particles[0];
    set_mesh(plist, mlist, mattrs3_1, &membuffer, stream);
    mesh1 = mlist[0];

    plist[0] = list_particles[1];
    set_mesh(plist, mlist, mattrs3_2, &membuffer, stream);
    mesh2 = mlist[0];

    plist[0] = list_particles[2];
    set_mesh(plist, mlist, mattrs3_3, &membuffer, stream);
    mesh3 = mlist[0];

    // count3 has no (2, 3) axis, so those members stay default.
    const Count3Attrs attrs{
        MeshAttrs{}, mattrs3_2, mattrs3_3,
        battrs3_12, battrs3_13, BinAttrs{},
        wattrs3,
        sattrs3_12, sattrs3_13, SelectionAttrs{},
        veto3_12, veto3_13, SelectionAttrs{}};

    count3(
        counts->typed_data(),
        mesh1,
        mesh2,
        mesh3,
        attrs,
        &membuffer,
        stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }

    return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    count3ffi, count3Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);

// -----------------------------------------------------------------------------
// Capsule helper
// -----------------------------------------------------------------------------

template <typename T>
py::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be an XLA FFI handler");
    return py::capsule(reinterpret_cast<void *>(fn));
}

// -----------------------------------------------------------------------------
// Module
// -----------------------------------------------------------------------------

PYBIND11_MODULE(ffi_cuda, m) {
    register_attrs(m);

    // count2 setup
    m.def("set_count2_attrs", &set_count2_attrs_py, "Set count2 attributes",
        py::arg("mattrs"),
        py::arg("battrs"),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs") = SelectionAttrs_py(),
        py::arg("spattrs") = SplitAttrs_py());

    m.def("set_count2_index_value", &set_count2_index_value_py, "Set count2 value indices",
        py::arg("iparticle"),
        py::arg("size_split") = 0,
        py::arg("size_spin") = 0,
        py::arg("size_individual_weight") = 0,
        py::arg("size_bitwise_weight") = 0,
        py::arg("size_negative_weight") = 0);

    m.def("count2", []() { return EncapsulateFfiCall(count2ffi); });

    // Soft (Gaussian-softmax) binning: an extension with its own request setter,
    // its own forward target and its own adjoint target, so the count2 path above
    // is untouched.
    m.def("set_count2_soft_attrs", &set_count2_soft_attrs_py, "Set soft count2 attributes",
        py::arg("mattrs"),
        py::arg("battrs"),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs") = SelectionAttrs_py(),
        py::arg("spattrs") = SplitAttrs_py(),
        py::arg("soft_sigma") = 0.,
        py::arg("soft_cutoff") = 1e-13,
        py::arg("soft_taps") = "exact",
        py::arg("soft_accum") = "float32",
        py::arg("soft_mu_sigma") = 0.,
        py::arg("soft_mu_bins") = 0,
        py::arg("soft_mu_edges") = py::none());

    m.def("count2_soft", []() { return EncapsulateFfiCall(count2Softffi); });
    m.def("count2_soft_vjp", []() { return EncapsulateFfiCall(count2SoftVjpffi); });
    // float32 twins, for callers that run with JAX_ENABLE_X64 unset: the samples
    // are widened here and the histogram is narrowed once on the way out.
    m.def("count2_soft_f32", []() { return EncapsulateFfiCall(count2SoftF32ffi); });
    m.def("count2_soft_vjp_f32", []() { return EncapsulateFfiCall(count2SoftVjpF32ffi); });

    // backward-compatible aliases
    m.def("set_attrs", &set_count2_attrs_py, "Set count2 attributes",
        py::arg("mattrs"),
        py::arg("battrs"),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs") = SelectionAttrs_py(),
        py::arg("spattrs") = SplitAttrs_py());

    m.def("set_index_value", &set_count2_index_value_py, "Set count2 value indices",
        py::arg("iparticle"),
        py::arg("size_split") = 0,
        py::arg("size_spin") = 0,
        py::arg("size_individual_weight") = 0,
        py::arg("size_bitwise_weight") = 0,
        py::arg("size_negative_weight") = 0);

    m.def(
        "get_count2_layout",
        &get_count2_layout_py,
        "Return (names, shape) for count2 outputs."
    );

    // count3close setup
    m.def("set_count3close_attrs", &set_count3close_attrs_py,
        py::arg("mattrs1"),
        py::arg("mattrs2"),
        py::arg("mattrs3"),
        py::arg("battrs12"),
        py::arg("battrs13"),
        py::arg("battrs23") = py::none(),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs12") = SelectionAttrs_py(),
        py::arg("sattrs13") = SelectionAttrs_py(),
        py::arg("sattrs23") = SelectionAttrs_py(),
        py::arg("veto12") = SelectionAttrs_py(),
        py::arg("veto13") = SelectionAttrs_py(),
        py::arg("veto23") = SelectionAttrs_py(),
        py::arg("close_pair") = py::make_tuple(1, 2));

    m.def("set_count3close_index_value", &set_count3close_index_value_py,
        "Set count3close value indices",
        py::arg("iparticle"),
        py::arg("size_split") = 0,
        py::arg("size_spin") = 0,
        py::arg("size_individual_weight") = 0,
        py::arg("size_bitwise_weight") = 0,
        py::arg("size_negative_weight") = 0);

    m.def(
        "get_count3close_layout",
        &get_count3close_layout_py,
        "Return (names, shape) for count3close outputs."
    );

    m.def("count3close", []() { return EncapsulateFfiCall(count3closeffi); });

    m.def("set_count3_attrs", &set_count3_attrs_py,
        py::arg("mattrs1"),
        py::arg("mattrs2"),
        py::arg("mattrs3"),
        py::arg("battrs12"),
        py::arg("battrs13"),
        py::arg("wattrs") = WeightAttrs_py(),
        py::arg("sattrs12") = SelectionAttrs_py(),
        py::arg("sattrs13") = SelectionAttrs_py(),
        py::arg("veto12") = SelectionAttrs_py(),
        py::arg("veto13") = SelectionAttrs_py());

    m.def("set_count3_index_value", &set_count3close_index_value_py,
        "Set count3 value indices",
        py::arg("iparticle"),
        py::arg("size_split") = 0,
        py::arg("size_spin") = 0,
        py::arg("size_individual_weight") = 0,
        py::arg("size_bitwise_weight") = 0,
        py::arg("size_negative_weight") = 0);

    m.def(
        "get_count3_layout",
        &get_count3_layout_py,
        "Return (names, shape) for count3 outputs."
    );

    m.def("count3", []() { return EncapsulateFfiCall(count3ffi); });

}