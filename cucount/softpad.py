"""Edge padding ("buffer bins") for the soft histogram pair counts.

Why padding is needed
---------------------
With soft histogramming, the weight of a pair with separation ``s`` is spread
over the bins with the softmax of a Gaussian kernel of width ``sigma``.  The
Gaussian has infinite support, but the traversal only counts pairs inside the
binning range ``[s_edges[0], s_edges[-1])``, so the tails that reach beyond the
outermost edges are truncated.  The outermost bins therefore average ``xi(s)``
over a *half* kernel: mass leaking out of the last bin towards larger
separations (where ``xi`` is smaller for a steep correlation function) is not
compensated by mass leaking in from beyond ``s_edges[-1]``, because those pairs
are never counted.  The result is a positive bias (excess power) in the last
separation bin and a smaller one in the first, growing with the steepness of
``xi`` and with the kernel width.

How padding works
-----------------
``soft_pad = z`` extends the counting range by ``z`` kernel widths beyond the
outermost edges::

    s_edges[-1] -> s_edges[-1] * exp(z f)   (geometric continuation)
    s_edges[-1] -> s_edges[-1] + z sigma    (arithmetic continuation)

and symmetrically at the lower edge.  The extension is realised with *buffer
bins*: continuation bins appended to the separation grid, with centres on the
natural continuation of the grid (geometric for a log-spaced grid, arithmetic
otherwise, using the local spacing at each edge).  The softmax therefore
normalises over the real *and* the buffer centres, exactly like an infinite
virtual continuation of the grid, so pairs beyond the original edges contribute
their proper (Gaussian-decayed) weight to the outermost real bins, while pairs
near the edges lose the spurious boost from the truncated normalisation.  The
buffer-bin counts are discarded from the results; the real bins are returned
with their original edges.

The padding is quantised to whole continuation bins: the number of buffer bins
per side is ``ceil(z * sigma_edge / spacing)``.  ``soft_pad = 0`` disables
padding and restores the previous behaviour (softmax over the real bins only,
pairs in ``[s_edges[0], s_edges[-1])``).

Cost
----
Padding enumerates the pairs in the buffer shells as well.  At the upper edge
their number grows with the enclosed volume, ~ ``exp(3 z f) - 1`` relative to
the pairs inside ``s_edges[-1]`` for log-spaced bins, and ~
``((smax + z sigma)/smax)^3 - 1`` for linear bins.  Lower-edge padding is cheap
(the volume below ``s_edges[0]`` is tiny).  In a periodic box the extended
maximum edge is clamped below half the box size (the minimum-image limit), so
setups binning close to ``L/2`` get reduced or no upper padding.
"""

import numpy as np

__all__ = ['SOFT_PAD_DEFAULT', 'validate_soft_pad', 'plan_padding',
           'pad_cotangent', 'slice_soft_result']

#: Default edge padding, in kernel widths.
SOFT_PAD_DEFAULT = 2.0


def validate_soft_pad(soft_pad):
    """Validate and normalise a ``soft_pad`` request.

    ``None`` selects :data:`SOFT_PAD_DEFAULT`; booleans are rejected because
    ``True`` silently meaning "one kernel width" is more likely a mistake than
    an intent.  Returns a finite, non-negative ``float``.
    """
    if soft_pad is None:
        return float(SOFT_PAD_DEFAULT)
    if isinstance(soft_pad, (bool, np.bool_)):
        raise TypeError(f'soft_pad must be a number, got {soft_pad!r}')
    try:
        pad = float(soft_pad)
    except (TypeError, ValueError):
        raise TypeError(f'soft_pad must be a number, got {soft_pad!r}') from None
    if not np.isfinite(pad) or pad < 0.0:
        raise ValueError(f'soft_pad must be finite and non-negative, got {soft_pad!r}')
    return pad


def _is_uniform(edges):
    """True when the edges are an arithmetic progression."""
    edges = np.asarray(edges, dtype=np.float64)
    d = np.diff(edges)
    return bool(np.all(np.abs(d - d[0]) <= 1e-9 * abs(d[0]) + 1e-14))


def _is_log_uniform(edges):
    """True when the positive edges are a geometric progression."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges[0] <= 0.0:
        return False
    d = np.diff(np.log(edges))
    return bool(np.all(np.abs(d - d[0]) <= 1e-9 * abs(d[0]) + 1e-14))


def plan_padding(edges, soft_pad=None, soft_sigma=None, soft_sigma_log=False,
                 max_edge=None):
    """Extended separation grid with buffer bins for soft-histogram padding.

    Parameters
    ----------
    edges : ndarray
        Original separation bin edges, length ``ns + 1``.
    soft_pad : float, optional
        Padding in kernel widths (``z``); ``0`` disables padding, ``None``
        selects :data:`SOFT_PAD_DEFAULT`.
    soft_sigma : float or None
        Kernel width in separation units.  ``None`` means the library default:
        half of each bin's width at the relevant edge (half the mean width on a
        uniform grid).  With ``soft_sigma_log`` it is instead the scale-free
        fraction ``f`` of ``sigma(s) = f s``.
    soft_sigma_log : bool
        Treat ``soft_sigma`` as the scale-free fraction rather than an absolute
        width, and force the geometric continuation.  Without it a log-uniform
        grid is still continued geometrically, because that is the natural
        continuation of the grid; an arbitrary grid uses the local spacing.
    max_edge : float or None
        Upper limit for the extended maximum edge (half the periodic box size);
        ``None`` means no limit.

    Returns
    -------
    (edges_ext, nlo, nhi, sigma) :
        Extended edges (identical to ``edges`` when no padding applies), the
        numbers of buffer bins below and above, and a kernel width to pass
        explicitly to the C library when it is not ``None``.  The latter pins a
        default width to the *original* grid; it is ``None`` when an explicit
        width was given or the kernel's per-bin default already reproduces the
        plan (a non-uniform grid).
    """
    edges = np.asarray(edges, dtype=np.float64)
    pad = validate_soft_pad(soft_pad)
    if pad <= 0.0:
        return edges, 0, 0, None
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError('soft padding needs a 1D array of at least two edges')

    fractional = bool(soft_sigma_log)
    if fractional and edges[0] <= 0.0 and soft_sigma is None:
        raise ValueError('soft_sigma_log with soft_pad requires positive '
                         'edges[0] (or an explicit soft_sigma)')
    geometric = fractional or _is_log_uniform(edges)

    # Kernel width at each end.  A width-scaled default is per-bin, so the two
    # edges can differ; an absolute sigma is the same everywhere.
    if soft_sigma is None:
        if fractional:
            sig = 0.5 * float(np.log(edges[-1] / edges[0])) / (edges.size - 1)
            sig_lo = sig_hi = sig
        else:
            sig_lo = 0.5 * float(edges[1] - edges[0])
            sig_hi = 0.5 * float(edges[-1] - edges[-2])
            sig = sig_lo
    else:
        sig = float(soft_sigma)
        if not np.isfinite(sig) or sig <= 0.0:
            raise ValueError(f'soft_sigma must be positive and finite, got {soft_sigma!r}')
        sig_lo = sig_hi = sig

    if geometric:
        if edges[0] <= 0.0:
            nlo = 0  # no geometric continuation below zero
        else:
            dlo = float(np.log(edges[1] / edges[0]))
            if fractional:
                span_lo = pad * sig_lo / dlo  # sigma_lo = f * edges[0]
            else:
                span_lo = pad * sig_lo / (edges[0] * dlo)
            nlo = max(int(np.ceil(span_lo - 1e-12)), 0)
        dhi = float(np.log(edges[-1] / edges[-2]))
        if fractional:
            span_hi = pad * sig_hi / dhi
        else:
            span_hi = pad * sig_hi / (edges[-1] * dhi)
        nhi = max(int(np.ceil(span_hi - 1e-12)), 0)
        if max_edge is not None:
            if edges[-1] >= max_edge:
                nhi = 0
            elif nhi > 0:
                room = np.log(max_edge / edges[-1]) / dhi
                nhi = min(nhi, max(int(np.floor(room + 1e-12)), 0))
        lo = (edges[0] * np.exp(-dlo * np.arange(nlo, 0, -1)) if nlo
              else np.empty(0))
        hi = (edges[-1] * np.exp(dhi * np.arange(1, nhi + 1)) if nhi
              else np.empty(0))
    else:
        dlo = float(edges[1] - edges[0])
        dhi = float(edges[-1] - edges[-2])
        # The lower edge must stay non-negative: no pair exists below zero.
        nlo = max(int(np.ceil(pad * sig_lo / dlo - 1e-12)), 0)
        nlo = min(nlo, max(int(np.floor(edges[0] / dlo + 1e-12)), 0))
        nhi = max(int(np.ceil(pad * sig_hi / dhi - 1e-12)), 0)
        if max_edge is not None:
            if edges[-1] >= max_edge:
                nhi = 0
            elif nhi > 0:
                room = (max_edge - edges[-1]) / dhi
                nhi = min(nhi, max(int(np.floor(room + 1e-12)), 0))
        lo = (edges[0] - dlo * np.arange(nlo, 0, -1) if nlo else np.empty(0))
        hi = (edges[-1] + dhi * np.arange(1, nhi + 1) if nhi else np.empty(0))

    # On a uniform grid the kernel's default is one absolute width, so pin it
    # explicitly (the arithmetic extension reproduces it either way, but this
    # keeps the plan self-describing and matches pyfcfc).  A non-uniform
    # default is per-bin: the real bins keep their local spacing, so the kernel
    # reproduces the same widths without a pin.  A scale-free default is pinned
    # because the C library has no separate fractional-width request here.
    if soft_sigma is None and fractional:
        sigma_pin = sig
    elif soft_sigma is None and _is_uniform(edges):
        sigma_pin = sig_lo
    else:
        sigma_pin = None
    if nlo == 0 and nhi == 0:
        return edges, 0, 0, None
    edges_ext = np.concatenate([lo, edges, hi]).astype(np.float64)
    return edges_ext, nlo, nhi, sigma_pin


def pad_cotangent(cot, nlo, ns_ext):
    """Embed a cotangent of the real bins in the extended bin grid.

    The buffer bins get zero cotangents: they are discarded from the forward
    counts, so no upstream gradient flows through them.  They do still enter the
    softmax normalisation of the real-bin weights, which the adjoint kernel
    replays exactly.
    """
    cot = np.asarray(cot)
    if cot.ndim == 1:
        out = np.zeros((ns_ext,), dtype=np.float64)
        out[nlo:nlo + cot.shape[0]] = cot
    elif cot.ndim == 2:
        out = np.zeros((ns_ext, cot.shape[1]), dtype=np.float64)
        out[nlo:nlo + cot.shape[0]] = cot
    else:
        raise ValueError(f'cotangent must be 1D or 2D, got shape {cot.shape}')
    return out


def slice_soft_result(result, nlo, nhi, ns_real, edges_ext=None, soft_pad=None):
    """Drop the buffer bins from a soft-count result, in place.

    ``result`` is ``{'weight': counts, 'soft': meta}``.  The weight is sliced
    along its separation axis and the metadata is annotated with the real and
    padded bin counts, the offsets and (when padded) the extended edges, which
    the lsstypes wrapper needs to build the soft self-pair profile.
    """
    meta = result.get('soft')
    weight = result.get('weight')
    if weight is not None:
        ns_ext = int(weight.shape[0])
    elif meta is not None:
        ns_ext = int(meta.get('nbins', ns_real))
    else:
        ns_ext = int(ns_real)
    if nlo == 0 and ns_real == ns_ext:
        if meta is not None:
            meta.setdefault('soft_pad',
                            float(SOFT_PAD_DEFAULT) if soft_pad is None else float(soft_pad))
            meta.setdefault('pad_lo', 0)
            meta.setdefault('pad_hi', 0)
            meta.setdefault('nbins_padded', ns_ext)
        return result

    if weight is not None:
        result['weight'] = weight[nlo:nlo + ns_real]
    if meta is not None:
        meta['nbins'] = int(ns_real)
        meta['nbins_padded'] = int(ns_ext)
        meta['soft_pad'] = float(soft_pad) if soft_pad is not None else 0.0
        meta['pad_lo'] = int(nlo)
        meta['pad_hi'] = int(nhi)
        if 'slots' in meta:
            meta['slots'] = int(ns_real) * max(1, int(meta.get('nell', 0)))
        if edges_ext is not None:
            meta['s_edges_ext'] = np.asarray(edges_ext, dtype=np.float64)
    return result
