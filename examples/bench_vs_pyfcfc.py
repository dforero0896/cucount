"""Soft pair counts on real data: cucount (GPU and CPU) versus pyfcfc (CPU, SIMD).

The same request goes through every engine that can serve it, and the cost of
the forward pass and of one reverse-mode pass is recorded:

* **pyfcfc** -- ``pyfcfc.jax.soft_pair_counts``: a ``custom_vjp`` around
  ``jax.pure_callback`` into the FCFC C library (SIMD kernels), with the
  multipoles integrated by ``pyfcfc.jax.multipoles`` inside the traced function,
  exactly as their examples do;
* **cucount, GPU** -- ``cucount.jax.soft.count2_soft``, reached through the XLA
  FFI custom call; the adjoint is the hand-written CUDA kernel behind the
  frontend's ``custom_vjp``;
* **cucount, CPU** -- the ``cucount.numpy.soft`` entry points, because the jax
  frontend is CUDA-only.  Its adjoint is again the analytic kernel, never finite
  differences.

Data: an AbacusSummit v2 cubic-box HOD mock (real space, ``LRG`` at z=0.5, box
2000 Mpc/h).  The positions are wrapped the way the holi-mocks reference loader
does, and the sample is used in full unless ``--nmax`` asks for a random
subsample.

Both codes are asked for the same thing: soft separation binning with
``sigma = --sigma-bins * ds`` (default one bin, so 1 Mpc/h here), and for the
multipoles a soft ``mu`` axis with cells of the same width in both codes
(pyfcfc keeps ``|mu|`` in [0, 1) with ``--nmu`` cells, cucount keeps the sign on
[-1, 1) with twice as many), so the two Legendre quadratures coincide up to the
folding, which is immaterial for even multipoles.

Both codes are also asked for the same *normalisation*.  pyfcfc's
``soft_pair_counts`` renormalises the kernel per pair (a softmax over the window
it keeps), so cucount is pinned to ``soft_norm='softmax'`` -- that is what
``--soft-norm`` defaults to here -- and the two engines then count the same
quantity.  cucount's own default is ``soft_norm='gaussian'``, a
discrete-quadrature kernel with no per-pair renormalisation, which saves the
second pass but leaves the row sums at quadrature accuracy (~0.4% at the default
half-bin width) and changes the adjoint cost; benchmarking that against the
pyfcfc oracle would compare two different estimators.  Ask for it explicitly
with ``--soft-norm gaussian`` if that is the point of the run, and the report
records which mode was used.

Typical use, inside one GPU allocation (``scripts/run_bench_vs_pyfcfc.sh``)::

    JAX_PLATFORMS=cpu python examples/bench_vs_pyfcfc.py \
        --engines pyfcfc,cucount_cpu
    python examples/bench_vs_pyfcfc.py --engines cucount_gpu

Every stage appends to ``examples/figures/bench_vs_pyfcfc.json`` and rewrites
the report (``BENCHMARK_SOFT_VS_PYFCFC.md``) and the figure, so partial results
are readable while a long run is still going.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault('MPLCONFIGDIR',
                      os.path.join(tempfile.gettempdir(), 'cucount-mpl'))

import numpy as np  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                       # .../cucount
WORKSPACE = os.path.dirname(REPO)                  # .../cucount_dev
FIGDIR = os.path.join(HERE, 'figures')
REPORT = os.path.join(REPO, 'BENCHMARK_SOFT_VS_PYFCFC.md')
STATE = os.path.join(FIGDIR, 'bench_vs_pyfcfc.json')


def _bootstrap():
    """Use the development build (and the local pyfcfc wheel), not site-packages."""
    sys.meta_path[:] = [f for f in sys.meta_path
                        if type(f).__name__ != 'ScikitBuildRedirectingFinder']
    for path in (os.path.join(WORKSPACE, '.devlib'), REPO,
                 os.path.join(WORKSPACE, '.pydeps')):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_bootstrap()


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------

def abacus_path(tracer, z, version, phase, root):
    """Reference path of an AbacusSummit mock (the holi-mocks convention)."""
    if version != 2:
        raise SystemExit('only AbacusSummit version 2 is wired up here')
    box = 'desi/cosmosim/SecondGenMocks/CubicBox'
    return (f'{root}/{box}/{tracer}/z{z:.3f}/AbacusSummit_base_c000_'
            f'ph{phase:03d}/{tracer}_real_space.fits')


def load_catalog(args):
    """Return (positions (N, 3), weights (N,), box, description)."""
    path = args.path or abacus_path(args.tracer, args.z, args.version,
                                    args.phase, args.root)
    if path.endswith(('.npy', '.npz')):
        data = np.load(path)
        if hasattr(data, 'files'):
            pos = np.asarray(data['pos'], dtype=np.float64)
            w = (np.asarray(data['w'], dtype=np.float64) if 'w' in data.files
                 else np.ones(len(pos)))
        else:
            pos = np.asarray(data, dtype=np.float64)
            w = np.ones(len(pos))
    else:
        from astropy.table import Table
        table = Table.read(path, memmap=True)
        pos = np.column_stack([np.asarray(table[c], dtype=np.float64)
                               for c in 'xyz'])
        w = np.ones(len(pos))
    # the reference loader's convention: shift a box and wrap
    pos += 0.5 * args.box
    pos %= args.box
    n_all = len(pos)
    if args.nmax and args.nmax < n_all:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(n_all, size=args.nmax, replace=False))
        pos, w = pos[keep], w[keep]
    description = f'{os.path.basename(path)}: {n_all} objects'
    if len(pos) < n_all:
        description += (f', random subsample of {len(pos)} '
                        f'(seed {args.seed})')
    return pos, w, args.box, description


# --------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------

def _block(tree):
    """jax.block_until_ready over a nested dict/tuple of arrays."""
    import jax
    if isinstance(tree, dict):
        for value in tree.values():
            _block(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            _block(value)
    elif tree is not None:
        try:
            jax.block_until_ready(tree)
        except (AttributeError, TypeError):
            pass


def _timed(fn, repeat):
    """Best and median wall time of `fn`, after one warm-up call."""
    fn()
    times = []
    out = None
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn()
        _block(out)
        times.append(time.perf_counter() - start)
    times = np.asarray(times)
    return float(times.min()), float(np.median(times)), out


# --------------------------------------------------------------------------
# binning and normalisation
# --------------------------------------------------------------------------

def make_grid(args):
    edges = np.arange(args.smin, args.smax + 1e-9, args.ds)
    if edges.size < 2:
        raise SystemExit('need at least one bin: check --smin/--smax/--ds')
    return {'edges': edges, 'ds': args.ds,
            'sigma': args.sigma_bins * args.ds,
            'nmu': args.nmu, 'dmu': 1.0 / args.nmu,
            'mu_bins_cucount': 2 * args.nmu,
            'mu_sigma': args.mu_sigma_bins / args.nmu}


def shell_volumes(edges, box):
    return (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3) / box ** 3


def pair_norm(w):
    """Ordered-pair normalisation (the FCFC convention).  Both codes count
    autocorrelation pairs twice, so one number normalises both."""
    return float(np.sum(w) ** 2 - np.sum(w ** 2))


def xi_from_counts(counts, edges, box, norm):
    """Normalise raw pair counts into xi (1-D or per multipole column)."""
    volume = norm * shell_volumes(edges, box)
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim == 1:
        return counts / volume
    return counts / volume[:, None]


# --------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------

def _store(kind, counts, grid, args):
    """xi curves of one request, keyed by ell, as plain lists for JSON."""
    xi = xi_from_counts(counts, grid['edges'], args.box, args._pair_norm)
    if xi.ndim == 1:
        return {'0': xi.tolist()}
    return {str(ell): xi[:, k].tolist() for k, ell in enumerate(args.ells)}


def engine_pyfcfc(args, pos, w, grid, cot_iso, cot_pole):
    """pyfcfc's jax front-end: custom_vjp -> pure_callback -> FCFC C kernels."""
    import jax
    import jax.numpy as jnp
    from pyfcfc.jax import multipoles, soft_pair_counts
    import pyfcfc.boxes as pboxes

    jax.config.update('jax_enable_x64', True)
    edges = grid['edges']
    common = dict(module='boxes', s_edges=edges, pairs=('DD',), labels=['D'],
                  box=args.box)

    def iso_counts(p, w_):
        # pyfcfc fixes its own truncation (1e-13); only cucount exposes it.
        return soft_pair_counts([p], [w_], bin=0, soft_sigma=grid['sigma'],
                                **common)['DD']

    def pole_counts(p, w_):
        cells = soft_pair_counts([p], [w_], bin=1, nmu=grid['nmu'],
                                 soft_mu=True, soft_mu_sigma=grid['mu_sigma'],
                                 soft_sigma=grid['sigma'], **common)['DD']
        # multipoles() expects xi(s, mu) (a density per unit mu), so undo the
        # cell width first; the result is then the same (2 ell + 1) sum that
        # cucount's pole axis returns.
        return multipoles(cells / grid['dmu'], args.ells,
                          grid['nmu']).T                       # (ns, nell)

    pos_j, w_j = jnp.asarray(pos), jnp.asarray(w)
    cot_iso_j, cot_pole_j = jnp.asarray(cot_iso), jnp.asarray(cot_pole)
    out = {'label': 'pyfcfc (CPU, SIMD)', 'backend': 'FCFC C + SIMD, OpenMP'}

    for kind, counts_fn, cot in (('iso', iso_counts, cot_iso_j),
                                 ('pole', pole_counts, cot_pole_j)):
        fwd = jax.jit(counts_fn)
        vg = jax.jit(jax.value_and_grad(
            lambda p, w_: jnp.sum(counts_fn(p, w_) * cot), argnums=(0, 1)))
        best, median, counts = _timed(lambda: fwd(pos_j, w_j), args.repeat)
        vg_best, _, _ = _timed(lambda: vg(pos_j, w_j), args.repeat)
        out[kind] = {'fwd': best, 'fwd_median': median, 'vg': vg_best,
                     'bwd': vg_best - best,
                     'xi': _store(kind, np.asarray(_blocked(counts)),
                                  grid, args)}
    return out


def _blocked(tree):
    _block(tree)
    return tree


def engine_cucount_gpu(args, pos, w, grid, cot_iso, cot_pole):
    """cucount's jax front-end: XLA FFI custom call into the CUDA soft kernels."""
    import jax
    import jax.numpy as jnp
    from cucount.numpy import BinAttrs, MeshAttrs
    from cucount.numpy import Particles as NParticles
    import cucount.jax as cjax

    jax.config.update('jax_enable_x64', True)
    edges = grid['edges']
    ref = NParticles(pos, w)
    requests = {
        'iso': (BinAttrs(s=edges),
                MeshAttrs(ref, ref, boxsize=args.box,
                          battrs=BinAttrs(s=edges), periodic=True)),
        'pole': (BinAttrs(s=edges, pole=(args.ells, 'z')),
                 MeshAttrs(ref, ref, boxsize=args.box,
                           battrs=BinAttrs(s=edges, pole=(args.ells, 'z')),
                           periodic=True)),
    }
    pos_j, w_j = jnp.asarray(pos), jnp.asarray(w)
    cots = {'iso': jnp.asarray(cot_iso), 'pole': jnp.asarray(cot_pole)}
    out = {'label': 'cucount (GPU)', 'backend': f'cuda, accum={args.acc}'}

    for kind, (battrs, mattrs) in requests.items():
        def counts(p, w_, battrs=battrs, mattrs=mattrs):
            return cjax.count2_soft(
                cjax.Particles(p, w_), battrs=battrs,
                soft_sigma=grid['sigma'], soft_cutoff=args.cutoff,
                soft_accum=args.acc, soft_norm=args.soft_norm,
                soft_mu_bins=grid['mu_bins_cucount'],
                soft_mu_sigma=grid['mu_sigma'], mattrs=mattrs,
                backend='cuda')['weight']

        fwd = jax.jit(counts)
        vg = jax.jit(jax.value_and_grad(
            lambda p, w_: jnp.sum(counts(p, w_) * cots[kind]),
            argnums=(0, 1)))
        best, median, result = _timed(lambda: fwd(pos_j, w_j), args.repeat)
        vg_best, _, _ = _timed(lambda: vg(pos_j, w_j), args.repeat)
        out[kind] = {'fwd': best, 'fwd_median': median, 'vg': vg_best,
                     'bwd': vg_best - best,
                     'xi': _store(kind, np.asarray(_blocked(result)),
                                  grid, args)}
    return out


def engine_cucount_gpu_hard(args, pos, w, grid):
    """The hard kernel, as the reference curve the soft results smooth."""
    from cucount.numpy import BinAttrs, MeshAttrs, Particles, count2
    edges = grid['edges']
    ref = Particles(pos, w)
    out = {'label': 'cucount (GPU), hard bins', 'backend': 'cuda, hard'}
    for kind, battrs in (('iso', BinAttrs(s=edges)),
                         ('pole', BinAttrs(s=edges, pole=(args.ells, 'z')))):
        mattrs = MeshAttrs(ref, ref, boxsize=args.box, battrs=battrs,
                           periodic=True)
        best, median, result = _timed(
            lambda: count2(ref, ref, battrs=battrs, mattrs=mattrs,
                           backend='cuda')['weight'], args.repeat)
        out[kind] = {'fwd': best, 'fwd_median': median,
                     'xi': _store(kind, np.asarray(result), grid, args)}
    return out


def engine_cucount_cpu(args, pos, w, grid, cot_iso, cot_pole):
    """cucount's CPU backend (numpy frontend: the jax one is CUDA-only)."""
    from cucount.numpy import BinAttrs, MeshAttrs, Particles
    from cucount.numpy import soft as nsoft

    edges = grid['edges']
    ref = Particles(pos, w)
    p1, p2 = Particles(pos, w), Particles(pos, w)
    tuning = {'nthreads': args.threads}
    common = dict(soft_sigma=grid['sigma'], soft_cutoff=args.cutoff,
                  soft_accum='float64', soft_norm=args.soft_norm,
                  soft_mu_bins=grid['mu_bins_cucount'],
                  soft_mu_sigma=grid['mu_sigma'], backend='cpu', tuning=tuning)
    out = {'label': f'cucount (CPU, {args.threads} threads)',
           'backend': 'cpu scalar kernel, OpenMP'}

    for kind, battrs, cot in (('iso', BinAttrs(s=edges), cot_iso),
                              ('pole', BinAttrs(s=edges, pole=(args.ells, 'z')),
                               cot_pole)):
        mattrs = MeshAttrs(ref, ref, boxsize=args.box, battrs=battrs,
                           periodic=True)
        cot = np.ascontiguousarray(cot, dtype=np.float64)
        best, median, counts = _timed(
            lambda: nsoft.count2_soft(p1, p2, battrs=battrs, mattrs=mattrs,
                                      **common)['weight'], args.repeat)
        vjp_best, _, _ = _timed(
            lambda: nsoft.count2_soft_vjp(p1, p2, battrs=battrs,
                                          cotangent=cot, mattrs=mattrs,
                                          **common), args.repeat)
        out[kind] = {'fwd': best, 'fwd_median': median, 'bwd': vjp_best,
                     'vg': best + vjp_best,
                     'xi': _store(kind, np.asarray(counts), grid, args)}
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _load_state():
    if os.path.exists(STATE):
        with open(STATE) as handle:
            return json.load(handle)
    return {'meta': {}, 'engines': {}}


def gpu_name():
    """Name of the visible GPU, for the report (empty when there is none).

    The kernels are fat binaries: a build that only ships one arch refuses to
    launch on a different card (an sm_86-only build on an A30 reports "named
    symbol not found"), so the report has to say which card was timed.
    """
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=20)
        names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return ', '.join(names)
    except Exception:
        return ''


def run_signature(args, grid):
    """Identifies one configuration; entries recorded under another are dropped.

    The JSON is shared by the two stages of scripts/run_bench_vs_pyfcfc.sh (and
    by successive runs), so a stale engine must not be mixed into a report for
    different bins.
    """
    return {'n': int(args._n), 'box': round(float(args.box), 6),
            'smin': float(args.smin), 'smax': float(args.smax),
            'ds': float(args.ds), 'sigma_bins': float(args.sigma_bins),
            'ells': [int(e) for e in args.ells], 'nmu': int(grid['nmu']),
            'mu_sigma_bins': float(args.mu_sigma_bins),
            'cutoff': float(args.cutoff), 'acc': args.acc,
            'soft_norm': args.soft_norm}


def _log(state, message):
    state.setdefault('log', []).append(
        {'t': time.strftime('%H:%M:%S'), 'msg': message})


def _fmt(value):
    if value is None:
        return '--'
    if value < 1e-3:
        return f'{value * 1e6:.0f} us'
    if value < 1.0:
        return f'{value * 1e3:.0f} ms'
    return f'{value:.2f} s'


def render_report(state):
    meta = state.get('meta', {})
    engines = state.get('engines', {})
    edges = np.asarray(meta.get('edges', []))
    lines = ['# Soft pair counts on an AbacusSummit mock: cucount versus pyfcfc',
             '',
             'Forward and reverse-mode (adjoint) cost of one soft pair-count '
             'request on a real HOD catalogue, and what the kernel does to the '
             'counts. Generated by `examples/bench_vs_pyfcfc.py`; every timing '
             f'is the best of {meta.get("repeat", "?")} calls after one '
             'warm-up.', '',
             '## Data and request', '']
    if meta:
        sigma = meta.get('sigma', 0.0)
        lines += ['| item | value |', '| --- | --- |',
                  f'| catalogue | {meta.get("catalog", "")} |',
                  f'| box | {meta.get("box", "")} Mpc/h, periodic |',
                  f'| separation bins | {meta.get("smin")} to '
                  f'{meta.get("smax")} Mpc/h, {meta.get("ds")} Mpc/h wide |',
                  f'| kernel width | sigma_s = {meta.get("sigma_bins")} x ds '
                  f'= {sigma:.3f} Mpc/h |',
                  f'| kernel normalisation | cucount `soft_norm='
                  f'{meta.get("soft_norm", "?")}` (pinned to match pyfcfc\'s '
                  f'per-pair softmax) |',
                  f'| soft cutoff | {meta.get("cutoff")} |',
                  f'| multipoles | ell = {meta.get("ells")} |',
                  f'| mu quadrature | {meta.get("nmu")} folded cells '
                  f'(cucount: {meta.get("mu_bins_cucount")} unfolded cells of '
                  f'the same width), sigma_mu = '
                  f'{meta.get("mu_sigma_bins")} cell |',
                  f'| cpu threads (cucount) | {meta.get("threads")} |',
                  f'| host | {meta.get("host", "")} |',
                  f'| gpu | {meta.get("gpu") or "--"} |',
                  f'| generated | {meta.get("timestamp", "")} |', '']

    rows_any = False
    for kind, title in (('iso', 'Isotropic soft-s counts'),
                        ('pole', 'Soft-s + soft-mu multipoles')):
        present = [name for name, entry in engines.items()
                   if entry.get(kind, {}).get('fwd') is not None]
        if not present:
            continue
        rows_any = True
        lines += [f'## {title}', '',
                  '| engine | path | forward | adjoint | forward + adjoint |',
                  '| --- | --- | --- | --- | --- |']
        for name in present:
            entry = engines[name]
            block = entry[kind]
            fwd, bwd = block.get('fwd'), block.get('bwd')
            total = None if bwd is None else fwd + bwd
            lines.append(f'| {entry.get("label", name)} | '
                         f'{entry.get("backend", "")} | {_fmt(fwd)} | '
                         f'{_fmt(bwd)} | {_fmt(total)} |')
        best = min(engines[name][kind]['fwd'] for name in present)
        lines += ['', 'Forward pass relative to the fastest engine:']
        for name in present:
            fwd = engines[name][kind]['fwd']
            lines.append(f'* {engines[name].get("label", name)}: '
                         f'{fwd / best:.2f}x')
        lines.append('')

    # Headline ratios: only across engines that compute the same quantity.
    soft_names = [name for name in engines if 'hard' not in name]
    if 'cucount_gpu' in soft_names and engines['cucount_gpu'].get('iso', {}).get('fwd'):
        gpu = engines['cucount_gpu']
        lines += ['## Headline', '']
        for name in soft_names:
            if name == 'cucount_gpu':
                continue
            other = engines[name]
            for kind, what in (('iso', 'isotropic'), ('pole', 'multipoles')):
                a = gpu.get(kind, {}).get('fwd')
                b = other.get(kind, {}).get('fwd')
                if not a or not b:
                    continue
                lines.append(f'* {what}: `{other.get("label", name)}` forward '
                             f'takes {b / a:.1f}x the cucount GPU pass '
                             f'({_fmt(b)} versus {_fmt(a)}).')
        for kind, what in (('iso', 'isotropic'), ('pole', 'multipoles')):
            a = engines.get('cucount_gpu', {}).get(kind, {}).get('fwd')
            b = engines.get('cucount_cpu', {}).get(kind, {}).get('fwd')
            if a and b:
                lines.append(f'* {what}: the GPU pass is {b / a:.1f}x faster '
                             f'than cucount on {meta.get("threads", "?")} CPU '
                             'threads.')
        lines.append('')

    ref_name = next((name for name in ('cucount_gpu', 'cucount_cpu')
                     if engines.get(name, {}).get('pole', {}).get('xi')),
                    None)
    ref = engines.get(ref_name, {}).get('pole', {}).get('xi') if ref_name else None
    if ref:
        lines += ['## Cross-check: do the engines agree?', '',
                  'Largest relative difference of the multipoles with respect '
                  f'to {engines[ref_name].get("label", ref_name)} '
                  '(same bins, same mu grid):', '']
        for name, entry in engines.items():
            # The hard-binned reference is *meant* to differ from a soft
            # histogram, so it is not part of the agreement check.
            if name == ref_name or 'hard' in name:
                continue
            xi = entry.get('pole', {}).get('xi')
            if not xi:
                continue
            dev = max(np.max(np.abs(np.asarray(xi[key]) - np.asarray(ref[key]))
                             / max(np.max(np.abs(ref[key])), 1e-30))
                      for key in xi)
            lines.append(f'* {entry.get("label", name)}: {dev:.2e}')
        lines.append('')

    if os.path.exists(os.path.join(FIGDIR, 'bench_vs_pyfcfc.png')):
        lines += ['![soft counts versus pyfcfc]'
                  '(examples/figures/bench_vs_pyfcfc.png)', '']

    lines += ['## Caveats', '',
              '* pyfcfc is CPU-only, so the GPU rows have no counterpart; the '
              'CPU rows are the honest like-for-like comparison.',
              '* Both codes count autocorrelation pairs twice here (pyfcfc by '
              'convention, cucount because the catalogue is passed as two '
              'copies), so a single ordered-pair normalisation applies to all '
              'curves.  pyfcfc folds mu into [0, 1) while cucount keeps the '
              'sign: even multipoles are unaffected.',
              '* cucount on the CPU goes through the numpy frontend: the jax '
              'frontend is CUDA-only, so a jax-to-jax comparison on the CPU is '
              'not available yet.',
              '* Timings are wall clock around one call: they include the '
              'Python/pybind11 dispatch and the host/device copies, but not jax '
              'tracing or compilation, which are warmed up first.',
              '* The kernels are compared at the same normalisation: pyfcfc '
              "renormalises per pair, and cucount runs `soft_norm='softmax'` "
              "(`--soft-norm`, its default) to match. cucount's own default is "
              "now `soft_norm='gaussian'`, which skips that second pass: keep "
              'the pinned mode when the point is to compare against the oracle, '
              'and pass `--soft-norm gaussian` when the point is to measure the '
              'default.',
              '']
    if not rows_any:
        lines += ['_No engine has finished yet._', '']
    log = state.get('log', [])
    if log:
        lines += ['## Run log', '']
        lines += [f'- `{item["t"]}` {item["msg"]}' for item in log[-40:]]
        lines += ['']
    return '\n'.join(lines)


def write_outputs(state):
    os.makedirs(FIGDIR, exist_ok=True)
    with open(STATE, 'w') as handle:
        json.dump(state, handle, indent=1)
    with open(REPORT, 'w') as handle:
        handle.write(render_report(state))
    try:
        make_figure(state)
    except Exception as exc:                      # plotting must not kill a run
        print(f'  (figure not updated: {type(exc).__name__}: {exc})')
        sys.stdout.flush()


def make_figure(state):
    engines = state.get('engines', {})
    if not engines:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    meta = state['meta']
    edges = np.asarray(meta['edges'])
    centres = 0.5 * (edges[1:] + edges[:-1])
    colours = {'pyfcfc': 'C0', 'gpu': 'C3', 'cpu': 'C2', 'hard': '0.55'}

    def colour(name):
        if 'pyfcfc' in name:
            return colours['pyfcfc']
        if 'hard' in name:
            return colours['hard']
        return colours['gpu'] if 'gpu' in name else colours['cpu']

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    for col, (kind, ell, label) in enumerate((('iso', 0, r'$\xi_0(s)$'),
                                              ('pole', 2, r'$\xi_2(s)$'))):
        ax = axes[0][col]
        for name, entry in engines.items():
            xi = entry.get(kind, {}).get('xi')
            if not xi or str(ell) not in xi:
                continue
            label_ = f'{entry.get("label", name).split(" (")[0]}'
            if 'gpu' in name:
                label_ += ' GPU'
            elif 'cpu' in name:
                label_ += ' CPU'
            style = dict(lw=1.0, ls='--') if 'hard' in name else dict(lw=1.6)
            ax.plot(centres, xi[str(ell)], color=colour(name), label=label_,
                    alpha=0.9, **style)
        ax.axhline(0, color='0.85', lw=0.6)
        ax.set_xlabel(r'$s\ [{\rm Mpc}/h]$')
        ax.set_ylabel(label)
        ax.set_title(f'{label}, $\\sigma_s$ = '
                     f'{meta.get("sigma_bins")}$\\Delta s$')
        if col == 0:
            ax.legend(fontsize=7, frameon=False)

    ax = axes[1][0]
    names = [name for name in engines if 'hard' not in name]
    width = 0.8 / max(len(names), 1)
    kinds = [kind for kind in ('iso', 'pole')
             if any(engines[name].get(kind) for name in names)]
    for i, name in enumerate(names):
        entry = engines[name]
        x = np.arange(len(kinds)) + (i - (len(names) - 1) / 2) * width
        fwd = [entry[kind]['fwd'] if entry.get(kind) else 0.0 for kind in kinds]
        bwd = [(entry[kind].get('bwd') or 0.0) if entry.get(kind) else 0.0
               for kind in kinds]
        ax.bar(x, fwd, width, color=colour(name), alpha=0.95,
               label=entry.get('label', name))
        ax.bar(x, bwd, width, bottom=fwd, color=colour(name), alpha=0.45)
        for xk, f, b in zip(x, fwd, bwd):
            ax.text(xk, f + b, _fmt(f + b), ha='center', va='bottom',
                    fontsize=6, rotation=90)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(kinds)))
    ax.set_xticklabels(['isotropic', 'multipoles'][:len(kinds)])
    ax.set_ylabel('s (solid forward, pale adjoint)')
    ax.set_title('Cost of one pass')
    ax.legend(fontsize=7, frameon=False, loc='upper left')

    ax = axes[1][1]
    hard = engines.get('hard', {}).get('iso', {}).get('xi')
    if hard:
        hard0 = np.asarray(hard['0'])
        for name, entry in engines.items():
            if 'hard' in name:
                continue
            xi = entry.get('iso', {}).get('xi')
            if not xi:
                continue
            ax.plot(centres, np.asarray(xi['0']) / hard0 - 1.0, lw=1.5,
                    color=colour(name),
                    label=entry.get('label', name).split(' (')[0])
        ax.axhline(0, color='0.85', lw=0.6)
    ax.set_xlabel(r'$s\ [{\rm Mpc}/h]$')
    ax.set_ylabel(r'$\xi_0^{\rm soft}/\xi_0^{\rm hard} - 1$')
    ax.set_title('What the kernel does to the counts')
    if hard:
        ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'bench_vs_pyfcfc.png'), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--engines', default='pyfcfc,cucount_cpu,cucount_gpu',
                   help='comma-separated subset of pyfcfc, cucount_cpu, '
                        'cucount_gpu (the GPU stage needs JAX on a GPU)')
    p.add_argument('--path', default=None,
                   help='catalogue instead of the AbacusSummit path '
                        '(.fits with x/y/z columns, or .npy/.npz)')
    p.add_argument('--tracer', default='LRG')
    p.add_argument('--z', type=float, default=0.5)
    p.add_argument('--version', type=int, default=2)
    p.add_argument('--phase', type=int, default=0)
    p.add_argument('--root', default=os.environ.get(
        'WEBJAX_COSMO3D_ROOT', os.path.expanduser('~/cosmo3d')))
    p.add_argument('--nmax', type=int, default=0,
                   help='random subsample size (0 = the whole catalogue)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--box', type=float, default=2000.0)
    p.add_argument('--smin', type=float, default=1.0)
    p.add_argument('--smax', type=float, default=40.0)
    p.add_argument('--ds', type=float, default=1.0)
    p.add_argument('--sigma-bins', type=float, default=1.0,
                   help='soft-s kernel width in units of the bin width')
    p.add_argument('--soft-norm', default='softmax',
                   choices=['softmax', 'gaussian'],
                   help="cucount separation-axis normalisation; keep the "
                        "default 'softmax' to compare against pyfcfc, which "
                        "renormalises the kernel per pair")
    p.add_argument('--ells', type=int, nargs='+', default=[0, 2, 4])
    p.add_argument('--nmu', type=int, default=200,
                   help='mu cells for the multipoles (same in both codes)')
    p.add_argument('--mu-sigma-bins', type=float, default=0.5,
                   help='mu kernel width in units of a folded mu cell')
    p.add_argument('--cutoff', type=float, default=1e-13)
    p.add_argument('--acc', default='float32',
                   choices=['float32', 'float64'],
                   help='cucount per-block accumulator precision on the GPU')
    p.add_argument('--threads', type=int, default=0,
                   help='CPU threads for cucount (0 = affinity mask)')
    p.add_argument('--repeat', type=int, default=3)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    engines = [e.strip() for e in args.engines.split(',') if e.strip()]
    if args.threads == 0:
        args.threads = len(os.sched_getaffinity(0))

    grid = make_grid(args)
    path = args.path or abacus_path(args.tracer, args.z, args.version,
                                    args.phase, args.root)
    print(f'loading {path}')
    sys.stdout.flush()
    pos, w, box, description = load_catalog(args)
    args.box = box
    args._pair_norm = pair_norm(w)
    print(f'catalogue: {description}')
    print(f'gpu: {gpu_name() or "none"}')
    sys.stdout.flush()

    state = _load_state()
    state['meta'] = {
        'catalog': description, 'n': int(len(pos)), 'box': box,
        'smin': args.smin, 'smax': args.smax, 'ds': args.ds,
        'sigma_bins': args.sigma_bins, 'sigma': grid['sigma'],
        'cutoff': args.cutoff, 'ells': list(args.ells), 'nmu': grid['nmu'],
        'mu_sigma_bins': args.mu_sigma_bins, 'mu_sigma': grid['mu_sigma'],
        'mu_bins_cucount': grid['mu_bins_cucount'],
        'soft_norm': args.soft_norm,
        'edges': grid['edges'].tolist(), 'repeat': args.repeat,
        'host': os.uname().nodename, 'gpu': gpu_name(),
        'threads': args.threads,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'acc': args.acc,
    }
    state.setdefault('engines', {})
    args._n = len(pos)
    sig = run_signature(args, grid)
    state['sig'] = sig
    for name in list(state['engines']):
        if state['engines'][name].get('sig') != sig:
            print(f'  (dropping stale {name} entry: different configuration)')
            del state['engines'][name]
    _log(state, f'run started: {description}')
    write_outputs(state)

    rng = np.random.default_rng(args.seed + 1)
    nbins = grid['edges'].size - 1
    cot_iso = rng.normal(size=nbins)
    cot_pole = rng.normal(size=(nbins, len(args.ells)))

    for name in engines:
        start = time.time()
        print(f'[{name}] running ...')
        sys.stdout.flush()
        if name == 'pyfcfc':
            entry = engine_pyfcfc(args, pos, w, grid, cot_iso, cot_pole)
        elif name == 'cucount_gpu':
            entry = engine_cucount_gpu(args, pos, w, grid, cot_iso, cot_pole)
            state['engines']['hard'] = engine_cucount_gpu_hard(
                args, pos, w, grid)
        elif name == 'cucount_cpu':
            entry = engine_cucount_cpu(args, pos, w, grid, cot_iso, cot_pole)
        elif name == 'hard':
            entry = engine_cucount_gpu_hard(args, pos, w, grid)
        else:
            raise SystemExit(f'unknown engine {name!r}')
        entry['sig'] = sig
        state['engines'][name] = entry
        write_outputs(state)
        _log(state, f'{name} finished in {time.time() - start:.1f} s')
        write_outputs(state)
        print(f'[{name}] done in {time.time() - start:.1f} s')
        sys.stdout.flush()

    print(f'report: {REPORT}')
    print(f'figure: {os.path.join(FIGDIR, "bench_vs_pyfcfc.png")}')


if __name__ == '__main__':
    main()
