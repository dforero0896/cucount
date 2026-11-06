"""Real Abacus halo catalogues for the soft-binning examples.

The examples in this directory are more convincing on a clustered catalogue than
on a uniform one: a Gaussian kernel's quadrature error and its edge truncation
both show up as a modification of the measured correlation function, and on a
uniform catalogue there is no correlation function to modify.

Path resolution and the box wrap here mirror
``holi-refactor/y1-holimocks/src/utils.py:load_abacus_reference`` (the loader
used by the differentiable 2PCF pipeline): the catalogue path comes from
``holimocks_config``, the positions are wrapped with
``(position + 1.5 * box_size) % box_size``, and the halo finder's box is
2000 Mpc/h.  We read only the position columns, with ``memmap``, because the
full catalogue is ~7 million rows and the examples only need a subsample.

Nothing in this module is cucount-specific: it is example scaffolding, kept
separate so both the comparison and the benchmark use the same catalogue.
"""

import glob
import os

import numpy as np

#: ``.../cucount_dev``: the examples live in ``cucount_dev/cucount/examples``.
_DEV_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Where the module looks for ``holimocks_config`` when the caller does not
#: point at it.  Override with ``CUC0UNT_HOLI_SRC``.
DEFAULT_HOLI_SRC = os.path.join(
    os.path.dirname(_DEV_ROOT), 'holi-refactor', 'y1-holimocks', 'src')

#: Fallback search roots for the DR2_v1.0 halo catalogues, in priority order.
DEFAULT_ABACUS_ROOTS = (
    os.environ.get('CUC0UNT_ABACUS_ROOT', ''),
    os.path.expanduser('~/cosmo3d/desi/mocks/cai/abacus_HF/DR2_v1.0'),
    '/global/cfs/cdirs/desicollab/mocks/cai/abacus_HF/DR2_v1.0',
)


def _holimocks_config():
    """The ``holimocks_config`` module, or ``None`` when it is not importable."""
    import sys

    source = os.environ.get('CUC0UNT_HOLI_SRC', DEFAULT_HOLI_SRC)
    if os.path.isfile(os.path.join(source, 'holimocks_config.py')):
        if source not in sys.path:
            sys.path.insert(0, source)
    try:
        import holimocks_config
    except Exception:
        return None
    return holimocks_config


def abacus_reference_path(tracer='LRG', redshift=0.5, version=3, phase=0):
    """Path of an Abacus halo catalogue, plus the box side it lives in.

    Tries the ``holimocks_config`` registry first (the same resolution the
    pipeline uses).  When that is unavailable, or the registered file is not
    mirrored on this cluster, searches the DR2_v1.0 tree for the requested
    tracer and falls back to the nearest available redshift.

    Returns ``(path, box_size, redshift, note)``.
    """
    config = _holimocks_config()
    if config is not None:
        try:
            cfg = config.get_config()
            path = cfg.abacus_reference_path(tracer, redshift, version=version,
                                             phase=phase)
            if os.path.isfile(path):
                return path, float(cfg.abacus_box_size(version)), float(redshift), \
                    'holimocks_config'
            box_size = float(cfg.abacus_box_size(version))
        except Exception:
            box_size = 2000.0
    else:
        box_size = 2000.0

    pattern = os.path.join('{root}', 'AbacusSummit_base_c000_ph*', 'Boxes',
                           tracer.upper(), f'*_{tracer.upper()}_*')
    candidates = []
    for root in DEFAULT_ABACUS_ROOTS:
        if not root:
            continue
        candidates.extend(sorted(glob.glob(pattern.format(root=root))))
    if not candidates:
        raise FileNotFoundError(
            'no Abacus catalogue found for tracer %r at z=%g\n'
            'searched:\n  %s\n'
            'pass --catalogue PATH, or set CUC0UNT_ABACUS_ROOT, or run with '
            '--synthetic' % (tracer, redshift,
                             '\n  '.join(root for root in DEFAULT_ABACUS_ROOTS if root)))

    # Nearest available redshift wins; ties keep the sorted order.
    redshifts = np.array([_catalogue_redshift(path) for path in candidates])
    best = int(np.nanargmin(np.abs(redshifts - redshift)))
    path = candidates[best]
    return path, box_size, float(redshifts[best]), 'directory search'


def _catalogue_redshift(path):
    """Redshift parsed from a DR2 filename such as ``..._LRG_0p500_...``."""
    for part in os.path.basename(path).split('_'):
        try:
            return float(part.replace('p', '.'))
        except ValueError:
            continue
    return float('nan')


def read_positions(path, box_size=2000.0, n_particles=None, seed=7):
    """Positions from an Abacus FITS catalogue, wrapped into ``[0, box_size)``.

    Only the ``X``/``Y``/``Z`` columns are touched, through the FITS memmap, so
    a 7-million-row catalogue costs one column copy.  ``n_particles``
    subsamples rows with ``seed``; a subsample keeps the clustering (it is a
    random subset of the same tracers) and is what makes the examples fast
    enough to run interactively.
    """
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        table = None
        for hdu in hdul:
            columns = getattr(hdu, 'columns', None)
            if columns is None:            # the primary HDU has no columns
                continue
            names = [name.strip().upper() for name in (columns.names or [])]
            if all(axis in names for axis in ('X', 'Y', 'Z')):
                table = hdu
                break
        if table is None:
            raise ValueError(f'{path}: no X/Y/Z columns found')
        lookup = {name.strip().upper(): name for name in table.columns.names}
        columns = [np.asarray(table.data[lookup[axis]], dtype=np.float64)
                   for axis in ('X', 'Y', 'Z')]
    positions = np.column_stack(columns)
    del columns

    if n_particles is not None and n_particles < positions.shape[0]:
        rng = np.random.default_rng(seed)
        index = rng.choice(positions.shape[0], size=int(n_particles), replace=False)
        index.sort()
        positions = positions[index]

    # The same wrap the pipeline applies: the halo finder's box is centred on
    # the origin for some versions and on a corner for others.
    positions = np.mod(positions + 1.5 * box_size, box_size)
    return positions


def cut_sub_box(positions, box_size, sub_box, seed=7):
    """Cut a periodic sub-box of side ``sub_box`` out of the catalogue.

    A sub-box keeps the *real* number density (unlike a random subsample of the
    same size, which would thin the sample and inflate the shot noise).  The
    origin is random, so the cut does not align with the halo finder's grid.
    """
    rng = np.random.default_rng(seed)
    origin = rng.uniform(0.0, box_size - sub_box, size=3)
    inside = np.all((positions >= origin) & (positions < origin + sub_box), axis=1)
    return np.mod(positions[inside] - origin, sub_box)


def load_abacus(tracer='LRG', redshift=0.5, n_particles=None, seed=7,
                version=3, path=None, sub_box=None):
    """Load an Abacus halo catalogue, as a sub-box or a random subsample.

    Exactly one of ``n_particles`` and ``sub_box`` is normally used: a
    ``sub_box`` cut keeps the true density (more pairs at small separations,
    smaller volume), a random subsample keeps the true volume (more pairs at
    large separations, more shot noise).  Returns ``(positions, box_size,
    info)``; ``path`` bypasses the path resolution entirely.
    """
    if path is None:
        path, box_size, redshift, how = abacus_reference_path(
            tracer=tracer, redshift=redshift, version=version)
    else:
        box_size, how = 2000.0, '--catalogue'
    info = dict(path=path, tracer=tracer.upper(), box_size=float(box_size),
                redshift=_catalogue_redshift(path), resolved_by=how)

    positions = read_positions(path, box_size=box_size, n_particles=None)
    info['n_parent'] = int(positions.shape[0])
    if sub_box is not None:
        parent_box = box_size
        positions = cut_sub_box(positions, parent_box, float(sub_box), seed=seed)
        box_size = float(sub_box)
        info['sub_box'] = float(sub_box)
        info['n_cut'] = int(positions.shape[0])
    if n_particles is not None and n_particles < positions.shape[0]:
        rng = np.random.default_rng(seed)
        index = rng.choice(positions.shape[0], size=int(n_particles), replace=False)
        index.sort()
        positions = positions[index]
    info['box_size'] = float(box_size)
    info['n_total'] = int(positions.shape[0])
    return positions, float(box_size), info


def synthetic_clustered(box_size=1000.0, n_particles=50000, seed=11,
                        n_haloes=None, scale_radius=6.0):
    """A clustered fallback catalogue: Poisson haloes with Gaussian profiles.

    Only used when no Abacus catalogue is reachable, so that the examples still
    run on a laptop.  ``scale_radius`` sets the halo size in Mpc/h, which puts
    the one-halo term where a real halo catalogue has it.
    """
    rng = np.random.default_rng(seed)
    n_haloes = int(n_haloes if n_haloes is not None else n_particles // 40)
    centres = rng.uniform(0.0, box_size, (n_haloes, 3))
    members = rng.integers(0, n_haloes, n_particles)
    offsets = rng.normal(scale=scale_radius, size=(n_particles, 3))
    positions = np.mod(centres[members] + offsets, box_size)
    return positions
