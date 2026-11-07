"""Scale-indexed representations S(theta) = Phi_theta(X).

A representation is a metric-measure space with a field attached: unit
centroids (the support of the induced metric), unit values, and a measure
mu over units. Every representation in a family also records which unit
each base (finest) unit was mapped to, so that inter-scale maps
A_{s -> s'} and change-of-support simulations can be computed exactly.

theta = (s, zeta): s is the scale index and zeta the placement of the
aggregation lattice (the zoning coordinate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse


@dataclass
class Representation:
    """One member S(theta) of a family of representations."""

    values: np.ndarray
    coords: np.ndarray
    mass: np.ndarray
    s: float
    zeta: tuple = (0.0, 0.0)
    label: str = ""
    members: np.ndarray | None = None
    adjacency: dict = field(default_factory=dict)

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.coords = np.asarray(self.coords, dtype=float).reshape(-1, 2)
        self.mass = np.asarray(self.mass, dtype=float)
        if self.mass.sum() > 0:
            self.mass = self.mass / self.mass.sum()
        if not (len(self.values) == len(self.coords) == len(self.mass)):
            raise ValueError("values, coords and mass must have the same length")
        if not self.label:
            self.label = _default_label(self.s, self.zeta)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def theta(self) -> tuple:
        return (self.s, self.zeta)

    def with_values(self, values: np.ndarray) -> "Representation":
        """Same geometry and measure, different field values."""
        return Representation(values, self.coords, self.mass, self.s, self.zeta,
                              self.label, self.members, self.adjacency)


@dataclass
class Family:
    """A family {S(theta)} along a scale path, with optional zoning variants.

    path    : representations ordered from fine to coarse; path[0] is the base.
    zoning  : maps a position on the path to a list of representations at the
              same s but a different lattice placement zeta.
    origin, extent : distances are normalized as (x - origin) / extent so that
              every representation in the family shares one unit of length.
    """

    path: list
    zoning: dict = field(default_factory=dict)
    origin: np.ndarray = None
    extent: float = None
    kind: str = "custom"
    lattice: dict = field(default_factory=dict)

    def __post_init__(self):
        base = self.path[0]
        if self.origin is None:
            self.origin = base.coords.min(axis=0)
        if self.extent is None:
            span = base.coords.max(axis=0) - base.coords.min(axis=0)
            self.extent = float(max(span.max(), 1e-12))
        self.origin = np.asarray(self.origin, dtype=float)

    @property
    def base(self) -> Representation:
        return self.path[0]

    @property
    def scales(self) -> list:
        return [r.s for r in self.path]

    def normalized(self, rep: Representation) -> np.ndarray:
        return (rep.coords - self.origin) / self.extent

    def reference_moments(self) -> tuple:
        """Mean and SD of the base field: the common standardization used
        for semivariograms and FGW feature costs across the family."""
        v = self.base.values
        sd = float(np.std(v))
        return float(np.mean(v)), (sd if sd > 0 else 1.0)

    def aggregate_values(self, base_values: np.ndarray, rep: Representation) -> np.ndarray:
        """Apply the family's aggregation operator for `rep` to new base values.

        This is how simulated fields are pushed through exactly the same
        Phi_theta as the observed field."""
        return _weighted_group_mean(base_values, self.base.mass, rep.members, rep.n)

    def all_representations(self) -> list:
        out = list(self.path)
        for variants in self.zoning.values():
            out.extend(variants)
        return out


# ---------------------------------------------------------------- helpers

def _default_label(s, zeta) -> str:
    s_txt = f"{s:g}" if isinstance(s, (int, float, np.floating, np.integer)) else str(s)
    if zeta is None or (np.ndim(zeta) and not np.any(np.asarray(zeta, dtype=object) != 0)):
        return f"s={s_txt}"
    if isinstance(zeta, str):
        return f"s={s_txt} [{zeta}]"
    return f"s={s_txt} zeta=({zeta[0]:g},{zeta[1]:g})"


def _weighted_group_mean(values, weights, labels, n_groups):
    ok = labels >= 0
    w = weights[ok]
    lab = labels[ok]
    wsum = np.bincount(lab, weights=w, minlength=n_groups)
    vsum = np.bincount(lab, weights=w * values[ok], minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return vsum / wsum


def indicator_matrix(labels: np.ndarray, n_groups: int) -> sparse.csr_matrix:
    """Sparse base-to-unit membership matrix P (base x units)."""
    ok = np.flatnonzero(labels >= 0)
    return sparse.csr_matrix((np.ones(len(ok)), (ok, labels[ok])),
                             shape=(len(labels), n_groups))


def coarsen_adjacency(adjacency: dict, labels: np.ndarray, n_groups: int) -> dict:
    """Two units are neighbours if any of their base members are neighbours."""
    if not adjacency:
        return {}
    P = indicator_matrix(labels, n_groups)
    out = {}
    for kind, A in adjacency.items():
        B = (P.T @ A @ P).tocsr()
        B.setdiag(0)
        B.eliminate_zeros()
        B.data[:] = 1.0
        out[kind] = B
    return out


def aggregate_by_labels(base: Representation, labels: Sequence[int], s: float,
                        zeta=(0.0, 0.0), label: str = "", coords=None,
                        keep_order: bool = False) -> Representation:
    """Aggregate a base representation by integer (or hashable) labels.

    Values are mu-weighted means of member values, centroids are mu-weighted
    centroids (unless `coords` gives one point per label), and the measure of
    each unit is the summed base measure. Labels < 0 or None drop a base unit.
    """
    labels = np.asarray(labels)
    if len(labels) != base.n:
        raise ValueError("one label per base unit is required")
    if labels.dtype.kind in "iu":
        valid = labels >= 0
        keys = labels[valid]
    elif labels.dtype.kind == "f":
        valid = np.isfinite(labels) & (labels >= 0)
        keys = labels[valid]
    else:
        valid = np.array([l is not None and str(l) not in ("", "nan", "None", "NULL")
                          for l in labels], dtype=bool)
        keys = labels[valid].astype(str)
    if not valid.any():
        raise ValueError("no valid labels")
    compact = np.full(len(labels), -1, dtype=np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    compact[valid] = inv
    n_groups = len(uniq)
    vals = _weighted_group_mean(base.values, base.mass, compact, n_groups)
    if coords is None:
        cx = _weighted_group_mean(base.coords[:, 0], base.mass, compact, n_groups)
        cy = _weighted_group_mean(base.coords[:, 1], base.mass, compact, n_groups)
        xy = np.column_stack([cx, cy])
    else:
        xy = np.asarray(coords, dtype=float)
    mass = np.bincount(compact[compact >= 0], weights=base.mass[compact >= 0],
                       minlength=n_groups)
    adj = coarsen_adjacency(base.adjacency, compact, n_groups)
    rep = Representation(vals, xy, mass, s, zeta, label, compact, adj)
    rep.unit_keys = uniq
    return rep


# ---------------------------------------------------------------- lattices

def lattice_adjacency(rows: np.ndarray, cols: np.ndarray, kind: str = "queen") -> sparse.csr_matrix:
    """Adjacency among occupied cells of a regular lattice."""
    n = len(rows)
    key = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if kind == "queen":
        offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    ii, jj = [], []
    nr = int(rows.max()) + 2 if n else 0
    nc = int(cols.max()) + 2 if n else 0
    # vectorized via a dense index grid
    grid = -np.ones((nr + 1, nc + 1), dtype=np.int64)
    grid[rows, cols] = np.arange(n)
    for dr, dc in offs:
        r2, c2 = rows + dr, cols + dc
        ok = (r2 >= 0) & (c2 >= 0) & (r2 <= nr) & (c2 <= nc)
        j = np.full(n, -1)
        j[ok] = grid[r2[ok], c2[ok]]
        hit = j >= 0
        ii.append(np.flatnonzero(hit))
        jj.append(j[hit])
    ii = np.concatenate(ii) if ii else np.array([], int)
    jj = np.concatenate(jj) if jj else np.array([], int)
    return sparse.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))


def from_raster(array: np.ndarray, cellsize: float = 1.0, origin=(0.0, 0.0),
                label: str = "s=1") -> tuple:
    """Base representation of a raster band (NaN = no data).

    origin is the lower-left corner of the raster in map units; row 0 is the
    top row. Returns (representation, rows, cols)."""
    a = np.asarray(array, dtype=float)
    nr, nc = a.shape
    rows, cols = np.nonzero(np.isfinite(a))
    x = origin[0] + (cols + 0.5) * cellsize
    y = origin[1] + (nr - rows - 0.5) * cellsize
    rep = Representation(a[rows, cols], np.column_stack([x, y]), np.ones(len(rows)),
                         1, (0, 0), label, np.arange(len(rows)),
                         {"queen": lattice_adjacency(rows, cols, "queen"),
                          "rook": lattice_adjacency(rows, cols, "rook")})
    return rep, rows, cols


def raster_family(array: np.ndarray, scales: Iterable[int], cellsize: float = 1.0,
                  origin=(0.0, 0.0), zoning="half", min_coverage: float = 1.0) -> Family:
    """Block-mean aggregation family of a raster, as in Vaz (2026) Section 9.

    scales  : block side lengths in cells, e.g. (1, 2, 4, 5, 8, 10, 16).
    zoning  : "half" adds a half-block shifted lattice at every s >= 2; a list
              of (row, col) offsets in cells adds those; None adds nothing.
    min_coverage : minimum share of valid cells for a block to be kept.
    """
    a = np.asarray(array, dtype=float)
    nr, nc = a.shape
    base, rows, cols = from_raster(a, cellsize, origin)
    scales = sorted(set(int(s) for s in scales))
    if scales[0] != 1:
        scales = [1] + scales

    def block_rep(s, off):
        zr, zc = off
        nbr, nbc = (nr - zr) // s, (nc - zc) // s
        br, bc = (rows - zr) // s, (cols - zc) // s
        inside = (rows >= zr) & (cols >= zc) & (br < nbr) & (bc < nbc)
        lab = np.where(inside, br * nbc + bc, -1)
        counts = np.bincount(lab[lab >= 0], minlength=nbr * nbc)
        keep_block = counts >= min_coverage * s * s
        lab = np.where((lab >= 0) & keep_block[np.maximum(lab, 0)], lab, -1)
        used = np.unique(lab[lab >= 0])
        remap = -np.ones(nbr * nbc, dtype=np.int64)
        remap[used] = np.arange(len(used))
        compact = np.where(lab >= 0, remap[np.maximum(lab, 0)], -1)
        ub_r, ub_c = used // nbc, used % nbc
        # geometric block centres
        cx = origin[0] + (zc + (ub_c + 0.5) * s) * cellsize
        cy = origin[1] + (nr - zr - (ub_r + 0.5) * s) * cellsize
        vals = _weighted_group_mean(base.values, base.mass, compact, len(used))
        mass = np.bincount(compact[compact >= 0], minlength=len(used)).astype(float)
        zeta = (zr * cellsize, zc * cellsize)
        rep = Representation(vals, np.column_stack([cx, cy]), mass, s, zeta,
                             _default_label(s, (zr, zc)), compact,
                             {"queen": lattice_adjacency(ub_r, ub_c, "queen"),
                              "rook": lattice_adjacency(ub_r, ub_c, "rook")})
        rep.lattice_rc = (ub_r, ub_c, nbr, nbc)
        return rep

    path = [base] + [block_rep(s, (0, 0)) for s in scales if s != 1]
    zon = {}
    for i, s in enumerate(scales):
        offsets = []
        if zoning == "half" and s >= 2:
            offsets = [(s // 2, s // 2)]
        elif zoning not in (None, "half"):
            offsets = [tuple(o) for o in zoning if s >= 2]
        if offsets:
            zon[i] = [block_rep(s, o) for o in offsets]
    fam = Family(path, zon, origin=np.asarray(origin, float),
                 extent=float(max(nr, nc) * cellsize), kind="raster",
                 lattice={"shape": (nr, nc), "rows": rows, "cols": cols,
                          "cellsize": cellsize, "origin": tuple(origin)})
    return fam


# ---------------------------------------------------------------- vector data

def from_points(coords, values, mass=None, adjacency: dict | None = None,
                s: float | None = None, label: str = "base") -> Representation:
    """Base representation from unit centroids (points or polygon centroids).

    mass : optional weights (for example population); values are aggregated
           as mass-weighted means and the measure mu is proportional to mass.
    adjacency : optional {"queen": csr, "rook": csr} contiguity of the units.
    s    : scale index of the base; defaults to the mean spacing
           sqrt(bbox area / n)."""
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    mass = np.ones(len(values)) if mass is None else np.asarray(mass, dtype=float)
    ok = np.isfinite(values) & np.all(np.isfinite(coords), axis=1) & np.isfinite(mass) & (mass > 0)
    if not ok.all():
        raise ValueError(f"{(~ok).sum()} units have missing values, coordinates or "
                         "non-positive weights; drop them first")
    if s is None:
        span = coords.max(axis=0) - coords.min(axis=0)
        s = float(np.sqrt(max(span[0] * span[1], 1e-12) / len(values)))
    return Representation(values, coords, mass, s, (0.0, 0.0), label,
                          np.arange(len(values)), adjacency or {})


def grid_family(base: Representation, cell_sizes: Iterable[float], zoning="half",
                origin=None, centroids: str = "weighted") -> Family:
    """Aggregate a base representation onto square grids of increasing size.

    cell_sizes : grid cell sides in map units.
    zoning     : "half" adds a grid shifted by half a cell in x and y.
    centroids  : "weighted" uses mu-weighted member centroids; "cell" uses
                 the geometric centre of each grid cell.
    """
    xy = base.coords
    o = xy.min(axis=0) if origin is None else np.asarray(origin, float)
    span = xy.max(axis=0) - o

    def grid_rep(cs, shift):
        ox, oy = o[0] - shift, o[1] - shift
        gc = np.floor((xy[:, 0] - ox) / cs).astype(np.int64)
        gr = np.floor((xy[:, 1] - oy) / cs).astype(np.int64)
        ncol = int(gc.max()) + 1
        key = gr * ncol + gc
        uniq, inv = np.unique(key, return_inverse=True)
        cell_xy = None
        if centroids == "cell":
            ur, uc = uniq // ncol, uniq % ncol
            cell_xy = np.column_stack([ox + (uc + 0.5) * cs, oy + (ur + 0.5) * cs])
        rep = aggregate_by_labels(base, inv, cs, (shift, shift),
                                  _default_label(cs, (shift, shift)), coords=cell_xy)
        ur, uc = uniq // ncol, uniq % ncol
        grid_adj = {"queen": lattice_adjacency(ur, uc, "queen"),
                    "rook": lattice_adjacency(ur, uc, "rook")}
        # keep areal contiguity if the base had it, otherwise grid contiguity
        if not rep.adjacency:
            rep.adjacency = grid_adj
        rep.grid_adjacency = grid_adj
        return rep

    sizes = sorted(float(c) for c in cell_sizes)
    path = [base] + [grid_rep(cs, 0.0) for cs in sizes]
    zon = {}
    if zoning == "half":
        for i, cs in enumerate(sizes, start=1):
            zon[i] = [grid_rep(cs, cs / 2.0)]
    ext = float(max(span.max(), 1e-12))
    return Family(path, zon, origin=o, extent=ext, kind="grid")


def nested_family(base: Representation, levels: Sequence[Sequence], names: Sequence[str] | None = None,
                  scale_index: Sequence[float] | None = None) -> Family:
    """Family over nested administrative supports (e.g. DA -> CT -> CSD).

    levels : one label array per coarser level, each giving the parent unit
             of every base unit. Levels need not be strictly nested; each is
             aggregated directly from the base.
    scale_index : optional s for each level; defaults to sqrt(bbox area / n).
    """
    names = list(names) if names is not None else [f"level{i + 1}" for i in range(len(levels))]
    span = base.coords.max(axis=0) - base.coords.min(axis=0)
    area = max(span[0] * span[1], 1e-12)
    path = [base]
    for i, lab in enumerate(levels):
        rep = aggregate_by_labels(base, lab, 0.0, (0, 0), names[i])
        rep.s = float(scale_index[i + 1]) if scale_index is not None else float(np.sqrt(area / rep.n))
        path.append(rep)
    if scale_index is not None:
        base.s = float(scale_index[0])
    return Family(path, {}, kind="nested")


def voronoi_zoning(base: Representation, n_zones: int, seed: int = 0, label: str = "") -> Representation:
    """Random size-matched zoning: n_zones zones grown from random seed units
    by nearest-seed assignment (a Voronoi approximation to contiguous
    aggregation). Used for the zoning null of nested families."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    n_zones = int(min(max(2, n_zones), base.n))
    seeds = rng.choice(base.n, n_zones, replace=False)
    _, lab = cKDTree(base.coords[seeds]).query(base.coords)
    return aggregate_by_labels(base, lab, float("nan"), "random",
                               label or f"random zoning ({n_zones}) #{seed}")
