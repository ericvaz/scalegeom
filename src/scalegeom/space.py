"""The family of representations as a measured space.

Each representation is treated as a metric-measure space (S_theta, d_theta,
mu_theta) carrying field values; the Fused Gromov-Wasserstein distance
(Memoli 2011; Vayer et al. 2018) jointly compares the induced geometries and
the fields. This gives scale velocity along the scale path and zoning
displacement at fixed s. Inter-scale maps are checked against the
bi-Lipschitz condition c d_s(x, y) <= d_s'(A x, A y) <= C d_s(x, y).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .table import Table


def _require_pot():
    try:
        import ot  # noqa: F401
        return ot
    except ImportError as exc:  # pragma: no cover
        raise ImportError("FGW distances need the POT package: pip install pot") from exc


def mm_space(rep, family, max_nodes: int = 400, rng=None):
    """Structure matrix, standardized features and measure for one representation.

    Representations with more than max_nodes units are subsampled with
    probability proportional to mu, so the subsample carries a uniform measure."""
    rng = np.random.default_rng(rng)
    mu, sd = family.reference_moments()
    if rep.n > max_nodes:
        idx = rng.choice(rep.n, max_nodes, replace=False, p=rep.mass)
        p = np.full(max_nodes, 1.0 / max_nodes)
    else:
        idx = np.arange(rep.n)
        p = rep.mass / rep.mass.sum()
    xy = family.normalized(rep)[idx]
    C = cdist(xy, xy)
    f = (rep.values[idx] - mu) / sd
    return C, f, p, xy


def fgw_distance(rep_a, rep_b, family, alpha: float = 0.5, max_nodes: int = 800, seed=0) -> float:
    """FGW distance between two representations of the same family.

    alpha weights the structure (metric) term against the feature term, as in
    POT; alpha = 0.5 is the default used in Vaz (2026). Because all members of
    a family share one coordinate frame, the solver starts from the optimal
    transport plan between unit locations, which avoids most of the local
    optima of the non-convex FGW problem."""
    ot = _require_pot()
    rng = np.random.default_rng(seed)
    Ca, fa, pa, xa = mm_space(rep_a, family, max_nodes, rng)
    Cb, fb, pb, xb = mm_space(rep_b, family, max_nodes, rng)
    M = (fa[:, None] - fb[None, :]) ** 2
    G0 = ot.emd(pa, pb, cdist(xa, xb, "sqeuclidean"))
    return float(ot.gromov.fused_gromov_wasserstein2(
        M, Ca, Cb, pa, pb, loss_fun="square_loss", alpha=alpha, symmetric=True, G0=G0))


def sampling_floor(rep, family, alpha=0.5, max_nodes=800, repeats=2, seed=0) -> float:
    """FGW between independent subsamples of the same representation: the
    distance attributable to subsampling alone. Zero if no subsampling."""
    if rep.n <= max_nodes:
        return 0.0
    vals = [fgw_distance(rep, rep, family, alpha, max_nodes, seed=seed + 1000 + r) for r in range(repeats)]
    return float(np.mean(vals))


def representation_space(family, alpha: float = 0.5, max_nodes: int = 800, seed: int = 0,
                         floor: bool = True) -> dict:
    """Scale velocity along the path and zoning displacement at fixed s.

    When a representation has more than max_nodes units it is subsampled, and
    subsampling alone produces a positive FGW distance. With floor=True each
    such representation's floor F is estimated from independent subsamples
    and the adjusted distance FGW(a, b) - (F_a + F_b) / 2 is reported next to
    the raw value. This is an empirical correction, analogous in form to the
    self-distance terms of debiased Sinkhorn divergences but without their
    guarantees: the adjusted value can be slightly negative and is not a
    metric. Velocity and the cumulative path use the adjusted distance."""
    path = family.path
    floors = {}

    def fl(rep, k):
        if not floor:
            return 0.0
        if rep.label not in floors:
            floors[rep.label] = sampling_floor(rep, family, alpha, max_nodes, seed=seed + 7 * k)
        return floors[rep.label]

    steps = Table(title="Representation space: FGW distance and scale velocity",
                  note=f"alpha = {alpha:g}; representations above {max_nodes} units are subsampled. "
                       "fgw_debiased subtracts half of each side's subsampling floor; velocity = "
                       "fgw_debiased / (s' - s).")
    cum = 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        d = fgw_distance(a, b, family, alpha, max_nodes, seed + i)
        fa, fb = fl(a, i), fl(b, i + 1)
        dd = d - 0.5 * (fa + fb)
        ds = (b.s - a.s) if np.isfinite(b.s - a.s) else np.nan
        cum += max(dd, 0.0)
        steps.append({"step": f"{a.label} -> {b.label}", "s_from": a.s, "s_to": b.s, "fgw_raw": d,
                      "floor_from": fa, "floor_to": fb, "fgw_debiased": dd,
                      "velocity": dd / ds if ds else np.nan, "cumulative_fgw": cum})
    zon = Table(title="Zoning displacement at fixed s (FGW to the shifted lattice)")
    for i, variants in sorted(family.zoning.items()):
        for j, v in enumerate(variants):
            d = fgw_distance(path[i], v, family, alpha, max_nodes, seed + 100 + 10 * i + j)
            dd = d - 0.5 * (fl(path[i], i) + fl(v, 50 + 10 * i + j))
            zon.append({"s": path[i].s, "variant": v.label, "fgw_raw": d, "fgw_debiased": dd})
    out = {"path": steps, "zoning": zon, "alpha": alpha, "max_nodes": max_nodes, "floors": floors}
    if floor:
        out["sampling_floor"] = fl(family.base, 0)
    return out


# ---------------------------------------------------------------- invariance

def interscale_map(fine, coarse) -> np.ndarray:
    """A_{s -> s'}: the coarse unit holding the largest share of each fine
    unit's base members (-1 where undefined)."""
    f, c = fine.members, coarse.members
    ok = (f >= 0) & (c >= 0)
    key = f[ok].astype(np.int64) * (coarse.n + 1) + c[ok]
    uk, cnt = np.unique(key, return_counts=True)
    fu, cu = uk // (coarse.n + 1), uk % (coarse.n + 1)
    order = np.lexsort((-cnt, fu))
    fu, cu = fu[order], cu[order]
    first = np.r_[True, fu[1:] != fu[:-1]]
    A = np.full(fine.n, -1, dtype=np.int64)
    A[fu[first]] = cu[first]
    return A


def bilipschitz(fine, coarse, family=None, n_sample: int = 150, method: str = "centroid",
                quantiles=(0.0, 1.0), n_pairs: int = 20_000, seed: int = 0) -> dict:
    """Empirical distortion constants (c, C) of the inter-scale centroid map on matched pairs.

    Aggregation is many-to-one, so two fine units in the same coarse unit are
    sent to the same point and no positive lower bound c can hold for the whole
    map. The constants are therefore computed on pairs where the map separates
    the units: either coarse units matched to their nearest fine unit, or
    random pairs of fine units lying in distinct coarse units. They are the
    empirical counterpart of the bi-Lipschitz condition of Vaz (2026),
    Section 7, restricted to those pairs.

    method="centroid" (Vaz 2026, Section 9.7): a sample of coarse units is
        matched to the nearest fine-unit centroid and pairwise distances are
        compared; quantiles=(0, 1) gives the min and max ratio.
    method="pairs": random pairs of fine units falling in distinct coarse
        units, with robust quantiles (e.g. 0.01, 0.99) for irregular supports.
    The distortion C / c equals 1 for an isometry."""
    rng = np.random.default_rng(seed)
    xf = fine.coords if family is None else family.normalized(fine)
    xc = coarse.coords if family is None else family.normalized(coarse)
    if method == "centroid":
        m = min(n_sample, coarse.n)
        cu = rng.choice(coarse.n, m, replace=False)
        _, fu = cKDTree(xf).query(xc[cu])
        i, j = np.triu_indices(m, k=1)
        df = np.hypot(*(xf[fu[i]] - xf[fu[j]]).T)
        dc = np.hypot(*(xc[cu[i]] - xc[cu[j]]).T)
    elif method == "pairs":
        A = interscale_map(fine, coarse)
        ok = np.flatnonzero(A >= 0)
        a = rng.choice(ok, n_pairs)
        b = rng.choice(ok, n_pairs)
        keep = A[a] != A[b]
        a, b = a[keep], b[keep]
        df = np.hypot(*(xf[a] - xf[b]).T)
        dc = np.hypot(*(xc[A[a]] - xc[A[b]]).T)
    else:
        raise ValueError("method must be 'centroid' or 'pairs'")
    good = df > 0
    r = dc[good] / df[good]
    if len(r) == 0:
        return {"c": np.nan, "C": np.nan, "distortion": np.nan, "pairs": 0}
    c, C = np.quantile(r, quantiles[0]), np.quantile(r, quantiles[1])
    return {"c": float(c), "C": float(C), "distortion": float(C / c) if c > 0 else np.inf,
            "pairs": int(len(r))}


def invariance_table(family, n_sample=150, method="centroid", quantiles=(0.0, 1.0), seed=0) -> Table:
    t = Table(title="Inter-scale distortion: empirical bi-Lipschitz constants on matched pairs",
              note=f"Pairs: {'coarse units matched to their nearest fine unit' if method == 'centroid' else 'fine units in distinct coarse units'}. "
                   "C/c = 1 for an isometry on these pairs; larger values mean neighbourhood-based "
                   "quantities cannot be presumed transportable across the step. Pairs inside one "
                   "coarse unit collapse to distance zero and are excluded by construction.")
    for i in range(len(family.path) - 1):
        a, b = family.path[i], family.path[i + 1]
        r = bilipschitz(a, b, family, n_sample, method, quantiles, seed=seed + i)
        t.append({"step": f"{a.label} -> {b.label}", "c": r["c"], "C": r["C"],
                  "C/c": r["distortion"], "pairs": r["pairs"]})
    return t
