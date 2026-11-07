"""Structural diagnostics evaluated on each representation.

Moran's I and local Moran (LISA) follow Moran (1950) and Anselin (1995) with
permutation inference matching the PySAL/esda conventions; semivariograms and
the correlation-threshold range proxy follow Vaz (2026), Section 9.1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .metrics import row_standardize

DEFAULT_LAG_EDGES = np.linspace(0.0, 0.5, 16)  # 15 bins over normalized lags 0-0.5


# ---------------------------------------------------------------- Moran's I

@dataclass
class MoranResult:
    I: float
    EI: float
    p_sim: float
    n: int
    permutations: int


def morans_i(y: np.ndarray, A: sparse.spmatrix, permutations: int = 99, seed=None) -> MoranResult:
    """Global Moran's I with row-standardized weights from adjacency A."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    W = row_standardize(A)
    s0 = W.sum()
    z = y - y.mean()
    den = z @ z
    if n < 3 or s0 == 0 or den == 0:
        return MoranResult(np.nan, -1.0 / max(n - 1, 1), np.nan, n, permutations)
    scale = n / s0
    I = float(scale * (z @ (W @ z)) / den)
    p = np.nan
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty(permutations)
        batch = max(1, min(permutations, int(2e7 // max(n, 1))))
        done = 0
        while done < permutations:
            b = min(batch, permutations - done)
            Z = rng.permuted(np.broadcast_to(z, (b, n)), axis=1)
            sims[done:done + b] = scale * np.einsum("ij,ij->i", Z, (W @ Z.T).T) / den
            done += b
        larger = np.sum(sims >= I)
        if permutations - larger < larger:
            larger = permutations - larger
        p = (larger + 1.0) / (permutations + 1.0)
    return MoranResult(I, -1.0 / (n - 1), float(p), n, int(permutations or 0))


def morans_i_fast(z_or_y: np.ndarray, W: sparse.spmatrix) -> float:
    """Moran's I without inference, for Monte Carlo loops (W row-standardized)."""
    z = z_or_y - z_or_y.mean()
    den = z @ z
    s0 = W.sum()
    if den == 0 or s0 == 0:
        return np.nan
    return float(len(z) / s0 * (z @ (W @ z)) / den)


# ---------------------------------------------------------------- local Moran

CLUSTER_NAMES = {0: "not significant", 1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}


@dataclass
class LocalMoranResult:
    Is: np.ndarray
    p_sim: np.ndarray
    quadrant: np.ndarray
    cluster: np.ndarray  # 0 = ns, 1 HH, 2 LH, 3 LL, 4 HL at the chosen alpha

    def counts(self) -> dict:
        return {CLUSTER_NAMES[k]: int(np.sum(self.cluster == k)) for k in range(5)}

    @property
    def n_significant(self) -> int:
        return int(np.sum(self.cluster > 0))


def local_moran(y: np.ndarray, A: sparse.spmatrix, permutations: int = 999,
                alpha: float = 0.05, seed=None) -> LocalMoranResult:
    """Local Moran's I with conditional permutation inference.

    As in esda, one set of random draws is shared by all units and the
    focal unit is excluded from its own reference distribution."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    W = row_standardize(A).tocsr()
    z = y - y.mean()
    m2 = (z @ z) / n
    lag = W @ z
    Is = z * lag / m2
    quad = np.zeros(n, dtype=int)
    quad[(z > 0) & (lag > 0)] = 1
    quad[(z < 0) & (lag > 0)] = 2
    quad[(z < 0) & (lag < 0)] = 3
    quad[(z > 0) & (lag < 0)] = 4
    p = np.full(n, np.nan)
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        card = np.diff(W.indptr)
        kmax = int(card.max()) if n else 0
        if kmax > 0:
            draws = np.stack([rng.choice(n - 1, kmax, replace=False) for _ in range(permutations)])
            for i in range(n):
                k = card[i]
                if k == 0:
                    continue
                w = W.data[W.indptr[i]:W.indptr[i + 1]]
                j = draws[:, :k].copy()
                j[j >= i] += 1
                lag_sim = z[j] @ w
                Isim = z[i] * lag_sim / m2
                larger = np.sum(Isim >= Is[i])
                if permutations - larger < larger:
                    larger = permutations - larger
                p[i] = (larger + 1.0) / (permutations + 1.0)
    cluster = np.where(p <= alpha, quad, 0)
    return LocalMoranResult(Is, p, quad, cluster)


# ---------------------------------------------------------------- pairs, variograms

def sample_pairs(n: int, max_pairs: int = 200_000, seed=None) -> tuple:
    """All pairs when there are few, otherwise a random sample of pairs."""
    total = n * (n - 1) // 2
    if total <= max_pairs:
        i, j = np.triu_indices(n, k=1)
        return i, j
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=int(max_pairs * 1.05))
    j = rng.integers(0, n, size=len(i))
    keep = i != j
    return i[keep][:max_pairs], j[keep][:max_pairs]


def _bin(d, edges):
    # right-closed bins (a, b]; the first bin also includes 0
    b = np.searchsorted(edges, d, side="left") - 1
    b[d == edges[0]] = 0
    b[(b < 0) | (b >= len(edges) - 1)] = -1
    return b


def semivariogram(values: np.ndarray, xy: np.ndarray, edges=DEFAULT_LAG_EDGES,
                  max_pairs: int = 200_000, seed=None, pairs=None) -> dict:
    """Empirical semivariogram on a fixed lag grid.

    values should already be standardized with the family's reference
    moments so that levels are comparable across scales."""
    edges = np.asarray(edges, dtype=float)
    i, j = pairs if pairs is not None else sample_pairs(len(values), max_pairs, seed)
    d = np.hypot(*(xy[i] - xy[j]).T)
    b = _bin(d, edges)
    ok = b >= 0
    sq = 0.5 * (values[i] - values[j]) ** 2
    nb = len(edges) - 1
    cnt = np.bincount(b[ok], minlength=nb).astype(float)
    tot = np.bincount(b[ok], weights=sq[ok], minlength=nb)
    dsum = np.bincount(b[ok], weights=d[ok], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(cnt > 0, tot / cnt, np.nan)
        mean_lag = np.where(cnt > 0, dsum / cnt, np.nan)
    return {"centers": 0.5 * (edges[:-1] + edges[1:]), "gamma": gamma, "counts": cnt, "edges": edges,
            "mean_lag": mean_lag}


def correlation_profile(values: np.ndarray, xy: np.ndarray, edges=DEFAULT_LAG_EDGES,
                        max_pairs: int = 200_000, seed=None, pairs=None) -> dict:
    """Pairwise correlation rho(h) of the field standardized within this scale."""
    edges = np.asarray(edges, dtype=float)
    sd = values.std()
    z = (values - values.mean()) / (sd if sd > 0 else 1.0)
    i, j = pairs if pairs is not None else sample_pairs(len(values), max_pairs, seed)
    d = np.hypot(*(xy[i] - xy[j]).T)
    b = _bin(d, edges)
    ok = b >= 0
    nb = len(edges) - 1
    cnt = np.bincount(b[ok], minlength=nb).astype(float)
    tot = np.bincount(b[ok], weights=(z[i] * z[j])[ok], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(cnt > 0, tot / cnt, np.nan)
    return {"centers": 0.5 * (edges[:-1] + edges[1:]), "rho": rho, "counts": cnt, "edges": edges}


def range_proxy(centers: np.ndarray, rho: np.ndarray, tau: float = 0.1) -> float:
    """Smallest normalized lag at which the correlation profile falls below
    tau, linearly interpolated between bin centres."""
    ok = np.isfinite(rho)
    c, r = np.asarray(centers)[ok], np.asarray(rho)[ok]
    below = np.flatnonzero(r < tau)
    if len(below) == 0:
        return np.nan
    k = below[0]
    if k == 0:
        return float(c[0])
    x0, x1, y0, y1 = c[k - 1], c[k], r[k - 1], r[k]
    if y0 == y1:
        return float(x1)
    return float(x0 + (y0 - tau) * (x1 - x0) / (y0 - y1))
