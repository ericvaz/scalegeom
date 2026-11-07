"""Induced metrics and the neighbourhood structures built from them.

Each function returns a binary sparse adjacency A_s (n x n, zero diagonal).
Spatial weights are obtained by row standardization. A metric is named by a
short specification string so that analyses can be reported and replicated:

    "knn:8"        k nearest neighbours under Euclidean centroid distance
    "band:0.05"    Euclidean distance band, radius in normalized units
    "queen"        queen contiguity (shared vertex)
    "rook"         rook contiguity (shared edge)
    "hops:2"       network metric: units within h hops on the queen graph
    "hops:2:rook"  same, on the rook graph
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def knn(coords: np.ndarray, k: int) -> sparse.csr_matrix:
    n = len(coords)
    k = int(min(k, n - 1))
    if k < 1:
        return sparse.csr_matrix((n, n))
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    # guard against duplicate points returning self
    keep = rows != cols
    return sparse.csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n))


def distance_band(coords: np.ndarray, radius: float) -> sparse.csr_matrix:
    n = len(coords)
    pairs = cKDTree(coords).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return sparse.csr_matrix((n, n))
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))


def hops(adjacency: sparse.spmatrix, h: int) -> sparse.csr_matrix:
    """Units connected by a path of at most h edges in the adjacency graph."""
    A = sparse.csr_matrix(adjacency, dtype=float)
    A.data[:] = 1.0
    n = A.shape[0]
    reach = A.copy()
    frontier = A.copy()
    for _ in range(int(h) - 1):
        frontier = (frontier @ A).tocsr()
        frontier.data[:] = 1.0
        reach = reach + frontier
        reach.data[:] = 1.0
    reach = reach.tolil()
    reach.setdiag(0)
    reach = reach.tocsr()
    reach.eliminate_zeros()
    return reach


def from_distance_matrix(D: np.ndarray, k: int | None = None, radius: float | None = None) -> sparse.csr_matrix:
    """Neighbours under an arbitrary induced distance (travel time, cost,
    network distance, learned embedding). Give k or radius."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    if k is not None:
        k = int(min(k, n - 1))
        idx = np.argpartition(Dm, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(n), k)
        return sparse.csr_matrix((np.ones(n * k), (rows, idx.ravel())), shape=(n, n))
    if radius is not None:
        i, j = np.nonzero(Dm <= radius)
        return sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    raise ValueError("give k or radius")


def row_standardize(A: sparse.spmatrix) -> sparse.csr_matrix:
    A = sparse.csr_matrix(A, dtype=float)
    rs = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(rs)
    inv[rs > 0] = 1.0 / rs[rs > 0]
    return sparse.diags(inv) @ A


def parse_spec(spec: str) -> tuple:
    parts = str(spec).strip().lower().split(":")
    return parts[0], parts[1:]


def induced_adjacency(rep, spec: str, family=None) -> sparse.csr_matrix:
    """Adjacency A_s for representation `rep` under the metric `spec`.

    Euclidean metrics are computed on coordinates normalized by the family
    extent when a family is given, so that band radii are comparable across
    scales."""
    kind, args = parse_spec(spec)
    xy = family.normalized(rep) if family is not None else rep.coords
    if kind == "knn":
        return knn(xy, int(args[0]) if args else 8)
    if kind == "band":
        return distance_band(xy, float(args[0]))
    if kind in ("queen", "rook"):
        if kind not in rep.adjacency:
            raise ValueError(f"{rep.label}: no {kind} contiguity available for this "
                             "representation (supply polygon adjacency or use a lattice)")
        return rep.adjacency[kind]
    if kind == "hops":
        h = int(args[0]) if args else 2
        graph = args[1] if len(args) > 1 else "queen"
        if graph not in rep.adjacency:
            raise ValueError(f"{rep.label}: no {graph} contiguity for the hop metric")
        return hops(rep.adjacency[graph], h)
    if kind == "matrix":
        D = getattr(rep, "distance_matrix", None)
        if D is None:
            raise ValueError("attach rep.distance_matrix to use the 'matrix' metric")
        return from_distance_matrix(D, k=int(args[0]) if args else 6)
    raise ValueError(f"unknown metric specification: {spec!r}")


def describe(spec: str) -> str:
    kind, args = parse_spec(spec)
    return {
        "knn": lambda: f"Euclidean k-NN (k={args[0] if args else 8})",
        "band": lambda: f"Euclidean distance band (r={args[0]})",
        "queen": lambda: "Queen contiguity",
        "rook": lambda: "Rook contiguity",
        "hops": lambda: f"Network (<= {args[0] if args else 2} hops)",
        "matrix": lambda: f"Supplied distance, k-NN (k={args[0] if args else 6})",
    }.get(kind, lambda: spec)()
