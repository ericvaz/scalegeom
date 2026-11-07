"""Helpers for polygon and point data held in geopandas GeoDataFrames."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .representation import from_points, nested_family


def polygon_contiguity(geoms) -> dict:
    """Queen (shared vertex) and rook (shared edge) contiguity among polygons."""
    from shapely import STRtree
    geoms = list(geoms)
    n = len(geoms)
    tree = STRtree(geoms)
    qi, qj, ri, rj = [], [], [], []
    for i, g in enumerate(geoms):
        for j in tree.query(g, predicate="intersects"):
            if j <= i:
                continue
            inter = g.intersection(geoms[j])
            if inter.is_empty:
                continue
            qi += [i, j]; qj += [j, i]
            if inter.length > 0:
                ri += [i, j]; rj += [j, i]
    mk = lambda a, b: sparse.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return {"queen": mk(qi, qj), "rook": mk(ri, rj)}


def from_geodataframe(gdf, column: str, weight: str | None = None, contiguity: bool = True,
                      s: float | None = None, points: str = "centroid"):
    """Base representation from a GeoDataFrame of polygons or points.

    points : "centroid" (geometric centroid, the usual PySAL choice) or
             "representative" (a point guaranteed inside each polygon).
    Rows with a missing value or non-positive weight are dropped (the kept
    rows are in rep.index)."""
    g = gdf[gdf[column].notna()]
    if weight is not None:
        g = g[g[weight].notna() & (g[weight] > 0)]
    geom = g.geometry
    poly = geom.geom_type.isin(["Polygon", "MultiPolygon"]).any()
    if poly:
        pts = geom.representative_point() if points == "representative" else geom.centroid
    else:
        pts = geom
    xy = np.column_stack([pts.x.values, pts.y.values])
    adj = polygon_contiguity(geom.values) if contiguity and geom.geom_type.isin(
        ["Polygon", "MultiPolygon"]).all() else None
    rep = from_points(xy, g[column].values.astype(float),
                      None if weight is None else g[weight].values.astype(float), adj, s)
    rep.index = g.index
    return rep


def nested_from_geodataframe(gdf, column: str, levels, weight: str | None = None, contiguity: bool = True):
    """Nested family from a base polygon layer with parent-code columns,
    e.g. levels=["CTUID", "CSDUID"] on dissemination areas."""
    base = from_geodataframe(gdf, column, weight, contiguity)
    g = gdf.loc[base.index]
    return nested_family(base, [g[lv].values for lv in levels], names=list(levels))
