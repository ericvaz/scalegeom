"""Open datasets bundled with scalegeom: a test bank for scale-conditioned analysis.

Every dataset is in the public domain and is rebuilt from its source by
scripts/build_datasets.py. See DATASETS.md for provenance.

    nc_tracts            North Carolina census tracts, ACS 2023 5-year estimates
                         (income, population, broadband, education), with
                         tract -> county nesting and polygon contiguity.
    etopo_blue_ridge     NOAA ETOPO1 elevation, 160 x 160 cells at 1 arc-minute,
                         from the Blue Ridge to the Piedmont.
    simulated            The two benchmark fields of Vaz (2026), Section 9.8.
"""
from __future__ import annotations

import csv
import gzip
import os

import numpy as np
from scipy import sparse

from ..representation import from_points, grid_family, nested_family, raster_family
from ..simulate import gaussian_random_field, mixed_scale_field
from ..table import Table

DATA = os.path.join(os.path.dirname(__file__), "data")
NC_VARIABLES = ("median_income", "pop", "pct_broadband", "pct_bachelors")


def available() -> Table:
    """List the bundled datasets."""
    return Table([
        {"name": "nc_tracts", "kind": "polygons (centroids + contiguity)", "units": 2660,
         "source": "US Census Bureau, ACS 2023 5-year; cartographic boundaries 2023", "licence": "public domain"},
        {"name": "etopo_blue_ridge", "kind": "raster 160 x 160", "units": 25600,
         "source": "NOAA ETOPO1 via CoastWatch ERDDAP (etopo180)", "licence": "public domain"},
        {"name": "simulated", "kind": "raster 160 x 160", "units": 25600,
         "source": "Vaz (2026) Scenarios A and B, regenerated from seeds", "licence": "MIT"},
    ], title="Bundled open datasets")


# ---------------------------------------------------------------- North Carolina tracts

def _nc_table():
    with gzip.open(os.path.join(DATA, "nc_tracts.csv.gz"), "rt") as fh:
        rows = list(csv.DictReader(fh))
    out = {"GEOID": np.array([r["GEOID"] for r in rows]), "county": np.array([r["county"] for r in rows])}
    for c in ("x", "y", "area_km2") + NC_VARIABLES:
        out[c] = np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rows])
    return out


def _nc_contiguity(n):
    z = np.load(os.path.join(DATA, "nc_tracts_contiguity.npz"))
    out = {}
    for kind in ("queen", "rook"):
        p = z[kind]
        i = np.r_[p[:, 0], p[:, 1]]
        j = np.r_[p[:, 1], p[:, 0]]
        out[kind] = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return out


def nc_tracts(variable: str = "median_income", weight: str | None = "pop", contiguity: bool = True):
    """Base representation of North Carolina census tracts.

    Coordinates are tract centroids in NAD83 / North Carolina (EPSG:32119),
    metres. Tracts with a missing value or zero population are dropped; the
    kept rows are described by rep.attributes (a dict of arrays, including
    GEOID and county)."""
    if variable not in NC_VARIABLES:
        raise ValueError(f"variable must be one of {NC_VARIABLES}")
    t = _nc_table()
    keep = np.isfinite(t[variable])
    if weight:
        keep &= np.isfinite(t[weight]) & (t[weight] > 0)
    idx = np.flatnonzero(keep)
    adj = None
    if contiguity:
        full = _nc_contiguity(len(t["GEOID"]))
        adj = {k: A[idx][:, idx].tocsr() for k, A in full.items()}
    rep = from_points(np.column_stack([t["x"][idx], t["y"][idx]]), t[variable][idx],
                      t[weight][idx] if weight else None, adj, label="tracts")
    rep.attributes = {k: v[idx] for k, v in t.items()}
    return rep


def nc_tracts_family(variable: str = "median_income", weight: str | None = "pop"):
    """Nested family: tracts -> counties (100 counties)."""
    base = nc_tracts(variable, weight)
    return nested_family(base, [base.attributes["county"]], names=["counties"])


def nc_tracts_grid_family(variable: str = "median_income", weight: str | None = "pop",
                          cell_sizes_km=(5, 10, 20, 40), zoning="half"):
    """Tracts aggregated to square grids (cell sizes in km), with optional half-cell shifts."""
    base = nc_tracts(variable, weight)
    return grid_family(base, [1000.0 * c for c in cell_sizes_km], zoning=zoning)


def nc_tracts_geometry():
    """Simplified tract polygons (WGS84) as a GeoDataFrame; needs geopandas."""
    import io

    import geopandas as gpd
    import pandas as pd
    with gzip.open(os.path.join(DATA, "nc_tracts.geojson.gz"), "rb") as fh:
        g = gpd.read_file(io.BytesIO(fh.read()))
    return g.merge(pd.DataFrame(_nc_table()), on="GEOID", how="left")


# ---------------------------------------------------------------- elevation

def etopo_blue_ridge():
    """Elevation (m) on a 160 x 160 grid of 1 arc-minute cells.

    Returns (array, meta). Cells are geographic: about 1.85 km north-south and
    1.51 km east-west at 36 N, so distances are in degrees and slightly
    anisotropic. Use for scale trajectories, not for exact distances."""
    z = np.load(os.path.join(DATA, "etopo_blue_ridge.npz"))
    a = z["elevation"].astype(float)
    meta = {"cellsize": float(z["cellsize_deg"]), "west": float(z["west"]), "north": float(z["north"]),
            "units": "degrees", "source": "NOAA ETOPO1 (etopo180, CoastWatch ERDDAP)"}
    meta["origin"] = (meta["west"], meta["north"] - a.shape[0] * meta["cellsize"])
    return a, meta


def etopo_family(scales=(1, 2, 4, 5, 8, 10, 16), zoning="half"):
    a, m = etopo_blue_ridge()
    return raster_family(a, scales, m["cellsize"], m["origin"], zoning)


# ---------------------------------------------------------------- simulated

def simulated(scenario: str = "A", seed: int = 7, shape=(160, 160)):
    """Scenario A: one correlation length (12 cells). Scenario B: lengths 3 and 30."""
    if scenario.upper() == "A":
        return gaussian_random_field(shape, 12, seed=seed)
    if scenario.upper() == "B":
        return mixed_scale_field(shape, (3, 30), seed=seed)
    raise ValueError("scenario must be 'A' or 'B'")
