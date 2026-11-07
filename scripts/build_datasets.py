"""Rebuild the open datasets bundled with scalegeom from their public sources.

    python scripts/build_datasets.py                       # needs CENSUS_API_KEY
    python scripts/build_datasets.py --acs-csv tracts_37.csv --boundaries cb_2023_37_tract_500k.zip

Needs: requests (or urllib), pandas, geopandas, shapely. Writes compact files to
src/scalegeom/datasets/data/. All sources are works of the United States
government and are in the public domain.

1. North Carolina census tracts
   - Attributes: American Community Survey 2023 5-year estimates, variables
     B19013_001E (median household income), B01003_001E (total population),
     S2801_C02_014E (share of households with a broadband subscription) and
     S1501_C02_015E (share of adults 25+ with a bachelor's degree or higher).
     Fetched from the Census Data API with a free key (set CENSUS_API_KEY;
     https://api.census.gov/data/key_signup.html), or read from a CSV with
     GEOID and those four columns (--acs-csv), for example tables exported
     from data.census.gov.
   - Boundaries: 2023 cartographic boundary file cb_2023_37_tract_500k.
   - Projected to NAD83 / North Carolina (EPSG:32119), metres. Queen and rook
     contiguity are computed from the polygons.
2. ETOPO elevation, Blue Ridge to Piedmont
   - NOAA ETOPO1 (1 arc-minute) through the CoastWatch ERDDAP server
     (dataset etopo180), latitude 34.70-37.35 N, longitude 83.70-81.05 W,
     160 x 160 cells.
"""
import argparse
import gzip
import io
import json
import os
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "src", "scalegeom", "datasets", "data")
CACHE = os.path.join(HERE, "..", ".data_cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
UA = {"User-Agent": "scalegeom-dataset-builder (https://github.com/ericvaz/scalegeom)"}
CENSUS_API = "https://api.census.gov/data/2023/acs/acs5"
ACS_VARS = {"B19013_001E": "median_income", "B01003_001E": "pop",
            "S2801_C02_014E": "pct_broadband", "S1501_C02_015E": "pct_bachelors"}


def get(url, name):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        for i in range(5):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as fh:
                    fh.write(r.read())
                break
            except Exception as exc:  # pragma: no cover
                print(f"  retry {i + 1}: {exc}")
                time.sleep(10 * (i + 1))
        else:
            raise RuntimeError(f"could not download {url}")
    return path


def acs_from_api(key):
    frames = []
    for var, name in ACS_VARS.items():
        endpoint = CENSUS_API + ("/subject" if var.startswith("S") else "")
        path = get(f"{endpoint}?get={var}&for=tract:*&in=state:37&key={key}", f"acs2023_37_{var}.json")
        rows = json.load(open(path))
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df["GEOID"] = df["state"] + df["county"] + df["tract"]
        frames.append(df[["GEOID", var]].rename(columns={var: name}).set_index("GEOID"))
    return pd.concat(frames, axis=1).reset_index()


def acs_from_csv(path):
    df = pd.read_csv(path, dtype={"GEOID": str})
    df = df.rename(columns={k: v for k, v in ACS_VARS.items() if k in df.columns})
    missing = [v for v in ACS_VARS.values() if v not in df.columns]
    if missing:
        raise SystemExit(f"{path} lacks columns {missing}")
    df["GEOID"] = df["GEOID"].str.zfill(11)
    return df[["GEOID"] + list(ACS_VARS.values())]


def build_nc_tracts(acs_csv=None, boundaries=None):
    import geopandas as gpd
    from scalegeom.geo import polygon_contiguity
    if acs_csv:
        df = acs_from_csv(acs_csv)
    elif os.environ.get("CENSUS_API_KEY"):
        df = acs_from_api(os.environ["CENSUS_API_KEY"])
    else:
        raise SystemExit("Set CENSUS_API_KEY or pass --acs-csv")
    for c in ACS_VARS.values():
        df[c] = pd.to_numeric(df[c], errors="coerce").where(lambda x: x > -100)
    z = boundaries or get("https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_37_tract_500k.zip",
                          "cb_2023_37_tract_500k.zip")
    g = gpd.read_file(f"zip://{z}").to_crs(32119)
    g = g.merge(df, on="GEOID", how="left")
    g = g.sort_values("GEOID").reset_index(drop=True)
    g["county"] = g["GEOID"].str[:5]
    c = g.geometry.centroid
    g["x"], g["y"] = c.x.round(1), c.y.round(1)
    g["area_km2"] = (g.geometry.area / 1e6).round(4)
    keep = ["GEOID", "county", "x", "y", "area_km2", "pop", "median_income", "pct_broadband", "pct_bachelors"]
    with gzip.open(os.path.join(OUT, "nc_tracts.csv.gz"), "wt", newline="") as fh:
        g[keep].to_csv(fh, index=False)
    adj = polygon_contiguity(g.geometry.values)
    pairs = {}
    for kind, A in adj.items():
        U = A.tocoo()
        m = U.row < U.col
        pairs[kind] = np.column_stack([U.row[m], U.col[m]]).astype(np.int32)
    np.savez_compressed(os.path.join(OUT, "nc_tracts_contiguity.npz"), **pairs)
    # simplified polygons for maps (about 100 m tolerance)
    gs = g[["GEOID", "geometry"]].copy()
    gs["geometry"] = gs.geometry.simplify(100, preserve_topology=True)
    gs.to_crs(4326).to_file(os.path.join(OUT, "nc_tracts.geojson.gz".replace(".gz", "")), driver="GeoJSON")
    raw = open(os.path.join(OUT, "nc_tracts.geojson"), "rb").read()
    with gzip.open(os.path.join(OUT, "nc_tracts.geojson.gz"), "wb") as fh:
        fh.write(raw)
    os.remove(os.path.join(OUT, "nc_tracts.geojson"))
    print(f"nc_tracts: {len(g)} tracts, {g['median_income'].notna().sum()} with income, "
          f"{len(pairs['queen'])} queen pairs")


def build_etopo():
    lat0, lat1, lon0, lon1 = 34.70, 37.35, -83.70, -81.05
    url = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.csv?"
           f"altitude%5B({lat0}):1:({lat1})%5D%5B({lon0}):1:({lon1})%5D")
    path = get(url, "etopo_blue_ridge.csv")
    d = pd.read_csv(path, skiprows=[1])
    lats = np.sort(d["latitude"].unique())[::-1]
    lons = np.sort(d["longitude"].unique())
    grid = d.pivot(index="latitude", columns="longitude", values="altitude").loc[lats, lons].values
    grid = grid[:160, :160].astype(np.float32)
    np.savez_compressed(os.path.join(OUT, "etopo_blue_ridge.npz"), elevation=grid,
                        north=np.float64(lats[0] + 1 / 120), west=np.float64(lons[0] - 1 / 120),
                        cellsize_deg=np.float64(1 / 60))
    print(f"etopo: {grid.shape}, elevation {grid.min():.0f} to {grid.max():.0f} m")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acs-csv", help="CSV with GEOID and the four ACS variables")
    ap.add_argument("--boundaries", help="local copy of cb_2023_37_tract_500k.zip")
    a = ap.parse_args()
    build_etopo()
    build_nc_tracts(a.acs_csv, a.boundaries)
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}: {os.path.getsize(os.path.join(OUT, f)) / 1024:.0f} KB")
