"""Known-answer test bank (see tests/benchmarks.json)."""
import json
import os

import numpy as np
import pytest

import scalegeom as sg
from scalegeom import datasets as D

BANK = json.load(open(os.path.join(os.path.dirname(__file__), "benchmarks.json")))
PUB, DATA, REG = BANK["published"], BANK["data"], BANK["regression"]
SCALES = [1, 2, 4, 5, 8, 10, 16]


# ------------------------------------------------------------------ published results

def test_nc_sids_table3():
    gpd = pytest.importorskip("geopandas")
    examples = pytest.importorskip("libpysal.examples")
    from scalegeom.geo import from_geodataframe
    rep = from_geodataframe(gpd.read_file(examples.get_path("sids2.shp")), "SIDR74")
    exp = PUB["nc_sids74_moran"]
    for spec in ("knn:6", "queen", "hops:2"):
        I = sg.morans_i(rep.values, sg.induced_adjacency(rep, spec), 0).I
        assert I == pytest.approx(exp[spec], abs=exp["tolerance"]), spec


@pytest.mark.parametrize("scenario", ["A", "B"])
def test_scenarios_against_null(scenario):
    fam = sg.raster_family(D.simulated(scenario), SCALES, zoning=None)
    r = sg.information_loss_null(fam, "knn:8", "auto", n_sim=60, seed=1)
    exp = PUB[f"scenario_{scenario}_exits"]
    if "max" in exp:
        assert r.exits <= exp["max"]
    else:
        assert r.exits >= exp["min"]


# ------------------------------------------------------------------ bundled data

def test_nc_tracts_integrity():
    exp = DATA["nc_tracts"]
    t = D._nc_table()
    assert len(t["GEOID"]) == exp["tracts"]
    assert np.isfinite(t["median_income"]).sum() == exp["with_income"]
    assert np.isfinite(t["pct_broadband"]).sum() == exp["with_broadband"]
    assert len(set(t["county"])) == exp["counties"]
    i = list(t["GEOID"]).index(exp["check_geoid"])
    assert t["median_income"][i] == exp["check_income"] and t["pop"][i] == exp["check_pop"]
    z = np.load(os.path.join(D.DATA, "nc_tracts_contiguity.npz"))
    assert len(z["queen"]) == exp["queen_pairs"]
    assert len(z["rook"]) <= len(z["queen"])


def test_etopo_integrity():
    exp = DATA["etopo_blue_ridge"]
    a, meta = D.etopo_blue_ridge()
    assert list(a.shape) == exp["shape"]
    assert a.min() == exp["min"] and a.max() == exp["max"]
    assert a.mean() == pytest.approx(exp["mean"], abs=exp["tolerance"])


def test_nc_tracts_geometry_matches_table():
    pytest.importorskip("geopandas")
    g = D.nc_tracts_geometry()
    assert len(g) == DATA["nc_tracts"]["tracts"]
    assert g["median_income"].notna().sum() == DATA["nc_tracts"]["with_income"]


# ------------------------------------------------------------------ regression values

@pytest.fixture(scope="module")
def nc_income():
    return D.nc_tracts_family("median_income")


def test_nc_income_trajectory(nc_income):
    b, c = nc_income.path
    for key, rep in (("nc_income_tract_moran_queen", b), ("nc_income_county_moran_queen", c)):
        assert sg.morans_i(rep.values, rep.adjacency["queen"], 0).I == pytest.approx(
            REG[key]["value"], abs=REG[key]["tolerance"])
    exp = REG["nc_income_county_var_retained"]
    assert np.var(c.values) / np.var(b.values) == pytest.approx(exp["value"], abs=exp["tolerance"])


def test_nc_income_zoning_null(nc_income):
    exp = REG["nc_income_zoning_null_z_var"]
    t = sg.zoning_null(nc_income, "queen", exp["n_sim"], seed=exp["seed"])
    assert t.rows[0]["z_var"] < exp["max"]


def test_nc_broadband_moran():
    rep = D.nc_tracts("pct_broadband")
    exp = REG["nc_broadband_tract_moran_queen"]
    assert sg.morans_i(rep.values, rep.adjacency["queen"], 0).I == pytest.approx(exp["value"], abs=exp["tolerance"])


def test_nc_income_grid_null():
    exp = REG["nc_income_grid_null_exits"]
    fam = D.nc_tracts_grid_family(cell_sizes_km=(5, 10, 20, 40), zoning=None)
    r = sg.information_loss_null(fam, "knn:6", "auto", n_sim=exp["n_sim"], seed=0)
    assert r.exits >= exp["min"]


def test_etopo_model_choice_and_null():
    fam = D.etopo_family(zoning=None)
    m = sg.fit_covariance(fam, "auto")
    assert m.model == REG["etopo_selected_model"]["value"]
    r = sg.information_loss_null(fam, "knn:8", m, n_sim=REG["etopo_null_exits"]["n_sim"], seed=0)
    assert r.exits <= REG["etopo_null_exits"]["max"]


def test_datasets_listing():
    t = D.available()
    assert set(t["name"]) == {"nc_tracts", "etopo_blue_ridge", "simulated"}


def test_manifest_checksums():
    import hashlib
    manifest = json.load(open(os.path.join(os.path.dirname(D.__file__), "manifest.json")))
    for d in manifest["datasets"]:
        for fname, digest in d["files"].items():
            data = open(os.path.join(D.DATA, fname), "rb").read()
            assert hashlib.sha256(data).hexdigest() == digest, fname


def test_null_calibration_smoke():
    """With parameter uncertainty propagated, rough stationary fields rarely leave the band
    (examples/null_calibration.py measures the rate on 100 fields per model)."""
    from scalegeom.simulate import CovModel, simulate_grid
    truth = CovModel("exponential", 1.0, 10 / 96)
    rng = np.random.default_rng(7)
    exits = 0
    for k in range(5):
        fam = sg.raster_family(simulate_grid(truth, (96, 96), 1 / 96, rng), [1, 2, 4, 8, 16], zoning=None)
        exits += sg.information_loss_null(fam, "knn:8", "auto", n_sim=30, seed=k).exits > 0
    assert exits <= 1
