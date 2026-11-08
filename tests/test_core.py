import numpy as np
import pytest

import scalegeom as sg
from scalegeom import datasets as D
from scalegeom.metrics import hops, induced_adjacency, knn
from scalegeom.space import bilipschitz, interscale_map


@pytest.fixture(scope="module")
def field():
    return sg.gaussian_random_field((80, 80), 6, seed=3)


@pytest.fixture(scope="module")
def family(field):
    return sg.raster_family(field, [1, 2, 4, 8])


def test_block_means_are_exact(field, family):
    rep = family.path[2]  # s = 4
    blocks = field.reshape(20, 4, 20, 4).mean(axis=(1, 3))
    r, c, _, _ = rep.lattice_rc
    assert np.allclose(rep.values, blocks[r, c])
    assert rep.n == 400


def test_zoning_shift_has_fewer_full_blocks(family):
    v = family.zoning[1][0]  # s = 2 shifted by one cell
    assert v.n == 39 * 39


def test_members_and_aggregate_values_round_trip(family):
    for rep in family.path:
        assert np.allclose(family.aggregate_values(family.base.values, rep), rep.values)


def test_lattice_contiguity_counts(family):
    q = family.base.adjacency["queen"]
    r = family.base.adjacency["rook"]
    assert q.sum() == 2 * (2 * 79 * 80) + 2 * (2 * 79 * 79)
    assert r.sum() == 2 * (2 * 79 * 80)


def test_hops_metric():
    rows, cols = np.divmod(np.arange(25), 5)
    from scalegeom.representation import lattice_adjacency
    A = lattice_adjacency(rows, cols, "rook")
    H = hops(A, 2)
    assert H[12].sum() == 12  # centre of a 5x5 rook lattice: 4 + 8 within two steps
    assert H.diagonal().sum() == 0


@pytest.fixture(scope="module")
def irregular():
    # irregular points avoid k-NN distance ties, which libraries break differently
    rng = np.random.default_rng(11)
    xy = rng.uniform(0, 1, (600, 2))
    vals = np.sin(6 * xy[:, 0]) + np.cos(5 * xy[:, 1]) + rng.normal(0, 0.5, 600)
    return sg.from_points(xy, vals)


def test_morans_i_matches_esda(irregular):
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    rep = irregular
    A = knn(rep.coords, 8)
    ours = sg.morans_i(rep.values, A, permutations=0)
    w = libpysal.weights.KNN.from_array(rep.coords, k=8)
    w.transform = "R"
    ref = esda.Moran(rep.values, w, permutations=0)
    assert ours.I == pytest.approx(ref.I, abs=1e-10)


def test_local_moran_quadrants_match_esda(irregular):
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    rep = irregular
    A = knn(rep.coords, 6)
    ours = sg.local_moran(rep.values, A, permutations=499, seed=1)
    w = libpysal.weights.KNN.from_array(rep.coords, k=6)
    w.transform = "R"
    ref = esda.Moran_Local(rep.values, w, permutations=499, seed=1)
    assert np.array_equal(ours.quadrant, ref.q)
    # permutation p-values agree in distribution, not draw for draw
    assert np.corrcoef(ours.p_sim, ref.p_sim)[0, 1] > 0.9
    assert abs(np.mean(ours.p_sim <= 0.05) - np.mean(ref.p_sim <= 0.05)) < 0.05


def test_moran_permutation_p_value_is_small_for_smooth_field(family):
    m = sg.morans_i(family.path[1].values, induced_adjacency(family.path[1], "knn:8", family), 99, seed=0)
    assert m.I > 0.8 and m.p_sim == pytest.approx(0.01)


def test_range_proxy_interpolates():
    centers = np.array([0.1, 0.2, 0.3])
    rho = np.array([0.5, 0.0, -0.1])
    assert sg.range_proxy(centers, rho, 0.1) == pytest.approx(0.18)


def test_covariance_fit_recovers_length():
    fits = [sg.fit_covariance(sg.raster_family(sg.gaussian_random_field((120, 120), 9, seed=s), [1]), "gaussian")
            for s in range(4)]
    assert np.mean([m.length for m in fits]) == pytest.approx(9 / 120, rel=0.1)
    assert max(m.nugget for m in fits) < 0.02


def test_regularization_prediction_tracks_monte_carlo(family):
    m = sg.fit_covariance(family, "gaussian")
    res = sg.information_loss_null(family, "knn:8", m, n_sim=40, seed=1)
    pred = np.array(res.table["var_predicted"])
    mc = np.array(res.table["var_null_mean"])
    assert np.max(np.abs(pred - mc)) < 0.02


def test_null_accepts_single_scale_and_rejects_mixed_scale():
    a = sg.raster_family(sg.gaussian_random_field((96, 96), 8, seed=5), [1, 2, 4, 8, 16], zoning=None)
    b = sg.raster_family(sg.mixed_scale_field((96, 96), (2, 20), seed=5), [1, 2, 4, 8, 16], zoning=None)
    ra = sg.information_loss_null(a, "knn:8", "gaussian", n_sim=60, seed=0)
    rb = sg.information_loss_null(b, "knn:8", "gaussian", n_sim=60, seed=0)
    assert ra.exits <= 1
    assert rb.exits >= 3


def test_interscale_map_and_bilipschitz(family):
    A = interscale_map(family.path[1], family.path[2])
    assert (A >= 0).all() and A.max() == family.path[2].n - 1
    r = bilipschitz(family.path[2], family.path[3], family, method="pairs", quantiles=(0.01, 0.99))
    assert 0 < r["c"] <= 1 <= r["C"]
    iso = bilipschitz(family.base, family.base, family)
    assert iso["distortion"] == pytest.approx(1.0)


def test_fgw_zero_for_identical_and_grows_with_scale(family):
    pytest.importorskip("ot")
    small = sg.raster_family(sg.gaussian_random_field((32, 32), 4, seed=1), [1, 2, 4, 8])
    d0 = sg.fgw_distance(small.base, small.base, small, max_nodes=2000)
    d1 = sg.fgw_distance(small.base, small.path[1], small, max_nodes=2000)
    d3 = sg.fgw_distance(small.base, small.path[3], small, max_nodes=2000)
    assert d0 == pytest.approx(0.0, abs=1e-8)
    assert d1 < d3


def test_grid_and_nested_families_from_points():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (1500, 2))
    vals = np.sin(xy[:, 0] / 15) + rng.normal(0, 0.1, 1500)
    pop = rng.integers(50, 500, 1500)
    base = sg.from_points(xy, vals, pop)
    g = sg.grid_family(base, [10, 25])
    assert g.path[1].n <= 100 and len(g.zoning[1]) == 1
    # mass-weighted means
    lab = g.path[1].members
    k = lab[0]
    m = lab == k
    assert g.path[1].values[k] == pytest.approx(np.average(vals[m], weights=pop[m]))
    nest = sg.nested_family(base, [np.floor(xy[:, 0] / 20).astype(int)], names=["strips"])
    assert nest.path[1].n == 5


def test_protocol_runs_and_writes_report(tmp_path, family):
    p = sg.ScaleProtocol(family, "knn:8", alt_metrics=["queen", "hops:2"], permutations=19)
    rep = p.run(n_sim=10, fgw=False)
    assert {"representations", "trajectory", "null", "invariance"} <= set(rep.tables)
    out = rep.save(str(tmp_path / "r"))
    assert (tmp_path / "r" / "report.html").exists()
    assert (tmp_path / "r" / "trajectory.csv").read_text().startswith("label,")


def test_metric_sensitivity(family):
    rep = family.path[2]
    r = sg.metric_sensitivity(rep, ["queen", "knn:6", "hops:2"], family, permutations=99, figures=False)
    t = r["metrics"]
    assert len(t) == 3
    I = dict(zip(t["spec"], t["moran_I"]))
    assert I["queen"] > 0.6 and I["knn:6"] > 0.6
    assert I["hops:2"] < I["queen"]  # wider neighbourhoods dilute dependence


def test_north_carolina_table3():
    """Global Moran's I for 1974 SIDS rates under three induced metrics (Vaz 2026, Table 3)."""
    gpd = pytest.importorskip("geopandas")
    examples = pytest.importorskip("libpysal.examples")
    from scalegeom.geo import from_geodataframe
    rep = from_geodataframe(gpd.read_file(examples.get_path("sids2.shp")), "SIDR74")
    r = sg.metric_sensitivity(rep, ["knn:6", "queen", "hops:2"], permutations=99, figures=False)
    I = dict(zip(r["metrics"]["spec"], r["metrics"]["moran_I"]))
    assert I["knn:6"] == pytest.approx(0.192, abs=5e-4)
    assert I["queen"] == pytest.approx(0.231, abs=5e-4)
    assert I["hops:2"] == pytest.approx(0.152, abs=5e-4)


def test_citation():
    assert "10.1111/gean.70061" in sg.citation()
    assert "@article{vaz2026scale" in sg.citation("bibtex")
    assert "gean.70061" in sg.Report("x").to_html()


# ------------------------------------------------------------------ added in 0.2.0

def _irregular_family(seed=0, n=400):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (n, 2))
    vals = rng.normal(size=n)
    mass = rng.uniform(1, 10, n)
    base = sg.from_points(xy, vals, mass)
    return base, sg.grid_family(base, [10.0, 25.0], zoning=None)


def test_weighted_variance_is_between_share():
    """Weighted variance of mass-weighted zone means over the base variance is
    the between-zone share: it equals 1 - within-zone share exactly."""
    base, fam = _irregular_family()
    w = base.mass / base.mass.sum()
    for rep in fam.path[1:]:
        lab = rep.members
        means = rep.values[lab]
        m = np.sum(w * base.values)
        total = np.sum(w * (base.values - m) ** 2)
        within = np.sum(w * (base.values - means) ** 2)
        ratio = sg.weighted_variance(rep.values, rep.mass) / sg.weighted_variance(base.values, base.mass)
        assert ratio == pytest.approx(1 - within / total, abs=1e-12)
        assert 0 <= ratio <= 1


def test_aggregation_of_zone_constant_and_zone_centred_fields():
    """A field constant within zones keeps all its variance; a field whose
    zone means are all equal keeps none."""
    base, fam = _irregular_family(1)
    rep = fam.path[-1]
    lab = rep.members
    const = np.arange(rep.n, dtype=float)[lab]
    v = fam.aggregate_values(const, rep)
    assert np.allclose(v, np.arange(rep.n))
    assert sg.weighted_variance(v, rep.mass) / sg.weighted_variance(const, base.mass) == pytest.approx(1.0)
    w = base.mass
    centred = base.values - (np.bincount(lab, w * base.values) / np.bincount(lab, w))[lab]
    v2 = fam.aggregate_values(centred, rep)
    assert np.allclose(v2, 0, atol=1e-12)


def test_contiguous_zoning_is_contiguous_and_size_matched():
    from scipy.sparse.csgraph import connected_components
    from scalegeom.representation import _contiguity_graph
    fam = D.nc_tracts_family("median_income")
    base, cty = fam.path
    sizes = np.bincount(cty.members)
    G = _contiguity_graph(base)
    z = sg.contiguous_zoning(base, sizes, seed=5, graph=G)
    assert z.n == cty.n
    for g in range(z.n):
        idx = np.flatnonzero(z.members == g)
        assert connected_components(G[idx][:, idx], directed=False)[0] == 1
    got = np.sort(np.bincount(z.members))[::-1]
    want = np.sort(sizes)[::-1]
    assert got.sum() == want.sum()
    assert np.corrcoef(got, want)[0, 1] > 0.9


def test_fdr_matches_benjamini_hochberg():
    from scalegeom.diagnostics import fdr_significant
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    # BH at q = 0.05 with n = 10: thresholds 0.005, 0.010, 0.015, ...; the two smallest pass
    assert fdr_significant(p, 0.05).tolist() == [True, True] + [False] * 8
    assert fdr_significant(np.full(5, 0.5), 0.05).sum() == 0


def test_simultaneous_test_is_reported():
    a = sg.raster_family(sg.gaussian_random_field((48, 48), 6, seed=3), [1, 2, 4], zoning=None)
    r = sg.information_loss_null(a, "knn:8", "gaussian", n_sim=40, seed=0)
    s = r.simultaneous
    assert s["critical"] > 0 and 0 < s["p"] <= 1
    assert s["rejects"] == (s["statistic"] > s["critical"])
