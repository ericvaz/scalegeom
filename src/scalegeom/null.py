"""Testing change under aggregation against an information-loss null.

The change-of-support null (Gotway and Young 2002; Cressie 2015) asks what a
diagnostic's trajectory along the scale path would be if the only thing
happening were the loss of fine-scale degrees of freedom under a single,
fine-scale correlation structure. A covariance model is fitted at the finest
representation, fields are simulated from it on the base support, pushed
through exactly the same aggregation operators Phi_theta as the data, and the
diagnostic is recomputed. Observed values outside mean +/- width * SD exit
the null (Vaz 2026, Section 9.8).

A second null, `zoning_null`, compares an observed (e.g. administrative)
aggregation with random size-matched zonings of the same base data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from .diagnostics import DEFAULT_LAG_EDGES, morans_i_fast, semivariogram
from .metrics import induced_adjacency, row_standardize
from .representation import voronoi_zoning
from .simulate import CovModel, point_simulator, simulate_grid
from .table import Table


# ---------------------------------------------------------------- model fitting

MODELS = ("gaussian", "exponential", "spherical")


def fit_covariance(family, model: str = "auto", edges=DEFAULT_LAG_EDGES,
                   max_pairs: int = 200_000, seed: int = 0, nugget: bool = True) -> CovModel:
    """Fit a covariance model to the empirical semivariogram of the base
    representation (values standardized by the base moments, normalized lags).

    model="auto" fits the Gaussian, exponential and spherical models and keeps
    the one with the smallest weighted residual sum of squares; the scores of
    all three are stored in `.scores`. The choice matters: a smooth (Gaussian)
    model fitted to a rough field predicts too little information loss under
    aggregation and makes the null reject spuriously."""
    if model == "auto":
        fits = [fit_covariance(family, m, edges, max_pairs, seed, nugget) for m in MODELS]
        best = min(fits, key=lambda m: m.wrss)
        best.scores = {m.model: m.wrss for m in fits}
        return best
    from scipy.spatial import cKDTree
    base = family.base
    mu, sd = family.reference_moments()
    z = (base.values - mu) / sd
    xy = family.normalized(base)
    v = semivariogram(z, xy, edges, max_pairs, seed)
    ok = np.isfinite(v["gamma"]) & (v["counts"] > 0)
    # fit at the mean pair distance of each bin, with Cressie (1985) weights N(h) / gamma(h)^2
    h, g, n_h = v["mean_lag"][ok], v["gamma"][ok], v["counts"][ok]
    # the exact nearest-neighbour semivariance anchors the short-lag behaviour (and the nugget),
    # which random pair sampling resolves poorly
    dnn, inn = cKDTree(xy).query(xy, k=2)
    good = dnn[:, 1] > 0
    if good.sum() > 10:
        h = np.r_[np.mean(dnn[good, 1]), h]
        g = np.r_[np.mean(0.5 * (z[good] - z[inn[good, 1]]) ** 2), g]
        n_h = np.r_[good.sum(), n_h]
    sigma = np.maximum(g, 1e-3 * max(g.max(), 1e-9)) / np.sqrt(n_h)
    if len(h) < 3:
        raise ValueError("too few populated lag bins to fit a covariance model")

    def f(hh, sill, length, nug):
        return CovModel(model, sill, length, nug if nugget else 0.0).gamma(hh)

    p0 = [max(g.max(), 1e-6), max(h[np.argmax(g >= 0.63 * g.max())], h[0]) / 2, 0.0]
    bounds = ([1e-9, 1e-6, 0.0], [10 * max(g.max(), 1e-6), 10.0, max(g.max(), 1e-6)])
    try:
        (sill, length, nug), _ = curve_fit(f, h, g, p0=p0, sigma=sigma, bounds=bounds, maxfev=20_000)
    except RuntimeError:
        sill, length, nug = p0
    out = CovModel(model, float(sill), float(length), float(nug) if nugget else 0.0)
    out.wrss = float(np.sum(((g - out.gamma(h)) / sigma) ** 2))
    return out


def regularization_prediction(family, model: CovModel, n_pairs: int = 20_000, seed: int = 0) -> np.ndarray:
    """Predicted share of base variance retained at each scale on the path,
    from Krige's relation D^2(v|D) = gbar(D,D) - gbar(v,v) with the fitted
    model (block regularization of the covariance)."""
    rng = np.random.default_rng(seed)
    base = family.base
    xy = family.normalized(base)
    p = base.mass / base.mass.sum()
    a = rng.choice(base.n, n_pairs, p=p)
    b = rng.choice(base.n, n_pairs, p=p)
    gDD = float(np.mean(model.gamma(np.hypot(*(xy[a] - xy[b]).T))))
    out = []
    for rep in family.path:
        lab = rep.members
        order = np.argsort(lab, kind="stable")
        lab_s = lab[order]
        starts = np.searchsorted(lab_s, np.arange(rep.n))
        ends = np.searchsorted(lab_s, np.arange(rep.n), side="right")
        gvv = np.zeros(rep.n)
        for u in range(rep.n):
            idx = order[starts[u]:ends[u]]
            m = len(idx)
            if m < 2:
                continue
            pts = xy[idx]
            if m <= 256:
                d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
                gvv[u] = model.gamma(d).sum() / m ** 2
            else:
                i = rng.integers(0, m, 4000)
                j = rng.integers(0, m, 4000)
                gvv[u] = float(np.mean(model.gamma(np.hypot(*(pts[i] - pts[j]).T))))
        out.append(1.0 - float(np.sum(rep.mass * gvv)) / gDD)
    return np.array(out)


# ---------------------------------------------------------------- simulation null

@dataclass
class NullResult:
    model: CovModel
    table: Table
    sims: dict = field(default_factory=dict)
    width: float = 2.5
    n_sim: int = 0

    @property
    def exits(self) -> int:
        return int(sum(1 for r in self.table.rows if r.get("exits")))


def _base_simulator(family, model):
    """A function rng -> simulated base values for this family."""
    if family.kind == "raster":
        lat = family.lattice

        def draw(rng):
            g = simulate_grid(model, lat["shape"], lat["cellsize"] / family.extent, rng)
            return g[lat["rows"], lat["cols"]]
        return draw
    return point_simulator(model, family.normalized(family.base))


def _trajectory(family, base_values, weights):
    v0 = np.var(base_values)
    var_ratio, moran = [], []
    for rep, W in zip(family.path, weights):
        vals = base_values if rep is family.base else family.aggregate_values(base_values, rep)
        var_ratio.append(np.var(vals) / v0 if v0 > 0 else np.nan)
        moran.append(morans_i_fast(vals, W))
    return np.array(var_ratio), np.array(moran)


def information_loss_null(family, metric: str = "knn:8", model="auto", n_sim: int = 100,
                          width: float = 2.5, seed: int = 0, observed: bool = True) -> NullResult:
    """Monte Carlo change-of-support band for the variance and Moran trajectories.

    model : a CovModel, or the name of a model to fit at the base scale."""
    if isinstance(model, str):
        model = fit_covariance(family, model, seed=seed)
    weights = [row_standardize(induced_adjacency(r, metric, family)) for r in family.path]
    rng = np.random.default_rng(seed)
    V = np.empty((n_sim, len(family.path)))
    M = np.empty_like(V)
    draw = _base_simulator(family, model)
    for k in range(n_sim):
        f = draw(rng)
        V[k], M[k] = _trajectory(family, f, weights)
    obs_v, obs_m = _trajectory(family, family.base.values, weights) if observed else (None, None)
    pred = regularization_prediction(family, model, seed=seed)
    t = Table(title=f"Change-of-support null ({n_sim} simulations, band = mean +/- {width:g} SD)",
              note=f"Covariance fitted at the base scale: {model.model}, sill {model.sill:.3f}, "
                   f"length {model.length:.4f} (normalized units), nugget {model.nugget:.3f}"
                   + (" (selected by weighted fit among " + ", ".join(f"{k} {v:.1f}" for k, v in
                                                                    model.scores.items()) + "). "
                      if getattr(model, "scores", None) else ". ") +
                   f"Weights: {metric}. Both diagnostics are tested as change from the base: the "
                   "variance as a share of base variance, Moran's I as I(s) - I(base). Bands are "
                   "drawn around the observed base value.")
    dM = M - M[:, [0]]
    for i, rep in enumerate(family.path):
        vm, vs = V[:, i].mean(), V[:, i].std(ddof=1)
        dm, ds = dM[:, i].mean(), dM[:, i].std(ddof=1)
        row = {"label": rep.label, "s": rep.s, "var_predicted": pred[i],
               "var_null_mean": vm, "var_null_sd": vs,
               "moran_null_mean": M[:, i].mean(), "dmoran_null_mean": dm, "dmoran_null_sd": ds}
        if observed:
            zv = (obs_v[i] - vm) / vs if vs > 0 else 0.0
            zm = (obs_m[i] - obs_m[0] - dm) / ds if ds > 0 else 0.0
            row.update({"var_observed": obs_v[i], "z_var": zv, "moran_observed": obs_m[i], "z_moran": zm,
                        "var_band_lo": vm - width * vs, "var_band_hi": vm + width * vs,
                        "moran_band_lo": obs_m[0] + dm - width * ds,
                        "moran_band_hi": obs_m[0] + dm + width * ds,
                        "exits": bool(i > 0 and (abs(zv) > width or abs(zm) > width))})
        t.append(row)
    return NullResult(model, t, {"var": V, "moran": M}, width, n_sim)


def zoning_null(family, metric: str = "knn:6", n_sim: int = 30, width: float = 2.5, seed: int = 3000) -> Table:
    """Each coarser support on the path compared with random size-matched
    zonings of the base (Voronoi growth from random seeds). Large |z| means
    the observed zoning carries structure beyond its number of units."""
    base = family.base
    v0 = np.var(base.values)
    t = Table(title=f"Zoning null: observed supports vs {n_sim} random size-matched zonings",
              note=f"Band = mean +/- {width:g} SD. Weights: {metric}.")
    for rep in family.path[1:]:
        W = row_standardize(induced_adjacency(rep, metric, family))
        ov, om = np.var(rep.values) / v0, morans_i_fast(rep.values, W)
        sv, sm = [], []
        for r in range(n_sim):
            z = voronoi_zoning(base, rep.n, seed + r)
            Wz = row_standardize(induced_adjacency(z, metric, family))
            sv.append(np.var(z.values) / v0)
            sm.append(morans_i_fast(z.values, Wz))
        sv, sm = np.array(sv), np.array(sm)
        zv = (ov - sv.mean()) / sv.std(ddof=1) if sv.std() > 0 else 0.0
        zm = (om - sm.mean()) / sm.std(ddof=1) if sm.std() > 0 else 0.0
        t.append({"support": rep.label, "units": rep.n, "var_observed": ov, "var_null_mean": sv.mean(),
                  "z_var": zv, "moran_observed": om, "moran_null_mean": sm.mean(), "z_moran": zm,
                  "exits": bool(abs(zv) > width or abs(zm) > width)})
    return t
