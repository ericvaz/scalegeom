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
aggregation with random zonings of the same base data that match its number
of zones and, by default, its distribution of zone sizes and contiguity.

Variance retained is measured with the representation's measure (weighted
variance) by default, which is consistent with mass-weighted aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from .diagnostics import DEFAULT_LAG_EDGES, morans_i_fast, semivariogram
from .metrics import induced_adjacency, row_standardize
from .representation import _contiguity_graph, contiguous_zoning, voronoi_zoning, weighted_variance
from .simulate import CovModel, point_simulator, simulate_grid
from .table import Table


# ---------------------------------------------------------------- model fitting

MODELS = ("gaussian", "exponential", "spherical")


def _short_lag_semivariogram(z, xy, h_max, k=10, short_bins=8, min_pairs=30):
    """Exact semivariance of near-neighbour pairs (each unit with its k
    nearest neighbours) closer than h_max, in distance-quantile bins.
    Returns (mean lag, gamma, count) per bin.

    Near-neighbour pairs give every unit equal weight, as the k-nearest-
    neighbour diagnostics do. On irregular supports, where spacing varies
    with location, the binned values mix distance with location: under
    stationarity this does not bias the fit (see
    examples/null_calibration_irregular.py), but in non-stationary data a
    semivariance that falls with distance at short lags is a sign that local
    variability differs between dense and sparse areas."""
    from scipy.spatial import cKDTree
    n = len(z)
    k = int(min(k, n - 1))
    if k < 1:
        return np.array([]), np.array([]), np.array([])
    d, j = cKDTree(xy).query(xy, k=k + 1)
    i = np.repeat(np.arange(n), k)
    jj, dd = j[:, 1:].ravel(), d[:, 1:].ravel()
    a, b = np.minimum(i, jj), np.maximum(i, jj)
    key = a.astype(np.int64) * n + b
    _, first = np.unique(key, return_index=True)
    a, b, dd = a[first], b[first], dd[first]
    sel = (dd > 0) & (dd <= h_max)
    a, b, dd = a[sel], b[sel], dd[sel]
    if len(dd) < min_pairs:
        return np.array([]), np.array([]), np.array([])
    nb = int(max(1, min(short_bins, len(dd) // min_pairs)))
    q = np.quantile(dd, np.linspace(0, 1, nb + 1))
    q[-1] += 1e-12
    idx = np.clip(np.searchsorted(q, dd, side="right") - 1, 0, nb - 1)
    sq = 0.5 * (z[a] - z[b]) ** 2
    cnt = np.bincount(idx, minlength=nb).astype(float)
    ok = cnt > 0
    hs = np.bincount(idx, dd, nb)[ok] / cnt[ok]
    gs = np.bincount(idx, sq, nb)[ok] / cnt[ok]
    return hs, gs, cnt[ok]


def fit_covariance(family, model: str = "auto", edges=DEFAULT_LAG_EDGES,
                   max_pairs: int = 200_000, seed: int = 0, nugget="auto") -> CovModel:
    """Fit a covariance model to the empirical semivariogram of the base
    representation (values standardized by the base moments, normalized lags).

    model="auto" fits the Gaussian, exponential and spherical models and keeps
    the best. nugget="auto" fits each model with and without a nugget. The
    candidates are compared by an AIC-type criterion on the weighted residual
    sum of squares, m log(WRSS / m) + 2 p, so a nugget is kept only when it
    improves the fit enough to justify the extra parameter. Both choices
    matter for the null: a smooth model fitted to a rough field, or a spurious
    nugget, changes the predicted behaviour at the first aggregation steps and
    makes the test reject when it should not (see examples/null_calibration.py).
    The criterion of every candidate is stored in `.scores`."""
    if model == "auto" or nugget == "auto":
        models = MODELS if model == "auto" else (model,)
        nuggets = (False, True) if nugget == "auto" else (bool(nugget),)
        fits = [fit_covariance(family, m, edges, max_pairs, seed, ng) for m in models for ng in nuggets]
        best = min(fits, key=lambda m: m.aic)
        best.scores = {(m.model + (" + nugget" if m.has_nugget else "")): m.aic for m in fits}
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
    # Short lags: random pair sampling resolves them poorly, and on irregular supports the first
    # bin of the reporting grid can be much wider than the spacing of the units. The fit therefore
    # replaces the first bin by the exact semivariance of near-neighbour pairs inside it, in
    # distance-quantile bins.
    hs, gs, ns = _short_lag_semivariogram(z, xy, float(edges[1]))
    if len(hs):
        keep = h > edges[1]
        h, g, n_h = np.r_[hs, h[keep]], np.r_[gs, g[keep]], np.r_[ns, n_h[keep]]
    sigma = np.maximum(g, 1e-3 * max(g.max(), 1e-9)) / np.sqrt(n_h)
    if len(h) < 3:
        raise ValueError("too few populated lag bins to fit a covariance model")

    gmax = max(g.max(), 1e-6)
    p0 = [gmax, max(h[np.argmax(g >= 0.63 * g.max())], h[0]) / 2, 0.0]
    lo, hi = [1e-9, 1e-6, 0.0], [10 * gmax, 10.0, gmax]
    k = 3 if nugget else 2

    def f(hh, *p):
        return CovModel(model, p[0], p[1], p[2] if nugget else 0.0).gamma(hh)

    try:
        p, _ = curve_fit(f, h, g, p0=p0[:k], sigma=sigma, bounds=(lo[:k], hi[:k]), maxfev=20_000)
    except RuntimeError:
        p = p0[:k]
    out = CovModel(model, float(p[0]), float(p[1]), float(p[2]) if nugget else 0.0)
    out.wrss = float(np.sum(((g - out.gamma(h)) / sigma) ** 2))
    m = len(h)
    out.aic = float(m * np.log(max(out.wrss, 1e-12) / m) + 2 * k)
    out.has_nugget = bool(nugget)
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
    simultaneous: dict = field(default_factory=dict)

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


def _trajectory(family, base_values, weights, weighted=True):
    v0 = weighted_variance(base_values, family.base.mass if weighted else None)
    var_ratio, moran = [], []
    for rep, W in zip(family.path, weights):
        vals = base_values if rep is family.base else family.aggregate_values(base_values, rep)
        v = weighted_variance(vals, rep.mass if weighted else None)
        var_ratio.append(v / v0 if v0 > 0 else np.nan)
        moran.append(morans_i_fast(vals, W))
    return np.array(var_ratio), np.array(moran)


def _with_base_values(family, values):
    import copy
    fam = copy.copy(family)
    fam.path = [family.base.with_values(values)] + list(family.path[1:])
    return fam


def information_loss_null(family, metric: str = "knn:8", model="auto", n_sim: int = 100,
                          width: float = 2.5, seed: int = 0, observed: bool = True,
                          uncertainty: bool = True, weighted: bool = True,
                          alpha: float = 0.05) -> NullResult:
    """Monte Carlo change-of-support band for the variance and Moran trajectories.

    model : a CovModel, or the name of a model to fit at the base scale.
    uncertainty : propagate the uncertainty of the fitted covariance parameters
        by parametric bootstrap. Each simulation draws a field from the fitted
        model, refits the same model form to that field, and simulates the
        trajectory from the refitted parameters. Without this the band treats
        the fitted parameters as known and is too narrow for rough fields,
        where the first aggregation steps depend strongly on the estimated
        short-range covariance (see examples/null_calibration.py).
    weighted : measure variance retained with each representation's measure
        (consistent with mass-weighted aggregation). False gives the
        unweighted variance of unit values.
    alpha : level of the simultaneous test. Besides the per-scale band, the
        result carries a test across all scales and both diagnostics at once:
        the statistic is the largest absolute standardized departure, and its
        reference distribution is the same statistic computed for each
        simulated trajectory (result.simultaneous)."""
    if isinstance(model, str):
        model = fit_covariance(family, model, seed=seed)
    weights = [row_standardize(induced_adjacency(r, metric, family)) for r in family.path]
    rng = np.random.default_rng(seed)
    V = np.empty((n_sim, len(family.path)))
    M = np.empty_like(V)
    draw = _base_simulator(family, model)
    nug = getattr(model, "has_nugget", model.nugget > 0)
    for k in range(n_sim):
        f = draw(rng)
        if uncertainty:
            refit = fit_covariance(_with_base_values(family, f), model.model, seed=seed + 1 + k, nugget=nug)
            f = _base_simulator(family, refit)(rng)
        V[k], M[k] = _trajectory(family, f, weights, weighted)
    obs_v, obs_m = _trajectory(family, family.base.values, weights, weighted) if observed else (None, None)
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
    simultaneous = {}
    if len(family.path) > 1 and n_sim > 2:
        X = np.concatenate([V[:, 1:], dM[:, 1:]], axis=1)
        mu, sd = X.mean(0), X.std(0, ddof=1)
        sd = np.where(sd > 0, sd, np.inf)
        sim_max = np.max(np.abs((X - mu) / sd), axis=1)
        crit = float(np.quantile(sim_max, 1 - alpha))
        simultaneous = {"critical": crit, "alpha": alpha, "n_sim": n_sim}
        if observed:
            x = np.concatenate([obs_v[1:], obs_m[1:] - obs_m[0]])
            stat = float(np.max(np.abs((x - mu) / sd)))
            simultaneous.update({"statistic": stat,
                                 "p": float((1 + np.sum(sim_max >= stat)) / (n_sim + 1)),
                                 "rejects": bool(stat > crit)})
        t.note += (f" Simultaneous test across scales and both diagnostics: critical max |z| = {crit:.2f} "
                   f"at level {alpha:g}" + (f"; observed {simultaneous['statistic']:.2f}, "
                                            f"p = {simultaneous['p']:.3f}." if observed else "."))
    return NullResult(model, t, {"var": V, "moran": M}, width, n_sim, simultaneous)


def zoning_null(family, metric: str = "knn:6", n_sim: int = 100, width: float = 2.5, seed: int = 3000,
                method: str = "size_matched", weighted: bool = True) -> Table:
    """Each coarser support on the path compared with random zonings of the
    base into the same number of zones.

    method : "size_matched" (default) grows contiguous zones whose sizes, in
        base units, follow those of the observed support (see
        contiguous_zoning). "voronoi" assigns base units to the nearest of
        randomly chosen seeds, which gives zones of roughly equal size; it is
        kept for comparison, since a partition with very unequal zones can
        differ from it for that reason alone.
    weighted : measure variance retained with the representation's measure.

    Large |z| means the observed zoning differs from random zonings of this
    kind. With method="voronoi" that can reflect the size distribution of the
    zones as well as where their boundaries fall."""
    base = family.base
    wb = base.mass if weighted else None
    v0 = weighted_variance(base.values, wb)
    graph = _contiguity_graph(base) if method == "size_matched" else None
    t = Table(title=f"Zoning null: observed supports vs {n_sim} random zonings ({method.replace('_', '-')})",
              note=f"Band = mean +/- {width:g} SD. Weights: {metric}. Variance "
                   f"{'weighted by the measure' if weighted else 'unweighted'}. "
                   + ("Random zones are contiguous and follow the observed distribution of zone sizes."
                      if method == "size_matched" else
                      "Random zones are nearest-seed (Voronoi) partitions of roughly equal size."))
    for rep in family.path[1:]:
        W = row_standardize(induced_adjacency(rep, metric, family))
        ov = weighted_variance(rep.values, rep.mass if weighted else None) / v0
        om = morans_i_fast(rep.values, W)
        sizes = np.bincount(rep.members[rep.members >= 0], minlength=rep.n)
        sv, sm = [], []
        for r in range(n_sim):
            if method == "size_matched":
                z = contiguous_zoning(base, sizes, seed + r, graph=graph)
            elif method == "voronoi":
                z = voronoi_zoning(base, rep.n, seed + r)
            else:
                raise ValueError("method must be 'size_matched' or 'voronoi'")
            Wz = row_standardize(induced_adjacency(z, metric, family))
            sv.append(weighted_variance(z.values, z.mass if weighted else None) / v0)
            sm.append(morans_i_fast(z.values, Wz))
        sv, sm = np.array(sv), np.array(sm)
        zv = (ov - sv.mean()) / sv.std(ddof=1) if sv.std() > 0 else 0.0
        zm = (om - sm.mean()) / sm.std(ddof=1) if sm.std() > 0 else 0.0
        t.append({"support": rep.label, "units": rep.n, "method": method, "var_observed": ov,
                  "var_null_mean": sv.mean(), "var_null_sd": sv.std(ddof=1), "z_var": zv,
                  "var_share_below": float(np.mean(sv <= ov)), "moran_observed": om,
                  "moran_null_mean": sm.mean(), "z_moran": zm,
                  "exits": bool(abs(zv) > width or abs(zm) > width)})
    return t
