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


def _near_neighbour_pairs(xy, h_max, k=10):
    """Unique pairs formed by each unit and its k nearest neighbours, closer
    than h_max. Returns (i, j, distance)."""
    from scipy.spatial import cKDTree
    n = len(xy)
    k = int(min(k, n - 1))
    if k < 1:
        return np.array([], int), np.array([], int), np.array([])
    d, j = cKDTree(xy).query(xy, k=k + 1)
    i = np.repeat(np.arange(n), k)
    jj, dd = j[:, 1:].ravel(), d[:, 1:].ravel()
    a, b = np.minimum(i, jj), np.maximum(i, jj)
    key = a.astype(np.int64) * n + b
    _, first = np.unique(key, return_index=True)
    a, b, dd = a[first], b[first], dd[first]
    sel = (dd > 0) & (dd <= h_max)
    return a[sel], b[sel], dd[sel]


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
    a, b, dd = _near_neighbour_pairs(xy, h_max, k)
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


def regularization_prediction(family, model: CovModel, n_pairs: int = 20_000, seed: int = 0,
                              scale=None) -> np.ndarray:
    """Predicted share of base variance retained at each scale on the path,
    from Krige's relation D^2(v|D) = gbar(D,D) - gbar(v,v) with the fitted
    model (block regularization of the covariance).

    Mean semivariances are weighted by the base measure, which makes the
    prediction the expected between-unit share of the weighted variance,
    the quantity the trajectories report. With `scale`, a local standard
    deviation s_i for every base unit (a variance surface), the semivariance
    between units i and j is (sill + nugget)(s_i^2 + s_j^2) / 2 -
    s_i s_j sill rho(h_ij), and the same relation gives the share retained
    when local variability differs from place to place."""
    rng = np.random.default_rng(seed)
    base = family.base
    xy = family.normalized(base)
    p = base.mass / base.mass.sum()
    s = np.ones(base.n) if scale is None else np.asarray(scale, dtype=float)
    total = model.sill + model.nugget

    def gam(i, j):
        h = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
        g = total * 0.5 * (s[i] ** 2 + s[j] ** 2) - s[i] * s[j] * model.sill * model.rho(h)
        return np.where(i != j, g, 0.0)

    a = rng.choice(base.n, n_pairs, p=p)
    b = rng.choice(base.n, n_pairs, p=p)
    gDD = float(np.mean(gam(a, b)))
    out = []
    for rep in family.path:
        lab = rep.members
        order = np.argsort(lab, kind="stable")
        lab_s = lab[order]
        starts = np.searchsorted(lab_s, np.arange(rep.n))
        ends = np.searchsorted(lab_s, np.arange(rep.n), side="right")
        within = 0.0
        for u in range(rep.n):
            idx = order[starts[u]:ends[u]]
            m = len(idx)
            if m < 2:
                continue
            wu = p[idx]
            Wu = wu.sum()
            if m <= 256:
                I, J = np.meshgrid(idx, idx, indexing="ij")
                gvv = float(wu @ gam(I, J) @ wu) / Wu ** 2
            else:
                q = wu / Wu
                i = rng.choice(idx, 4000, p=q)
                j = rng.choice(idx, 4000, p=q)
                gvv = float(np.mean(gam(i, j)))
            within += Wu * gvv
        out.append(1.0 - within / gDD)
    return np.array(out)


# ---------------------------------------------------------------- variance surface

@dataclass
class VarianceSurface:
    """Local standard deviation that changes with a covariate:
    log sigma(x) = a + b c(x), with c the covariate standardized to mean 0
    and SD 1 over the base units.

    scale : sigma for every base unit, normalized so that its mean square
        under the base measure is one.
    slope : b, the change in log local SD per SD of the covariate. b = 0 is
        the stationary null."""

    slope: float
    scale: np.ndarray
    covariate: np.ndarray
    name: str = "covariate"
    raw: np.ndarray | None = None
    n_pairs: int = 0
    n_cells: int = 0

    def rescale(self, values, mass):
        """Centred values divided by the local SD: the field the stationary
        covariance is fitted to."""
        v = np.asarray(values, dtype=float)
        w = np.asarray(mass, dtype=float)
        return (v - np.sum(w * v) / w.sum()) / self.scale

    def sd_ratio(self, lo: float = 10, hi: float = 90) -> float:
        """Ratio of local SDs between the hi-th and lo-th percentiles of the covariate."""
        c = self.covariate
        return float(np.exp(self.slope * (np.percentile(c, hi) - np.percentile(c, lo))))


def _surface_scale(slope, c, mass):
    e = np.exp(2.0 * slope * c)
    w = mass / mass.sum()
    return np.sqrt(e / np.sum(w * e))


def _covariate_array(family, covariate):
    if isinstance(covariate, str):
        attrs = getattr(family.base, "attributes", None) or {}
        if covariate not in attrs:
            raise ValueError(f"the base representation has no attribute {covariate!r}")
        return covariate, np.asarray(attrs[covariate], dtype=float)
    x = np.asarray(covariate, dtype=float)
    if x.shape != (family.base.n,):
        raise ValueError("covariate must have one value per base unit")
    return "covariate", x


def fit_variance_surface(family, covariate, values=None, name: str | None = None, k: int = 10,
                         distance_bins: int = 12, classes: int = 5, min_pairs: int = 30,
                         edges=DEFAULT_LAG_EDGES) -> VarianceSurface:
    """Estimate how local variability changes with a covariate.

    Under the model Z(x) = mu + sigma(x) Y(x), with Y stationary and
    log sigma = a + b c, the semivariance of a pair of units is

        gamma_ij = T (s_i - s_j)^2 / 2 + s_i s_j gamma_Y(h_ij),   s = exp(b c),

    where T sets the overall variance and gamma_Y is the unknown semivariogram
    of Y. The estimator uses the pairs formed by each base unit and its k
    nearest neighbours that fall in the first lag bin. Pairs are grouped by
    distance (distance_bins quantile bins) and, within each distance bin, by
    the mean covariate of the pair (classes quantile classes). For a given b,
    T is fixed by the standardization of the values, gamma_Y takes a free
    value in every distance bin, and the group means of the semivariance are
    fitted by least squares with weights N / gamma^2; b minimizes the
    weighted residual sum of squares.

    The free value per distance bin absorbs the correlation function, so b
    compares pairs at the same separation in places with different covariate
    values. On irregular supports this matters: close pairs come mostly from
    dense areas, and a comparison that ignored distance would confound
    density with separation. b = 0 is the stationary null.

    covariate : one value per base unit, or the name of an attribute of the
        base representation. Transform it first if needed (for example the
        log of population density); it is standardized here.
    values : base values to use in place of the observed ones (used by the
        bootstrap)."""
    from scipy.optimize import minimize_scalar
    label, x = _covariate_array(family, covariate)
    if not np.all(np.isfinite(x)):
        raise ValueError("covariate has missing or infinite values")
    sd = x.std()
    c = (x - x.mean()) / (sd if sd > 0 else 1.0)
    base = family.base
    v = base.values if values is None else np.asarray(values, dtype=float)
    z = (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)
    a, b, d = _near_neighbour_pairs(family.normalized(base), float(edges[1]), k)
    if len(d) < 2 * min_pairs:
        raise ValueError("too few near-neighbour pairs to estimate a variance surface")
    q = np.quantile(d, np.linspace(0, 1, distance_bins + 1))
    q[-1] += 1e-12
    di = np.clip(np.searchsorted(q, d, side="right") - 1, 0, distance_bins - 1)
    cm = 0.5 * (c[a] + c[b])
    cq = np.quantile(cm, np.linspace(0, 1, classes + 1))
    cq[-1] += 1e-12
    ci = np.clip(np.searchsorted(cq, cm, side="right") - 1, 0, classes - 1)
    cell = di * classes + ci
    ncell = distance_bins * classes
    cnt = np.bincount(cell, minlength=ncell).astype(float)
    ok = cnt >= min_pairs
    keep = ok[cell]
    a, b, cell = a[keep], b[keep], cell[keep]
    _, cell = np.unique(cell, return_inverse=True)
    n_k = np.bincount(cell).astype(float)
    g_k = np.bincount(cell, 0.5 * (z[a] - z[b]) ** 2) / n_k
    d_k = np.unique(np.flatnonzero(ok) // classes, return_inverse=True)[1]
    D = (d_k[:, None] == np.arange(d_k.max() + 1)[None, :]).astype(float)
    wt = np.sqrt(n_k) / np.maximum(g_k, 1e-3 * max(g_k.max(), 1e-9))

    def fit(slope):
        s = np.exp(slope * c)
        T = 1.0 / np.mean(s ** 2)
        A = np.bincount(cell, 0.5 * T * (s[a] - s[b]) ** 2) / n_k
        B = np.bincount(cell, s[a] * s[b]) / n_k
        X = D * B[:, None]
        coef = np.linalg.lstsq(X * wt[:, None], (g_k - A) * wt, rcond=None)[0]
        return float(np.sum(((g_k - A - X @ coef) * wt) ** 2))

    slope = float(minimize_scalar(fit, bounds=(-2.0, 2.0), method="bounded",
                                  options={"xatol": 1e-4}).x)
    return VarianceSurface(slope, _surface_scale(slope, c, base.mass), c, name or label, x,
                           int(n_k.sum()), int(len(n_k)))


def variance_surface_test(family, covariate, model="auto", n_sim: int = 200, seed: int = 0,
                          name: str | None = None) -> dict:
    """Does local variability change with the covariate?

    The slope b of fit_variance_surface is compared with its distribution
    under the stationary change-of-support null: fields are simulated from
    the covariance fitted at the base, which has no variance surface, and the
    slope is re-estimated on each. Returns the observed slope, the mean and SD
    of the simulated slopes, and a two-sided Monte Carlo p-value."""
    surface = fit_variance_surface(family, covariate, name=name)
    if isinstance(model, str):
        model = fit_covariance(family, model, seed=seed)
    draw = _base_simulator(family, model)
    rng = np.random.default_rng(seed)
    sims = np.array([fit_variance_surface(family, surface.raw, values=draw(rng)).slope for _ in range(n_sim)])
    p = float((1 + np.sum(np.abs(sims) >= abs(surface.slope))) / (n_sim + 1))
    return {"name": surface.name, "slope": surface.slope, "sd_ratio_10_90": surface.sd_ratio(),
            "null_mean": float(sims.mean()), "null_sd": float(sims.std(ddof=1)), "p": p, "n_sim": n_sim,
            "surface": surface, "sims": sims}


# ---------------------------------------------------------------- simulation null

@dataclass
class NullResult:
    model: CovModel
    table: Table
    sims: dict = field(default_factory=dict)
    width: float = 2.5
    n_sim: int = 0
    simultaneous: dict = field(default_factory=dict)
    surface: VarianceSurface | None = None

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
                          alpha: float = 0.05, covariate=None) -> NullResult:
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
        simulated trajectory (result.simultaneous).
    covariate : None for the stationary null. Otherwise one value per base
        unit, the name of a base attribute, or a fitted VarianceSurface: the
        null then lets local variability change with the covariate,
        Z(x) = mu + sigma(x) Y(x) with log sigma linear in the covariate and Y
        stationary (fit_variance_surface). The covariance of Y is fitted to
        the centred values divided by sigma, simulated fields are multiplied
        by sigma before aggregation, and the bootstrap re-estimates both the
        surface and the covariance on every simulated field. With a slope of
        zero this is the stationary null, so comparing the two tests whether
        variability tied to the covariate accounts for a departure from the
        stationary null (see examples/surface_calibration.py)."""
    base = family.base
    surface = None
    if covariate is not None:
        surface = covariate if isinstance(covariate, VarianceSurface) else fit_variance_surface(family, covariate)
        fit_family = _with_base_values(family, surface.rescale(base.values, base.mass))
    else:
        fit_family = family
    if isinstance(model, str):
        model = fit_covariance(fit_family, model, seed=seed)
    weights = [row_standardize(induced_adjacency(r, metric, family)) for r in family.path]
    rng = np.random.default_rng(seed)
    V = np.empty((n_sim, len(family.path)))
    M = np.empty_like(V)
    slopes = np.full(n_sim, np.nan)
    base_draw = _base_simulator(family, model)
    nug = getattr(model, "has_nugget", model.nugget > 0)
    for k in range(n_sim):
        f = base_draw(rng)
        if surface is not None:
            f = surface.scale * f
        if uncertainty:
            if surface is not None:
                sf = fit_variance_surface(family, surface.raw, values=f)
                slopes[k] = sf.slope
                refit = fit_covariance(_with_base_values(family, sf.rescale(f, base.mass)), model.model,
                                       seed=seed + 1 + k, nugget=nug)
                f = sf.scale * _base_simulator(family, refit)(rng)
            else:
                refit = fit_covariance(_with_base_values(family, f), model.model, seed=seed + 1 + k, nugget=nug)
                f = _base_simulator(family, refit)(rng)
        V[k], M[k] = _trajectory(family, f, weights, weighted)
    obs_v, obs_m = _trajectory(family, base.values, weights, weighted) if observed else (None, None)
    pred = regularization_prediction(family, model, seed=seed, scale=surface.scale if surface else None)
    kind = "Change-of-support null" if surface is None else f"Change-of-support null with a variance surface in {surface.name}"
    t = Table(title=f"{kind} ({n_sim} simulations, band = mean +/- {width:g} SD)",
              note=(f"Variance surface: log local SD = a + b c, c = standardized {surface.name}; "
                    f"b = {surface.slope:.3f} from {surface.n_pairs} near-neighbour pairs in {surface.n_cells} "
                    f"distance-by-covariate groups, local SD ratio between the 90th and 10th percentiles of c "
                    f"= {surface.sd_ratio():.2f}"
                    + (f", bootstrap SD of b = {np.nanstd(slopes, ddof=1):.3f}" if uncertainty and n_sim > 2 else "")
                    + ". The covariance below is that of the rescaled field. " if surface is not None else "") +
                   f"Covariance fitted at the base scale: {model.model}, sill {model.sill:.3f}, "
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
    sims = {"var": V, "moran": M}
    if surface is not None:
        sims["slope"] = slopes
    return NullResult(model, t, sims, width, n_sim, simultaneous, surface)


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
