"""The scale-explicit protocol of Vaz (2026), Section 10.4.

    (i)   state the representation theta = (s, zeta) and the induced metric;
    (ii)  recompute the core diagnostics across scales (and, where relevant,
          lattice placements and alternative induced metrics), reporting
          trajectories rather than single values;
    (iii) test those trajectories against the change-of-support null;
    (iv)  where neighbourhood-based conclusions are transported across
          resolutions, report the bi-Lipschitz distortion of inter-scale maps.

The representation space (FGW distance, scale velocity, zoning displacement)
is available as an additional measurement of the family itself.
"""
from __future__ import annotations

import numpy as np

from . import __version__
from .diagnostics import (DEFAULT_LAG_EDGES, correlation_profile, local_moran, morans_i,
                          range_proxy, sample_pairs, semivariogram)
from .metrics import describe, induced_adjacency
from .report import Report
from .table import Table


class ScaleProtocol:
    """Run the scale-explicit workflow on a Family of representations.

    Parameters
    ----------
    family       : a Family (raster_family, grid_family, nested_family, ...).
    metric       : induced metric used for Moran's I, e.g. "knn:8", "queen", "hops:2".
    alt_metrics  : further metrics whose Moran's I is reported at every scale.
    tau          : correlation threshold for the range proxy.
    lag_edges    : common normalized lag grid for semivariograms and rho(h).
    gamma_bin    : index of the lag bin reported as gamma(h) in the trajectory
                   table (default 2, the bin 0.067-0.100 on the default grid).
    permutations : permutations for Moran's I inference.
    """

    def __init__(self, family, metric: str = "knn:8", alt_metrics=(), tau: float = 0.1,
                 lag_edges=DEFAULT_LAG_EDGES, gamma_bin: int = 2, permutations: int = 99,
                 max_pairs: int = 200_000, seed: int = 0):
        self.family = family
        self.metric = metric
        self.alt_metrics = list(alt_metrics)
        self.tau = tau
        self.lag_edges = np.asarray(lag_edges, float)
        self.gamma_bin = gamma_bin
        self.permutations = permutations
        self.max_pairs = max_pairs
        self.seed = seed
        self.profiles = {}
        self.results = {}

    # ---------------------------------------------------------- step (i)
    def describe(self) -> Table:
        f = self.family
        t = Table(title="Step (i): representations and induced metric",
                  note=f"Induced metric: {describe(self.metric)}. Distances normalized by the "
                       f"domain extent ({f.extent:g} map units).")
        for i, rep in enumerate(f.path):
            t.append({"label": rep.label, "s": rep.s, "zeta": str(rep.zeta), "units": rep.n,
                      "role": "base" if i == 0 else "scale path"})
            for v in f.zoning.get(i, []):
                t.append({"label": v.label, "s": v.s, "zeta": str(v.zeta), "units": v.n,
                          "role": "zoning variant"})
        return t

    # ---------------------------------------------------------- step (ii)
    def _diagnose(self, rep, zoning: bool, idx: int) -> dict:
        f = self.family
        mu, sd = f.reference_moments()
        xy = f.normalized(rep)
        A = induced_adjacency(rep, self.metric, f)
        m = morans_i(rep.values, A, self.permutations, seed=self.seed + idx)
        pairs = sample_pairs(rep.n, self.max_pairs, self.seed + idx)
        g = semivariogram((rep.values - mu) / sd, xy, self.lag_edges, pairs=pairs)
        c = correlation_profile(rep.values, xy, self.lag_edges, pairs=pairs)
        if not zoning:
            self.profiles[rep.label] = {"centers": g["centers"], "gamma": g["gamma"], "rho": c["rho"],
                                        "counts": g["counts"]}
        v0 = np.var(f.base.values)
        row = {"label": rep.label, "s": rep.s, "zoning": bool(zoning), "units": rep.n,
               "var_retained": float(np.var(rep.values) / v0) if v0 > 0 else np.nan,
               "moran_I": m.I, "moran_p": m.p_sim,
               "gamma_h": float(g["gamma"][self.gamma_bin]) if self.gamma_bin < len(g["gamma"]) else np.nan,
               "range_proxy": range_proxy(c["centers"], c["rho"], self.tau)}
        for spec in self.alt_metrics:
            try:
                row[f"I[{spec}]"] = morans_i(rep.values, induced_adjacency(rep, spec, f), 0).I
            except ValueError:
                row[f"I[{spec}]"] = np.nan
        return row

    def trajectory(self, include_zoning: bool = True) -> Table:
        e = self.lag_edges
        lo, hi = e[self.gamma_bin], e[self.gamma_bin + 1]
        t = Table(title="Step (ii): diagnostic trajectories along the scale path",
                  note=f"Moran's I under {describe(self.metric)} with {self.permutations} permutations; "
                       f"gamma_h is the semivariance in the lag bin {lo:.3f} to {hi:.3f} (field standardized "
                       f"by base moments); range proxy is the lag where rho(h) falls below {self.tau:g}; "
                       "all lags normalized by the domain extent.")
        k = 0
        for i, rep in enumerate(self.family.path):
            t.append(self._diagnose(rep, False, k)); k += 1
            if include_zoning:
                for v in self.family.zoning.get(i, []):
                    t.append(self._diagnose(v, True, k)); k += 1
        self.results["trajectory"] = t
        return t

    # ---------------------------------------------------------- step (iii)
    def null_test(self, model="gaussian", n_sim: int = 100, width: float = 2.5, kind: str = "auto",
                  seed: int | None = None):
        """kind="covariance": information-loss (change-of-support) null;
        kind="zoning": random size-matched zonings (nested supports);
        kind="auto": covariance for raster and grid families, both for nested."""
        from .null import information_loss_null, zoning_null
        seed = self.seed if seed is None else seed
        out = {}
        if kind in ("auto", "covariance"):
            out["covariance"] = information_loss_null(self.family, self.metric, model, n_sim, width, seed)
            self.results["null"] = out["covariance"]
        if kind == "zoning" or (kind == "auto" and self.family.kind == "nested"):
            out["zoning"] = zoning_null(self.family, self.metric, min(n_sim, 30), width, seed + 3000)
            self.results["zoning_null"] = out["zoning"]
        return out

    # ---------------------------------------------------------- step (iv)
    def invariance(self, n_sample: int = 150, method: str = "centroid", quantiles=(0.0, 1.0)) -> Table:
        from .space import invariance_table
        t = invariance_table(self.family, n_sample, method, quantiles, self.seed)
        self.results["invariance"] = t
        return t

    # ---------------------------------------------------------- representation space
    def representation_space(self, alpha: float = 0.5, max_nodes: int = 800) -> dict:
        from .space import representation_space
        sp = representation_space(self.family, alpha, max_nodes, self.seed)
        self.results["space"] = sp
        return sp

    # ---------------------------------------------------------- everything
    def run(self, null: bool = True, fgw: bool = True, invariance: bool = True, n_sim: int = 100,
            model="gaussian", width: float = 2.5, alpha: float = 0.5, max_nodes: int = 800,
            figures: bool = True, title: str = "Scale-conditioned analysis", progress=None) -> Report:
        """Run steps (i) to (iv) and the representation space; progress, if
        given, is called as progress(percent, message)."""
        say = progress or (lambda pct, msg: None)
        rep = Report(title)
        rep.meta.update({"scalegeom_version": __version__, "metric": self.metric,
                         "metric_description": describe(self.metric), "alt_metrics": self.alt_metrics,
                         "tau": self.tau, "permutations": self.permutations, "seed": self.seed,
                         "family_kind": self.family.kind, "extent": self.family.extent})
        rep.add_table("representations", self.describe())
        say(5, "Step (ii): diagnostic trajectories")
        traj = self.trajectory()
        rep.add_table("trajectory", traj)
        nres = None
        if null:
            say(30, f"Step (iii): change-of-support null ({n_sim} simulations)")
            res = self.null_test(model, n_sim, width)
            if "covariance" in res:
                nres = res["covariance"]
                rep.add_table("null", nres.table)
                rep.meta["null_exits"] = nres.exits
                rep.meta["null_model"] = vars(nres.model)
            if "zoning" in res:
                rep.add_table("zoning_null", res["zoning"])
        if invariance and len(self.family.path) > 1:
            say(55, "Step (iv): bi-Lipschitz invariance")
            rep.add_table("invariance", self.invariance())
        sp = None
        if fgw and len(self.family.path) > 1:
            try:
                say(60, "Representation space (FGW)")
                sp = self.representation_space(alpha, max_nodes)
                rep.add_table("fgw_path", sp["path"])
                if len(sp["zoning"]):
                    rep.add_table("fgw_zoning", sp["zoning"])
                rep.meta["fgw_alpha"] = alpha
                rep.meta["fgw_max_nodes"] = max_nodes
                rep.meta["fgw_sampling_floor"] = sp.get("sampling_floor")
            except ImportError as exc:
                rep.notes.append(f"Representation space skipped: {exc}")
        if figures:
            say(92, "Figures")
            from . import plotting as P
            rep.add_figure("family", P.family_figure(self.family), "The field at several representations.")
            rep.add_figure("moran", P.trajectory_figure(traj, nres, "moran_I", "Moran's I"),
                           "Moran's I along the scale path" + (" against the information-loss null." if nres else "."))
            if nres:
                rep.add_figure("variance", P.trajectory_figure(traj, nres, "var_retained", "Variance retained"),
                               "Share of base variance retained against the information-loss null.")
            rep.add_figure("semivariograms", P.profiles_figure(self.profiles, "gamma"),
                           "Empirical semivariograms on the common lag grid.")
            rep.add_figure("correlation", P.profiles_figure(self.profiles, "rho", self.tau),
                           "Correlation decay profiles; the dashed line marks tau.")
            if sp is not None:
                rep.add_figure("space", P.space_figure(sp), "The family of representations as a measured space.")
        return rep


# ---------------------------------------------------------------- metric sensitivity

def metric_sensitivity(rep, metrics=("queen", "knn:6", "hops:2"), family=None, permutations: int = 999,
                       alpha: float = 0.05, seed: int = 0, figures: bool = True,
                       title: str = "Induced-metric sensitivity") -> Report:
    """Global and local Moran's I for one support under several induced
    metrics (Vaz 2026, Section 9.9). Reports how global dependence, the
    number of significant local clusters, and cluster membership change."""
    results = {}
    t = Table(title="Global and local dependence under alternative induced metrics",
              note=f"{permutations} permutations; local clusters at p <= {alpha:g}.")
    for k, spec in enumerate(metrics):
        A = induced_adjacency(rep, spec, family)
        g = morans_i(rep.values, A, permutations, seed + k)
        loc = local_moran(rep.values, A, permutations, alpha, seed + 100 + k)
        results[spec] = {"name": describe(spec), "I": g.I, "p": g.p_sim, "local": loc,
                         "mean_neighbours": float(A.sum() / rep.n)}
        cnt = loc.counts()
        t.append({"metric": describe(spec), "spec": spec, "mean_neighbours": A.sum() / rep.n, "moran_I": g.I,
                  "p_sim": g.p_sim, "significant_units": loc.n_significant,
                  "HH": cnt["High-High"], "LL": cnt["Low-Low"], "LH": cnt["Low-High"], "HL": cnt["High-Low"]})
    agree = Table(title="Agreement of local cluster labels between metrics (share of units with the same label)")
    for a in metrics:
        row = {"metric": a}
        for b in metrics:
            row[b] = float(np.mean(results[a]["local"].cluster == results[b]["local"].cluster))
        agree.append(row)
    report = Report(title)
    report.meta.update({"scalegeom_version": __version__, "units": rep.n, "permutations": permutations,
                        "alpha": alpha, "seed": seed, "metrics": list(metrics)})
    report.add_table("metrics", t)
    report.add_table("agreement", agree)
    report.payload["clusters"] = {spec: r["local"] for spec, r in results.items()}
    if figures:
        from .plotting import cluster_figure
        report.add_figure("clusters", cluster_figure(rep, results, family),
                          "LISA cluster membership (unit centroids) under each induced metric.")
    return report
