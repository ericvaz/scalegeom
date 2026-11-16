"""Size and power of the change-of-support test with a variance surface.

    python examples/surface_calibration.py [--fields 100] [--n-sim 100] [--workers 10]

The support is the one used by examples/null_calibration_irregular.py: the
2,628 North Carolina census tract centroids, weighted by population and
aggregated to square grids of 5, 10, 20 and 40 km. The covariate is the log
of tract population density, standardized.

Fields are simulated at the tract centroids as sigma(x) Y(x), where
log sigma = b c with c the standardized covariate, plus, in two scenarios,
structure that the variance surface does not describe:

    surface b = 0.3, Gaussian     Y Gaussian, length 8 km
    surface b = 0.3, exponential  Y exponential, length 8 km
    surface b = 0.15, Gaussian    weaker surface
    stationary, Gaussian          b = 0 (the stationary null)
    surface + two scales, 20%     sigma times 80% of the variance at 8 km, plus 20% at 64 km
    surface + trend, 10%          sigma times 90% at 8 km, plus a linear east-west trend with 10%

Every field is tested twice, with the stationary null and with the null
that has a variance surface in the covariate, each with the covariance
model selected from the data and 100 simulations. Under the first four
scenarios the surface test's rejection rate is its false-rejection rate,
and the stationary test's rejection rate shows how often a variance surface
alone is mistaken for structure beyond aggregation. Under the last two it is
the surface test's power against structure it does not include.
"""
import argparse
import os
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402

SCENARIOS = {
    "surface b = 0.3, Gaussian": (0.3, [("gaussian", 1.0, 8.0)]),
    "surface b = 0.3, exponential": (0.3, [("exponential", 1.0, 8.0)]),
    "surface b = 0.15, Gaussian": (0.15, [("gaussian", 1.0, 8.0)]),
    "stationary, Gaussian": (0.0, [("gaussian", 1.0, 8.0)]),
    "surface + two scales, 20%": (0.3, [("gaussian", 0.8, 8.0), ("regional", 0.2, 64.0)]),
    "surface + trend, 10%": (0.3, [("gaussian", 0.9, 8.0), ("trend", 0.1, None)]),
}
CELLS_KM = (5, 10, 20, 40)


def covariate(base):
    a = base.attributes
    return np.log(a["pop"] / a["area_km2"])


def _one(args):
    name, k, n_sim = args
    import scalegeom as sg
    from scalegeom import datasets as D
    from scalegeom.simulate import CovModel, point_simulator

    base = D.nc_tracts("median_income", contiguity=False)
    fam0 = sg.grid_family(base, [1000.0 * c for c in CELLS_KM], zoning=None)
    xy = fam0.normalized(base)
    km = 1000.0 / fam0.extent
    x = covariate(base)
    c = (x - x.mean()) / x.std()
    b, parts = SCENARIOS[name]
    sigma = np.exp(b * c)
    rng = np.random.default_rng(50_000 + 1_000 * list(SCENARIOS).index(name) + k)
    f = np.zeros(base.n)
    for model, share, length in parts:
        if model == "trend":
            e = (xy[:, 0] - xy[:, 0].mean()) / xy[:, 0].std()
            f += np.sqrt(share) * e * np.sqrt(np.average(sigma ** 2))
        elif model == "regional":
            f += point_simulator(CovModel("gaussian", share, length * km), xy)(rng) * np.sqrt(np.average(sigma ** 2))
        else:
            f += sigma * point_simulator(CovModel(model, share, length * km), xy)(rng)
    fam = sg.grid_family(base.with_values(f), [1000.0 * c for c in CELLS_KM], zoning=None)
    fam.base.attributes = base.attributes
    out = {"scenario": name, "field": k}
    t0 = time.time()
    r0 = sg.information_loss_null(fam, "knn:6", "auto", n_sim=n_sim, seed=k)
    r1 = sg.information_loss_null(fam, "knn:6", "auto", n_sim=n_sim, seed=k, covariate=x)
    out.update({"stationary_model": r0.model.model, "stationary_any": r0.exits > 0,
                "stationary_simultaneous": r0.simultaneous["rejects"], "stationary_p": r0.simultaneous["p"],
                "surface_model": r1.model.model, "surface_slope": r1.surface.slope,
                "surface_any": r1.exits > 0, "surface_simultaneous": r1.simultaneous["rejects"],
                "surface_p": r1.simultaneous["p"], "seconds": time.time() - t0})
    return out


def run(n_fields, n_sim, workers, out, only=None):
    from multiprocessing import Pool

    import scalegeom as sg
    names = [n for n in SCENARIOS if not only or any(o in n for o in only)]
    jobs = [(name, k, n_sim) for name in names for k in range(n_fields)]
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(_one, jobs, chunksize=1)
    raw = sg.Table(res, title="Raw results")
    summary = sg.Table(title=f"Rejection rates of the stationary and variance-surface tests ({n_fields} fields "
                             f"per scenario, {n_sim} null simulations each)")
    for name in names:
        rows = [r for r in res if r["scenario"] == name]
        sl = np.array([r["surface_slope"] for r in rows])
        summary.append({"scenario": name, "fields": len(rows),
                        "true_slope": SCENARIOS[name][0], "slope_mean": sl.mean(), "slope_sd": sl.std(ddof=1),
                        "stationary_any_2.5sd": np.mean([r["stationary_any"] for r in rows]),
                        "stationary_simultaneous_5pct": np.mean([r["stationary_simultaneous"] for r in rows]),
                        "surface_any_2.5sd": np.mean([r["surface_any"] for r in rows]),
                        "surface_simultaneous_5pct": np.mean([r["surface_simultaneous"] for r in rows])})
    os.makedirs(out, exist_ok=True)
    raw.to_csv(os.path.join(out, "surface_calibration_raw.csv"))
    summary.to_csv(os.path.join(out, "surface_calibration.csv"))
    print(summary)
    print(f"{time.time() - t0:.0f} s")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, default=100)
    ap.add_argument("--n-sim", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "calibration"))
    ap.add_argument("--only", nargs="*", help="run only scenarios whose name contains one of these words")
    a = ap.parse_args()
    run(a.fields, a.n_sim, a.workers, a.out, a.only)
