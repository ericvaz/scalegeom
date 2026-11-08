"""Size and power of the change-of-support test on an irregular support.

    python examples/null_calibration_irregular.py [--fields 100] [--n-sim 100] [--workers 10]

The support is the set of 2,628 North Carolina census tract centroids, with
tract population as the measure, aggregated to square grids of 5, 10, 20 and
40 km. Grid cells contain very different numbers of tracts, so this is an
irregular, unequal-area aggregation.

Fields are simulated at the tract centroids from known structures:

    null, Gaussian       stationary Gaussian covariance, length 8 km
    null, exponential    stationary exponential covariance, length 8 km
    two scales, 20%      80% of the variance at 8 km, 20% at 64 km
    two scales, 10%      90% at 8 km, 10% at 64 km
    trend, 10%           90% at 8 km plus a linear east-west trend with 10%

For each field the covariance model is selected from the data and the test
is run as in practice. Under the two null scenarios the share of fields that
leave the band estimates the false-rejection rate; under the others it
estimates power. Both the per-scale rule (any scale outside mean +/- 2.5 SD)
and the simultaneous max-statistic test at the 5% level are recorded.
"""
import argparse
import os
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402

SCENARIOS = {
    "null, Gaussian": [("gaussian", 1.0, 8.0)],
    "null, exponential": [("exponential", 1.0, 8.0)],
    "two scales, 20%": [("gaussian", 0.8, 8.0), ("gaussian", 0.2, 64.0)],
    "two scales, 10%": [("gaussian", 0.9, 8.0), ("gaussian", 0.1, 64.0)],
    "trend, 10%": [("gaussian", 0.9, 8.0), ("trend", 0.1, None)],
}
CELLS_KM = (5, 10, 20, 40)


def _one(args):
    name, k, n_sim = args
    import scalegeom as sg
    from scalegeom import datasets as D
    from scalegeom.simulate import CovModel, point_simulator

    base = D.nc_tracts("median_income", contiguity=False)
    fam0 = sg.grid_family(base, [1000.0 * c for c in CELLS_KM], zoning=None)
    xy = fam0.normalized(base)
    km = 1000.0 / fam0.extent
    rng = np.random.default_rng(10_000 + 1_000 * list(SCENARIOS).index(name) + k)
    f = np.zeros(base.n)
    for model, share, length in SCENARIOS[name]:
        if model == "trend":
            x = (xy[:, 0] - xy[:, 0].mean()) / xy[:, 0].std()
            f += np.sqrt(share) * x
        else:
            f += point_simulator(CovModel(model, share, length * km), xy)(rng)
    fam = sg.grid_family(base.with_values(f), [1000.0 * c for c in CELLS_KM], zoning=None)
    r = sg.information_loss_null(fam, "knn:6", "auto", n_sim=n_sim, seed=k)
    ex = [bool(x) for x in r.table["exits"]][1:]
    return {"scenario": name, "field": k, "model": r.model.model, "any": any(ex),
            "simultaneous": r.simultaneous["rejects"], **{f"s{c}": e for c, e in zip(CELLS_KM, ex)}}


def run(n_fields, n_sim, workers, out, only=None):
    from multiprocessing import Pool

    import scalegeom as sg
    names = [n for n in SCENARIOS if not only or any(o in n for o in only)]
    jobs = [(name, k, n_sim) for name in names for k in range(n_fields)]
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(_one, jobs, chunksize=2)
    raw = sg.Table(res, title="Raw results")
    summary = sg.Table(title=f"Rejection rates on an irregular support ({n_fields} fields per scenario, "
                             f"{n_sim} null simulations each)")
    for name in names:
        rows = [r for r in res if r["scenario"] == name]
        models = {}
        for r in rows:
            models[r["model"]] = models.get(r["model"], 0) + 1
        row = {"scenario": name, "fields": len(rows),
               "any_scale_2.5sd": np.mean([r["any"] for r in rows]),
               "simultaneous_5pct": np.mean([r["simultaneous"] for r in rows])}
        for c in CELLS_KM:
            row[f"rate_{c}km"] = np.mean([r[f"s{c}"] for r in rows])
        row["models_selected"] = ", ".join(f"{m} {n}" for m, n in sorted(models.items()))
        summary.append(row)
    os.makedirs(out, exist_ok=True)
    raw.to_csv(os.path.join(out, "calibration_irregular_raw.csv"))
    summary.to_csv(os.path.join(out, "calibration_irregular.csv"))
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
