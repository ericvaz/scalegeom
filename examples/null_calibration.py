"""Size of the change-of-support test when its null hypothesis is true.

    python examples/null_calibration.py [--fields 100] [--out examples/output/calibration]

Fields are simulated from a known stationary covariance (Gaussian or
exponential) on a 96 x 96 grid and aggregated to blocks of 2, 4, 8 and 16
cells. For each field the covariance model is selected and fitted from the
data (model="auto"), the null band is simulated, and we record whether the
observed trajectory leaves the band. Because each field satisfies the null,
the share of fields that leave the band estimates the false-rejection rate,
per scale and for the trajectory as a whole (any scale). Covariance
parameters are estimated, not known, so this measures the procedure as used.
"""
import argparse
import os
import time

import numpy as np

import scalegeom as sg
from scalegeom.simulate import CovModel, simulate_grid

SCALES = [1, 2, 4, 8, 16]


def run(n_fields=100, n_sim=60, width=2.5, out=None):
    rows = sg.Table(title=f"False-rejection rate of the change-of-support test ({n_fields} fields per model, "
                          f"band = mean +/- {width:g} SD, {n_sim} null simulations)")
    for model, length in (("gaussian", 8 / 96), ("exponential", 10 / 96)):
        truth = CovModel(model, 1.0, length)
        rng = np.random.default_rng(2026)
        per_scale = np.zeros(len(SCALES) - 1)
        any_exit = 0
        chosen = {}
        for k in range(n_fields):
            f = simulate_grid(truth, (96, 96), 1 / 96, rng)
            fam = sg.raster_family(f, SCALES, zoning=None)
            r = sg.information_loss_null(fam, "knn:8", "auto", n_sim=n_sim, width=width, seed=k)
            chosen[r.model.model] = chosen.get(r.model.model, 0) + 1
            ex = np.array([bool(x) for x in r.table["exits"]])[1:]
            per_scale += ex
            any_exit += ex.any()
        row = {"true_model": model, "fields": n_fields,
               "any_scale_rate": any_exit / n_fields,
               "models_selected": ", ".join(f"{m} {c}" for m, c in sorted(chosen.items()))}
        for s, v in zip(SCALES[1:], per_scale / n_fields):
            row[f"rate_s{s}"] = v
        rows.append(row)
        print(row, flush=True)
    if out:
        os.makedirs(out, exist_ok=True)
        rows.to_csv(os.path.join(out, "calibration.csv"))
    print(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, default=100)
    ap.add_argument("--n-sim", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "calibration"))
    a = ap.parse_args()
    t0 = time.time()
    run(a.fields, a.n_sim, out=a.out)
    print(f"done in {time.time() - t0:.0f} s")
