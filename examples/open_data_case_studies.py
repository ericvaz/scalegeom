"""Case studies on the bundled open data.

    python examples/open_data_case_studies.py [--out examples/output/open_data]

1  North Carolina median household income, census tracts -> counties:
   trajectory under three induced metrics, change-of-support null, zoning
   null against random size-matched zonings, bi-Lipschitz distortion.
2  The same variable aggregated to 5, 10, 20 and 40 km grids with
   half-cell shifts: trajectories and the change-of-support null.
3  Household broadband, tract support: global and local Moran's I under
   k-nearest neighbours, queen contiguity and a two-hop network.
4  Elevation from the Blue Ridge to the Piedmont (NOAA ETOPO1): block
   aggregation and the change-of-support null.

Every null is repeated under each of the three covariance models so that
no verdict rests on a single model choice.
"""
import argparse
import os
import time

import scalegeom as sg
from scalegeom import datasets as D
from scalegeom.null import MODELS


def model_robustness(family, metric, n_sim, seed=0):
    t = sg.Table(title="Change-of-support null under each covariance model")
    best = sg.fit_covariance(family, "auto", seed=seed)
    for m in MODELS:
        r = sg.information_loss_null(family, metric, m, n_sim=n_sim, seed=seed)
        t.append({"model": m, "selected": m == best.model, "weighted_rss": best.scores[m],
                  "exits": r.exits, "of": len(family.path) - 1})
    return t


def run(out, n_sim=100):
    os.makedirs(out, exist_ok=True)
    summary = sg.Table(title="Open-data case studies: summary")

    # 1 nested supports
    fam = D.nc_tracts_family("median_income")
    rep = sg.ScaleProtocol(fam, "queen", alt_metrics=["knn:6", "hops:2"], permutations=999).run(
        n_sim=n_sim, fgw=False, title="NC median income: tracts to counties")
    rep.add_table("model_robustness", model_robustness(fam, "queen", n_sim))
    rep.add_table("invariance_pairs", sg.space.invariance_table(fam, method="pairs", quantiles=(0.01, 0.99)))
    rep.save(os.path.join(out, "1_nc_income_counties"))
    zn = rep["zoning_null"].rows[0]
    summary.append({"case": "1 NC income, tracts to counties", "units": f"{fam.path[0].n} to {fam.path[1].n}",
                    "model": rep.meta["null_model"]["model"], "null_exits": rep.meta["null_exits"],
                    "finding": f"counties keep {zn['var_observed']:.0%} of tract variance vs "
                               f"{zn['var_null_mean']:.0%} for random zonings (z = {zn['z_var']:.1f})"})

    # 2 grids
    fam2 = D.nc_tracts_grid_family("median_income", cell_sizes_km=(5, 10, 20, 40))
    rep2 = sg.ScaleProtocol(fam2, "knn:6", alt_metrics=["queen"], permutations=999).run(
        n_sim=n_sim, title="NC median income: 5 to 40 km grids")
    rep2.add_table("model_robustness", model_robustness(fam2, "knn:6", n_sim))
    rep2.save(os.path.join(out, "2_nc_income_grids"))
    summary.append({"case": "2 NC income, 5 to 40 km grids", "units": f"{fam2.path[0].n} to {fam2.path[-1].n}",
                    "model": rep2.meta["null_model"]["model"], "null_exits": rep2.meta["null_exits"],
                    "finding": "variance falls faster and dependence persists longer than a single "
                               "stationary structure predicts"})

    # 3 metric sensitivity
    rep3 = sg.metric_sensitivity(D.nc_tracts("pct_broadband"), ["knn:6", "queen", "hops:2"], permutations=999,
                                 title="NC household broadband: three induced metrics")
    rep3.save(os.path.join(out, "3_nc_broadband_metrics"))
    ag = rep3["agreement"].rows
    summary.append({"case": "3 NC broadband, three metrics", "units": D.nc_tracts("pct_broadband").n,
                    "model": "", "null_exits": "",
                    "finding": f"Moran's I {min(rep3['metrics']['moran_I']):.2f} to {max(rep3['metrics']['moran_I']):.2f}; "
                               f"k-NN and two-hop cluster labels agree for {ag[0]['hops:2']:.0%} of tracts"})

    # 4 elevation
    fam4 = D.etopo_family()
    rep4 = sg.ScaleProtocol(fam4, "knn:8", permutations=99).run(n_sim=n_sim, title="Blue Ridge elevation (ETOPO1)")
    rep4.add_table("model_robustness", model_robustness(fam4, "knn:8", n_sim))
    rep4.save(os.path.join(out, "4_etopo_elevation"))
    summary.append({"case": "4 Blue Ridge elevation", "units": f"{fam4.path[0].n} to {fam4.path[-1].n}",
                    "model": rep4.meta["null_model"]["model"], "null_exits": rep4.meta["null_exits"],
                    "finding": "trajectory consistent with information loss under the selected model"})

    print(summary)
    summary.to_csv(os.path.join(out, "summary.csv"))
    for r in (rep, rep2, rep4):
        print("\n", r.title)
        print(r["model_robustness"])
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "open_data"))
    ap.add_argument("--n-sim", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    run(a.out, a.n_sim)
    print(f"\ndone in {time.time() - t0:.0f} s; reports in {a.out}")
