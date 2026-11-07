"""Reproduce the simulation and North Carolina results of Vaz (2026) with scalegeom.

    python examples/reproduce_vaz2026.py [--reps 30] [--out examples/output]

Part 1  Table 1: Monte Carlo trajectories of Moran's I, variance retained,
        short-lag semivariance and the range proxy (160 x 160 field,
        correlation length 12, block sizes 1-16, 30 realizations).
Part 2  Section 9.8: Scenario A (one correlation length) and Scenario B
        (lengths 3 and 30) against the change-of-support null.
Part 3  Section 9.7: representation space (FGW) and bi-Lipschitz distortion.
Part 4  Section 9.9 / Table 3: North Carolina SIDS rates under three induced
        metrics (needs geopandas and the libpysal example data).
"""
import argparse
import os
import time

import numpy as np

import scalegeom as sg

SCALES = [1, 2, 4, 5, 8, 10, 16]


def part1(reps, out):
    rows = {s: [] for s in SCALES}
    for r in range(reps):
        fam = sg.raster_family(sg.gaussian_random_field((160, 160), 12, seed=r), SCALES, zoning=None)
        traj = sg.ScaleProtocol(fam, "knn:8", permutations=99, seed=r).trajectory(include_zoning=False)
        for row in traj:
            rows[row["s"]].append(row)
    t = sg.Table(title=f"Table 1 (scalegeom): Monte Carlo means over {reps} realizations")
    for s in SCALES:
        R = rows[s]
        g = lambda k: np.array([x[k] for x in R], float)
        t.append({"s": s, "units": R[0]["units"], "var_retained": g("var_retained").mean(),
                  "moran_I": g("moran_I").mean(), "moran_I_sd": g("moran_I").std(ddof=1),
                  "p_max": g("moran_p").max(), "gamma_h3": g("gamma_h").mean(),
                  "range_proxy": g("range_proxy").mean()})
    rp = np.array(t["range_proxy"])
    t.note = (f"Paper: Moran's I 0.994 -> 0.297, variance retained 1.000 -> 0.752, range proxy 0.160 to 0.189 "
              f"(CV 0.062). Here: range proxy CV across scale means = {rp.std() / rp.mean():.3f}.")
    print(t, "\n", t.note, "\n")
    t.to_csv(os.path.join(out, "table1.csv"))
    return t


def part2(out, n_sim=100):
    res = {}
    for name, f in [("A", sg.gaussian_random_field((160, 160), 12, seed=7)),
                    ("B", sg.mixed_scale_field((160, 160), (3, 30), seed=7))]:
        fam = sg.raster_family(f, SCALES, zoning=None)
        p = sg.ScaleProtocol(fam, "knn:8", permutations=99)
        traj = p.trajectory(include_zoning=False)
        nres = sg.information_loss_null(fam, "knn:8", "gaussian", n_sim=n_sim, seed=1)
        print(f"Scenario {name}: exits the information-loss null at {nres.exits} of {len(SCALES) - 1} "
              f"aggregated scales (fitted length {nres.model.length:.4f}, nugget {nres.model.nugget:.3f})")
        nres.table.to_csv(os.path.join(out, f"null_scenario_{name}.csv"))
        from scalegeom.plotting import trajectory_figure
        fig = trajectory_figure(traj, nres, "moran_I", "Moran's I")
        fig.savefig(os.path.join(out, f"null_scenario_{name}.png"), dpi=200, bbox_inches="tight")
        res[name] = nres
    print()
    return res


def part3(out):
    fam = sg.raster_family(sg.gaussian_random_field((160, 160), 12, seed=0), SCALES)
    p = sg.ScaleProtocol(fam, "knn:8")
    inv = p.invariance(n_sample=150)
    sp = p.representation_space(alpha=0.5, max_nodes=800)
    print(inv, "\n")
    print(sp["path"], "\n")
    print(sp["zoning"], "\n")
    inv.to_csv(os.path.join(out, "table2_bilipschitz.csv"))
    sp["path"].to_csv(os.path.join(out, "table2_fgw.csv"))
    sp["zoning"].to_csv(os.path.join(out, "table2_zoning.csv"))


def part4(out):
    try:
        import geopandas as gpd
        from libpysal import examples
        from scalegeom.geo import from_geodataframe
    except ImportError:
        print("Part 4 skipped: needs geopandas and libpysal")
        return
    gdf = gpd.read_file(examples.get_path("sids2.shp"))
    for var in ["SIDR74", "SIDR79"]:
        rep = from_geodataframe(gdf, var)
        r = sg.metric_sensitivity(rep, ["knn:6", "queen", "hops:2"], permutations=999, seed=0,
                                  title=f"North Carolina {var}: induced-metric sensitivity")
        print(var)
        print(r["metrics"], "\n")
        r.save(os.path.join(out, f"nc_{var}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    part1(a.reps, a.out)
    part2(a.out)
    part3(a.out)
    part4(a.out)
    print(f"done in {time.time() - t0:.0f} s; outputs in {a.out}")
