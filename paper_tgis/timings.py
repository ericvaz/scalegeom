"""Run time and memory of each analysis stage for the Transactions in GIS article.

    python paper_tgis/timings.py

Each stage runs in a fresh process so that peak resident memory can be
attributed to it. Writes paper_tgis/robustness/timings.csv.
"""
import csv
import os
import platform
import subprocess
import sys
import time
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    ("Elevation, blocks", "fit", "fam = D.etopo_family(); sg.fit_covariance(fam, 'auto')"),
    ("Elevation, blocks", "change-of-support null, 100 sims", "fam = D.etopo_family(); sg.information_loss_null(fam, 'knn:8', 'auto', n_sim=100)"),
    ("Elevation, blocks", "FGW along the path", "fam = D.etopo_family(); sg.representation_space(fam)"),
    ("Income, grids", "fit", "fam = D.nc_tracts_grid_family('median_income'); sg.fit_covariance(fam, 'auto')"),
    ("Income, grids", "change-of-support null, 100 sims", "fam = D.nc_tracts_grid_family('median_income', zoning=None); sg.information_loss_null(fam, 'knn:6', 'auto', n_sim=100)"),
    ("Income, counties", "zoning null, 100 size-matched zonings", "fam = D.nc_tracts_family('median_income'); sg.zoning_null(fam, 'queen', 100)"),
    ("Broadband, tracts", "local Moran, 3 rules, 999 permutations", "rep = D.nc_tracts('pct_broadband'); sg.metric_sensitivity(rep, ['knn:6', 'queen', 'hops:2'], figures=False)"),
]


def _stage(code, q):
    import resource
    import scalegeom as sg  # noqa: F401
    from scalegeom import datasets as D  # noqa: F401
    t0 = time.perf_counter()
    exec(code, {"sg": sg, "D": D})
    dt = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    q.put((dt, peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024))


def main():
    ctx = get_context("spawn")
    rows = []
    for case, stage, code in STAGES:
        q = ctx.Queue()
        p = ctx.Process(target=_stage, args=(code, q))
        p.start()
        dt, mb = q.get()
        p.join()
        rows.append({"case": case, "stage": stage, "seconds": round(dt, 1), "peak_memory_mb": round(mb)})
        print(rows[-1], flush=True)
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        chip = platform.processor()
    rows.append({"case": "hardware", "stage": f"{chip}; {platform.system()} {platform.release()}; Python {platform.python_version()}; single process",
                 "seconds": "", "peak_memory_mb": ""})
    os.makedirs(os.path.join(HERE, "robustness"), exist_ok=True)
    with open(os.path.join(HERE, "robustness", "timings.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
