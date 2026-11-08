"""Figures for the Transactions in GIS article on scalegeom.

    python paper_tgis/make_figures.py [--n-zonings 200]

Reads the case-study outputs written by examples/open_data_case_studies.py
(examples/output/open_data) and recomputes the zoning distribution and the
local cluster maps from the bundled data. Writes 300 dpi PNG files to
paper_tgis/figures.
"""
import argparse
import csv
import os
import shutil

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import scalegeom as sg
from scalegeom import datasets as D
from scalegeom.diagnostics import local_moran
from scalegeom.metrics import induced_adjacency
from scalegeom.representation import voronoi_zoning

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
CASES = os.path.join(ROOT, "examples", "output", "open_data")
INK, MUTED, BAND, ALERT = "#1f2a44", "#6b7280", "#c9ced8", "#b42318"
LISA = {0: "#eeeeee", 1: "#c0392b", 2: "#9ecae1", 3: "#2166ac", 4: "#f4a582"}
LISA_NAMES = {1: "High-High", 3: "Low-Low", 2: "Low-High", 4: "High-Low", 0: "Not significant"}


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(True, color="#eceef2", linewidth=0.6)


def save(fig, name):
    FigureCanvasAgg(fig)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300)
    print("wrote", path)


def read_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


# ------------------------------------------------------------------ Figure 1
def workflow():
    fig = Figure(figsize=(7.2, 1.75))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    boxes = [
        (1, "Family of\nrepresentations", "S(θ), θ = (s, ζ)\nmembership of each\nbase unit recorded"),
        (21, "Induced\nmetrics", "knn:k, band:r,\nqueen, rook,\nhops:h, matrix:k"),
        (41, "Diagnostics on\none lag grid", "variance retained,\nMoran's I, LISA,\nsemivariogram, range"),
        (61, "Null models", "change of support\n(fitted at base),\nsize-matched zonings"),
        (81, "Distortion and\ndistance", "matched-pair\nconstants,\nFGW between supports"),
    ]
    for x, head, body in boxes:
        ax.add_patch(FancyBboxPatch((x, 2), 17, 26, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc="white", ec=INK, lw=0.9))
        ax.text(x + 8.5, 21.5, head, ha="center", va="center", fontsize=8.5, color=INK, weight="bold")
        ax.text(x + 8.5, 9, body, ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.3)
    for x in (18.6, 38.6, 58.6, 78.6):
        ax.add_patch(FancyArrowPatch((x, 15), (x + 2.2, 15), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1))
    save(fig, "fig1_workflow.png")


# ------------------------------------------------------------------ Figure 2
def benchmark():
    src = os.path.join(ROOT, "paper", "figure1.png")
    shutil.copy(src, os.path.join(OUT, "fig2_benchmark.png"))
    print("copied", src)


# ------------------------------------------------------------------ Figure 3
def trajectories():
    cases = [("4_etopo_elevation", "Elevation", "Block size (cells)", 1.0),
             ("2_nc_income_grids", "Tract income", "Grid cell (km)", 1e-3)]
    fig = Figure(figsize=(7.2, 5.2))
    axes = fig.subplots(2, 2)
    for row, (case, title, xlabel, scale) in enumerate(cases):
        null = read_csv(os.path.join(CASES, case, "null.csv"))
        s = np.array([r["s"] for r in null]) * scale
        if case.startswith("2_"):
            s[0] = 0  # tracts are irregular; draw them at the left edge
        cols = [("var_observed", "var_band_lo", "var_band_hi", "Variance retained"),
                ("moran_observed", "moran_band_lo", "moran_band_hi", "Moran's I")]
        for c, (obs, lo, hi, ylab) in enumerate(cols):
            ax = axes[row, c]
            style(ax)
            o = np.array([r[obs] for r in null])
            a = np.array([r[lo] for r in null])
            b = np.array([r[hi] for r in null])
            ax.fill_between(s, a, b, color=BAND, alpha=0.8, lw=0, label="Null band (±2.5 SD)")
            out = (o < a - 1e-12) | (o > b + 1e-12)
            ax.plot(s, o, color=INK, lw=1.2, zorder=3)
            ax.scatter(s[~out], o[~out], s=18, color=INK, zorder=4, label="Inside band")
            ax.scatter(s[out], o[out], s=22, color=ALERT, marker="D", zorder=4, label="Outside band")
            ax.set_ylabel(ylab, fontsize=8.5, color=INK)
            ax.set_xlabel(xlabel, fontsize=8.5, color=INK)
            if case.startswith("2_"):
                ax.set_xticks(s)
                ax.set_xticklabels(["tracts"] + [f"{v:g}" for v in s[1:]])
            ax.set_title(f"({'abcd'[2 * row + c]}) {title}: {ylab[0].lower() + ylab[1:] if ylab[0] == 'V' else ylab}", fontsize=9, color=INK, loc="left")
    h, lab = axes[0, 0].get_legend_handles_labels()
    h2, lab2 = axes[1, 0].get_legend_handles_labels()
    seen = dict(zip(lab, h))
    seen.update(dict(zip(lab2, h2)))
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig3_trajectories.png")


# ------------------------------------------------------------------ Figure 4
def semivariogram_figure():
    """Empirical semivariogram of tract income (short-lag pairs and the
    reporting lag grid) with the covariance model selected for the null."""
    from scalegeom.diagnostics import semivariogram
    from scalegeom.null import _short_lag_semivariogram
    fam = D.nc_tracts_grid_family("median_income", zoning=None)
    m = sg.fit_covariance(fam, "auto")
    b = fam.base
    mu, sd = fam.reference_moments()
    z = (b.values - mu) / sd
    xy = fam.normalized(b)
    v = semivariogram(z, xy, seed=0)
    hs, gs, ns = _short_lag_semivariogram(z, xy, float(v["edges"][1]))
    km = fam.extent / 1000.0
    fig = Figure(figsize=(7.2, 3.0))
    ax = fig.add_subplot(111)
    style(ax)
    hh = np.geomspace(min(hs.min(), 1e-3), v["mean_lag"][np.isfinite(v["mean_lag"])].max(), 200)
    ax.plot(hh * km, m.gamma(hh), color=ALERT, lw=1.3,
            label=f"Fitted {m.model} model (length {m.length * km:.1f} km)")
    ax.scatter(hs * km, gs, s=22, color=INK, marker="s", label="Near-neighbour pairs")
    ok = np.isfinite(v["gamma"])
    ax.scatter(v["mean_lag"][ok][1:] * km, v["gamma"][ok][1:], s=22, color=INK, label="Lag grid")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Lag (km, log scale)", fontsize=8.5, color=INK)
    ax.set_ylabel("Semivariance (standardized)", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig4_income_semivariogram.png")
    return {"model": m.model, "length_km": m.length * km, "sill": m.sill, "nugget": m.nugget,
            "short": list(zip(np.round(hs * km, 1), np.round(gs, 3))),
            "grid": list(zip(np.round(v["mean_lag"][ok] * km, 1), np.round(v["gamma"][ok], 3)))}


# ------------------------------------------------------------------ Figure 5
def counties(n_zonings, seed=3000):
    from scalegeom.representation import _contiguity_graph
    fam = D.nc_tracts_family("median_income")
    base, cty = fam.path[0], fam.path[1]
    v0 = sg.weighted_variance(base.values, base.mass)
    observed = sg.weighted_variance(cty.values, cty.mass) / v0
    sizes = np.bincount(cty.members)
    G = _contiguity_graph(base)
    ens = {"Equal-size (Voronoi) zonings": [], "Tract-count-matched contiguous zonings": []}
    for r in range(n_zonings):
        z1 = voronoi_zoning(base, cty.n, seed + r)
        z2 = sg.contiguous_zoning(base, sizes, seed + r, graph=G)
        ens["Equal-size (Voronoi) zonings"].append(sg.weighted_variance(z1.values, z1.mass) / v0)
        ens["Tract-count-matched contiguous zonings"].append(sg.weighted_variance(z2.values, z2.mass) / v0)

    g = D.nc_tracts_geometry().to_crs(32119)
    keep = dict(zip(base.attributes["GEOID"], base.values / 1000))
    g = g[g["GEOID"].isin(keep)].copy()
    g["value"] = g["GEOID"].map(keep)
    county_of = dict(zip(base.attributes["GEOID"], base.attributes["county"]))
    g["county_code"] = g["GEOID"].map(county_of)
    cvals = dict(zip(cty.unit_keys, cty.values / 1000))
    gc = g.dissolve("county_code").reset_index()
    gc["geometry"] = gc.buffer(300).buffer(-300)
    gc["value"] = gc["county_code"].map(cvals)
    vmin, vmax = np.nanpercentile(g["value"], [2, 98])

    fig = Figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=(0.75, 1))
    for k, (frame, title) in enumerate([(g, "(a) Census tracts (n = 2,628)"),
                                         (gc, "(b) Counties (n = 100)")]):
        ax = fig.add_subplot(gs[0, k])
        frame.plot(column="value", ax=ax, cmap="viridis", vmin=vmin, vmax=vmax, linewidth=0.05,
                   edgecolor="white")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_axis_off()
    import matplotlib as mpl
    cax = fig.add_axes((0.25, 0.60, 0.50, 0.014))
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap="viridis"),
                      cax=cax, orientation="horizontal")
    cb.set_label("Median household income ($ thousands)", fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[1, :])
    style(ax)
    bins = np.linspace(min(min(v) for v in ens.values()) - 0.01, max(max(v) for v in ens.values()) + 0.01, 40)
    stats = {}
    for (name, vals), col in zip(ens.items(), ("#9aa3b5", "#2166ac")):
        vals = np.array(vals)
        z = (observed - vals.mean()) / vals.std(ddof=1)
        stats[name] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "z": float(z),
                       "share_below": float(np.mean(vals <= observed)), "min": float(vals.min()),
                       "max": float(vals.max())}
        ax.hist(vals, bins=bins, color=col, alpha=0.6, edgecolor="white",
                label=f"{name} (mean {vals.mean():.2f}, z = {z:.1f})")
    ax.axvline(observed, color=ALERT, lw=1.6)
    ax.text(observed, ax.get_ylim()[1] * 0.95, f" counties: {observed:.2f}", color=ALERT, fontsize=8, va="top")
    ax.set_xlabel("Share of tract-level variance retained in 100 zones (population-weighted)",
                  fontsize=8.5, color=INK)
    ax.set_ylabel("Random zonings", fontsize=8.5, color=INK)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title(f"(c) Counties against {n_zonings} random zonings of each kind", fontsize=9, color=INK, loc="left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.10, hspace=0.75, wspace=0.05)
    save(fig, "fig5_counties.png")
    return {"observed": float(observed), **stats}


# ------------------------------------------------------------------ Figure 6
def lisa_maps(permutations=999, seed=0):
    rep = D.nc_tracts("pct_broadband")
    g = D.nc_tracts_geometry().to_crs(32119)
    order = {gid: i for i, gid in enumerate(rep.attributes["GEOID"])}
    g = g[g["GEOID"].isin(order)].copy()
    g["i"] = g["GEOID"].map(order)
    specs = [("knn:6", "k-nearest neighbours (k = 6)"), ("queen", "Queen contiguity"),
             ("hops:2", "Network, two hops")]
    fig = Figure(figsize=(7.2, 2.0))
    axes = fig.subplots(1, 3)
    stats = {}
    for k, ((spec, name), ax) in enumerate(zip(specs, axes)):
        A = induced_adjacency(rep, spec)
        gI = sg.morans_i(rep.values, A, 0).I
        loc = local_moran(rep.values, A, permutations, 0.05, seed + 100 + k, fdr=True)
        g["cl"] = loc.cluster[g["i"].to_numpy()]
        for code in (0, 2, 4, 3, 1):
            sub = g[g["cl"] == code]
            if len(sub):
                sub.plot(ax=ax, color=LISA[code], linewidth=0.03, edgecolor="white")
        ax.set_title(f"({'abc'[k]}) {name}\nI = {gI:.2f}; {int(np.sum(loc.cluster > 0))} tracts in clusters",
                     fontsize=8, color=INK, loc="left")
        ax.set_axis_off()
        stats[spec] = loc.cluster
    from matplotlib.patches import Patch
    fig.legend([Patch(color=LISA[c]) for c in (1, 3, 2, 4, 0)], [LISA_NAMES[c] for c in (1, 3, 2, 4, 0)],
               loc="lower center", ncol=5, frameon=False, fontsize=7.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.13, wspace=0.05)
    save(fig, "fig6_broadband_lisa.png")
    c = [stats[s] for s in ("knn:6", "queen", "hops:2")]
    sig = (c[0] > 0) | (c[1] > 0) | (c[2] > 0)
    same = (c[0] == c[1]) & (c[1] == c[2])
    return {"significant": {s: int(np.sum(v > 0)) for s, v in stats.items()},
            "HH": {s: int(np.sum(v == 1)) for s, v in stats.items()},
            "LL": {s: int(np.sum(v == 3)) for s, v in stats.items()},
            "all_three_same_all": float(np.mean(same)), "all_three_same_significant": float(np.mean(same[sig]))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-zonings", type=int, default=200)
    ap.add_argument("--only", default="123456")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for old in ("fig4_counties.png", "fig5_broadband_lisa.png"):
        if os.path.exists(os.path.join(OUT, old)):
            os.remove(os.path.join(OUT, old))
    if "1" in a.only:
        workflow()
    if "2" in a.only:
        benchmark()
    if "3" in a.only:
        trajectories()
    if "4" in a.only:
        print("semivariogram:", semivariogram_figure())
    if "5" in a.only:
        print("counties:", counties(a.n_zonings))
    if "6" in a.only:
        print("lisa:", lisa_maps())
