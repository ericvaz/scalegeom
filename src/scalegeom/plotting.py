"""Figures for scale-conditioned analyses.

Uses matplotlib.figure.Figure directly (no pyplot), so figures render
headless on servers and in batch jobs."""
from __future__ import annotations

import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure

INK = "#1f2a44"
MUTED = "#6b7280"
BAND = "#c9ced8"
ALERT = "#b42318"
CLUSTER_COLOURS = {0: "#e5e7eb", 1: "#0b2e6b", 2: "#9ec5e8", 3: "#4a90c9", 4: "#e8907a"}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color="#eceef2", linewidth=0.6)
    ax.set_axisbelow(True)


def _seq(n):
    import matplotlib
    cmap = matplotlib.colormaps["viridis"]
    return [cmap(x) for x in np.linspace(0.0, 0.9, max(n, 1))]


def trajectory_figure(traj, null=None, column="moran_I", ylabel="Moran's I"):
    """Diagnostic trajectory along the scale path, optionally over the
    change-of-support null band (cf. Vaz 2026, Figures 2 and 7)."""
    rows = [r for r in traj.rows if r.get("zoning") in (None, "", "0", 0, False)]
    s = np.array([r["s"] for r in rows], float)
    y = np.array([r[column] for r in rows], float)
    fig = Figure(figsize=(6.0, 3.6), dpi=150)
    ax = fig.add_subplot(111)
    _style(ax)
    if null is not None:
        key = "moran" if column == "moran_I" else "var"
        lo = np.array(null.table[f"{key}_band_lo"], float)
        hi = np.array(null.table[f"{key}_band_hi"], float)
        ax.fill_between(s, lo, hi, color=BAND, alpha=0.6, linewidth=0,
                        label=f"information-loss null (±{null.width:g} SD, n={null.n_sim})")
        ax.plot(s, 0.5 * (lo + hi), color=MUTED, linewidth=1, linestyle="--")
        zkey = "z_moran" if key == "moran" else "z_var"
        out = np.array([abs(v) > null.width for v in null.table[zkey]])
        out[0] = False
        if out.any():
            ax.scatter(s[out], y[out], s=90, facecolors="none", edgecolors=ALERT, linewidths=1.5,
                       zorder=4, label="outside the null band")
    ax.plot(s, y, color=INK, linewidth=2, marker="o", markersize=5, label="observed", zorder=3)
    zr = [r for r in traj.rows if r.get("zoning") not in (None, "", "0", 0, False)]
    if zr:
        ax.scatter([r["s"] for r in zr], [r[column] for r in zr], marker="^", s=30, color="#4a90c9",
                   zorder=3, label="shifted lattice (zoning)")
    ax.set_xlabel("Scale s", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def profiles_figure(profiles, which="gamma", tau=None):
    """Semivariograms (which='gamma') or correlation profiles (which='rho')
    for every scale, on the common normalized lag grid (cf. Figures 3 and 4)."""
    fig = Figure(figsize=(6.0, 3.6), dpi=150)
    ax = fig.add_subplot(111)
    _style(ax)
    labels = list(profiles)
    for colour, lab in zip(_seq(len(labels)), labels):
        p = profiles[lab]
        ax.plot(p["centers"], p[which], color=colour, linewidth=1.6, marker="o", markersize=3, label=lab)
    if tau is not None and which == "rho":
        ax.axhline(tau, color=ALERT, linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[1], tau, f" τ = {tau:g}", color=ALERT, va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("Normalized lag distance h", color=INK)
    ax.set_ylabel("Semivariance γ(h)" if which == "gamma" else "Pairwise correlation ρ(h)", color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def space_figure(space):
    """Representations along the scale path by cumulative FGW distance, with
    zoning variants branching off at fixed s (cf. Figure 6)."""
    path = space["path"].rows
    fig = Figure(figsize=(6.4, 3.0), dpi=150)
    ax = fig.add_subplot(111)
    xs = [0.0] + [r["cumulative_fgw"] for r in path]
    labels = [path[0]["step"].split(" -> ")[0]] + [r["step"].split(" -> ")[1] for r in path]
    svals = [path[0]["s_from"]] + [r["s_to"] for r in path]
    cols = _seq(len(xs))
    ax.plot(xs, [0] * len(xs), color="#4a5a85", linewidth=1.2, zorder=1)
    zon = {}
    for r in space["zoning"].rows:
        zon.setdefault(r["s"], []).append(r)
    ymin = 0.0
    for x, lab, s, c in zip(xs, labels, svals, cols):
        ax.scatter([x], [0], s=90, color=c, edgecolors=INK, linewidths=0.8, zorder=3)
        ax.text(x, 0.08, lab, ha="center", va="bottom", fontsize=7, color=INK)
        for z in zon.get(s, []):
            dz = max(z["fgw_debiased"], 0.0)
            ax.plot([x, x], [0, -dz], color=ALERT, linewidth=1, zorder=2)
            ax.scatter([x], [-dz], marker="^", s=30, facecolors="none", edgecolors=ALERT, zorder=3)
            ymin = min(ymin, -dz)
    span = max(xs[-1], abs(ymin), 1e-6)
    # stagger labels of nodes that sit close together on the path
    level, last_x = [], {}
    for x in xs:
        k = 0
        while k in last_x and x - last_x[k] < 0.06 * span:
            k += 1
        last_x[k] = x
        level.append(k)
    top = max(level) + 1
    ax.set_ylim(min(ymin, -0.1 * span) * 1.3, (0.2 + 0.09 * top) * span)
    for t, k in zip(ax.texts, level):
        t.set_position((t.get_position()[0], (0.06 + 0.09 * k) * span))
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("Cumulative debiased FGW distance along the scale path (offshoots: zoning displacement)", color=INK)
    fig.tight_layout()
    return fig


def family_figure(family, max_panels: int = 5):
    """Maps of the field at several representations of the family (cf. Figure 1)."""
    reps = family.path
    if len(reps) > max_panels:
        idx = np.unique(np.round(np.linspace(0, len(reps) - 1, max_panels)).astype(int))
        reps = [reps[i] for i in idx]
    vals = np.concatenate([r.values for r in reps])
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    fig = Figure(figsize=(2.0 * len(reps) + 0.6, 2.4), dpi=150)
    axes = fig.subplots(1, len(reps), squeeze=False)[0]
    im = None
    for ax, rep in zip(axes, reps):
        xy = family.normalized(rep)
        if hasattr(rep, "lattice_rc"):
            r, c, nbr, nbc = rep.lattice_rc
            grid = np.full((nbr, nbc), np.nan)
            grid[r, c] = rep.values
            im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        elif family.kind == "raster" and rep is family.base:
            lat = family.lattice
            grid = np.full(lat["shape"], np.nan)
            grid[lat["rows"], lat["cols"]] = rep.values
            im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        else:
            size = max(1.0, 4000.0 / max(rep.n, 1))
            im = ax.scatter(xy[:, 0], xy[:, 1], c=rep.values, s=size, marker="s", cmap="viridis",
                            vmin=vmin, vmax=vmax, linewidths=0)
            ax.set_aspect("equal")
        ax.set_title(f"{rep.label}\n(n = {rep.n:,})", fontsize=7, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(BAND)
    if im is not None:
        cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
    return fig


def cluster_figure(rep, results: dict, family=None):
    """LISA cluster maps (unit centroids) for each induced metric (cf. Figure 8)."""
    specs = list(results)
    xy = family.normalized(rep) if family is not None else rep.coords
    fig = Figure(figsize=(2.6 * len(specs), 2.9), dpi=150)
    axes = fig.subplots(1, len(specs), squeeze=False)[0]
    size = max(2.0, 6000.0 / max(rep.n, 1))
    names = {0: "not significant", 1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    for ax, spec in zip(axes, specs):
        res = results[spec]
        for k in (0, 2, 3, 4, 1):
            m = res["local"].cluster == k
            ax.scatter(xy[m, 0], xy[m, 1], s=size, color=CLUSTER_COLOURS[k], linewidths=0,
                       label=names[k] if spec == specs[0] else None)
        ax.set_title(f"{res['name']}\nI = {res['I']:.3f}, {res['local'].n_significant} clusters",
                     fontsize=7, color=INK)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=7, markerscale=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig
