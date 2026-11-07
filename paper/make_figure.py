"""Build the figure for paper.md: Moran's I under aggregation for a field with
one correlation length (A) and with two (B), each against the change-of-support
null fitted at the finest scale.  Run: python paper/make_figure.py"""
import os

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import scalegeom as sg

SCALES = [1, 2, 4, 5, 8, 10, 16]
INK, BAND, ALERT, MUTED = "#1f2a44", "#c9ced8", "#b42318", "#6b7280"

fig = Figure(figsize=(7.2, 3.0), dpi=300)
FigureCanvasAgg(fig)
axes = fig.subplots(1, 2, sharey=True)
fields = [("A  one correlation length (12 cells)", sg.gaussian_random_field((160, 160), 12, seed=7)),
          ("B  two correlation lengths (3 and 30 cells)", sg.mixed_scale_field((160, 160), (3, 30), seed=7))]
for ax, (title, f) in zip(axes, fields):
    fam = sg.raster_family(f, SCALES, zoning=None)
    traj = sg.ScaleProtocol(fam, "knn:8", permutations=0).trajectory(include_zoning=False)
    res = sg.information_loss_null(fam, "knn:8", "auto", n_sim=100, seed=1)
    s = np.array(traj["s"], float)
    y = np.array(traj["moran_I"], float)
    lo, hi = np.array(res.table["moran_band_lo"]), np.array(res.table["moran_band_hi"])
    out = np.array([abs(z) > res.width for z in res.table["z_moran"]])
    ax.fill_between(s, lo, hi, color=BAND, alpha=0.7, linewidth=0, label="information-loss null (±2.5 SD)")
    ax.plot(s, y, color=INK, linewidth=1.8, marker="o", markersize=4, label="observed")
    ax.scatter(s[out], y[out], s=70, facecolors="none", edgecolors=ALERT, linewidths=1.3, zorder=4,
               label="outside the band")
    ax.set_title(title, fontsize=8, loc="left", color=INK)
    ax.set_xlabel("Block size s (cells)", fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#eceef2", linewidth=0.5)
axes[0].set_ylabel("Moran's I (k = 8)", fontsize=8, color=INK)
axes[1].legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "figure1.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(out)
