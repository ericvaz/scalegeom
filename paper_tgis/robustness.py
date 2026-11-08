"""Robustness analyses for the Transactions in GIS article.

    python paper_tgis/robustness.py [--n-zonings 200] [--n-sim 100]

Responds to points raised in pre-submission review:
  A  variance retained with consistent population weights at both supports
  B  a zoning null that matches the counties' tract-count distribution and
     keeps every zone contiguous on the queen graph
  C  the change-of-support null for income on grids after removing a smooth
     regional trend
  D  local clusters of broadband with false discovery rate control, and
     agreement measured on significant tracts only
Writes CSV tables to paper_tgis/robustness/.
"""
import argparse
import csv
import os
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

import scalegeom as sg
from scalegeom import datasets as D
from scalegeom.diagnostics import local_moran, morans_i_fast
from scalegeom.metrics import induced_adjacency, row_standardize
from scalegeom.representation import aggregate_by_labels, grid_family, voronoi_zoning

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "robustness")


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(v)) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    print("wrote", name)


# ------------------------------------------------------------------ helpers
def weighted_retained(values, weights, labels):
    """Between-zone share of the weighted variance: sum_g W_g (m_g - m)^2 / sum_i w_i (y_i - m)^2."""
    w = weights / weights.sum()
    m = np.sum(w * values)
    total = np.sum(w * (values - m) ** 2)
    W = np.bincount(labels, weights=w)
    mg = np.bincount(labels, weights=w * values) / np.where(W > 0, W, 1)
    return float(np.sum(W * (mg - m) ** 2) / total)


def unweighted_retained(values, weights, labels):
    W = np.bincount(labels, weights=weights)
    mg = np.bincount(labels, weights=weights * values) / np.where(W > 0, W, 1)
    return float(np.var(mg[W > 0]) / np.var(values))


def contiguous_size_matched(adj, sizes, rng, max_tries=50):
    """Random partition of a graph into len(sizes) contiguous zones whose
    sizes follow `sizes` as closely as the graph allows. Zones grow from
    random seeds by adding random frontier units; units left over at the
    end join an adjacent zone. Disconnected islands join the nearest zone
    through the graph's components."""
    n = adj.shape[0]
    A = adj.tocsr()
    target = np.array(sorted(sizes, reverse=True))
    k = len(target)
    for _ in range(max_tries):
        lab = -np.ones(n, dtype=int)
        seeds = rng.choice(n, k, replace=False)
        lab[seeds] = np.arange(k)
        count = np.ones(k, dtype=int)
        frontier = [set(A.indices[A.indptr[s]:A.indptr[s + 1]]) for s in seeds]
        active = list(range(k))
        while active:
            g = active[rng.integers(len(active))]
            cand = [u for u in frontier[g] if lab[u] < 0]
            if not cand or count[g] >= target[g]:
                active.remove(g)
                continue
            u = cand[rng.integers(len(cand))]
            lab[u] = g
            count[g] += 1
            frontier[g].update(A.indices[A.indptr[u]:A.indptr[u + 1]])
        # leftovers: attach to an adjacent zone, repeatedly
        for _ in range(n):
            left = np.flatnonzero(lab < 0)
            if len(left) == 0:
                break
            changed = False
            for u in left:
                nb = A.indices[A.indptr[u]:A.indptr[u + 1]]
                nb = nb[lab[nb] >= 0]
                if len(nb):
                    lab[u] = lab[rng.choice(nb)]
                    changed = True
            if not changed:
                break
        if np.all(lab >= 0):
            return lab
    return None


def contiguous_size_matched_seq(adj, sizes, rng):
    """Like contiguous_size_matched, but zones are grown one at a time from
    the largest target to the smallest, each from a random unassigned seed,
    so that large zones reach their target sizes. Units that cannot be
    reached join an adjacent zone at the end."""
    n = adj.shape[0]
    A = adj.tocsr()
    target = np.array(sorted(sizes, reverse=True))
    lab = -np.ones(n, dtype=int)
    for g, size in enumerate(target):
        free = np.flatnonzero(lab < 0)
        if len(free) == 0:
            break
        s = rng.choice(free)
        lab[s] = g
        members = [s]
        frontier = set(A.indices[A.indptr[s]:A.indptr[s + 1]])
        while len(members) < size:
            cand = [u for u in frontier if lab[u] < 0]
            if not cand:
                break
            u = cand[rng.integers(len(cand))]
            lab[u] = g
            members.append(u)
            frontier.update(A.indices[A.indptr[u]:A.indptr[u + 1]])
    for _ in range(n):
        left = np.flatnonzero(lab < 0)
        if len(left) == 0:
            break
        for u in left:
            nb = A.indices[A.indptr[u]:A.indptr[u + 1]]
            nb = nb[lab[nb] >= 0]
            if len(nb):
                lab[u] = lab[rng.choice(nb)]
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def bridge_islands(adj, coords):
    """Connect every component of the contiguity graph to its nearest unit in
    another component, so that region growing can reach islands."""
    from scipy.spatial import cKDTree
    A = adj.tolil(copy=True)
    while True:
        ncomp, comp = connected_components(A.tocsr(), directed=False)
        if ncomp == 1:
            return A.tocsr()
        big = np.argmax(np.bincount(comp))
        for c in range(ncomp):
            if c == big:
                continue
            inside = np.flatnonzero(comp == c)
            outside = np.flatnonzero(comp != c)
            d, j = cKDTree(coords[outside]).query(coords[inside])
            i = inside[np.argmin(d)]
            o = outside[j[np.argmin(d)]]
            A[i, o] = 1
            A[o, i] = 1
            break


def zone_contiguous_share(adj, lab):
    ok = 0
    k = lab.max() + 1
    for g in range(k):
        idx = np.flatnonzero(lab == g)
        sub = adj[idx][:, idx]
        ok += connected_components(sub, directed=False)[0] == 1
    return ok / k


# ------------------------------------------------------------------ A and B
def counties(n_zonings, seed):
    fam = D.nc_tracts_family("median_income")
    base, cty = fam.path[0], fam.path[1]
    y, w = base.values, base.mass
    codes = base.attributes["county"]
    _, clab = np.unique(codes, return_inverse=True)
    adj = bridge_islands(base.adjacency["queen"], base.coords)
    sizes = np.bincount(clab)
    rng = np.random.default_rng(seed)
    obs = {"unweighted": unweighted_retained(y, w, clab), "weighted": weighted_retained(y, w, clab),
           "contiguous_share": zone_contiguous_share(adj, clab)}
    ens = {"voronoi": [], "size_matched_contiguous": [], "size_matched_sequential": []}
    t0 = time.time()
    for r in range(n_zonings):
        z = voronoi_zoning(base, cty.n, seed + r)
        lab = z_labels(base, cty.n, seed + r)
        ens["voronoi"].append((unweighted_retained(y, w, lab), weighted_retained(y, w, lab),
                               np.std(np.bincount(lab)), zone_contiguous_share(adj, lab)))
        lab2 = contiguous_size_matched(adj, sizes, rng)
        ens["size_matched_contiguous"].append((unweighted_retained(y, w, lab2), weighted_retained(y, w, lab2),
                                               np.std(np.bincount(lab2)), zone_contiguous_share(adj, lab2)))
        lab3 = contiguous_size_matched_seq(adj, sizes, rng)
        ens["size_matched_sequential"].append((unweighted_retained(y, w, lab3), weighted_retained(y, w, lab3),
                                               np.std(np.bincount(lab3)), zone_contiguous_share(adj, lab3)))
    rows = []
    for name, vals in ens.items():
        a = np.array(vals)
        for j, est in enumerate(("unweighted", "weighted")):
            rows.append({"ensemble": name, "estimand": est, "observed": obs[est], "mean": a[:, j].mean(),
                         "sd": a[:, j].std(ddof=1), "min": a[:, j].min(), "max": a[:, j].max(),
                         "z": (obs[est] - a[:, j].mean()) / a[:, j].std(ddof=1),
                         "share_below_observed": float(np.mean(a[:, j] <= obs[est])),
                         "zone_size_sd_mean": a[:, 2].mean(), "contiguous_zone_share_mean": a[:, 3].mean(),
                         "n": n_zonings})
    rows.append({"ensemble": "counties", "estimand": "zone size SD", "observed": float(np.std(sizes)),
                 "mean": "", "sd": "", "min": float(sizes.min()), "max": float(sizes.max()), "z": "",
                 "share_below_observed": "", "zone_size_sd_mean": "",
                 "contiguous_zone_share_mean": obs["contiguous_share"], "n": ""})
    print(f"  zonings: {time.time() - t0:.0f} s")
    write("A_B_county_zoning.csv", rows)
    np.savez(os.path.join(OUT, "A_B_county_zoning_samples.npz"),
             **{k: np.array(v) for k, v in ens.items()}, observed=np.array([obs["unweighted"], obs["weighted"]]))
    return rows


def z_labels(base, n_zones, seed):
    """The labels voronoi_zoning uses (same seeds, same assignment)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    seeds = rng.choice(base.n, n_zones, replace=False)
    return cKDTree(base.coords[seeds]).query(base.coords)[1]


# ------------------------------------------------------------------ C
def detrended_grids(n_sim, seed, degree=3):
    base = D.nc_tracts("median_income")
    xy = base.coords
    c = (xy - xy.mean(0)) / xy.std(0)
    cols = [np.ones(len(c))]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            cols.append(c[:, 0] ** (d - i) * c[:, 1] ** i)
    X = np.column_stack(cols)
    w = base.mass / base.mass.sum()
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], base.values * np.sqrt(w), rcond=None)[0]
    trend = X @ beta
    r2 = 1 - np.sum(w * (base.values - trend) ** 2) / np.sum(w * (base.values - np.sum(w * base.values)) ** 2)
    rows = []
    variants = (("original", base.values), (f"residual (cubic trend, R2 = {r2:.2f})", base.values - trend),
                ("log income", np.log(base.values)))
    for label, vals in variants:
        b = base.with_values(vals)
        fam = grid_family(b, [5000.0, 10000.0, 20000.0, 40000.0], zoning=None)
        t0 = time.time()
        res = sg.information_loss_null(fam, "knn:6", "auto", n_sim=n_sim, seed=seed)
        for r in res.table.rows:
            rows.append({"variable": label, "model": res.model.model, "length": res.model.length,
                         "support": r["label"], "var_observed": r["var_observed"], "var_null_mean": r["var_null_mean"],
                         "z_var": r["z_var"], "moran_observed": r["moran_observed"],
                         "moran_null_mean": r["moran_null_mean"], "z_moran": r["z_moran"], "exits": r["exits"],
                         "simultaneous_p": res.simultaneous["p"], "simultaneous_stat": res.simultaneous["statistic"],
                         "simultaneous_crit": res.simultaneous["critical"], "seconds": time.time() - t0})
        print(f"  {label}: {res.exits} exits, model {res.model.model}, {time.time() - t0:.0f} s")
    write("C_detrended_grids.csv", rows)
    return rows


# ------------------------------------------------------------------ E
def local_variability():
    """Semivariance of nearest-neighbour pairs in dense and sparse areas.

    Tracts are split at the median distance to their nearest neighbour. Under
    stationarity the two halves would differ only through the distance
    itself, with the sparse half larger; a larger value in the dense half
    shows that local variability differs between urban and rural areas."""
    from scipy.spatial import cKDTree
    rows = []
    for var in ("median_income", "pct_broadband"):
        base = D.nc_tracts(var, contiguity=False)
        z = (base.values - base.values.mean()) / base.values.std()
        d, j = cKDTree(base.coords).query(base.coords, 2)
        dn, jn = d[:, 1], j[:, 1]
        g = 0.5 * (z - z[jn]) ** 2
        med = np.median(dn)
        for name, m in (("dense half", dn <= med), ("sparse half", dn > med)):
            rows.append({"variable": var, "group": name, "tracts": int(m.sum()),
                         "mean_nn_distance_km": float(dn[m].mean() / 1000),
                         "nn_semivariance": float(g[m].mean())})
    write("E_local_variability.csv", rows)
    return rows


# ------------------------------------------------------------------ F and G
def distance_controlled(bands_km=(0.5, 1.5, 2.5, 4.0, 6.0)):
    """Semivariance of tract pairs in dense and sparse areas at the same
    separation. Tracts are split at the median distance to their nearest
    neighbour; a pair counts as dense (sparse) when both tracts are in the
    dense (sparse) half. Pairs are grouped into common distance bands."""
    from scipy.spatial import cKDTree
    rows = []
    for var in ("median_income", "pct_broadband"):
        base = D.nc_tracts(var, contiguity=False)
        z = (base.values - base.values.mean()) / base.values.std()
        tree = cKDTree(base.coords)
        dn = tree.query(base.coords, 2)[0][:, 1]
        dense = dn <= np.median(dn)
        pairs = tree.query_pairs(bands_km[-1] * 1000, output_type="ndarray")
        a, b = pairs[:, 0], pairs[:, 1]
        d = np.hypot(*(base.coords[a] - base.coords[b]).T) / 1000
        g = 0.5 * (z[a] - z[b]) ** 2
        for lo, hi in zip(bands_km[:-1], bands_km[1:]):
            inb = (d > lo) & (d <= hi)
            for name, m in (("dense", dense[a] & dense[b]), ("sparse", ~dense[a] & ~dense[b])):
                k = inb & m
                rows.append({"variable": var, "band_km": f"{lo:g} to {hi:g}", "group": name, "pairs": int(k.sum()),
                             "mean_distance_km": float(d[k].mean()) if k.any() else np.nan,
                             "semivariance": float(g[k].mean()) if k.sum() >= 30 else np.nan})
    write("F_distance_controlled.csv", rows)
    return rows


def zoning_size_match(n_zonings=200, seed=3000):
    """How closely the tract-count-matched zonings reproduce the counties'
    tract counts: zones sorted by size and compared rank by rank with the
    sorted county counts."""
    from scalegeom.representation import _contiguity_graph
    fam = D.nc_tracts_family("median_income")
    base, cty = fam.path
    target = np.sort(np.bincount(cty.members))[::-1]
    G = _contiguity_graph(base)
    dev, exact, nzones = [], [], []
    for r in range(n_zonings):
        z = sg.contiguous_zoning(base, target, seed + r, graph=G)
        got = np.sort(np.bincount(z.members))[::-1]
        nzones.append(len(got))
        k = min(len(got), len(target))
        d = np.abs(got[:k] - target[:k])
        dev.append(d)
        exact.append(np.mean(d == 0))
    dev = np.concatenate(dev)
    rows = [{"zonings": n_zonings, "zones_per_zoning_min": int(min(nzones)), "zones_per_zoning_max": int(max(nzones)),
             "abs_dev_median": float(np.median(dev)), "abs_dev_p95": float(np.percentile(dev, 95)),
             "abs_dev_max": int(dev.max()), "share_exact": float(np.mean(exact)),
             "share_within_2": float(np.mean(dev <= 2))}]
    write("G_zoning_size_match.csv", rows)
    return rows


# ------------------------------------------------------------------ D
def bh(p, q=0.05):
    p = np.asarray(p)
    n = len(p)
    o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    k = np.max(np.flatnonzero(ok)) + 1 if ok.any() else 0
    out = np.zeros(n, bool)
    out[o[:k]] = True
    return out


def lisa_fdr(permutations=999, seed=0):
    rep = D.nc_tracts("pct_broadband")
    specs = ["knn:6", "queen", "hops:2"]
    res = {}
    for k, spec in enumerate(specs):
        A = induced_adjacency(rep, spec)
        loc = local_moran(rep.values, A, permutations, 0.05, seed + 100 + k)
        sig_fdr = bh(loc.p_sim, 0.05)
        res[spec] = {"p05": loc.cluster, "fdr": np.where(sig_fdr, loc.quadrant, 0)}
    rows = []
    for rule in ("p05", "fdr"):
        for spec in specs:
            c = res[spec][rule]
            rows.append({"rule": rule, "metric": spec, "significant": int(np.sum(c > 0)),
                         "HH": int(np.sum(c == 1)), "LL": int(np.sum(c == 3)),
                         "LH": int(np.sum(c == 2)), "HL": int(np.sum(c == 4))})
    agree = []
    for rule in ("p05", "fdr"):
        for i, a in enumerate(specs):
            for b in specs[i + 1:]:
                ca, cb = res[a][rule], res[b][rule]
                sig = (ca > 0) | (cb > 0)
                both = (ca > 0) & (cb > 0)
                row = {"rule": rule, "pair": f"{a} vs {b}", "all_tracts_same_label": float(np.mean(ca == cb)),
                       "significant_in_either_same_label": float(np.mean(ca[sig] == cb[sig])),
                       "same_label_given_both_significant": float(np.mean(ca[both] == cb[both])) if both.any() else np.nan}
                for q, name in ((1, "HH"), (3, "LL")):
                    A_, B_ = ca == q, cb == q
                    row[f"jaccard_{name}"] = float(np.sum(A_ & B_) / max(np.sum(A_ | B_), 1))
                agree.append(row)
        c3 = [res[s][rule] for s in specs]
        sig = (c3[0] > 0) | (c3[1] > 0) | (c3[2] > 0)
        agree.append({"rule": rule, "pair": "all three", "all_tracts_same_label":
                      float(np.mean((c3[0] == c3[1]) & (c3[1] == c3[2]))),
                      "significant_in_either_same_label": float(np.mean(((c3[0] == c3[1]) & (c3[1] == c3[2]))[sig])),
                      "same_label_given_both_significant": "", "jaccard_HH": "", "jaccard_LL": ""})
    write("D_lisa_counts.csv", rows)
    write("D_lisa_agreement.csv", agree)
    return rows, agree


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-zonings", type=int, default=200)
    ap.add_argument("--n-sim", type=int, default=100)
    ap.add_argument("--only", default="ABCD")
    a = ap.parse_args()
    if "A" in a.only or "B" in a.only:
        for r in counties(a.n_zonings, 3000):
            print(" ", r)
    if "C" in a.only:
        detrended_grids(a.n_sim, 0)
    if "E" in a.only:
        for r in local_variability():
            print(" ", r)
    if "F" in a.only:
        for r in distance_controlled():
            print(" ", r)
    if "G" in a.only:
        for r in zoning_size_match():
            print(" ", r)
    if "D" in a.only:
        c, g = lisa_fdr()
        for r in c + g:
            print(" ", r)
