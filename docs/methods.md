# Methods and reproducibility notes

This document records how scalegeom implements the protocol of Vaz (2026), *Scale-Conditioned Geometry: How Measurement Resolution Induces Distance, Proximity, and Spatial Dependence*, Geographical Analysis 58: e70061 ([doi:10.1111/gean.70061](https://doi.org/10.1111/gean.70061)), with the choices a reviewer or user needs in order to reproduce or question a result.

## Representations and families

A representation is a set of units with values, centroids, a measure (weights summing to one) and, where available, queen and rook contiguity. A family is an ordered list of representations of one dataset from fine to coarse, plus optional variants at the same scale. Every representation stores, for each base unit, the index of the unit it was aggregated into. This membership array is what lets simulated fields pass through exactly the same aggregation as the data.

Each representation carries a scale s and a zoning label. For block and grid families the zoning is the offset of the aggregation lattice (the paper's ζ). For administrative or random zonings there is no lattice, and the zoning is identified by the partition itself, which the membership array records.

Aggregated values are measure-weighted means of base values. This is the right operator for continuous fields and for intensive quantities whose weights are their denominators. It is not the right operator for medians or other non-additive statistics. In the bundled North Carolina data, the aggregated median income of a county is the population-weighted mean of its tract medians, not the county median, and the reports say so.

## Induced metrics and neighbourhoods

A metric specification names how neighbours are defined on each representation:

| Specification | Neighbourhood |
|---|---|
| `knn:k` | the k nearest centroids under Euclidean distance |
| `band:r` | centroids within distance r, in units of the domain extent |
| `queen`, `rook` | units sharing a vertex or an edge |
| `hops:h` | units within h steps on the queen graph (`hops:h:rook` for the rook graph) |
| `matrix:k` | the k nearest units under a supplied distance matrix, for example travel time |

Strictly, `knn` and `hops` are neighbourhood rules built from a metric. Weights are row-standardized. Euclidean distances use projected coordinates as supplied; the package does not reproject, and geographic coordinates give distances in degrees. All distances are divided by one domain extent shared by the whole family, so bands and lags mean the same thing at every scale.

## Diagnostics

- Moran's I and local Moran follow esda's conventions, including conditional permutation for local statistics and folded pseudo p-values. On the same weights, global I agrees with esda to machine precision and local quadrants agree exactly (tests/test_core.py).
- Semivariograms and correlation profiles use one lag grid for all scales (15 bins from 0 to 0.5 of the domain extent by default, right-closed). Semivariances standardize values by the base mean and standard deviation, so levels are comparable across scales. Correlation profiles standardize within each scale.
- The range proxy is the smallest lag at which the correlation profile falls below τ (0.1 by default), interpolated linearly between bin centres.
- Variance retained is the variance of unit values weighted by each unit's measure, divided by the weighted variance of the base. With measure-weighted means as the aggregation operator, this is the between-unit share of the base variance, between 0 and 1. Trajectories also report the unweighted ratio (`var_retained_unweighted`), which counts every unit once; the nulls take `weighted=False` for it.

## Change-of-support null

1. **Fit.** The empirical semivariogram of the base is computed on the lag grid. Each bin enters the fit at its mean pair distance, not its centre, with Cressie (1985) weights N(h) / γ(h)². The first bin is replaced by the exact semivariance of all pairs formed by each unit and its ten nearest neighbours that fall inside it, in eight bins of equal pair count by distance. Random pair sampling resolves short lags poorly, and on irregular supports the first bin can be much wider than the unit spacing (26 km against a median neighbour distance of 3 km for North Carolina tracts). Two earlier versions failed calibration: fitting at bin centres produced a spurious nugget, and a single averaged nearest-neighbour value overestimated short correlation lengths on irregular supports by about 30%. Near-neighbour pairs give every unit equal weight; using all pairs within the bin instead calibrated worse in simulation, because close pairs come mostly from dense areas.
2. **Model choice.** Gaussian, exponential and spherical models are each fitted with and without a nugget. The six candidates are compared by m log(WRSS / m) + 2p, where WRSS is the weighted residual sum of squares, m the number of lags and p the number of parameters. Every candidate's criterion is reported. On the bundled elevation data a Gaussian model, too smooth for terrain, made the null reject at five or six scales, while the selected exponential model did not.
3. **Simulation.** Lattices are simulated by circulant embedding. Points and polygons with up to 3,000 units use one Cholesky factor per test; larger sets are simulated on a covering grid and sampled at the nearest cell.
4. **Parameter uncertainty.** By default each simulation draws a field from the fitted model, refits the same model form to it, and simulates the trajectory from the refitted parameters (a parametric bootstrap). This widens the band by the uncertainty of the fitted parameters. Without it the test is too liberal for rough fields (see Calibration).
5. **Test.** Variance is tested as the share of base variance retained, and Moran's I as the change from the base, I(s) − I(base). The base scale is in the band by construction. A scale exits the descriptive band when either diagnostic lies more than 2.5 simulation standard deviations from the simulated mean. The formal decision is a simultaneous test across all scales and both diagnostics: the statistic is the largest absolute standardized departure, and its reference distribution is that statistic computed for every simulated trajectory (`result.simultaneous`, level 0.05 by default).
6. **Prediction.** The analytic share of variance retained, 1 − γ̄(v, v) / γ̄(D, D) from Krige's relation, is reported next to the Monte Carlo mean as a check.

A departure from the band means the trajectory is inconsistent with information loss under a single stationary structure fitted at the base. Possible causes include structure at several scales, non-stationarity, anisotropy and edge effects. The test does not separate these.

### Calibration

Two scripts estimate the false-rejection rate and the power of the test. In each run the covariance model is selected from the simulated data and the test uses 100 null simulations, as in practice.

`examples/null_calibration.py` uses a regular 96 x 96 lattice aggregated into blocks of 2, 4, 8 and 16 cells, with fields from a stationary Gaussian or exponential covariance and no nugget (100 fields each). `examples/null_calibration_irregular.py` uses the 2,628 North Carolina tract centroids, weighted by population, aggregated to 5, 10, 20 and 40 km grids (cells hold 1 to 242 tracts), with two null scenarios (Gaussian and exponential, length 8 km) and three alternatives.

| Support | Scenario | Any scale outside 2.5 SD | Simultaneous test, 5% |
|---|---|---|---|
| Regular lattice | Null, Gaussian | 1% | 1% |
| Regular lattice | Null, exponential | 0% | 0% |
| Tract centroids | Null, Gaussian | 8% | 7% |
| Tract centroids | Null, exponential | 7% | 7% |
| Tract centroids | Two scales, 20% of variance at 64 km | 78% | 78% |
| Tract centroids | Two scales, 10% at 64 km | 50% | 47% |
| Tract centroids | Linear trend, 10% | 97% | 96% |

Raw results are in `docs/calibration.csv` and `docs/calibration_irregular.csv`. With parameters treated as known (an earlier calibration on the lattice), exponential fields were rejected at s = 2 in 24 to 28% of fields; the parametric bootstrap removes this. With a single averaged nearest-neighbour value in place of the short-lag bins, an 8 km Gaussian field on the tract centroids was fitted with a median length of 10.3 km and the test rejected about half of the null fields.

### Zoning null

For nested supports, each observed support is compared with random zonings of the base into the same number of zones. By default (`method="size_matched"`) the zones are contiguous and follow the observed distribution of zone sizes: they are grown one at a time from the largest target to the smallest, each from a random unassigned seed, by adding random unassigned neighbours on the queen graph (islands are first linked to their nearest unit). Units left at the end join an adjacent zone, so sizes are matched approximately. The earlier nearest-seed zonings, with zones of roughly equal size, remain available (`method="voronoi"`). The choice matters: North Carolina counties, which range from 1 to 301 tracts, look unusual against equal-size zonings (z = −5.5 for population-weighted variance retained) and typical against size-matched contiguous zonings (z near 0).

## Inter-scale distortion

The paper's bi-Lipschitz condition c d_s(x, y) ≤ d_s'(A x, A y) ≤ C d_s(x, y) cannot hold with c > 0 for a many-to-one aggregation map, since two units in the same coarse unit are sent to one point. scalegeom reports empirical constants on pairs the map separates: coarse units matched to their nearest fine unit (`method="centroid"`, as in Section 9.7 of the paper, minimum and maximum ratio), or random pairs of fine units in distinct coarse units (`method="pairs"`, with 1% and 99% quantiles for irregular supports).

## Representation space (FGW)

Each representation is a metric-measure space with features, compared by fused Gromov-Wasserstein distance (POT, α = 0.5, square loss). The solver starts from the optimal transport plan between unit locations, which in our tests made repeated estimates much less variable than the default start; the problem is non-convex, so results are local optima. Representations with more than `max_nodes` units (800 by default) are subsampled with probability proportional to their measure. Subsampling alone produces a positive distance, so each subsampled representation's floor F is estimated from independent subsamples and the adjusted distance FGW(a, b) − (F_a + F_b) / 2 is reported with the raw value. The adjustment is empirical: it can be slightly negative and is not a metric.

## Reproducing the paper

`examples/reproduce_vaz2026.py` uses different random seeds from the original study. With 30 realizations, Moran's I at s = 16 is 0.320 against the published 0.297; the published Monte Carlo standard deviation at that scale is 0.083, so the difference is within sampling variation.
