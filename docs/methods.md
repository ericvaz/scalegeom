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
- Variance retained is the unweighted variance of unit values divided by that of the base.

## Change-of-support null

1. **Fit.** The empirical semivariogram of the base is computed on the lag grid. Each bin enters the fit at its mean pair distance, not its centre, with Cressie (1985) weights N(h) / γ(h)². The exact nearest-neighbour semivariance is added as the shortest lag, because random pair sampling resolves short lags poorly. Fitting at bin centres produced a spurious nugget that pushed even the base scale outside its own band.
2. **Model choice.** Gaussian, exponential and spherical models are each fitted with and without a nugget. The six candidates are compared by m log(WRSS / m) + 2p, where WRSS is the weighted residual sum of squares, m the number of lags and p the number of parameters. Every candidate's criterion is reported. On the bundled elevation data a Gaussian model, too smooth for terrain, made the null reject at five or six scales, while the selected exponential model did not.
3. **Simulation.** Lattices are simulated by circulant embedding. Points and polygons with up to 3,000 units use one Cholesky factor per test; larger sets are simulated on a covering grid and sampled at the nearest cell.
4. **Parameter uncertainty.** By default each simulation draws a field from the fitted model, refits the same model form to it, and simulates the trajectory from the refitted parameters (a parametric bootstrap). This widens the band by the uncertainty of the fitted parameters. Without it the test is too liberal for rough fields (see Calibration).
5. **Test.** Variance is tested as the share of base variance retained, and Moran's I as the change from the base, I(s) − I(base). The base scale is in the band by construction. A scale exits when either diagnostic lies more than 2.5 simulation standard deviations from the simulated mean. No multiplicity correction is applied across scales; the calibration below reports the any-scale rate.
6. **Prediction.** The analytic share of variance retained, 1 − γ̄(v, v) / γ̄(D, D) from Krige's relation, is reported next to the Monte Carlo mean as a check.

A departure from the band means the trajectory is inconsistent with information loss under a single stationary structure fitted at the base. Possible causes include structure at several scales, non-stationarity, anisotropy and edge effects. The test does not separate these.

### Calibration

`examples/null_calibration.py` simulates fields that satisfy the null (stationary, isotropic, Gaussian or exponential covariance, no nugget, 96 x 96 cells, blocks of 2 to 16) and records how often the test rejects.

| True model | Any scale | s = 2 | s = 4 | s = 8 | s = 16 |
|---|---|---|---|---|---|
| Gaussian (100 fields) | 1% | 0% | 0% | 0% | 1% |
| Exponential (100 fields) | 2% | 2% | 0% | 0% | 0% |

The covariance model was selected from the data each time: Gaussian in all 100 Gaussian fields, and exponential in 82 and spherical in 18 of the exponential fields. Each test used 60 null simulations. Raw results are in `docs/calibration.csv`.

With parameters treated as known, the exponential fields were rejected at s = 2 in 24 to 28% of fields, because the first aggregation step depends strongly on the estimated short-range covariance. The parametric bootstrap removes this.

### Zoning null

For nested supports, each observed support is compared with random zonings of the base into the same number of zones, grown by nearest-seed assignment from randomly chosen base units. These zones approximate contiguous regions but are not guaranteed to be contiguous, and they have roughly equal numbers of base units, while administrative units often do not. A large |z| therefore says the observed boundaries differ from size-matched random partitions in this sense.

## Inter-scale distortion

The paper's bi-Lipschitz condition c d_s(x, y) ≤ d_s'(A x, A y) ≤ C d_s(x, y) cannot hold with c > 0 for a many-to-one aggregation map, since two units in the same coarse unit are sent to one point. scalegeom reports empirical constants on pairs the map separates: coarse units matched to their nearest fine unit (`method="centroid"`, as in Section 9.7 of the paper, minimum and maximum ratio), or random pairs of fine units in distinct coarse units (`method="pairs"`, with 1% and 99% quantiles for irregular supports).

## Representation space (FGW)

Each representation is a metric-measure space with features, compared by fused Gromov-Wasserstein distance (POT, α = 0.5, square loss). The solver starts from the optimal transport plan between unit locations, which in our tests made repeated estimates much less variable than the default start; the problem is non-convex, so results are local optima. Representations with more than `max_nodes` units (800 by default) are subsampled with probability proportional to their measure. Subsampling alone produces a positive distance, so each subsampled representation's floor F is estimated from independent subsamples and the adjusted distance FGW(a, b) − (F_a + F_b) / 2 is reported with the raw value. The adjustment is empirical: it can be slightly negative and is not a metric.

## Reproducing the paper

`examples/reproduce_vaz2026.py` uses different random seeds from the original study. With 30 realizations, Moran's I at s = 16 is 0.320 against the published 0.297; the published Monte Carlo standard deviation at that scale is 0.083, so the difference is within sampling variation.
