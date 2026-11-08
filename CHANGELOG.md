# Changelog

## Unreleased (0.2.0)

Changes made after a pre-submission review of the methods. Several change results, so regression values in `tests/benchmarks.json` were updated and are explained here.

- Variance retained is now measured with each representation's measure (`weighted_variance`), the estimand consistent with mass-weighted aggregation: the ratio is the between-unit share of the base variance. Trajectories also report the unweighted ratio (`var_retained_unweighted`), and the nulls take `weighted=False` for the old behaviour. For North Carolina tract income aggregated to counties the share is 0.302 weighted and 0.192 unweighted.
- The zoning null now compares an observed partition with random contiguous zonings that follow its distribution of zone sizes (`zoning_null(method="size_matched")`, using the new `contiguous_zoning`). The earlier Voronoi null, with zones of roughly equal size, remains as `method="voronoi"`. With equal-size zones, North Carolina counties appear to hide income differences (z = -5.5); against size-matched contiguous zones they are typical (z near 0). A zoning null that ignores zone sizes can mistake a size effect for a boundary effect.
- Covariance fitting uses the exact semivariance of near-neighbour pairs, in distance-quantile bins, in place of the first lag bin and the single nearest-neighbour anchor. On irregular supports the old fit overestimated short correlation lengths and the test rejected 40 to 55% of fields that satisfied its null; with the new fit the rate is about 5% (`examples/null_calibration_irregular.py`).
- `information_loss_null` adds a simultaneous test across all scales and both diagnostics (`result.simultaneous`): a maximum-statistic test whose reference distribution comes from the same simulations.
- `local_moran(..., fdr=True)` declares significance by the Benjamini-Hochberg procedure; `metric_sensitivity` uses it by default and reports cluster agreement both over all units and over significant units only.
- New tests: weighted variance as the between-zone share, exact aggregation of zone-constant fields, contiguity and size matching of random zonings, the FDR procedure, and the simultaneous test.


- Open-data test bank in `scalegeom.datasets`: North Carolina census tracts (ACS 2023, public domain) with tract to county nesting and polygon contiguity, NOAA ETOPO1 elevation for the Blue Ridge, and the simulated scenarios of Vaz (2026). Provenance in `DATASETS.md`; rebuild with `scripts/build_datasets.py`.
- Known-answer tests in `tests/benchmarks.json`: published values from Vaz (2026), dataset integrity checks, and pinned regression values.
- `examples/open_data_case_studies.py`: four case studies on the bundled data, each null repeated under every covariance model.
- The change-of-support null now selects its covariance model by weighted fit (`model="auto"`, the new default). On the elevation data a Gaussian model made the null reject at every scale while the better-fitting exponential model did not.
- The covariance of point and polygon supports is factored once per null test, which makes 100 simulations on a few thousand units take seconds.
- The null propagates the uncertainty of the fitted covariance parameters by parametric bootstrap (`uncertainty=True`, the default). A calibration study (`examples/null_calibration.py`, `docs/methods.md`) showed that with parameters treated as known the test rejected about a quarter of rough stationary fields at the first aggregation step; with the bootstrap the false-rejection rate across all scales is about 2%. Bands are wider than in 0.1.0.
- Model selection now also decides whether to include a nugget, using an AIC-type criterion on the weighted fit.
- The inter-scale distortion is documented and labelled as empirical bi-Lipschitz constants on matched pairs, since aggregation is many-to-one. The FGW adjustment for subsampling is described as an empirical correction.
- S-maup (Duque, Laniado and Polo 2018, via esda) is reported next to the nulls in the nested case study.
- Bundled data have a manifest with sources, licences and SHA-256 checksums, checked by the tests.
- Fixed: CSV output wrote NumPy floats as `np.float64(...)`.

## 0.1.0 (2026-11-07)

First public release.

- Families of representations: raster block aggregation, square grids over points or polygons, nested administrative supports, and random size-matched zonings, with half-lattice shifts for the zoning direction.
- Induced metrics: k-nearest neighbours, distance bands, queen and rook contiguity, network hops, and supplied distance matrices.
- Diagnostics: global and local Moran's I with permutation inference, semivariograms and correlation profiles on a common normalized lag grid, and the correlation-threshold range proxy.
- Change-of-support null fitted at the finest scale, with the analytic regularization prediction, and a zoning null for nested supports.
- Bi-Lipschitz distortion of inter-scale maps and debiased fused Gromov-Wasserstein distances between representations.
- `ScaleProtocol` and `metric_sensitivity` reports with CSV tables, figures and HTML output.
- Script reproducing the results of Vaz (2026).
