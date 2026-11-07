# Changelog

## Unreleased

- Open-data test bank in `scalegeom.datasets`: North Carolina census tracts (ACS 2023, public domain) with tract to county nesting and polygon contiguity, NOAA ETOPO1 elevation for the Blue Ridge, and the simulated scenarios of Vaz (2026). Provenance in `DATASETS.md`; rebuild with `scripts/build_datasets.py`.
- Known-answer tests in `tests/benchmarks.json`: published values from Vaz (2026), dataset integrity checks, and pinned regression values.
- `examples/open_data_case_studies.py`: four case studies on the bundled data, each null repeated under every covariance model.
- The change-of-support null now selects its covariance model by weighted fit (`model="auto"`, the new default). On the elevation data a Gaussian model made the null reject at every scale while the better-fitting exponential model did not.
- The covariance of point and polygon supports is factored once per null test, which makes 100 simulations on a few thousand units take seconds.

## 0.1.0 (2026-11-07)

First public release.

- Families of representations: raster block aggregation, square grids over points or polygons, nested administrative supports, and random size-matched zonings, with half-lattice shifts for the zoning direction.
- Induced metrics: k-nearest neighbours, distance bands, queen and rook contiguity, network hops, and supplied distance matrices.
- Diagnostics: global and local Moran's I with permutation inference, semivariograms and correlation profiles on a common normalized lag grid, and the correlation-threshold range proxy.
- Change-of-support null fitted at the finest scale, with the analytic regularization prediction, and a zoning null for nested supports.
- Bi-Lipschitz distortion of inter-scale maps and debiased fused Gromov-Wasserstein distances between representations.
- `ScaleProtocol` and `metric_sensitivity` reports with CSV tables, figures and HTML output.
- Script reproducing the results of Vaz (2026).
