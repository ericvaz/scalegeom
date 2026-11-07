# scalegeom

[![tests](https://github.com/ericvaz/scalegeom/actions/workflows/tests.yml/badge.svg)](https://github.com/ericvaz/scalegeom/actions/workflows/tests.yml)

Scale-conditioned geometry for multiscale spatial analysis in Python.

scalegeom implements the scale-explicit workflow of Vaz (2026), *Scale-Conditioned Geometry: How Measurement Resolution Induces Distance, Proximity, and Spatial Dependence*, Geographical Analysis 58: e70061 ([doi:10.1111/gean.70061](https://doi.org/10.1111/gean.70061)). It treats scale as a parameter of the analysis: the same variable is represented at a family of resolutions and lattice placements, and distance, neighbourhood and dependence are recomputed on each representation, so their variation across scales becomes part of the evidence.

## What it does

The package follows the four steps of the protocol in Section 10.4 of the paper.

1. **State the representation.** Build a family of representations S(θ), θ = (s, ζ), from a raster (block means), points or polygons on square grids, or nested administrative supports such as dissemination areas, census tracts and subdivisions. Every family records how each base unit was aggregated.
2. **Report trajectories.** At every scale and lattice placement, compute Moran's I under a stated induced metric, the share of variance retained, short-lag semivariance, and the correlation-threshold range proxy. Alternative metrics (k-nearest neighbours, distance bands, queen or rook contiguity, network hops, or a supplied distance matrix such as travel time) are reported side by side.
3. **Test against the change-of-support null.** A covariance model is fitted at the finest scale, fields are simulated from it and pushed through the same aggregation operators, and the observed trajectories are compared with the Monte Carlo band. For nested supports, a second null compares each administrative zoning with random size-matched zonings.
4. **Check invariance.** Report empirical bi-Lipschitz constants of the inter-scale maps, and measure the family itself with fused Gromov-Wasserstein distances (scale velocity and zoning displacement).

A separate function reports global and local Moran's I for one support under several induced metrics, with LISA cluster labels and their agreement (Section 9.9 of the paper).

## Install

```bash
pip install git+https://github.com/ericvaz/scalegeom
pip install "scalegeom[all] @ git+https://github.com/ericvaz/scalegeom"   # adds POT and geopandas
```

The core needs NumPy, SciPy and Matplotlib. Fused Gromov-Wasserstein distances need POT, and reading polygon files needs geopandas.

## Quick start

```python
import scalegeom as sg

field = sg.gaussian_random_field((160, 160), corr_length=12, seed=0)
family = sg.raster_family(field, scales=[1, 2, 4, 5, 8, 10, 16])

protocol = sg.ScaleProtocol(family, metric="knn:8", alt_metrics=["queen"])
report = protocol.run(n_sim=100)
print(report["trajectory"])
print(report["null"])
report.save("my_analysis")      # CSV tables, 300 dpi figures, settings.json, report.html
```

Polygons with a population weight, aggregated through census geography:

```python
import geopandas as gpd
from scalegeom.geo import nested_from_geodataframe

da = gpd.read_file("dissemination_areas.gpkg")
family = nested_from_geodataframe(da, "median_income", levels=["CTUID", "CSDUID"], weight="pop")
report = sg.ScaleProtocol(family, metric="queen", alt_metrics=["knn:6", "hops:2"]).run()
```

One support, three induced metrics:

```python
from scalegeom.geo import from_geodataframe

rep = from_geodataframe(gpd.read_file("sids2.shp"), "SIDR74")
sg.metric_sensitivity(rep, ["knn:6", "queen", "hops:2"]).save("nc_metrics")
```

## Reproducing the paper

```bash
python examples/reproduce_vaz2026.py
```

The script reruns the simulation study and the North Carolina analysis in about 20 seconds. With 30 Monte Carlo realizations it gives Moran's I from 0.995 to 0.320 across block sizes 1 to 16 (paper: 0.994 to 0.297), variance retained 0.760 at s = 16 (0.752), and a range proxy between 0.164 and 0.184 (0.160 to 0.189). The single-length scenario stays inside the change-of-support band and the two-length scenario leaves it. For North Carolina SIDS rates (1974), global Moran's I is 0.192, 0.231 and 0.152 under k-nearest neighbours, queen contiguity and the two-hop network, matching Table 3 of the paper.

## Implementation notes

- Moran's I agrees with PySAL's esda to machine precision and local Moran quadrants agree exactly; permutation inference follows esda's conventions.
- Semivariograms and correlation profiles use one normalized lag grid for every scale, with lags normalized by the domain extent and semivariances standardized by the base moments.
- The covariance model for the null is fitted at the mean pair distance of each lag bin with Cressie weights, anchored by the exact nearest-neighbour semivariance. The base scale is inside the band by construction, and Moran's I is tested as change from the base.
- Fused Gromov-Wasserstein distances start the solver from the optimal transport plan between unit locations. Representations larger than `max_nodes` are subsampled; because subsampling alone produces a positive distance, each such representation's floor is estimated and a debiased distance is reported next to the raw one.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Contributing and support

See [CONTRIBUTING.md](CONTRIBUTING.md). Questions and bug reports go to the issue tracker.

## Citation

Please cite the method paper (see `CITATION.cff`). Licence: MIT.
