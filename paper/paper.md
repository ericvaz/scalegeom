---
title: 'scalegeom: scale-conditioned geometry for multiscale spatial analysis in Python'
tags:
  - Python
  - spatial analysis
  - modifiable areal unit problem
  - change of support
  - spatial autocorrelation
  - optimal transport
authors:
  - name: Eric Vaz
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Department of Geography and Environmental Studies, Toronto Metropolitan University, Toronto, Ontario, Canada
    index: 1
date: 7 November 2026
bibliography: paper.bib
---

<!-- Before submission (May 2027 at the earliest): replace the ORCID, update the
Research impact statement with the studies that have used the package, and
check the AI usage disclosure against what actually happened. -->

# Summary

The same place can look clustered at one resolution and nearly random at another. Spatial analysts have known this since @gehlke1934 and @openshaw1984, yet most workflows still compute distance, neighbourhoods and spatial dependence at a single resolution and treat them as fixed inputs. `scalegeom` makes the resolution an explicit parameter. It builds a family of representations of one dataset (a raster aggregated to coarser blocks, points or polygons aggregated to grids, or nested census geographies), recomputes the usual spatial diagnostics on every member of the family, and tests whether the way those diagnostics change is what aggregation alone would produce. The package implements the protocol set out in @vaz2026scale and reproduces the simulation and observational results reported there.

# Statement of need

In @vaz2026scale I argued that distance, proximity and dependence are induced by the representation, so that a Moran's I, a semivariogram or a cluster map is a property of the data at a stated resolution and under a stated metric. The paper closed with a four-step protocol for applied work: state the representation $\theta = (s, \zeta)$, where $s$ is the scale and $\zeta$ the placement of the aggregation lattice, together with the induced metric; recompute the core diagnostics across scales, placements and metrics and report trajectories; test those trajectories against a change-of-support null; and, where neighbourhood-based conclusions are carried across resolutions, report the bi-Lipschitz distortion of the inter-scale maps. Each step uses standard estimators, but no existing package strings them together, and doing so by hand requires care that is easy to get wrong. The aggregation operator has to be recorded so that simulated fields pass through exactly the same blocks or zones as the data, lag grids and standardization have to be held fixed across scales, and the null has to be calibrated at the finest support.

`scalegeom` is written for geographers, ecologists, epidemiologists and remote sensing researchers who report spatial statistics at one resolution and want to know whether their conclusions hold at others. It is also meant for teaching the modifiable areal unit problem with something more concrete than a cautionary paragraph.

# State of the field

The Python spatial ecosystem covers each ingredient separately. PySAL's `libpysal` and `esda` build spatial weights and compute global and local Moran statistics [@rey2022pysal], and `scalegeom` matches `esda` to machine precision on the same weights. `scikit-gstat` [@malicke2022skgstat] and `GSTools` [@muller2022gstools] estimate variograms, fit covariance models and simulate random fields. `tobler` in the PySAL family moves attributes between incompatible supports. POT [@flamary2021pot] computes Gromov-Wasserstein and fused Gromov-Wasserstein distances [@memoli2011; @vayer2020fgw]. None of these treat a family of representations as the object of analysis, and none provide the change-of-support test for a diagnostic's trajectory [@gotway2002; @cressie1996].

I chose to build a small package that sits on top of this ecosystem rather than contribute a module to one of these libraries, because the unit of analysis is different. Every function in `scalegeom` takes a family, and the family carries the membership of each base unit in every aggregated unit. Adding that structure to a weights library or a geostatistics library would change their core abstractions. Where an existing tool does the job, `scalegeom` uses it: POT for transport, optional `geopandas` for reading polygons, and `esda` as the reference in the test suite.

# Software design

The central object is a `Family`: an ordered scale path of `Representation` objects, each with unit values, centroids, a measure, an optional contiguity graph and an integer array mapping every base unit to its parent unit. Families are built by block aggregation of rasters, by square grids over points or polygons, by nested label columns, or by random size-matched zonings. Because the membership array is kept, the same aggregation operator can be applied to any vector of base values, which is what the null model needs.

Induced metrics are named by short strings (`"knn:8"`, `"queen"`, `"hops:2"`, `"band:0.05"`) and resolved on each representation, so a report states exactly which neighbourhood rule produced each number. Distances are normalized by one domain extent shared across the family, and semivariances use the moments of the finest representation, so levels can be compared across scales.

Three choices needed particular care. First, the covariance model behind the change-of-support null is fitted at the mean pair distance of each lag bin with the weights of @cressie1985, and anchored by the exact nearest-neighbour semivariance; fitting at bin centres produced a spurious nugget that pushed even the finest scale outside its own band. Second, the null tests change from the base, the variance as a share of base variance and Moran's I as $I(s) - I(\text{base})$, so that the test addresses aggregation and not the residual misfit of the covariance model. Third, fused Gromov-Wasserstein distances between large representations require subsampling, and subsampling alone produces a positive distance. The package starts the solver from the transport plan between unit locations, which removes most local optima, estimates each representation's subsampling floor, and reports a debiased distance next to the raw one, in the spirit of debiased Sinkhorn divergences [@feydy2019].

The core depends only on NumPy, SciPy and Matplotlib, so it installs anywhere these do. Results are returned as plain tables and gathered in a `Report` that writes CSV tables, 300 dpi figures, the settings used, and a self-contained HTML page.

# Research impact statement

The package reproduces the published results of @vaz2026scale from a single script. Across 30 simulated fields, Moran's I falls from 0.995 at the finest resolution to 0.320 at 16-cell blocks (published: 0.994 to 0.297), 76.0% of the variance is retained at the coarsest scale (75.2%), and the correlation-threshold range proxy stays between 0.164 and 0.184 (0.160 to 0.189). A field with one correlation length stays within the change-of-support band, and a field with two leaves it, as in the paper (\autoref{fig:null}). For North Carolina sudden infant death rates, global Moran's I is 0.192, 0.231 and 0.152 under six nearest neighbours, queen contiguity and a two-hop network, matching the published values exactly.

![Moran's I under block aggregation against the change-of-support null fitted at the finest scale. A field with a single correlation length (A) follows the band; a field with two well-separated lengths (B) leaves it because the short-range component is averaged out within blocks while the long-range component persists.\label{fig:null}](figure1.png)

<!-- Add here, before submission: the studies that have used scalegeom (for example
the nested census analysis of Toronto and the digital inequality paper if it is
rerun with the package), with citations, and any external users. -->

# AI usage disclosure

The code, tests, documentation and a first draft of this paper were produced with the assistance of Claude Opus 5.5 (Anthropic), used through Claude Code in November 2026. The assistance covered code generation, test scaffolding, documentation drafting and drafting of this text. The method, its parameterization and the protocol come from @vaz2026scale. I reviewed, edited and validated all AI-assisted output, checked the implementation against PySAL's `esda` and against the published results, and made the final design decisions described above.

# Acknowledgements

<!-- Add funding sources, or state that the work received no specific funding. -->

# References
