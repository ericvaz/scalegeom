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

<!-- Before submission (May 2027 at the earliest): replace the ORCID and update
the Research impact statement with the studies that have used the package. -->

# Summary

The same place can look clustered at one resolution and nearly random at another. Spatial analysts have known this since @gehlke1934 and @openshaw1984, yet most workflows still compute distance, neighbourhoods, and spatial dependence at a single resolution and treat them as fixed. `scalegeom` examines one dataset at several levels of geographic detail: fine raster cells combined into larger blocks, points or polygons aggregated into grid cells of increasing size, or neighbourhoods grouped into tracts, municipalities, and regions. At each level, it recalculates the usual measures of spatial similarity and clustering. It then tests whether the changes it finds are what aggregation alone would produce, or whether they point to structure that a single resolution would miss. The package implements @vaz2026scale and reproduces the results reported there.

# Statement of need

In @vaz2026scale, I argued that distance, proximity, and dependence are induced by the representation, so that a Moran's I, a semivariogram, or a cluster map describes the data at a stated resolution and under a stated rule for who counts as a neighbour. That paper ended with a four-step protocol for applied work. First, state how observations are grouped and how neighbours are defined. Second, recalculate the spatial statistics across resolutions, boundary placements, and neighbour rules, and report how they change. Third, compare those changes with the changes expected from aggregation alone. Fourth, when neighbourhood-based conclusions are carried from one resolution to another, measure how much the geometry is distorted along the way.

Each step uses standard estimators, but the comparison is valid only when the simulated and observed data undergo identical aggregation, when distances and lag intervals mean the same thing at every scale, and when the expected change is calibrated at the finest resolution. Without these controls, an apparent scale effect can be an artifact of the procedure. `scalegeom` records which original observations belong to each larger unit and enforces these conditions so that the protocol can be applied in a few lines and reported in full.

The package is written for geographers, ecologists, epidemiologists, and remote sensing researchers who report spatial statistics at one resolution and want to know whether their conclusions hold at others. It is also useful for teaching the modifiable areal unit problem with something more concrete than a cautionary paragraph.

# State of the field

Mature tools cover the ingredients. PySAL's `libpysal` and `esda` build spatial weights and compute global and local Moran statistics [@rey2022pysal]; `scalegeom` matches `esda` to machine precision on the same weights. `esda` also implements S-maup [@duque2018smaup], which asks whether a variable is likely to change when its units are aggregated into a given number of regions. `spopt` builds random and optimal regionalizations [@feng2022spopt], `maup` and `tobler` move attributes between supports, `scikit-gstat` [@malicke2022skgstat] and `GSTools` [@muller2022gstools] fit variograms and simulate fields, and POT [@flamary2021pot] computes fused Gromov-Wasserstein distances [@memoli2011; @vayer2020fgw]. In R, `spdep` [@bivand2018lisa], `gstat` [@pebesma2004gstat], and `spatstat` [@baddeley2005spatstat] offer the same building blocks.

These libraries are organized around one dataset, one partition, or one diagnostic. `scalegeom` organizes the analysis around an ordered family of representations of the same data, records the aggregation between them, and applies the same diagnostics and simulated null across that family. Its question also differs from S-maup's. S-maup indicates whether aggregation is likely to matter for a variable; `scalegeom` asks how each diagnostic changes from one support to the next, whether that change exceeds what a single fine-scale structure predicts, and whether particular boundaries behave differently from random partitions of the same size. On the bundled North Carolina data, both agree that income is sensitive to aggregation, and only `scalegeom` shows that the county boundaries themselves matter.

I built a separate package on top of these tools, without extending one of them, because the family is a new unit of analysis that cuts across their scopes: it needs weights from PySAL, simulation from geostatistics, and transport distances from POT, and the membership records it carries are what every step depends on. The package uses existing tools where they fit (POT, optional `geopandas`, and `esda` as the reference in its tests) and could take its regionalizations from `spopt`.

# Software design

The central object is a `Family`: representations ordered from fine to coarse, each with unit values, centroids, weights, optional contiguity, and an array giving the parent unit of every base unit. A representation is labelled by its scale $s$ and its zoning, which for block and grid families is the offset $\zeta$ of the aggregation lattice and for administrative or random zonings is the partition itself. Families are built from rasters, points or polygons on square grids, nested code columns, or random size-matched zonings. Neighbour rules are named by short strings (`"knn:8"`, `"queen"`, `"hops:2"`), so a report states exactly which rule produced each number, and all distances are scaled by one domain extent shared across the family.

The change-of-support null [@cressie1996; @gotway2002] fits a covariance model at the finest resolution, simulates fields from it, pushes them through the recorded aggregation, and builds a band for each diagnostic. Three design choices came out of validation. Short distances need care: fitting at the mean pair distance of each lag bin with @cressie1985 weights and anchoring the fit with the nearest-neighbour semivariance removed an artificial discontinuity that otherwise pushed even the finest scale out of its own band. The model form matters: Gaussian, exponential, and spherical models, each with and without a nugget, are compared by a penalized weighted fit, because on terrain data a model that was too smooth made the test reject at almost every scale. Parameter uncertainty matters most: a calibration study on simulated fields that satisfy the null showed the test rejecting a quarter of rough fields at the first aggregation step when fitted parameters were treated as known. Propagating their uncertainty with a parametric bootstrap brought the false-rejection rate across all scales to about 2% (documented, with code, in the repository).

I report two further measures and state their limits. Aggregation is many-to-one, so a strict bi-Lipschitz bound cannot hold; the package reports empirical distortion constants on the pairs that the inter-scale map keeps apart. Fused Gromov-Wasserstein distances between large representations require subsampling, so the package estimates the distance that subsampling alone produces and reports an adjusted value alongside the raw one, as an empirical correction.

The core depends only on NumPy, SciPy, and Matplotlib. Results are returned as plain tables and gathered in a report that writes CSV tables, 300 dpi figures, the settings used, and a self-contained HTML page.

# Research impact statement

The package reproduces the published results of @vaz2026scale from one script. Across 30 simulated fields, Moran's I falls from 0.995 at the finest resolution to 0.320 at 16-cell blocks (published: 0.994 to 0.297, within the published Monte Carlo standard deviation of 0.083 at that scale), and the correlation-threshold range proxy stays between 0.164 and 0.184 (0.160 to 0.189). A field with one correlation length stays inside the change-of-support band and a field with two leaves it, as in the paper (\autoref{fig:null}). For North Carolina sudden infant death rates, Moran's I is 0.192, 0.231, and 0.152 under six nearest neighbours, queen contiguity, and a two-hop network, the published values.

The package ships a test bank of public-domain data with a manifest of sources and checksums: 2,660 North Carolina census tracts with American Community Survey estimates and polygon contiguity [@acs2023], and ETOPO1 elevation from the Blue Ridge to the Piedmont [@amante2009etopo1]. On these data, the protocol separates cases a single-resolution analysis would treat alike. The elevation trajectories are consistent with the change predicted by the fitted aggregation model. Tract income, aggregated as a population-weighted mean of tract medians, departs from that model on grids from 5 to 40 km under every candidate covariance model, and the 100 counties retain 19% of the tract-level variance of this quantity where random partitions into 100 zones retain 45%.

![Moran's I under block aggregation against the change-of-support null fitted at the finest scale. A field with a single correlation length (A) follows the band; a field with two well-separated lengths (B) leaves it, because the short-range component is averaged out within blocks while the long-range component persists.\label{fig:null}](figure1.png)

<!-- Add here, before submission: the studies that have used scalegeom (for example
the nested census analysis of Toronto and the digital inequality paper if it is
rerun with the package), with citations, and any external users. -->

# Acknowledgements

<!-- Add funding sources, or state that the work received no specific funding. -->

# References
