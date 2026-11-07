# Bundled open datasets

scalegeom ships a small test bank of open data in `src/scalegeom/datasets/data/` (about 650 KB). Every file can be rebuilt from its public source with `scripts/build_datasets.py`.

| Name | Content | Source | Licence |
|---|---|---|---|
| `nc_tracts` | 2,660 North Carolina census tracts: median household income (B19013), population (B01003), share of households with broadband (S2801), share of adults 25+ with a bachelor's degree (S1501); county codes; centroids in NAD83 / North Carolina (EPSG:32119); queen and rook contiguity; simplified polygons | US Census Bureau, American Community Survey 2023 5-year estimates; 2023 cartographic boundary file `cb_2023_37_tract_500k` | Public domain (US government work) |
| `etopo_blue_ridge` | Elevation in metres, 160 x 160 cells at 1 arc-minute, 34.70 to 37.35 N, 83.70 to 81.05 W | NOAA ETOPO1 through the CoastWatch ERDDAP server (dataset `etopo180`) | Public domain (US government work) |
| `simulated` | Scenarios A and B of Vaz (2026), regenerated from seeds | scalegeom | MIT |

## Rebuilding

The ACS values come from the Census Data API, which needs a free key:

```bash
export CENSUS_API_KEY=...   # https://api.census.gov/data/key_signup.html
python scripts/build_datasets.py
```

or from a CSV of the same four variables keyed by an 11-digit `GEOID`:

```bash
python scripts/build_datasets.py --acs-csv tracts_37.csv --boundaries cb_2023_37_tract_500k.zip
```

The bundled files were built on 7 November 2026 from ACS 2023 5-year tables downloaded from data.census.gov in July 2026, and the 2023 cartographic boundaries.

## Notes

- ACS values are survey estimates with margins of error, which are not included here.
- Thirty-two tracts (water, institutions, parks) have no income estimate and are dropped by the loaders.
- ETOPO cells are geographic, so distances are in degrees and slightly anisotropic (about 1.85 km by 1.51 km at 36 N).
