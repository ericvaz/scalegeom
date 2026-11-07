"""How to cite scalegeom."""

REFERENCE = ("Vaz, E. (2026). Scale-conditioned geometry: How measurement resolution induces "
             "distance, proximity, and spatial dependence. Geographical Analysis, 58(4), e70061. "
             "https://doi.org/10.1111/gean.70061")

BIBTEX = """@article{vaz2026scale,
  author  = {Vaz, Eric},
  title   = {Scale-Conditioned Geometry: How Measurement Resolution Induces Distance, Proximity, and Spatial Dependence},
  journal = {Geographical Analysis},
  year    = {2026},
  volume  = {58},
  number  = {4},
  pages   = {e70061},
  doi     = {10.1111/gean.70061}
}"""


def citation(style: str = "text") -> str:
    """The reference to cite when using scalegeom: style "text" or "bibtex"."""
    if style == "bibtex":
        return BIBTEX
    return REFERENCE
