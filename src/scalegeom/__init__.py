"""scalegeom: scale-conditioned geometry for spatial analysis.

Implements the scale-explicit workflow of Vaz (2026), Geographical Analysis
58(4): e70061 (doi:10.1111/gean.70061); cite it with scalegeom.citation().
The package provides scale-indexed families of representations, induced metrics,
diagnostic trajectories, the change-of-support null, inter-scale distortion
checks, and the Fused Gromov-Wasserstein representation space.
"""
__version__ = "0.2.0"

from .representation import (Family, Representation, aggregate_by_labels, from_points, from_raster,
                             grid_family, nested_family, raster_family, voronoi_zoning,
                             contiguous_zoning, weighted_variance)
from .metrics import induced_adjacency, row_standardize
from .diagnostics import (correlation_profile, local_moran, morans_i, range_proxy, semivariogram)
from .simulate import CovModel, gaussian_random_field, mixed_scale_field
from .null import fit_covariance, information_loss_null, regularization_prediction, zoning_null
from .space import bilipschitz, fgw_distance, representation_space
from .protocol import ScaleProtocol, metric_sensitivity
from .report import Report
from .table import Table
from ._citation import citation

__all__ = [
    "Family", "Representation", "aggregate_by_labels", "from_points", "from_raster", "grid_family",
    "nested_family", "raster_family", "voronoi_zoning", "contiguous_zoning", "weighted_variance", "induced_adjacency", "row_standardize",
    "correlation_profile", "local_moran", "morans_i", "range_proxy", "semivariogram", "CovModel",
    "gaussian_random_field", "mixed_scale_field", "fit_covariance", "information_loss_null",
    "regularization_prediction", "zoning_null", "bilipschitz", "fgw_distance", "representation_space",
    "ScaleProtocol", "metric_sensitivity", "Report", "Table", "citation",
]
