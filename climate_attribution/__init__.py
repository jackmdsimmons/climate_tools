"""
climate_attribution -- constituent-level attribution of changes in portfolio
climate KPIs.

Phase 0-1: the metric-agnostic LMDI engine and the partitioning layer that makes
it safe on real portfolios. See README.md for the framework and roadmap.
"""

from .core import (
    Decomposition,
    NonPositiveDriver,
    lmdi_decompose,
    log_mean,
    log_mean_array,
)
from .partition import (
    Block,
    GroupReclassified,
    Level,
    Partition,
    PartitionedDecomposition,
    classify,
    decompose_blocks,
    nested_weight_drivers,
    nested_weights,
)

__version__ = "0.1.0"

__all__ = [
    "Block",
    "Decomposition",
    "GroupReclassified",
    "Level",
    "NonPositiveDriver",
    "Partition",
    "PartitionedDecomposition",
    "classify",
    "decompose_blocks",
    "lmdi_decompose",
    "log_mean",
    "log_mean_array",
    "nested_weight_drivers",
    "nested_weights",
]
