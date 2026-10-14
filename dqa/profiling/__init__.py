"""Profiling: describing what a dataset contains, before judging its quality."""

from dqa.profiling.cardinality import (
    CardinalityProfile,
    ColumnCardinalityProfile,
    profile_cardinality,
)
from dqa.profiling.correlations import (
    CorrelationPair,
    CorrelationProfile,
    profile_correlations,
)
from dqa.profiling.dtypes import ColumnTypeProfile, DTypeProfile, profile_dtypes
from dqa.profiling.duplicates import DuplicateProfile, profile_duplicates
from dqa.profiling.missing import (
    ColumnMissingProfile,
    MissingProfile,
    profile_missing,
)
from dqa.profiling.shape import ShapeProfile, profile_shape
from dqa.profiling.statistics import (
    ColumnStatisticsProfile,
    StatisticsProfile,
    profile_statistics,
)
from dqa.profiling.value_counts import (
    ValueCount,
    ValueCountsProfile,
    profile_value_counts,
)

__all__ = [
    "CardinalityProfile",
    "ColumnCardinalityProfile",
    "ColumnMissingProfile",
    "ColumnStatisticsProfile",
    "ColumnTypeProfile",
    "CorrelationPair",
    "CorrelationProfile",
    "DTypeProfile",
    "DuplicateProfile",
    "MissingProfile",
    "ShapeProfile",
    "StatisticsProfile",
    "ValueCount",
    "ValueCountsProfile",
    "profile_cardinality",
    "profile_correlations",
    "profile_dtypes",
    "profile_duplicates",
    "profile_missing",
    "profile_shape",
    "profile_statistics",
    "profile_value_counts",
]
