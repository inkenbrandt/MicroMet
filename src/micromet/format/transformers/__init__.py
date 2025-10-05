"""
Data transformation functions for the reformatter pipeline.

This package contains modular transformation functions organized by category:
- timestamps: Datetime handling and resampling
- columns: Column naming, renaming, and organization
- validation: Data quality checks and boundary enforcement
- corrections: Variable-specific data fixes
- cleanup: Column filtering and type setting

For backward compatibility, all functions are re-exported at the package level.
"""

# Re-export constants (for backward compatibility)
from .cleanup import SOIL_SENSOR_SKIP_INDEX, DEFAULT_SOIL_DROP_LIMIT

MISSING_VALUE: int = -9999

# Import and re-export all transformation functions for backward compatibility
# This allows: from micromet.format.transformers import fix_timestamps
# to continue working exactly as before

from .timestamps import (
    infer_datetime_col,
    fix_timestamps,
    resample_timestamps,
    timestamp_reset,
)

from .columns import (
    rename_columns,
    normalize_prefixes,
    modernize_soil_legacy,
    make_unique,
    make_unique_cols,
    col_order,
)

from .validation import (
    apply_physical_limits,
    mask_stuck_values,
)

from .corrections import (
    apply_fixes,
    tau_fixer,
    fix_swc_percent,
    ssitc_scale,
    scale_and_convert,
    rating,
    fill_na_drop_dups,
)

from .cleanup import (
    drop_extra_soil_columns,
    set_number_types,
    drop_extras,
    process_and_match_columns,
)

# Explicit __all__ for clarity about public API
__all__ = [
    # Constants
    "MISSING_VALUE",
    "SOIL_SENSOR_SKIP_INDEX",
    "DEFAULT_SOIL_DROP_LIMIT",
    
    # Timestamp functions
    "infer_datetime_col",
    "fix_timestamps",
    "resample_timestamps",
    "timestamp_reset",
    
    # Column functions
    "rename_columns",
    "normalize_prefixes",
    "modernize_soil_legacy",
    "make_unique",
    "make_unique_cols",
    "col_order",
    
    # Validation functions
    "apply_physical_limits",
    "mask_stuck_values",
    
    # Correction functions
    "apply_fixes",
    "tau_fixer",
    "fix_swc_percent",
    "ssitc_scale",
    "scale_and_convert",
    "rating",
    "fill_na_drop_dups",
    
    # Cleanup functions
    "drop_extra_soil_columns",
    "set_number_types",
    "drop_extras",
    "process_and_match_columns",
]