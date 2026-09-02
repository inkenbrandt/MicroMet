import pandas as pd

def _parse_factor(val: tuple[float, float] | float) -> float:
    if isinstance(val, (tuple, list)):
        old, new = val
        return old / new
    return float(val)

def apply_netrad_calibration(
    df: pd.DataFrame,
    sw_in: tuple[float, float] | float,
    sw_out: tuple[float, float] | float,
    lw_in: tuple[float, float] | float,
    lw_out: tuple[float, float] | float,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    suffix: str = "1_1_1"
) -> pd.DataFrame:
    """Applies radiation calibration factors across a specified date range."""
    df_fixed = df.copy()

    # Convert index to DatetimeIndex for comparison
    idx_dt = pd.to_datetime(df_fixed.index)

    # Calculate ratios from (old, new) inputs or direct floats
    cf = {
        "SW_IN": _parse_factor(sw_in),
        "SW_OUT": _parse_factor(sw_out),
        "LW_IN": _parse_factor(lw_in),
        "LW_OUT": _parse_factor(lw_out)
    }

    # Construct column names based on suffix
    sw_in_col = f"SW_IN_{suffix}"
    sw_out_col = f"SW_OUT_{suffix}"
    r_lw_in_col = f"R_LW_IN_MEAS_{suffix}"
    r_lw_out_col = f"R_LW_OUT_MEAS_{suffix}"
    lw_in_col = f"LW_IN_{suffix}"
    lw_out_col = f"LW_OUT_{suffix}"
    netrad_col = f"NETRAD_{suffix}"
    alb_col = f"ALB_{suffix}"

    # Build date filter mask using parsed datetime index
    mask = pd.Series(True, index=df_fixed.index)
    if start_date is not None:
        mask &= idx_dt >= pd.to_datetime(start_date)
    if end_date is not None:
        mask &= idx_dt <= pd.to_datetime(end_date)

    if not mask.any():
        print("No matching dates found in date range.")
        return df_fixed

    # 1. Correct SW IN and SW OUT
    df_fixed.loc[mask, sw_in_col] *= cf["SW_IN"]
    df_fixed.loc[mask, sw_out_col] *= cf["SW_OUT"]

    # 2. Correct measured LW IN and OUT values
    r_lw_in_meas_fix = df_fixed.loc[mask, r_lw_in_col] * cf["LW_IN"]
    r_lw_out_meas_fix = df_fixed.loc[mask, r_lw_out_col] * cf["LW_OUT"]

    # 3. Calculate corrected net LW radiation and update overall NETRAD
    lw_net_corrected = r_lw_in_meas_fix - r_lw_out_meas_fix
    df_fixed.loc[mask, netrad_col] = (
        df_fixed.loc[mask, sw_in_col]
        - df_fixed.loc[mask, sw_out_col]
        + lw_net_corrected
    )

    # 4. Correct final LW OUT (retaining temperature/sigma adjustments)
    lw_out_sigma = df_fixed.loc[mask, lw_out_col] - df_fixed.loc[mask, r_lw_out_col]
    df_fixed.loc[mask, lw_out_col] = r_lw_out_meas_fix + lw_out_sigma

    # 5. Correct final LW IN (retaining temperature/sigma adjustments)
    lw_in_sigma = df_fixed.loc[mask, lw_in_col] - df_fixed.loc[mask, r_lw_in_col]
    df_fixed.loc[mask, lw_in_col] = r_lw_in_meas_fix + lw_in_sigma

    return df_fixed


def apply_correction_factors(
    df: pd.DataFrame, 
    correction_df: pd.DataFrame, 
    stationid: str | int, 
    omit_vars: str | list[str] = None,
) -> pd.DataFrame:
    """Applies time-bounded calibration factors to target columns in a DataFrame by 
    multipying the original values by a correction factor. All of the boundaries on the  
    corrections are found in the correction_df table, which must be formatted correctly.

        Parameters
        ----------
        df : pd.DataFrame
            Target time-series DataFrame indexed by date/time.
        correction_df : pd.DataFrame
            DataFrame containing calibration metadata. Must include 'start_date',
            'end_date', 'variables', and 'correction_factor` and can optionally include
            'stationid'. 
        stationid : str or int, optional
            Filters `correction_df` to rules matching a specific station identifier.
            If None, applies rules across all stations in `correction_df`.
        omit_vars : str or list of str, optional
            Variable name(s) to exclude from being corrected.

        Returns
        -------
        pd.DataFrame
            A updated copy of `df` with calibration factors applied to matching
            date ranges and variable columns.
        """
    
    df_update = df.copy()

    if stationid is not None:
        correction_df = correction_df[correction_df["stationid"] == stationid]

    if omit_vars:
        if isinstance(omit_vars, str):
            omit_vars = [omit_vars]
        correction_df = correction_df[~correction_df["variables"].isin(omit_vars)]

    if correction_df.empty:
        print("No calibration factors to correct in data")
        return df_update

    for row in correction_df.itertuples():
        mask = (df_update.index >= row.start_date) & (df_update.index <= row.end_date)

        # Process variable names
        if isinstance(row.variables, str):
            var_list = [v.strip() for v in row.variables.split(",")]
        else:
            var_list = [row.variables]

        factor = getattr(row, "correction_factor")

        df_update.loc[mask, var_list] = df_update.loc[mask, var_list] * factor
        if mask.sum()>0:
            print(f'Updated {mask.sum()} rows for {var_list} with correction factor of {factor}')

    return df_update