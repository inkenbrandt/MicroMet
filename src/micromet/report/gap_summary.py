import pandas as pd
from pandas.tseries.frequencies import to_offset


def summarize_gaps(
    df: pd.DataFrame,
    station_level: str = "STATIONID",
    time_level: str = "DATETIME_END",
    expected_freq: str = "30min",
    columns: list | None = None,
) -> pd.DataFrame:
    """
    Summarize runs of missing data (NaNs) per column for each station in a
    MultiIndex DataFrame indexed by (station, datetime).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a MultiIndex (station_level, time_level).
    station_level : str, default "STATIONID"
        Name of the station level in the index.
    time_level : str, default "DATETIME_START"
        Name of the datetime level in the index.
    expected_freq : str, default "30min"
        The expected sampling frequency. Used to build a complete timeline per station
        so that missing timestamps become explicit NaNs.
    columns : list[str] | None
        Subset of columns to analyze. Defaults to all columns.

    Returns
    -------
    pd.DataFrame
        Columns:
            - STATIONID
            - COLUMN
            - GAP_START
            - GAP_END
            - N_STEPS_MISSING
            - HOURS_MISSING
            - GAP_KIND  ("MissingTimestamp", "NaN", or "Mixed")
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("df must have a MultiIndex (station, datetime).")

    if station_level not in df.index.names or time_level not in df.index.names:
        raise KeyError(
            "MultiIndex must contain the specified station_level and time_level."
        )

    # Work on a sorted copy
    df = df.copy()
    df = df.sort_index()

    if columns is None:
        columns = list(df.columns)

    # Frequency as a Timedelta (e.g., 30 minutes)
    freq_td = pd.Timedelta(to_offset(expected_freq))
    hours_per_step = freq_td / pd.Timedelta(hours=1)

    records = []

    # Iterate station by station
    stations = df.index.get_level_values(station_level).unique()
    for stn in stations:
        # Slice one station: index becomes time_level
        dfx = df.xs(stn, level=station_level)

        # Ensure the time index is datetime and sorted
        time_idx = pd.to_datetime(dfx.index)
        dfx = dfx.set_index(time_idx).sort_index()
        original_idx = dfx.index

        # Build a complete timeline so *missing timestamps* are turned into NaNs
        full_idx = pd.date_range(
            start=original_idx.min(), end=original_idx.max(), freq=expected_freq
        )
        # Mask telling which timestamps were missing in the original index
        missing_row_mask = pd.Series(
            ~pd.Index(full_idx).isin(original_idx), index=full_idx
        )

        # Reindex to full timeline
        dfr = dfx.reindex(full_idx)

        for col in columns:
            col_na = dfr[col].isna()
            if not col_na.any():
                continue  # no gaps for this column

            # Label contiguous runs (True/False) and keep only True-runs (gaps)
            run_id = (col_na != col_na.shift(1)).cumsum()
            for rid, run_mask in col_na.groupby(run_id):
                if not run_mask.iloc[0]:
                    continue  # this run is of non-NaNs

                run_times = run_mask.index
                gap_start = run_times[0]
                gap_end = run_times[-1]
                n_steps = int(run_mask.sum())

                # Determine the kind of gap: missing timestamps vs NaNs vs mixed
                row_missing_in_run = missing_row_mask.loc[run_times]
                if row_missing_in_run.all():
                    kind = "MissingTimestamp"
                elif not row_missing_in_run.any():
                    kind = "NaN"
                else:
                    kind = "Mixed"

                records.append(
                    {
                        "STATIONID": stn,
                        "COLUMN": col,
                        "GAP_START": gap_start,
                        "GAP_END": gap_end,
                        "N_STEPS_MISSING": n_steps,
                        "HOURS_MISSING": n_steps * hours_per_step,
                        "GAP_KIND": kind,
                    }
                )

    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out = out.sort_values(["STATIONID", "COLUMN", "GAP_START"]).reset_index(
            drop=True
        )
    else:
        # Ensure expected columns even when no gaps
        out = pd.DataFrame(
            columns=[
                "STATIONID",
                "COLUMN",
                "GAP_START",
                "GAP_END",
                "N_STEPS_MISSING",
                "HOURS_MISSING",
                "GAP_KIND",
            ]
        )
    return out


def compare_gap_summaries(
    gaps_a: pd.DataFrame,
    gaps_b: pd.DataFrame,
    expected_freq: str = "30min",
    min_steps: int = 1,
) -> pd.DataFrame:
    """
    Compare two gap-summary DataFrames (from `summarize_gaps`) and highlight
    where one dataset has coverage that could fill the other's gaps.

    Parameters
    ----------
    gaps_a, gaps_b : pd.DataFrame
        DataFrames returned by `summarize_gaps`. Must include the columns:
        ['STATIONID','COLUMN','GAP_START','GAP_END','N_STEPS_MISSING','HOURS_MISSING','GAP_KIND'].
    expected_freq : str, default "30min"
        Sampling frequency. Used to compute discrete step counts and to
        treat intervals on the expected time grid.
    min_steps : int, default 1
        Only report fillable segments with at least this many steps.

    Returns
    -------
    pd.DataFrame
        One row per *fillable segment*.
        Columns:
            - TARGET_DATASET   ("A" or "B")
            - SOURCE_DATASET   ("B" or "A")
            - STATIONID
            - COLUMN
            - TARGET_GAP_START
            - TARGET_GAP_END
            - FILLABLE_START
            - FILLABLE_END
            - N_STEPS_FILLABLE
            - HOURS_FILLABLE
            - TARGET_N_STEPS_MISSING
            - COVERAGE_RATIO    (steps_fillable / TARGET_N_STEPS_MISSING)
            - TARGET_GAP_KIND
    """
    req = {"STATIONID", "COLUMN", "GAP_START", "GAP_END", "N_STEPS_MISSING"}
    for name, g in [("gaps_a", gaps_a), ("gaps_b", gaps_b)]:
        missing = req - set(g.columns)
        if missing:
            raise KeyError(f"{name} missing required columns: {missing}")

    # Normalize dtypes and sort
    def _prep(g):
        g = g.copy()
        g["GAP_START"] = pd.to_datetime(g["GAP_START"])
        g["GAP_END"] = pd.to_datetime(g["GAP_END"])
        if "GAP_KIND" not in g.columns:
            g["GAP_KIND"] = "Unknown"
        return g.sort_values(
            ["STATIONID", "COLUMN", "GAP_START", "GAP_END"]
        ).reset_index(drop=True)

    gaps_a = _prep(gaps_a)
    gaps_b = _prep(gaps_b)

    freq_td = pd.Timedelta(to_offset(expected_freq))
    hours_per_step = freq_td / pd.Timedelta(hours=1)

    # Build a quick lookup: for each (station, column), list of (start, end) gaps
    def _build_lookup(g):
        d = {}
        for (stn, col), sub in g.groupby(["STATIONID", "COLUMN"], sort=False):
            d[(stn, col)] = list(zip(sub["GAP_START"], sub["GAP_END"]))
        return d

    gapsB_lookup = _build_lookup(gaps_b)
    gapsA_lookup = _build_lookup(gaps_a)

    def _steps_inclusive(s, e):
        # number of discrete samples on the regular grid from s..e inclusive
        return int(((e - s) // freq_td) + 1)

    def _subtract_interval(base, subtracts):
        """Given a base [a0,a1] (inclusive, on grid) and a list of
        subtract intervals (inclusive), return list of remaining
        inclusive intervals on the same grid."""
        a0, a1 = base
        if a0 > a1:
            return []
        # Clip subtracts to base
        cl = []
        for s, e in subtracts:
            s1 = max(s, a0)
            e1 = min(e, a1)
            if s1 <= e1:
                cl.append((s1, e1))
        if not cl:
            return [(a0, a1)]
        cl.sort(key=lambda x: x[0])

        segs = []
        cur = a0
        for s, e in cl:
            # segment before s (subtract is inclusive)
            before_end = s - freq_td
            if cur <= before_end:
                segs.append((cur, before_end))
            # skip the subtracted run
            cur = e + freq_td
            if cur > a1:
                break
        if cur <= a1:
            segs.append((cur, a1))
        return segs

    def _direction_fill(target_gaps, source_lookup, target_label, source_label):
        """Compute fillable segments where `source` can fill `target`."""
        out_rows = []
        for _, r in target_gaps.iterrows():
            key = (r["STATIONID"], r["COLUMN"])
            base = (r["GAP_START"], r["GAP_END"])
            subtracts = source_lookup.get(key, [])
            fill_segments = _subtract_interval(base, subtracts)
            for fs, fe in fill_segments:
                steps = _steps_inclusive(fs, fe)
                if steps < min_steps:
                    continue
                out_rows.append(
                    {
                        "TARGET_DATASET": target_label,
                        "SOURCE_DATASET": source_label,
                        "STATIONID": r["STATIONID"],
                        "COLUMN": r["COLUMN"],
                        "TARGET_GAP_START": r["GAP_START"],
                        "TARGET_GAP_END": r["GAP_END"],
                        "FILLABLE_START": fs,
                        "FILLABLE_END": fe,
                        "N_STEPS_FILLABLE": steps,
                        "HOURS_FILLABLE": steps * hours_per_step,
                        "TARGET_N_STEPS_MISSING": int(r["N_STEPS_MISSING"]),
                        "COVERAGE_RATIO": steps / int(r["N_STEPS_MISSING"]),
                        "TARGET_GAP_KIND": r.get("GAP_KIND", "Unknown"),
                    }
                )
        if not out_rows:
            return pd.DataFrame(
                columns=[
                    "TARGET_DATASET",
                    "SOURCE_DATASET",
                    "STATIONID",
                    "COLUMN",
                    "TARGET_GAP_START",
                    "TARGET_GAP_END",
                    "FILLABLE_START",
                    "FILLABLE_END",
                    "N_STEPS_FILLABLE",
                    "HOURS_FILLABLE",
                    "TARGET_N_STEPS_MISSING",
                    "COVERAGE_RATIO",
                    "TARGET_GAP_KIND",
                ]
            )
        return (
            pd.DataFrame(out_rows)
            .sort_values(["STATIONID", "COLUMN", "TARGET_GAP_START", "FILLABLE_START"])
            .reset_index(drop=True)
        )

    # B can fill A (subtract A's gaps by B's gaps)
    fill_B_to_A = _direction_fill(
        gaps_a, gapsB_lookup, target_label="A", source_label="B"
    )
    # A can fill B
    fill_A_to_B = _direction_fill(
        gaps_b, gapsA_lookup, target_label="B", source_label="A"
    )

    # Combine
    combined = pd.concat([fill_B_to_A, fill_A_to_B], ignore_index=True)
    return combined.sort_values(
        ["STATIONID", "COLUMN", "TARGET_DATASET", "TARGET_GAP_START", "FILLABLE_START"]
    ).reset_index(drop=True)
