"""
micromet.qaqc.compare
=====================

Tools for comparing redundant sensors or validating one signal against another.

Features
--------
- Pairwise comparison with OLS or Deming regression (EIV)
- Robust outlier rejection (sigma or MAD-based)
- Rich stats: R, R², slope, intercept, bias, RMSE, MBE, Spearman/Kendall (optional)
- Bland–Altman analysis (mean difference + limits of agreement)
- Rolling (windowed) comparisons to detect drift over time
- Batch utilities for multiple variable pairs

Typical use
-----------
from micromet.qaqc.compare import (
    CompareOptions, compare_pair, rolling_compare, bland_altman, batch_compare
)

res = compare_pair(df["SW_IN_primary"], df["SW_IN_backup"])
print(res)

roll = rolling_compare(df["SW_IN_primary"], df["SW_IN_backup"], window="7D")
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

# Optional SciPy: Kendall/Spearman and ODR; we handle absence gracefully
try:
    from scipy import stats as _spstats

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _spstats = None
    SCIPY_AVAILABLE = False


# ============================================================================
# Options / Results
# ============================================================================


@dataclass
class CompareOptions:
    """Configuration for pairwise comparison."""

    method: str = "ols"  # "ols" | "deming"
    deming_delta: float = 1.0  # variance ratio sigma_x^2 / sigma_y^2
    intercept: bool = True
    min_overlap: int = 30  # minimum aligned samples to compute stats
    dropna: str = "both"  # "both" | "left" | "right"
    outlier_sigma: Optional[float] = 5.0  # None to disable; applies to residuals
    outlier_strategy: str = "mad"  # "mad" | "sigma"
    return_masks: bool = False  # include inlier mask in result


@dataclass
class CompareResult:
    """Summary statistics for a comparison between two aligned series."""

    n: int
    n_inliers: int
    inlier_fraction: float

    slope: float
    intercept: float
    r: float
    r2: float
    spearman_r: float
    kendall_tau: float

    bias: float  # mean(y - x)
    mbe: float  # same as bias, named explicitly
    mae: float
    rmse: float
    residual_std: float

    ba_mean: float  # Bland–Altman mean difference
    ba_lo: float  # lower LoA (mean - 1.96 * sd)
    ba_hi: float  # upper LoA (mean + 1.96 * sd)

    # Optional: masks (boolean arrays aligned with input index)
    inlier_mask: Optional[pd.Series] = None

    def as_dict(self) -> Dict:
        d = asdict(self)
        if self.inlier_mask is not None:
            d["inlier_mask"] = self.inlier_mask.astype(bool).tolist()
        return d


# ============================================================================
# Internals
# ============================================================================


def _align_pair(
    a: pd.Series, b: pd.Series, dropna: str = "both"
) -> Tuple[pd.Series, pd.Series]:
    """Align two series on index and drop NaNs according to policy."""
    a1, b1 = a.align(b, join="inner")
    mask = pd.Series(True, index=a1.index)
    if dropna in ("both", "left"):
        mask &= a1.notna()
    if dropna in ("both", "right"):
        mask &= b1.notna()
    return a1[mask], b1[mask]


def _ols_fit(
    x: np.ndarray, y: np.ndarray, intercept: bool = True
) -> Tuple[float, float]:
    """Simple OLS using numpy; returns slope, intercept."""
    if not intercept:
        # slope-only fit through origin
        denom = np.dot(x, x)
        slope = float(np.dot(x, y) / denom) if denom != 0 else np.nan
        return slope, 0.0
    # add intercept column
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope = float(beta[0])
    intercept_v = float(beta[1])
    return slope, intercept_v


def _deming_fit(
    x: np.ndarray, y: np.ndarray, delta: float = 1.0
) -> Tuple[float, float]:
    """
    Deming regression (errors-in-variables) closed-form for intercept=True case.
    delta = variance ratio sigma_x^2 / sigma_y^2. If SciPy available with ODR, that’s preferred.
    """
    if SCIPY_AVAILABLE:
        # Use ODR with linear model y = m*x + b
        def f(B, x):
            return B[0] * x + B[1]

        model = _spstats.odr.Model(f)  # type: ignore[no-untyped-def]
        data = _spstats.odr.RealData(x, y, sx=np.sqrt(delta), sy=1.0)  # type: ignore[no-untyped-def]
        odr = _spstats.odr.ODR(data, model, beta0=[1.0, 0.0])  # type: ignore[no-untyped-def]
        out = odr.run()
        return float(out.beta[0]), float(out.beta[1])

    # Closed-form Deming (Linnet 1993)
    xbar, ybar = np.mean(x), np.mean(y)
    sxx = np.var(x, ddof=1)
    syy = np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    # Guard against degenerate cases
    if np.isnan(sxx) or np.isnan(syy) or np.isnan(sxy):
        return np.nan, np.nan
    # Slope
    lam = delta
    term = syy - lam * sxx
    slope = (term + np.sqrt(term**2 + 4 * lam * sxy**2)) / (2 * sxy)
    intercept = ybar - slope * xbar
    return float(slope), float(intercept)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    vx = x - x.mean()
    vy = y - y.mean()
    denom = np.sqrt((vx * vx).sum() * (vy * vy).sum())
    return float((vx * vy).sum() / denom) if denom != 0 else np.nan


def _robust_outlier_mask(
    x: np.ndarray,
    y: np.ndarray,
    slope: float,
    intercept: float,
    sigma: Optional[float],
    strategy: str = "mad",
) -> np.ndarray:
    """Return boolean mask of inliers based on residuals."""
    if sigma is None:
        return np.ones_like(x, dtype=bool)
    resid = y - (slope * x + intercept)
    r = resid.copy()
    if strategy == "mad":
        med = np.nanmedian(r)
        mad = np.nanmedian(np.abs(r - med))
        # Consistent MAD estimator for normal dist
        s = 1.4826 * mad if mad > 0 else np.nanstd(r)
    else:
        s = np.nanstd(r, ddof=1)
    s = s if (s is not None and s > 0 and np.isfinite(s)) else np.nan
    if not np.isfinite(s) or s == 0:
        return np.ones_like(x, dtype=bool)
    return np.abs(r - np.nanmean(r)) <= sigma * s


def _basic_errors(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """Return (bias, mbe, mae, rmse, resid_std)."""
    resid = y_true - y_pred
    bias = float(np.nanmean(resid))
    mbe = bias
    mae = float(np.nanmean(np.abs(resid)))
    rmse = float(np.sqrt(np.nanmean(resid**2)))
    resid_std = float(np.nanstd(resid, ddof=1))
    return bias, mbe, mae, rmse, resid_std


def _spearman_kendall(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if not SCIPY_AVAILABLE or x.size < 3:
        return np.nan, np.nan
    try:
        sr = _spstats.spearmanr(x, y, nan_policy="omit").correlation  # type: ignore[no-untyped-call]
        kt = _spstats.kendalltau(x, y, nan_policy="omit").correlation  # type: ignore[no-untyped-call]
        return float(sr), float(kt)
    except Exception:  # pragma: no cover
        return np.nan, np.nan


def _bland_altman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return (mean_diff, lower_loa, upper_loa) with 1.96*sd limits."""
    d = y - x
    m = float(np.nanmean(d))
    sd = float(np.nanstd(d, ddof=1))
    return m, m - 1.96 * sd, m + 1.96 * sd


# ============================================================================
# Public API
# ============================================================================


def compare_pair(
    a: pd.Series, b: pd.Series, options: CompareOptions | None = None
) -> CompareResult:
    """
    Compare two series, returning regression, association, and error metrics.

    - Aligns on index and drops NaN per options.dropna
    - Fits OLS (default) or Deming regression
    - Outlier rejection on residuals (MAD or sigma)
    - Computes R/R², bias/MBE, MAE, RMSE, residual std
    - Computes Spearman/Kendall if SciPy available
    - Computes Bland–Altman mean and limits

    Returns CompareResult. If insufficient overlap, most fields are NaN.
    """
    opts = options or CompareOptions()
    a1, b1 = _align_pair(a, b, dropna=opts.dropna)
    x = a1.to_numpy(dtype=float)
    y = b1.to_numpy(dtype=float)
    n = int(x.size)

    if n < opts.min_overlap:
        # Not enough points to compute stable stats
        return CompareResult(
            n=n,
            n_inliers=0,
            inlier_fraction=0.0,
            slope=np.nan,
            intercept=np.nan,
            r=np.nan,
            r2=np.nan,
            spearman_r=np.nan,
            kendall_tau=np.nan,
            bias=np.nan,
            mbe=np.nan,
            mae=np.nan,
            rmse=np.nan,
            residual_std=np.nan,
            ba_mean=np.nan,
            ba_lo=np.nan,
            ba_hi=np.nan,
            inlier_mask=(
                pd.Series([False] * n, index=a1.index) if opts.return_masks else None
            ),
        )

    # Initial fit
    if opts.method.lower() == "deming":
        slope0, intercept0 = _deming_fit(x, y, delta=opts.deming_delta)
    else:
        slope0, intercept0 = _ols_fit(x, y, intercept=opts.intercept)

    # Outlier rejection on residuals
    inlier_mask = _robust_outlier_mask(
        x,
        y,
        slope0,
        intercept0,
        sigma=opts.outlier_sigma,
        strategy=opts.outlier_strategy,
    )
    xi, yi = x[inlier_mask], y[inlier_mask]
    n_in = int(xi.size)
    inlier_fraction = float(n_in / n) if n > 0 else 0.0

    # Refit with inliers only
    if n_in >= max(3, opts.min_overlap // 3):
        if opts.method.lower() == "deming":
            slope, intercept = _deming_fit(xi, yi, delta=opts.deming_delta)
        else:
            slope, intercept = _ols_fit(xi, yi, intercept=opts.intercept)
    else:
        slope, intercept = slope0, intercept0

    # Final stats
    yhat = slope * xi + intercept
    r = _pearson_r(xi, yi)
    r2 = float(r**2) if np.isfinite(r) else np.nan
    spearman_r, kendall_tau = _spearman_kendall(xi, yi)
    bias, mbe, mae, rmse, resid_std = _basic_errors(yi, yhat)
    ba_mean, ba_lo, ba_hi = _bland_altman(xi, yi)

    mask_series = pd.Series(False, index=a1.index)
    mask_series.loc[a1.index[inlier_mask]] = True

    return CompareResult(
        n=n,
        n_inliers=n_in,
        inlier_fraction=inlier_fraction,
        slope=float(slope),
        intercept=float(intercept),
        r=float(r),
        r2=float(r2),
        spearman_r=spearman_r,
        kendall_tau=kendall_tau,
        bias=bias,
        mbe=mbe,
        mae=mae,
        rmse=rmse,
        residual_std=resid_std,
        ba_mean=ba_mean,
        ba_lo=ba_lo,
        ba_hi=ba_hi,
        inlier_mask=mask_series if opts.return_masks else None,
    )


def bland_altman(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
      mean   = (a+b)/2
      diff   = (b-a)
      mean_d = global mean difference
      lo     = lower limit of agreement
      hi     = upper limit of agreement
    Useful for plotting.
    """
    a1, b1 = _align_pair(a, b, dropna="both")
    mean = (a1 + b1) / 2.0
    diff = b1 - a1
    m, lo, hi = _bland_altman(a1.to_numpy(dtype=float), b1.to_numpy(dtype=float))
    return pd.DataFrame(
        {"mean": mean, "diff": diff, "mean_d": m, "lo": lo, "hi": hi}, index=a1.index
    )


def rolling_compare(
    a: pd.Series,
    b: pd.Series,
    window: str | int = "7D",
    options: CompareOptions | None = None,
) -> pd.DataFrame:
    """
    Rolling window comparison. Returns a time-indexed DataFrame with columns:
      n, n_inliers, slope, intercept, r, r2, bias, mae, rmse, residual_std

    - `window` can be a time offset string (e.g., "3D", "14D") if the index is DateTimeIndex,
      or an integer number of samples.
    - Uses the same CompareOptions as compare_pair (including outlier rejection).
    """
    opts = options or CompareOptions()
    a1, b1 = _align_pair(a, b, dropna=opts.dropna)

    # Choose rolling API based on index type
    if isinstance(a1.index, pd.DatetimeIndex) and isinstance(window, str):
        roller = zip(a1.rolling(window), b1.rolling(window))
        idx = a1.index
    else:
        # sample-count window
        w = int(window)
        roller = (
            (a1.iloc[i - w + 1 : i + 1], b1.iloc[i - w + 1 : i + 1])
            for i in range(len(a1))
        )
        idx = a1.index

    rows = []
    for sa, sb in roller:
        if len(sa) < max(5, opts.min_overlap // 5):
            rows.append((np.nan,) * 10)
            continue
        res = compare_pair(sa, sb, options=opts)
        rows.append(
            (
                res.n_inliers,
                res.slope,
                res.intercept,
                res.r,
                res.r2,
                res.bias,
                res.mae,
                res.rmse,
                res.residual_std,
                res.inlier_fraction,
            )
        )

    out = pd.DataFrame(
        rows,
        index=idx,
        columns=[
            "n_inliers",
            "slope",
            "intercept",
            "r",
            "r2",
            "bias",
            "mae",
            "rmse",
            "residual_std",
            "inlier_frac",
        ],
    )
    return out


def batch_compare(
    df: pd.DataFrame,
    pairs: Iterable[Tuple[str, str]],
    options: CompareOptions | None = None,
) -> Dict[Tuple[str, str], CompareResult]:
    """
    Compare multiple (left,right) column pairs in a single call.

    Example:
        pairs = [("SW_IN_primary","SW_IN_backup"), ("H_main","H_ref")]
        results = batch_compare(df, pairs)
    """
    opts = options or CompareOptions()
    out: Dict[Tuple[str, str], CompareResult] = {}
    for left, right in pairs:
        if left not in df.columns or right not in df.columns:
            out[(left, right)] = CompareResult(
                n=0,
                n_inliers=0,
                inlier_fraction=0.0,
                slope=np.nan,
                intercept=np.nan,
                r=np.nan,
                r2=np.nan,
                spearman_r=np.nan,
                kendall_tau=np.nan,
                bias=np.nan,
                mbe=np.nan,
                mae=np.nan,
                rmse=np.nan,
                residual_std=np.nan,
                ba_mean=np.nan,
                ba_lo=np.nan,
                ba_hi=np.nan,
                inlier_mask=None,
            )
            continue
        out[(left, right)] = compare_pair(df[left], df[right], options=opts)
    return out
