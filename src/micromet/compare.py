# micromet/compare.py
# -*- coding: utf-8 -*-
"""
Relationship comparison and coordinated outlier plots (SciPy version).

- Aligns two time series (treats -9999 as NaN)
- Fits y ~ x with scipy.stats.linregress
- Flags outliers from residuals using robust MAD (or STD)
- Produces three coordinated plots:
  (1) scatter + regression line + highlighted outliers
  (2) x time series with the same outliers highlighted
  (3) y time series with the same outliers highlighted
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------
# Data containers
# ---------------------------


@dataclass
class FitResult:
    """Linear regression summary."""

    coef: float
    intercept: float
    r2: float
    y_hat: np.ndarray
    residuals: np.ndarray


# ---------------------------
# Utilities
# ---------------------------


def _to_series(
    obj: Union[pd.Series, pd.DataFrame, np.ndarray, Iterable],
    index: Optional[pd.DatetimeIndex] = None,
    name: Optional[str] = None,
) -> pd.Series:
    """Coerce input to a pandas Series and normalize missing values."""
    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError("DataFrame must have exactly 1 column.")
        s = obj.iloc[:, 0].copy()
        if name is None:
            name = obj.columns[0]
        s.name = name
    else:
        s = pd.Series(obj, index=index, name=name)  # type: ignore

    # Standardize sentinel to NaN
    s = s.replace(-9999, np.nan)
    return s


def align(
    x: Union[pd.Series, pd.DataFrame, np.ndarray, Iterable],
    y: Union[pd.Series, pd.DataFrame, np.ndarray, Iterable],
    x_index: Optional[pd.DatetimeIndex] = None,
    y_index: Optional[pd.DatetimeIndex] = None,
    x_name: str = "X",
    y_name: str = "Y",
    how: str = "inner",
) -> pd.DataFrame:
    """
    Align two series on their index and drop rows with NaNs.
    """
    sx = _to_series(x, index=x_index, name=x_name)
    sy = _to_series(y, index=y_index, name=y_name)
    df = pd.concat({x_name: sx, y_name: sy}, axis=1, join=how)  # type: ignore
    return df.dropna(how="any")


def fit_linear(
    x: Union[pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray],
) -> FitResult:
    """
    Fit linear model y ~ x using scipy.stats.linregress.
    """
    X = np.asarray(x).ravel()
    Y = np.asarray(y).ravel()
    lr = stats.linregress(
        X, Y
    )  # slope, intercept, rvalue, pvalue, stderr, intercept_stderr
    y_hat = lr.slope * X + lr.intercept  # type: ignore
    resid = Y - y_hat
    r2 = float(lr.rvalue**2)  # type: ignore

    return FitResult(
        coef=float(lr.slope),  # type: ignore
        intercept=float(lr.intercept),  # type: ignore
        r2=r2,
        y_hat=y_hat,
        residuals=resid,
    )


def outlier_mask_from_residuals(
    residuals: Union[pd.Series, np.ndarray],
    method: Literal["mad", "std"] = "mad",
    k: float = 3.0,
) -> np.ndarray:
    """
    Flag outliers from residuals using MAD (robust, default) or STD.
    """
    r = np.asarray(residuals).ravel()

    if method == "mad":
        # SciPy returns MAD; with scale="normal" it’s already scaled to sigma-equivalent.
        # nan_policy="omit" ensures robustness in the unlikely presence of NaNs.
        sigma = float(stats.median_abs_deviation(r, scale="normal", nan_policy="omit"))  # type: ignore
        # fallback if all residuals are identical (sigma==0)
        if sigma == 0:
            sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
        center = float(np.nanmedian(r))
        thr = k * sigma
        mask = np.abs(r - center) > thr
    elif method == "std":
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        thr = k * sigma
        mask = np.abs(r - mu) > thr
    else:
        raise ValueError("method must be 'mad' or 'std'.")

    return mask


# ---------------------------
# Plotting helpers
# ---------------------------


def _scatter_with_fit(
    ax: plt.Axes,  # type: ignore
    x: pd.Series,
    y: pd.Series,
    fit: FitResult,
    outliers: np.ndarray,
    x_label: str,
    y_label: str,
    point_size: int = 8,
) -> None:
    ax.scatter(x, y, s=point_size, color="lightgray", alpha=0.7, label="data")

    # regression line
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    y_line = fit.coef * x_line + fit.intercept
    ax.plot(x_line, y_line, linewidth=2, label="Linear regression")

    # outliers
    ax.scatter(
        x.iloc[outliers],
        y.iloc[outliers],
        s=40,
        facecolors="none",
        edgecolors="purple",
        linewidths=1.6,
        label="Identified outliers",
        zorder=3,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def _timeseries_panel(
    ax: plt.Axes,  # type: ignore
    t: pd.DatetimeIndex,
    v: pd.Series,
    outliers: np.ndarray,
    ylabel: str,
    point_size: int = 8,
) -> None:
    ax.scatter(t, v, s=point_size, color="lightgray", alpha=0.7)
    ax.scatter(
        t[outliers],
        v.iloc[outliers],
        s=40,
        facecolors="none",
        edgecolors="purple",
        linewidths=1.6,
        zorder=3,
    )
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


# ---------------------------
# Public API
# ---------------------------


def compare_and_plot(
    x: Union[pd.Series, pd.DataFrame, np.ndarray, Iterable],
    y: Union[pd.Series, pd.DataFrame, np.ndarray, Iterable],
    *,
    x_index: Optional[pd.DatetimeIndex] = None,
    y_index: Optional[pd.DatetimeIndex] = None,
    x_label: str = "X",
    y_label: str = "Y",
    title: Optional[str] = None,
    method: Literal["mad", "std"] = "mad",
    k: float = 3.0,
    join: str = "inner",
    point_size: int = 8,
) -> Tuple[plt.Figure, dict]:  # type: ignore
    """
    Align, fit (SciPy), detect outliers, and render three coordinated plots.
    """
    df = align(
        x, y, x_index=x_index, y_index=y_index, x_name=x_label, y_name=y_label, how=join
    )
    if df.empty:
        raise ValueError("No overlapping, non-NaN data between inputs.")

    fit = fit_linear(df[x_label], df[y_label])
    mask = outlier_mask_from_residuals(fit.residuals, method=method, k=k)
    mask_series = pd.Series(mask, index=df.index, name="outlier")

    # Layout
    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2, ncols=2, width_ratios=[1.1, 1.4], height_ratios=[1, 1]
    )

    # (1) Scatter (left, spans both rows)
    ax_sc = fig.add_subplot(gs[:, 0])
    _scatter_with_fit(
        ax_sc, df[x_label], df[y_label], fit, mask, x_label, y_label, point_size
    )
    if title:
        ax_sc.set_title(title)

    # (2) Top-right: x time series
    ax_x = fig.add_subplot(gs[0, 1])
    _timeseries_panel(ax_x, df.index, df[x_label], mask, x_label, point_size)  # type: ignore

    # (3) Bottom-right: y time series (sharex)
    ax_y = fig.add_subplot(gs[1, 1], sharex=ax_x)
    _timeseries_panel(ax_y, df.index, df[y_label], mask, y_label, point_size)  # type: ignore

    for lbl in ax_x.get_xticklabels():
        lbl.set_visible(False)

    # Caption with fit stats
    fig.suptitle(
        f"Linear fit: {y_label} = {fit.coef:.3g} · {x_label} + {fit.intercept:.3g}   (R² = {fit.r2:.3f})",
        y=1.02,
        fontsize=11,
    )

    results = {
        "data": df,
        "mask_outliers": mask_series,
        "fit": fit,
        "threshold_k": k,
        "method": method,
    }
    return fig, results


def compare_report(
    x: Union[pd.Series, pd.DataFrame, np.ndarray, Iterable],
    y: Union[pd.Series, pd.DataFrame, np.ndarray, Iterable],
    **kwargs,
) -> pd.DataFrame:
    """
    Return a tidy per-record report with predictions, residuals, and outlier flags.
    """
    fig, res = compare_and_plot(x, y, **kwargs)
    df = res["data"].copy()
    fit: FitResult = res["fit"]
    df["y_hat"] = fit.y_hat
    df["residual"] = fit.residuals
    df["outlier"] = res["mask_outliers"].reindex(df.index)
    plt.close(fig)
    return df


__all__ = [
    "FitResult",
    "align",
    "fit_linear",
    "outlier_mask_from_residuals",
    "compare_and_plot",
    "compare_report",
]
