"""Jurkevich phase-binning period search."""

from __future__ import annotations

import numpy as np

from ..results import JurkevichResult
from .peaks import find_periodogram_peaks

__all__ = ["jurkevich", "compute_jurkevich", "find_jurkevich_peaks"]


def _clean_inputs(time, values):
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
        raise ValueError("time and values must be one-dimensional arrays of equal length")
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < 3:
        raise ValueError("Jurkevich requires at least three finite observations")
    order = np.argsort(t, kind="mergesort")
    return t[order], y[order]


def _period_grid(periods, t):
    if periods is None:
        raise ValueError("periods is required")
    grid = np.asarray(periods, dtype=float)
    if grid.ndim != 1 or len(grid) == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("periods must be a non-empty one-dimensional finite array")
    if np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise ValueError("periods must be strictly increasing and greater than zero")
    if t[-1] <= t[0]:
        raise ValueError("time must span a positive duration")
    return grid


def jurkevich(time, values, periods, *, m=10, time_unit="unknown") -> JurkevichResult:
    """Compute the normalized Jurkevich within-phase variance.

    Smaller ``v_norm = V_m^2 / V^2`` indicates a stronger periodic signal.
    ``m`` is the number of phase bins.
    """
    t, y = _clean_inputs(time, values)
    period_grid = _period_grid(periods, t)
    if int(m) != m or int(m) < 2:
        raise ValueError("m must be an integer greater than or equal to 2")
    m = int(m)
    total_variance = float(np.var(y))
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise ValueError("values must have non-zero finite variance")

    relative_time = t - t[0]
    v_norm = np.full(len(period_grid), np.nan, dtype=float)
    for i, period in enumerate(period_grid):
        phases = (relative_time / period) % 1.0
        bins = np.minimum((phases * m).astype(np.int64), m - 1)
        counts = np.bincount(bins, minlength=m).astype(float)
        sums = np.bincount(bins, weights=y, minlength=m)
        squared_sums = np.bincount(bins, weights=y * y, minlength=m)
        valid = counts > 1
        if np.any(valid):
            within_variance = squared_sums[valid] - sums[valid] ** 2 / counts[valid]
            # Match the reference implementation: singleton bins are excluded
            # from both the numerator and the effective sample count.
            v_norm[i] = np.sum(np.maximum(within_variance, 0.0)) / (np.sum(counts[valid]) * total_variance)
    return JurkevichResult(
        t, y, period_grid, v_norm, m, str(time_unit), float(t[0]),
        float(period_grid[0]), float(period_grid[-1]),
        float(np.median(np.diff(period_grid))) if len(period_grid) > 1 else np.nan,
        len(y), total_variance,
    )


compute_jurkevich = jurkevich


def find_jurkevich_peaks(result: JurkevichResult, *, top_n=3, min_period=0.0,
                          prominence=None, distance=1, fit_width=5):
    """Find the lowest Jurkevich variance minima using the shared peak logic."""
    score = 1.0 - np.asarray(result.v_norm, dtype=float)
    candidates = find_periodogram_peaks(
        result.period, score, top_n=top_n, min_period=min_period,
        prominence=prominence, distance=distance, fit_width=fit_width,
        coordinate_is_frequency=False,
    )
    output = []
    for candidate in candidates:
        index = candidate["index"]
        period = float(candidate["frequency"])
        period_error = float(candidate["frequency_err"])
        output.append({
            "frequency": 1.0 / period,
            "frequency_err": period_error / period ** 2,
            "period": period,
            "period_err": period_error,
            "v_norm": float(result.v_norm[index]),
            "statistic": float(result.v_norm[index]),
            "power": float(score[index]),
            "score": float(score[index]),
            "index": index,
        })
    return output
