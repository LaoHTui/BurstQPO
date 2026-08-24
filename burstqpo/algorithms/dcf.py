"""Discrete correlation function for a single unevenly sampled series."""

from __future__ import annotations

import numpy as np

from ..results import DCFResult
from .peaks import find_periodogram_peaks

__all__ = ["dcf", "compute_dcf", "find_dcf_peaks"]


def _clean_inputs(time, values):
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
        raise ValueError("time and values must be one-dimensional arrays of equal length")
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < 3:
        raise ValueError("DCF requires at least three finite observations")
    order = np.argsort(t, kind="mergesort")
    t, y = t[order], y[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must contain strictly increasing values")
    return t, y


def dcf(time, values, delta_tau, c, max_tau, *, time_unit="unknown") -> DCFResult:
    """Compute the Edelson--Krolik discrete autocorrelation function.

    ``delta_tau`` is the lag-grid step, ``c`` is the lag-bin width, and
    ``max_tau`` is the largest non-negative lag to evaluate.
    """
    t, y = _clean_inputs(time, values)
    delta_tau, c, max_tau = float(delta_tau), float(c), float(max_tau)
    if not np.isfinite(delta_tau) or delta_tau <= 0:
        raise ValueError("delta_tau must be finite and greater than zero")
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and greater than zero")
    if not np.isfinite(max_tau) or max_tau < 0:
        raise ValueError("max_tau must be finite and non-negative")
    std = float(np.std(y, ddof=1))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("values must have non-zero finite standard deviation")
    normalized = (y - np.mean(y)) / std
    lag = np.arange(0.0, max_tau + delta_tau * 0.5, delta_tau)
    dt_matrix = t[:, None] - t[None, :]
    udcf = normalized[:, None] * normalized[None, :]
    correlation = np.full(len(lag), np.nan, dtype=float)
    error = np.full(len(lag), np.nan, dtype=float)
    pair_count = np.zeros(len(lag), dtype=int)
    for index, center in enumerate(lag):
        selected = udcf[(dt_matrix >= center - c / 2.0) & (dt_matrix < center + c / 2.0)]
        pair_count[index] = len(selected)
        if len(selected) > 1:
            correlation[index] = np.mean(selected)
            error[index] = np.sqrt(np.sum((selected - correlation[index]) ** 2)) / (len(selected) - 1)
    return DCFResult(
        t, y, lag, correlation, error, pair_count, delta_tau, c, max_tau,
        str(time_unit), float(t[0]), len(t), std,
    )


compute_dcf = dcf


def _fwhm_error(lag, correlation, index):
    peak = correlation[index]
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    half = peak / 2.0
    left = index
    while left > 0 and np.isfinite(correlation[left - 1]) and correlation[left - 1] > half:
        left -= 1
    right = index
    while right < len(correlation) - 1 and np.isfinite(correlation[right + 1]) and correlation[right + 1] > half:
        right += 1
    if right == left:
        return 2.0 * (lag[1] - lag[0]) if len(lag) > 1 else np.nan
    return 0.5 * (lag[right] - lag[left])


def find_dcf_peaks(result: DCFResult, *, top_n=3, min_period=0.0,
                   prominence=None, distance=1):
    """Find positive DCF peaks and estimate their lag uncertainty by FWHM."""
    lag, correlation = result.lag, result.correlation
    candidates = find_periodogram_peaks(
        lag, correlation, top_n=top_n, min_period=min_period,
        prominence=prominence, distance=distance, coordinate_is_frequency=False,
    )
    output = []
    for candidate in candidates:
        index = candidate["index"]
        uncertainty = _fwhm_error(lag, correlation, index)
        output.append({
            "lag": float(lag[index]), "lag_err": float(uncertainty),
            "period": float(lag[index]), "period_err": float(uncertainty),
            "correlation": float(correlation[index]),
            "power": float(correlation[index]), "index": int(index),
        })
    return sorted(output, key=lambda item: item["correlation"], reverse=True)[:int(top_n)]
