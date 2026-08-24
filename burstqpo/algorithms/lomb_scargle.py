"""Lomb--Scargle and generalized Lomb--Scargle periodograms."""

from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle

from ..results import LombScargleResult
from .peaks import find_periodogram_peaks

__all__ = ["lomb_scargle", "lsp", "compute_lomb_scargle", "compute_lsp", "find_lomb_scargle_peaks"]


def _clean_inputs(time, values, uncertainty):
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
        raise ValueError("time and values must be one-dimensional arrays of equal length")
    dy = None if uncertainty is None else np.asarray(uncertainty, dtype=float)
    if dy is not None and (dy.ndim != 1 or len(dy) != len(t)):
        raise ValueError("uncertainty must have the same length as time")
    mask = np.isfinite(t) & np.isfinite(y)
    # Keep the uncertainty array aligned with the cleaned observations.
    # Its validity is evaluated separately so mode='auto' can correctly
    # fall back to ordinary LSP when even one error value is unusable.
    t, y = t[mask], y[mask]
    dy = None if dy is None else dy[mask]
    if len(t) < 3:
        raise ValueError("Lomb--Scargle requires at least three finite observations")
    order = np.argsort(t, kind="mergesort")
    return t[order], y[order], None if dy is None else dy[order]


def _make_frequency_grid(t, frequencies, frequency_min, frequency_max, frequency_step, divide_freq_step):
    supplied = frequencies is not None
    params = [frequency_min is not None, frequency_max is not None, frequency_step is not None]
    if supplied and any(params):
        raise ValueError("pass frequencies or frequency_min/frequency_max/frequency_step, not both")
    if supplied:
        f = np.asarray(frequencies, dtype=float)
        if f.ndim != 1 or len(f) == 0 or not np.all(np.isfinite(f)) or np.any(f <= 0) or np.any(np.diff(f) <= 0):
            raise ValueError("frequencies must be a strictly increasing positive array")
        return f
    duration = float(t[-1] - t[0])
    default_min = 1.0 / duration
    fmin = default_min if frequency_min is None else float(frequency_min)
    if frequency_max is None:
        fmax = fmin * len(t) / 2.0
        if fmax <= fmin:
            fmax = fmin * 10.0
    else:
        fmax = float(frequency_max)
    if frequency_step is None:
        div = float(divide_freq_step)
        if not np.isfinite(div) or div <= 0:
            raise ValueError("divide_freq_step must be finite and greater than zero")
        step = fmin / div
    else:
        step = float(frequency_step)
    if not (np.isfinite(fmin) and np.isfinite(fmax) and np.isfinite(step) and 0 < fmin <= fmax and step > 0):
        raise ValueError("frequency range must satisfy 0 < fmin <= fmax and step > 0")
    return fmin + step * np.arange(int(np.floor((fmax - fmin) / step + 1e-12)) + 1)


def lomb_scargle(time, values, uncertainty=None, *, frequencies=None, frequency_min=None,
                 frequency_max=None, frequency_step=None, divide_freq_step=10.0,
                 mode="auto", fit_mean=True, center_data=True, normalization="standard",
                 time_unit="unknown") -> LombScargleResult:
    """Compute an LSP/GLSP and return a reusable result object.

    ``mode='auto'`` selects GLSP when all supplied uncertainties are finite and
    positive; otherwise it selects ordinary LSP.
    """
    t, y, dy = _clean_inputs(time, values, uncertainty)
    requested = str(mode).lower()
    if requested not in {"auto", "lsp", "glsp"}:
        raise ValueError("mode must be 'auto', 'lsp', or 'glsp'")
    errors_valid = dy is not None and np.all(np.isfinite(dy) & (dy > 0))
    selected = "glsp" if requested == "auto" and errors_valid else ("lsp" if requested == "auto" else requested)
    if selected == "glsp" and not errors_valid:
        raise ValueError("mode='glsp' requires finite, strictly positive uncertainty values")
    if selected == "lsp":
        dy = None
    f = _make_frequency_grid(t, frequencies, frequency_min, frequency_max, frequency_step, divide_freq_step)
    ls = LombScargle(t, y, dy=dy if selected == "glsp" else None, fit_mean=bool(fit_mean),
                     center_data=bool(center_data), normalization=normalization)
    power = np.asarray(ls.power(f), dtype=float)
    step = float(np.median(np.diff(f))) if len(f) > 1 else np.nan
    return LombScargleResult(t, y, f, power, selected, dy, str(time_unit), float(t[0]),
                             float(f[0]), float(f[-1]), step, float(divide_freq_step),
                             str(normalization), bool(fit_mean))


compute_lomb_scargle = lomb_scargle
lsp = lomb_scargle
compute_lsp = lomb_scargle


def find_lomb_scargle_peaks(result: LombScargleResult, *, top_n=3, min_period=0.0,
                            prominence=None, distance=1, fit_width=5):
    """Find LSP/GLSP candidates from the computed power spectrum."""
    return find_periodogram_peaks(
        result.frequency, result.power, top_n=top_n, min_period=min_period,
        prominence=prominence, distance=distance, fit_width=fit_width,
    )
