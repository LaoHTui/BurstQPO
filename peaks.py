"""Shared peak detection for one-dimensional periodograms."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def _gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def find_periodogram_peaks(frequency, power, *, top_n=3, min_period=0.0,
                           prominence=None, distance=1, fit_width=5,
                           coordinate_is_frequency=True):
    """Find and refine peaks in any one-dimensional frequency spectrum.

    This is the common implementation used by both WWZ projections and
    Lomb--Scargle periodograms. Gaussian-fit errors are one-sigma values.
    """
    freq = np.asarray(frequency, dtype=float)
    spectrum = np.asarray(power, dtype=float)
    if freq.ndim != 1 or spectrum.ndim != 1 or len(freq) != len(spectrum):
        raise ValueError("frequency and power must be one-dimensional arrays of equal length")
    if (len(freq) == 0 or np.any(~np.isfinite(freq))
            or np.any(freq < 0 if not coordinate_is_frequency else freq <= 0)
            or np.any(np.diff(freq) <= 0)):
        raise ValueError("coordinate must be strictly increasing and non-negative")
    if int(top_n) < 1 or int(distance) < 1 or int(fit_width) < 1:
        raise ValueError("top_n, distance, and fit_width must be positive integers")

    finite = np.isfinite(spectrum)
    if not np.any(finite):
        return []
    peak_prominence = (max(0.0, float(np.nanmax(spectrum)) * 0.05)
                       if prominence is None else float(prominence))
    indices, _ = find_peaks(
        np.where(finite, spectrum, -np.inf),
        prominence=peak_prominence, distance=int(distance),
    )

    output = []
    step = float(np.median(np.diff(freq))) if len(freq) > 1 else np.nan
    for index in indices:
        raw_frequency = float(freq[index])
        if ((1.0 / raw_frequency < float(min_period)) if coordinate_is_frequency
                else (raw_frequency < float(min_period))):
            continue

        best_frequency = raw_frequency
        frequency_error = step if np.isfinite(step) else np.nan
        left = max(0, index - int(fit_width))
        right = min(len(freq), index + int(fit_width) + 1)
        fit_x, fit_y = freq[left:right], spectrum[left:right]
        valid_fit = np.isfinite(fit_y)
        if np.count_nonzero(valid_fit) >= 4 and np.isfinite(step) and step > 0:
            try:
                fitted, _ = curve_fit(
                    _gaussian, fit_x[valid_fit], fit_y[valid_fit],
                    p0=[spectrum[index], raw_frequency, 2.0 * step],
                    bounds=([0.0, fit_x[valid_fit].min(), np.finfo(float).eps],
                            [np.inf, fit_x[valid_fit].max(), np.inf]),
                    maxfev=2000,
                )
                best_frequency = float(fitted[1])
                frequency_error = float(abs(fitted[2]))
            except (RuntimeError, ValueError, FloatingPointError):
                pass

        period = 1.0 / best_frequency
        period_error = frequency_error / best_frequency ** 2
        output.append({
            "frequency": best_frequency,
            "frequency_err": frequency_error,
            "period": period,
            "period_err": period_error,
            "power": float(spectrum[index]),
            "index": int(index),
        })
    return sorted(output, key=lambda item: item["power"], reverse=True)[:int(top_n)]
