"""Weighted Wavelet Z-transform (Foster WWZ).

The numerical functions in this module deliberately operate on plain NumPy
arrays. Unit conversion and application-level dispatch belong in ``detection``.
"""

from __future__ import annotations

import math
import numpy as np
from ..results import WWZResult
from .peaks import find_periodogram_peaks

__all__ = [
    "WWZResult", "wwz", "compute_wwz", "find_wwz_peaks", "project_wwz",
]

try:  # Numba is optional: the NumPy implementation remains the reference API.
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised only in minimal installs
    njit = None
    prange = range


if njit is not None:

    @njit(cache=True)
    def _solve_3x3(a00, a01, a02, a10, a11, a12, a20, a21, a22, b0, b1, b2):
        matrix = np.empty((3, 4), dtype=np.float64)
        matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3] = a00, a01, a02, b0
        matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3] = a10, a11, a12, b1
        matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3] = a20, a21, a22, b2
        for column in range(3):
            pivot = column
            max_abs = abs(matrix[column, column])
            for row in range(column + 1, 3):
                if abs(matrix[row, column]) > max_abs:
                    max_abs = abs(matrix[row, column])
                    pivot = row
            if max_abs < 1e-14:
                return np.nan, np.nan, np.nan
            if pivot != column:
                for k in range(4):
                    temporary = matrix[column, k]
                    matrix[column, k] = matrix[pivot, k]
                    matrix[pivot, k] = temporary
            divisor = matrix[column, column]
            for k in range(column, 4):
                matrix[column, k] /= divisor
            for row in range(3):
                if row != column:
                    factor = matrix[row, column]
                    if factor != 0.0:
                        for k in range(column, 4):
                            matrix[row, k] -= factor * matrix[column, k]
        return matrix[0, 3], matrix[1, 3], matrix[2, 3]

    @njit(parallel=True, cache=True)
    def _wwz_core_numba(series, flux, tau, omega, c):
        tau_len, f_len = tau.shape[0], omega.shape[0]
        n_eff = np.empty((tau_len, f_len), dtype=np.float64)
        vx_out = np.empty((tau_len, f_len), dtype=np.float64)
        vy_out = np.empty((tau_len, f_len), dtype=np.float64)
        amplitude = np.empty((tau_len, f_len), dtype=np.float64)
        cutoff = 1e-12
        for i in prange(tau_len):
            for j in range(f_len):
                w = omega[j]
                sw = w2 = sy = sy2 = sc = ss = scc = sss = scs = scy = ssy = 0.0
                valid = 0
                for k in range(series.shape[0]):
                    phase = w * (series[k] - tau[i])
                    weight = math.exp(-c * phase * phase)
                    if weight <= cutoff:
                        continue
                    co, si, y = math.cos(phase), math.sin(phase), flux[k]
                    sw += weight
                    w2 += weight * weight
                    sy += weight * y
                    sy2 += weight * y * y
                    sc += weight * co
                    ss += weight * si
                    scc += weight * co * co
                    sss += weight * si * si
                    scs += weight * co * si
                    scy += weight * co * y
                    ssy += weight * si * y
                    valid += 1
                if valid == 0 or sw <= 0.0 or w2 <= 0.0:
                    n_eff[i, j] = vx_out[i, j] = vy_out[i, j] = amplitude[i, j] = np.nan
                    continue
                ne = sw * sw / w2
                mean = sy / sw
                vx = sy2 / sw - mean * mean
                n_eff[i, j] = ne
                vx_out[i, j] = vx
                if not np.isfinite(vx) or vx <= 0.0:
                    vy_out[i, j] = amplitude[i, j] = np.nan
                    continue
                s01, s02 = sc / sw, ss / sw
                s11, s12, s22 = scc / sw, scs / sw, sss / sw
                a0, a1, a2 = _solve_3x3(1.0, s01, s02, s01, s11, s12, s02, s12, s22,
                                        mean, scy / sw, ssy / sw)
                if not (np.isfinite(a0) and np.isfinite(a1) and np.isfinite(a2)):
                    vy_out[i, j] = amplitude[i, j] = np.nan
                    continue
                ey = a0 + a1 * s01 + a2 * s02
                ey2 = (a0 * a0 + 2 * a0 * a1 * s01 + 2 * a0 * a2 * s02
                       + a1 * a1 * s11 + 2 * a1 * a2 * s12 + a2 * a2 * s22)
                vy = ey2 - ey * ey
                tolerance = 1e-12 * max(1.0, vx)
                if vy < -tolerance:
                    vy_out[i, j] = amplitude[i, j] = np.nan
                    continue
                vy_out[i, j] = max(0.0, vy)
                amplitude[i, j] = math.sqrt(a1 * a1 + a2 * a2)
        return n_eff, vx_out, vy_out, amplitude
else:  # pragma: no cover - optional-dependency fallback
    _wwz_core_numba = None


def _wwz_core_numpy(series, flux, tau, omega, c):
    """Readable NumPy fallback used when Numba is unavailable."""
    shape = (len(tau), len(omega))
    n_eff = np.full(shape, np.nan)
    vx_out = np.full(shape, np.nan)
    vy_out = np.full(shape, np.nan)
    amplitude = np.full(shape, np.nan)
    for i, center in enumerate(tau):
        for j, w in enumerate(omega):
            phase = w * (series - center)
            weights = np.exp(-c * phase * phase)
            weights[weights <= 1e-12] = 0.0
            sw = weights.sum()
            w2 = np.dot(weights, weights)
            if sw <= 0 or w2 <= 0:
                continue
            mean = np.dot(weights, flux) / sw
            vx = np.dot(weights, flux * flux) / sw - mean * mean
            n_eff[i, j], vx_out[i, j] = sw * sw / w2, vx
            if not np.isfinite(vx) or vx <= 0:
                continue
            co, si = np.cos(phase), np.sin(phase)
            s01, s02 = np.dot(weights, co) / sw, np.dot(weights, si) / sw
            s11 = np.dot(weights, co * co) / sw
            s12 = np.dot(weights, co * si) / sw
            s22 = np.dot(weights, si * si) / sw
            matrix = np.array([[1, s01, s02], [s01, s11, s12], [s02, s12, s22]])
            try:
                a0, a1, a2 = np.linalg.solve(matrix, [mean, np.dot(weights, co * flux) / sw,
                                                       np.dot(weights, si * flux) / sw])
            except np.linalg.LinAlgError:
                continue
            ey = a0 + a1 * s01 + a2 * s02
            ey2 = a0 * a0 + 2 * a0 * a1 * s01 + 2 * a0 * a2 * s02 + a1 * a1 * s11 + 2 * a1 * a2 * s12 + a2 * a2 * s22
            vy = ey2 - ey * ey
            if vy >= -1e-12 * max(1.0, vx):
                vy_out[i, j] = max(0.0, vy)
                amplitude[i, j] = np.hypot(a1, a2)
    return n_eff, vx_out, vy_out, amplitude


def _frequency_grid(frequencies, frequency_parameters):
    if frequencies is not None and frequency_parameters is not None:
        raise ValueError("pass frequencies or frequency_parameters, not both")
    if frequency_parameters is not None:
        params = np.asarray(frequency_parameters, dtype=float)
        if params.size != 3:
            raise ValueError("frequency_parameters must be (fmin, fmax, step)")
        fmin, fmax, step = params
        if not (np.isfinite(fmin) and np.isfinite(fmax) and np.isfinite(step)) or fmin <= 0 or fmax < fmin or step <= 0:
            raise ValueError("frequency_parameters must satisfy 0 < fmin <= fmax and step > 0")
        return fmin + step * np.arange(int(np.floor((fmax - fmin) / step + 1e-12)) + 1)
    if frequencies is None:
        raise ValueError("frequencies or frequency_parameters is required")
    grid = np.asarray(frequencies, dtype=float)
    if grid.ndim != 1 or len(grid) == 0 or not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError("frequencies must be a non-empty one-dimensional positive array")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("frequencies must be strictly increasing")
    return grid


def wwz(time, values, frequencies=None, *, uncertainty=None, frequency_parameters=None,
        tau=None, tau_number=1000, c=0.0125, time_unit="unknown") -> WWZResult:
    """Compute the Foster weighted wavelet Z-transform.

    The input time unit is arbitrary but must be consistent with frequency:
    seconds with Hz, days with ``1/day``, and so on. The returned ``tau`` and
    ``frequency`` arrays use those same numerical units.
    """
    series = np.asarray(time, dtype=float)
    flux = np.asarray(values, dtype=float)
    if series.ndim != 1 or flux.ndim != 1 or len(series) != len(flux):
        raise ValueError("time and values must be one-dimensional arrays of equal length")
    errors = None if uncertainty is None else np.asarray(uncertainty, dtype=float)
    if errors is not None:
        if errors.ndim != 1 or len(errors) != len(series):
            raise ValueError("uncertainty must be one-dimensional and have the same length as time")
        if np.any(np.isfinite(errors) & (errors < 0)):
            raise ValueError("finite uncertainty values must be non-negative")
    finite = np.isfinite(series) & np.isfinite(flux)
    if errors is not None:
        errors = errors[finite]
    series, flux = series[finite], flux[finite]
    if len(series) < 4:
        raise ValueError("WWZ requires at least four finite observations")
    order = np.argsort(series)
    series, flux = np.ascontiguousarray(series[order]), np.ascontiguousarray(flux[order])
    if errors is not None:
        errors = np.ascontiguousarray(errors[order])
    time_origin = float(series[0])
    original_series = series
    series = series - time_origin
    if not np.isfinite(c) or float(c) <= 0:
        raise ValueError("c must be finite and greater than zero")
    c = float(c)
    freq = _frequency_grid(frequencies, frequency_parameters)
    if tau is None:
        if int(tau_number) != tau_number or int(tau_number) < 1:
            raise ValueError("tau_number must be a positive integer")
        tau_values = np.linspace(series[0], series[-1], int(tau_number))
    else:
        tau_values = np.asarray(tau, dtype=float)
        # The data series is shifted so that the first observation is zero;
        # apply the same origin to a user-supplied tau grid.
        if tau_values.ndim == 1 and np.all(np.isfinite(tau_values)):
            tau_values = tau_values - time_origin
    if tau_values.ndim != 1 or len(tau_values) == 0 or not np.all(np.isfinite(tau_values)):
        raise ValueError("tau must be a non-empty one-dimensional finite array")
    omega = 2.0 * np.pi * freq
    if _wwz_core_numba is not None:
        n_eff, vx, vy, amplitude = _wwz_core_numba(series, flux, tau_values, omega, c)
    else:
        n_eff, vx, vy, amplitude = _wwz_core_numpy(series, flux, tau_values, omega, c)
    denominator = 2.0 * (vx - vy)
    power = np.full_like(vx, np.nan)
    valid = np.isfinite(denominator) & np.isfinite(n_eff) & np.isfinite(vy) & (denominator > 0) & (n_eff > 3) & (vy >= 0)
    power[valid] = (n_eff[valid] - 3.0) * vy[valid] / denominator[valid]
    mid = (series[0] + series[-1]) / 2.0
    delta = 1.0 / (2.0 * np.pi * freq * np.sqrt(c))
    coi = np.array([np.minimum(series[0] + delta, mid), np.maximum(series[-1] - delta, mid)])
    frequency_step = float(np.median(np.diff(freq))) if len(freq) > 1 else np.nan
    # Keep the public grid and COI in the caller's time coordinates. The WWZ
    # kernel above uses the origin-shifted arrays for numerical conditioning.
    result = WWZResult(
        tau_values + time_origin, freq, power, coi + time_origin,
        float(np.pi * np.sqrt(c) * np.ptp(series)), amplitude, n_eff, c,
        str(time_unit), time_origin, float(freq[0]), float(freq[-1]),
        frequency_step, int(len(tau_values)), int(len(series)), original_series, flux, errors,
    )
    return result


# Backward-compatible alias. New code should use the concise method name.
compute_wwz = wwz


def project_wwz(power, tau, frequency, c=0.0125, use_coi=True) -> np.ndarray:
    """Average WWZ power over time.

    Parameters
    ----------
    use_coi : bool, default=True
        If true, exclude the cone-of-influence region before averaging.
    """
    matrix, taus, freqs = np.asarray(power, dtype=float), np.asarray(tau, dtype=float), np.asarray(frequency, dtype=float)
    if matrix.shape != (len(taus), len(freqs)):
        raise ValueError("power shape must be (len(tau), len(frequency))")
    if c <= 0 or not np.isfinite(c):
        raise ValueError("c must be finite and greater than zero")
    masked = matrix.copy()
    if use_coi:
        for j, freq in enumerate(freqs):
            if not np.isfinite(freq) or freq <= 0:
                masked[:, j] = np.nan
                continue
            delta = 1.0 / (2.0 * np.pi * freq * np.sqrt(c))
            masked[(taus < taus.min() + delta) | (taus > taus.max() - delta), j] = np.nan
    else:
        masked[:, ~np.isfinite(freqs) | (freqs <= 0)] = np.nan
    valid_count = np.sum(np.isfinite(masked), axis=0)
    totals = np.nansum(masked, axis=0)
    projection = np.full(len(freqs), np.nan, dtype=float)
    valid_columns = valid_count > 0
    projection[valid_columns] = totals[valid_columns] / valid_count[valid_columns]
    return projection


def find_wwz_peaks(result: WWZResult, *, top_n=3, min_period=0.0, prominence=None, distance=1,
                   fit_width=5, use_coi=True) -> list[dict]:
    """Find strong projected WWZ peaks and estimate period uncertainty.

    Parameters
    ----------
    use_coi : bool, default=True
        If true, find peaks from the projection after excluding COI.
    """
    projection = project_wwz(
        result.power, result.tau, result.frequency, result._c, use_coi=use_coi
    )
    return find_periodogram_peaks(
        result.frequency, projection, top_n=top_n, min_period=min_period,
        prominence=prominence, distance=distance, fit_width=fit_width,
    )
