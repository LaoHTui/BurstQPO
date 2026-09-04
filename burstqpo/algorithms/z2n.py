"""Z2n periodogram for event arrival times."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ..results import Z2nResult
from .peaks import find_periodogram_peaks

__all__ = [
    "z2n", "compute_z2n", "find_z2n_peaks", "z2n_single_pvalue",
    "global_pvalue_from_trials", "estimate_z2n_trials",
    "z2n_local_threshold", "z2n_global_threshold",
]


def z2n_single_pvalue(z_value, n_harmonics=1):
    """Return the fixed-frequency tail probability for a Z^2_n value.

    Under the uniform-phase null hypothesis, Z^2_n is asymptotically
    chi-square distributed with ``2 * n_harmonics`` degrees of freedom.
    """
    if int(n_harmonics) != n_harmonics or int(n_harmonics) < 1:
        raise ValueError("n_harmonics must be a positive integer")
    values = np.asarray(z_value, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("z_value must contain finite non-negative values")
    probability = chi2.sf(values, 2 * int(n_harmonics))
    return float(probability) if probability.ndim == 0 else probability


def global_pvalue_from_trials(p_single, n_trials):
    """Apply a Sidak trial correction to one or more local p-values."""
    probabilities = np.asarray(p_single, dtype=float)
    trials = float(n_trials)
    if (np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0)
            or np.any(probabilities > 1)):
        raise ValueError("p_single must contain finite values between 0 and 1")
    if not np.isfinite(trials) or trials < 1:
        raise ValueError("n_trials must be finite and at least one")
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = -np.expm1(trials * np.log1p(-probabilities))
    corrected = np.clip(corrected, 0.0, 1.0)
    return float(corrected) if corrected.ndim == 0 else corrected


def _significance_level(alpha):
    """Parse a tail probability, accepting e.g. ``"0.1%"`` or ``0.001``."""
    if isinstance(alpha, str):
        text = alpha.strip()
        if text.endswith("%"):
            value = float(text[:-1]) / 100.0
        else:
            value = float(text)
    else:
        value = float(alpha)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must be between 0 and 1, e.g. '0.1%' or 0.001")
    return value


def z2n_local_threshold(alpha="0.1%", n_harmonics=1):
    """Return the local Z^2_n threshold for tail probability ``alpha``."""
    probability = _significance_level(alpha)
    if int(n_harmonics) != n_harmonics or int(n_harmonics) < 1:
        raise ValueError("n_harmonics must be a positive integer")
    return float(chi2.isf(probability, 2 * int(n_harmonics)))


def z2n_global_threshold(alpha="0.1%", n_harmonics=1, n_trials=1):
    """Return the Z^2_n threshold after a Sidak trial correction."""
    probability = _significance_level(alpha)
    if int(n_harmonics) != n_harmonics or int(n_harmonics) < 1:
        raise ValueError("n_harmonics must be a positive integer")
    trials = float(n_trials)
    if not np.isfinite(trials) or trials < 1:
        raise ValueError("n_trials must be finite and at least one")
    local_probability = -np.expm1(np.log1p(-probability) / trials)
    return float(chi2.isf(local_probability, 2 * int(n_harmonics)))


def estimate_z2n_trials(event_times, frequencies):
    """Roughly estimate the number of independent searched frequencies.

    The independent Fourier spacing is approximately ``1 / T``, where
    ``T`` is the event-time span.  Thus an evenly sampled frequency grid
    with spacing ``df`` has oversampling factor ``1 / (df * T)``.  The
    estimate is the grid size divided by that factor, capped at the number
    of evaluated frequencies.
    """
    events = np.asarray(event_times, dtype=float)
    frequency = np.asarray(frequencies, dtype=float)
    if events.ndim != 1 or len(events) < 2 or np.any(~np.isfinite(events)):
        raise ValueError("event_times must contain at least two finite values")
    if (frequency.ndim != 1 or len(frequency) == 0
            or np.any(~np.isfinite(frequency)) or np.any(np.diff(frequency) <= 0)):
        raise ValueError("frequencies must be a finite, strictly increasing array")
    if len(frequency) == 1:
        return 1.0
    duration = float(np.max(events) - np.min(events))
    if duration <= 0:
        return 1.0
    frequency_step = float(np.median(np.diff(frequency)))
    oversampling_factor = max(1.0, 1.0 / (frequency_step * duration))
    return float(np.clip(len(frequency) / oversampling_factor, 1.0, len(frequency)))


def z2n(event_times, frequencies=None, *, frequency_min=None, frequency_max=None,
        frequency_step=None, n_harmonics=1, t0=None, time_unit="unknown") -> Z2nResult:
    """Compute the event-based Z^2_n periodogram and analytic p-values.

    Local p-values use the asymptotic chi-square null distribution at each
    fixed frequency. The result also stores a default global p-value based on
    an effective trial count estimated from the duration and grid spacing.
    This trial count does not affect the Z^2_n statistic.
    """
    events = np.asarray(event_times, dtype=float)
    if events.ndim != 1:
        raise ValueError("event_times must be one-dimensional")
    events = np.sort(events[np.isfinite(events)])
    if len(events) < 2:
        raise ValueError("Z2n requires at least two finite event times")
    if int(n_harmonics) != n_harmonics or int(n_harmonics) < 1:
        raise ValueError("n_harmonics must be a positive integer")
    n_harmonics = int(n_harmonics)
    if frequencies is not None and any(v is not None for v in (frequency_min, frequency_max, frequency_step)):
        raise ValueError("pass frequencies or frequency_min/frequency_max/frequency_step, not both")
    if frequencies is None:
        if any(v is None for v in (frequency_min, frequency_max, frequency_step)):
            raise ValueError("frequencies or all frequency grid parameters are required")
        fmin, fmax, step = map(float, (frequency_min, frequency_max, frequency_step))
        if not (np.isfinite(fmin) and np.isfinite(fmax) and np.isfinite(step) and 0 < fmin <= fmax and step > 0):
            raise ValueError("frequency range must satisfy 0 < fmin <= fmax and step > 0")
        frequencies = fmin + step * np.arange(int(np.floor((fmax - fmin) / step + 1e-12)) + 1)
    else:
        frequencies = np.asarray(frequencies, dtype=float)
        if (frequencies.ndim != 1 or len(frequencies) == 0 or not np.all(np.isfinite(frequencies))
                or np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0)):
            raise ValueError("frequencies must be strictly increasing and positive")
    origin = float(events[0] if t0 is None else t0)
    if not np.isfinite(origin):
        raise ValueError("t0 must be finite")
    phase = 2.0 * np.pi * frequencies[:, None] * (events - origin)[None, :]
    statistic = np.zeros(len(frequencies), dtype=float)
    for harmonic in range(1, n_harmonics + 1):
        statistic += np.sum(np.cos(harmonic * phase), axis=1) ** 2
        statistic += np.sum(np.sin(harmonic * phase), axis=1) ** 2
    statistic *= 2.0 / len(events)
    local_pvalue = z2n_single_pvalue(statistic, n_harmonics)
    effective_trials = estimate_z2n_trials(events, frequencies)
    global_pvalue = global_pvalue_from_trials(local_pvalue, effective_trials)
    return Z2nResult(
        events, frequencies, statistic, n_harmonics, origin, str(time_unit),
        float(frequencies[0]), float(frequencies[-1]),
        float(np.median(np.diff(frequencies))) if len(frequencies) > 1 else np.nan,
        len(events),
        local_pvalue,
        global_pvalue,
        effective_trials,
    )


compute_z2n = z2n


def find_z2n_peaks(result: Z2nResult, *, top_n=3, min_period=0.0,
                   prominence=None, distance=1, fit_width=5):
    """Find Z²_n frequency peaks using the shared periodogram detector."""
    peaks = find_periodogram_peaks(
        result.frequency, result.statistic, top_n=top_n, min_period=min_period,
        prominence=prominence, distance=distance, fit_width=fit_width,
    )
    for peak in peaks:
        index = peak["index"]
        peak["local_pvalue"] = float(result.local_pvalue[index])
        peak["global_pvalue"] = float(result.global_pvalues[index])
    return peaks
