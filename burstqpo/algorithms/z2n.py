"""Z2n periodogram for event arrival times."""

from __future__ import annotations

import numpy as np

from ..results import Z2nResult
from .peaks import find_periodogram_peaks

__all__ = ["z2n", "compute_z2n", "find_z2n_peaks"]


def z2n(event_times, frequencies=None, *, frequency_min=None, frequency_max=None,
        frequency_step=None, n_harmonics=1, t0=None, time_unit="unknown") -> Z2nResult:
    """Compute the event-based Z^2_n periodogram."""
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
    return Z2nResult(
        events, frequencies, statistic, n_harmonics, origin, str(time_unit),
        float(frequencies[0]), float(frequencies[-1]),
        float(np.median(np.diff(frequencies))) if len(frequencies) > 1 else np.nan,
        len(events),
    )


compute_z2n = z2n


def find_z2n_peaks(result: Z2nResult, *, top_n=3, min_period=0.0,
                   prominence=None, distance=1, fit_width=5):
    """Find Z²_n frequency peaks using the shared periodogram detector."""
    return find_periodogram_peaks(
        result.frequency, result.statistic, top_n=top_n, min_period=min_period,
        prominence=prominence, distance=distance, fit_width=fit_width,
    )
