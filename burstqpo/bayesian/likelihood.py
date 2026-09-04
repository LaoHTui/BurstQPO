"""Gaussian-process likelihoods."""

import numpy as np

__all__ = ["gp_log_likelihood"]


def gp_log_likelihood(theta, time, counts, errors, mean_function, kernel_function):
    """Return the Gaussian-process log likelihood for a dense covariance."""
    time = np.asarray(time, dtype=float)
    counts = np.asarray(counts, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if time.ndim != 1 or counts.ndim != 1 or errors.ndim != 1 or not (time.size == counts.size == errors.size):
        return -np.inf
    if not time.size or np.any(~np.isfinite(time)) or np.any(~np.isfinite(counts)) or np.any(~np.isfinite(errors)) or np.any(errors <= 0):
        return -np.inf
    try:
        mean = np.asarray(mean_function(time, theta), dtype=float)
        covariance = np.asarray(kernel_function(time, theta), dtype=float)
    except (ValueError, TypeError, FloatingPointError, OverflowError):
        return -np.inf
    if mean.shape != counts.shape or covariance.shape != (counts.size, counts.size) or np.any(~np.isfinite(mean)) or np.any(~np.isfinite(covariance)):
        return -np.inf
    covariance = 0.5 * (covariance + covariance.T)
    covariance = covariance.copy()
    covariance[np.diag_indices_from(covariance)] += errors ** 2
    try:
        cholesky = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(cholesky, counts - mean)
    except np.linalg.LinAlgError:
        return -np.inf
    log_determinant = 2.0 * np.sum(np.log(np.diag(cholesky)))
    return float(-0.5 * (np.dot(whitened, whitened) + log_determinant + counts.size * np.log(2.0 * np.pi)))
