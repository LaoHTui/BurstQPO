"""Stationary covariance kernels for Gaussian-process inference."""

import numpy as np

__all__ = [
    "k_red_noise", "k_qpo", "add_kernels", "qpo_conditional"
]


def _indices(parameter_indices, expected, kernel_name):
    indices = tuple(int(index) for index in parameter_indices)
    if len(indices) != expected:
        raise ValueError(
            f"{kernel_name} requires exactly {expected} parameter indices"
        )
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError("parameter_indices must contain unique non-negative indices")
    return indices


def _inputs(time, theta, indices):
    time = np.asarray(time, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if theta.ndim != 1:
        raise ValueError("theta must be one-dimensional")
    if indices and max(indices) >= theta.size:
        raise ValueError("parameter_indices refer outside theta")
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(theta)):
        raise ValueError("time and theta must contain only finite values")
    return time, theta


def k_red_noise(parameter_indices):
    """Create ``a * exp(-c |dt|)`` using log-amplitude/log-rate parameters."""
    indices = _indices(parameter_indices, 2, "k_red_noise")
    log_a_index, log_c_index = indices

    def kernel(time, theta):
        time, theta = _inputs(time, theta, indices)
        amplitude = np.exp(theta[log_a_index])
        decay_rate = np.exp(theta[log_c_index])
        lag = np.abs(time[:, None] - time[None, :])
        return amplitude * np.exp(-decay_rate * lag)

    kernel.__name__ = "k_red_noise"
    kernel.metadata = {
        "type": "kernel",
        "name": "red_noise",
        "parameter_indices": list(indices),
        "parameterization": ["log_amplitude", "log_decay_rate"],
    }
    return kernel


def k_qpo(parameter_indices):
    """Create an exponentially damped cosine QPO covariance kernel."""
    indices = _indices(parameter_indices, 3, "k_qpo")
    log_a_index, log_c_index, log_f_index = indices

    def kernel(time, theta):
        time, theta = _inputs(time, theta, indices)
        amplitude = np.exp(theta[log_a_index])
        decay_rate = np.exp(theta[log_c_index])
        frequency = np.exp(theta[log_f_index])
        lag = np.abs(time[:, None] - time[None, :])
        return (
            amplitude
            * np.exp(-decay_rate * lag)
            * np.cos(2.0 * np.pi * frequency * lag)
        )

    kernel.__name__ = "k_qpo"
    kernel.metadata = {
        "type": "kernel",
        "name": "qpo",
        "parameter_indices": list(indices),
        "parameterization": [
            "log_amplitude",
            "log_decay_rate",
            "log_frequency",
        ],
    }
    return kernel


def add_kernels(*kernels):
    """Return the covariance obtained by adding one or more kernels."""
    if not kernels:
        raise ValueError("at least one kernel is required")
    if not all(callable(kernel) for kernel in kernels):
        raise TypeError("every kernel must be callable")

    def combined_kernel(time, theta):
        terms = [np.asarray(kernel(time, theta), dtype=float) for kernel in kernels]
        expected_shape = terms[0].shape
        if any(term.shape != expected_shape for term in terms[1:]):
            raise ValueError("all kernels must return covariance matrices of one shape")
        return np.sum(terms, axis=0)

    combined_kernel.__name__ = "add_kernels"
    combined_kernel.metadata = {
        "type": "kernel",
        "name": "sum",
        "terms": [
            getattr(kernel, "metadata", {"name": getattr(kernel, "__name__", type(kernel).__name__)})
            for kernel in kernels
        ],
    }
    return combined_kernel


def qpo_conditional(
    theta,
    training_time,
    counts,
    errors,
    mean_function,
    kernel_function,
    qpo_kernel,
    prediction_time=None,
):
    """Condition the latent QPO GP component on observed data.

    The model is ``y = mean + red_noise + qpo + measurement_noise``. This
    returns the conditional mean and covariance of ``qpo`` at
    ``prediction_time``. The QPO kernel must be supplied separately from the
    total kernel because the latent component is not identifiable from the
    total covariance alone.
    """
    training_time = np.asarray(training_time, dtype=float)
    counts = np.asarray(counts, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if prediction_time is None:
        prediction_time = training_time
    prediction_time = np.asarray(prediction_time, dtype=float)
    if (training_time.ndim != 1 or counts.ndim != 1 or errors.ndim != 1
            or prediction_time.ndim != 1
            or not (training_time.size == counts.size == errors.size)):
        raise ValueError("training data and prediction_time must be 1D arrays")
    if not training_time.size or np.any(~np.isfinite(training_time)):
        raise ValueError("training_time must contain finite values")
    if (np.any(~np.isfinite(counts)) or np.any(~np.isfinite(errors))
            or np.any(errors <= 0) or np.any(~np.isfinite(prediction_time))):
        raise ValueError("data arrays must contain finite values and positive errors")

    mean = np.asarray(mean_function(training_time, theta), dtype=float)
    total_covariance = np.asarray(kernel_function(training_time, theta), dtype=float)
    joint_time = np.concatenate([prediction_time, training_time])
    qpo_joint = np.asarray(qpo_kernel(joint_time, theta), dtype=float)
    n_prediction = prediction_time.size
    qpo_cross = qpo_joint[:n_prediction, n_prediction:]
    qpo_prediction = qpo_joint[:n_prediction, :n_prediction]
    total_covariance = 0.5 * (total_covariance + total_covariance.T)
    total_covariance = total_covariance.copy()
    total_covariance[np.diag_indices_from(total_covariance)] += errors ** 2
    try:
        chol = np.linalg.cholesky(total_covariance)
        residual = counts - mean
        solved_residual = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
        solved_cross = np.linalg.solve(chol.T, np.linalg.solve(chol, qpo_cross.T))
    except np.linalg.LinAlgError as exc:
        raise ValueError("total GP covariance is not positive definite") from exc
    conditional_mean = qpo_cross @ solved_residual
    conditional_covariance = qpo_prediction - qpo_cross @ solved_cross
    conditional_covariance = 0.5 * (
        conditional_covariance + conditional_covariance.T
    )
    return conditional_mean, conditional_covariance
