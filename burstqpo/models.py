"""Light-curve models used by burst QPO fitting workflows."""

import numpy as np

__all__ = [
    "add_mean_models",
    "constant_background",
    "constant_background_model",
    "ercod",
    "ercod_model",
    "fred",
    "fred_model",
    "linear_background",
    "linear_background_model",
    "polynomial_background",
    "polynomial_background_model",
]


def _parameterized_model(model, parameter_indices, model_name):
    indices = tuple(int(index) for index in parameter_indices)
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError("parameter_indices must contain unique non-negative indices")

    def bound_model(time, theta):
        parameters = np.asarray(theta, dtype=float)
        if parameters.ndim != 1:
            raise ValueError("theta must be one-dimensional")
        if indices and max(indices) >= parameters.size:
            raise ValueError("parameter_indices refer outside theta")
        return model(time, parameters[list(indices)])

    bound_model.__name__ = model_name
    bound_model.__qualname__ = model_name
    bound_model.metadata = {
        "type": "mean_model",
        "name": model_name,
        "parameter_indices": list(indices),
    }
    return bound_model


def ercod_model(time, theta):
    """Evaluate the background-free paper ERCOD pulse.

    Parameters
    ----------
    time : array-like
        Time coordinates at which to evaluate the model.
    theta : sequence of float
        ``(amplitude, t_max, sigma_1, sigma_2, nu, t_cut)``.

    Returns
    -------
    numpy.ndarray
        Model count rate at each time coordinate.
    """
    amplitude, t_max, sigma_1, sigma_2, nu, t_cut = theta
    time = np.asarray(time, dtype=float)
    separation = t_cut - t_max

    # This normalization makes the decay branch meet the rise at t_max.
    normalization = 1.0 / (-np.expm1(-separation / sigma_2))
    rise = amplitude * np.exp(
        -(np.maximum(t_max - time, 0.0) / sigma_1) ** nu
    )
    decay = amplitude * normalization * (
        -np.expm1(-np.maximum(t_cut - time, 0.0) / sigma_2)
    )
    pulse = np.where(
        time <= t_max, rise, np.where(time <= t_cut, decay, 0.0)
    )
    return pulse


def fred_model(time, theta):
    """Evaluate the background-free paper FRED pulse.

    Parameters
    ----------
    time : array-like
        Time coordinates at which to evaluate the model.
    theta : sequence of float
        ``(amplitude, t_max, sigma_1, sigma_2, nu)``.

    Returns
    -------
    numpy.ndarray
        Model count rate at each time coordinate.
    """
    amplitude, t_max, sigma_1, sigma_2, nu = theta
    time = np.asarray(time, dtype=float)
    delta = time - t_max
    rise = amplitude * np.exp(-(np.abs(delta) / sigma_1) ** nu)
    decay = amplitude * np.exp(-(np.abs(delta) / sigma_2) ** nu)
    pulse = np.where(delta <= 0, rise, decay)
    return pulse


def fred(parameter_indices=(0, 1, 2, 3, 4)):
    """Bind the FRED model to selected positions in a global parameter vector."""
    indices = tuple(parameter_indices)
    if len(indices) != 5:
        raise ValueError("FRED requires exactly 5 parameter indices")
    return _parameterized_model(fred_model, indices, "fred")


def ercod(parameter_indices=(0, 1, 2, 3, 4, 5)):
    """Bind the ERCOD model to selected positions in a global parameter vector."""
    indices = tuple(parameter_indices)
    if len(indices) != 6:
        raise ValueError("ERCOD requires exactly 6 parameter indices")
    return _parameterized_model(ercod_model, indices, "ercod")


def constant_background_model(time, theta):
    """Evaluate a constant background with ``theta=(level,)``."""
    (level,) = theta
    return np.full(np.asarray(time, dtype=float).shape, level, dtype=float)


def linear_background_model(time, theta, *, time_origin=0.0):
    """Evaluate ``intercept + slope * (time - time_origin)``."""
    intercept, slope = theta
    centered_time = np.asarray(time, dtype=float) - float(time_origin)
    return intercept + slope * centered_time


def polynomial_background_model(time, theta, *, time_origin=0.0):
    """Evaluate an ascending-order polynomial about ``time_origin``.

    ``theta=(c0, c1, ..., cn)`` represents
    ``c0 + c1*dt + ... + cn*dt**n``, where ``dt=time-time_origin``.
    """
    coefficients = np.asarray(theta, dtype=float)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("theta must contain at least one coefficient")
    centered_time = np.asarray(time, dtype=float) - float(time_origin)
    return np.polynomial.polynomial.polyval(centered_time, coefficients)


def constant_background(parameter_index):
    """Bind a constant background level to one global theta position."""
    return _parameterized_model(
        constant_background_model, (parameter_index,), "constant_background"
    )


def linear_background(parameter_indices, *, time_origin=0.0):
    """Bind intercept and slope parameters to a linear background."""
    indices = tuple(parameter_indices)
    if len(indices) != 2:
        raise ValueError("linear background requires exactly 2 parameter indices")

    def model(time, parameters):
        return linear_background_model(
            time, parameters, time_origin=time_origin
        )

    bound = _parameterized_model(model, indices, "linear_background")
    bound.metadata["time_origin"] = float(time_origin)
    return bound


def polynomial_background(parameter_indices, *, time_origin=0.0):
    """Bind ascending polynomial coefficients to global theta positions."""
    indices = tuple(parameter_indices)
    if not indices:
        raise ValueError("polynomial background requires at least one parameter")

    def model(time, parameters):
        return polynomial_background_model(
            time, parameters, time_origin=time_origin
        )

    bound = _parameterized_model(model, indices, "polynomial_background")
    bound.metadata.update({
        "degree": len(indices) - 1,
        "time_origin": float(time_origin),
    })
    return bound


def add_mean_models(*models):
    """Return a mean function equal to the sum of all supplied models."""
    if not models:
        raise ValueError("at least one mean model is required")
    if not all(callable(model) for model in models):
        raise TypeError("every mean model must be callable")

    def combined_model(time, theta):
        expected_shape = np.asarray(time, dtype=float).shape
        total = np.zeros(expected_shape, dtype=float)
        for model in models:
            values = np.asarray(model(time, theta), dtype=float)
            try:
                values = np.broadcast_to(values, expected_shape)
            except ValueError as exc:
                raise ValueError(
                    "every mean model must return values matching time"
                ) from exc
            total = total + values
        return total

    combined_model.__name__ = "add_mean_models"
    combined_model.metadata = {
        "type": "mean_model",
        "name": "sum",
        "terms": [
            getattr(
                model,
                "metadata",
                {"name": getattr(model, "__name__", type(model).__name__)},
            )
            for model in models
        ],
    }
    return combined_model
