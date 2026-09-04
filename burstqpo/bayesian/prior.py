"""Prior builders for Bayesian fitting."""

import numpy as np

__all__ = ["set_prior"]


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def set_prior(prior_specs, constraints=()):
    """Build a log-prior callable and attach inspectable metadata.

    Specifications are ``("uniform", lower, upper)``,
    ``("log_uniform", lower, upper)``, ``("normal", mean, sigma)``,
    or a scalar callable. Uniform bounds are open, matching the previous
    implementation.
    """
    prior_specs = tuple(prior_specs)
    if callable(constraints):
        constraints = (constraints,)
    else:
        constraints = tuple(constraints)
    normalized_specs = []
    for spec in prior_specs:
        if callable(spec):
            normalized_specs.append({"type": "callable", "name": getattr(spec, "__name__", type(spec).__name__)})
        else:
            if len(spec) != 3:
                raise ValueError("prior specifications must contain three items")
            normalized_specs.append({"type": str(spec[0]), "lower_or_mean": _json_value(spec[1]), "upper_or_sigma": _json_value(spec[2])})
    metadata = {
        "type": "prior",
        "specs": normalized_specs,
        "constraints": [getattr(c, "__name__", type(c).__name__) for c in constraints],
        "normalized": not constraints,
    }

    def log_prior(theta):
        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 1 or theta.size != len(prior_specs) or np.any(~np.isfinite(theta)):
            return -np.inf
        total = 0.0
        for value, spec in zip(theta, prior_specs):
            if callable(spec):
                contribution = spec(value)
            else:
                kind, first, second = spec
                if kind == "uniform":
                    if not first < value < second:
                        return -np.inf
                    contribution = -np.log(second - first)
                elif kind == "log_uniform":
                    if not first < value < second or value <= 0:
                        return -np.inf
                    contribution = -np.log(value) - np.log(np.log(second / first))
                elif kind == "normal":
                    if second <= 0:
                        raise ValueError("normal-prior sigma must be positive")
                    contribution = -0.5 * (((value - first) / second) ** 2 + np.log(2.0 * np.pi * second ** 2))
                else:
                    raise ValueError(f"unsupported prior type: {kind}")
            if isinstance(contribution, (bool, np.bool_)):
                if not contribution:
                    return -np.inf
                contribution = 0.0
            try:
                contribution = float(contribution)
            except (TypeError, ValueError):
                return -np.inf
            if not np.isfinite(contribution):
                return -np.inf
            total += contribution
        if not all(bool(constraint(theta)) for constraint in constraints):
            return -np.inf
        return total

    log_prior.metadata = metadata
    if not constraints and all(not callable(spec) for spec in prior_specs):
        def prior_transform(unit_cube):
            unit_cube = np.asarray(unit_cube, dtype=float)
            if (unit_cube.ndim != 1 or unit_cube.size != len(prior_specs)
                    or np.any(~np.isfinite(unit_cube))
                    or np.any((unit_cube < 0.0) | (unit_cube > 1.0))):
                raise ValueError("unit_cube must contain values in [0, 1]")
            transformed = []
            for unit_value, spec in zip(unit_cube, prior_specs):
                kind, first, second = spec
                if kind == "uniform":
                    transformed.append(first + unit_value * (second - first))
                elif kind == "log_uniform":
                    if first <= 0 or second <= first:
                        raise ValueError("log_uniform bounds must be positive and ordered")
                    transformed.append(np.exp(
                        np.log(first) + unit_value * np.log(second / first)
                    ))
                elif kind == "normal":
                    if second <= 0:
                        raise ValueError("normal-prior sigma must be positive")
                    from scipy.special import ndtri
                    transformed.append(first + second * ndtri(unit_value))
                else:
                    raise ValueError(f"unsupported prior type: {kind}")
            return np.asarray(transformed, dtype=float)

        log_prior.prior_transform = prior_transform
    return log_prior
