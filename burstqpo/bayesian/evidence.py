"""Small helpers for comparing Bayesian model evidences."""

from __future__ import annotations

import numpy as np

__all__ = ["bayes_factor", "calculate_log_evidence", "run_nested_sampling"]


def bayes_factor(z_h1, z_h0, *, log_input=False):
    """Return the Bayes factor ``BF_10`` comparing H1 against H0.

    Parameters
    ----------
    z_h1, z_h0 : float
        Evidences ``Z(H1)`` and ``Z(H0)``. With ``log_input=True``, pass
        log-evidences such as ``dynesty.results.logz[-1]`` instead.
    log_input : bool, default=False
        Interpret the inputs as ``log Z`` values.

    Returns
    -------
    float
        ``BF_10 = Z(H1) / Z(H0)``. Values too large for floating point return
        ``numpy.inf``.

    Notes
    -----
    This combines already-computed evidences. It cannot turn MCMC posterior
    samples, posterior log-probabilities, or Z2n statistics into a Bayes
    factor.
    """
    try:
        first = float(z_h1)
        second = float(z_h0)
    except (TypeError, ValueError) as exc:
        raise TypeError("z_h1 and z_h0 must be scalar finite numbers") from exc
    if not np.isfinite(first) or not np.isfinite(second):
        raise ValueError("z_h1 and z_h0 must be finite")

    if log_input:
        difference = first - second
        if difference >= np.log(np.finfo(float).max):
            return float("inf")
        return float(np.exp(difference))

    if first < 0.0 or second <= 0.0:
        raise ValueError(
            "ordinary evidences must satisfy z_h1 >= 0 and z_h0 > 0"
        )
    return float(first / second)


def calculate_log_evidence(
    time,
    counts,
    errors,
    likelihood_function,
    model_group,
    prior_transform=None,
    ndim=None,
    *,
    nlive=500,
    dlogz=0.1,
    seed=None,
    bound="multi",
    sample="rwalk",
    print_progress=False,
):
    """Run nested sampling and return ``(logZ, logZerr, dynesty_result)``.

    ``prior_transform`` must map a point from the unit cube to the declared
    physical parameter prior. The likelihood and transform must describe the
    same prior; otherwise the evidence and Bayes factor are not meaningful.
    """
    try:
        import dynesty
    except ImportError as exc:
        raise ImportError(
            "nested evidence requires 'dynesty'; install it with "
            "`pip install dynesty`"
        ) from exc
    if not callable(likelihood_function):
        raise TypeError("likelihood_function must be callable")
    if prior_transform is None:
        raise TypeError(
            "prior_transform must be supplied; use prior.prior_transform "
            "for an unconstrained set_prior()"
        )
    if not callable(prior_transform):
        raise TypeError("prior_transform must be callable")
    if not hasattr(model_group, "items"):
        raise TypeError("model_group must be a dictionary-like object")
    if ndim is None:
        raise TypeError("ndim must be supplied")
    ndim = int(ndim)
    if ndim <= 0:
        raise ValueError("ndim must be positive")
    if int(nlive) <= 0 or float(dlogz) <= 0:
        raise ValueError("nlive and dlogz must be positive")

    time = np.asarray(time)
    counts = np.asarray(counts, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if (time.ndim != 1 or counts.ndim != 1 or errors.ndim != 1
            or not (time.size == counts.size == errors.size)
            or not time.size):
        raise ValueError("time, counts and errors must be non-empty 1D arrays of equal length")
    if (np.any(~np.isfinite(time)) or np.any(~np.isfinite(counts))
            or np.any(~np.isfinite(errors)) or np.any(errors <= 0)):
        raise ValueError("data arrays must be finite and errors must be positive")

    def loglike(theta):
        value = likelihood_function(
            np.asarray(theta, dtype=float),
            time,
            counts,
            errors,
            **dict(model_group),
        )
        value = float(value)
        return value if np.isfinite(value) else -np.inf

    sampler_kwargs = {
        "nlive": int(nlive),
        "bound": bound,
        "sample": sample,
    }
    if seed is not None:
        sampler_kwargs["rstate"] = np.random.default_rng(seed)
    sampler = dynesty.NestedSampler(
        loglike, prior_transform, ndim, **sampler_kwargs
    )
    sampler.run_nested(dlogz=float(dlogz), print_progress=bool(print_progress))
    nested_result = sampler.results
    logz = float(np.asarray(nested_result.logz)[-1])
    logzerr = float(np.asarray(nested_result.logzerr)[-1])
    return logz, logzerr, nested_result


run_nested_sampling = calculate_log_evidence
