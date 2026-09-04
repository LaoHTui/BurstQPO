import numpy as np

from .results import MCMCResult

__all__ = ["MCMCResult", "MCMCResults", "run_mcmc"]

MCMCResults = MCMCResult


def _callable_metadata(function):
    metadata = getattr(function, "metadata", None)
    if metadata is not None:
        return dict(metadata)
    return {
        "name": getattr(function, "__name__", type(function).__name__),
        "module": getattr(function, "__module__", ""),
    }


def run_mcmc(
        time,
        counts,
        errors,
        prior,
        mcmc_config,
        likelihood_function,
        model_group=None,
        output_path=None,
        seed=None,
        model_id=None,
        parameter_names=None,
        metadata=None,
):
    """Run emcee using an externally defined prior and likelihood.

    Parameters
    ----------
    time, counts, errors
        Observed data passed unchanged to ``likelihood_function``.

    prior
        External prior function. It may return either:

        - ``True`` or ``False`` for a hard uniform prior;
        - a finite scalar log-prior value, or ``-np.inf`` outside the prior.

    mcmc_config
        Dictionary containing at least ``initial`` and ``start_scale``.
        Optional keys are ``n_walkers``, ``n_steps``, ``burn_in`` and ``thin``.

    likelihood_function
        External likelihood function with the signature::

            likelihood_function(
                theta, time, counts, errors, **model_group
            )

        It must return the log-likelihood.

    model_group
        Dictionary of model objects or functions passed to the likelihood
        using keyword arguments. For example::

            {
                "mean_function": fred_model,
                "kernel_function": qpo_kernel,
            }

    output_path
        Path used to save the MCMC result. If ``None``, no file is written.

    seed
        Random seed. If ``None``, random initialization is used.

    model_id
        Optional model label saved in the output file.

    parameter_names
        Optional names corresponding to entries in ``theta``. They may also
        be supplied as ``mcmc_config["parameter_names"]``.

    metadata
        Optional JSON-serializable user metadata saved with the result.

    Returns
    -------
    MCMCResult
        Result object exposing the posterior samples and sampler diagnostics.
        It can be saved or loaded with :meth:`MCMCResult.save_npz` and
        :meth:`MCMCResult.load_npz`.
    """
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            "run_mcmc requires the 'emcee' dependency; "
            "install it with `pip install emcee`"
        ) from exc

    if not callable(prior):
        raise TypeError("prior must be callable")
    if not callable(likelihood_function):
        raise TypeError("likelihood_function must be callable")
    if model_group is None:
        model_group = {}
    elif not hasattr(model_group, "items"):
        raise TypeError("model_group must be a dictionary-like object")
    else:
        model_group = dict(model_group)

    required_config = ("initial", "start_scale")
    missing_config = [
        key for key in required_config if key not in mcmc_config
    ]
    if missing_config:
        raise KeyError(
            "mcmc_config is missing required keys: "
            + ", ".join(missing_config)
        )

    initial = np.asarray(mcmc_config["initial"], dtype=float)
    start_scale = np.asarray(mcmc_config["start_scale"], dtype=float)

    if initial.ndim != 1:
        raise ValueError("mcmc_config['initial'] must be one-dimensional")
    if start_scale.ndim != 1:
        raise ValueError("mcmc_config['start_scale'] must be one-dimensional")
    if initial.size == 0:
        raise ValueError("initial must contain at least one parameter")
    if initial.size != start_scale.size:
        raise ValueError(
            "initial and start_scale must have the same length"
        )
    if np.any(~np.isfinite(initial)):
        raise ValueError("initial must contain only finite values")
    if np.any(~np.isfinite(start_scale)) or np.any(start_scale <= 0):
        raise ValueError(
            "start_scale must contain only positive finite values"
        )

    n_walkers = int(mcmc_config.get("n_walkers", 48))
    n_steps = int(mcmc_config.get("n_steps", 6000))
    burn_in = int(mcmc_config.get("burn_in", 1500))
    thin = int(mcmc_config.get("thin", 1))
    ndim = initial.size
    if parameter_names is None:
        parameter_names = mcmc_config.get("parameter_names")
    if parameter_names is None:
        parameter_names = tuple(f"theta_{index}" for index in range(ndim))
    else:
        parameter_names = tuple(str(name) for name in parameter_names)
    if len(parameter_names) != ndim:
        raise ValueError("parameter_names must have the same length as initial")
    if len(set(parameter_names)) != ndim:
        raise ValueError("parameter_names must be unique")
    if metadata is None:
        metadata = {}
    elif not hasattr(metadata, "items"):
        raise TypeError("metadata must be a dictionary-like object")
    else:
        metadata = dict(metadata)

    if n_walkers < 2 * ndim:
        raise ValueError(
            "n_walkers must be at least twice the number of parameters"
        )
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not 0 <= burn_in < n_steps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps")
    if thin <= 0:
        raise ValueError("thin must be positive")

    time = np.asarray(time)
    counts = np.asarray(counts, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if time.ndim != 1 or counts.ndim != 1 or errors.ndim != 1:
        raise ValueError("time, counts and errors must be one-dimensional")
    if not (time.size == counts.size == errors.size):
        raise ValueError(
            "time, counts and errors must have the same length"
        )
    if counts.size == 0:
        raise ValueError("the data arrays must not be empty")
    if np.any(~np.isfinite(time)):
        raise ValueError("time must contain only finite values")
    if np.any(~np.isfinite(counts)):
        raise ValueError("counts must contain only finite values")
    if np.any(~np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError("errors must contain only positive finite values")

    def log_prior_value(theta):
        value = prior(theta)

        if isinstance(value, (bool, np.bool_)):
            return 0.0 if bool(value) else -np.inf

        try:
            value = float(value)
        except (TypeError, ValueError):
            return -np.inf

        return value if np.isfinite(value) else -np.inf

    def log_probability(theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != initial.shape or np.any(~np.isfinite(theta)):
            return -np.inf, -np.inf, -np.inf

        log_prior = log_prior_value(theta)
        if not np.isfinite(log_prior):
            return -np.inf, -np.inf, -np.inf

        try:
            log_likelihood = likelihood_function(
                theta,
                time,
                counts,
                errors,
                **model_group,
            )
            log_likelihood = float(log_likelihood)
        except (ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
            return -np.inf, log_prior, -np.inf

        if not np.isfinite(log_likelihood):
            return -np.inf, log_prior, -np.inf

        return log_prior + log_likelihood, log_prior, log_likelihood

    rng = np.random.default_rng(seed)
    walkers = np.empty((n_walkers, ndim), dtype=float)

    for index in range(n_walkers):
        initialized = False
        for _ in range(10000):
            candidate = initial + rng.normal(
                size=ndim
            ) * start_scale
            if np.isfinite(log_probability(candidate)[0]):
                walkers[index] = candidate
                initialized = True
                break

        if not initialized:
            raise ValueError(
                "Could not initialize a valid walker after 10000 attempts; "
                "check initial, start_scale, prior, and likelihood"
            )

    # emcee versions commonly use NumPy's legacy global RNG internally.
    # Seed it as well so the complete chain is reproducible when seed is set.
    old_random_state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(seed)

        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            log_probability,
            blobs_dtype=[
                ("log_prior", float),
                ("log_likelihood", float),
            ],
        )
        sampler.run_mcmc(walkers, n_steps, progress=False)
    finally:
        np.random.set_state(old_random_state)

    samples = sampler.get_chain(
        discard=burn_in,
        thin=thin,
        flat=True,
    )
    log_prob = sampler.get_log_prob(
        discard=burn_in,
        thin=thin,
        flat=True,
    )
    probability_parts = sampler.get_blobs(
        discard=burn_in,
        thin=thin,
        flat=True,
    )

    try:
        autocorrelation_time = sampler.get_autocorr_time(tol=0)
    except emcee.autocorr.AutocorrError:
        autocorrelation_time = np.full(ndim, np.nan)

    result = MCMCResult(
        samples=np.asarray(samples, dtype=float),
        log_probability=np.asarray(log_prob, dtype=float),
        acceptance_fraction=np.asarray(sampler.acceptance_fraction, dtype=float),
        autocorrelation_time=np.asarray(autocorrelation_time, dtype=float),
        model_id="" if model_id is None else str(model_id),
        seed=None if seed is None else int(seed),
        parameter_names=parameter_names,
        initial=initial.copy(),
        start_scale=start_scale.copy(),
        sampler_config={
            "n_walkers": n_walkers,
            "n_steps": n_steps,
            "burn_in": burn_in,
            "thin": thin,
        },
        prior_metadata=_callable_metadata(prior),
        model_metadata={
            name: _callable_metadata(component)
            for name, component in model_group.items()
            if callable(component)
        },
        likelihood_name=getattr(
            likelihood_function, "__name__", type(likelihood_function).__name__
        ),
        extra_metadata=metadata,
        log_prior_values=np.asarray(
            probability_parts["log_prior"], dtype=float
        ),
        log_likelihood_values=np.asarray(
            probability_parts["log_likelihood"], dtype=float
        ),
        n_observations=int(time.size),
        time=np.asarray(time, dtype=float).copy(),
        counts=counts.copy(),
        errors=errors.copy(),
    )

    if output_path is not None:
        result.save_npz(output_path)

    return result
