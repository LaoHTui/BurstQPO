"""Posterior-predictive light curves and simulation-calibrated significance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import norm

from .bayesian.gp import add_kernels, k_qpo, k_red_noise
from .mcmc import MCMCResult
from .models import (
    add_mean_models,
    constant_background,
    ercod,
    fred,
    linear_background,
    polynomial_background,
)
from .results import DCFResult, JurkevichResult, LombScargleResult, WWZResult

__all__ = [
    "LightCurveSimulationResult",
    "PeakSignificance",
    "SignificanceResult",
    "SignificanceResults",
    "calculate_significance",
    "simulate_light_curves",
]

_SIGMA_LEVELS = np.asarray([1.0, 2.0, 3.0])
_SIGMA_PERCENTILES = 100.0 * norm.cdf(_SIGMA_LEVELS)


def _npz_path(path):
    output = Path(path)
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _load_posterior(posterior):
    if isinstance(posterior, MCMCResult):
        return posterior
    return MCMCResult.load_npz(posterior)


def _random_generator(random_state):
    rng = np.random.default_rng(random_state)
    seed = (
        int(random_state)
        if isinstance(random_state, (int, np.integer)) else None
    )
    return rng, seed


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _observations(result, time, counts, errors):
    arrays = (
        result.time if time is None else time,
        result.counts if counts is None else counts,
        result.errors if errors is None else errors,
    )
    time, counts, errors = (np.asarray(value, dtype=float) for value in arrays)
    if time.ndim != 1 or counts.ndim != 1 or errors.ndim != 1:
        raise ValueError("time, counts, and errors must be one-dimensional")
    if not time.size:
        raise ValueError(
            "the posterior does not contain observations; pass time, counts, "
            "and errors explicitly"
        )
    if not (time.size == counts.size == errors.size):
        raise ValueError("time, counts, and errors must have the same length")
    if (np.any(~np.isfinite(time)) or np.any(~np.isfinite(counts))
            or np.any(~np.isfinite(errors)) or np.any(errors <= 0)):
        raise ValueError(
            "observations must be finite and errors must be strictly positive"
        )
    return time, counts, errors


def _mean_from_metadata(metadata):
    name = metadata.get("name")
    indices = metadata.get("parameter_indices", ())
    if name == "sum":
        terms = metadata.get("terms", ())
        if not terms:
            raise ValueError("mean-model metadata contains an empty sum")
        return add_mean_models(*(_mean_from_metadata(term) for term in terms))
    if name == "fred":
        return fred(indices)
    if name == "ercod":
        return ercod(indices)
    if name == "constant_background":
        if len(indices) != 1:
            raise ValueError("invalid constant-background metadata")
        return constant_background(indices[0])
    if name == "linear_background":
        return linear_background(indices, time_origin=metadata.get("time_origin", 0.0))
    if name == "polynomial_background":
        return polynomial_background(indices, time_origin=metadata.get("time_origin", 0.0))
    raise ValueError(
        f"cannot restore mean model {name!r} from posterior metadata; "
        "pass mean_function explicitly"
    )


def _kernel_from_metadata(metadata):
    name = metadata.get("name")
    indices = metadata.get("parameter_indices", ())
    if name == "sum":
        terms = metadata.get("terms", ())
        if not terms:
            raise ValueError("kernel metadata contains an empty sum")
        return add_kernels(*(_kernel_from_metadata(term) for term in terms))
    if name == "red_noise":
        return k_red_noise(indices)
    if name == "qpo":
        return k_qpo(indices)
    raise ValueError(
        f"cannot restore kernel {name!r} from posterior metadata; "
        "pass kernel_function explicitly"
    )


def _contains_qpo(metadata):
    if isinstance(metadata, dict):
        return metadata.get("name") == "qpo" or any(
            _contains_qpo(value) for value in metadata.values()
        )
    if isinstance(metadata, (list, tuple)):
        return any(_contains_qpo(value) for value in metadata)
    return False


def _resolve_models(result, mean_function, kernel_function):
    models = result.model_metadata
    if mean_function is None:
        metadata = models.get("mean_function")
        if not metadata:
            raise ValueError(
                "posterior metadata does not define a mean model; "
                "pass mean_function explicitly"
            )
        mean_function = _mean_from_metadata(metadata)
    if kernel_function is None:
        metadata = models.get("kernel_function")
        kernel_function = (
            None if not metadata else _kernel_from_metadata(metadata)
        )
    if not callable(mean_function):
        raise TypeError("mean_function must be callable")
    if kernel_function is not None and not callable(kernel_function):
        raise TypeError("kernel_function must be callable or None")
    return mean_function, kernel_function


def _reference_curve(result, time, mean_function, reference_mean, residual):
    if reference_mean is not None:
        reference = np.asarray(reference_mean, dtype=float)
        if reference.shape != time.shape or np.any(~np.isfinite(reference)):
            raise ValueError("reference_mean must be finite and match time")
        return reference.copy()
    if not residual:
        return np.zeros_like(time)
    if result.log_likelihood_values.size != result.n_samples:
        raise ValueError(
            "residual=True requires saved log-likelihood values or an explicit "
            "reference_mean"
        )
    reference = np.asarray(
        mean_function(time, result.maximum_likelihood_sample), dtype=float
    )
    if reference.shape != time.shape or np.any(~np.isfinite(reference)):
        raise ValueError("mean_function returned an invalid reference curve")
    return reference


def _draw_curve(rng, time, errors, theta, mean_function, kernel_function,
                observation_sampler):
    mean = np.asarray(mean_function(time, theta), dtype=float)
    if mean.shape != time.shape or np.any(~np.isfinite(mean)):
        raise ValueError("mean_function must return one finite value per time")
    latent = mean.copy()
    if kernel_function is not None:
        covariance = np.asarray(kernel_function(time, theta), dtype=float)
        if covariance.shape != (time.size, time.size):
            raise ValueError("kernel_function returned a covariance with invalid shape")
        covariance = 0.5 * (covariance + covariance.T)
        if np.any(~np.isfinite(covariance)):
            raise ValueError("kernel_function returned a non-finite covariance")
        try:
            latent += rng.multivariate_normal(
                np.zeros(time.size), covariance, check_valid="raise"
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError("kernel covariance is not positive semidefinite") from exc
    if observation_sampler is None:
        values = latent + rng.normal(0.0, errors, size=time.size)
    else:
        values = observation_sampler(rng, latent.copy(), errors.copy(), theta.copy())
    values = np.asarray(values, dtype=float)
    if values.shape != time.shape or np.any(~np.isfinite(values)):
        raise ValueError(
            "observation_sampler must return one finite value per time"
        )
    return values


@dataclass(frozen=True)
class LightCurveSimulationResult:
    time: np.ndarray
    values: np.ndarray
    sample_indices: np.ndarray
    posterior_parameters: np.ndarray
    errors: np.ndarray
    reference_mean: np.ndarray
    residual: bool = False
    seed: Optional[int] = None

    @property
    def light_curves(self):
        if self.residual:
            return self.values + self.reference_mean[None, :]
        return self.values

    @property
    def residuals(self):
        if self.residual:
            return self.values
        return self.values - self.reference_mean[None, :]

    @property
    def n_simulations(self):
        return int(self.values.shape[0])

    def save_npz(self, path):
        output = _npz_path(path)
        np.savez_compressed(
            output, time=self.time, values=self.values,
            sample_indices=self.sample_indices,
            posterior_parameters=self.posterior_parameters,
            errors=self.errors, reference_mean=self.reference_mean,
            residual=self.residual, seed=-1 if self.seed is None else self.seed,
        )
        return output

    @classmethod
    def load_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            required = {
                "time", "values", "sample_indices", "posterior_parameters",
                "errors", "reference_mean",
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(
                    f"NPZ file is missing simulation fields: {sorted(missing)}"
                )
            seed = int(data["seed"]) if "seed" in data.files else -1
            return cls(
                np.asarray(data["time"], float),
                np.asarray(data["values"], float),
                np.asarray(data["sample_indices"], int),
                np.asarray(data["posterior_parameters"], float),
                np.asarray(data["errors"], float),
                np.asarray(data["reference_mean"], float),
                bool(data["residual"]) if "residual" in data.files else False,
                None if seed < 0 else seed,
            )


def simulate_light_curves(
    posterior,
    n_simulations,
    *,
    time=None,
    counts=None,
    errors=None,
    mean_function=None,
    kernel_function=None,
    observation_sampler=None,
    residual=False,
    reference_mean=None,
    random_state=None,
    save_path=None,
):
    """Draw posterior-predictive binned light curves from an MCMC result."""
    result = _load_posterior(posterior)
    if int(n_simulations) != n_simulations or int(n_simulations) <= 0:
        raise ValueError("n_simulations must be a positive integer")
    if result.n_samples <= 0:
        raise ValueError("posterior contains no samples")
    time, counts, errors = _observations(result, time, counts, errors)
    mean_function, kernel_function = _resolve_models(
        result, mean_function, kernel_function
    )
    reference = _reference_curve(
        result, time, mean_function, reference_mean, bool(residual)
    )
    rng, seed = _random_generator(random_state)
    indices = rng.integers(0, result.n_samples, size=int(n_simulations))
    curves = np.empty((int(n_simulations), time.size), dtype=float)
    for row, index in enumerate(indices):
        curves[row] = _draw_curve(
            rng, time, errors, result.samples[index], mean_function,
            kernel_function, observation_sampler,
        ) - reference
    output = LightCurveSimulationResult(
        time.copy(), curves, indices, result.samples[indices].copy(),
        errors.copy(), reference, bool(residual),
        seed,
    )
    if save_path is not None:
        output.save_npz(save_path)
    return output


def _method_name(method):
    return getattr(method, "__name__", type(method).__name__)


def _run_method(method, time, values, errors, method_kwargs):
    name = _method_name(method).lower()
    if name in {"z2n", "compute_z2n"}:
        raise ValueError(
            "Z2n requires event arrival times and is not supported by the "
            "binned-light-curve significance interface"
        )
    kwargs = dict(method_kwargs)
    if name in {
        "lsp", "lomb_scargle", "compute_lsp", "compute_lomb_scargle",
        "wwz", "compute_wwz",
    } and "uncertainty" not in kwargs:
        kwargs["uncertainty"] = errors
    return method(time, values, **kwargs)


def _extract_statistic(result, statistic_extractor):
    if statistic_extractor is not None:
        extracted = statistic_extractor(result)
        if not isinstance(extracted, (tuple, list)) or len(extracted) != 2:
            raise ValueError(
                "statistic_extractor must return (coordinate, score)"
            )
        coordinate, score = extracted
    elif isinstance(result, LombScargleResult):
        coordinate, score = result.frequency, result.power
    elif isinstance(result, WWZResult):
        coordinate, score = result.frequency, result.project(use_coi=True)
    elif isinstance(result, JurkevichResult):
        coordinate, score = result.period, 1.0 - result.v_norm
    elif isinstance(result, DCFResult):
        coordinate, score = result.lag, result.correlation
    else:
        raise TypeError(
            "unsupported method result; pass statistic_extractor returning "
            "(coordinate, score), with larger scores more significant"
        )
    coordinate = np.asarray(coordinate, dtype=float)
    score = np.asarray(score, dtype=float)
    if (coordinate.ndim != 1 or score.ndim != 1
            or coordinate.size == 0 or coordinate.shape != score.shape
            or np.any(~np.isfinite(coordinate))):
        raise ValueError("coordinate and score must be aligned 1D arrays")
    return coordinate, score


def _search_mask(coordinate, search_range):
    mask = np.isfinite(coordinate)
    if search_range is not None:
        if len(search_range) != 2:
            raise ValueError("search_range must be (minimum, maximum)")
        lower, upper = map(float, search_range)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("search_range must contain finite ordered bounds")
        mask &= (coordinate >= lower) & (coordinate <= upper)
    if not np.any(mask):
        raise ValueError("search_range contains no statistic coordinates")
    return mask


def _empirical_pvalues(observed, simulated):
    output = np.full(observed.shape, np.nan)
    for index, value in enumerate(observed):
        column = simulated[:, index]
        valid = np.isfinite(column)
        if np.isfinite(value) and np.any(valid):
            output[index] = (
                np.count_nonzero(column[valid] >= value) + 1.0
            ) / (np.count_nonzero(valid) + 1.0)
    return output


def _global_empirical_pvalues(observed, simulated_max):
    valid = np.isfinite(simulated_max)
    output = np.full(observed.shape, np.nan)
    if np.any(valid):
        maxima = simulated_max[valid]
        finite_observed = np.isfinite(observed)
        output[finite_observed] = (
            np.count_nonzero(
                maxima[:, None] >= observed[finite_observed][None, :], axis=0
            ) + 1.0
        ) / (maxima.size + 1.0)
    return output


@dataclass(frozen=True)
class PeakSignificance:
    index: int
    coordinate: float
    observed_statistic: float
    local_pvalue: float
    global_pvalue: float
    local_significance_sigma: float
    global_significance_sigma: float
    local_thresholds: dict
    global_thresholds: dict


@dataclass(frozen=True)
class SignificanceResult:
    coordinate: np.ndarray
    observed_statistic: np.ndarray
    simulated_statistic: np.ndarray
    simulated_max: np.ndarray
    local_pvalue: np.ndarray
    global_pvalue: np.ndarray
    local_sigma_thresholds: np.ndarray
    global_sigma_thresholds: np.ndarray
    sigma_levels: np.ndarray = field(default_factory=lambda: _SIGMA_LEVELS.copy())
    method_name: str = ""
    method_kwargs: dict = field(default_factory=dict)
    search_range: Optional[tuple] = None
    n_simulations_requested: int = 0
    seed: Optional[int] = None
    residual: bool = True
    reference_mean: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )

    @property
    def n_simulations_completed(self):
        return int(self.simulated_statistic.shape[0])

    @property
    def frequency(self):
        return self.coordinate

    @property
    def simulated_best(self):
        return self.simulated_max

    def local_threshold(self, sigma):
        percentile = 100.0 * norm.cdf(float(sigma))
        return np.nanpercentile(self.simulated_statistic, percentile, axis=0)

    def global_threshold(self, sigma):
        return float(np.nanpercentile(
            self.simulated_max, 100.0 * norm.cdf(float(sigma))
        ))

    def _local_sigma(self, sigma):
        return self.local_sigma_thresholds[
            int(np.argmin(np.abs(self.sigma_levels - sigma)))
        ]

    def _global_sigma(self, sigma):
        value = self.global_sigma_thresholds[
            int(np.argmin(np.abs(self.sigma_levels - sigma)))
        ]
        return np.full(self.coordinate.shape, value, dtype=float)

    @property
    def local_sigma1(self): return self._local_sigma(1.0)
    @property
    def local_sigma2(self): return self._local_sigma(2.0)
    @property
    def local_sigma3(self): return self._local_sigma(3.0)
    @property
    def global_sigma1(self): return self._global_sigma(1.0)
    @property
    def global_sigma2(self): return self._global_sigma(2.0)
    @property
    def global_sigma3(self): return self._global_sigma(3.0)

    def at(self, *, coordinate=None, index=None):
        if (coordinate is None) == (index is None):
            raise ValueError("pass exactly one of coordinate or index")
        if coordinate is not None:
            value = float(coordinate)
            if not np.isfinite(value):
                raise ValueError("coordinate must be finite")
            lower, upper = np.min(self.coordinate), np.max(self.coordinate)
            if value < lower or value > upper:
                raise ValueError("coordinate lies outside the statistic grid")
            index = int(np.argmin(np.abs(self.coordinate - value)))
        else:
            index = int(index)
            if index < 0 or index >= self.coordinate.size:
                raise IndexError("index lies outside the statistic grid")
        local_p = float(self.local_pvalue[index])
        global_p = float(self.global_pvalue[index])
        return PeakSignificance(
            index=index,
            coordinate=float(self.coordinate[index]),
            observed_statistic=float(self.observed_statistic[index]),
            local_pvalue=local_p,
            global_pvalue=global_p,
            local_significance_sigma=float(norm.isf(local_p)),
            global_significance_sigma=float(norm.isf(global_p)),
            local_thresholds={
                float(level): float(values[index])
                for level, values in zip(
                    self.sigma_levels, self.local_sigma_thresholds
                )
            },
            global_thresholds={
                float(level): float(value)
                for level, value in zip(
                    self.sigma_levels, self.global_sigma_thresholds
                )
            },
        )

    def save_npz(self, path):
        output = _npz_path(path)
        np.savez_compressed(
            output,
            coordinate=self.coordinate,
            frequency=self.coordinate,
            observed_statistic=self.observed_statistic,
            simulated_statistic=self.simulated_statistic,
            simulated_max=self.simulated_max,
            simulated_best=self.simulated_max,
            local_pvalue=self.local_pvalue,
            global_pvalue=self.global_pvalue,
            local_sigma_thresholds=self.local_sigma_thresholds,
            global_sigma_thresholds=self.global_sigma_thresholds,
            sigma_levels=self.sigma_levels,
            sigma_percentiles=100.0 * norm.cdf(self.sigma_levels),
            method_name=self.method_name,
            method_kwargs_json=json.dumps(_json_safe(self.method_kwargs)),
            search_range=(np.asarray([]) if self.search_range is None
                          else np.asarray(self.search_range, dtype=float)),
            n_simulations_requested=self.n_simulations_requested,
            n_simulations_completed=self.n_simulations_completed,
            seed=-1 if self.seed is None else self.seed,
            residual=self.residual,
            reference_mean=self.reference_mean,
        )
        return output

    @classmethod
    def load_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            coordinate_key = "coordinate" if "coordinate" in data.files else "frequency"
            max_key = "simulated_max" if "simulated_max" in data.files else "simulated_best"
            required = {
                coordinate_key, "observed_statistic", "simulated_statistic", max_key,
                "local_sigma_thresholds", "global_sigma_thresholds",
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(
                    f"NPZ file is missing significance fields: {sorted(missing)}"
                )
            coordinate = np.asarray(data[coordinate_key], float)
            observed = np.asarray(data["observed_statistic"], float)
            simulated = np.asarray(data["simulated_statistic"], float)
            simulated_max = np.asarray(data[max_key], float)
            local_p = (
                np.asarray(
                    data["local_pvalue"]
                    if "local_pvalue" in data.files else data["local_p"],
                    float,
                )
                if "local_pvalue" in data.files or "local_p" in data.files
                else _empirical_pvalues(observed, simulated)
            )
            if local_p.shape != observed.shape:
                local_p = _empirical_pvalues(observed, simulated)
            global_p = (
                np.asarray(
                    data["global_pvalue"]
                    if "global_pvalue" in data.files else data["global_p"],
                    float,
                )
                if "global_pvalue" in data.files or "global_p" in data.files
                else _global_empirical_pvalues(observed, simulated_max)
            )
            if global_p.shape != observed.shape:
                global_p = _global_empirical_pvalues(observed, simulated_max)
            levels = (
                np.asarray(data["sigma_levels"], float)
                if "sigma_levels" in data.files else _SIGMA_LEVELS.copy()
            )
            search = np.asarray(data["search_range"], float) if "search_range" in data.files else np.asarray([])
            seed = int(data["seed"]) if "seed" in data.files else -1
            method_kwargs = {}
            if "method_kwargs_json" in data.files:
                method_kwargs = json.loads(str(data["method_kwargs_json"].item()))
            return cls(
                coordinate, observed, simulated, simulated_max, local_p,
                global_p,
                np.asarray(data["local_sigma_thresholds"], float),
                np.asarray(data["global_sigma_thresholds"], float), levels,
                str(data["method_name"].item()) if "method_name" in data.files else "",
                method_kwargs, None if search.size == 0 else tuple(search.tolist()),
                int(data["n_simulations_requested"])
                if "n_simulations_requested" in data.files else simulated.shape[0],
                None if seed < 0 else seed,
                bool(data["residual"]) if "residual" in data.files else True,
                np.asarray(data["reference_mean"], float)
                if "reference_mean" in data.files else np.asarray([], dtype=float),
            )


SignificanceResults = SignificanceResult


def calculate_significance(
    posterior,
    method,
    n_simulations,
    *,
    method_kwargs=None,
    statistic_extractor=None,
    time=None,
    counts=None,
    errors=None,
    mean_function=None,
    kernel_function=None,
    observation_sampler=None,
    residual=True,
    reference_mean=None,
    search_range=None,
    random_state=None,
    progress=True,
    save_path=None,
):
    """Calibrate local and global peak significance from an H0 posterior."""
    if not callable(method):
        raise TypeError("method must be callable")
    if int(n_simulations) != n_simulations or int(n_simulations) <= 0:
        raise ValueError("n_simulations must be a positive integer")
    result = _load_posterior(posterior)
    kernel_metadata = result.model_metadata.get("kernel_function", {})
    if _contains_qpo(kernel_metadata):
        raise ValueError(
            "significance calibration requires a null-model posterior without "
            "a QPO kernel"
        )
    time, counts, errors = _observations(result, time, counts, errors)
    mean_function, kernel_function = _resolve_models(
        result, mean_function, kernel_function
    )
    if _contains_qpo(getattr(kernel_function, "metadata", {})):
        raise ValueError(
            "significance calibration requires a null-model kernel without QPO"
        )
    reference = _reference_curve(
        result, time, mean_function, reference_mean, bool(residual)
    )
    analysis_values = counts - reference
    kwargs = {} if method_kwargs is None else dict(method_kwargs)
    observed_result = _run_method(method, time, analysis_values, errors, kwargs)
    coordinate, observed = _extract_statistic(observed_result, statistic_extractor)
    mask = _search_mask(coordinate, search_range)
    if not np.any(np.isfinite(observed[mask])):
        raise ValueError("observed statistic is non-finite throughout search_range")

    rng, seed = _random_generator(random_state)
    simulated = np.full((int(n_simulations), coordinate.size), np.nan)
    interval = max(1, int(n_simulations) // 100)
    for row in range(int(n_simulations)):
        index = int(rng.integers(0, result.n_samples))
        curve = _draw_curve(
            rng, time, errors, result.samples[index], mean_function,
            kernel_function, observation_sampler,
        ) - reference
        simulation_result = _run_method(method, time, curve, errors, kwargs)
        simulation_coordinate, score = _extract_statistic(
            simulation_result, statistic_extractor
        )
        if (simulation_coordinate.shape != coordinate.shape
                or not np.allclose(simulation_coordinate, coordinate, rtol=1e-10,
                                   atol=1e-12)):
            raise ValueError(
                "method returned a different coordinate grid for a simulation"
            )
        simulated[row] = score
        if progress and ((row + 1) % interval == 0 or row + 1 == n_simulations):
            print(
                f"\rSignificance simulations: {row + 1}/{int(n_simulations)}",
                end="", flush=True,
            )
    if progress:
        print()

    valid_rows = np.any(np.isfinite(simulated[:, mask]), axis=1)
    if not np.any(valid_rows):
        raise RuntimeError("all simulated statistics are non-finite in search_range")
    simulated = simulated[valid_rows]
    simulated_max = np.nanmax(simulated[:, mask], axis=1)
    local_p = _empirical_pvalues(observed, simulated)
    global_p = _global_empirical_pvalues(observed, simulated_max)
    global_p[~mask] = np.nan
    local_thresholds = np.nanpercentile(
        simulated, _SIGMA_PERCENTILES, axis=0
    )
    global_thresholds = np.nanpercentile(
        simulated_max, _SIGMA_PERCENTILES
    )
    output = SignificanceResult(
        coordinate=coordinate,
        observed_statistic=observed,
        simulated_statistic=simulated,
        simulated_max=simulated_max,
        local_pvalue=local_p,
        global_pvalue=global_p,
        local_sigma_thresholds=local_thresholds,
        global_sigma_thresholds=global_thresholds,
        sigma_levels=_SIGMA_LEVELS.copy(),
        method_name=_method_name(method),
        method_kwargs=kwargs,
        search_range=None if search_range is None else tuple(map(float, search_range)),
        n_simulations_requested=int(n_simulations),
        seed=seed,
        residual=bool(residual),
        reference_mean=reference.copy(),
    )
    if save_path is not None:
        output.save_npz(save_path)
    return output
