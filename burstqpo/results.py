"""Small result containers shared by the independent period-search methods."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


def _output_path(path, suffix):
    output = Path(path)
    if output.suffix.lower() != suffix:
        output = output.with_suffix(suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _scalar(data, name, default):
    return data[name].item() if name in data.files else default


def _json_safe(value):
    """Return a strict-JSON representation of nested metadata."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class WWZResult:
    tau: np.ndarray
    frequency: np.ndarray
    power: np.ndarray
    coi: np.ndarray
    p_max: float
    amplitude: np.ndarray
    n_eff: np.ndarray
    _c: float = field(default=0.0125, repr=False, compare=False)
    time_unit: str = "unknown"
    time_origin: float = 0.0
    frequency_min: float = np.nan
    frequency_max: float = np.nan
    frequency_step: float = np.nan
    tau_number: int = 0
    n_observations: int = 0
    time: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    values: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    uncertainty: Optional[np.ndarray] = None

    @property
    def period(self): return 1.0 / self.frequency
    @property
    def c(self): return self._c
    @property
    def wwz(self): return self.power
    @property
    def Z(self): return self.power
    @property
    def z_projection(self): return self.project()
    def project(self, *, use_coi=True):
        from .algorithms.wwz import project_wwz
        return project_wwz(self.power, self.tau, self.frequency, self._c, use_coi=use_coi)
    def find_peaks(self, *, use_coi=True, **kwargs):
        from .algorithms.wwz import find_wwz_peaks
        return find_wwz_peaks(self, use_coi=use_coi, **kwargs)
    def plot_wwz(self, **kwargs):
        from .visualization import plot_wwz
        return plot_wwz(self, **kwargs)
    @property
    def metadata(self):
        return {"time_unit": self.time_unit, "time_origin": self.time_origin,
                "frequency_min": self.frequency_min, "frequency_max": self.frequency_max,
                "frequency_step": self.frequency_step, "tau_number": self.tau_number or len(self.tau),
                "n_observations": self.n_observations, "c": self._c}
    def save_npz(self, path):
        output = _output_path(path, ".npz")
        np.savez_compressed(output, tau=self.tau, frequency=self.frequency, power=self.power,
                            coi=self.coi, p_max=self.p_max, amplitude=self.amplitude, n_eff=self.n_eff,
                            c=self._c, time_unit=self.time_unit, time_origin=self.time_origin,
                            frequency_min=self.frequency_min, frequency_max=self.frequency_max,
                            frequency_step=self.frequency_step, tau_number=self.tau_number,
                            n_observations=self.n_observations, time=self.time, values=self.values,
                            uncertainty=(np.asarray([]) if self.uncertainty is None
                                         else self.uncertainty))
        return output
    @classmethod
    def load_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            required = {"tau", "frequency", "power", "coi", "p_max", "amplitude", "n_eff"}
            missing = required.difference(data.files)
            if missing: raise ValueError(f"NPZ file is missing WWZ fields: {sorted(missing)}")
            frequency = np.asarray(data["frequency"], float)
            time = np.asarray(data["time"], float) if "time" in data.files else np.asarray([], dtype=float)
            values = np.asarray(data["values"], float) if "values" in data.files else np.asarray([], dtype=float)
            uncertainty = (np.asarray(data["uncertainty"], float)
                           if "uncertainty" in data.files and data["uncertainty"].size else None)
            return cls(np.asarray(data["tau"], float), frequency, np.asarray(data["power"], float),
                       np.asarray(data["coi"], float), float(data["p_max"]),
                       np.asarray(data["amplitude"], float), np.asarray(data["n_eff"], float),
                       float(_scalar(data, "c", 0.0125)), str(_scalar(data, "time_unit", "unknown")),
                       float(_scalar(data, "time_origin", 0.0)), float(_scalar(data, "frequency_min", frequency[0])),
                       float(_scalar(data, "frequency_max", frequency[-1])),
                       float(_scalar(data, "frequency_step", np.nan)), int(_scalar(data, "tau_number", len(data["tau"]))),
                       int(_scalar(data, "n_observations", 0)), time, values, uncertainty)


WWZResults = WWZResult


@dataclass(frozen=True)
class LombScargleResult:
    time: np.ndarray
    values: np.ndarray
    frequency: np.ndarray
    power: np.ndarray
    mode: str
    uncertainty: Optional[np.ndarray] = None
    time_unit: str = "unknown"
    time_origin: float = 0.0
    frequency_min: float = np.nan
    frequency_max: float = np.nan
    frequency_step: float = np.nan
    divide_freq_step: float = 10.0
    normalization: str = "standard"
    fit_mean: bool = True
    @property
    def period(self): return 1.0 / self.frequency
    @property
    def is_generalized(self): return self.mode == "glsp"
    @property
    def metadata(self):
        return {"mode": self.mode, "time_unit": self.time_unit, "time_origin": self.time_origin,
                "frequency_min": self.frequency_min, "frequency_max": self.frequency_max,
                "frequency_step": self.frequency_step, "divide_freq_step": self.divide_freq_step,
                "normalization": self.normalization, "fit_mean": self.fit_mean,
                "n_observations": len(self.time)}
    def find_peaks(self, **kwargs):
        from .algorithms.lomb_scargle import find_lomb_scargle_peaks
        return find_lomb_scargle_peaks(self, **kwargs)
    def plot_lomb_scargle(self, **kwargs):
        from .visualization import plot_lomb_scargle
        return plot_lomb_scargle(self, **kwargs)
    def save_npz(self, path):
        output = _output_path(path, ".npz")
        np.savez_compressed(output, time=self.time, values=self.values, frequency=self.frequency,
                            power=self.power, mode=self.mode,
                            uncertainty=np.asarray([]) if self.uncertainty is None else self.uncertainty,
                            time_unit=self.time_unit, time_origin=self.time_origin,
                            frequency_min=self.frequency_min, frequency_max=self.frequency_max,
                            frequency_step=self.frequency_step, divide_freq_step=self.divide_freq_step,
                            normalization=self.normalization, fit_mean=self.fit_mean)
        return output
    @classmethod
    def load_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            required = {"time", "values", "frequency", "power", "mode"}
            missing = required.difference(data.files)
            if missing: raise ValueError(f"NPZ file is missing LSP fields: {sorted(missing)}")
            frequency = np.asarray(data["frequency"], float)
            uncertainty = np.asarray(data["uncertainty"], float) if "uncertainty" in data.files and data["uncertainty"].size else None
            return cls(np.asarray(data["time"], float), np.asarray(data["values"], float), frequency,
                       np.asarray(data["power"], float), str(data["mode"]), uncertainty,
                       str(_scalar(data, "time_unit", "unknown")), float(_scalar(data, "time_origin", 0.0)),
                       float(_scalar(data, "frequency_min", frequency[0])), float(_scalar(data, "frequency_max", frequency[-1])),
                       float(_scalar(data, "frequency_step", np.nan)), float(_scalar(data, "divide_freq_step", 10.0)),
                       str(_scalar(data, "normalization", "standard")), bool(_scalar(data, "fit_mean", True)))


LombScargleResults = LombScargleResult


@dataclass(frozen=True)
class JurkevichResult:
    time: np.ndarray
    values: np.ndarray
    period: np.ndarray
    v_norm: np.ndarray
    m: int
    time_unit: str = "unknown"
    time_origin: float = 0.0
    period_min: float = np.nan
    period_max: float = np.nan
    period_step: float = np.nan
    n_observations: int = 0
    total_variance: float = np.nan
    @property
    def statistic(self): return self.v_norm
    @property
    def score(self): return 1.0 - self.v_norm
    @property
    def metadata(self):
        return {"m": self.m, "time_unit": self.time_unit, "time_origin": self.time_origin,
                "period_min": self.period_min, "period_max": self.period_max,
                "period_step": self.period_step, "n_observations": self.n_observations,
                "total_variance": self.total_variance}
    def find_peaks(self, **kwargs):
        from .algorithms.jurkevich import find_jurkevich_peaks
        return find_jurkevich_peaks(self, **kwargs)
    def plot_jurkevich(self, **kwargs):
        from .visualization import plot_jurkevich
        return plot_jurkevich(self, **kwargs)
    def save_npz(self, path):
        output = _output_path(path, ".npz")
        np.savez_compressed(output, time=self.time, values=self.values, period=self.period,
                            v_norm=self.v_norm, m=self.m, time_unit=self.time_unit,
                            time_origin=self.time_origin, period_min=self.period_min,
                            period_max=self.period_max, period_step=self.period_step,
                            n_observations=self.n_observations, total_variance=self.total_variance)
        return output
    @classmethod
    def load_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            period = np.asarray(data["period"], float)
            return cls(np.asarray(data["time"], float), np.asarray(data["values"], float), period,
                       np.asarray(data["v_norm"], float), int(data["m"]), str(_scalar(data, "time_unit", "unknown")),
                       float(_scalar(data, "time_origin", 0.0)), float(_scalar(data, "period_min", period[0])),
                       float(_scalar(data, "period_max", period[-1])), float(_scalar(data, "period_step", np.nan)),
                       int(_scalar(data, "n_observations", 0)), float(_scalar(data, "total_variance", np.nan)))


JurkevichResults = JurkevichResult


@dataclass(frozen=True)
class DCFResult:
    time: np.ndarray
    values: np.ndarray
    lag: np.ndarray
    correlation: np.ndarray
    error: np.ndarray
    pair_count: np.ndarray
    delta_tau: float
    c: float
    max_tau: float
    time_unit: str = "unknown"
    time_origin: float = 0.0
    n_observations: int = 0
    sample_std: float = np.nan
    @property
    def tau(self): return self.lag
    @property
    def dcf(self): return self.correlation
    @property
    def metadata(self):
        return {"delta_tau": self.delta_tau, "c": self.c, "max_tau": self.max_tau,
                "time_unit": self.time_unit, "time_origin": self.time_origin,
                "n_observations": self.n_observations, "sample_std": self.sample_std}
    def find_peaks(self, **kwargs):
        from .algorithms.dcf import find_dcf_peaks
        return find_dcf_peaks(self, **kwargs)
    def plot_dcf(self, **kwargs):
        from .visualization import plot_dcf
        return plot_dcf(self, **kwargs)
    def save_npz(self, path):
        output = _output_path(path, ".npz")
        np.savez_compressed(output, time=self.time, values=self.values, lag=self.lag,
                            correlation=self.correlation, error=self.error, pair_count=self.pair_count,
                            delta_tau=self.delta_tau, c=self.c, max_tau=self.max_tau,
                            time_unit=self.time_unit, time_origin=self.time_origin,
                            n_observations=self.n_observations, sample_std=self.sample_std)
        return output
    @classmethod
    def load_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(np.asarray(data["time"], float), np.asarray(data["values"], float),
                       np.asarray(data["lag"], float), np.asarray(data["correlation"], float),
                       np.asarray(data["error"], float), np.asarray(data["pair_count"], int),
                       float(data["delta_tau"]), float(data["c"]), float(data["max_tau"]),
                       str(_scalar(data, "time_unit", "unknown")), float(_scalar(data, "time_origin", 0.0)),
                       int(_scalar(data, "n_observations", 0)), float(_scalar(data, "sample_std", np.nan)))


DCFResults = DCFResult


@dataclass(frozen=True)
class Z2nResult:
    event_times: np.ndarray
    frequency: np.ndarray
    statistic: np.ndarray
    n_harmonics: int
    time_origin: float
    time_unit: str = "unknown"
    frequency_min: float = np.nan
    frequency_max: float = np.nan
    frequency_step: float = np.nan
    n_events: int = 0
    local_pvalue: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    global_pvalues: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    n_trials: float = 1.0
    @property
    def z2n(self): return self.statistic
    @property
    def period(self): return 1.0 / self.frequency
    def global_pvalue(self, n_trials=None):
        """Return trial-corrected p-values, optionally with another trial count."""
        if n_trials is None:
            return self.global_pvalues
        from .algorithms.z2n import global_pvalue_from_trials
        return global_pvalue_from_trials(self.local_pvalue, n_trials)
    def z_local_threshold(self, alpha="0.1%"):
        """Return the local Z^2_n threshold for a significance level."""
        from .algorithms.z2n import z2n_local_threshold
        return z2n_local_threshold(alpha, self.n_harmonics)
    def z_global_threshold(self, alpha="0.1%", n_trials=None):
        """Return the global Z^2_n threshold for a significance level."""
        from .algorithms.z2n import z2n_global_threshold
        return z2n_global_threshold(alpha, self.n_harmonics,
                                    self.n_trials if n_trials is None else n_trials)
    z2n_local_threshold = z_local_threshold
    z2n_global_threshold = z_global_threshold
    @property
    def metadata(self):
        return {"n_harmonics": self.n_harmonics, "time_origin": self.time_origin,
                "time_unit": self.time_unit, "frequency_min": self.frequency_min,
                "frequency_max": self.frequency_max, "frequency_step": self.frequency_step,
                "n_events": self.n_events, "n_trials": self.n_trials}
    def find_peaks(self, **kwargs):
        from .algorithms.z2n import find_z2n_peaks
        return find_z2n_peaks(self, **kwargs)
    def plot_z2n(self, **kwargs):
        from .visualization import plot_z2n
        return plot_z2n(self, **kwargs)
    def save_npz(self, path):
        output = _output_path(path, ".npz")
        np.savez_compressed(output, event_times=self.event_times, frequency=self.frequency,
                            statistic=self.statistic, n_harmonics=self.n_harmonics,
                            time_origin=self.time_origin, time_unit=self.time_unit,
                            frequency_min=self.frequency_min, frequency_max=self.frequency_max,
                            frequency_step=self.frequency_step, n_events=self.n_events,
                            local_pvalue=self.local_pvalue, global_pvalues=self.global_pvalues,
                            n_trials=self.n_trials)
        return output
    @classmethod
    def load_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            frequency = np.asarray(data["frequency"], float)
            statistic = np.asarray(data["statistic"], float)
            n_harmonics = int(data["n_harmonics"])
            if "local_pvalue" in data.files or "local_pvalues" in data.files:
                local_pvalue = np.asarray(data["local_pvalue" if "local_pvalue" in data.files else "local_pvalues"], float)
                global_pvalue = np.asarray(data["global_pvalues" if "global_pvalues" in data.files else "global_pvalue"], float)
                n_trials = float(_scalar(data, "n_trials", 1.0))
            else:
                from .algorithms.z2n import (estimate_z2n_trials,
                                             global_pvalue_from_trials,
                                             z2n_single_pvalue)
                local_pvalue = z2n_single_pvalue(statistic, n_harmonics)
                n_trials = estimate_z2n_trials(data["event_times"], frequency)
                global_pvalue = global_pvalue_from_trials(local_pvalue, n_trials)
            return cls(np.asarray(data["event_times"], float), frequency, statistic,
                       n_harmonics, float(data["time_origin"]), str(_scalar(data, "time_unit", "unknown")),
                       float(_scalar(data, "frequency_min", frequency[0])), float(_scalar(data, "frequency_max", frequency[-1])),
                       float(_scalar(data, "frequency_step", np.nan)), int(_scalar(data, "n_events", len(data["event_times"]))),
                       local_pvalue, global_pvalue, n_trials)


Z2nResults = Z2nResult


@dataclass(frozen=True)
class MCMCResult:
    """Posterior samples and diagnostics produced by :func:`run_mcmc`.

    ``samples`` contains the flattened post-burn-in chain, with one parameter
    vector per row.  The ``posterior`` property is provided as the descriptive
    name for this array, while ``samples`` remains available for compatibility
    with the original tuple-returning implementation.
    """

    samples: np.ndarray
    log_probability: np.ndarray
    acceptance_fraction: np.ndarray
    autocorrelation_time: np.ndarray
    model_id: str = ""
    seed: Optional[int] = None
    parameter_names: tuple = ()
    initial: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    start_scale: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    sampler_config: dict = field(default_factory=dict)
    prior_metadata: dict = field(default_factory=dict)
    model_metadata: dict = field(default_factory=dict)
    likelihood_name: str = ""
    extra_metadata: dict = field(default_factory=dict)
    log_prior_values: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    log_likelihood_values: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    n_observations: Optional[int] = None
    time: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    counts: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    errors: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))

    @property
    def posterior(self):
        """Flattened posterior samples, shaped ``(n_samples, n_parameters)``."""
        return self.samples

    @property
    def posterior_samples(self):
        """Alias for :attr:`posterior` useful in downstream analysis code."""
        return self.samples

    @property
    def log_prob(self):
        """Short alias for :attr:`log_probability`."""
        return self.log_probability

    @property
    def log_posterior(self):
        """Alias for :attr:`log_probability`."""
        return self.log_probability

    @property
    def log_prior(self):
        """Per-sample log-prior values, when available."""
        return self.log_prior_values

    @property
    def log_likelihood(self):
        """Per-sample log-likelihood values, when available."""
        return self.log_likelihood_values

    @property
    def acceptance(self):
        """Alias for the per-walker acceptance fractions."""
        return self.acceptance_fraction

    @property
    def autocorr_time(self):
        """Short alias for :attr:`autocorrelation_time`."""
        return self.autocorrelation_time

    @property
    def mean_acceptance_fraction(self):
        """Mean acceptance fraction across walkers."""
        return float(np.mean(self.acceptance_fraction))

    @property
    def ndim(self):
        """Number of fitted parameters."""
        if self.samples.ndim != 2:
            return 0
        return self.samples.shape[1]

    @property
    def n_samples(self):
        """Number of flattened posterior samples."""
        if self.samples.ndim == 0:
            return 0
        return self.samples.shape[0]

    @property
    def n_walkers(self):
        """Number of walkers represented by :attr:`acceptance_fraction`."""
        return self.acceptance_fraction.size

    @property
    def prior(self):
        """Serializable description attached by the prior builder."""
        return self.prior_metadata

    @property
    def models(self):
        """Serializable descriptions of the likelihood model components."""
        return self.model_metadata

    @property
    def parameters(self):
        """Map parameter names to their initial values."""
        return dict(zip(self.parameter_names, self.initial.tolist()))

    @property
    def posterior_by_parameter(self):
        """Map each parameter name to its posterior sample column."""
        return {
            name: self.samples[:, index]
            for index, name in enumerate(self.parameter_names)
        }

    @property
    def posterior_means(self):
        """Map parameter names to posterior sample means."""
        return {
            name: float(value)
            for name, value in zip(
                self._parameter_labels, np.mean(self.samples, axis=0)
            )
        }

    @property
    def posterior_mean(self):
        """Alias for :attr:`posterior_means`."""
        return self.posterior_means

    @property
    def posterior_medians(self):
        """Map parameter names to posterior medians."""
        return {
            name: interval[0]
            for name, interval in self.posterior_1sigma.items()
        }

    @property
    def posterior_standard_deviations(self):
        """Map parameter names to posterior sample standard deviations."""
        return {
            name: float(value)
            for name, value in zip(
                self._parameter_labels, np.std(self.samples, axis=0)
            )
        }

    @property
    def posterior_1sigma(self):
        """Return median and asymmetric central 68% errors by parameter.

        Each value is ``(median, lower_error, upper_error)``. The lower error
        is negative, so the tuple can be displayed directly as
        ``median_{lower_error}^{+upper_error}``.
        """
        quantiles = np.quantile(self.samples, [0.158655, 0.5, 0.841345], axis=0)
        lower, median, upper = quantiles
        return {
            name: (float(center), float(low - center), float(high - center))
            for name, low, center, high in zip(
                self._parameter_labels, lower, median, upper
            )
        }

    @property
    def posterior_1sigma_values(self):
        """Central 68% summaries in parameter-column order."""
        return list(self.posterior_1sigma.values())

    @property
    def maximum_log_posterior(self):
        """Largest sampled log posterior value."""
        return float(np.max(self.log_probability))

    @property
    def maximum_a_posteriori_sample(self):
        """Sample vector with the largest sampled log posterior."""
        return self.samples[int(np.argmax(self.log_probability))].copy()

    @property
    def maximum_a_posteriori_parameters(self):
        """Named parameters at the largest sampled log posterior."""
        return dict(zip(
            self._parameter_labels, self.maximum_a_posteriori_sample.tolist()
        ))

    @property
    def maximum_log_likelihood(self):
        """Largest sampled log likelihood, or NaN for legacy results."""
        if self.log_likelihood_values.size != self.n_samples:
            return float("nan")
        return float(np.max(self.log_likelihood_values))

    @property
    def maximum_likelihood(self):
        """Largest sampled likelihood in linear space.

        This can underflow to zero for long data sets;
        :attr:`maximum_log_likelihood` is normally preferable.
        """
        return float(np.exp(self.maximum_log_likelihood))

    @property
    def maximum_log_prior(self):
        """Largest sampled log-prior value, or NaN for legacy results."""
        if self.log_prior_values.size != self.n_samples:
            return float("nan")
        return float(np.max(self.log_prior_values))

    @property
    def aic(self):
        """AIC using the largest likelihood found in the retained chain."""
        if not np.isfinite(self.maximum_log_likelihood):
            return float("nan")
        return float(2 * self.ndim - 2 * self.maximum_log_likelihood)

    @property
    def bic(self):
        """BIC using the largest retained likelihood and data size."""
        if (self.n_observations is None or self.n_observations <= 0
                or not np.isfinite(self.maximum_log_likelihood)):
            return float("nan")
        return float(
            self.ndim * np.log(self.n_observations)
            - 2 * self.maximum_log_likelihood
        )

    @property
    def maximum_likelihood_sample(self):
        """Sample vector with the largest likelihood."""
        if self.log_likelihood_values.size != self.n_samples:
            raise ValueError(
                "log-likelihood values are unavailable in this result; "
                "rerun MCMC with the current burstqpo version"
            )
        return self.samples[int(np.argmax(self.log_likelihood_values))].copy()

    @property
    def maximum_likelihood_parameters(self):
        """Named parameters at the largest sampled likelihood."""
        return dict(zip(
            self._parameter_labels, self.maximum_likelihood_sample.tolist()
        ))

    @property
    def _parameter_labels(self):
        if len(self.parameter_names) == self.ndim:
            return self.parameter_names
        return tuple(f"theta_{index}" for index in range(self.ndim))

    def parameter(self, name):
        """Return posterior samples for one named parameter."""
        try:
            index = self.parameter_names.index(str(name))
        except ValueError as exc:
            raise KeyError(f"unknown parameter: {name}") from exc
        return self.samples[:, index]

    def summary(self):
        """Return commonly used posterior summaries as a serializable dict."""
        return _json_safe({
            "mean": self.posterior_means,
            "standard_deviation": self.posterior_standard_deviations,
            "one_sigma": self.posterior_1sigma,
            "maximum_log_posterior": self.maximum_log_posterior,
            "maximum_a_posteriori_parameters": (
                self.maximum_a_posteriori_parameters
            ),
            "maximum_log_likelihood": self.maximum_log_likelihood,
            "maximum_likelihood": self.maximum_likelihood,
            "maximum_likelihood_parameters": (
                self.maximum_likelihood_parameters
                if self.log_likelihood_values.size == self.n_samples else None
            ),
            "aic": self.aic,
            "bic": self.bic,
        })

    def plot_corner(self, **kwargs):
        """Plot one- and two-dimensional posterior projections."""
        from .visualization import plot_mcmc_corner

        return plot_mcmc_corner(self, **kwargs)

    def qpo_posterior(self, qpo_index=0):
        """Return physical QPO quantities for every posterior sample.

        For ``a exp(-c|tau|) cos(2*pi*f*tau)``, this returns ``a``, ``c``,
        ``f``, period ``1/f``, damping time ``1/c``, coherence cycles ``f/c``,
        and quality factor ``Q=pi*f/c``.
        """
        terms = []

        def collect(value):
            if isinstance(value, dict):
                if value.get("name") == "qpo":
                    terms.append(value)
                for item in value.values():
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(self.model_metadata.get("kernel_function", {}))
        if not terms:
            raise ValueError("result metadata does not contain a QPO kernel")
        try:
            indices = tuple(int(index) for index in terms[qpo_index]["parameter_indices"])
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("invalid QPO kernel metadata") from exc
        if len(indices) != 3 or max(indices) >= self.ndim:
            raise ValueError("invalid QPO parameter indices in result metadata")

        log_a, log_c, log_f = self.samples[:, indices].T
        amplitude = np.exp(log_a)
        decay_rate = np.exp(log_c)
        frequency = np.exp(log_f)
        return {
            "amplitude": amplitude,
            "decay_rate": decay_rate,
            "frequency": frequency,
            "period": 1.0 / frequency,
            "damping_time": 1.0 / decay_rate,
            "coherence_cycles": frequency / decay_rate,
            "quality_factor": np.pi * frequency / decay_rate,
        }

    def qpo_summary(self, qpo_index=0):
        """Summarize QPO posterior quantities with central 68% intervals.

        Each result is ``(median, lower_error, upper_error)`` with a negative
        lower error. A constrained QPO posterior describes the fitted QPO
        component; it does not by itself establish evidence for a QPO over a
        red-noise-only model.
        """
        derived = self.qpo_posterior(qpo_index=qpo_index)

        def summarize(values):
            low, center, high = np.quantile(
                values, [0.158655, 0.5, 0.841345]
            )
            return (
                float(center), float(low - center), float(high - center)
            )

        return {name: summarize(values) for name, values in derived.items()}

    def predict_qpo(
        self,
        time,
        counts,
        errors,
        mean_function,
        kernel_function,
        qpo_kernel,
        *,
        prediction_time=None,
        max_samples=256,
        random_state=None,
    ):
        """Return the posterior predictive latent QPO component.

        The returned mapping contains ``time``, ``mean``, ``std`` and
        ``covariance``. ``mean`` and ``std`` include uncertainty from both the
        conditional GP and variation of parameters across the retained chain.
        ``qpo_kernel`` must be the QPO component, while ``kernel_function`` is
        the full covariance used in the likelihood.
        """
        if not callable(mean_function) or not callable(kernel_function):
            raise TypeError("mean_function and kernel_function must be callable")
        if not callable(qpo_kernel):
            raise TypeError("qpo_kernel must be callable")
        training_time = np.asarray(time, dtype=float)
        if prediction_time is None:
            prediction_time = training_time
        else:
            prediction_time = np.asarray(prediction_time, dtype=float)
        if training_time.ndim != 1 or not training_time.size:
            raise ValueError("time must be a non-empty one-dimensional array")
        if prediction_time.ndim != 1 or not prediction_time.size:
            raise ValueError(
                "prediction_time must be a non-empty one-dimensional array"
            )
        if max_samples is None or self.n_samples <= int(max_samples):
            indices = np.arange(self.n_samples)
        else:
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError("max_samples must be positive or None")
            rng = np.random.default_rng(random_state)
            indices = rng.choice(self.n_samples, max_samples, replace=False)

        from .bayesian.gp import qpo_conditional

        conditional_means = []
        conditional_covariances = []
        for index in indices:
            conditional_mean, conditional_covariance = qpo_conditional(
                self.samples[index],
                training_time,
                counts,
                errors,
                mean_function,
                kernel_function,
                qpo_kernel,
                prediction_time,
            )
            conditional_means.append(conditional_mean)
            conditional_covariances.append(conditional_covariance)

        means = np.asarray(conditional_means, dtype=float)
        covariances = np.asarray(conditional_covariances, dtype=float)
        posterior_mean = np.mean(means, axis=0)
        posterior_covariance = np.mean(covariances, axis=0)
        centered = means - posterior_mean
        posterior_covariance = posterior_covariance + (
            centered.T @ centered / len(means)
        )
        posterior_covariance = 0.5 * (
            posterior_covariance + posterior_covariance.T
        )
        return {
            "time": prediction_time.copy(),
            "mean": posterior_mean,
            "std": np.sqrt(np.maximum(np.diag(posterior_covariance), 0.0)),
            "covariance": posterior_covariance,
            "sample_indices": np.asarray(indices, dtype=int),
        }

    def __iter__(self):
        """Allow legacy four-value unpacking of an MCMC result."""
        yield self.samples
        yield self.log_probability
        yield self.mean_acceptance_fraction
        yield self.autocorrelation_time

    @property
    def metadata(self):
        return _json_safe({
            "model_id": self.model_id,
            "seed": self.seed,
            "ndim": self.ndim,
            "n_samples": self.n_samples,
            "n_walkers": self.n_walkers,
            "n_observations": self.n_observations,
            "parameter_names": list(self.parameter_names),
            "initial": self.initial.tolist(),
            "start_scale": self.start_scale.tolist(),
            "sampler_config": dict(self.sampler_config),
            "prior": dict(self.prior_metadata),
            "models": dict(self.model_metadata),
            "likelihood": self.likelihood_name,
            "extra": dict(self.extra_metadata),
            "mean_acceptance_fraction": self.mean_acceptance_fraction,
            "autocorrelation_time": self.autocorrelation_time.tolist(),
            "posterior_summary": self.summary(),
            "has_log_prior_values": (
                self.log_prior_values.size == self.n_samples
            ),
            "has_log_likelihood_values": (
                self.log_likelihood_values.size == self.n_samples
            ),
            "has_observations": (
                self.time.ndim == self.counts.ndim == self.errors.ndim == 1
                and self.time.size > 0
                and self.time.size == self.counts.size == self.errors.size
            ),
        })

    def save_metadata_json(self, path):
        """Write metadata as JSON and return the output path."""
        output = _output_path(path, ".json")
        output.write_text(
            json.dumps(
                self.metadata, indent=2, ensure_ascii=False, allow_nan=False
            ),
            encoding="utf-8",
        )
        return output

    def save_metadata_txt(self, path):
        """Write human-readable metadata and return the output path."""
        output = _output_path(path, ".txt")
        lines = ["MCMC result metadata", "===================="]
        for key, value in self.metadata.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(
                    value, indent=2, ensure_ascii=False, allow_nan=False
                )
            lines.append(f"{key}: {value}")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output

    def save_metadata(self, path):
        """Write metadata as JSON, or TXT when the path ends in ``.txt``."""
        return (
            self.save_metadata_txt(path)
            if Path(path).suffix.lower() == ".txt"
            else self.save_metadata_json(path)
        )

    def save_npz(self, path):
        """Save this result to a compressed NPZ file and return its path."""
        output = _output_path(path, ".npz")
        np.savez_compressed(
            output,
            samples=self.samples,
            log_probability=self.log_probability,
            acceptance_fraction=self.acceptance_fraction,
            autocorrelation_time=self.autocorrelation_time,
            model_id=self.model_id,
            seed=-1 if self.seed is None else int(self.seed),
            parameter_names=np.asarray(self.parameter_names, dtype=str),
            initial=self.initial,
            start_scale=self.start_scale,
            log_prior_values=self.log_prior_values,
            log_likelihood_values=self.log_likelihood_values,
            n_observations=(
                -1 if self.n_observations is None else self.n_observations
            ),
            time=self.time,
            counts=self.counts,
            errors=self.errors,
            metadata_json=json.dumps(
                self.metadata, ensure_ascii=False, allow_nan=False
            ),
        )
        return output

    @classmethod
    def load_npz(cls, path):
        """Load an :class:`MCMCResult` from a compressed NPZ file."""
        with np.load(Path(path), allow_pickle=False) as data:
            required = {
                "samples",
                "log_probability",
                "acceptance_fraction",
                "autocorrelation_time",
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(
                    f"NPZ file is missing MCMC fields: {sorted(missing)}"
                )

            seed = _scalar(data, "seed", -1)
            seed = None if int(seed) < 0 else int(seed)
            metadata = {}
            if "metadata_json" in data.files:
                metadata = json.loads(str(data["metadata_json"].item()))
            parameter_names = tuple(
                str(name) for name in (
                    data["parameter_names"] if "parameter_names" in data.files
                    else metadata.get("parameter_names", [])
                )
            )
            return cls(
                np.asarray(data["samples"], dtype=float),
                np.asarray(data["log_probability"], dtype=float),
                np.asarray(data["acceptance_fraction"], dtype=float),
                np.asarray(data["autocorrelation_time"], dtype=float),
                str(_scalar(data, "model_id", "")),
                seed,
                parameter_names,
                np.asarray(data["initial"], dtype=float) if "initial" in data.files else np.asarray(metadata.get("initial", []), dtype=float),
                np.asarray(data["start_scale"], dtype=float) if "start_scale" in data.files else np.asarray(metadata.get("start_scale", []), dtype=float),
                dict(metadata.get("sampler_config", {})),
                dict(metadata.get("prior", {})),
                dict(metadata.get("models", {})),
                str(metadata.get("likelihood", "")),
                dict(metadata.get("extra", {})),
                np.asarray(data["log_prior_values"], dtype=float)
                if "log_prior_values" in data.files else np.asarray([], dtype=float),
                np.asarray(data["log_likelihood_values"], dtype=float)
                if "log_likelihood_values" in data.files else np.asarray([], dtype=float),
                (lambda value: None if value < 0 else value)(
                    int(_scalar(
                        data, "n_observations",
                        metadata.get("n_observations", -1),
                    ))
                ),
                np.asarray(data["time"], dtype=float)
                if "time" in data.files else np.asarray([], dtype=float),
                np.asarray(data["counts"], dtype=float)
                if "counts" in data.files else np.asarray([], dtype=float),
                np.asarray(data["errors"], dtype=float)
                if "errors" in data.files else np.asarray([], dtype=float),
            )


MCMCResults = MCMCResult
