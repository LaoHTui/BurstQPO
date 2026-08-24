"""Small result containers shared by the independent period-search methods."""

from __future__ import annotations

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
                            n_observations=self.n_observations)
        return output
    @classmethod
    def load_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            required = {"tau", "frequency", "power", "coi", "p_max", "amplitude", "n_eff"}
            missing = required.difference(data.files)
            if missing: raise ValueError(f"NPZ file is missing WWZ fields: {sorted(missing)}")
            frequency = np.asarray(data["frequency"], float)
            return cls(np.asarray(data["tau"], float), frequency, np.asarray(data["power"], float),
                       np.asarray(data["coi"], float), float(data["p_max"]),
                       np.asarray(data["amplitude"], float), np.asarray(data["n_eff"], float),
                       float(_scalar(data, "c", 0.0125)), str(_scalar(data, "time_unit", "unknown")),
                       float(_scalar(data, "time_origin", 0.0)), float(_scalar(data, "frequency_min", frequency[0])),
                       float(_scalar(data, "frequency_max", frequency[-1])),
                       float(_scalar(data, "frequency_step", np.nan)), int(_scalar(data, "tau_number", len(data["tau"]))),
                       int(_scalar(data, "n_observations", 0)))


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
    @property
    def z2n(self): return self.statistic
    @property
    def period(self): return 1.0 / self.frequency
    @property
    def metadata(self):
        return {"n_harmonics": self.n_harmonics, "time_origin": self.time_origin,
                "time_unit": self.time_unit, "frequency_min": self.frequency_min,
                "frequency_max": self.frequency_max, "frequency_step": self.frequency_step,
                "n_events": self.n_events}
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
                            frequency_step=self.frequency_step, n_events=self.n_events)
        return output
    @classmethod
    def load_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            frequency = np.asarray(data["frequency"], float)
            return cls(np.asarray(data["event_times"], float), frequency, np.asarray(data["statistic"], float),
                       int(data["n_harmonics"]), float(data["time_origin"]), str(_scalar(data, "time_unit", "unknown")),
                       float(_scalar(data, "frequency_min", frequency[0])), float(_scalar(data, "frequency_max", frequency[-1])),
                       float(_scalar(data, "frequency_step", np.nan)), int(_scalar(data, "n_events", len(data["event_times"]))))


Z2nResults = Z2nResult
