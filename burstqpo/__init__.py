"""Reusable QPO period-search algorithms for plain time-series arrays."""

from .algorithms.wwz import compute_wwz, find_wwz_peaks, project_wwz, wwz
from .algorithms.lomb_scargle import compute_lomb_scargle, compute_lsp, find_lomb_scargle_peaks, lomb_scargle, lsp
from .algorithms.jurkevich import compute_jurkevich, find_jurkevich_peaks, jurkevich
from .algorithms.dcf import compute_dcf, dcf, find_dcf_peaks
from .algorithms.z2n import (compute_z2n, estimate_z2n_trials, find_z2n_peaks,
                             global_pvalue_from_trials, z2n, z2n_global_threshold,
                             z2n_local_threshold, z2n_single_pvalue)
from .results import (JurkevichResult, JurkevichResults, LombScargleResult,
                      LombScargleResults, DCFResult, DCFResults, Z2nResult,
                      Z2nResults, WWZResult, WWZResults, MCMCResult,
                      MCMCResults)
from .mcmc import run_mcmc
from .significance import (LightCurveSimulationResult, PeakSignificance,
                           SignificanceResult, SignificanceResults,
                           calculate_significance,
                           simulate_light_curves)
from .bayesian.evidence import (bayes_factor, calculate_log_evidence,
                                run_nested_sampling)
from .models import (add_mean_models, constant_background,
                     constant_background_model, ercod, ercod_model, fred,
                     fred_model, linear_background, linear_background_model,
                     polynomial_background, polynomial_background_model)
from .visualization import (plot_dcf, plot_jurkevich, plot_lomb_scargle,
                            plot_wwz, plot_z2n)

__all__ = [
    "WWZResult", "WWZResults", "wwz", "compute_wwz",
    "find_wwz_peaks", "plot_wwz", "project_wwz",
    "LombScargleResult", "LombScargleResults", "lomb_scargle", "lsp",
    "compute_lomb_scargle", "compute_lsp", "find_lomb_scargle_peaks", "plot_lomb_scargle",
    "JurkevichResult", "JurkevichResults", "jurkevich", "compute_jurkevich",
    "find_jurkevich_peaks", "plot_jurkevich",
    "DCFResult", "DCFResults", "dcf", "compute_dcf", "find_dcf_peaks", "plot_dcf",
    "Z2nResult", "Z2nResults", "z2n", "compute_z2n", "find_z2n_peaks", "plot_z2n",
    "z2n_single_pvalue", "global_pvalue_from_trials", "estimate_z2n_trials",
    "z2n_local_threshold", "z2n_global_threshold",
    "MCMCResult", "MCMCResults", "run_mcmc",
    "LightCurveSimulationResult", "PeakSignificance", "SignificanceResult",
    "SignificanceResults",
    "simulate_light_curves", "calculate_significance",
    "bayes_factor",
    "calculate_log_evidence", "run_nested_sampling",
    "fred_model", "ercod_model",
    "fred", "ercod",
    "add_mean_models", "constant_background",
    "constant_background_model", "linear_background",
    "linear_background_model", "polynomial_background",
    "polynomial_background_model",

]
