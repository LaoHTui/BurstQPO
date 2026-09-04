"""Numerical algorithms."""

from .wwz import compute_wwz, find_wwz_peaks, project_wwz, wwz
from .lomb_scargle import compute_lomb_scargle, compute_lsp, find_lomb_scargle_peaks, lomb_scargle, lsp
from .jurkevich import compute_jurkevich, find_jurkevich_peaks, jurkevich
from .dcf import compute_dcf, dcf, find_dcf_peaks
from .z2n import (compute_z2n, estimate_z2n_trials, find_z2n_peaks,
                  global_pvalue_from_trials, z2n, z2n_global_threshold,
                  z2n_local_threshold, z2n_single_pvalue)


__all__ = [
    "wwz", "compute_wwz", "find_wwz_peaks", "project_wwz",
    "lomb_scargle", "lsp", "compute_lomb_scargle", "compute_lsp", "find_lomb_scargle_peaks",
    "jurkevich", "compute_jurkevich", "find_jurkevich_peaks",
    "dcf", "compute_dcf", "find_dcf_peaks",
    "z2n", "compute_z2n", "find_z2n_peaks", "z2n_single_pvalue",
    "global_pvalue_from_trials", "estimate_z2n_trials", "z2n_local_threshold",
    "z2n_global_threshold",
]
