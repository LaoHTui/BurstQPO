"""Reusable QPO period-search algorithms for plain time-series arrays."""

from .algorithms.wwz import compute_wwz, find_wwz_peaks, project_wwz, wwz
from .algorithms.lomb_scargle import compute_lomb_scargle, compute_lsp, find_lomb_scargle_peaks, lomb_scargle, lsp
from .algorithms.jurkevich import compute_jurkevich, find_jurkevich_peaks, jurkevich
from .algorithms.dcf import compute_dcf, dcf, find_dcf_peaks
from .algorithms.z2n import compute_z2n, find_z2n_peaks, z2n
from .results import (JurkevichResult, JurkevichResults, LombScargleResult,
                      LombScargleResults, DCFResult, DCFResults, Z2nResult,
                      Z2nResults, WWZResult, WWZResults)
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

]
