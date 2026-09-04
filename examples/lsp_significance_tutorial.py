"""End-to-end posterior-predictive LSP significance example.

Run from the BurstQPO repository root::

    python examples/lsp_significance_tutorial.py

This tutorial intentionally builds a small H0 posterior directly so it can be
run after installation without spending time on a new MCMC fit. In a real
analysis, replace ``build_demo_posterior`` with the NPZ produced by
``run_mcmc`` for the FRED+red-noise (no-QPO) model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running this file directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from burstqpo import MCMCResult, lsp
from burstqpo.bayesian.gp import k_red_noise
from burstqpo.models import add_mean_models, constant_background, fred
from burstqpo.significance import calculate_significance, simulate_light_curves


def build_demo_posterior(output_path: Path) -> MCMCResult:
    """Create a compact synthetic FRED+red-noise H0 posterior NPZ."""
    rng = np.random.default_rng(20260903)
    time = np.linspace(-0.25, 0.55, 48)
    errors = np.full(time.size, 0.08)
    mean_function = add_mean_models(
        fred(parameter_indices=(0, 1, 2, 3, 4)),
        constant_background(parameter_index=5),
    )
    kernel_function = k_red_noise(parameter_indices=(6, 7))
    center = np.asarray([
        2.4, 0.04, 0.09, 0.20, 2.0, 0.35,
        np.log(0.035), np.log(3.5),
    ])
    samples = center + rng.normal(
        0.0,
        [0.12, 0.008, 0.006, 0.012, 0.08, 0.03, 0.08, 0.08],
        size=(240, center.size),
    )
    # Keep the constrained positive FRED scales and shape valid.
    samples[:, 2:4] = np.abs(samples[:, 2:4])
    samples[:, 4] = np.maximum(samples[:, 4], 0.3)
    counts = mean_function(time, center)
    covariance = kernel_function(time, center) + np.diag(errors ** 2)
    counts = rng.multivariate_normal(counts, covariance)
    log_likelihood = np.zeros(samples.shape[0], dtype=float)
    log_likelihood[0] = 1.0  # deterministic maximum-likelihood reference sample
    result = MCMCResult(
        samples=samples,
        log_probability=log_likelihood.copy(),
        acceptance_fraction=np.asarray([0.4, 0.45, 0.5, 0.55]),
        autocorrelation_time=np.full(center.size, np.nan),
        model_id="tutorial_fred_red_noise_h0",
        seed=20260903,
        parameter_names=(
            "amplitude", "t_max", "sigma_rise", "sigma_decay", "nu",
            "background", "log_a_red", "log_c_red",
        ),
        model_metadata={
            "mean_function": mean_function.metadata,
            "kernel_function": kernel_function.metadata,
        },
        log_likelihood_values=log_likelihood,
        n_observations=time.size,
        time=time,
        counts=counts,
        errors=errors,
    )
    result.save_npz(output_path)
    return result


def run_tutorial(output_dir=None):
    output_dir = Path(output_dir or Path(__file__).resolve().parent / "significance_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = output_dir / "fred_red_noise_h0_posterior.npz"
    simulated_path = output_dir / "fred_red_noise_h0_simulated_lc.npz"
    significance_path = output_dir / "fred_red_noise_h0_lsp_significance.npz"

    posterior = build_demo_posterior(posterior_path)
    # Optional standalone posterior-predictive matrix (100 curves x 48 bins).
    simulations = simulate_light_curves(
        posterior,
        n_simulations=100,
        residual=True,
        random_state=42,
        save_path=simulated_path,
    )

    frequencies = np.arange(2.0, 25.0 + 0.05, 0.05)
    observed_lsp = lsp(
        posterior.time,
        posterior.counts - simulations.reference_mean,
        uncertainty=posterior.errors,
        frequencies=frequencies,
        mode="auto",
    )
    peak_index = int(np.nanargmax(observed_lsp.power))

    significance = calculate_significance(
        posterior,
        method=lsp,
        method_kwargs={
            "frequencies": frequencies,
            "mode": "auto",
        },
        n_simulations=1000,
        residual=True,
        random_state=42,
        progress=True,
        save_path=significance_path,
    )
    peak = significance.at(index=peak_index)
    print("\nPosterior:", posterior_path)
    print("Standalone simulated curves:", simulated_path)
    print("Significance result:", significance_path)
    print(f"Peak frequency: {peak.coordinate:.4f}")
    print(f"Observed LSP power: {peak.observed_statistic:.6g}")
    print(f"Local p-value / sigma: {peak.local_pvalue:.6g} / "
          f"{peak.local_significance_sigma:.3f}")
    print(f"Global p-value / sigma: {peak.global_pvalue:.6g} / "
          f"{peak.global_significance_sigma:.3f}")
    print("Local 1/2/3 sigma thresholds at peak:", peak.local_thresholds)
    print("Global 1/2/3 sigma thresholds:", peak.global_thresholds)
    return posterior, simulations, significance


if __name__ == "__main__":
    run_tutorial()
