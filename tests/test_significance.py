from types import SimpleNamespace

import numpy as np
import pytest

from burstqpo import MCMCResult, dcf, jurkevich, lomb_scargle, wwz
from burstqpo.bayesian.gp import k_red_noise
from burstqpo.models import add_mean_models, constant_background, fred
from burstqpo.significance import (
    LightCurveSimulationResult,
    SignificanceResult,
    calculate_significance,
    simulate_light_curves,
)


def _h0_result():
    time = np.linspace(-0.2, 0.5, 16)
    errors = np.full(time.size, 0.05)
    samples = np.asarray([
        [2.0, 0.0, 0.08, 0.18, 2.0, 0.4, np.log(0.02), np.log(3.0)],
        [2.2, 0.01, 0.09, 0.20, 2.1, 0.5, np.log(0.03), np.log(2.5)],
        [1.8, -0.01, 0.07, 0.17, 1.9, 0.3, np.log(0.01), np.log(4.0)],
    ])
    mean = add_mean_models(fred((0, 1, 2, 3, 4)), constant_background(5))
    counts = mean(time, samples[1]) + 0.02 * np.sin(2 * np.pi * 6 * time)
    return MCMCResult(
        samples=samples,
        log_probability=np.asarray([-3.0, -1.0, -2.0]),
        acceptance_fraction=np.asarray([0.4, 0.5]),
        autocorrelation_time=np.full(samples.shape[1], np.nan),
        parameter_names=tuple(f"theta_{i}" for i in range(samples.shape[1])),
        model_metadata={
            "mean_function": mean.metadata,
            "kernel_function": k_red_noise((6, 7)).metadata,
        },
        log_likelihood_values=np.asarray([-2.0, -0.5, -1.0]),
        n_observations=time.size,
        time=time,
        counts=counts,
        errors=errors,
    )


def test_mcmc_observations_round_trip(tmp_path):
    result = _h0_result()
    restored = MCMCResult.load_npz(result.save_npz(tmp_path / "posterior"))

    np.testing.assert_array_equal(restored.time, result.time)
    np.testing.assert_array_equal(restored.counts, result.counts)
    np.testing.assert_array_equal(restored.errors, result.errors)
    assert restored.metadata["has_observations"] is True


def test_simulated_light_curves_are_reproducible_and_round_trip(tmp_path):
    result = _h0_result()
    first = simulate_light_curves(
        result, 5, residual=True, random_state=12,
        save_path=tmp_path / "simulations",
    )
    second = simulate_light_curves(result, 5, residual=True, random_state=12)

    np.testing.assert_allclose(first.values, second.values)
    np.testing.assert_array_equal(first.sample_indices, second.sample_indices)
    np.testing.assert_allclose(
        first.light_curves,
        first.residuals + first.reference_mean[None, :],
    )
    assert first.values.shape == (5, result.time.size)

    restored = LightCurveSimulationResult.load_npz(tmp_path / "simulations.npz")
    np.testing.assert_allclose(restored.values, first.values)
    np.testing.assert_allclose(restored.light_curves, first.light_curves)
    assert restored.residual is True


def test_residual_baseline_uses_maximum_likelihood_sample():
    result = _h0_result()
    mean = add_mean_models(fred((0, 1, 2, 3, 4)), constant_background(5))
    simulations = simulate_light_curves(result, 2, residual=True, random_state=4)

    np.testing.assert_allclose(
        simulations.reference_mean,
        mean(result.time, result.samples[1]),
    )


def test_calculate_lsp_significance_and_query_round_trip(tmp_path):
    result = _h0_result()
    significance = calculate_significance(
        result,
        method=lomb_scargle,
        n_simulations=12,
        method_kwargs={"frequencies": np.linspace(2.0, 10.0, 9)},
        random_state=21,
        progress=False,
        save_path=tmp_path / "significance",
    )

    assert significance.simulated_statistic.shape == (12, 9)
    assert significance.local_sigma1.shape == (9,)
    assert significance.global_sigma3.shape == (9,)
    assert np.all(significance.global_pvalue >= significance.local_pvalue)
    peak = significance.at(index=int(np.nanargmax(significance.observed_statistic)))
    same_peak = significance.at(coordinate=peak.coordinate)
    assert same_peak == peak
    assert peak.global_pvalue >= peak.local_pvalue

    restored = SignificanceResult.load_npz(tmp_path / "significance.npz")
    np.testing.assert_allclose(restored.local_pvalue, significance.local_pvalue)
    np.testing.assert_allclose(restored.global_pvalue, significance.global_pvalue)
    np.testing.assert_allclose(restored.local_sigma2, significance.local_sigma2)
    assert restored.at(index=peak.index).global_pvalue == peak.global_pvalue


def test_custom_statistic_extractor_is_supported():
    result = _h0_result()

    def method(time, values, coordinates):
        return SimpleNamespace(x=np.asarray(coordinates), y=np.asarray(coordinates) * np.var(values))

    significance = calculate_significance(
        result,
        method=method,
        method_kwargs={"coordinates": [1.0, 2.0, 3.0]},
        statistic_extractor=lambda output: (output.x, output.y),
        n_simulations=5,
        random_state=2,
        progress=False,
    )

    np.testing.assert_array_equal(significance.coordinate, [1.0, 2.0, 3.0])
    assert significance.simulated_statistic.shape == (5, 3)


@pytest.mark.parametrize(
    ("method", "method_kwargs", "expected_size"),
    [
        (wwz, {"frequencies": [2.0, 4.0], "tau_number": 8, "c": 0.1}, 2),
        (jurkevich, {"periods": [0.15, 0.25], "m": 3}, 2),
        (dcf, {"delta_tau": 0.1, "c": 0.12, "max_tau": 0.3}, 4),
    ],
)
def test_builtin_method_adapters(method, method_kwargs, expected_size):
    significance = calculate_significance(
        _h0_result(), method, 2, method_kwargs=method_kwargs,
        random_state=5, progress=False,
    )

    assert significance.coordinate.size == expected_size
    assert significance.simulated_statistic.shape == (2, expected_size)


def test_search_range_limits_global_trials():
    result = _h0_result()
    significance = calculate_significance(
        result, lomb_scargle, 3,
        method_kwargs={"frequencies": [2.0, 3.0, 4.0]},
        search_range=(2.5, 3.5), random_state=3, progress=False,
    )

    assert np.isnan(significance.global_pvalue[[0, 2]]).all()
    assert np.isfinite(significance.global_pvalue[1])


def test_qpo_posterior_is_rejected_for_significance():
    result = _h0_result()
    result = MCMCResult(
        **{
            **result.__dict__,
            "model_metadata": {
                **result.model_metadata,
                "kernel_function": {
                    "name": "sum",
                    "terms": [
                        result.model_metadata["kernel_function"],
                        {"name": "qpo", "parameter_indices": [0, 1, 2]},
                    ],
                },
            },
        }
    )

    with pytest.raises(ValueError, match="null-model posterior"):
        calculate_significance(
            result, lomb_scargle, 2,
            method_kwargs={"frequencies": [2.0, 3.0]}, progress=False,
        )


def test_reference_significance_npz_field_names_are_loaded(tmp_path):
    path = tmp_path / "reference.npz"
    coordinate = np.asarray([1.0, 2.0])
    simulated = np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    observed = np.asarray([2.5, 3.5])
    np.savez_compressed(
        path,
        frequency=coordinate,
        observed_statistic=observed,
        simulated_statistic=simulated,
        simulated_best=np.max(simulated, axis=1),
        sigma_levels=np.asarray([1.0, 2.0, 3.0]),
        local_sigma_thresholds=np.tile(coordinate, (3, 1)),
        global_sigma_thresholds=np.asarray([2.0, 3.0, 4.0]),
        local_p=np.asarray(0.5),
        global_p=np.asarray(0.75),
    )

    restored = SignificanceResult.load_npz(path)
    np.testing.assert_array_equal(restored.coordinate, coordinate)
    assert restored.local_pvalue.shape == coordinate.shape
    assert restored.global_pvalue.shape == coordinate.shape
    assert np.all(np.isfinite(restored.global_pvalue))


def test_at_rejects_out_of_range_coordinate():
    result = _h0_result()
    significance = calculate_significance(
        result, lomb_scargle, 2,
        method_kwargs={"frequencies": [2.0, 3.0]},
        random_state=1, progress=False,
    )
    with pytest.raises(ValueError, match="outside"):
        significance.at(coordinate=5.0)
