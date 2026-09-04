import sys
import types

import numpy as np
import pytest
import matplotlib.pyplot as plt

from burstqpo import MCMCResult
from burstqpo.bayesian.gp import add_kernels, k_qpo, k_red_noise
from burstqpo.mcmc import run_mcmc


def _result():
    return MCMCResult(
        np.arange(6.0).reshape(3, 2),
        np.array([-1.0, -0.5, -0.25]),
        np.array([0.4, 0.5, 0.6, 0.7]),
        np.array([1.2, 1.3]),
        "test-model",
        12,
        parameter_names=("alpha", "beta"),
        log_prior_values=np.array([-0.5, -0.4, -0.3]),
        log_likelihood_values=np.array([-0.5, -0.1, 0.05]),
        n_observations=10,
    )


def test_mcmc_result_exposes_posterior_and_diagnostics():
    result = _result()

    np.testing.assert_array_equal(result.posterior, result.samples)
    np.testing.assert_array_equal(result.posterior_samples, result.samples)
    np.testing.assert_array_equal(result.log_prob, result.log_probability)
    np.testing.assert_array_equal(result.acceptance, result.acceptance_fraction)
    np.testing.assert_array_equal(result.autocorr_time, result.autocorrelation_time)
    assert result.mean_acceptance_fraction == pytest.approx(0.55)
    assert result.ndim == 2
    assert result.n_samples == 3
    assert result.posterior_means == {"alpha": 2.0, "beta": 3.0}
    assert result.maximum_log_posterior == pytest.approx(-0.25)
    assert result.maximum_log_likelihood == pytest.approx(0.05)
    np.testing.assert_array_equal(result.maximum_likelihood_sample, [4.0, 5.0])
    assert result.aic == pytest.approx(3.9)
    assert result.bic == pytest.approx(2 * np.log(10) - 0.1)
    assert list(result.posterior_1sigma) == ["alpha", "beta"]
    assert len(result.posterior_1sigma_values) == 2


def test_mcmc_result_round_trips_through_npz(tmp_path):
    result = _result()

    path = result.save_npz(tmp_path / "nested" / "chain")
    restored = MCMCResult.load_npz(path)

    assert path.name == "chain.npz"
    np.testing.assert_array_equal(restored.samples, result.samples)
    np.testing.assert_array_equal(restored.log_probability, result.log_probability)
    np.testing.assert_array_equal(restored.acceptance_fraction, result.acceptance_fraction)
    np.testing.assert_array_equal(restored.autocorrelation_time, result.autocorrelation_time)
    np.testing.assert_array_equal(
        restored.log_likelihood_values, result.log_likelihood_values
    )
    assert restored.n_observations == 10
    assert restored.model_id == result.model_id
    assert restored.seed == result.seed


def test_mcmc_result_loads_legacy_npz_without_observations(tmp_path):
    result = _result()
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        samples=result.samples,
        log_probability=result.log_probability,
        acceptance_fraction=result.acceptance_fraction,
        autocorrelation_time=result.autocorrelation_time,
    )

    restored = MCMCResult.load_npz(path)

    assert restored.time.size == 0
    assert restored.counts.size == 0
    assert restored.errors.size == 0


def test_mcmc_result_load_rejects_incomplete_npz(tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez(path, samples=np.zeros((1, 1)))

    with pytest.raises(ValueError, match="missing MCMC fields"):
        MCMCResult.load_npz(path)


def test_mcmc_result_corner_plot_and_qpo_summary(tmp_path):
    samples = np.array([
        [0.0, np.log(2.0), np.log(4.0)],
        [np.log(2.0), np.log(1.0), np.log(8.0)],
        [np.log(4.0), np.log(0.5), np.log(16.0)],
    ])
    result = MCMCResult(
        samples=samples,
        log_probability=np.array([-3.0, -2.0, -1.0]),
        acceptance_fraction=np.array([0.4, 0.5]),
        autocorrelation_time=np.full(3, np.nan),
        parameter_names=("log_a_qpo", "log_c_qpo", "log_f_qpo"),
        model_metadata={
            "kernel_function": {
                "name": "qpo",
                "parameter_indices": [0, 1, 2],
            }
        },
    )

    qpo = result.qpo_posterior()
    np.testing.assert_allclose(qpo["frequency"], [4.0, 8.0, 16.0])
    np.testing.assert_allclose(qpo["period"], [0.25, 0.125, 0.0625])
    np.testing.assert_allclose(
        qpo["quality_factor"], np.pi * qpo["frequency"] / qpo["decay_rate"]
    )
    assert set(result.qpo_summary()) == {
        "amplitude", "decay_rate", "frequency", "period",
        "damping_time", "coherence_cycles", "quality_factor",
    }

    figure, axes = result.plot_corner(
        parameters=("log_a_qpo", "log_f_qpo"),
        truths={"log_f_qpo": np.log(8.0)},
        save_path=tmp_path / "corner.png",
    )
    assert axes.shape == (2, 2)
    assert (tmp_path / "corner.png").exists()
    plt.close(figure)


def test_mcmc_result_predicts_latent_qpo_component():
    result = MCMCResult(
        samples=np.tile(
            [1.0, 0.0, 0.0], (4, 1)
        ),
        log_probability=np.zeros(4),
        acceptance_fraction=np.array([0.5, 0.5]),
        autocorrelation_time=np.full(3, np.nan),
    )
    time = np.linspace(0.0, 1.0, 4)
    counts = np.zeros(4)
    errors = np.full(4, 0.1)
    red = k_red_noise((0, 1))
    qpo = k_qpo((0, 1, 2))
    total = add_kernels(red, qpo)
    prediction = result.predict_qpo(
        time,
        counts,
        errors,
        lambda t, theta: np.zeros_like(t),
        total,
        qpo,
    )
    assert prediction["time"].shape == time.shape
    assert prediction["mean"].shape == time.shape
    assert prediction["std"].shape == time.shape
    assert prediction["covariance"].shape == (time.size, time.size)
    assert np.all(np.isfinite(prediction["mean"]))
    assert np.all(prediction["std"] >= 0.0)


def test_run_mcmc_returns_result_and_saves_it(monkeypatch, tmp_path):
    class AutocorrError(Exception):
        pass

    class FakeSampler:
        acceptance_fraction = np.array([0.25, 0.5, 0.75, 1.0])

        def __init__(self, n_walkers, ndim, log_probability, blobs_dtype=None):
            self.n_walkers = n_walkers
            self.ndim = ndim
            self.log_probability = log_probability
            self.blobs_dtype = blobs_dtype

        def run_mcmc(self, walkers, n_steps, progress=False):
            self.walkers = walkers
            self.n_steps = n_steps

        def get_chain(self, discard, thin, flat):
            assert (discard, thin, flat) == (1, 1, True)
            return np.arange(6.0).reshape(3, 2)

        def get_log_prob(self, discard, thin, flat):
            assert (discard, thin, flat) == (1, 1, True)
            return np.array([-1.0, -0.5, -0.25])

        def get_blobs(self, discard, thin, flat):
            assert (discard, thin, flat) == (1, 1, True)
            return np.array(
                [(0.0, -1.0), (0.0, -0.5), (0.0, -0.25)],
                dtype=[("log_prior", float), ("log_likelihood", float)],
            )

        def get_autocorr_time(self, tol):
            raise AutocorrError()

    fake_emcee = types.ModuleType("emcee")
    fake_emcee.EnsembleSampler = FakeSampler
    fake_emcee.autocorr = types.SimpleNamespace(AutocorrError=AutocorrError)
    monkeypatch.setitem(sys.modules, "emcee", fake_emcee)

    output = tmp_path / "run" / "result"
    result = run_mcmc(
        [0.0, 1.0], [1.0, 2.0], [0.1, 0.1],
        prior=lambda theta: True,
        mcmc_config={
            "initial": [0.0, 0.0], "start_scale": [0.1, 0.1],
            "n_walkers": 4, "n_steps": 4, "burn_in": 1,
        },
        likelihood_function=lambda theta, time, counts, errors: -1.0,
        output_path=output,
        seed=3,
    )

    assert isinstance(result, MCMCResult)
    assert output.with_suffix(".npz").exists()
    assert result.mean_acceptance_fraction == pytest.approx(0.625)
    np.testing.assert_array_equal(
        MCMCResult.load_npz(output.with_suffix(".npz")).posterior,
        result.posterior,
    )
