"""Integration test for the executable GP-MCMC tutorial."""

import numpy as np

from examples.gp_mcmc_tutorial import PARAMETER_NAMES, run_tutorial


def test_real_fred_red_noise_qpo_gp_mcmc_workflow(tmp_path):
    result, restored, output = run_tutorial(tmp_path)

    # This short run tests plumbing, not statistical convergence.
    assert result.posterior.shape == (360, 11)
    assert np.all(np.isfinite(result.log_probability))
    assert result.parameter_names == PARAMETER_NAMES
    np.testing.assert_array_equal(
        result.parameter("log_f_qpo"), result.posterior[:, 10]
    )
    assert result.prior["type"] == "prior"
    assert result.models["mean_function"]["name"] == "sum"
    mean_terms = result.models["mean_function"]["terms"]
    assert [term["name"] for term in mean_terms] == [
        "fred", "constant_background"
    ]
    assert result.models["kernel_function"]["name"] == "sum"

    np.testing.assert_allclose(restored.posterior, result.posterior)
    assert restored.parameter_names == PARAMETER_NAMES
    assert restored.sampler_config["n_walkers"] == 24
    assert restored.log_prior_values.shape == (360,)
    assert restored.log_likelihood_values.shape == (360,)
    assert np.isfinite(restored.maximum_log_likelihood)
    assert set(restored.qpo_summary()) >= {
        "frequency", "period", "damping_time", "quality_factor"
    }
    figure, axes = restored.plot_corner(
        parameters=("log_a_qpo", "log_c_qpo", "log_f_qpo"),
        save_path=tmp_path / "qpo_corner.png",
    )
    assert axes.shape == (3, 3)
    assert (tmp_path / "qpo_corner.png").exists()
    import matplotlib.pyplot as plt
    plt.close(figure)
    assert output.exists()
