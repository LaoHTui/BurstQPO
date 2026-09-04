import numpy as np
import pytest

from burstqpo import (Z2nResult, estimate_z2n_trials,
                      global_pvalue_from_trials, z2n,
                      z2n_global_threshold, z2n_local_threshold,
                      z2n_single_pvalue)


def test_single_frequency_pvalue_matches_chi_square_two_dof():
    statistic = np.array([0.0, 2.0, 10.0])

    np.testing.assert_allclose(
        z2n_single_pvalue(statistic, n_harmonics=1),
        np.exp(-statistic / 2.0),
    )


def test_global_pvalue_is_stable_at_probability_boundaries():
    assert global_pvalue_from_trials(0.0, 100) == 0.0
    assert global_pvalue_from_trials(1.0, 100) == 1.0
    assert global_pvalue_from_trials(0.01, 10) == pytest.approx(1.0 - 0.99**10)


def test_trial_estimate_accounts_for_frequency_oversampling():
    duration = 10.0
    oversample_factor = 8.0
    frequency_step = 1.0 / (duration * oversample_factor)
    frequencies = 0.1 + frequency_step * np.arange(800)

    n_trials = estimate_z2n_trials([0.0, duration], frequencies)

    assert n_trials == pytest.approx(len(frequencies) / oversample_factor)


def test_z2n_returns_local_and_global_pvalues_and_round_trips(tmp_path):
    rng = np.random.default_rng(4)
    events = np.sort(rng.uniform(0.0, 20.0, 500))
    frequencies = np.linspace(0.2, 2.0, 300)

    result = z2n(events, frequencies=frequencies, n_harmonics=2)

    assert result.local_pvalue.shape == frequencies.shape
    assert result.global_pvalue().shape == frequencies.shape
    assert 1.0 <= result.n_trials < len(frequencies)
    assert np.all(result.global_pvalue() >= result.local_pvalue)
    peak = result.find_peaks(top_n=1, prominence=0.0)[0]
    assert peak["local_pvalue"] == result.local_pvalue[peak["index"]]
    assert peak["global_pvalue"] == result.global_pvalue()[peak["index"]]

    restored = Z2nResult.load_npz(result.save_npz(tmp_path / "z2n.npz"))
    np.testing.assert_allclose(restored.local_pvalue, result.local_pvalue)
    np.testing.assert_allclose(restored.global_pvalue(), result.global_pvalue())
    assert restored.n_trials == result.n_trials


def test_trial_count_can_be_overridden_only_for_significance():
    result = z2n([0.0, 0.2, 0.7, 1.0], frequencies=[1.0, 2.0])

    np.testing.assert_allclose(
        result.global_pvalue(n_trials=25),
        global_pvalue_from_trials(result.local_pvalue, 25),
    )


def test_local_and_global_z_thresholds_accept_percent_strings():
    local = z2n_local_threshold("0.1%", n_harmonics=2)
    global_ = z2n_global_threshold("0.1%", n_harmonics=2, n_trials=100)

    assert local == pytest.approx(18.4668269529)
    assert global_ > local
    assert global_pvalue_from_trials(z2n_single_pvalue(global_, 2), 100) == pytest.approx(0.001)

    result = z2n([0.0, 0.2, 0.7, 1.0], frequencies=[1.0, 2.0], n_harmonics=2)
    assert result.z_local_threshold("0.1%") == pytest.approx(local)
    assert result.z_global_threshold("0.1%", n_trials=100) == pytest.approx(global_)
