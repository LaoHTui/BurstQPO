import numpy as np
import pytest

from burstqpo import (add_mean_models, constant_background,
                      constant_background_model, ercod, ercod_model, fred, fred_model,
                      linear_background, linear_background_model,
                      polynomial_background, polynomial_background_model)


def test_fred_model_uses_separate_rise_and_decay_scales():
    time = np.array([-1.0, 0.0, 1.0, 2.0])
    theta = (10.0, 0.0, 1.0, 2.0, 2.0)

    result = fred_model(time, theta)
    expected = np.array([
        10.0 * np.exp(-1.0),
        10.0,
        10.0 * np.exp(-0.25),
        10.0 * np.exp(-1.0),
    ])

    np.testing.assert_allclose(result, expected)


def test_ercod_model_is_continuous_at_peak_and_zeroes_pulse_after_cutoff():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    theta = (10.0, 1.0, 0.5, 1.0, 2.0, 2.0)

    result = ercod_model(time, theta)

    assert result[1] == 10.0
    assert result[2] == 0.0
    assert result[3] == 0.0
    np.testing.assert_allclose(
        ercod_model([1.0 - 1e-8], theta),
        ercod_model([1.0 + 1e-8], theta),
        rtol=1e-6,
        atol=1e-6,
    )


def test_add_mean_models_sums_ercod_and_constant_background():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    theta = np.array([10.0, 1.0, 0.5, 1.0, 2.0, 2.0, 3.0])
    pulse = ercod(parameter_indices=(0, 1, 2, 3, 4, 5))
    background = constant_background(parameter_index=6)
    combined = add_mean_models(pulse, background)

    np.testing.assert_allclose(
        combined(time, theta),
        ercod_model(time, theta[:6]) + 3.0,
    )
    assert [term["name"] for term in combined.metadata["terms"]] == [
        "ercod", "constant_background"
    ]


def test_constant_linear_and_polynomial_background_models():
    time = np.array([9.0, 10.0, 11.0])

    np.testing.assert_allclose(
        constant_background_model(time, [2.5]), [2.5, 2.5, 2.5]
    )
    np.testing.assert_allclose(
        linear_background_model(time, [2.0, 0.5], time_origin=10.0),
        [1.5, 2.0, 2.5],
    )
    np.testing.assert_allclose(
        polynomial_background_model(
            time, [1.0, 2.0, 3.0], time_origin=10.0
        ),
        [2.0, 1.0, 6.0],
    )


def test_add_mean_models_sums_fred_and_constant_background():
    time = np.array([-1.0, 0.0, 1.0])
    theta = np.array([10.0, 0.0, 1.0, 2.0, 2.0, 3.0])
    pulse = fred(parameter_indices=(0, 1, 2, 3, 4))
    background = constant_background(parameter_index=5)
    combined = add_mean_models(pulse, background)

    np.testing.assert_allclose(
        combined(time, theta),
        fred_model(time, theta[:5]) + 3.0,
    )
    assert combined.metadata["name"] == "sum"
    assert [term["name"] for term in combined.metadata["terms"]] == [
        "fred", "constant_background"
    ]


def test_background_factories_select_global_parameters():
    time = np.array([1.0, 2.0, 3.0])
    theta = np.array([99.0, 4.0, 0.5, 2.0, -1.0, 0.25])

    np.testing.assert_allclose(
        linear_background((1, 2), time_origin=2.0)(time, theta),
        [3.5, 4.0, 4.5],
    )
    np.testing.assert_allclose(
        polynomial_background((3, 4, 5), time_origin=2.0)(time, theta),
        [3.25, 2.0, 1.25],
    )


def test_add_mean_models_rejects_invalid_components_and_shapes():
    with pytest.raises(ValueError, match="at least one"):
        add_mean_models()
    with pytest.raises(TypeError, match="callable"):
        add_mean_models(None)

    combined = add_mean_models(lambda time, theta: np.ones(2))
    with pytest.raises(ValueError, match="matching time"):
        combined(np.arange(3.0), np.array([]))
