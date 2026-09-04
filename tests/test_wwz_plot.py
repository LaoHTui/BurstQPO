import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from burstqpo import WWZResult, wwz


def _result():
    time = np.linspace(10.0, 12.0, 30)
    values = np.sin(2 * np.pi * time)
    uncertainty = np.linspace(0.1, 0.2, len(time))
    return wwz(
        time, values, frequencies=np.linspace(0.5, 2.0, 8),
        uncertainty=uncertainty, tau_number=12,
    )


def test_wwz_plot_can_add_light_curve_panel():
    result = _result()

    figure, axes = result.plot_wwz(show_light_curve=True)
    figure.canvas.draw()

    assert len(figure.axes) == 4
    assert len(axes) == 2
    assert figure.axes[0].get_ylabel() == "Value"
    assert figure.axes[0].get_shared_x_axes().joined(figure.axes[0], axes[0])
    assert len(figure.axes[0].lines) == 1
    assert len(figure.axes[0].collections) > 0
    assert all(spine.get_visible() for spine in figure.axes[0].spines.values())
    assert all(spine.get_visible() for spine in axes[0].spines.values())
    assert any(tick.tick2line.get_visible() for tick in axes[0].xaxis.majorTicks)
    assert figure.axes[0].get_position().y0 == pytest.approx(axes[0].get_position().y1)
    assert min(figure.axes[0].get_yticks()) > figure.axes[0].get_ylim()[0]
    assert max(axes[0].get_yticks()) < axes[0].get_ylim()[1]
    assert min(axes[1].get_xticks()) > axes[1].get_xlim()[0]
    assert axes[1].get_position().width < axes[0].get_position().width / 4


def test_wwz_light_curve_round_trips_through_npz(tmp_path):
    result = _result()

    restored = WWZResult.load_npz(result.save_npz(tmp_path / "result.npz"))

    np.testing.assert_allclose(restored.time, result.time)
    np.testing.assert_allclose(restored.values, result.values)
    np.testing.assert_allclose(restored.uncertainty, result.uncertainty)
    restored.plot_wwz(show_light_curve=True)


def test_wwz_tau_and_plot_keep_the_input_time_coordinates():
    time = np.linspace(-0.1, 0.1, 30)
    result = wwz(
        time, np.sin(2 * np.pi * 10 * time),
        frequencies=np.linspace(5.0, 15.0, 8), tau_number=12,
    )

    np.testing.assert_allclose(result.tau[[0, -1]], [-0.1, 0.1])
    np.testing.assert_allclose(result.time, time)
    assert np.all(result.coi >= time.min())
    assert np.all(result.coi <= time.max())

    figure, (axis, _) = result.plot_wwz(show_coi=False, show_projection=False)
    np.testing.assert_allclose(axis.get_xlim(), [-0.1, 0.1])
    figure.clear()


def test_wwz_rejects_misaligned_uncertainty():
    with pytest.raises(ValueError, match="same length"):
        wwz(np.arange(4.0), np.arange(4.0), frequencies=[1.0], uncertainty=[0.1])


def test_wwz_plot_can_mask_coi_and_scale_colors_from_inside_power():
    tau = np.arange(4.0)
    frequency = np.array([1.0, 2.0])
    power = np.array([
        [1000.0, 2000.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [3000.0, 4000.0],
    ])
    result = WWZResult(
        tau, frequency, power, np.array([[1.0, 1.0], [2.0, 2.0]]),
        0.0, np.zeros_like(power), np.ones_like(power),
    )

    figure, (axis, _) = result.plot_wwz(
        mask_coi=True, show_coi=False, show_projection=False,
    )
    image = axis.collections[0]

    assert image.get_clim() == pytest.approx((1.0, 4.0))
    assert np.ma.count_masked(image.get_array()) == 4
    figure.clear()


def test_wwz_plot_keeps_full_color_range_when_coi_mask_is_disabled():
    result = WWZResult(
        np.arange(3.0), np.array([1.0]), np.array([[1000.0], [2.0], [2000.0]]),
        np.array([[1.0], [1.0]]), 0.0, np.zeros((3, 1)), np.ones((3, 1)),
    )

    figure, (axis, _) = result.plot_wwz(
        mask_coi=False, show_coi=False, show_projection=False,
    )

    assert axis.collections[0].get_clim() == pytest.approx((2.0, 2000.0))
    figure.clear()
