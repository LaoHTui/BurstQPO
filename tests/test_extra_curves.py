import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from burstqpo import (
    DCFResult, JurkevichResult, LombScargleResult, WWZResult, Z2nResult,
)


@pytest.fixture
def grids():
    return np.linspace(1.0, 2.0, 5)


def _results(x):
    return [
        (
            LombScargleResult(x, x, x, x, "lsp"),
            "plot_lomb_scargle",
        ),
        (
            JurkevichResult(x, x, x, x, 5),
            "plot_jurkevich",
        ),
        (
            DCFResult(x, x, x, x, np.zeros_like(x), np.ones_like(x, dtype=int), 1.0, 0.0, 2.0),
            "plot_dcf",
        ),
        (
            Z2nResult(x, x, x, 2, 0.0),
            "plot_z2n",
        ),
    ]


@pytest.mark.parametrize("result_index", range(4))
def test_plot_methods_add_multiple_styled_curves_and_legend(grids, result_index):
    result, method_name = _results(grids)[result_index]
    curves = [
        {"values": grids + 1, "label": "1 sigma", "color": "tab:red",
         "linestyle": "--", "linewidth": 0.8},
        {"values": grids + 2, "color": "tab:blue"},
    ]

    figure, axis = getattr(result, method_name)(extra_curves=curves)

    first_extra, second_extra = axis.lines[-2:]
    np.testing.assert_allclose(first_extra.get_ydata(), grids + 1)
    np.testing.assert_allclose(second_extra.get_ydata(), grids + 2)
    assert first_extra.get_label() == "1 sigma"
    assert first_extra.get_color() == "tab:red"
    assert first_extra.get_linestyle() == "--"
    assert first_extra.get_linewidth() == pytest.approx(0.8)
    assert [text.get_text() for text in axis.get_legend().get_texts()][-1] == "1 sigma"
    assert not axis.get_legend().get_frame_on()
    figure.clear()


def test_wwz_adds_curve_to_frequency_projection(grids):
    tau = np.linspace(0.0, 1.0, 4)
    result = WWZResult(
        tau, grids, np.ones((len(tau), len(grids))), np.empty((0, len(grids))),
        1.0, np.ones((len(tau), len(grids))), np.ones((len(tau), len(grids))),
    )

    threshold = grids + 10
    figure, (axis, projection_axis) = result.plot_wwz(
        show_coi=False,
        extra_curves={"values": threshold, "label": "1 sigma", "color": "tab:red"},
    )

    assert len(axis.lines) == 0
    np.testing.assert_allclose(projection_axis.lines[-1].get_xdata(), threshold)
    np.testing.assert_allclose(projection_axis.lines[-1].get_ydata(), result.frequency)
    assert projection_axis.get_legend().get_texts()[-1].get_text() == "1 sigma"
    assert not projection_axis.get_legend().get_frame_on()
    figure.clear()


def test_wwz_extra_curve_must_match_frequency_length(grids):
    result = WWZResult(
        grids, grids, np.ones((len(grids), len(grids))), np.empty((0, len(grids))),
        1.0, np.ones((len(grids), len(grids))), np.ones((len(grids), len(grids))),
    )

    with pytest.raises(ValueError, match="same length as frequency"):
        result.plot_wwz(extra_curves={"values": grids[:-1]})


def test_wwz_extra_curve_requires_projection_panel(grids):
    result = WWZResult(
        grids, grids, np.ones((len(grids), len(grids))), np.empty((0, len(grids))),
        1.0, np.ones((len(grids), len(grids))), np.ones((len(grids), len(grids))),
    )

    with pytest.raises(ValueError, match="require the projection panel"):
        result.plot_wwz(show_projection=False, extra_curves={"values": grids})


def test_extra_curve_without_label_is_not_added_to_legend(grids):
    result, method_name = _results(grids)[0]

    figure, axis = getattr(result, method_name)(
        extra_curves={"values": grids, "color": "tab:green"},
    )

    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Lomb-Scargle power",
    ]
    assert not axis.get_legend().get_frame_on()
    figure.clear()


@pytest.mark.parametrize("result_index", range(4))
def test_plot_methods_broadcast_scalar_extra_curve(grids, result_index):
    result, method_name = _results(grids)[result_index]

    figure, axis = getattr(result, method_name)(
        extra_curves={"values": 1.5, "label": "threshold", "color": "tab:purple"},
    )

    curve = axis.lines[-1]
    np.testing.assert_allclose(curve.get_ydata(), 1.5)
    assert curve.get_label() == "threshold"
    figure.clear()


def test_wwz_broadcasts_scalar_extra_curve_on_frequency_projection(grids):
    result = WWZResult(
        grids, grids, np.ones((len(grids), len(grids))), np.empty((0, len(grids))),
        1.0, np.ones((len(grids), len(grids))), np.ones((len(grids), len(grids))),
    )

    figure, (_, projection_axis) = result.plot_wwz(
        extra_curves={"values": 2.0, "label": "2 sigma", "color": "tab:orange"},
    )

    curve = projection_axis.lines[-1]
    np.testing.assert_allclose(curve.get_xdata(), 2.0)
    np.testing.assert_allclose(curve.get_ydata(), result.frequency)
    figure.clear()


@pytest.mark.parametrize(
    ("extra_curves", "message"),
    [
        ({"color": "red"}, "must contain 'values'"),
        ({"values": [[1, 2], [3, 4]]}, "must be one-dimensional"),
        ({"values": [1, 2]}, "same length as the x-axis"),
        ([{"values": [1, 2, 3, 4, 5]}, 1], "must be a mapping"),
    ],
)
def test_extra_curves_are_validated(grids, extra_curves, message):
    result, method_name = _results(grids)[0]

    with pytest.raises((TypeError, ValueError), match=message):
        getattr(result, method_name)(extra_curves=extra_curves)
