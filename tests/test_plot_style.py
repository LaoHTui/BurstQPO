import matplotlib
import numpy as np

matplotlib.use("Agg")

from burstqpo import LombScargleResult, Z2nResult


def test_default_plot_uses_exact_range_full_ticks_font_and_frameless_legend():
    frequency = np.linspace(10.0, 60.0, 101)
    result = Z2nResult(frequency, frequency, np.sin(frequency) + 2.0, 1, 0.0)

    figure, axis = result.plot_z2n(
        extra_curves={"values": 3.0, "label": "0.1%",
                      "color": "red", "linestyle": "--"},
    )
    figure.canvas.draw()

    np.testing.assert_allclose(axis.get_xlim(), [frequency.min(), frequency.max()])
    assert all(spine.get_visible() for spine in axis.spines.values())
    assert any(tick.tick2line.get_visible() for tick in axis.xaxis.majorTicks)
    assert any(tick.tick2line.get_visible() for tick in axis.yaxis.majorTicks)
    assert len(axis.xaxis.get_minorticklocs()) > 0
    assert len(axis.yaxis.get_minorticklocs()) > 0
    assert axis.get_legend() is not None
    assert not axis.get_legend().get_frame_on()
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "$Z^2_{1}$ statistics", "0.1%",
    ]
    assert all(text.get_fontfamily()[0] == "Times New Roman"
               for text in figure.findobj(match=matplotlib.text.Text))
    figure.clear()


def test_peak_markers_are_opaque_red_dashed_lines(monkeypatch):
    frequency = np.linspace(1.0, 3.0, 5)
    result = LombScargleResult(frequency, frequency, frequency, frequency, "lsp")
    monkeypatch.setattr(
        LombScargleResult, "find_peaks",
        lambda self, **kwargs: [{"frequency": 2.0, "period": .5,
                                "power": 2.0, "index": 2}],
    )

    figure, axis = result.plot_lomb_scargle(show_peaks=True)
    peak_line = axis.lines[-1]

    assert peak_line.get_color() == "red"
    assert peak_line.get_linestyle() == "--"
    assert peak_line.get_alpha() is None
    figure.clear()


def test_peak_label_is_positioned_at_half_peak_height(monkeypatch):
    frequency = np.linspace(1.0, 3.0, 5)
    result = LombScargleResult(frequency, frequency, frequency, frequency, "lsp")
    monkeypatch.setattr(
        LombScargleResult, "find_peaks",
        lambda self, **kwargs: [{"frequency": 2.0, "period": .5,
                                "power": 4.0, "index": 2}],
    )

    figure, axis = result.plot_lomb_scargle(show_peaks=True)

    assert axis.texts[-1].xy == (2.0, 2.0)
    figure.clear()
