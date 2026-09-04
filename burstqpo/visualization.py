"""Optional Matplotlib plots for the five independent search methods."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

from .algorithms.wwz import project_wwz
from .results import (DCFResult, JurkevichResult, LombScargleResult,
                      MCMCResult, WWZResult, Z2nResult)


def _finish(figure, save_path, dpi, show):
    for text in figure.findobj(match=Text):
        text.set_fontfamily("Times New Roman")
        if hasattr(text, "set_math_fontfamily"):
            text.set_math_fontfamily("stix")
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def plot_mcmc_corner(result: MCMCResult, *, parameters=None, truths=None,
                     bins=30, max_points=5000, color="tab:blue",
                     save_path=None, dpi=200, show=False, figsize=None):
    """Plot marginal and pairwise posterior projections.

    ``parameters`` may contain parameter names or integer column indices.
    This implementation does not require the optional third-party ``corner``
    package.
    """
    if not isinstance(result, MCMCResult):
        raise TypeError("result must be an MCMCResult")
    if result.samples.ndim != 2 or not result.n_samples or not result.ndim:
        raise ValueError("result must contain a non-empty 2D posterior")

    labels = result._parameter_labels
    if parameters is None:
        indices = tuple(range(result.ndim))
    else:
        if isinstance(parameters, (str, int)):
            parameters = (parameters,)
        else:
            parameters = tuple(parameters)
        selected = []
        for parameter in parameters:
            if isinstance(parameter, str):
                try:
                    index = labels.index(parameter)
                except ValueError as exc:
                    raise KeyError(f"unknown parameter: {parameter}") from exc
            else:
                index = int(parameter)
                if not 0 <= index < result.ndim:
                    raise IndexError("parameter index is outside the posterior")
            selected.append(index)
        indices = tuple(selected)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("parameters must select unique posterior columns")

    selected_labels = tuple(labels[index] for index in indices)
    values = result.samples[:, indices]
    if max_points is not None and len(values) > int(max_points):
        if int(max_points) <= 0:
            raise ValueError("max_points must be positive or None")
        positions = np.linspace(
            0, len(values) - 1, int(max_points), dtype=int
        )
        scatter_values = values[positions]
    else:
        scatter_values = values

    if truths is None:
        truth_values = [None] * len(indices)
    elif isinstance(truths, Mapping):
        truth_values = [truths.get(label) for label in selected_labels]
    else:
        truth_values = list(truths)
        if len(truth_values) != len(indices):
            raise ValueError("truths must match the selected parameters")

    count = len(indices)
    if figsize is None:
        side = max(3.0, min(24.0, 2.1 * count))
        figsize = (side, side)
    figure, axes = plt.subplots(
        count, count, figsize=figsize, squeeze=False
    )
    summaries = result.posterior_1sigma

    for row in range(count):
        for column in range(count):
            axis = axes[row, column]
            if column > row:
                axis.set_visible(False)
                continue
            if row == column:
                axis.hist(
                    values[:, column], bins=bins, color=color, alpha=.75,
                    histtype="stepfilled",
                )
                center, lower, upper = summaries[selected_labels[column]]
                axis.set_title(
                    f"{center:.4g} {lower:.2g}/+{upper:.2g}", fontsize=9
                )
            else:
                axis.scatter(
                    scatter_values[:, column], scatter_values[:, row],
                    s=3, alpha=.18, color=color, rasterized=True,
                    linewidths=0,
                )
            if truth_values[column] is not None:
                axis.axvline(
                    truth_values[column], color="tab:red",
                    linestyle="--", linewidth=1,
                )
            if row != column and truth_values[row] is not None:
                axis.axhline(
                    truth_values[row], color="tab:red",
                    linestyle="--", linewidth=1,
                )
            if row == count - 1:
                axis.set_xlabel(selected_labels[column])
            else:
                axis.tick_params(labelbottom=False)
            if column == 0 and row > 0:
                axis.set_ylabel(selected_labels[row])
            elif column > 0:
                axis.tick_params(labelleft=False)
            axis.tick_params(
                which="both", direction="in", top=True, right=True
            )

    figure.tight_layout(h_pad=.15, w_pad=.15)
    return _finish(figure, save_path, dpi, show), axes


def _peak_label(peak, coordinate="frequency"):
    value = peak[coordinate]
    error = peak.get(f"{coordinate}_err")
    return (f"{coordinate[0]} = {value:.5g} +/- {error:.3g}"
            if error is not None and np.isfinite(error)
            else f"{coordinate[0]} = {value:.5g}")


def _remove_joining_y_ticks(upper_ax, lower_ax):
    upper_min, upper_max = sorted(upper_ax.get_ylim())
    upper_ticks = upper_ax.get_yticks()
    upper_ticks = upper_ticks[(upper_ticks >= upper_min) & (upper_ticks <= upper_max)]
    if len(upper_ticks) > 1:
        upper_ax.set_yticks(upper_ticks[1:])

    lower_min, lower_max = sorted(lower_ax.get_ylim())
    lower_ticks = lower_ax.get_yticks()
    lower_ticks = lower_ticks[(lower_ticks >= lower_min) & (lower_ticks <= lower_max)]
    if len(lower_ticks) > 1:
        lower_ax.set_yticks(lower_ticks[:-1])


def _remove_leftmost_x_tick(axis):
    x_min, x_max = sorted(axis.get_xlim())
    ticks = axis.get_xticks()
    ticks = ticks[(ticks >= x_min) & (ticks <= x_max)]
    if len(ticks) > 1:
        axis.set_xticks(ticks[1:])


def _set_exact_limits(axis, *, x=None, y=None):
    for values, setter in ((x, axis.set_xlim), (y, axis.set_ylim)):
        if values is None:
            continue
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size and finite.min() < finite.max():
            setter(finite.min(), finite.max())


def _style_axis(axis, *, x=None, y=None):
    """Apply the shared publication-style axes without changing plotted data."""
    _set_exact_limits(axis, x=x, y=y)
    axis.minorticks_on()
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.tick_params(which="major", length=6, width=1.1)
    axis.tick_params(which="minor", length=3, width=.9)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
    axis.grid(False)
    handles, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend(handles, labels, loc="upper right", frameon=False)


def _draw_peak_line(axis, position, *, horizontal=False):
    method = axis.axhline if horizontal else axis.axvline
    return method(position, color="red", linestyle="--", linewidth=1.2)


def _plot_extra_curves(axis, coordinate, extra_curves, *, values_on_x=False,
                       coordinate_name="the x-axis"):
    """Plot user-supplied values against an existing coordinate grid."""
    if extra_curves is None:
        return
    curves = [extra_curves] if isinstance(extra_curves, Mapping) else extra_curves
    if isinstance(curves, (str, bytes)):
        raise TypeError("extra_curves must be a mapping or an iterable of mappings")
    try:
        curves = list(curves)
    except TypeError as exc:
        raise TypeError("extra_curves must be a mapping or an iterable of mappings") from exc

    coordinate = np.asarray(coordinate)
    for index, curve in enumerate(curves):
        if not isinstance(curve, Mapping):
            raise TypeError(f"extra_curves[{index}] must be a mapping")
        if "values" not in curve:
            raise ValueError(f"extra_curves[{index}] must contain 'values'")
        values = np.asarray(curve["values"])
        if values.ndim == 0:
            values = np.full(len(coordinate), values.item())
        elif values.ndim != 1:
            raise ValueError(f"extra_curves[{index}]['values'] must be one-dimensional")
        if len(values) != len(coordinate):
            raise ValueError(
                f"extra_curves[{index}]['values'] must have the same length as "
                f"{coordinate_name} ({len(coordinate)}), got {len(values)}"
            )
        plot_kwargs = {key: value for key, value in curve.items() if key != "values"}
        if values_on_x:
            axis.plot(values, coordinate, **plot_kwargs)
        else:
            axis.plot(coordinate, values, **plot_kwargs)


def plot_wwz(result: WWZResult, *, ax=None, projection_ax=None, light_curve_ax=None,
             show_coi=True, mask_coi=False, projection_use_coi=True, coi_alpha=0.18,
             show_projection=True, show_light_curve=False,
             save_path=None, title="WWZ", cmap="viridis", figsize=(12, 7),
             dpi=200, show=False, show_peaks=False, peak_kwargs=None,
             extra_curves=None):
    """Plot a WWZ time-frequency map and its optional projections.

    When ``mask_coi`` is true, power outside the cone of influence is removed
    before plotting. Matplotlib therefore derives the color scale exclusively
    from finite power values inside the COI.
    """
    if not isinstance(result, WWZResult):
        raise TypeError("result must be a WWZResult")
    if (projection_ax is not None or light_curve_ax is not None) and ax is None:
        raise ValueError("ax must be provided with projection_ax or light_curve_ax")
    if ax is not None and show_light_curve and light_curve_ax is None:
        raise ValueError("light_curve_ax is required with an external ax when show_light_curve=True")
    if show_light_curve and (len(result.time) == 0 or len(result.values) == 0):
        raise ValueError("light-curve data are unavailable; recompute WWZ with the current version")
    if extra_curves is not None and (not show_projection if ax is None else projection_ax is None):
        raise ValueError("WWZ extra_curves require the projection panel")
    if ax is None:
        figure = plt.figure(figsize=figsize)
        row_count = 2 if show_light_curve else 1
        main_row = row_count - 1
        grid = figure.add_gridspec(
            row_count, 3 if show_projection else 2,
            width_ratios=(5, 1, .16) if show_projection else (1, .035),
            height_ratios=(1.2, 4) if show_light_curve else None,
            wspace=0, hspace=0 if show_light_curve else None,
            left=.08, right=.98, bottom=.1, top=.93,
        )
        if show_light_curve:
            light_curve_ax = figure.add_subplot(grid[0, 0])
            ax = figure.add_subplot(grid[main_row, 0], sharex=light_curve_ax)
        else:
            ax = figure.add_subplot(grid[main_row, 0])
        projection_ax = figure.add_subplot(grid[main_row, 1], sharey=ax) if show_projection else None
        colorbar_ax = (figure.add_subplot(grid[main_row, 2]) if show_projection
                       else figure.add_subplot(grid[main_row, 1]))
    else:
        figure = ax.figure
        colorbar_ax = None
    plot_power = result.power
    if mask_coi:
        if result.coi.shape != (2, len(result.frequency)):
            raise ValueError("result.coi must have shape (2, len(result.frequency))")
        tau_grid = result.tau[:, np.newaxis]
        inside_coi = ((tau_grid >= result.coi[0][np.newaxis, :])
                      & (tau_grid <= result.coi[1][np.newaxis, :]))
        plot_power = np.where(inside_coi, result.power, np.nan)
        if not np.any(np.isfinite(plot_power)):
            raise ValueError("no finite WWZ power remains inside the COI")
    image = ax.pcolormesh(result.tau, result.frequency, plot_power.T, shading="auto", cmap=cmap)
    ax.set(xlabel="Time", ylabel="Frequency", title=title)
    if show_light_curve:
        light_curve_ax.step(result.time, result.values, where="mid", color="blue", linewidth=1.3)
        if result.uncertainty is not None:
            light_curve_ax.errorbar(
                result.time, result.values, yerr=result.uncertainty, fmt="none",
                ecolor="0.45", elinewidth=.8, alpha=.65, capsize=0,
            )
        light_curve_ax.set(ylabel="Value")
        light_curve_ax.tick_params(axis="x", which="both", labelbottom=False)
        _remove_joining_y_ticks(light_curve_ax, ax)
        ax.set_title("")
        light_curve_ax.set_title(title)
    if colorbar_ax is None:
        figure.colorbar(image, ax=ax, label="WWZ power")
    else:
        figure.colorbar(image, cax=colorbar_ax, label="WWZ power")
    if show_coi and result.coi.shape == (2, len(result.frequency)):
        left, right = result.coi
        ax.fill_betweenx(result.frequency, result.tau.min(), left, color="white", alpha=coi_alpha)
        ax.fill_betweenx(result.frequency, right, result.tau.max(), color="white", alpha=coi_alpha)
        ax.plot(left, result.frequency, color="white", linewidth=.8)
        ax.plot(right, result.frequency, color="white", linewidth=.8)
    if projection_ax is not None:
        projection = project_wwz(result.power, result.tau, result.frequency, result.c, use_coi=projection_use_coi)
        projection_ax.plot(
            projection, result.frequency, color="blue", linewidth=1.3,
            label="Mean WWZ power",
        )
        _plot_extra_curves(
            projection_ax, result.frequency, extra_curves,
            values_on_x=True, coordinate_name="frequency",
        )
        projection_ax.set(xlabel="Mean WWZ power")
        projection_ax.tick_params(axis="y", labelleft=False)
        _remove_leftmost_x_tick(projection_ax)
        if show_peaks:
            for peak in result.find_peaks(use_coi=projection_use_coi, **(peak_kwargs or {})):
                _draw_peak_line(projection_ax, peak["frequency"], horizontal=True)
                projection_ax.annotate(
                    _peak_label(peak),
                    (projection[peak["index"]], peak["frequency"]),
                    xytext=(-4, 3), textcoords="offset points",
                    ha="right", va="bottom", clip_on=True,
                )
        _style_axis(projection_ax, y=result.frequency)
    _style_axis(ax, x=result.tau, y=result.frequency)
    if light_curve_ax is not None:
        _style_axis(light_curve_ax, x=result.tau)
    return _finish(figure, save_path, dpi, show), (ax, projection_ax)


def _plot_periodogram(result, *, ax, x_axis, log_x, title, ylabel, show_peaks, peak_kwargs,
                      extra_curves, series_label):
    if x_axis not in {"frequency", "period"}:
        raise ValueError("x_axis must be 'frequency' or 'period'")
    x = result.frequency if x_axis == "frequency" else result.period
    ax.plot(
        x, result.power if hasattr(result, "power") else result.statistic,
        color="blue", linewidth=1.2, label=series_label,
    )
    _plot_extra_curves(ax, x, extra_curves)
    if show_peaks:
        for peak in result.find_peaks(**(peak_kwargs or {})):
            px = peak["frequency"] if x_axis == "frequency" else peak["period"]
            peak_height = peak.get("power", peak.get("statistic", 0))
            _draw_peak_line(ax, px)
            ax.annotate(
                _peak_label(peak, "frequency" if x_axis == "frequency" else "period"),
                (px, peak_height * .5),
            )
    if log_x: ax.set_xscale("log")
    ax.set(xlabel=x_axis.title(), ylabel=ylabel, title=title)
    _style_axis(ax, x=x)


def plot_lomb_scargle(result: LombScargleResult, *, ax=None, save_path=None, title=None,
                       x_axis="frequency", log_x=False, figsize=(8, 4.8), dpi=200,
                       show=False, show_peaks=False, peak_kwargs=None,
                       extra_curves=None):
    if not isinstance(result, LombScargleResult): raise TypeError("result must be a LombScargleResult")
    figure, axis = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    _plot_periodogram(result, ax=axis, x_axis=x_axis, log_x=log_x, title=title or ("Generalized Lomb-Scargle" if result.is_generalized else "Lomb-Scargle"), ylabel="Power", show_peaks=show_peaks, peak_kwargs=peak_kwargs, extra_curves=extra_curves, series_label="Lomb-Scargle power")
    return _finish(figure, save_path, dpi, show), axis


def plot_jurkevich(result: JurkevichResult, *, ax=None, save_path=None, title="Jurkevich",
                   x_axis="period", log_x=False, figsize=(8, 4.8), dpi=200,
                   show=False, show_peaks=False, peak_kwargs=None,
                   extra_curves=None):
    if not isinstance(result, JurkevichResult): raise TypeError("result must be a JurkevichResult")
    figure, axis = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    x = result.period
    axis.plot(x, result.v_norm, color="blue", linewidth=1.2, label="Normalized variance")
    _plot_extra_curves(axis, x, extra_curves)
    if log_x: axis.set_xscale("log")
    if show_peaks:
        for peak in result.find_peaks(**(peak_kwargs or {})):
            _draw_peak_line(axis, peak["period"])
    axis.set(xlabel="Period", ylabel="Normalized variance", title=title)
    _style_axis(axis, x=x)
    return _finish(figure, save_path, dpi, show), axis


def plot_dcf(result: DCFResult, *, ax=None, save_path=None, title="DCF",
             figsize=(8, 4.8), dpi=200, show=False, show_peaks=False, peak_kwargs=None,
             extra_curves=None):
    if not isinstance(result, DCFResult): raise TypeError("result must be a DCFResult")
    figure, axis = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    axis.errorbar(result.lag, result.correlation, yerr=result.error, fmt=".", color="0.55", alpha=.5)
    axis.plot(result.lag, result.correlation, color="blue", linewidth=1.1, label="DCF coefficient")
    _plot_extra_curves(axis, result.lag, extra_curves)
    if show_peaks:
        for peak in result.find_peaks(**(peak_kwargs or {})):
            _draw_peak_line(axis, peak["lag"])
    axis.set(xlabel="Lag tau", ylabel="DCF coefficient", title=title)
    _style_axis(axis, x=result.lag)
    return _finish(figure, save_path, dpi, show), axis


def plot_z2n(result: Z2nResult, *, ax=None, save_path=None, title=None,
             x_axis="frequency", log_x=False, figsize=(8, 4.8), dpi=200,
             show=False, show_peaks=False, peak_kwargs=None,
             extra_curves=None):
    if not isinstance(result, Z2nResult): raise TypeError("result must be a Z2nResult")
    figure, axis = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    _plot_periodogram(result, ax=axis, x_axis=x_axis, log_x=log_x,
                      title=title or rf"$Z^2_{{{result.n_harmonics}}}$ periodogram",
                      ylabel=rf"$Z^2_{{{result.n_harmonics}}}$",
                      show_peaks=show_peaks, peak_kwargs=peak_kwargs, extra_curves=extra_curves,
                      series_label=rf"$Z^2_{{{result.n_harmonics}}}$ statistics")
    return _finish(figure, save_path, dpi, show), axis
