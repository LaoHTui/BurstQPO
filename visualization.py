"""Optional Matplotlib plots for the five independent search methods."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .algorithms.wwz import project_wwz
from .results import DCFResult, JurkevichResult, LombScargleResult, WWZResult, Z2nResult


def _finish(figure, save_path, dpi, show):
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def _peak_label(peak, coordinate="frequency"):
    value = peak[coordinate]
    error = peak.get(f"{coordinate}_err")
    return (f"{coordinate[0]} = {value:.5g} +/- {error:.3g}"
            if error is not None and np.isfinite(error)
            else f"{coordinate[0]} = {value:.5g}")


def plot_wwz(result: WWZResult, *, ax=None, projection_ax=None, show_coi=True,
             projection_use_coi=True, coi_alpha=0.18, show_projection=True,
             save_path=None, title="WWZ", cmap="viridis", figsize=(12, 7),
             dpi=200, show=False, show_peaks=False, peak_kwargs=None):
    if not isinstance(result, WWZResult):
        raise TypeError("result must be a WWZResult")
    if projection_ax is not None and ax is None:
        raise ValueError("ax and projection_ax must be provided together")
    if ax is None:
        figure = plt.figure(figsize=figsize)
        grid = figure.add_gridspec(1, 3 if show_projection else 2,
                                   width_ratios=(5, 1.35, .16) if show_projection else (1, .035),
                                   wspace=0, left=.08, right=.98, bottom=.1, top=.93)
        ax = figure.add_subplot(grid[0, 0])
        projection_ax = figure.add_subplot(grid[0, 1], sharey=ax) if show_projection else None
        colorbar_ax = figure.add_subplot(grid[0, 2]) if show_projection else figure.add_subplot(grid[0, 1])
    else:
        figure = ax.figure
        colorbar_ax = None
    image = ax.pcolormesh(result.tau, result.frequency, result.power.T, shading="auto", cmap=cmap)
    ax.set(xlabel="Time", ylabel="Frequency", title=title)
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
        projection_ax.plot(projection, result.frequency, color="black", linewidth=1.3)
        projection_ax.set(xlabel="Mean WWZ power")
        projection_ax.tick_params(axis="y", labelleft=False)
        if show_peaks:
            for peak in result.find_peaks(use_coi=projection_use_coi, **(peak_kwargs or {})):
                projection_ax.axhline(peak["frequency"], color="tab:red", alpha=.5)
                projection_ax.annotate(_peak_label(peak), (projection[peak["index"]], peak["frequency"]))
    return _finish(figure, save_path, dpi, show), (ax, projection_ax)


def _plot_periodogram(result, *, ax, x_axis, log_x, title, ylabel, show_peaks, peak_kwargs):
    if x_axis not in {"frequency", "period"}:
        raise ValueError("x_axis must be 'frequency' or 'period'")
    x = result.frequency if x_axis == "frequency" else result.period
    ax.plot(x, result.power if hasattr(result, "power") else result.statistic, color="black", linewidth=1.2)
    if show_peaks:
        for peak in result.find_peaks(**(peak_kwargs or {})):
            px = peak["frequency"] if x_axis == "frequency" else peak["period"]
            ax.axvline(px, color="tab:red", alpha=.35)
            ax.annotate(_peak_label(peak, "frequency" if x_axis == "frequency" else "period"), (px, peak.get("power", peak.get("statistic", 0))))
    if log_x: ax.set_xscale("log")
    ax.set(xlabel=x_axis.title(), ylabel=ylabel, title=title)
    ax.grid(True, alpha=.18)


def plot_lomb_scargle(result: LombScargleResult, *, ax=None, save_path=None, title=None,
                       x_axis="frequency", log_x=False, figsize=(8, 4.8), dpi=200,
                       show=False, show_peaks=False, peak_kwargs=None):
    if not isinstance(result, LombScargleResult): raise TypeError("result must be a LombScargleResult")
    figure, axis = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    _plot_periodogram(result, ax=axis, x_axis=x_axis, log_x=log_x, title=title or ("Generalized Lomb-Scargle" if result.is_generalized else "Lomb-Scargle"), ylabel="Power", show_peaks=show_peaks, peak_kwargs=peak_kwargs)
    return _finish(figure, save_path, dpi, show), axis


def plot_jurkevich(result: JurkevichResult, *, ax=None, save_path=None, title="Jurkevich",
                   x_axis="period", log_x=False, figsize=(8, 4.8), dpi=200,
                   show=False, show_peaks=False, peak_kwargs=None):
    if not isinstance(result, JurkevichResult): raise TypeError("result must be a JurkevichResult")
    figure, axis = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    x = result.period
    axis.plot(x, result.v_norm, color="black", linewidth=1.2)
    if log_x: axis.set_xscale("log")
    if show_peaks:
        for peak in result.find_peaks(**(peak_kwargs or {})):
            axis.axvline(peak["period"], color="tab:red", alpha=.4)
    axis.set(xlabel="Period", ylabel="Normalized variance", title=title)
    axis.grid(True, alpha=.18)
    return _finish(figure, save_path, dpi, show), axis


def plot_dcf(result: DCFResult, *, ax=None, save_path=None, title="DCF",
             figsize=(8, 4.8), dpi=200, show=False, show_peaks=False, peak_kwargs=None):
    if not isinstance(result, DCFResult): raise TypeError("result must be a DCFResult")
    figure, axis = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    axis.errorbar(result.lag, result.correlation, yerr=result.error, fmt=".", color="0.55", alpha=.5)
    axis.plot(result.lag, result.correlation, color="black", linewidth=1.1)
    if show_peaks:
        for peak in result.find_peaks(**(peak_kwargs or {})): axis.axvline(peak["lag"], color="tab:red", alpha=.4)
    axis.set(xlabel="Lag tau", ylabel="DCF coefficient", title=title)
    axis.grid(True, alpha=.18)
    return _finish(figure, save_path, dpi, show), axis


def plot_z2n(result: Z2nResult, *, ax=None, save_path=None, title=None,
             x_axis="frequency", log_x=False, figsize=(8, 4.8), dpi=200,
             show=False, show_peaks=False, peak_kwargs=None):
    if not isinstance(result, Z2nResult): raise TypeError("result must be a Z2nResult")
    figure, axis = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    _plot_periodogram(result, ax=axis, x_axis=x_axis, log_x=log_x,
                      title=title or f"Z2({result.n_harmonics}) periodogram", ylabel="Z2 statistic",
                      show_peaks=show_peaks, peak_kwargs=peak_kwargs)
    return _finish(figure, save_path, dpi, show), axis
