#!/usr/bin/env python3
"""
Loss Cone Orbit Studio: multi-panel orbit diagnostics with flux correction.

Usage:
  uv run python scripts/diagnostics/losscone_orbit_studio.py data/1999/091_120APR/3D990429.TAB
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import panel as pn
import spiceypy as spice
from bokeh.models import (
    ColumnDataSource,
    CustomJSTickFormatter,
    FixedTicker,
    Label,
    LinearColorMapper,
    Range1d,
)
from bokeh.palettes import Category20, Viridis256, Turbo256
from bokeh.plotting import figure

from src import config
from src.flux import ERData, PitchAngle
from src.potential_mapper.spice import load_spice_files
from src.spacecraft_potential import calculate_potential
from src.utils.geometry import get_intersection_or_none
from src.utils.spice_ops import (
    get_lp_position_wrt_moon,
    get_lp_vector_to_sun_in_lunar_frame,
)


@dataclass(frozen=True)
class OrbitData:
    spec_nos: np.ndarray
    times: list[datetime]
    time_labels: list[str]
    date_label: str
    energies: np.ndarray
    flux_mean: np.ndarray
    flux_median: np.ndarray
    flux_sum: np.ndarray
    u_sc: np.ndarray
    spacecraft_in_sun: np.ndarray


def _finite_range(
    data: np.ndarray, fallback: tuple[float, float], pct: tuple[float, float]
) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return fallback
    vmin = float(np.nanpercentile(finite, pct[0]))
    vmax = float(np.nanpercentile(finite, pct[1]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return fallback
    return vmin, vmax


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            et = float(text)
        except ValueError:
            return None
        try:
            utc = spice.et2utc(et, "ISOC", 0)
        except Exception:
            return None
        try:
            return datetime.fromisoformat(utc)
        except ValueError:
            return None


def _utc_to_et(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(spice.str2et(str(value)))
    except Exception:
        return None


def _compute_spacecraft_in_sun(et: float | None) -> bool:
    if et is None:
        return False
    lp_pos = get_lp_position_wrt_moon(et)
    lp_to_sun = get_lp_vector_to_sun_in_lunar_frame(et)
    if lp_pos is None or lp_to_sun is None:
        return False
    intersection = get_intersection_or_none(
        lp_pos, lp_to_sun, config.LUNAR_RADIUS
    )
    return intersection is None


def _interpolate_log_flux(
    energies: np.ndarray, flux: np.ndarray, log_energy_grid: np.ndarray
) -> np.ndarray:
    valid = (
        np.isfinite(energies)
        & np.isfinite(flux)
        & (energies > 0)
        & (flux > 0)
    )
    if np.count_nonzero(valid) < 2:
        return np.full_like(log_energy_grid, np.nan)
    log_e = np.log10(energies[valid])
    log_f = np.log10(flux[valid])
    order = np.argsort(log_e)
    log_e_sorted = log_e[order]
    log_f_sorted = log_f[order]
    interp = np.interp(log_energy_grid, log_e_sorted, log_f_sorted)
    interp[log_energy_grid < log_e_sorted[0]] = np.nan
    interp[log_energy_grid > log_e_sorted[-1]] = np.nan
    return interp


def _collapse_flux(flux: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_energy = flux.shape[0]
    means = np.full(n_energy, np.nan)
    medians = np.full(n_energy, np.nan)
    sums = np.full(n_energy, np.nan)
    for i in range(n_energy):
        row_vals = flux[i]
        row_vals = row_vals[np.isfinite(row_vals)]
        if row_vals.size:
            means[i] = float(np.mean(row_vals))
            medians[i] = float(np.median(row_vals))
            sums[i] = float(np.sum(row_vals))
    return means, medians, sums


def _rebin_spectrogram(
    energies: np.ndarray,
    flux: np.ndarray,
    log_energy_grid: np.ndarray,
) -> np.ndarray:
    n_specs = energies.shape[0]
    log_flux = np.full((len(log_energy_grid), n_specs), np.nan)
    for i in range(n_specs):
        log_flux[:, i] = _interpolate_log_flux(
            energies[i], flux[i], log_energy_grid
        )
    return log_flux


def _build_time_formatter(ticks: list[int], labels: list[str]) -> CustomJSTickFormatter:
    mapping = {str(int(tick)): label for tick, label in zip(ticks, labels, strict=False)}
    code = f"""
        const mapping = {json.dumps(mapping)};
        const key = Math.round(tick).toString();
        return mapping[key] || "";
    """
    return CustomJSTickFormatter(code=code)


def _default_energy_selection(energies: np.ndarray) -> list[float]:
    targets = np.array([100.0, 200.0, 500.0, 1000.0])
    selections = []
    for target in targets:
        idx = int(np.nanargmin(np.abs(energies - target)))
        selections.append(float(energies[idx]))
    return sorted(set(selections))


def _hours_since_midnight(times: list[datetime]) -> np.ndarray:
    if not times:
        return np.array([])
    base = datetime(times[0].year, times[0].month, times[0].day)
    return np.array(
        [((dt - base).total_seconds() / 3600.0) for dt in times], dtype=float
    )


def _hour_ticks() -> tuple[list[float], list[str]]:
    ticks = [float(4 * i) for i in range(7)]
    labels = [f"{4 * i:02d}:00" for i in range(7)]
    return ticks, labels


def _load_orbit_data(er_file: Path, theta_file: Path) -> OrbitData:
    load_spice_files()

    er_data = ERData(str(er_file))
    pitch_angle = PitchAngle(er_data)

    spec_nos = er_data.data[config.SPEC_NO_COLUMN].unique()
    n_specs = len(spec_nos)
    n_energy = config.SWEEP_ROWS

    energies = np.full((n_specs, n_energy), np.nan)
    flux_mean = np.full((n_specs, n_energy), np.nan)
    flux_median = np.full((n_specs, n_energy), np.nan)
    flux_sum = np.full((n_specs, n_energy), np.nan)
    u_sc = np.full(n_specs, np.nan)
    spacecraft_in_sun = np.zeros(n_specs, dtype=bool)

    times: list[datetime] = []
    time_labels: list[str] = []

    energy_backup = er_data.data[config.ENERGY_COLUMN].to_numpy(copy=True)

    for idx, spec_no in enumerate(spec_nos):
        chunk = er_data.data[er_data.data[config.SPEC_NO_COLUMN] == spec_no]
        if chunk.empty:
            continue

        energies[idx, : len(chunk)] = chunk[config.ENERGY_COLUMN].to_numpy(
            dtype=np.float64
        )
        flux = chunk[config.FLUX_COLS].to_numpy(dtype=np.float64)
        pitches = pitch_angle.pitch_angles[chunk.index.to_numpy()]

        # TODO: Confirm Earthward pitch-angle convention; polarity/sign pending.
        earthward_mask = pitches < 90.0
        flux_masked = np.where(earthward_mask, flux, np.nan)
        flux_masked = np.where(flux_masked > 0, flux_masked, np.nan)

        (
            flux_mean[idx],
            flux_median[idx],
            flux_sum[idx],
        ) = _collapse_flux(flux_masked)

        utc_val = chunk.iloc[0].get(
            config.UTC_COLUMN, chunk.iloc[0].get(config.TIME_COLUMN)
        )
        dt = _parse_datetime(utc_val)
        if dt is None:
            dt = datetime.utcfromtimestamp(0)
        times.append(dt)
        time_labels.append(dt.strftime("%H%M"))

        et = _utc_to_et(utc_val)
        spacecraft_in_sun[idx] = _compute_spacecraft_in_sun(et)

        try:
            result = calculate_potential(er_data, int(spec_no))
        except Exception:
            result = None
        if result is not None:
            _params, usc = result
            u_sc[idx] = float(usc.magnitude)

        er_data.data[config.ENERGY_COLUMN] = energy_backup

    date_label = times[0].strftime("%Y %b %d") if times else "Unknown date"

    return OrbitData(
        spec_nos=spec_nos,
        times=times,
        time_labels=time_labels,
        date_label=date_label,
        energies=energies,
        flux_mean=flux_mean,
        flux_median=flux_median,
        flux_sum=flux_sum,
        u_sc=u_sc,
        spacecraft_in_sun=spacecraft_in_sun,
    )


def build_app(args: argparse.Namespace) -> pn.template.FastListTemplate:
    data = _load_orbit_data(args.er_file, args.theta_file)

    n_specs = len(data.spec_nos)
    time_hours = _hours_since_midnight(data.times)
    if time_hours.size:
        time_min = float(np.nanmin(time_hours))
        time_max = float(np.nanmax(time_hours))
        if not np.isfinite(time_min) or not np.isfinite(time_max):
            time_min, time_max = 0.0, 24.0
    else:
        time_min, time_max = 0.0, 24.0

    nominal_energies = np.nanmedian(data.energies, axis=0)
    default_selection = _default_energy_selection(nominal_energies)
    energy_options = {
        f"{energy:.0f} eV": float(energy) for energy in nominal_energies
    }

    collapse = pn.widgets.Select(
        name="Earthward collapse", options=["mean", "median", "sum"], value=args.collapse
    )
    n_bins = pn.widgets.IntSlider(
        name="Energy bins (log)",
        start=32,
        end=256,
        value=args.n_bins,
        step=16,
    )
    time_range = pn.widgets.RangeSlider(
        name="Time range (hours)",
        start=0.0,
        end=24.0,
        value=(
            time_min if time_min < time_max else 0.0,
            time_max if time_min < time_max else 24.0,
        ),
        step=0.25,
    )
    auto_scale = pn.widgets.Checkbox(name="Auto color scale", value=False)
    flux_min = pn.widgets.FloatInput(
        name="Flux min (log10)",
        value=3.0,
        step=0.1,
    )
    flux_max = pn.widgets.FloatInput(
        name="Flux max (log10)",
        value=7.0,
        step=0.1,
    )
    energy_select = pn.widgets.MultiChoice(
        name="Line energies (uncorrected labels)",
        options=energy_options,
        value=default_selection,
    )

    status = pn.pane.Alert("", alert_type="info", sizing_mode="stretch_width")

    x_range = Range1d(0.0, 24.0)
    y_range = Range1d(0.0, 1.0)

    log_energy_ticks = FixedTicker(ticks=[1, 2, 3, 4, 5])
    log_energy_formatter = CustomJSTickFormatter(
        code="return (Math.pow(10, tick)).toExponential(0);"
    )

    tick_positions, tick_labels = _hour_ticks()
    time_ticker = FixedTicker(ticks=tick_positions)
    time_formatter = _build_time_formatter(tick_positions, tick_labels)

    uncorr_source = ColumnDataSource(
        data=dict(image=[np.zeros((2, 2))], x=[0.0], y=[0.0], dw=[1.0], dh=[1.0])
    )
    corr_source = ColumnDataSource(
        data=dict(image=[np.zeros((2, 2))], x=[0.0], y=[0.0], dw=[1.0], dh=[1.0])
    )
    illum_source = ColumnDataSource(
        data=dict(image=[np.zeros((1, 2))], x=[0.0], y=[0.0], dw=[1.0], dh=[1.0])
    )

    uncorr_mapper = LinearColorMapper(palette=Turbo256, low=0.0, high=1.0)
    corr_mapper = LinearColorMapper(palette=Turbo256, low=0.0, high=1.0)
    illum_mapper = LinearColorMapper(palette=["black", "blue"], low=0.0, high=1.0)

    uncorr_fig = figure(
        title="Uncorrected Earthward differential energy flux (log10)",
        x_range=x_range,
        y_range=y_range,
        height=300,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )
    uncorr_fig.image(
        image="image",
        x="x",
        y="y",
        dw="dw",
        dh="dh",
        source=uncorr_source,
        color_mapper=uncorr_mapper,
    )
    uncorr_fig.xaxis.visible = False
    uncorr_fig.yaxis.ticker = log_energy_ticks
    uncorr_fig.yaxis.formatter = log_energy_formatter
    uncorr_fig.yaxis.axis_label = "Energy [eV]"

    usc_source = ColumnDataSource(data=dict(x=time_hours, y=data.u_sc))
    usc_fig = figure(
        title="Spacecraft potential (U_sc)",
        x_range=x_range,
        y_axis_label="U_sc [V]",
        height=200,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )
    usc_fig.line("x", "y", source=usc_source, line_color="black", line_width=1.5)
    usc_fig.xaxis.visible = False

    corr_fig = figure(
        title="Corrected Earthward differential energy flux (log10)",
        x_range=x_range,
        y_range=y_range,
        height=300,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )
    corr_fig.image(
        image="image",
        x="x",
        y="y",
        dw="dw",
        dh="dh",
        source=corr_source,
        color_mapper=corr_mapper,
    )
    corr_fig.xaxis.visible = False
    corr_fig.yaxis.ticker = log_energy_ticks
    corr_fig.yaxis.formatter = log_energy_formatter
    corr_fig.yaxis.axis_label = "Corrected Energy [eV]"

    line_fig = figure(
        title="Corrected Earthward differential flux (selected energies)",
        x_range=x_range,
        y_axis_type="log",
        y_axis_label="Diff. flux",
        height=260,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )
    line_fig.xaxis.visible = False

    illum_fig = figure(
        title="Spacecraft illumination (black=shadow, blue=sun)",
        x_range=x_range,
        y_range=Range1d(0, 1),
        height=90,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )
    illum_fig.image(
        image="image",
        x="x",
        y="y",
        dw="dw",
        dh="dh",
        source=illum_source,
        color_mapper=illum_mapper,
    )
    illum_fig.yaxis.visible = False
    illum_fig.xaxis.ticker = time_ticker
    illum_fig.xaxis.formatter = time_formatter
    illum_fig.xaxis.axis_label = f"HHMM ({data.date_label})"

    date_label = Label(
        x=0.0,
        y=0.02,
        x_units="data",
        y_units="screen",
        text=f"hh:mm   {data.date_label}",
        text_font_size="10pt",
    )
    illum_fig.add_layout(date_label)

    line_renderers: dict[float, ColumnDataSource] = {}
    line_glyphs = []
    palette = Category20[20]
    for idx, energy in enumerate(nominal_energies):
        color = palette[idx % len(palette)]
        source = ColumnDataSource(
            data=dict(x=time_hours, y=np.full(n_specs, np.nan))
        )
        renderer = line_fig.line(
            "x",
            "y",
            source=source,
            line_width=1.5,
            color=color,
            legend_label=f"{energy:.0f} eV",
        )
        line_renderers[float(energy)] = source
        line_glyphs.append((float(energy), renderer))

    line_fig.legend.click_policy = "hide"
    line_fig.legend.location = "top_left"

    def update_view(*_events) -> None:
        collapse_mode = collapse.value
        if collapse_mode == "mean":
            flux = data.flux_mean
        elif collapse_mode == "median":
            flux = data.flux_median
        else:
            flux = data.flux_sum

        log_energy_grid = np.log10(
            np.geomspace(args.emin, args.emax, int(n_bins.value))
        )
        energy_grid = np.power(10.0, log_energy_grid)

        uncorr_log = _rebin_spectrogram(data.energies, flux, log_energy_grid)
        corr_energies = data.energies - data.u_sc[:, None]
        corr_log = _rebin_spectrogram(corr_energies, flux, log_energy_grid)
        corr_linear = np.power(10.0, corr_log)

        y_range.start = float(log_energy_grid.min())
        y_range.end = float(log_energy_grid.max())

        if time_hours.size:
            time_start = float(np.nanmin(time_hours))
            time_end = float(np.nanmax(time_hours))
        else:
            time_start = 0.0
            time_end = 24.0
        time_span = max(time_end - time_start, 1e-6)

        uncorr_source.data = dict(
            image=[uncorr_log],
            x=[time_start],
            y=[y_range.start],
            dw=[time_span],
            dh=[y_range.end - y_range.start],
        )
        corr_source.data = dict(
            image=[corr_log],
            x=[time_start],
            y=[y_range.start],
            dw=[time_span],
            dh=[y_range.end - y_range.start],
        )

        illum_source.data = dict(
            image=[data.spacecraft_in_sun.astype(float)[None, :]],
            x=[time_start],
            y=[0.0],
            dw=[time_span],
            dh=[1.0],
        )

        if auto_scale.value:
            uncorr_mapper.low, uncorr_mapper.high = _finite_range(
                uncorr_log, fallback=(0.0, 1.0), pct=(5, 95)
            )
            corr_mapper.low, corr_mapper.high = _finite_range(
                corr_log, fallback=(0.0, 1.0), pct=(5, 95)
            )
            status.object = ""
            status.alert_type = "info"
        else:
            vmin = float(flux_min.value)
            vmax = float(flux_max.value)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                status.object = "Invalid flux min/max; using auto scale."
                status.alert_type = "warning"
                uncorr_mapper.low, uncorr_mapper.high = _finite_range(
                    uncorr_log, fallback=(0.0, 1.0), pct=(5, 95)
                )
                corr_mapper.low, corr_mapper.high = _finite_range(
                    corr_log, fallback=(0.0, 1.0), pct=(5, 95)
                )
            else:
                uncorr_mapper.low = vmin
                uncorr_mapper.high = vmax
                corr_mapper.low = vmin
                corr_mapper.high = vmax
                status.object = ""
                status.alert_type = "info"

        log_energy = np.log10(np.maximum(energy_grid, config.EPS))
        for energy, source in line_renderers.items():
            idx = int(np.nanargmin(np.abs(log_energy - np.log10(energy))))
            y_vals = corr_linear[idx]
            y_vals = np.maximum(y_vals, config.EPS)
            source.data = dict(x=time_hours, y=y_vals)

        update_line_visibility()

    def update_time_range(*_events) -> None:
        start, end = time_range.value
        if not np.isfinite(start) or not np.isfinite(end) or start >= end:
            return
        x_range.start = float(start)
        x_range.end = float(end)

    def update_line_visibility(*_events) -> None:
        selected = set(float(val) for val in energy_select.value)
        visible_vals = []
        for energy, renderer in line_glyphs:
            renderer.visible = energy in selected
            if renderer.visible:
                source = line_renderers[energy]
                visible_vals.append(source.data["y"])

        if visible_vals:
            stacked = np.vstack(visible_vals)
            finite = stacked[np.isfinite(stacked)]
            if finite.size:
                low = float(np.nanpercentile(finite, 5))
                high = float(np.nanpercentile(finite, 95))
                low = max(low, config.EPS)
                line_fig.y_range.start = low
                line_fig.y_range.end = high

    collapse.param.watch(update_view, "value")
    n_bins.param.watch(update_view, "value")
    time_range.param.watch(update_time_range, "value")
    auto_scale.param.watch(update_view, "value")
    flux_min.param.watch(update_view, "value")
    flux_max.param.watch(update_view, "value")
    energy_select.param.watch(update_line_visibility, "value")

    update_view()
    update_time_range()

    stats = pn.pane.Markdown(
        (
            f"**Spectra:** {n_specs}\n\n"
            f"**U_sc finite:** {np.isfinite(data.u_sc).sum()} / {n_specs}\n\n"
            f"**Sunlit fraction:** {data.spacecraft_in_sun.mean():.2f}"
        ),
        sizing_mode="stretch_width",
    )

    controls = pn.Column(
        "## Controls",
        collapse,
        n_bins,
        time_range,
        auto_scale,
        flux_min,
        flux_max,
        energy_select,
        pn.layout.Divider(),
        stats,
        status,
        sizing_mode="stretch_width",
    )

    plots = pn.Column(
        uncorr_fig,
        usc_fig,
        corr_fig,
        line_fig,
        illum_fig,
        sizing_mode="stretch_both",
    )

    template = pn.template.FastListTemplate(
        title="Loss Cone Orbit Studio",
        sidebar=[controls],
        main=[plots],
        accent_base_color="#2c4f4f",
        header_background="#fbf1e6",
        main_max_width="1600px",
    )
    return template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Loss Cone Orbit Studio (multi-panel diagnostics)."
    )
    parser.add_argument("er_file", type=Path, help="Path to ER .TAB file")
    parser.add_argument(
        "--theta-file",
        type=Path,
        default=config.DATA_DIR / config.THETA_FILE,
        help="Theta file for pitch-angle calculations",
    )
    parser.add_argument("--port", type=int, default=5007, help="Server port")
    parser.add_argument(
        "--collapse",
        choices=["mean", "median", "sum"],
        default="mean",
        help="Earthward flux collapse mode",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=64,
        help="Number of log-spaced energy bins",
    )
    parser.add_argument(
        "--emin",
        type=float,
        default=7.0,
        help="Minimum energy for log grid [eV]",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=1.6e4,
        help="Maximum energy for log grid [eV]",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser window automatically",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pn.extension()
    app = build_app(args)
    pn.serve(
        {"losscone_orbit_studio": app},
        port=args.port,
        show=not args.no_open,
        websocket_origin=[f"localhost:{args.port}"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
