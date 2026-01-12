#!/usr/bin/env python3
"""
Loss Cone Studio: interactive loss-cone diagnostics in a browser UI.

Usage:
  uv run python scripts/diagnostics/losscone_studio.py data/1998/060_090MAR/3D980323.TAB
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import panel as pn
from bokeh.models import (
    ColumnDataSource,
    CustomJSTickFormatter,
    LinearColorMapper,
    Range1d,
)
from bokeh.palettes import RdBu11, Viridis256
from bokeh.plotting import figure

from src import config
from src.diagnostics import (
    LossConeSession,
    compute_loss_cone_boundary,
    interpolate_to_regular_grid,
)


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


def _build_frame(
    session: LossConeSession,
    chunk_idx: int,
    u_surface: float,
    bs_over_bm: float,
    beam_amp: float,
    beam_width_ev: float,
    u_spacecraft: float,
    n_pitch_bins: int,
) -> dict[str, np.ndarray | float | str | int]:
    chunk = session.get_chunk_data(chunk_idx)

    flux = np.where(chunk.flux > 0, chunk.flux, np.nan)
    raw_log = np.log10(np.maximum(flux, config.EPS))

    norm2d = session.get_norm2d(chunk_idx)
    norm2d = np.where(np.isfinite(norm2d) & (norm2d > 0), norm2d, np.nan)

    energies_reg, pitches_reg, raw_log_reg = interpolate_to_regular_grid(
        chunk.energies, chunk.pitches, raw_log, n_pitch_bins
    )
    _, _, norm_reg = interpolate_to_regular_grid(
        chunk.energies, chunk.pitches, norm2d, n_pitch_bins
    )

    model, model_mask = session.compute_model(
        energies=chunk.energies,
        pitches=chunk.pitches,
        u_surface=u_surface,
        bs_over_bm=bs_over_bm,
        beam_amp=beam_amp,
        beam_width_ev=beam_width_ev,
        u_spacecraft=u_spacecraft,
        return_mask=True,
    )
    _, _, model_reg = interpolate_to_regular_grid(
        chunk.energies, chunk.pitches, model, n_pitch_bins
    )
    residual = norm_reg - model_reg

    loss_cone = compute_loss_cone_boundary(
        chunk.energies, u_surface, bs_over_bm, u_spacecraft
    )

    energy_log = np.log10(np.maximum(energies_reg, config.EPS))
    if np.nanmean(np.diff(energy_log)) < 0:
        order = np.arange(len(energy_log) - 1, -1, -1)
        energy_log = energy_log[order]
        raw_log_reg = raw_log_reg[order]
        norm_reg = norm_reg[order]
        model_reg = model_reg[order]
        residual = residual[order]
        loss_cone = loss_cone[order]

    return {
        "energies": chunk.energies,
        "energy_log": energy_log,
        "pitches_reg": pitches_reg,
        "raw_log_reg": raw_log_reg,
        "norm_reg": norm_reg,
        "model_reg": model_reg,
        "residual": residual,
        "loss_cone": loss_cone,
        "chi2": session.compute_chi2(norm2d, model, model_mask),
        "spec_no": chunk.spec_no,
        "timestamp": chunk.timestamp,
    }


def _build_image_figure(
    title: str,
    source: ColumnDataSource,
    color_mapper: LinearColorMapper,
    x_range: Range1d,
    y_range: Range1d,
    x_formatter: CustomJSTickFormatter,
    y_label: str,
) -> figure:
    fig = figure(
        title=title,
        x_range=x_range,
        y_range=y_range,
        x_axis_label="Energy [eV]",
        y_axis_label=y_label,
        width=520,
        height=380,
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )
    fig.xaxis.formatter = x_formatter
    fig.image(
        image="image",
        x="x",
        y="y",
        dw="dw",
        dh="dh",
        source=source,
        color_mapper=color_mapper,
    )
    return fig


def build_app(args: argparse.Namespace) -> pn.template.FastListTemplate:
    session = LossConeSession(
        er_file=args.er_file,
        theta_file=args.theta_file,
        normalization_mode=args.normalization,
        incident_flux_stat=args.incident_stat,
        loss_cone_background=args.background,
        use_torch=args.fast,
    )

    max_chunk = max(session.chunk_count() - 1, 0)
    start_chunk = min(max(args.chunk, 0), max_chunk)

    chunk_slider = pn.widgets.IntSlider(
        name="Chunk", start=0, end=max_chunk, value=start_chunk, step=1
    )
    chunk_input = pn.widgets.IntInput(name="Chunk Jump", value=start_chunk, step=1)
    chunk_go = pn.widgets.Button(name="Go", button_type="primary")

    spec_input = pn.widgets.IntInput(name="Spec No", value=0, step=1)
    spec_go = pn.widgets.Button(name="Go To Spec", button_type="primary")

    normalization = pn.widgets.Select(
        name="Normalization",
        options=["global", "ratio", "ratio2", "ratio_rescaled"],
        value=args.normalization,
    )
    incident_stat = pn.widgets.Select(
        name="Incident Stat", options=["mean", "max"], value=args.incident_stat
    )

    u_surface = pn.widgets.FloatInput(
        name="U_surface [V]", value=args.u_surface, step=10.0
    )
    bs_over_bm = pn.widgets.FloatInput(
        name="Bs/Bm", value=args.bs_over_bm, step=0.01
    )
    beam_amp = pn.widgets.FloatInput(
        name="Beam Amp", value=args.beam_amp, step=0.1
    )
    beam_width = pn.widgets.FloatInput(
        name="Beam Width [eV]", value=args.beam_width, step=1.0
    )
    u_spacecraft = pn.widgets.FloatInput(
        name="U_sc [V]", value=args.u_spacecraft, step=5.0
    )
    n_pitch_bins = pn.widgets.IntInput(
        name="Pitch bins", value=args.n_pitch_bins, step=5
    )
    fit_mode = pn.widgets.Select(
        name="Fit Mode", options=["quick_lhs", "full"], value="quick_lhs"
    )
    fit_button = pn.widgets.Button(name="Fit Current Chunk", button_type="success")

    status = pn.pane.Alert("", alert_type="info", sizing_mode="stretch_width")

    raw_source = ColumnDataSource(data=dict(image=[np.zeros((2, 2))], x=[0], y=[0], dw=[1], dh=[1]))
    norm_source = ColumnDataSource(data=dict(image=[np.zeros((2, 2))], x=[0], y=[0], dw=[1], dh=[1]))
    model_source = ColumnDataSource(data=dict(image=[np.zeros((2, 2))], x=[0], y=[0], dw=[1], dh=[1]))
    residual_source = ColumnDataSource(
        data=dict(image=[np.zeros((2, 2))], x=[0], y=[0], dw=[1], dh=[1])
    )
    loss_cone_source = ColumnDataSource(data=dict(x=[], y=[]))

    raw_mapper = LinearColorMapper(palette=Viridis256, low=0.0, high=1.0)
    norm_mapper = LinearColorMapper(palette=Viridis256, low=0.0, high=1.0)
    model_mapper = LinearColorMapper(palette=Viridis256, low=0.0, high=1.0)
    residual_mapper = LinearColorMapper(palette=RdBu11[::-1], low=-1.0, high=1.0)

    x_range = Range1d(0.0, 1.0)
    y_range = Range1d(0.0, 180.0)
    x_formatter = CustomJSTickFormatter(
        code="return (Math.pow(10, tick)).toExponential(1);"
    )

    raw_fig = _build_image_figure(
        "Raw log10 flux", raw_source, raw_mapper, x_range, y_range, x_formatter, "Pitch [deg]"
    )
    norm_fig = _build_image_figure(
        "Normalized", norm_source, norm_mapper, x_range, y_range, x_formatter, "Pitch [deg]"
    )
    model_fig = _build_image_figure(
        "Model", model_source, model_mapper, x_range, y_range, x_formatter, "Pitch [deg]"
    )
    residual_fig = _build_image_figure(
        "Residual (obs - model)", residual_source, residual_mapper, x_range, y_range, x_formatter, "Pitch [deg]"
    )

    for fig in (raw_fig, norm_fig, model_fig, residual_fig):
        fig.line("x", "y", source=loss_cone_source, line_color="white", line_width=2)

    metrics = pn.pane.Markdown("", sizing_mode="stretch_width")

    guard = {"busy": False}

    def sync_chunk_input() -> None:
        chunk_input.value = chunk_slider.value

    def sync_chunk_slider() -> None:
        value = int(chunk_input.value)
        if 0 <= value <= max_chunk:
            chunk_slider.value = value
        else:
            status.object = f"Chunk {value} out of range (0..{max_chunk})."
            status.alert_type = "warning"

    def apply_spec_jump() -> None:
        target = session.spec_to_chunk(int(spec_input.value))
        if target is None:
            status.object = f"Spec {spec_input.value} not found."
            status.alert_type = "warning"
            return
        chunk_slider.value = target
        chunk_input.value = target

    def update_view(*_events) -> None:
        if guard["busy"]:
            return
        guard["busy"] = True
        try:
            session.set_normalization(normalization.value, incident_stat.value)
            frame = _build_frame(
                session=session,
                chunk_idx=int(chunk_slider.value),
                u_surface=float(u_surface.value),
                bs_over_bm=float(bs_over_bm.value),
                beam_amp=float(beam_amp.value),
                beam_width_ev=float(beam_width.value),
                u_spacecraft=float(u_spacecraft.value),
                n_pitch_bins=int(n_pitch_bins.value),
            )

            energy_log = frame["energy_log"]
            pitch_min = float(np.nanmin(frame["pitches_reg"]))
            pitch_max = float(np.nanmax(frame["pitches_reg"]))
            x_min = float(np.nanmin(energy_log))
            x_max = float(np.nanmax(energy_log))
            x_range.start = x_min
            x_range.end = x_max
            y_range.start = pitch_min
            y_range.end = pitch_max

            def pack_image(data: np.ndarray) -> dict[str, list[object]]:
                return dict(
                    image=[data.T],
                    x=[x_min],
                    y=[pitch_min],
                    dw=[x_max - x_min],
                    dh=[pitch_max - pitch_min],
                )

            raw_source.data = pack_image(frame["raw_log_reg"])
            norm_source.data = pack_image(frame["norm_reg"])
            model_source.data = pack_image(frame["model_reg"])
            residual_source.data = pack_image(frame["residual"])

            loss_cone_source.data = dict(
                x=energy_log,
                y=frame["loss_cone"],
            )

            raw_mapper.low, raw_mapper.high = _finite_range(
                frame["raw_log_reg"], fallback=(0.0, 1.0), pct=(5, 95)
            )
            norm_mapper.low, norm_mapper.high = _finite_range(
                frame["norm_reg"], fallback=(0.0, 1.0), pct=(1, 99)
            )
            model_mapper.low, model_mapper.high = 0.0, 1.0
            resid_low, resid_high = _finite_range(
                frame["residual"], fallback=(-0.5, 0.5), pct=(5, 95)
            )
            max_abs = max(abs(resid_low), abs(resid_high))
            residual_mapper.low = -max_abs
            residual_mapper.high = max_abs

            spec_input.value = int(frame["spec_no"])
            metrics.object = (
                f"Chunk: {chunk_slider.value}  |  Spec: {frame['spec_no']}  |  "
                f"UTC: {frame['timestamp']}  |  chi2: {frame['chi2']:.3g}"
            )
            status.object = ""
            status.alert_type = "info"
        except Exception as exc:
            status.object = f"Update failed: {exc}"
            status.alert_type = "danger"
        finally:
            guard["busy"] = False

    def run_fit(_event) -> None:
        if guard["busy"]:
            return
        guard["busy"] = True
        try:
            session.set_normalization(normalization.value, incident_stat.value)
            if fit_mode.value == "full":
                u_val, bs_val, beam_val, chi2 = session.fit_chunk_full(
                    int(chunk_slider.value)
                )
            else:
                u_val, bs_val, beam_val, chi2 = session.fit_chunk_lhs(
                    int(chunk_slider.value),
                    beam_width_ev=float(beam_width.value),
                    u_spacecraft=float(u_spacecraft.value),
                )
            if not np.isfinite(u_val):
                status.object = "Fit failed for this chunk."
                status.alert_type = "warning"
                return
            u_surface.value = float(u_val)
            bs_over_bm.value = float(bs_val)
            beam_amp.value = float(beam_val)
            status.object = f"Fit complete (chi2={chi2:.3g})."
            status.alert_type = "success"
            update_view()
        finally:
            guard["busy"] = False

    chunk_slider.param.watch(lambda *_: sync_chunk_input(), "value")
    chunk_go.on_click(lambda *_: sync_chunk_slider())
    spec_go.on_click(lambda *_: apply_spec_jump())

    for widget in [
        chunk_slider,
        normalization,
        incident_stat,
        u_surface,
        bs_over_bm,
        beam_amp,
        beam_width,
        u_spacecraft,
        n_pitch_bins,
    ]:
        widget.param.watch(update_view, "value")

    fit_button.on_click(run_fit)

    update_view()

    controls = pn.Column(
        "## Controls",
        chunk_slider,
        chunk_input,
        chunk_go,
        spec_input,
        spec_go,
        normalization,
        incident_stat,
        pn.layout.Divider(),
        u_surface,
        bs_over_bm,
        beam_amp,
        beam_width,
        u_spacecraft,
        n_pitch_bins,
        fit_mode,
        fit_button,
        pn.layout.Divider(),
        metrics,
        status,
        sizing_mode="stretch_width",
    )

    plots = pn.GridBox(
        raw_fig, norm_fig, model_fig, residual_fig, ncols=2, sizing_mode="stretch_both"
    )

    template = pn.template.FastListTemplate(
        title="Loss Cone Studio",
        sidebar=[controls],
        main=[plots],
        accent_base_color="#2c4f4f",
        header_background="#fbf1e6",
    )
    return template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Loss Cone Studio (interactive diagnostics UI)."
    )
    parser.add_argument("er_file", type=Path, help="Path to ER .TAB file")
    parser.add_argument(
        "--theta-file",
        type=Path,
        default=config.DATA_DIR / config.THETA_FILE,
        help="Theta file for pitch-angle calculations",
    )
    parser.add_argument("--port", type=int, default=5006, help="Server port")
    parser.add_argument("--chunk", type=int, default=0, help="Initial chunk index")
    parser.add_argument(
        "--normalization",
        choices=["global", "ratio", "ratio2", "ratio_rescaled"],
        default="ratio",
        help="Normalization mode",
    )
    parser.add_argument(
        "--incident-stat",
        choices=["mean", "max"],
        default="mean",
        help="Incident flux statistic",
    )
    parser.add_argument(
        "--u-surface",
        type=float,
        default=-200.0,
        help="Initial U_surface [V]",
    )
    parser.add_argument(
        "--bs-over-bm",
        type=float,
        default=1.0,
        help="Initial Bs/Bm",
    )
    parser.add_argument(
        "--beam-amp",
        type=float,
        default=1.0,
        help="Initial beam amplitude",
    )
    parser.add_argument(
        "--beam-width",
        type=float,
        default=config.LOSS_CONE_BEAM_WIDTH_EV,
        help="Initial beam width [eV]",
    )
    parser.add_argument(
        "--u-spacecraft",
        type=float,
        default=0.0,
        help="Initial spacecraft potential [V]",
    )
    parser.add_argument(
        "--n-pitch-bins",
        type=int,
        default=100,
        help="Number of pitch bins for interpolation",
    )
    parser.add_argument(
        "--background",
        type=float,
        default=config.LOSS_CONE_BACKGROUND,
        help="Loss-cone background level",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use torch model if available",
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
        {"losscone_studio": app},
        port=args.port,
        show=not args.no_open,
        websocket_origin=[f"localhost:{args.port}"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
