#!/usr/bin/env python3
"""
Beam Label Studio: blind image labeling for beam/no-beam judgments.

This tool labels pre-rendered diagnostic images (typically produced by
`scripts/diagnostics/view_norm2d.py`) while keeping any existing "truth" labels
in the manifest hidden during labeling. After labeling, you can generate a
report comparing your labels to the manifest truth field.

Usage (serve UI):
  uv run python scripts/diagnostics/beam_label_studio.py \\
    --manifest /path/to/selection.json

Usage (comparison report after labeling):
  uv run python scripts/diagnostics/beam_label_studio.py \\
    --manifest /path/to/selection.json \\
    --output /path/to/labels_user.json \\
    --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import panel as pn
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency `panel`.\n\n"
        "Install diagnostics UI deps with:\n"
        "  uv sync --extra diagnostics\n\n"
        "Or install just Panel:\n"
        "  uv add panel\n"
    ) from exc

LABEL_OPTIONS: tuple[str, ...] = ("beam", "non_beam", "uncertain", "skip")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


@dataclass(frozen=True, slots=True)
class Sample:
    file: str
    spec_no: int
    detector_label: str
    image_path: Path

    def key(self) -> str:
        return f"{self.file}|{self.spec_no}|{self.detector_label}"


def _load_manifest(manifest_path: Path) -> tuple[list[Sample], str]:
    raw = manifest_path.read_bytes()
    digest = _sha256_bytes(raw)
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict) or "selection" not in obj:
        raise ValueError("manifest must be a JSON object with a top-level 'selection' field")
    selection = obj["selection"]
    if not isinstance(selection, list):
        raise ValueError("manifest['selection'] must be a list")

    root = manifest_path.parent
    samples: list[Sample] = []
    for idx, entry in enumerate(selection):
        if not isinstance(entry, dict):
            raise ValueError(f"selection[{idx}] must be an object")
        for key in ("file", "spec_no", "detector_label"):
            if key not in entry:
                raise ValueError(f"selection[{idx}] missing '{key}'")
        file = str(entry["file"])
        spec_no = int(entry["spec_no"])
        detector_label = str(entry["detector_label"])
        file_stem = Path(file).stem
        image_path = root / file_stem / f"spec_{spec_no}_{detector_label}.png"
        samples.append(
            Sample(
                file=file,
                spec_no=spec_no,
                detector_label=detector_label,
                image_path=image_path,
            )
        )
    return samples, digest


def _load_existing_labels(path: Path) -> tuple[dict[str, str], dict[str, str], str | None]:
    if not path.exists():
        return {}, {}, None
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        return {}, {}, None
    labels_raw = obj.get("labels", {})
    notes_raw = obj.get("notes", {})
    manifest_digest = obj.get("manifest_sha256")

    labels: dict[str, str] = {}
    notes: dict[str, str] = {}
    if isinstance(labels_raw, dict):
        labels = {k: v for k, v in labels_raw.items() if isinstance(k, str) and isinstance(v, str)}
    if isinstance(notes_raw, dict):
        notes = {k: v for k, v in notes_raw.items() if isinstance(k, str) and isinstance(v, str)}
    if isinstance(manifest_digest, str):
        return labels, notes, manifest_digest
    return labels, notes, None


def _write_report(
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
    truth_field: str,
) -> None:
    manifest = json.loads(manifest_path.read_text())
    selection = manifest.get("selection", [])

    truth: dict[str, str] = {}
    if isinstance(selection, list):
        for entry in selection:
            if not isinstance(entry, dict):
                continue
            key = f"{entry.get('file')}|{entry.get('spec_no')}|{entry.get('detector_label')}"
            val = entry.get(truth_field)
            if isinstance(val, str):
                truth[key] = val

    labels_obj = json.loads(output_path.read_text())
    user_labels_raw = labels_obj.get("labels", {})
    user_labels: dict[str, str] = {}
    if isinstance(user_labels_raw, dict):
        user_labels = {
            k: v
            for k, v in user_labels_raw.items()
            if isinstance(k, str) and isinstance(v, str)
        }

    def norm(v: str) -> str:
        return v.strip().lower()

    tp = fp = tn = fn = 0
    missing = 0
    for key, y_true in truth.items():
        y_pred = user_labels.get(key)
        if y_pred is None:
            missing += 1
            continue
        y_true_n = norm(y_true)
        y_pred_n = norm(y_pred)
        if y_true_n == "beam" and y_pred_n == "beam":
            tp += 1
        elif y_true_n != "beam" and y_pred_n == "beam":
            fp += 1
        elif y_true_n != "beam" and y_pred_n != "beam":
            tn += 1
        elif y_true_n == "beam" and y_pred_n != "beam":
            fn += 1

    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")

    lines = [
        "# Beam Label Studio Report",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Labels: `{output_path}`",
        f"- Truth field: `{truth_field}`",
        f"- Generated: `{_utc_now_iso()}`",
        "",
        "## Coverage",
        f"- Truth rows: {len(truth)}",
        f"- Labeled rows: {len(user_labels)}",
        f"- Missing labels: {missing}",
        "",
        "## Confusion (beam vs not-beam)",
        f"- TP: {tp}",
        f"- FP: {fp}",
        f"- TN: {tn}",
        f"- FN: {fn}",
        "",
        "## Metrics",
        f"- Precision: {prec:.3f}" if prec == prec else "- Precision: NaN",
        f"- Recall: {rec:.3f}" if rec == rec else "- Recall: NaN",
        "",
    ]
    report_path.write_text("\n".join(lines))


def build_app(args: argparse.Namespace) -> pn.template.FastListTemplate:
    manifest_path = Path(args.manifest)
    samples_all, manifest_digest = _load_manifest(manifest_path)
    output_path = (
        Path(args.output) if args.output else manifest_path.with_name("labels_user.json")
    )

    labels, notes, existing_digest = _load_existing_labels(output_path)

    digest_warning = ""
    if existing_digest is not None and existing_digest != manifest_digest:
        digest_warning = (
            "⚠️ Existing label file was created for a different manifest (SHA256 mismatch). "
            "Continuing, but keys may not align."
        )

    if args.shuffle:
        import random

        rng = random.Random(args.seed)
        samples_all = samples_all.copy()
        rng.shuffle(samples_all)

    file_options = ["(all)", *sorted({s.file for s in samples_all})]
    file_select = pn.widgets.Select(name="File", options=file_options, value="(all)")

    idx_slider = pn.widgets.IntSlider(
        name="Index",
        start=0,
        end=max(len(samples_all) - 1, 0),
        value=0,
    )
    prev_btn = pn.widgets.Button(name="Prev", button_type="primary")
    next_btn = pn.widgets.Button(name="Next", button_type="primary")
    first_unlabeled_btn = pn.widgets.Button(name="First Unlabeled", button_type="default")

    label_radio = pn.widgets.RadioButtonGroup(
        name="Label",
        options=list(LABEL_OPTIONS),
        button_type="success",
    )
    note_input = pn.widgets.TextAreaInput(
        name="Notes",
        placeholder="Optional notes",
        height=120,
    )
    save_status = pn.pane.Alert("", alert_type="info", visible=False)
    header = pn.pane.Markdown("", sizing_mode="stretch_width")
    warning = pn.pane.Alert(digest_warning, alert_type="warning", visible=bool(digest_warning))

    image_pane = pn.pane.PNG(
        None,
        sizing_mode="stretch_width",
        height=args.img_height,
    )

    state: dict[str, Any] = {
        "samples": samples_all,
        "filtered": samples_all,
        "labels": labels,
        "notes": notes,
    }

    def current_sample() -> Sample:
        samples: list[Sample] = state["filtered"]
        if not samples:
            raise RuntimeError("no samples to label")
        idx = max(0, min(idx_slider.value, len(samples) - 1))
        return samples[idx]

    def sync_widgets_from_sample() -> None:
        samples: list[Sample] = state["filtered"]
        if not samples:
            header.object = "**No samples match the current filter.**"
            image_pane.object = None
            label_radio.value = None
            note_input.value = ""
            return

        sample = current_sample()
        key = sample.key()

        header.object = (
            f"**Sample {idx_slider.value + 1}/{len(samples)}**  \n"
            f"File: `{sample.file}`  \n"
            f"Spec No: `{sample.spec_no}`"
        )

        image_pane.object = str(sample.image_path) if sample.image_path.exists() else None

        label = state["labels"].get(key)
        label_radio.value = label if label in LABEL_OPTIONS else None
        note_input.value = state["notes"].get(key, "")

    def persist(reason: str) -> None:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_digest,
            "label_options": list(LABEL_OPTIONS),
            "labels": dict(state["labels"]),
            "notes": dict(state["notes"]),
        }
        if output_path.exists():
            try:
                existing = json.loads(output_path.read_text())
                if isinstance(existing, dict) and "created_at" in existing:
                    payload["created_at"] = existing.get("created_at")
            except Exception:
                pass

        _atomic_write_json(output_path, payload)
        save_status.object = f"Saved to `{output_path}` ({reason})"
        save_status.visible = True

    def apply_filter() -> None:
        chosen = file_select.value
        if chosen == "(all)":
            state["filtered"] = state["samples"]
        else:
            state["filtered"] = [s for s in state["samples"] if s.file == chosen]
        idx_slider.start = 0
        idx_slider.end = max(len(state["filtered"]) - 1, 0)
        idx_slider.value = 0
        sync_widgets_from_sample()

    def go(delta: int) -> None:
        if not state["filtered"]:
            return
        idx_slider.value = max(0, min(idx_slider.value + delta, idx_slider.end))
        sync_widgets_from_sample()

    def go_first_unlabeled() -> None:
        samples: list[Sample] = state["filtered"]
        for i, s in enumerate(samples):
            if s.key() not in state["labels"]:
                idx_slider.value = i
                break
        sync_widgets_from_sample()

    def on_label_change(event: Any) -> None:
        if not state["filtered"]:
            return
        sample = current_sample()
        key = sample.key()
        value = event.new
        if value is None:
            state["labels"].pop(key, None)
        elif value in LABEL_OPTIONS:
            state["labels"][key] = value
        persist("label")

    def on_note_change(event: Any) -> None:
        if not state["filtered"]:
            return
        sample = current_sample()
        key = sample.key()
        note = (event.new or "").strip()
        if note:
            state["notes"][key] = note
        else:
            state["notes"].pop(key, None)
        persist("notes")

    file_select.param.watch(lambda *_: apply_filter(), "value")
    idx_slider.param.watch(lambda *_: sync_widgets_from_sample(), "value")
    prev_btn.on_click(lambda *_: go(-1))
    next_btn.on_click(lambda *_: go(1))
    first_unlabeled_btn.on_click(lambda *_: go_first_unlabeled())
    label_radio.param.watch(on_label_change, "value")
    note_input.param.watch(on_note_change, "value")

    apply_filter()

    controls = pn.Column(
        warning,
        header,
        pn.Row(prev_btn, next_btn, first_unlabeled_btn),
        file_select,
        idx_slider,
        pn.layout.Divider(),
        label_radio,
        note_input,
        save_status,
        sizing_mode="stretch_width",
    )

    template = pn.template.FastListTemplate(
        title="Beam Label Studio",
        main=[pn.Row(controls, pn.Spacer(width=20), image_pane, sizing_mode="stretch_width")],
    )
    return template


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="Path to selection.json")
    p.add_argument(
        "--output",
        default=None,
        help="Path to output labels JSON (default: labels_user.json next to manifest)",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Write a markdown report comparing output labels to manifest truth",
    )
    p.add_argument(
        "--truth-field",
        default="human_label",
        help="Manifest field to treat as truth in --report mode (default: human_label)",
    )
    p.add_argument("--host", default="127.0.0.1", help="Host/interface for server")
    p.add_argument("--port", type=int, default=5007, help="Port for server")
    p.add_argument("--no-show", action="store_true", help="Do not open browser")
    p.add_argument("--shuffle", action="store_true", help="Shuffle sample order")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --shuffle")
    p.add_argument(
        "--img-height",
        type=int,
        default=800,
        help="Displayed image height in pixels (default: 800)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    output_path = Path(args.output) if args.output else manifest_path.with_name("labels_user.json")
    if args.report:
        report_path = output_path.with_suffix(".report.md")
        _write_report(
            manifest_path=manifest_path,
            output_path=output_path,
            report_path=report_path,
            truth_field=args.truth_field,
        )
        print(report_path)
        return

    pn.extension()
    app = build_app(args)
    pn.serve(
        app,
        address=args.host,
        port=args.port,
        show=not args.no_show,
        title="Beam Label Studio",
    )


if __name__ == "__main__":
    main()
