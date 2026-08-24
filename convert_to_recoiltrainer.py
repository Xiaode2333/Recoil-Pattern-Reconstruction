#!/usr/bin/env python3
"""Convert reconstructed recoil CSV data into a Recoil Trainer profile JSON.

The input coordinates are expected to use the convention produced by
``analyze_recoil.py``: X is positive to the right and Y is positive upward.
The conversion applies one uniform scale to both axes, so the measured recoil
direction and horizontal/vertical ratio are preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "recoil_output" / "keyframes_recoil.csv"
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "recoil_output" / "rm277_x1_opx2_recoiltrainer.json"
)
DEFAULT_RECOIL_TRAINER_ROOT = Path(r"C:\XiaodeDocuments\Programs\RecoilTrainer")

REQUIRED_COLUMNS = {
    "shot",
    "keyframe",
    "recoil_x_right_px",
    "recoil_y_up_px",
}


class ConversionError(ValueError):
    """Raised when the reconstruction data cannot form a valid profile."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert analyze_recoil.py keyframe output into a Recoil Trainer "
            "WeaponProfile JSON."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input keyframes_recoil.csv (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output profile JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--name", default="RM277 x1 OPX2")
    parser.add_argument(
        "--localized-name-en",
        default=None,
        help="English profile/weapon name stored under en-US (default: --name).",
    )
    parser.add_argument(
        "--localized-name-zh-cn",
        default=None,
        help="Simplified-Chinese profile/weapon name stored under zh-CN.",
    )
    parser.add_argument(
        "--weapon-name-en",
        default=None,
        help="English weapon-only metadata (default: --localized-name-en or --name).",
    )
    parser.add_argument(
        "--weapon-name-zh-cn",
        default=None,
        help="Simplified-Chinese weapon-only metadata.",
    )
    parser.add_argument("--game-name-en", default="Delta Force")
    parser.add_argument("--game-name-zh-cn", default="三角洲行动")
    parser.add_argument(
        "--workshop-title-en",
        default=None,
        help="English Workshop title (default: 'Delta Force | <weapon> Recoil').",
    )
    parser.add_argument(
        "--workshop-title-zh-cn",
        default=None,
        help="Simplified-Chinese Workshop title (default: '三角洲行动 | <weapon> 后坐力').",
    )
    parser.add_argument(
        "--profile-id",
        default=None,
        help="Profile ID. By default a deterministic 32-character ID is generated.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Video FPS used for t_ms. By default it is read from summary.json; "
            "time_sec is used as a fallback."
        ),
    )
    parser.add_argument(
        "--target-vertical-span",
        type=float,
        default=240.0,
        help=(
            "Normalize the trajectory to this Recoil Trainer vertical span while "
            "preserving aspect ratio (default: 240). Use 0 to keep pixel units."
        ),
    )
    parser.add_argument(
        "--coordinate-margin",
        type=float,
        default=90.0,
        help="Offset the minimum X/Y to this value (default: 90).",
    )
    parser.add_argument(
        "--recorded-pitch-deg",
        type=float,
        default=None,
        help=(
            "Physical vertical recoil represented by the profile. By default use "
            "the FOV/scope estimate from summary.json."
        ),
    )
    parser.add_argument(
        "--recoil-amplitude-divisor",
        type=float,
        default=1.0,
        help=(
            "Divide the physical recoil pitch by this value after FOV/scope "
            "estimation (default: 1). Use 0.89 for QBZ95-1."
        ),
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=104.0,
        help="Fallback base FOV for pitch estimation (default: 104 degrees).",
    )
    parser.add_argument(
        "--fov-axis",
        choices=("reference-horizontal", "horizontal", "vertical"),
        default="reference-horizontal",
        help=(
            "FOV model. Default matches Recoil Trainer's 4:3 reference-horizontal "
            "conversion."
        ),
    )
    parser.add_argument(
        "--scope-magnification",
        type=float,
        default=2.0,
        help="Fallback scope magnification for pitch estimation (default: 2x).",
    )
    parser.add_argument(
        "--smoothing",
        choices=("none", "spline"),
        default="spline",
        help="Recoil Trainer interpolation mode (default: spline).",
    )
    parser.add_argument("--smoothing-strength", type=float, default=0.2)
    parser.add_argument(
        "--segmentation",
        choices=("recoil-trainer", "single"),
        default="recoil-trainer",
        help=(
            "Use Recoil Trainer's own x-direction auto segmentation (default), "
            "or create one segment."
        ),
    )
    parser.add_argument("--drum-hit-strength", type=float, default=4.0)
    parser.add_argument(
        "--audio-spatial-style", default="pitch_ypos_sphere"
    )
    parser.add_argument("--device-mode", choices=("kbm", "controller"), default="kbm")
    parser.add_argument("--mouse-yx-ratio", type=float, default=1.0)
    parser.add_argument("--controller-yx-ratio", type=float, default=0.68)
    parser.add_argument(
        "--tags",
        default="delta-force,video-reconstruction",
        help="Comma-separated profile tags.",
    )
    parser.add_argument(
        "--recoil-trainer-root",
        type=Path,
        default=DEFAULT_RECOIL_TRAINER_ROOT,
        help=(
            "Recoil Trainer project root used to validate with its official model "
            f"(default: {DEFAULT_RECOIL_TRAINER_ROOT})."
        ),
    )
    parser.add_argument(
        "--skip-trainer-validation",
        action="store_true",
        help="Skip importing Recoil Trainer's WeaponProfile model for validation.",
    )
    return parser.parse_args(argv)


def _finite_float(value: str, column: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConversionError(
            f"Row {row_number}: {column} is not a number: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ConversionError(
            f"Row {row_number}: {column} must be finite, got {parsed!r}"
        )
    return parsed


def load_reconstruction(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        raise ConversionError(f"Input CSV does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ConversionError(
                "Input CSV is missing required columns: " + ", ".join(missing)
            )
        x_column = (
            "trainer_recoil_x_right_px"
            if "trainer_recoil_x_right_px" in columns
            else "recoil_x_right_px"
        )
        y_column = (
            "trainer_recoil_y_up_px"
            if "trainer_recoil_y_up_px" in columns
            else "recoil_y_up_px"
        )

        rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                shot = int(row["shot"])
                keyframe = int(row["keyframe"])
            except (TypeError, ValueError) as exc:
                raise ConversionError(
                    f"Row {row_number}: shot and keyframe must be integers"
                ) from exc
            rows.append(
                {
                    "shot": shot,
                    "keyframe": keyframe,
                    "time_sec": (
                        _finite_float(row["time_sec"], "time_sec", row_number)
                        if "time_sec" in columns and row.get("time_sec", "") != ""
                        else None
                    ),
                    "shot_time_ms": (
                        round(
                            _finite_float(
                                row["shot_time_ms"], "shot_time_ms", row_number
                            )
                        )
                        if "shot_time_ms" in columns
                        and row.get("shot_time_ms", "") != ""
                        else None
                    ),
                    "x": _finite_float(
                        row[x_column],
                        x_column,
                        row_number,
                    ),
                    "y": _finite_float(
                        row[y_column],
                        y_column,
                        row_number,
                    ),
                }
            )

    if len(rows) < 2:
        raise ConversionError("At least two reconstructed shots are required")

    expected_shots = list(range(1, len(rows) + 1))
    shots = [row["shot"] for row in rows]
    if shots != expected_shots:
        raise ConversionError(
            f"Shots must be contiguous and ordered 1..{len(rows)}; got {shots}"
        )

    keyframes = [row["keyframe"] for row in rows]
    if any(current <= previous for previous, current in zip(keyframes, keyframes[1:])):
        raise ConversionError("Keyframes must be strictly increasing")
    return rows


def load_summary(csv_path: Path) -> dict[str, Any]:
    summary_path = csv_path.with_name("summary.json")
    if not summary_path.is_file():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Cannot read {summary_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConversionError(f"Summary must be a JSON object: {summary_path}")
    return payload


def resolve_fps(summary: dict[str, Any], explicit_fps: float | None) -> float | None:
    if explicit_fps is not None:
        if not math.isfinite(explicit_fps) or explicit_fps <= 0:
            raise ConversionError("--fps must be a positive finite number")
        return explicit_fps
    if "fps" not in summary:
        return None
    try:
        fps = float(summary["fps"])
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"Invalid FPS in summary.json: {exc}") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise ConversionError(f"Invalid FPS in summary.json: {fps!r}")
    return fps


def make_timestamps(rows: list[dict[str, Any]], fps: float | None) -> list[int]:
    if all(row["shot_time_ms"] is not None for row in rows):
        timestamps = [int(row["shot_time_ms"]) for row in rows]
    elif fps is not None:
        first_frame = rows[0]["keyframe"]
        timestamps = [
            round((row["keyframe"] - first_frame) * 1000.0 / fps) for row in rows
        ]
    elif all(row["time_sec"] is not None for row in rows):
        first_time = rows[0]["time_sec"]
        timestamps = [round((row["time_sec"] - first_time) * 1000.0) for row in rows]
    else:
        raise ConversionError(
            "No timing source: provide --fps, place summary.json beside the CSV, "
            "or include time_sec in every CSV row"
        )

    if timestamps[0] != 0:
        raise ConversionError("The first t_ms must be zero")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ConversionError(
            "Rounded t_ms values are not strictly increasing; verify FPS/keyframes"
        )
    return timestamps


def normalize_coordinates(
    rows: list[dict[str, Any]], target_vertical_span: float, margin: float
) -> tuple[list[tuple[float, float]], float, dict[str, float]]:
    if not math.isfinite(target_vertical_span) or target_vertical_span < 0:
        raise ConversionError("--target-vertical-span must be finite and >= 0")
    if not math.isfinite(margin):
        raise ConversionError("--coordinate-margin must be finite")

    xs = [row["x"] for row in rows]
    ys = [row["y"] for row in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    vertical_span = max_y - min_y
    if vertical_span <= 1e-9:
        raise ConversionError("The reconstructed trajectory has no vertical span")

    scale = 1.0 if target_vertical_span == 0 else target_vertical_span / vertical_span
    points = [
        (
            round(margin + (x - min_x) * scale, 6),
            round(margin + (y - min_y) * scale, 6),
        )
        for x, y in zip(xs, ys)
    ]
    bounds = {
        "source_min_x": min_x,
        "source_max_x": max_x,
        "source_min_y": min_y,
        "source_max_y": max_y,
    }
    return points, scale, bounds


def resolve_recorded_pitch_deg(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, str]:
    if args.recorded_pitch_deg is not None:
        pitch = float(args.recorded_pitch_deg)
        source = "command line"
    elif "estimated_recoil_pitch_range_deg" in summary:
        pitch = float(summary["estimated_recoil_pitch_range_deg"])
        source = "summary.json FOV/scope estimate"
    else:
        try:
            width = int(summary["video_width"])
            height = int(summary["video_height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversionError(
                "Cannot estimate pitch without video_width/video_height in summary.json; "
                "provide --recorded-pitch-deg"
            ) from exc
        fov_deg = float(summary.get("fov_degrees", args.fov_deg))
        fov_axis = str(summary.get("fov_axis", args.fov_axis))
        magnification = float(
            summary.get("scope_magnification", args.scope_magnification)
        )
        if fov_axis not in {"reference-horizontal", "horizontal", "vertical"}:
            raise ConversionError(f"Invalid fov_axis in summary.json: {fov_axis!r}")
        if not 1.0 < fov_deg < 179.0:
            raise ConversionError("FOV must be between 1 and 179 degrees")
        if not math.isfinite(magnification) or magnification <= 0:
            raise ConversionError("Scope magnification must be a positive number")
        if fov_axis == "reference-horizontal":
            scoped_reference_horizontal = 2.0 * math.atan(
                math.tan(math.radians(fov_deg) / 2.0) / magnification
            )
            scoped_vertical = 2.0 * math.atan(
                math.tan(scoped_reference_horizontal / 2.0) / (4.0 / 3.0)
            )
            focal_length_px = height / (2.0 * math.tan(scoped_vertical / 2.0))
        else:
            sensor_size = width if fov_axis == "horizontal" else height
            focal_length_px = (
                sensor_size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
            ) * magnification
        angles = [math.atan(row["y"] / focal_length_px) for row in rows]
        pitch = math.degrees(max(angles) - min(angles))
        source = f"calculated from {fov_deg:g}° {fov_axis} FOV at {magnification:g}x"

    divisor = float(args.recoil_amplitude_divisor)
    if not math.isfinite(divisor) or divisor <= 0:
        raise ConversionError("--recoil-amplitude-divisor must be positive and finite")
    pitch /= divisor
    if divisor != 1.0:
        source += f"; amplitude divided by {divisor:g}"

    if not math.isfinite(pitch) or pitch <= 0:
        raise ConversionError("recorded recoil pitch must be a positive finite number")
    return pitch, source


def recoil_trainer_auto_segments(
    coordinates: list[tuple[float, float]],
    timestamps: list[int],
    project_root: Path,
    smoothing: str,
    smoothing_strength: float,
) -> list[dict[str, Any]]:
    """Call Recoil Trainer's own smoothing and Auto Segment implementation."""
    root = project_root.resolve()
    function_source = root / "src" / "app" / "desktop" / "views" / "pattern_lab_view.py"
    if not function_source.is_file():
        raise ConversionError(
            f"Recoil Trainer Auto Segment implementation was not found: {function_source}"
        )
    poetry = shutil.which("poetry")
    if poetry is None:
        raise ConversionError(
            "Poetry is required to call Recoil Trainer Auto Segment; "
            "install Poetry or use --segmentation single"
        )

    request = {
        "points": [
            {"shot_index": index, "x": x, "y": y, "t_ms": timestamp}
            for index, ((x, y), timestamp) in enumerate(zip(coordinates, timestamps))
        ],
        "smoothing": smoothing,
        "smoothing_strength": smoothing_strength,
    }
    # Poetry's Windows console entry point does not preserve newlines in the
    # ``python -c`` argument, so keep this runner as one logical line.
    runner = (
        "import json,os;"
        "payload=json.loads(os.environ.pop('RECOIL_SEGMENT_PAYLOAD'));"
        "from src.core.domain.models import TrajectoryPoint;"
        "from src.core.services.smoothing import apply_smoothing;"
        "from src.app.desktop.views.pattern_lab_view import _direction_change_segment_ranges;"
        "points=[TrajectoryPoint.from_dict(item) for item in payload['points']];"
        "smoothed=apply_smoothing(points,method=payload['smoothing'],"
        "strength=float(payload['smoothing_strength']));"
        "ranges=_direction_change_segment_ranges([point.x for point in smoothed]);"
        "print('RECOIL_SEGMENTS='+json.dumps(ranges,separators=(',',':')))"
    )
    environment = dict(os.environ)
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    environment["RECOIL_SEGMENT_PAYLOAD"] = json.dumps(request, separators=(",", ":"))
    try:
        result = subprocess.run(
            [poetry, "run", "python", "-c", runner],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConversionError(f"Cannot run Recoil Trainer Auto Segment: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ConversionError(
            f"Recoil Trainer Auto Segment failed with exit code {result.returncode}: {detail}"
        )

    prefix = "RECOIL_SEGMENTS="
    output_line = next(
        (line for line in reversed(result.stdout.splitlines()) if line.startswith(prefix)),
        None,
    )
    if output_line is None:
        raise ConversionError("Recoil Trainer Auto Segment returned no ranges")
    try:
        ranges = json.loads(output_line[len(prefix) :])
    except json.JSONDecodeError as exc:
        raise ConversionError("Recoil Trainer Auto Segment returned invalid JSON") from exc
    segments = [
        {"segment_id": f"Seq{index + 1}", "start_shot": int(start), "end_shot": int(end)}
        for index, (start, end) in enumerate(ranges)
    ]
    if not segments:
        raise ConversionError("Recoil Trainer Auto Segment produced no segments")
    return segments


def _profile_id(
    requested: str | None,
    name: str,
    rows: list[dict[str, Any]],
    target_vertical_span: float,
) -> str:
    if requested:
        cleaned = requested.strip()
        if not cleaned:
            raise ConversionError("--profile-id cannot be blank")
        return cleaned

    identity = {
        "name": name,
        "shots": [
            [row["shot"], row["keyframe"], row["x"], row["y"]] for row in rows
        ],
        "target_vertical_span": target_vertical_span,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def build_profile(
    rows: list[dict[str, Any]],
    timestamps: list[int],
    coordinates: list[tuple[float, float]],
    recorded_pitch_deg: float,
    segments: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.name.strip():
        raise ConversionError("--name cannot be blank")
    positive_options = {
        "recorded recoil pitch": recorded_pitch_deg,
        "--mouse-yx-ratio": args.mouse_yx_ratio,
        "--controller-yx-ratio": args.controller_yx_ratio,
    }
    for option, value in positive_options.items():
        if not math.isfinite(value) or value <= 0:
            raise ConversionError(f"{option} must be a positive finite number")
    if not 0.0 <= args.smoothing_strength <= 1.0:
        raise ConversionError("--smoothing-strength must be between 0 and 1")
    if not math.isfinite(args.drum_hit_strength) or args.drum_hit_strength < 0:
        raise ConversionError("--drum-hit-strength must be finite and >= 0")

    points = [
        {"shot_index": index, "x": x, "y": y, "t_ms": t_ms}
        for index, ((x, y), t_ms) in enumerate(zip(coordinates, timestamps))
    ]
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    localized_name_en = str(args.localized_name_en or args.name).strip()
    localized_name_zh_cn = str(args.localized_name_zh_cn or localized_name_en).strip()
    if not localized_name_en or not localized_name_zh_cn:
        raise ConversionError("Localized profile names cannot be blank")
    return {
        "profile_id": _profile_id(
            args.profile_id, args.name.strip(), rows, args.target_vertical_span
        ),
        "name": args.name.strip(),
        "points": points,
        "segments": segments,
        "smoothing": args.smoothing,
        "device_mode": args.device_mode,
        "smoothing_strength": args.smoothing_strength,
        "drum_hit_strength": args.drum_hit_strength,
        "audio_spatial_style": args.audio_spatial_style,
        "recorded_recoil_pitch_range_deg": recorded_pitch_deg,
        "mouse_yx_ratio": args.mouse_yx_ratio,
        "controller_yx_ratio": args.controller_yx_ratio,
        "schema_version": 1,
        "source_image_path": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "tags": tags,
        "localized_names": {
            "en-US": localized_name_en,
            "zh-CN": localized_name_zh_cn,
        },
    }


def build_workshop_payload(
    profile: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Build the same payload shape as Recoil Trainer's Workshop builder.

    Recoil Trainer stores Steam-language upload text separately from the core
    profile.  ``data.localizations`` keeps the full bilingual metadata inside
    the portable JSON while the official top-level Workshop fields remain the
    English primary localization.
    """
    try:
        weapon_en = str(
            args.weapon_name_en or args.localized_name_en or args.name
        ).strip()
        weapon_zh = str(args.weapon_name_zh_cn or weapon_en).strip()
        game_en = str(args.game_name_en).strip()
        game_zh = str(args.game_name_zh_cn).strip()
        title_en = str(
            args.workshop_title_en or f"{game_en} | {weapon_en} Recoil"
        ).strip()
        title_zh = str(
            args.workshop_title_zh_cn or f"{game_zh} | {weapon_zh} 后坐力"
        ).strip()
        fields = {
            "English game name": game_en,
            "Chinese game name": game_zh,
            "English weapon name": weapon_en,
            "Chinese weapon name": weapon_zh,
            "English Workshop title": title_en,
            "Chinese Workshop title": title_zh,
        }
        blanks = [label for label, value in fields.items() if not value]
        if blanks:
            raise ConversionError("Localized metadata cannot be blank: " + ", ".join(blanks))

        localizations = {
            "en-US": {
                "game_name": game_en,
                "weapon_name": weapon_en,
                "workshop_title": title_en,
            },
            "zh-CN": {
                "game_name": game_zh,
                "weapon_name": weapon_zh,
                "workshop_title": title_zh,
            },
        }
        payload = dict(profile)
        payload.pop("source_image_path", None)
        payload.update(
            {
                "format_version": 1,
                "profile_version": 1,
                "app": "Recoil Trainer",
                "game_name": game_en,
                "weapon_name": weapon_en,
                "workshop_title": title_en,
                "training_hints": "",
                "author_note": "",
                "data": {
                "localizations": localizations,
                "reconstruction": {
                    "source": "Feature Matching + RANSAC + reticle tracking",
                    "fov_degrees": float(args.fov_deg),
                    "scope_magnification": float(args.scope_magnification),
                    "recoil_amplitude_divisor": float(args.recoil_amplitude_divisor),
                },
            },
                "app_version": "0.1.0",
            }
        )
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f"Cannot build Recoil Trainer Workshop JSON: {exc}") from exc
    return payload


def validate_profile_shape(profile: dict[str, Any]) -> None:
    required = {
        "profile_id",
        "name",
        "points",
        "segments",
        "smoothing",
        "device_mode",
        "smoothing_strength",
        "drum_hit_strength",
        "audio_spatial_style",
        "recorded_recoil_pitch_range_deg",
        "mouse_yx_ratio",
        "controller_yx_ratio",
        "schema_version",
        "source_image_path",
        "updated_at",
        "tags",
        "localized_names",
    }
    missing = required - profile.keys()
    if missing:
        raise ConversionError("Generated profile is missing fields: " + ", ".join(sorted(missing)))

    points = profile["points"]
    if [point["shot_index"] for point in points] != list(range(len(points))):
        raise ConversionError("Generated shot_index values are not contiguous from zero")
    times = [point["t_ms"] for point in points]
    if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ConversionError("Generated t_ms values are invalid")
    segments = profile["segments"]
    if not segments:
        raise ConversionError("Generated profile has no segments")
    expected_start = 0
    seen_ids: set[str] = set()
    for segment in segments:
        segment_id = str(segment.get("segment_id", "")).strip()
        start = int(segment["start_shot"])
        end = int(segment["end_shot"])
        if not segment_id or segment_id in seen_ids:
            raise ConversionError("Generated segment IDs must be non-empty and unique")
        if start != expected_start or end < start:
            raise ConversionError("Generated segments must be contiguous and ordered")
        seen_ids.add(segment_id)
        expected_start = end + 1
    if expected_start != len(points):
        raise ConversionError("Generated segments do not cover the full trajectory")


def validate_with_recoil_trainer(profile: dict[str, Any], project_root: Path) -> None:
    root = project_root.resolve()
    model_path = root / "src" / "core" / "domain" / "models.py"
    if not model_path.is_file():
        raise ConversionError(f"Recoil Trainer model was not found: {model_path}")

    poetry = shutil.which("poetry")
    if poetry is None:
        raise ConversionError("Poetry is required for Recoil Trainer validation")
    runner = (
        "import json,os;"
        "payload=json.loads(os.environ.pop('RECOIL_PROFILE_PAYLOAD'));"
        "from src.core.domain.models import WeaponProfile;"
        "from src.core.workshop.workshop_validator import validate_workshop_payload;"
        "profile=WeaponProfile.from_dict(payload);"
        "round_trip=profile.to_dict();"
        "result=validate_workshop_payload(payload);"
        "assert result.ok,result.summary();"
        "assert len(round_trip.get('points',[]))==len(payload['points']);"
        "assert round_trip.get('schema_version')==1;"
        "print('RECOIL_PROFILE_VALID=1')"
    )
    environment = dict(os.environ)
    environment["RECOIL_PROFILE_PAYLOAD"] = json.dumps(profile, separators=(",", ":"))
    try:
        result = subprocess.run(
            [poetry, "run", "python", "-c", runner],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConversionError(
            f"Cannot run Recoil Trainer profile validation: {exc}"
        ) from exc
    if result.returncode != 0 or "RECOIL_PROFILE_VALID=1" not in result.stdout:
        detail = (result.stderr or result.stdout).strip()
        raise ConversionError(
            "Recoil Trainer rejected the generated Workshop JSON: " + detail
        )


def convert(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float, dict[str, float], float | None, str]:
    input_csv = args.input_csv.resolve()
    rows = load_reconstruction(input_csv)
    summary = load_summary(input_csv)
    fps = resolve_fps(summary, args.fps)
    timestamps = make_timestamps(rows, fps)
    coordinates, scale, bounds = normalize_coordinates(
        rows, args.target_vertical_span, args.coordinate_margin
    )
    recorded_pitch_deg, pitch_source = resolve_recorded_pitch_deg(rows, summary, args)
    if args.segmentation == "recoil-trainer":
        segments = recoil_trainer_auto_segments(
            coordinates,
            timestamps,
            args.recoil_trainer_root,
            args.smoothing,
            args.smoothing_strength,
        )
    else:
        segments = [
            {"segment_id": "Seq1", "start_shot": 0, "end_shot": len(rows) - 1}
        ]
    core_profile = build_profile(
        rows,
        timestamps,
        coordinates,
        recorded_pitch_deg,
        segments,
        args,
    )
    validate_profile_shape(core_profile)
    profile = build_workshop_payload(core_profile, args)
    profile["data"]["reconstruction"]["coordinate_source"] = (
        "trainer_recoil_*_px"
        if summary.get("trajectory_corrections")
        else "recoil_*_px"
    )
    profile["data"]["reconstruction"]["trajectory_corrections"] = list(
        summary.get("trajectory_corrections", [])
    )
    if not args.skip_trainer_validation:
        validate_with_recoil_trainer(profile, args.recoil_trainer_root)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return profile, scale, bounds, fps, pitch_source


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        profile, scale, bounds, fps, pitch_source = convert(args)
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    points = profile["points"]
    duration_ms = points[-1]["t_ms"]
    source_x_span = bounds["source_max_x"] - bounds["source_min_x"]
    source_y_span = bounds["source_max_y"] - bounds["source_min_y"]
    validation = "skipped" if args.skip_trainer_validation else "passed"
    print(f"Wrote: {args.output.resolve()}")
    print(f"Profile: {profile['name']} ({profile['profile_id']})")
    print(f"Shots: {len(points)}, duration: {duration_ms} ms, FPS: {fps or 'time_sec'}")
    print(
        f"Source span: x={source_x_span:.3f}px, y={source_y_span:.3f}px; "
        f"uniform scale={scale:.9f}"
    )
    print(
        "Recorded pitch: "
        f"{profile['recorded_recoil_pitch_range_deg']:.6f} deg ({pitch_source})"
    )
    print(
        f"Smoothing: {profile['smoothing']} {profile['smoothing_strength']:.3f}; "
        f"segments: {len(profile['segments'])} ({args.segmentation})"
    )
    print(f"Recoil Trainer model validation: {validation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
