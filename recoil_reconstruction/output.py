from __future__ import annotations
import argparse,csv,json,math,sys
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Iterable,Sequence
import cv2
import numpy as np
REFERENCE_WIDTH=2560
REFERENCE_HEIGHT=1440
from .models import *
from .common import *
from .tracking import create_outside_scope_mask


def imwrite(path: Path, image: np.ndarray, quality: int | None = None) -> None:
    suffix = path.suffix.lower()
    params: list[int] = []
    if quality is not None and suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        fail(f"无法编码图像: {path}")
    encoded.tofile(str(path))


def feature_mask_preview(frame: np.ndarray, output: Path) -> None:
    height, width = frame.shape[:2]
    mask = create_outside_scope_mask(width, height)
    preview = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(
        mask, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    tint = preview.copy()
    tint[:, :, 1] = np.maximum(tint[:, :, 1], 150)
    preview[mask_small > 0] = cv2.addWeighted(
        preview[mask_small > 0], 0.62, tint[mask_small > 0], 0.38, 0
    )
    preview[mask_small == 0] = (preview[mask_small == 0] * 0.35).astype(np.uint8)
    cv2.putText(
        preview,
        "GREEN = outside-scope RANSAC feature area",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    imwrite(output, preview)


def reticle_debug_crop(
    frame: np.ndarray,
    result: ReticleResult,
    shot: int,
    frame_index: int,
    ammo_after: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    sx, sy = width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT
    half_width, half_height = int(round(250 * sx)), int(round(200 * sy))
    cx, cy = int(round(width / 2)), int(round(height / 2))
    x0, x1 = max(0, cx - half_width), min(width, cx + half_width)
    y0, y1 = max(0, cy - half_height), min(height, cy + half_height)
    crop = frame[y0:y1, x0:x1].copy()
    point = (int(round(result.x - x0)), int(round(result.y - y0)))
    cv2.drawMarker(crop, point, (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
    cv2.circle(crop, point, 12, (0, 255, 0), 2, cv2.LINE_AA)
    angle = math.radians(result.angle_deg)
    direction = (int(round(60 * math.cos(angle))), int(round(60 * math.sin(angle))))
    cv2.line(
        crop,
        (point[0] - direction[0], point[1] - direction[1]),
        (point[0] + direction[0], point[1] + direction[1]),
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    label = f"shot {shot:02d}  f={frame_index}  ammo={ammo_after:02d}  q={result.confidence:.1f}"
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(
        crop,
        label,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.resize(crop, (400, 320), interpolation=cv2.INTER_AREA)


def make_contact_sheet(images: Sequence[np.ndarray], columns: int = 5) -> np.ndarray:
    if not images:
        fail("没有可生成联系表的图像")
    tile_height, tile_width = images[0].shape[:2]
    rows = math.ceil(len(images) / columns)
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        sheet[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = image
    return sheet


def draw_trajectory(points: Sequence[dict[str, float | int]], output: Path) -> None:
    canvas_width, canvas_height = 1400, 1050
    margin = 90
    canvas = np.full((canvas_height, canvas_width, 3), (28, 31, 36), dtype=np.uint8)
    x_field = (
        "trainer_recoil_x_right_px"
        if "trainer_recoil_x_right_px" in points[0]
        else "recoil_x_right_px"
    )
    y_field = (
        "trainer_recoil_y_up_px"
        if "trainer_recoil_y_up_px" in points[0]
        else "recoil_y_up_px"
    )
    xs = np.array([float(point[x_field]) for point in points])
    ys = np.array([float(point[y_field]) for point in points])
    xs = np.concatenate(([0.0], xs))
    ys = np.concatenate(([0.0], ys))
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    if x_max - x_min < 1.0:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_max - y_min < 1.0:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    x_pad = max(5.0, 0.12 * (x_max - x_min))
    y_pad = max(5.0, 0.08 * (y_max - y_min))
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def project(x: float, y: float) -> tuple[int, int]:
        px = margin + (x - x_min) / (x_max - x_min) * (canvas_width - 2 * margin)
        py = canvas_height - margin - (y - y_min) / (y_max - y_min) * (
            canvas_height - 2 * margin
        )
        return int(round(px)), int(round(py))

    if x_min <= 0 <= x_max:
        x_axis, _ = project(0.0, 0.0)
        cv2.line(canvas, (x_axis, margin), (x_axis, canvas_height - margin), (70, 75, 82), 1)
    if y_min <= 0 <= y_max:
        _, y_axis = project(0.0, 0.0)
        cv2.line(canvas, (margin, y_axis), (canvas_width - margin, y_axis), (70, 75, 82), 1)

    projected = [project(float(x), float(y)) for x, y in zip(xs, ys)]
    for index in range(1, len(projected)):
        fraction = index / max(1, len(projected) - 1)
        color = (int(255 * (1 - fraction)), int(190 + 50 * fraction), int(255 * fraction))
        cv2.line(canvas, projected[index - 1], projected[index], color, 3, cv2.LINE_AA)
    cv2.circle(canvas, projected[0], 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "0",
        (projected[0][0] + 8, projected[0][1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for index, pixel in enumerate(projected[1:], start=1):
        cv2.circle(canvas, pixel, 5, (70, 235, 255), -1, cv2.LINE_AA)
        if index == 1 or index % 5 == 0 or index == len(projected) - 1:
            cv2.putText(
                canvas,
                str(index),
                (pixel[0] + 7, pixel[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        canvas,
        "Reconstructed recoil trajectory (outside-scope RANSAC + reticle jitter)",
        (margin, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "X: right (+)    Y: up (+)    labels: shot number",
        (margin, canvas_height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (190, 195, 205),
        1,
        cv2.LINE_AA,
    )
    imwrite(output, canvas)


def apply_terminal_empty_animation_repair(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Remove a severe last-point drop caused by the post-empty weapon animation.

    Raw reconstructed coordinates remain untouched. Separate ``trainer_*``
    columns contain the trajectory exported to Recoil Trainer, and the returned
    audit record makes every repair explicit in ``summary.json``.
    """
    for row in rows:
        row["trainer_recoil_x_right_px"] = float(row["recoil_x_right_px"])
        row["trainer_recoil_y_up_px"] = float(row["recoil_y_up_px"])
        row["trainer_shot_delta_x_right_px"] = float(row["shot_delta_x_right_px"])
        row["trainer_shot_delta_y_up_px"] = float(row["shot_delta_y_up_px"])
        row["trajectory_correction"] = ""
    if len(rows) < 10:
        return []

    deltas = np.array(
        [
            [float(row["shot_delta_x_right_px"]), float(row["shot_delta_y_up_px"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    magnitudes = np.linalg.norm(deltas, axis=1)
    reference = magnitudes[:-1]
    median_magnitude = float(np.median(reference))
    mad_magnitude = float(np.median(np.abs(reference - median_magnitude)))
    robust_sigma = max(1e-6, 1.4826 * mad_magnitude)
    terminal_dx, terminal_dy = (float(value) for value in deltas[-1])
    terminal_magnitude = float(magnitudes[-1])
    threshold = max(
        100.0,
        median_magnitude * 5.0,
        median_magnitude + 8.0 * robust_sigma,
    )
    if terminal_dy >= -80.0 or terminal_magnitude <= threshold:
        return []

    history = deltas[max(0, len(deltas) - 11) : -1]
    history_magnitudes = np.linalg.norm(history, axis=1)
    reliable = history[
        history_magnitudes <= median_magnitude + 3.0 * robust_sigma
    ]
    if len(reliable) < 3:
        reliable = history
    replacement = np.median(reliable, axis=0)
    replacement[1] = max(0.0, float(replacement[1]))
    replacement_magnitude = float(np.linalg.norm(replacement))
    maximum_replacement = max(1.0, median_magnitude * 2.5)
    if replacement_magnitude > maximum_replacement:
        replacement *= maximum_replacement / replacement_magnitude

    previous_x = float(rows[-2]["trainer_recoil_x_right_px"])
    previous_y = float(rows[-2]["trainer_recoil_y_up_px"])
    rows[-1]["trainer_recoil_x_right_px"] = previous_x + float(replacement[0])
    rows[-1]["trainer_recoil_y_up_px"] = previous_y + float(replacement[1])
    rows[-1]["trainer_shot_delta_x_right_px"] = float(replacement[0])
    rows[-1]["trainer_shot_delta_y_up_px"] = float(replacement[1])
    rows[-1]["trajectory_correction"] = "terminal-empty-animation-robust-extrapolation"
    return [
        {
            "shot": int(rows[-1]["shot"]),
            "reason": "terminal-empty-animation-robust-extrapolation",
            "raw_delta_x_right_px": terminal_dx,
            "raw_delta_y_up_px": terminal_dy,
            "raw_delta_magnitude_px": terminal_magnitude,
            "replacement_delta_x_right_px": float(replacement[0]),
            "replacement_delta_y_up_px": float(replacement[1]),
            "detection_threshold_px": threshold,
        }
    ]


def apply_reviewed_tail_extrapolation(
    rows: list[dict[str, object]], count: int
) -> list[dict[str, object]]:
    """Replace a reviewed unrecorded tail in trainer columns, preserving raw data."""
    if count <= 0:
        return []
    if count >= len(rows) - 2:
        fail("trainer-tail-extrapolation-count 过大")
    first = len(rows) - count
    reliable = rows[max(1, first - 12) : first]
    dx_values = [float(row["shot_delta_x_right_px"]) for row in reliable]
    positive_dy = [
        float(row["shot_delta_y_up_px"])
        for row in reliable
        if float(row["shot_delta_y_up_px"]) > 0
    ]
    if not positive_dy:
        fail("无法从可靠尾段得到向上的后坐力增量")
    replacement_dx = float(np.median(dx_values))
    replacement_dy = float(np.median(positive_dy))
    corrections: list[dict[str, object]] = []
    for index in range(first, len(rows)):
        previous = rows[index - 1]
        row = rows[index]
        row["trainer_recoil_x_right_px"] = float(previous["trainer_recoil_x_right_px"]) + replacement_dx
        row["trainer_recoil_y_up_px"] = float(previous["trainer_recoil_y_up_px"]) + replacement_dy
        row["trainer_shot_delta_x_right_px"] = replacement_dx
        row["trainer_shot_delta_y_up_px"] = replacement_dy
        row["trajectory_correction"] = "reviewed-unrecorded-tail-robust-extrapolation"
        corrections.append(
            {
                "shot": int(row["shot"]),
                "reason": "reviewed-unrecorded-tail-robust-extrapolation",
                "raw_delta_x_right_px": float(row["shot_delta_x_right_px"]),
                "raw_delta_y_up_px": float(row["shot_delta_y_up_px"]),
                "replacement_delta_x_right_px": replacement_dx,
                "replacement_delta_y_up_px": replacement_dy,
            }
        )
    return corrections


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
