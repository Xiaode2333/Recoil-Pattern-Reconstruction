#!/usr/bin/env python3
"""Recover recoil from an OBS video pair and synchronized Raw Input sidecars.

The baseline take contains mouse movement without firing.  It calibrates raw
mouse counts against background motion measured in the video.  The firing take
is then decomposed into:

    observed camera motion = mouse-induced motion + weapon recoil

The OBS plugin's ``output_frame_count`` is used as the common timeline.  The
script also accounts for the field-of-view change between hip fire and ADS by
estimating the zoom from background features in the firing take.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


LEFT_DOWN = 0x0001
LEFT_UP = 0x0002
RIGHT_DOWN = 0x0004
RIGHT_UP = 0x0008


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


@dataclass
class ButtonInterval:
    start_output_frame: int
    end_output_frame: int
    start_session_ns: int
    end_session_ns: int

    @property
    def duration_frames(self) -> int:
        return self.end_output_frame - self.start_output_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a no-fire take and a firing take to recover mouse-compensated recoil."
    )
    parser.add_argument("baseline_video", type=Path)
    parser.add_argument("recoil_video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("synced_recoil_output"))
    parser.add_argument("--shot-count", type=int, default=30)
    parser.add_argument(
        "--reference-mode",
        choices=("hip", "ads"),
        default="hip",
        help="Use 'ads' when the no-fire reference was recorded while aiming down sights",
    )
    parser.add_argument("--baseline-start-frame", type=int, default=120)
    parser.add_argument(
        "--baseline-end-trim", type=int, default=120,
        help="Frames trimmed from the end of the baseline take",
    )
    parser.add_argument("--motion-scale", type=float, default=0.375)
    parser.add_argument("--max-shift", type=int, default=10)
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        fail(f"Cannot open video: {path}")
    info = VideoInfo(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    if info.fps <= 0 or info.frames <= 1:
        fail(f"Invalid video metadata: {path}")
    return info


def locate_sidecar_base(video: Path) -> Path:
    direct = video.with_suffix("")
    if Path(str(direct) + ".mouse.csv").exists():
        return direct

    parts = video.stem.split("_")
    for count in range(len(parts) - 1, 0, -1):
        candidate = video.parent / "_".join(parts[:count])
        if Path(str(candidate) + ".mouse.csv").exists():
            return candidate

    candidates = []
    for path in video.parent.glob("*.mouse-session.json"):
        stem = path.name[: -len(".mouse-session.json")]
        if video.stem.startswith(stem):
            candidates.append(video.parent / stem)
    if candidates:
        return max(candidates, key=lambda item: len(item.name))
    fail(f"No OBS mouse sidecars found for {video}")


def load_session(base: Path) -> dict[str, object]:
    with Path(str(base) + ".mouse-session.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_mouse_rows(base: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with Path(str(base) + ".mouse.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        for source in csv.DictReader(stream):
            rows.append(
                {
                    "session_time_ns": int(source["session_time_ns"]),
                    "output_frame_count": int(source["output_frame_count"]),
                    "dx": int(source["dx_counts"]),
                    "dy": int(source["dy_counts"]),
                    "movement_flags": int(source["movement_flags"]),
                    "button_flags": int(source["button_flags"]),
                }
            )
    if not rows:
        fail(f"Mouse CSV is empty: {base}")
    absolute = sum(row["movement_flags"] != 0 for row in rows)
    if absolute:
        fail(f"Expected relative Raw Input, found {absolute} absolute packets in {base}")
    return rows


def button_intervals(
    rows: Sequence[dict[str, int]], down_flag: int, up_flag: int
) -> list[ButtonInterval]:
    intervals: list[ButtonInterval] = []
    start: dict[str, int] | None = None
    for row in rows:
        flags = row["button_flags"]
        if flags & down_flag:
            start = row
        if flags & up_flag and start is not None:
            intervals.append(
                ButtonInterval(
                    start_output_frame=start["output_frame_count"],
                    end_output_frame=row["output_frame_count"],
                    start_session_ns=start["session_time_ns"],
                    end_session_ns=row["session_time_ns"],
                )
            )
            start = None
    return intervals


def main_interval(rows: Sequence[dict[str, int]], down: int, up: int) -> ButtonInterval:
    intervals = button_intervals(rows, down, up)
    if not intervals:
        fail("No complete button interval was found")
    return max(intervals, key=lambda item: item.duration_frames)


def aggregate_mouse(rows: Sequence[dict[str, int]], length: int) -> np.ndarray:
    values = np.zeros((length, 2), dtype=np.float64)
    for row in rows:
        index = row["output_frame_count"]
        if 0 <= index < length:
            values[index, 0] += row["dx"]
            values[index, 1] += row["dy"]
    return values


def motion_mask(width: int, height: int, ads: bool) -> np.ndarray:
    mask = np.full((height, width), 255, dtype=np.uint8)
    mask[: int(0.14 * height), :] = 0
    mask[int(0.69 * height) :, :] = 0
    mask[: int(0.48 * height), : int(0.26 * width)] = 0
    mask[int(0.28 * height) : int(0.67 * height), int(0.84 * width) :] = 0
    if ads:
        mask[
            int(0.24 * height) : int(0.69 * height),
            int(0.34 * width) : int(0.66 * width),
        ] = 0
    return mask


def _interpolate_missing(values: np.ndarray) -> tuple[np.ndarray, int]:
    result = values.copy()
    missing = ~np.isfinite(result)
    count = int(np.count_nonzero(missing))
    valid = np.flatnonzero(~missing)
    if not len(valid):
        fail("Every camera-motion estimate failed")
    indices = np.arange(len(result))
    result[missing] = np.interp(indices[missing], valid, result[valid])
    return result, count


def estimate_camera_motion(
    video: Path,
    info: VideoInfo,
    start_frame: int,
    end_frame: int,
    scale: float,
    ads: bool,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, float | int]]:
    if not 0.2 <= scale <= 1.0:
        fail("motion-scale must be between 0.2 and 1.0")
    start_frame = max(0, start_frame)
    end_frame = min(info.frames - 1, end_frame)
    if end_frame <= start_frame:
        fail("Invalid motion-analysis frame range")

    small_width = max(320, int(round(info.width * scale)))
    small_height = max(180, int(round(info.height * scale)))
    actual_scale_x = small_width / info.width
    actual_scale_y = small_height / info.height
    full_mask = motion_mask(info.width, info.height, ads)
    mask = cv2.resize(full_mask, (small_width, small_height), interpolation=cv2.INTER_NEAREST)

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, previous_frame = cap.read()
    if not ok:
        cap.release()
        fail(f"Cannot read frame {start_frame} from {video}")
    previous = cv2.cvtColor(
        cv2.resize(previous_frame, (small_width, small_height), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )

    motion = np.full((info.frames, 2), np.nan, dtype=np.float64)
    motion[start_frame] = 0.0
    rows: list[dict[str, object]] = [
        {
            "frame": start_frame,
            "background_dx_px": 0.0,
            "background_dy_px": 0.0,
            "tracked_points": 0,
            "inliers": 0,
            "inlier_ratio": 1.0,
            "median_error_px": 0.0,
            "status": "start",
        }
    ]
    reliable_inliers: list[int] = []
    reliable_ratios: list[float] = []

    for frame_index in range(start_frame + 1, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        current = cv2.cvtColor(
            cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        points0 = cv2.goodFeaturesToTrack(
            previous,
            mask=mask,
            maxCorners=700,
            qualityLevel=0.008,
            minDistance=7,
            blockSize=7,
        )
        status_name = "failed"
        tracked = inliers = 0
        inlier_ratio = 0.0
        median_error = math.nan
        dx = dy = math.nan
        if points0 is not None and len(points0) >= 24:
            points1, status1, _ = cv2.calcOpticalFlowPyrLK(
                previous,
                current,
                points0,
                None,
                winSize=(25, 25),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if points1 is not None and status1 is not None:
                points0_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
                    current,
                    previous,
                    points1,
                    None,
                    winSize=(25, 25),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if points0_back is not None and status_back is not None:
                    forward_backward = np.linalg.norm(
                        points0.reshape(-1, 2) - points0_back.reshape(-1, 2), axis=1
                    )
                    keep = (
                        status1.reshape(-1).astype(bool)
                        & status_back.reshape(-1).astype(bool)
                        & (forward_backward < 1.5)
                    )
                    source = points0.reshape(-1, 2)[keep]
                    target = points1.reshape(-1, 2)[keep]
                    tracked = len(source)
                    if tracked >= 20:
                        affine, inlier_mask = cv2.estimateAffinePartial2D(
                            source,
                            target,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=1.5,
                            maxIters=3000,
                            confidence=0.999,
                            refineIters=20,
                        )
                        if affine is not None and inlier_mask is not None:
                            selected = inlier_mask.reshape(-1).astype(bool)
                            inliers = int(np.count_nonzero(selected))
                            inlier_ratio = inliers / tracked
                            a, b = float(affine[0, 0]), float(affine[1, 0])
                            affine_scale = math.hypot(a, b)
                            center = np.array([small_width / 2.0, small_height / 2.0])
                            center_after = affine[:, :2] @ center + affine[:, 2]
                            step = center_after - center
                            projected = source[selected] @ affine[:, :2].T + affine[:, 2]
                            errors = np.linalg.norm(projected - target[selected], axis=1)
                            median_error = float(np.median(errors)) if len(errors) else math.nan
                            dx = float(step[0] / actual_scale_x)
                            dy = float(step[1] / actual_scale_y)
                            if (
                                inliers >= 18
                                and inlier_ratio >= 0.35
                                and 0.965 <= affine_scale <= 1.035
                                and math.hypot(dx, dy) <= 140.0
                            ):
                                status_name = "ok"
                                motion[frame_index] = (dx, dy)
                                reliable_inliers.append(inliers)
                                reliable_ratios.append(inlier_ratio)
        rows.append(
            {
                "frame": frame_index,
                "background_dx_px": dx,
                "background_dy_px": dy,
                "tracked_points": tracked,
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "median_error_px": median_error,
                "status": status_name,
            }
        )
        previous = current
        if (frame_index - start_frame) % 300 == 0:
            print(f"  camera motion: {frame_index - start_frame}/{end_frame - start_frame}", flush=True)
    cap.release()

    analyzed = motion[start_frame : end_frame + 1]
    for axis in range(2):
        analyzed[:, axis], _ = _interpolate_missing(analyzed[:, axis])
    motion[start_frame : end_frame + 1] = analyzed
    failed = sum(row["status"] == "failed" for row in rows)
    quality: dict[str, float | int] = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "analyzed_steps": end_frame - start_frame,
        "failed_steps_interpolated": failed,
        "failure_ratio": failed / max(1, end_frame - start_frame),
        "median_inliers": float(np.median(reliable_inliers)) if reliable_inliers else 0.0,
        "median_inlier_ratio": float(np.median(reliable_ratios)) if reliable_ratios else 0.0,
    }
    return motion, rows, quality


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64)
    return np.column_stack(
        [np.convolve(values[:, axis], kernel, mode="same") for axis in range(values.shape[1])]
    )


def robust_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    keep = np.isfinite(x) & np.isfinite(y) & (np.abs(x) >= 1.0)
    x = x[keep]
    y = y[keep]
    if len(x) < 20:
        return math.nan, -math.inf, len(x)
    slope = float(np.dot(x, y) / max(1e-12, np.dot(x, x)))
    for _ in range(8):
        residual = y - slope * x
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-6
        weights = np.minimum(1.0, (2.5 * scale) / np.maximum(np.abs(residual), 1e-9))
        denominator = float(np.dot(weights * x, x))
        if denominator <= 1e-12:
            break
        slope = float(np.dot(weights * x, y) / denominator)
    residual = y - slope * x
    centered = y - float(np.median(y))
    r2 = 1.0 - float(np.dot(residual, residual)) / max(1e-12, float(np.dot(centered, centered)))
    return slope, r2, len(x)


def calibrate_mouse(
    camera: np.ndarray,
    mouse: np.ndarray,
    reliable: np.ndarray,
    start: int,
    end: int,
    max_shift: int,
) -> dict[str, float | int]:
    window = 9
    camera_window = rolling_sum(camera, window)
    mouse_window = rolling_sum(mouse, window)
    reliable_window = np.convolve(
        reliable.astype(np.int32), np.ones(window, dtype=np.int32), mode="same"
    ) == window
    best: dict[str, float | int] | None = None
    for shift in range(-max_shift, max_shift + 1):
        frames = np.arange(start, end + 1)
        mouse_indices = frames + shift
        valid = (mouse_indices >= 0) & (mouse_indices < len(mouse_window))
        frames = frames[valid]
        mouse_indices = mouse_indices[valid]
        trustworthy = reliable_window[frames]
        frames = frames[trustworthy]
        mouse_indices = mouse_indices[trustworthy]
        selected_mouse = mouse_window[mouse_indices]
        selected_camera = camera_window[frames]

        horizontal = np.abs(selected_mouse[:, 0]) >= np.maximum(1.0, 1.5 * np.abs(selected_mouse[:, 1]))
        vertical = np.abs(selected_mouse[:, 1]) >= np.maximum(1.0, 1.5 * np.abs(selected_mouse[:, 0]))
        slope_x, r2_x, samples_x = robust_slope(
            selected_mouse[horizontal, 0], selected_camera[horizontal, 0]
        )
        slope_y, r2_y, samples_y = robust_slope(
            selected_mouse[vertical, 1], selected_camera[vertical, 1]
        )
        score = r2_x + r2_y
        candidate: dict[str, float | int] = {
            "mouse_frame_shift": shift,
            "background_dx_px_per_mouse_count": slope_x,
            "background_dy_px_per_mouse_count": slope_y,
            "horizontal_r2": r2_x,
            "vertical_r2": r2_y,
            "horizontal_samples": samples_x,
            "vertical_samples": samples_y,
            "score": score,
        }
        if best is None or float(candidate["score"]) > float(best["score"]):
            best = candidate
    assert best is not None
    if not math.isfinite(float(best["background_dx_px_per_mouse_count"])):
        fail("Horizontal mouse calibration failed")
    if not math.isfinite(float(best["background_dy_px_per_mouse_count"])):
        fail("Vertical mouse calibration failed")
    return best


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        fail(f"Cannot read frame {frame_index}: {path}")
    return frame


def estimate_ads_zoom(
    video: Path, info: VideoInfo, hip_frame: int, ads_frame: int
) -> dict[str, float | int]:
    hip = read_frame(video, hip_frame)
    ads = read_frame(video, ads_frame)
    scale = 0.5
    size = (int(round(info.width * scale)), int(round(info.height * scale)))
    hip_small = cv2.resize(hip, size, interpolation=cv2.INTER_AREA)
    ads_small = cv2.resize(ads, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(motion_mask(info.width, info.height, True), size, interpolation=cv2.INTER_NEAREST)
    detector = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.015, edgeThreshold=15)
    key0, descriptors0 = detector.detectAndCompute(cv2.cvtColor(hip_small, cv2.COLOR_BGR2GRAY), mask)
    key1, descriptors1 = detector.detectAndCompute(cv2.cvtColor(ads_small, cv2.COLOR_BGR2GRAY), mask)
    if descriptors0 is None or descriptors1 is None:
        fail("Could not extract ADS zoom features")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors0, descriptors1, k=2)
    good = [first for first, second in pairs if first.distance < 0.78 * second.distance]
    if len(good) < 30:
        fail("Too few ADS zoom matches")
    source = np.float32([key0[item.queryIdx].pt for item in good])
    target = np.float32([key1[item.trainIdx].pt for item in good])
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=8000,
        confidence=0.999,
        refineIters=30,
    )
    if affine is None or inlier_mask is None:
        fail("ADS zoom transform could not be estimated")
    inliers = int(np.count_nonzero(inlier_mask))
    zoom = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
    if not 0.8 <= zoom <= 2.0 or inliers < 25:
        fail(f"Implausible ADS zoom estimate: zoom={zoom:.4f}, inliers={inliers}")
    return {
        "hip_frame": hip_frame,
        "ads_frame": ads_frame,
        "zoom_scale": zoom,
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": inliers / len(good),
    }


def ammo_change_scores(
    video: Path, info: VideoInfo, start_frame: int, end_frame: int
) -> tuple[np.ndarray, int]:
    # Rainbow Six HUD current-magazine digits at the 2560x1440 reference size.
    x0, x1 = int(round(0.875 * info.width)), int(round(0.947 * info.width))
    y0, y1 = int(round(0.835 * info.height)), int(round(0.915 * info.height))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    features: list[np.ndarray] = []
    decoded = 0
    for _frame in range(start_frame, end_frame + 1):
        ok, image = cap.read()
        if not ok:
            break
        crop = image[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        blue, green, red = cv2.split(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark = gray < 72
        red_mask = (
            (red > 90)
            & (red.astype(np.int16) > green.astype(np.int16) + 22)
            & (red.astype(np.int16) > blue.astype(np.int16) + 22)
        )
        feature = (dark | red_mask).astype(np.uint8) * 255
        feature = cv2.morphologyEx(feature, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        feature = cv2.resize(feature, (96, 64), interpolation=cv2.INTER_NEAREST)
        features.append(feature.astype(np.float32) / 255.0)
        decoded += 1
    cap.release()
    if decoded < 3:
        fail("Could not decode firing HUD frames")
    array = np.asarray(features)
    differences = np.sqrt(np.mean(np.square(array[1:] - array[:-1]), axis=(1, 2)))
    scores = np.zeros(decoded, dtype=np.float64)
    scores[1:] = differences
    return scores, decoded


def detect_regular_shots(
    scores: np.ndarray,
    absolute_start: int,
    expected_count: int,
    first_min: int,
    first_max: int,
    fps: float,
) -> tuple[list[int], dict[str, float]]:
    best_score = -math.inf
    best_frames: list[int] = []
    best_period = math.nan
    for period in np.linspace(8.0, 12.5, 181):
        for first in np.linspace(first_min, first_max, (first_max - first_min) * 4 + 1):
            predicted = first + period * np.arange(expected_count)
            if predicted[-1] >= absolute_start + len(scores) - 2:
                continue
            chosen: list[int] = []
            strengths: list[float] = []
            for center in predicted:
                local_center = int(round(center)) - absolute_start
                left = max(1, local_center - 2)
                right = min(len(scores), local_center + 3)
                if right <= left:
                    break
                local = left + int(np.argmax(scores[left:right]))
                chosen.append(absolute_start + local)
                strengths.append(float(scores[local]))
            if len(chosen) != expected_count or any(b <= a for a, b in zip(chosen, chosen[1:])):
                continue
            intervals = np.diff(chosen)
            cadence_penalty = 0.15 * float(np.std(intervals))
            value = float(np.sum(strengths)) - cadence_penalty
            if value > best_score:
                best_score = value
                best_frames = chosen
                best_period = period
    if len(best_frames) != expected_count:
        fail("Could not find the requested regular shot sequence")
    interval = float(np.median(np.diff(best_frames)))
    return best_frames, {
        "grid_period_frames": best_period,
        "median_interval_frames": interval,
        "mean_interval_frames": float(np.mean(np.diff(best_frames))),
        "interval_std_frames": float(np.std(np.diff(best_frames))),
        "estimated_rpm": 60.0 * fps / float(np.mean(np.diff(best_frames))),
        "mean_change_score": float(np.mean([scores[frame - absolute_start] for frame in best_frames])),
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def draw_trajectory(rows: Sequence[dict[str, object]], path: Path) -> None:
    width = height = 1100
    margin = 110
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    x = np.array([float(row["recoil_x_right_px"]) for row in rows])
    y = np.array([float(row["recoil_y_up_px"]) for row in rows])
    x_min, x_max = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
    y_min, y_max = min(float(y.min()), 0.0), max(float(y.max()), 0.0)
    span_x = max(10.0, x_max - x_min)
    span_y = max(10.0, y_max - y_min)
    plot_scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)

    def point(x_value: float, y_value: float) -> tuple[int, int]:
        px = margin + (x_value - x_min) * plot_scale
        py = height - margin - (y_value - y_min) * plot_scale
        return int(round(px)), int(round(py))

    zero = point(0.0, 0.0)
    cv2.line(canvas, (margin, zero[1]), (width - margin, zero[1]), (205, 205, 205), 2)
    cv2.line(canvas, (zero[0], margin), (zero[0], height - margin), (205, 205, 205), 2)
    points = np.array([point(float(a), float(b)) for a, b in zip(x, y)], dtype=np.int32)
    cv2.polylines(canvas, [points], False, (36, 105, 214), 4, cv2.LINE_AA)
    for index, location in enumerate(points):
        color = (40, 160, 50) if index == 0 else (36, 105, 214)
        cv2.circle(canvas, tuple(location), 8, color, -1, cv2.LINE_AA)
        if index == 0 or (index + 1) % 5 == 0 or index + 1 == len(points):
            cv2.putText(
                canvas,
                str(index + 1),
                (int(location[0]) + 10, int(location[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (35, 35, 35),
                2,
                cv2.LINE_AA,
            )
    cv2.putText(canvas, "Mouse-compensated recoil: right / up", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "1", (zero[0] + 12, zero[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 120, 40), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        fail(f"Could not write {path}")


def main() -> int:
    args = parse_args()
    baseline_video = args.baseline_video.resolve()
    recoil_video = args.recoil_video.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_info = video_info(baseline_video)
    recoil_info = video_info(recoil_video)
    if (baseline_info.width, baseline_info.height) != (recoil_info.width, recoil_info.height):
        fail("The two videos have different resolutions")
    if abs(baseline_info.fps - recoil_info.fps) > 0.01:
        fail("The two videos have different frame rates")

    baseline_base = locate_sidecar_base(baseline_video)
    recoil_base = locate_sidecar_base(recoil_video)
    baseline_session = load_session(baseline_base)
    recoil_session = load_session(recoil_base)
    baseline_rows = load_mouse_rows(baseline_base)
    recoil_rows = load_mouse_rows(recoil_base)
    if int(baseline_session.get("output_dropped_frames", -1)) != 0:
        fail("Baseline OBS recording reports dropped frames")
    if int(recoil_session.get("output_dropped_frames", -1)) != 0:
        fail("Recoil OBS recording reports dropped frames")

    left_hold = main_interval(recoil_rows, LEFT_DOWN, LEFT_UP)
    right_hold = main_interval(recoil_rows, RIGHT_DOWN, RIGHT_UP)
    baseline_mouse = aggregate_mouse(baseline_rows, baseline_info.frames + 64)
    recoil_mouse = aggregate_mouse(recoil_rows, recoil_info.frames + 64)

    baseline_start = max(1, args.baseline_start_frame)
    baseline_end = baseline_info.frames - 1 - args.baseline_end_trim
    print(f"[1/5] Baseline camera motion {baseline_start}-{baseline_end} ...", flush=True)
    baseline_motion, baseline_motion_rows, baseline_quality = estimate_camera_motion(
        baseline_video,
        baseline_info,
        baseline_start,
        baseline_end,
        args.motion_scale,
        ads=False,
    )
    baseline_reliable = np.zeros(baseline_info.frames, dtype=bool)
    for row in baseline_motion_rows:
        if row["status"] in ("ok", "start"):
            baseline_reliable[int(row["frame"])] = True
    calibration = calibrate_mouse(
        baseline_motion,
        baseline_mouse,
        baseline_reliable,
        baseline_start,
        baseline_end,
        args.max_shift,
    )
    print(
        "[2/5] Mouse calibration: "
        f"dx={float(calibration['background_dx_px_per_mouse_count']):.5f}, "
        f"dy={float(calibration['background_dy_px_per_mouse_count']):.5f}, "
        f"shift={calibration['mouse_frame_shift']}",
        flush=True,
    )

    hip_frame = max(0, right_hold.start_output_frame - 25)
    ads_frame = max(hip_frame + 1, left_hold.start_output_frame - 40)
    if args.reference_mode == "ads":
        zoom: dict[str, float | int | str] = {
            "method": "same-view reference",
            "hip_frame": hip_frame,
            "ads_frame": ads_frame,
            "zoom_scale": 1.0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 1.0,
        }
    else:
        zoom = {
            "method": "cross-view feature estimate",
            **estimate_ads_zoom(recoil_video, recoil_info, hip_frame, ads_frame),
        }
    zoom_scale = float(zoom["zoom_scale"])
    ads_dx_per_count = float(calibration["background_dx_px_per_mouse_count"]) * zoom_scale
    ads_dy_per_count = float(calibration["background_dy_px_per_mouse_count"]) * zoom_scale
    print(f"[3/5] Hip-to-ADS zoom scale: {zoom_scale:.5f}", flush=True)

    fire_start = max(1, left_hold.start_output_frame - 20)
    fire_end = min(recoil_info.frames - 1, left_hold.end_output_frame + 10)
    print(f"[4/5] Firing camera motion {fire_start}-{fire_end} ...", flush=True)
    recoil_motion, recoil_motion_rows, recoil_quality = estimate_camera_motion(
        recoil_video,
        recoil_info,
        fire_start,
        fire_end,
        args.motion_scale,
        ads=True,
    )

    score_start = max(0, left_hold.start_output_frame - 10)
    score_end = min(recoil_info.frames - 1, left_hold.end_output_frame - 10)
    scores, _ = ammo_change_scores(recoil_video, recoil_info, score_start, score_end)
    shot_frames, cadence = detect_regular_shots(
        scores,
        score_start,
        args.shot_count,
        first_min=left_hold.start_output_frame,
        first_max=min(left_hold.start_output_frame + 45, score_end),
        fps=recoil_info.fps,
    )

    shift = int(calibration["mouse_frame_shift"])
    frame_rows: list[dict[str, object]] = []
    cumulative_observed_x = cumulative_observed_y = 0.0
    cumulative_mouse_x = cumulative_mouse_y = 0.0
    cumulative_recoil_x = cumulative_recoil_y = 0.0
    cumulative_by_frame: dict[int, tuple[float, float, float, float, float, float]] = {}
    for frame in range(fire_start, fire_end + 1):
        observed_bg_dx = float(recoil_motion[frame, 0])
        observed_bg_dy = float(recoil_motion[frame, 1])
        mouse_index = frame + shift
        mouse_dx = float(recoil_mouse[mouse_index, 0]) if 0 <= mouse_index < len(recoil_mouse) else 0.0
        mouse_dy = float(recoil_mouse[mouse_index, 1]) if 0 <= mouse_index < len(recoil_mouse) else 0.0
        predicted_bg_dx = ads_dx_per_count * mouse_dx
        predicted_bg_dy = ads_dy_per_count * mouse_dy
        residual_bg_dx = observed_bg_dx - predicted_bg_dx
        residual_bg_dy = observed_bg_dy - predicted_bg_dy

        observed_x = -observed_bg_dx
        observed_y = observed_bg_dy
        mouse_x = -predicted_bg_dx
        mouse_y = predicted_bg_dy
        recoil_x = -residual_bg_dx
        recoil_y = residual_bg_dy
        cumulative_observed_x += observed_x
        cumulative_observed_y += observed_y
        cumulative_mouse_x += mouse_x
        cumulative_mouse_y += mouse_y
        cumulative_recoil_x += recoil_x
        cumulative_recoil_y += recoil_y
        cumulative_by_frame[frame] = (
            cumulative_observed_x,
            cumulative_observed_y,
            cumulative_mouse_x,
            cumulative_mouse_y,
            cumulative_recoil_x,
            cumulative_recoil_y,
        )
        frame_rows.append(
            {
                "frame": frame,
                "time_sec": frame / recoil_info.fps,
                "mouse_output_frame": mouse_index,
                "mouse_dx_counts": mouse_dx,
                "mouse_dy_counts": mouse_dy,
                "observed_background_dx_px": observed_bg_dx,
                "observed_background_dy_px": observed_bg_dy,
                "predicted_mouse_background_dx_px": predicted_bg_dx,
                "predicted_mouse_background_dy_px": predicted_bg_dy,
                "recoil_step_x_right_px": recoil_x,
                "recoil_step_y_up_px": recoil_y,
                "cumulative_observed_x_right_px": cumulative_observed_x,
                "cumulative_observed_y_up_px": cumulative_observed_y,
                "cumulative_mouse_x_right_px": cumulative_mouse_x,
                "cumulative_mouse_y_up_px": cumulative_mouse_y,
                "cumulative_recoil_x_right_px": cumulative_recoil_x,
                "cumulative_recoil_y_up_px": cumulative_recoil_y,
            }
        )

    def sampled(frame: int) -> np.ndarray:
        values = [
            cumulative_by_frame[index]
            for index in range(frame - 1, frame + 2)
            if index in cumulative_by_frame
        ]
        return np.median(np.asarray(values, dtype=np.float64), axis=0)

    samples = [sampled(frame) for frame in shot_frames]
    origin = samples[0]
    shot_rows: list[dict[str, object]] = []
    previous_recoil_x = previous_recoil_y = 0.0
    for index, (frame, values) in enumerate(zip(shot_frames, samples), start=1):
        observed_x, observed_y, mouse_x, mouse_y, recoil_x, recoil_y = values - origin
        compensation_dx = recoil_x / ads_dx_per_count
        compensation_dy = -recoil_y / ads_dy_per_count
        row = {
            "shot": index,
            "video_frame": frame,
            "shot_time_ms": round((frame - shot_frames[0]) * 1000.0 / recoil_info.fps),
            "observed_x_right_px": observed_x,
            "observed_y_up_px": observed_y,
            "recorded_mouse_x_right_px": mouse_x,
            "recorded_mouse_y_up_px": mouse_y,
            "recoil_x_right_px": recoil_x,
            "recoil_y_up_px": recoil_y,
            "shot_delta_x_right_px": recoil_x - previous_recoil_x,
            "shot_delta_y_up_px": recoil_y - previous_recoil_y,
            "compensation_mouse_dx_counts": compensation_dx,
            "compensation_mouse_dy_counts": compensation_dy,
        }
        shot_rows.append(row)
        previous_recoil_x, previous_recoil_y = float(recoil_x), float(recoil_y)

    print("[5/5] Writing synchronized recoil outputs ...", flush=True)
    write_csv(output_dir / "baseline_camera_motion.csv", baseline_motion_rows, list(baseline_motion_rows[0]))
    write_csv(output_dir / "firing_camera_motion.csv", recoil_motion_rows, list(recoil_motion_rows[0]))
    write_csv(output_dir / "recoil_per_frame.csv", frame_rows, list(frame_rows[0]))
    write_csv(output_dir / "recoil_per_shot.csv", shot_rows, list(shot_rows[0]))
    draw_trajectory(shot_rows, output_dir / "recoil_trajectory.png")

    recoil_x_values = [float(row["recoil_x_right_px"]) for row in shot_rows]
    recoil_y_values = [float(row["recoil_y_up_px"]) for row in shot_rows]
    summary = {
        "pipeline": "obs-raw-input-paired-v1",
        "reference_mode": args.reference_mode,
        "baseline_video": baseline_video,
        "recoil_video": recoil_video,
        "video": {
            "width": recoil_info.width,
            "height": recoil_info.height,
            "fps": recoil_info.fps,
            "baseline_frames": baseline_info.frames,
            "recoil_frames": recoil_info.frames,
            "baseline_obs_dropped_frames": baseline_session["output_dropped_frames"],
            "recoil_obs_dropped_frames": recoil_session["output_dropped_frames"],
        },
        "firing_input_interval": {
            "left_down_output_frame": left_hold.start_output_frame,
            "left_up_output_frame": left_hold.end_output_frame,
            "duration_seconds": left_hold.duration_frames / recoil_info.fps,
            "right_down_output_frame": right_hold.start_output_frame,
            "right_up_output_frame": right_hold.end_output_frame,
        },
        "baseline_motion_quality": baseline_quality,
        "firing_motion_quality": recoil_quality,
        "mouse_calibration_hip_fire": calibration,
        "ads_zoom": zoom,
        "mouse_calibration_ads": {
            "background_dx_px_per_mouse_count": ads_dx_per_count,
            "background_dy_px_per_mouse_count": ads_dy_per_count,
        },
        "shot_detection": {
            "shot_count": len(shot_frames),
            "shot_frames": shot_frames,
            **cadence,
        },
        "recoil": {
            "horizontal_span_px": max(recoil_x_values) - min(recoil_x_values),
            "vertical_span_px": max(recoil_y_values) - min(recoil_y_values),
            "maximum_x_right_px": max(recoil_x_values),
            "minimum_x_right_px": min(recoil_x_values),
            "maximum_y_up_px": max(recoil_y_values),
            "minimum_y_up_px": min(recoil_y_values),
            "final_x_right_px": recoil_x_values[-1],
            "final_y_up_px": recoil_y_values[-1],
        },
        "formal_result_ready": (
            args.reference_mode == "ads"
            and float(calibration["horizontal_r2"]) >= 0.50
            and float(calibration["vertical_r2"]) >= 0.50
        ),
        "warnings": (
            ["The no-fire reference is hip fire; use this run only as a feasibility check and record an ADS reference for the formal result."]
            if args.reference_mode == "hip"
            else []
        ),
        "coordinate_convention": "x right positive; y up positive; compensation counts are the cumulative raw mouse counts needed to oppose recovered recoil",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(json_value(summary), stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"Done: {output_dir}")
    print(f"  shots: {shot_frames[0]}-{shot_frames[-1]}, {cadence['estimated_rpm']:.1f} RPM")
    print(f"  recoil span: x={summary['recoil']['horizontal_span_px']:.2f}px, y={summary['recoil']['vertical_span_px']:.2f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
