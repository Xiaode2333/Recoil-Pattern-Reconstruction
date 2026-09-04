#!/usr/bin/env python3
"""Reconstruct a recoil trajectory with background motion and reticle tracking.

Coordinate model (column vectors):

    A_i : 第 i-1 帧的镜外背景像素 -> 第 i 帧
    C_i = C_(i-1) @ inv(A_i) : 第 i 帧 -> 第 start_frame 帧
    p_i = C_i @ [reticle_x_i, reticle_y_i, 1]

Each keyframe uses the detected reticle location rather than a fixed image
center, so the final trajectory combines background SE(2) motion with reticle
motion inside the frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


REFERENCE_WIDTH = 2560
REFERENCE_HEIGHT = 1440
DEFAULT_AMMO_TEMPLATE_FILE = Path(__file__).resolve().with_name("ammo_digit_templates.npz")


@dataclass
class StepResult:
    frame: int
    matrix: np.ndarray | None
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    median_error_px: float = math.nan
    ransac_scale: float = math.nan
    status: str = "failed"


@dataclass
class ReticleResult:
    x: float
    y: float
    angle_deg: float
    confidence: float
    horizontal_score: float
    vertical_score: float


@dataclass
class AmmoDetection:
    analysis_start_frame: int
    analysis_end_frame: int
    start_ammo: int
    shot_count: int
    cadence_frames: int
    event_threshold: float
    baseline_score: float
    approximate_keyframes: list[int]
    change_scores: np.ndarray
    decoded_frame_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "检测弹药变化关键帧；在瞄准镜外做 SIFT/ORB Feature Matching + "
            "RANSAC；在镜内检测刻度线交点；重建后坐力轨迹。"
        )
    )
    parser.add_argument(
        "video",
        help="input video path",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="手动模式的起始帧；省略四个手动参数时自动扫描全视频",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="手动模式的结束帧",
    )
    parser.add_argument(
        "--start-ammo",
        type=int,
        default=None,
        help="手动模式的初始弹药数",
    )
    parser.add_argument(
        "--shot-count",
        type=int,
        default=None,
        help="手动模式的射击数",
    )
    parser.add_argument(
        "--ammo-roi",
        default="2348,1218,2464,1268",
        help=(
            "弹药三位数字 ROI: x0,y0,x1,y1；默认值按 2560x1440 标定，"
            "其他分辨率会按比例缩放"
        ),
    )
    parser.add_argument(
        "--ammo-template-file",
        type=Path,
        default=DEFAULT_AMMO_TEMPLATE_FILE,
        help="自动模式三位弹药 OCR 模板（默认 ammo_digit_templates.npz）",
    )
    parser.add_argument(
        "--ammo-min-segment", type=int, default=4, help="一个弹药数最少持续帧数"
    )
    parser.add_argument(
        "--ammo-max-segment", type=int, default=22, help="一个弹药数最多持续帧数"
    )
    parser.add_argument(
        "--ammo-cadence-weight",
        type=float,
        default=10.0,
        help="手动模式弹药分段的射速稳定性权重",
    )
    parser.add_argument(
        "--auto-ammo-cadence-weight",
        type=float,
        default=1000.0,
        help="自动模式细化关键帧时的射速稳定性权重",
    )
    parser.add_argument(
        "--auto-ammo-threshold-mad",
        type=float,
        default=3.5,
        help="自动弹药变化阈值的 MAD 倍数",
    )
    parser.add_argument(
        "--auto-min-shots",
        type=int,
        default=5,
        help="自动识别接受的最少连续射击数",
    )
    parser.add_argument(
        "--detector", choices=("sift", "orb"), default="sift", help="镜外特征检测器"
    )
    parser.add_argument(
        "--feature-scale",
        type=float,
        default=0.5,
        help="特征匹配时的图像缩放比例；0.5 在本视频上速度/精度较均衡",
    )
    parser.add_argument("--max-features", type=int, default=2500)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=2.0)
    parser.add_argument("--min-inliers", type=int, default=16)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.40)
    parser.add_argument("--max-step-px", type=float, default=120.0)
    parser.add_argument("--max-step-angle", type=float, default=5.0)
    parser.add_argument(
        "--reticle-max-angle",
        type=float,
        default=3.0,
        help="刻度线相对水平线的最大搜索角度（度）",
    )
    parser.add_argument(
        "--reticle-angle-step", type=float, default=0.25, help="刻度线角度搜索步长（度）"
    )
    parser.add_argument(
        "--reticle-mode",
        choices=("tick-lines", "red-dot"),
        default="tick-lines",
        help="准星检测模式：2x 刻度线或 1x 红点（默认 tick-lines）",
    )
    parser.add_argument(
        "--output-dir", default="reconstruction_output", help="output directory"
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=104.0,
        help="游戏基础 FOV（默认 104 度）",
    )
    parser.add_argument(
        "--fov-axis",
        choices=("reference-horizontal", "horizontal", "vertical"),
        default="reference-horizontal",
        help="FOV convention used for the optional angular estimate",
    )
    parser.add_argument(
        "--scope-magnification",
        type=float,
        default=2.0,
        help="录制时瞄准镜倍率（默认 2x）",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_video_info(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        fail(f"无法打开视频: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frame_count


def seek_and_read(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        fail(f"无法读取第 {frame_index} 帧")
    return frame


def parse_scaled_roi(text: str, width: int, height: int) -> tuple[int, int, int, int]:
    try:
        values = [float(item.strip()) for item in text.split(",")]
    except ValueError as exc:
        fail(f"ROI 格式错误: {text!r} ({exc})")
    if len(values) != 4:
        fail(f"ROI 必须是 x0,y0,x1,y1: {text!r}")
    if max(values) <= 1.0:
        x0, y0, x1, y1 = (
            int(round(values[0] * width)),
            int(round(values[1] * height)),
            int(round(values[2] * width)),
            int(round(values[3] * height)),
        )
    else:
        sx = width / REFERENCE_WIDTH
        sy = height / REFERENCE_HEIGHT
        x0, y0, x1, y1 = (
            int(round(values[0] * sx)),
            int(round(values[1] * sy)),
            int(round(values[2] * sx)),
            int(round(values[3] * sy)),
        )
    x0, x1 = sorted((max(0, x0), min(width, x1)))
    y0, y1 = sorted((max(0, y0), min(height, y1)))
    if x1 - x0 < 30 or y1 - y0 < 20:
        fail(f"ROI 太小或越界: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def normalized_ammo_feature(crop: np.ndarray) -> np.ndarray:
    """把三位数字变成对亮度/红色低弹药提示较不敏感的梯度特征。"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    # 默认 ROI 是 116px 宽，覆盖完整三位数字，不包含右侧弹种标签。
    ranges = (
        (0.05, 0.37),
        (0.38, 0.69),
        (0.70, 0.99),
    )
    cells: list[np.ndarray] = []
    for left, right in ranges:
        x0 = max(0, int(round(left * width)))
        x1 = min(width, int(round(right * width)))
        cell = gray[:, x0:x1]
        if cell.size == 0:
            continue
        cell = (cell - float(cell.mean())) / (float(cell.std()) + 10.0)
        cells.append(cell)
    if len(cells) != 3:
        fail("弹药 ROI 无法分成三个数字单元")
    joined = np.concatenate(cells, axis=1)
    joined = cv2.resize(joined, (48, 25), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(joined, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(joined, cv2.CV_32F, 0, 1, ksize=3)
    return np.concatenate((gx, gy), axis=1).reshape(-1)


def ammo_display_feature(crop: np.ndarray) -> np.ndarray:
    """提取当前弹匣两位数字的颜色不敏感字形，用于全片自动检测。

    正常弹药数字是低饱和亮色，低弹药数字会变红。逐帧选择白色或红色
    字形，可屏蔽大部分黄色火花、枪体和白色烟雾造成的短暂干扰。
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(crop)
    _hue, saturation, value = cv2.split(hsv)
    white = (value > 140) & (saturation < 70)
    red_i16 = red.astype(np.int16)
    red_mask = (
        (red > 90)
        & (red_i16 > green.astype(np.int16) + 20)
        & (red_i16 > blue.astype(np.int16) + 20)
    )

    height, width = white.shape
    current_x0 = int(round(0.37 * width))
    current_white = white[:, current_x0:]
    current_red = red_mask[:, current_x0:]
    # 低弹药状态的红色笔画足够多时只保留红色，避免白色烟雾穿过 HUD。
    selected = current_red if int(current_red.sum()) > max(30, height * 2) else current_white
    selected = cv2.resize(
        selected.astype(np.uint8),
        (66, 50),
        interpolation=cv2.INTER_NEAREST,
    )
    return selected.reshape(-1)


def ammo_ocr_cells(crop: np.ndarray) -> np.ndarray:
    """Return three aligned binary digit cells for template OCR."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(crop)
    _hue, saturation, value = cv2.split(hsv)
    white = (value > 140) & (saturation < 70)
    red_i16 = red.astype(np.int16)
    red_mask = (
        (red > 90)
        & (red_i16 > green.astype(np.int16) + 20)
        & (red_i16 > blue.astype(np.int16) + 20)
    )
    mask = white | red_mask
    width = mask.shape[1]
    ranges = ((0.05, 0.37), (0.38, 0.69), (0.70, 0.99))
    cells = [
        cv2.resize(
            mask[:, int(round(left * width)) : int(round(right * width))].astype(
                np.float32
            ),
            (36, 50),
            interpolation=cv2.INTER_NEAREST,
        )
        for left, right in ranges
    ]
    return np.asarray(cells, dtype=np.float32)


def extract_full_ammo_features(
    video: Path,
    roi: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一次解码全片，同时生成精细分段与自动扫描所需特征。"""
    cap = cv2.VideoCapture(str(video))
    x0, y0, x1, y1 = roi
    segment_features: list[np.ndarray] = []
    display_features: list[np.ndarray] = []
    ocr_features: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y0:y1, x0:x1]
        segment_features.append(normalized_ammo_feature(crop))
        display_features.append(ammo_display_feature(crop))
        ocr_features.append(ammo_ocr_cells(crop))
    cap.release()
    if not segment_features:
        fail("自动扫描弹药时没有解码出任何视频帧")
    return (
        np.asarray(segment_features, dtype=np.float64),
        np.asarray(display_features, dtype=np.float64),
        np.asarray(ocr_features, dtype=np.float32),
    )


def load_ammo_digit_templates(path: Path) -> np.ndarray:
    if not path.is_file():
        fail(
            f"弹药 OCR 模板不存在: {path}；先运行 calibrate_ammo_templates.py"
        )
    try:
        with np.load(path) as payload:
            templates = np.asarray(payload["templates"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        fail(f"无法读取弹药 OCR 模板 {path}: {exc}")
    if templates.shape != (10, 50, 36):
        fail(f"弹药 OCR 模板尺寸错误: {templates.shape} != (10, 50, 36)")
    return templates


def recognize_ammo_number(
    cells: np.ndarray, templates: np.ndarray
) -> tuple[int, float]:
    digits: list[int] = []
    errors: list[float] = []
    for position, cell in enumerate(cells):
        ink = float(np.mean(cell))
        # The red low-ammo glyph for digit 1 is very narrow (roughly 2-3% ink),
        # so only a nearly empty cell is a leading zero placeholder.
        if position < 2 and ink < 0.01:
            digits.append(0)
            errors.append(0.0)
            continue
        mse = np.mean(np.square(templates - cell[None, :, :]), axis=(1, 2))
        digit = int(np.argmin(mse))
        digits.append(digit)
        errors.append(float(mse[digit]))
    number = 100 * digits[0] + 10 * digits[1] + digits[2]
    return number, float(np.mean(errors))


def ocr_start_ammo(
    ocr_features: np.ndarray,
    templates: np.ndarray,
    first_event_frame: int,
    cadence: int,
    maximum_ammo: int = 200,
    expected_duration_frames: int | None = None,
) -> tuple[int, float]:
    """Read the full-mag value from stable frames immediately before firing."""
    left = max(0, first_event_frame - max(120, cadence * 8))
    right = max(left + 1, first_event_frame)
    observations: list[tuple[int, float]] = []
    for frame_index in range(left, right):
        number, error = recognize_ammo_number(ocr_features[frame_index], templates)
        # A bright background can leak through the translucent grey leading
        # zero and make 045/060 look like 145/160. Conversely, PKM really has
        # 125 rounds. Resolve an ambiguous leading 1 from the observed firing
        # duration instead of using a hard capacity ceiling.
        if 100 < number < 200:
            short_number = number % 100
            candidates = [
                candidate
                for candidate in (number, short_number)
                if 1 <= candidate <= maximum_ammo
            ]
            if expected_duration_frames is not None and candidates:
                number = min(
                    candidates,
                    key=lambda candidate: abs(
                        math.log(
                            max(
                                1e-6,
                                expected_duration_frames
                                / max(1, candidate * cadence),
                            )
                        )
                    ),
                )
            elif number > maximum_ammo:
                number = short_number
        if 1 <= number <= maximum_ammo and error <= 0.08:
            observations.append((number, error))
    if not observations:
        fail("弹药 OCR 无法读取开火前的完整弹匣数")
    counts: dict[int, list[float]] = {}
    for number, error in observations:
        counts.setdefault(number, []).append(error)
    # A valid idle HUD value persists for many frames. Prefer frequency first;
    # use the lower template error and larger ammo value only as tie-breakers.
    number = max(
        counts,
        key=lambda value: (len(counts[value]), -float(np.median(counts[value])), value),
    )
    if len(counts[number]) < 3:
        fail("弹药 OCR 的完整弹匣读数没有持续至少 3 帧")
    return number, float(np.median(counts[number]))


def ocr_first_stable_zero(
    ocr_features: np.ndarray,
    templates: np.ndarray,
    first_event_frame: int,
    minimum_run: int = 3,
) -> int:
    """Return the first frame of the stable 000 HUD state after firing."""
    run_start: int | None = None
    run_length = 0
    for frame_index in range(first_event_frame, len(ocr_features)):
        number, error = recognize_ammo_number(ocr_features[frame_index], templates)
        if number == 0 and error <= 0.08:
            if run_start is None:
                run_start = frame_index
            run_length += 1
            if run_length >= minimum_run:
                return int(run_start)
        else:
            run_start = None
            run_length = 0
    fail("弹药 OCR 未在开火后找到稳定的 000 状态")


def _ammo_change_scores(features: np.ndarray, window: int = 3) -> np.ndarray:
    """比较边界两侧的时域中位数字形，返回每帧作为变化点的强度。"""
    frame_total = len(features)
    scores = np.zeros(frame_total, dtype=np.float64)
    if frame_total < window * 2 + 1:
        return scores
    for frame_index in range(window, frame_total - window):
        before = np.median(features[frame_index - window : frame_index], axis=0)
        after = np.median(features[frame_index : frame_index + window], axis=0)
        scores[frame_index] = float(np.sqrt(np.mean(np.square(after - before))))
    return scores


def detect_ammo_sequence(
    display_features: np.ndarray,
    min_segment: int,
    max_segment: int,
    threshold_mad: float,
    minimum_shots: int,
) -> AmmoDetection:
    """自动寻找全视频中最长、稳定射速的连续弹匣递减序列。

    先从数字字形变化分数的自相关估计每发间隔，再在各相位上寻找最长的
    连续变化梳状序列。对完整打空的录制，变化次数就是弹匣最大弹药数。
    """
    if min_segment < 2 or max_segment < min_segment:
        fail("ammo-min-segment / ammo-max-segment 参数无效")
    if threshold_mad <= 0 or minimum_shots < 2:
        fail("自动弹药检测参数无效")

    scores = _ammo_change_scores(display_features)
    baseline = float(np.median(scores))
    mad = float(np.median(np.abs(scores - baseline))) + 1e-9
    threshold = baseline + threshold_mad * mad
    eventness = np.clip(
        (scores - (baseline + 2.0 * mad)) / (6.0 * mad),
        0.0,
        5.0,
    )

    correlations: dict[int, float] = {}
    for lag in range(min_segment, max_segment + 1):
        correlations[lag] = float(np.dot(eventness[:-lag], eventness[lag:]))
    peak_correlation = max(correlations.values(), default=0.0)
    if peak_correlation <= 0:
        fail("自动弹药检测未找到周期性数字变化")
    # 选择第一个达到主峰 90% 的周期，避免选择基本周期的 2 倍谐波。
    cadence = min(
        lag for lag, value in correlations.items() if value >= 0.90 * peak_correlation
    )
    jitter = max(2, int(round(cadence * 0.30)))
    window = 3

    best_length = 0
    best_strength = -math.inf
    best_keyframes: list[int] = []
    for phase in range(cadence):
        grid = range(max(window, phase), len(scores) - window, cadence)
        grid_scores: list[float] = []
        grid_frames: list[int] = []
        for position in grid:
            left = max(window, position - jitter)
            right = min(len(scores) - window, position + jitter + 1)
            if right <= left:
                continue
            local_frame = left + int(np.argmax(scores[left:right]))
            grid_frames.append(local_frame)
            grid_scores.append(float(scores[local_frame]))

        begin = 0
        while begin < len(grid_scores):
            if grid_scores[begin] < threshold:
                begin += 1
                continue
            end = begin
            while end + 1 < len(grid_scores) and grid_scores[end + 1] >= threshold:
                end += 1
            length = end - begin + 1
            strength = float(sum(grid_scores[begin : end + 1]))
            if length > best_length or (length == best_length and strength > best_strength):
                best_length = length
                best_strength = strength
                best_keyframes = grid_frames[begin : end + 1]
            begin = end + 1

    # Burst-fire and manually tapped weapons do not have one global cadence.
    # Each real HUD transition produces one short, contiguous score peak; form
    # an interval-bounded event chain as a fallback.  The regular cadence path
    # remains preferred for full-auto recordings because it rejects nearby
    # post-fire HUD animations particularly well.
    active_frames = np.flatnonzero(scores >= threshold)
    peak_groups = (
        np.split(active_frames, np.where(np.diff(active_frames) > 1)[0] + 1)
        if len(active_frames)
        else []
    )
    event_peaks = [
        int(group[int(np.argmax(scores[group]))]) for group in peak_groups if len(group)
    ]
    minimum_peak_distance = max(min_segment, int(round(cadence * 0.60)))
    suppressed_peaks: list[int] = []
    for event_frame in event_peaks:
        if (
            suppressed_peaks
            and event_frame - suppressed_peaks[-1] < minimum_peak_distance
        ):
            if scores[event_frame] > scores[suppressed_peaks[-1]]:
                suppressed_peaks[-1] = event_frame
            continue
        suppressed_peaks.append(event_frame)
    event_peaks = suppressed_peaks
    event_runs: list[list[int]] = []
    current_run: list[int] = []
    maximum_event_gap = max(max_segment * 2, max_segment + 8)
    for event_frame in event_peaks:
        if current_run and event_frame - current_run[-1] > maximum_event_gap:
            event_runs.append(current_run)
            current_run = []
        current_run.append(event_frame)
    if current_run:
        event_runs.append(current_run)
    irregular_keyframes = max(event_runs, key=len, default=[])
    prefer_irregular = len(irregular_keyframes) >= max(
        minimum_shots, best_length + max(3, math.ceil(best_length * 0.25))
    )
    if prefer_irregular:
        best_keyframes = irregular_keyframes
        best_length = len(best_keyframes)
        intervals = np.diff(best_keyframes)
        normal_intervals = intervals[intervals <= max_segment]
        if len(normal_intervals):
            cadence = max(min_segment, int(round(float(np.median(normal_intervals)))))

    if best_length < minimum_shots or not best_keyframes:
        fail(
            f"自动弹药检测只找到 {best_length} 次连续变化，少于 auto-min-shots={minimum_shots}"
        )

    analysis_start = max(0, best_keyframes[0] - cadence)
    analysis_end = min(len(display_features) - 1, best_keyframes[-1] + cadence)
    return AmmoDetection(
        analysis_start_frame=analysis_start,
        analysis_end_frame=analysis_end,
        start_ammo=best_length,
        shot_count=best_length,
        cadence_frames=cadence,
        event_threshold=threshold,
        baseline_score=baseline,
        approximate_keyframes=best_keyframes,
        change_scores=scores,
        decoded_frame_count=len(display_features),
    )


def refine_keyframes_from_display(
    keyframes: Sequence[int],
    display_features: np.ndarray,
    radius: int = 3,
) -> list[int]:
    """把分段边界吸附到字形相邻帧差最大的第一帧。"""
    adjacent_change = np.sqrt(
        np.mean(np.square(display_features[1:] - display_features[:-1]), axis=1)
    )
    refined: list[int] = []
    for index, keyframe in enumerate(keyframes):
        previous_midpoint = (
            (int(keyframes[index - 1]) + int(keyframe)) // 2 + 1
            if index > 0
            else 1
        )
        next_midpoint = (
            (int(keyframe) + int(keyframes[index + 1])) // 2 + 1
            if index + 1 < len(keyframes)
            else len(display_features)
        )
        left = max(1, previous_midpoint, int(keyframe) - radius)
        right = min(len(display_features), next_midpoint, int(keyframe) + radius + 1)
        if right <= left:
            frame = max(left, refined[-1] + 1 if refined else left)
            refined.append(frame)
            continue
        frame = left + int(np.argmax(adjacent_change[left - 1 : right - 1]))
        refined.append(frame)
    if any(current <= previous for previous, current in zip(refined, refined[1:])):
        fail("自动关键帧吸附后未严格递增，请改用手动弹药参数")
    return refined


def estimate_pitch_range_deg(
    recoil_y_up_px: Sequence[float],
    width: int,
    height: int,
    fov_deg: float,
    fov_axis: str,
    scope_magnification: float,
) -> tuple[float, float]:
    """用针孔投影和瞄准镜倍率把纵向像素轨迹换算为俯仰角跨度。"""
    if not 1.0 < fov_deg < 179.0:
        fail("fov-deg 必须在 1 到 179 度之间")
    if not math.isfinite(scope_magnification) or scope_magnification <= 0:
        fail("scope-magnification 必须是正数")
    if fov_axis == "reference-horizontal":
        scoped_reference_horizontal = 2.0 * math.atan(
            math.tan(math.radians(fov_deg) / 2.0) / scope_magnification
        )
        scoped_vertical = 2.0 * math.atan(
            math.tan(scoped_reference_horizontal / 2.0) / (4.0 / 3.0)
        )
        scoped_focal_length = height / (2.0 * math.tan(scoped_vertical / 2.0))
    else:
        sensor_size = width if fov_axis == "horizontal" else height
        focal_length = sensor_size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        scoped_focal_length = focal_length * scope_magnification
    angles = [math.atan(float(y) / scoped_focal_length) for y in recoil_y_up_px]
    pitch_range = math.degrees(max(angles) - min(angles))
    return pitch_range, scoped_focal_length


def extract_ammo_features(
    video: Path,
    start_frame: int,
    end_frame: int,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    x0, y0, x1, y1 = roi
    features: list[np.ndarray] = []
    for frame_index in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            fail(f"提取弹药数字时无法读取第 {frame_index} 帧")
        features.append(normalized_ammo_feature(frame[y0:y1, x0:x1]))
    cap.release()
    return np.asarray(features, dtype=np.float64)


def segment_fixed_count_sequence(
    features: np.ndarray,
    segment_count: int,
    min_length: int,
    max_length: int,
    cadence_weight: float,
) -> list[int]:
    """动态规划分成固定数量的连续状态；返回每段的右开区间终点。"""
    frame_total, dimension = features.shape
    if min_length * segment_count > frame_total:
        fail("ammo-min-segment 太大，无法容纳全部弹药状态")
    if max_length * segment_count < frame_total:
        fail("ammo-max-segment 太小，无法覆盖全部帧")

    cumulative = np.vstack(
        (np.zeros((1, dimension), dtype=np.float64), np.cumsum(features, axis=0))
    )
    cumulative_sq = np.concatenate(
        ([0.0], np.cumsum(np.einsum("ij,ij->i", features, features)))
    )
    dp = np.full((segment_count + 1, frame_total + 1), np.inf, dtype=np.float64)
    previous = np.full((segment_count + 1, frame_total + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    target_length = frame_total / segment_count

    for state in range(1, segment_count + 1):
        end_lo = max(
            state * min_length,
            frame_total - (segment_count - state) * max_length,
        )
        end_hi = min(
            state * max_length,
            frame_total - (segment_count - state) * min_length,
        )
        for end in range(end_lo, end_hi + 1):
            begin_lo = max((state - 1) * min_length, end - max_length)
            begin_hi = min((state - 1) * max_length, end - min_length)
            begins = np.arange(begin_lo, begin_hi + 1, dtype=np.int32)
            lengths = end - begins
            sums = cumulative[end] - cumulative[begins]
            sse = (
                cumulative_sq[end]
                - cumulative_sq[begins]
                - np.einsum("ij,ij->i", sums, sums) / lengths
            )
            costs = (
                dp[state - 1, begins]
                + sse
                + cadence_weight * np.square(lengths - target_length)
            )
            best = int(np.argmin(costs))
            dp[state, end] = costs[best]
            previous[state, end] = int(begins[best])

    if not np.isfinite(dp[segment_count, frame_total]):
        fail("弹药关键帧动态规划失败，请检查 ROI、段长和 shot-count")
    endpoints: list[int] = []
    end = frame_total
    for state in range(segment_count, 0, -1):
        endpoints.append(end)
        end = int(previous[state, end])
    endpoints.reverse()
    return endpoints


def create_outside_scope_mask(width: int, height: int) -> np.ndarray:
    """只保留镜外墙面；屏蔽瞄准镜、枪体和固定 HUD。"""
    mask = np.full((height, width), 255, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(0.50 * width)), int(round(0.50 * height))),
        (int(round(0.225 * width)), int(round(0.34 * height))),
        0,
        0,
        360,
        0,
        -1,
    )
    mask[int(round(0.70 * height)) :, :] = 0  # 枪体、手臂、底部 HUD
    mask[: int(round(0.20 * height)), : int(round(0.25 * width))] = 0
    mask[: int(round(0.09 * height)), int(round(0.73 * width)) :] = 0
    mask[
        int(round(0.30 * height)) : int(round(0.65 * height)),
        int(round(0.82 * width)) :,
    ] = 0  # 手柄输入显示
    mask[int(round(0.65 * height)) :, int(round(0.78 * width)) :] = 0
    mask[int(round(0.82 * height)) :, : int(round(0.24 * width))] = 0
    return mask


class OutsideFeatureMatcher:
    def __init__(
        self,
        width: int,
        height: int,
        detector_name: str,
        feature_scale: float,
        max_features: int,
        ratio_test: float,
        ransac_threshold: float,
        min_inliers: int,
        min_inlier_ratio: float,
        max_step_px: float,
        max_step_angle: float,
    ) -> None:
        if not 0.15 <= feature_scale <= 1.0:
            fail("feature-scale 必须在 0.15 到 1.0 之间")
        self.full_width = width
        self.full_height = height
        self.scale = feature_scale
        self.width = max(64, int(round(width * feature_scale)))
        self.height = max(64, int(round(height * feature_scale)))
        full_mask = create_outside_scope_mask(width, height)
        self.base_mask = cv2.resize(
            full_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        self.detector_name = detector_name
        if detector_name == "sift":
            self.detector = cv2.SIFT_create(
                nfeatures=max_features, contrastThreshold=0.02, edgeThreshold=15
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            self.detector = cv2.ORB_create(
                nfeatures=max_features,
                scaleFactor=1.2,
                nlevels=8,
                fastThreshold=12,
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio_test = ratio_test
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_step_px = max_step_px
        self.max_step_angle = max_step_angle
        self.center = np.array([width / 2.0, height / 2.0], dtype=np.float64)

    def extract(self, frame: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        small = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        # 高亮、带颜色的火花先剔除；剩余错误匹配交给 RANSAC。
        hot = ((hsv[:, :, 2] > 235) & (hsv[:, :, 1] > 50)).astype(np.uint8) * 255
        kernel_size = max(5, int(round(15 * self.scale / 0.5)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        hot = cv2.dilate(hot, np.ones((kernel_size, kernel_size), np.uint8))
        mask = self.base_mask.copy()
        mask[hot > 0] = 0
        return self.detector.detectAndCompute(gray, mask)

    @staticmethod
    def _rigid_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        covariance = src_centered.T @ dst_centered
        u, _, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = vt.T @ u.T
        translation = dst_mean - rotation @ src_mean
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :2] = rotation
        matrix[:2, 2] = translation
        return matrix

    def _match_once(
        self,
        previous: tuple[list[cv2.KeyPoint], np.ndarray | None],
        current: tuple[list[cv2.KeyPoint], np.ndarray | None],
        ratio: float,
        threshold: float,
    ) -> tuple[np.ndarray | None, dict[str, float | int]]:
        previous_keypoints, previous_descriptors = previous
        current_keypoints, current_descriptors = current
        quality: dict[str, float | int] = {
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_error_px": math.nan,
            "ransac_scale": math.nan,
        }
        if previous_descriptors is None or current_descriptors is None:
            return None, quality
        pairs = self.matcher.knnMatch(previous_descriptors, current_descriptors, k=2)
        good = [m for m, n in pairs if m.distance < ratio * n.distance]
        quality["good_matches"] = len(good)
        if len(good) < max(8, self.min_inliers):
            return None, quality

        src_small = np.float32([previous_keypoints[m.queryIdx].pt for m in good])
        dst_small = np.float32([current_keypoints[m.trainIdx].pt for m in good])
        ransac, inlier_mask = cv2.estimateAffinePartial2D(
            src_small,
            dst_small,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            maxIters=5000,
            confidence=0.999,
            refineIters=30,
        )
        if ransac is None or inlier_mask is None:
            return None, quality
        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / len(good)
        quality["inliers"] = inlier_count
        quality["inlier_ratio"] = inlier_ratio
        a, b = float(ransac[0, 0]), float(ransac[1, 0])
        ransac_scale = math.hypot(a, b)
        quality["ransac_scale"] = ransac_scale
        if inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return None, quality
        if not 0.96 <= ransac_scale <= 1.04:
            return None, quality

        src = src_small[inliers].astype(np.float64) / self.scale
        dst = dst_small[inliers].astype(np.float64) / self.scale
        rigid = self._rigid_fit(src, dst)
        projected = (rigid[:2, :2] @ src.T).T + rigid[:2, 2]
        errors = np.linalg.norm(projected - dst, axis=1)
        # 用全分辨率误差再精炼一次，防止 RANSAC 边缘点拉偏刚体拟合。
        refined = errors <= max(2.0, threshold / self.scale * 1.5)
        if int(np.count_nonzero(refined)) >= self.min_inliers:
            rigid = self._rigid_fit(src[refined], dst[refined])
            projected = (rigid[:2, :2] @ src[refined].T).T + rigid[:2, 2]
            errors = np.linalg.norm(projected - dst[refined], axis=1)
        quality["median_error_px"] = float(np.median(errors))

        angle = matrix_angle_deg(rigid)
        center_after = apply_matrix(rigid, self.center)
        center_step = float(np.linalg.norm(center_after - self.center))
        if center_step > self.max_step_px or abs(angle) > self.max_step_angle:
            return None, quality
        return rigid, quality

    def match(
        self,
        frame_index: int,
        previous: tuple[list[cv2.KeyPoint], np.ndarray | None],
        current: tuple[list[cv2.KeyPoint], np.ndarray | None],
    ) -> StepResult:
        matrix, quality = self._match_once(
            previous, current, self.ratio_test, self.ransac_threshold
        )
        status = "ok"
        if matrix is None:
            matrix, retry_quality = self._match_once(
                previous,
                current,
                min(0.88, self.ratio_test + 0.10),
                self.ransac_threshold * 1.75,
            )
            if matrix is not None:
                quality = retry_quality
                status = "retry"
        return StepResult(
            frame=frame_index,
            matrix=matrix,
            good_matches=int(quality["good_matches"]),
            inliers=int(quality["inliers"]),
            inlier_ratio=float(quality["inlier_ratio"]),
            median_error_px=float(quality["median_error_px"]),
            ransac_scale=float(quality["ransac_scale"]),
            status=status if matrix is not None else "failed",
        )


def matrix_angle_deg(matrix: np.ndarray) -> float:
    return math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))


def apply_matrix(matrix: np.ndarray, point: Sequence[float]) -> np.ndarray:
    vector = np.array([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
    return (matrix @ vector)[:2]


def rigid_from_center_step(
    angle_deg: float, center_step: np.ndarray, center: np.ndarray
) -> np.ndarray:
    angle = math.radians(angle_deg)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    translation = center + center_step - rotation @ center
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = translation
    return matrix


def fill_failed_steps(steps: list[StepResult], center: np.ndarray) -> int:
    """极少数 RANSAC 失败帧用相邻可靠 SE(2) 步长插值，并明确打标。"""
    valid = [i for i, step in enumerate(steps) if i > 0 and step.matrix is not None]
    if not valid:
        fail("所有镜外 Feature Matching + RANSAC 均失败")
    failed = 0
    for index in range(1, len(steps)):
        if steps[index].matrix is not None:
            continue
        failed += 1
        left_candidates = [i for i in valid if i < index]
        right_candidates = [i for i in valid if i > index]
        left = left_candidates[-1] if left_candidates else None
        right = right_candidates[0] if right_candidates else None

        def parameters(i: int) -> tuple[float, np.ndarray]:
            matrix = steps[i].matrix
            assert matrix is not None
            return matrix_angle_deg(matrix), apply_matrix(matrix, center) - center

        if left is not None and right is not None:
            weight = (index - left) / (right - left)
            left_angle, left_step = parameters(left)
            right_angle, right_step = parameters(right)
            angle = (1.0 - weight) * left_angle + weight * right_angle
            displacement = (1.0 - weight) * left_step + weight * right_step
        elif left is not None:
            angle, displacement = parameters(left)
        elif right is not None:
            angle, displacement = parameters(right)
        else:  # pragma: no cover - 上面已经保证至少存在一个有效变换
            angle, displacement = 0.0, np.zeros(2, dtype=np.float64)
        steps[index].matrix = rigid_from_center_step(angle, displacement, center)
        steps[index].status = "interpolated"
    return failed


def quadratic_peak(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, center, right = map(float, values[index - 1 : index + 2])
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-9:
        return float(index)
    offset = 0.5 * (left - right) / denominator
    return float(index) + float(np.clip(offset, -1.0, 1.0))


def detect_reticle_crosshair(
    frame: np.ndarray,
    max_angle_deg: float,
    angle_step_deg: float,
) -> ReticleResult:
    """以黑色横/纵刻度线的暗线对比度求交点，不依赖火花或红色准星。"""
    height, width = frame.shape[:2]
    nominal_x, nominal_y = width / 2.0, height / 2.0
    scale = min(width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT)
    half_width = int(round(360 * scale))
    up = int(round(170 * scale))
    down = int(round(340 * scale))
    x0 = max(0, int(round(nominal_x)) - half_width)
    x1 = min(width, int(round(nominal_x)) + half_width + 1)
    y0 = max(0, int(round(nominal_y)) - up)
    y1 = min(height, int(round(nominal_y)) + down + 1)
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_center = (nominal_x - x0, nominal_y - y0)
    search_radius = max(12, int(round(34 * scale)))
    line_half_width = max(1, int(round(1 * scale)))
    neighbor_offset = max(3, int(round(7 * scale)))
    central_gap = max(22, int(round(55 * scale)))
    margin = max(8, int(round(20 * scale)))

    if angle_step_deg <= 0:
        fail("reticle-angle-step 必须大于 0")
    angles = np.arange(
        -max_angle_deg,
        max_angle_deg + angle_step_deg * 0.5,
        angle_step_deg,
    )
    best: tuple[float, float, int, int, np.ndarray, np.ndarray, np.ndarray] | None = None
    for angle in angles:
        rotation = cv2.getRotationMatrix2D(local_center, -float(angle), 1.0)
        rotated = cv2.warpAffine(
            gray,
            rotation,
            (gray.shape[1], gray.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        center_x, center_y = int(round(local_center[0])), int(round(local_center[1]))
        left = np.arange(margin, max(margin + 1, center_x - central_gap))
        right = np.arange(
            min(rotated.shape[1] - margin - 1, center_x + central_gap),
            rotated.shape[1] - margin,
        )
        horizontal_x = np.concatenate((left, right))
        row_candidates = np.arange(center_y - search_radius, center_y + search_radius + 1)
        horizontal_response: list[float] = []
        for row in row_candidates:
            if row - neighbor_offset < 0 or row + neighbor_offset >= rotated.shape[0]:
                horizontal_response.append(-math.inf)
                continue
            line = np.mean(
                rotated[
                    row - line_half_width : row + line_half_width + 1,
                    horizontal_x,
                ],
                axis=0,
            )
            neighbors = 0.5 * (
                rotated[row - neighbor_offset, horizontal_x]
                + rotated[row + neighbor_offset, horizontal_x]
            )
            contrast = np.clip(neighbors - line, -10.0, 80.0)
            horizontal_response.append(float(np.mean(contrast)))
        horizontal_response_array = np.asarray(horizontal_response, dtype=np.float64)
        best_row_index = int(np.argmax(horizontal_response_array))
        best_row = int(row_candidates[best_row_index])

        column_candidates = np.arange(center_x - search_radius, center_x + search_radius + 1)
        vertical_ranges = (
            np.arange(
                min(rotated.shape[0] - neighbor_offset - 1, best_row + int(45 * scale)),
                min(rotated.shape[0] - neighbor_offset - 1, best_row + int(110 * scale)),
            ),
            np.arange(
                min(rotated.shape[0] - neighbor_offset - 1, best_row + int(170 * scale)),
                min(rotated.shape[0] - neighbor_offset - 1, best_row + int(310 * scale)),
            ),
        )
        vertical_y = np.concatenate([part for part in vertical_ranges if len(part) > 0])
        if len(vertical_y) < 10:
            continue
        vertical_response: list[float] = []
        for column in column_candidates:
            if (
                column - neighbor_offset < 0
                or column + neighbor_offset >= rotated.shape[1]
            ):
                vertical_response.append(-math.inf)
                continue
            line = np.mean(
                rotated[
                    vertical_y,
                    column - line_half_width : column + line_half_width + 1,
                ],
                axis=1,
            )
            neighbors = 0.5 * (
                rotated[vertical_y, column - neighbor_offset]
                + rotated[vertical_y, column + neighbor_offset]
            )
            contrast = np.clip(neighbors - line, -10.0, 80.0)
            vertical_response.append(float(np.mean(contrast)))
        vertical_response_array = np.asarray(vertical_response, dtype=np.float64)
        best_column_index = int(np.argmax(vertical_response_array))
        score = float(
            horizontal_response_array[best_row_index]
            + vertical_response_array[best_column_index]
        )
        if best is None or score > best[0]:
            best = (
                score,
                float(angle),
                best_row_index,
                best_column_index,
                horizontal_response_array,
                vertical_response_array,
                rotation,
            )
    if best is None:
        fail("刻度线检测失败")
    score, angle, row_index, column_index, horizontal, vertical, rotation = best
    row = (
        int(round(local_center[1]))
        - search_radius
        + quadratic_peak(horizontal, row_index)
    )
    column = (
        int(round(local_center[0]))
        - search_radius
        + quadratic_peak(vertical, column_index)
    )
    inverse_rotation = cv2.invertAffineTransform(rotation)
    original_x, original_y = inverse_rotation @ np.array([column, row, 1.0])
    return ReticleResult(
        x=float(x0 + original_x),
        y=float(y0 + original_y),
        angle_deg=angle,
        confidence=score,
        horizontal_score=float(horizontal[row_index]),
        vertical_score=float(vertical[column_index]),
    )


def detect_reticle_red_dot(frame: np.ndarray) -> ReticleResult:
    """Detect the compact pure-red aiming dot used by the 1x pistol optics.

    Muzzle flash and impact sparks are usually yellow/orange, while the sight
    dot has a much larger red-minus-green/blue chroma.  The optic housing is
    excluded by both the central search window and component compactness.
    """
    height, width = frame.shape[:2]
    center_x, center_y = width / 2.0, height / 2.0
    scale = min(width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT)
    radius = max(50, int(round(115 * scale)))
    x0 = max(0, int(round(center_x)) - radius)
    x1 = min(width, int(round(center_x)) + radius + 1)
    y0 = max(0, int(round(center_y)) - radius)
    y1 = min(height, int(round(center_y)) + radius + 1)
    crop = frame[y0:y1, x0:x1]
    blue, green, red = cv2.split(crop)
    redness = red.astype(np.float32) - np.maximum(green, blue).astype(np.float32)
    # Subtract a broad local background so a small red dot remains isolated
    # even when the entire optic window is covered by orange muzzle flash.
    highpass = redness - cv2.GaussianBlur(redness, (0, 0), 5.0)
    mask = ((red > 40) & (redness > 8) & (highpass > 5)).astype(np.uint8)
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    candidates: list[tuple[float, float, float, float]] = []
    max_size = max(14, int(round(32 * scale)))
    max_area = max(80, int(round(500 * scale * scale)))
    for label in range(1, labels_count):
        left, top, component_width, component_height, area = map(
            int, stats[label]
        )
        if (
            area < 2
            or area > max_area
            or component_width > max_size
            or component_height > max_size
        ):
            continue
        component = labels == label
        ys, xs = np.nonzero(component)
        weights = np.maximum(highpass[component].astype(np.float64), 1.0)
        local_x = float(np.average(xs, weights=weights))
        local_y = float(np.average(ys, weights=weights))
        absolute_x = x0 + local_x
        absolute_y = y0 + local_y
        distance = math.hypot(absolute_x - center_x, absolute_y - center_y)
        if (
            distance > radius
            or absolute_y > center_y + max(12.0, 24.0 * scale)
            or abs(absolute_x - center_x) > max(42.0, 70.0 * scale)
        ):
            continue
        purity = float(np.mean(redness[component]))
        local_contrast = float(np.mean(highpass[component]))
        compactness = area / max(1, component_width * component_height)
        score = (
            purity
            + local_contrast
            + 20.0 * compactness
            + 0.4 * min(area, 80)
            - 0.8 * abs(absolute_y - center_y)
            - 1.2 * abs(absolute_x - center_x)
        )
        candidates.append((score, absolute_x, absolute_y, purity))
    if not candidates:
        fail("1x 红点检测失败")
    score, x, y, purity = max(candidates, key=lambda item: item[0])
    return ReticleResult(
        x=x,
        y=y,
        angle_deg=0.0,
        confidence=score,
        horizontal_score=purity,
        vertical_score=score - purity,
    )


def detect_reticle(
    frame: np.ndarray,
    mode: str,
    max_angle_deg: float,
    angle_step_deg: float,
) -> ReticleResult:
    if mode == "red-dot":
        return detect_reticle_red_dot(frame)
    return detect_reticle_crosshair(frame, max_angle_deg, angle_step_deg)


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


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        fail(f"视频不存在: {video}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height, fps, reported_frame_count = read_video_info(video)
    ammo_roi = parse_scaled_roi(args.ammo_roi, width, height)

    manual_values = (
        args.start_frame,
        args.end_frame,
        args.start_ammo,
        args.shot_count,
    )
    if any(value is not None for value in manual_values) and not all(
        value is not None for value in manual_values
    ):
        fail("手动模式必须同时提供 start-frame、end-frame、start-ammo、shot-count")
    auto_ammo = all(value is None for value in manual_values)
    ammo_detection: AmmoDetection | None = None
    ammo_ocr_error: float | None = None
    ammo_zero_frame: int | None = None
    ammo_stable_zero_frame: int | None = None
    ammo_cadence_duration_ratio: float | None = None

    if auto_ammo:
        print(
            f"[1/4] 视频 {width}x{height} @ {fps:.3f} fps；全片扫描弹药 ROI {ammo_roi} ...",
            flush=True,
        )
        full_ammo_features, display_features, ocr_features = extract_full_ammo_features(
            video, ammo_roi
        )
        ammo_detection = detect_ammo_sequence(
            display_features,
            min_segment=args.ammo_min_segment,
            max_segment=args.ammo_max_segment,
            threshold_mad=args.auto_ammo_threshold_mad,
            minimum_shots=args.auto_min_shots,
        )
        templates = load_ammo_digit_templates(args.ammo_template_file.resolve())
        stable_zero_frame = ocr_first_stable_zero(
            ocr_features,
            templates,
            ammo_detection.approximate_keyframes[0],
        )
        maximum_ammo = min(
            200,
            max(
                args.auto_min_shots,
                (
                    stable_zero_frame
                    - ammo_detection.approximate_keyframes[0]
                    + ammo_detection.cadence_frames
                )
                // args.ammo_min_segment,
            ),
        )
        detected_count, ammo_ocr_error = ocr_start_ammo(
            ocr_features,
            templates,
            ammo_detection.approximate_keyframes[0],
            ammo_detection.cadence_frames,
            maximum_ammo=maximum_ammo,
            expected_duration_frames=(
                stable_zero_frame - ammo_detection.approximate_keyframes[0]
            ),
        )
        # The event detector is deliberately only a coarse firing-range finder.
        # It may miss a low-contrast digit transition and later mistake a HUD
        # animation for the missing Nth event (notably after the magazine is
        # already empty).  Keep the stable OCR zero plus one cadence as the
        # provisional tail; the fixed-state segmentation below will locate the
        # real 1 -> 0 boundary jointly with every preceding ammo state.
        zero_frame = stable_zero_frame
        ammo_zero_frame = None
        ammo_stable_zero_frame = stable_zero_frame
        ammo_cadence_duration_ratio = (
            (stable_zero_frame - ammo_detection.approximate_keyframes[0])
            / max(1, detected_count * ammo_detection.cadence_frames)
        )
        ammo_detection.start_ammo = detected_count
        ammo_detection.shot_count = detected_count
        ammo_detection.analysis_start_frame = max(
            0,
            ammo_detection.approximate_keyframes[0]
            - ammo_detection.cadence_frames,
        )
        ammo_detection.analysis_end_frame = min(
            ammo_detection.decoded_frame_count - 1,
            stable_zero_frame + ammo_detection.cadence_frames,
        )
        ammo_detection.approximate_keyframes = [
            frame
            for frame in ammo_detection.approximate_keyframes
            if frame <= stable_zero_frame + ammo_detection.cadence_frames
        ]
        start_frame = ammo_detection.analysis_start_frame
        end_frame = ammo_detection.analysis_end_frame
        start_ammo = ammo_detection.start_ammo
        shot_count = ammo_detection.shot_count
        frame_count = ammo_detection.decoded_frame_count
        ammo_features = full_ammo_features[start_frame : end_frame + 1]
        cadence_weight = (
            args.auto_ammo_cadence_weight
            if ammo_cadence_duration_ratio <= 1.25
            else min(args.auto_ammo_cadence_weight, 10.0)
        )
        print(
            "      自动识别: "
            f"OCR 弹匣 {start_ammo} 发，约 {ammo_detection.cadence_frames} 帧/发，"
            f"分析范围 [{start_frame}, {end_frame}]",
            flush=True,
        )
    else:
        start_frame = int(args.start_frame)
        end_frame = int(args.end_frame)
        start_ammo = int(args.start_ammo)
        shot_count = int(args.shot_count)
        frame_count = reported_frame_count
        if not 0 <= start_frame <= end_frame < frame_count:
            fail(
                f"帧范围 [{start_frame}, {end_frame}] 超出视频范围 [0, {frame_count - 1}]"
            )
        if shot_count <= 0 or start_ammo < shot_count:
            fail("shot-count 必须大于 0，且 start-ammo 不能小于 shot-count")
        print(
            f"[1/4] 视频 {width}x{height} @ {fps:.3f} fps；"
            f"手动提取弹药 ROI {ammo_roi} ...",
            flush=True,
        )
        ammo_features = extract_ammo_features(video, start_frame, end_frame, ammo_roi)
        cadence_weight = args.ammo_cadence_weight

    endpoints = segment_fixed_count_sequence(
        ammo_features,
        segment_count=shot_count + 1,
        min_length=args.ammo_min_segment,
        max_length=(
            max(args.ammo_max_segment, args.ammo_max_segment * 3)
            if ammo_detection is not None
            else args.ammo_max_segment
        ),
        cadence_weight=cadence_weight,
    )
    keyframes = [start_frame + endpoint for endpoint in endpoints[:-1]]
    if ammo_detection is not None:
        keyframes = refine_keyframes_from_display(keyframes, display_features)
    if len(keyframes) != shot_count:
        fail(f"关键帧数量异常: {len(keyframes)} != {shot_count}")
    if ammo_detection is not None:
        # The final fixed-state boundary is the actual first 000 frame used as
        # the last shot keyframe.  Shrink the provisional tail so feature/RANSAC
        # processing does not include unrelated post-fire HUD animations.
        ammo_zero_frame = keyframes[-1]
        ammo_cadence_duration_ratio = (
            (keyframes[-1] - keyframes[0])
            / max(1, shot_count * ammo_detection.cadence_frames)
        )
        end_frame = min(
            ammo_detection.decoded_frame_count - 1,
            keyframes[-1] + ammo_detection.cadence_frames,
        )
        ammo_detection.analysis_end_frame = end_frame
    print(f"      检出 {len(keyframes)} 个关键帧: {keyframes}", flush=True)

    if ammo_detection is not None:
        approximate_set = set(ammo_detection.approximate_keyframes)
        write_csv(
            output_dir / "ammo_detection.csv",
            (
                {
                    "frame": frame_index,
                    "change_score": float(score),
                    "event_threshold": ammo_detection.event_threshold,
                    "approximate_event": int(frame_index in approximate_set),
                }
                for frame_index, score in enumerate(ammo_detection.change_scores)
            ),
            ("frame", "change_score", "event_threshold", "approximate_event"),
        )

    matcher = OutsideFeatureMatcher(
        width=width,
        height=height,
        detector_name=args.detector,
        feature_scale=args.feature_scale,
        max_features=args.max_features,
        ratio_test=args.ratio_test,
        ransac_threshold=args.ransac_threshold,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        max_step_px=args.max_step_px,
        max_step_angle=args.max_step_angle,
    )
    print("[2/4] 镜外 Feature Matching + RANSAC（镜内/枪体/HUD/高亮火花已屏蔽）...", flush=True)
    cap = cv2.VideoCapture(str(video))
    first_frame = seek_and_read(cap, start_frame)
    feature_mask_preview(first_frame, output_dir / "feature_mask.png")
    previous_features = matcher.extract(first_frame)
    steps: list[StepResult] = [
        StepResult(frame=start_frame, matrix=np.eye(3), status="reference")
    ]
    reticle_by_frame: dict[int, ReticleResult] = {
        start_frame: detect_reticle(
            first_frame,
            args.reticle_mode,
            args.reticle_max_angle,
            args.reticle_angle_step,
        )
    }
    keyframe_set = set(keyframes)
    debug_crops: list[np.ndarray] = []
    for frame_index in range(start_frame + 1, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            fail(f"运动分析时无法读取第 {frame_index} 帧")
        current_features = matcher.extract(frame)
        steps.append(
            matcher.match(frame_index, previous_features, current_features)
        )
        previous_features = current_features
        if frame_index in keyframe_set:
            reticle = detect_reticle(
                frame,
                args.reticle_mode,
                args.reticle_max_angle,
                args.reticle_angle_step,
            )
            reticle_by_frame[frame_index] = reticle
            shot = keyframes.index(frame_index) + 1
            debug_crops.append(
                reticle_debug_crop(
                    frame,
                    reticle,
                    shot,
                    frame_index,
                    start_ammo - shot,
                )
            )
        if (frame_index - start_frame) % 50 == 0:
            print(
                f"      已处理 {frame_index - start_frame}/{end_frame - start_frame} 帧",
                flush=True,
            )
    cap.release()
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    failed_count = fill_failed_steps(steps, center)
    if failed_count:
        print(f"      警告: {failed_count} 个失败步长已用相邻可靠 SE(2) 步长插值", flush=True)

    cumulative_to_reference: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    for step in steps[1:]:
        assert step.matrix is not None
        cumulative_to_reference.append(
            cumulative_to_reference[-1] @ np.linalg.inv(step.matrix)
        )

    print("[3/4] 合成准星实际交点与镜外相机变换，生成轨迹 ...", flush=True)
    baseline_reticle = reticle_by_frame[start_frame]
    baseline_world = apply_matrix(
        cumulative_to_reference[0], (baseline_reticle.x, baseline_reticle.y)
    )
    baseline_center_world = apply_matrix(cumulative_to_reference[0], center)
    previous_world = baseline_world
    previous_pose = cumulative_to_reference[0]
    key_rows: list[dict[str, object]] = []
    for shot, frame_index in enumerate(keyframes, start=1):
        index = frame_index - start_frame
        pose = cumulative_to_reference[index]
        reticle = reticle_by_frame[frame_index]
        world = apply_matrix(pose, (reticle.x, reticle.y))
        center_world = apply_matrix(pose, center)
        reticle_component = world - center_world
        baseline_reticle_component = baseline_world - baseline_center_world
        previous_to_current = np.linalg.inv(pose) @ previous_pose
        camera_center_step = apply_matrix(previous_to_current, center) - center
        key_rows.append(
            {
                "shot": shot,
                "ammo_before": start_ammo - shot + 1,
                "ammo_after": start_ammo - shot,
                "keyframe": frame_index,
                "time_sec": (frame_index / fps),
                "shot_time_ms": round((frame_index - keyframes[0]) * 1000.0 / fps),
                "reticle_screen_x": reticle.x,
                "reticle_screen_y": reticle.y,
                "reticle_offset_x": reticle.x - center[0],
                "reticle_offset_y_down": reticle.y - center[1],
                "reticle_angle_deg": reticle.angle_deg,
                "reticle_confidence": reticle.confidence,
                "world_x_ref": world[0],
                "world_y_ref": world[1],
                "recoil_x_right_px": world[0] - baseline_world[0],
                "recoil_y_up_px": -(world[1] - baseline_world[1]),
                "shot_delta_x_right_px": world[0] - previous_world[0],
                "shot_delta_y_up_px": -(world[1] - previous_world[1]),
                "background_only_x_right_px": center_world[0] - baseline_center_world[0],
                "background_only_y_up_px": -(center_world[1] - baseline_center_world[1]),
                "reticle_jitter_contribution_x_px": (
                    reticle_component[0] - baseline_reticle_component[0]
                ),
                "reticle_jitter_contribution_y_up_px": -(
                    reticle_component[1] - baseline_reticle_component[1]
                ),
                "camera_step_center_dx": camera_center_step[0],
                "camera_step_center_dy_down": camera_center_step[1],
                "camera_rotation_since_prev_deg": matrix_angle_deg(previous_to_current),
                "cumulative_rotation_to_ref_deg": matrix_angle_deg(pose),
            }
        )
        previous_world = world
        previous_pose = pose

    trajectory_corrections = apply_terminal_empty_animation_repair(key_rows)

    all_rows: list[dict[str, object]] = []
    for index, (step, pose) in enumerate(zip(steps, cumulative_to_reference)):
        if step.matrix is None:  # fill_failed_steps 后不会发生
            step_dx = step_dy = step_angle = math.nan
        else:
            center_after = apply_matrix(step.matrix, center)
            step_dx, step_dy = center_after - center
            step_angle = matrix_angle_deg(step.matrix)
        center_ref = apply_matrix(pose, center)
        all_rows.append(
            {
                "frame": start_frame + index,
                "step_center_dx": step_dx,
                "step_center_dy_down": step_dy,
                "step_rotation_deg": step_angle,
                "ransac_scale_diagnostic": step.ransac_scale,
                "good_matches": step.good_matches,
                "inliers": step.inliers,
                "inlier_ratio": step.inlier_ratio,
                "median_reprojection_error_px": step.median_error_px,
                "status": step.status,
                "center_x_in_reference": center_ref[0],
                "center_y_in_reference": center_ref[1],
                "rotation_to_reference_deg": matrix_angle_deg(pose),
            }
        )

    key_fields = list(key_rows[0].keys())
    all_fields = list(all_rows[0].keys())
    write_csv(output_dir / "keyframes_recoil.csv", key_rows, key_fields)
    write_csv(output_dir / "all_frames_motion.csv", all_rows, all_fields)
    draw_trajectory(key_rows, output_dir / "recoil_trajectory.png")
    imwrite(
        output_dir / "reticle_keyframes_contact_sheet.jpg",
        make_contact_sheet(debug_crops),
        quality=92,
    )

    interval_lengths = [
        keyframes[0] - start_frame,
        *[b - a for a, b in zip(keyframes, keyframes[1:])],
        end_frame + 1 - keyframes[-1],
    ]
    estimated_pitch_deg, scoped_focal_length_px = estimate_pitch_range_deg(
        [float(row["trainer_recoil_y_up_px"]) for row in key_rows],
        width=width,
        height=height,
        fov_deg=args.fov_deg,
        fov_axis=args.fov_axis,
        scope_magnification=args.scope_magnification,
    )
    summary = {
        "pipeline_version": "delta-force-v6-joint-zero-boundary",
        "video": str(video),
        "video_width": width,
        "video_height": height,
        "fps": fps,
        "source_frame_count": frame_count,
        "analysis_start_frame": start_frame,
        "analysis_end_frame": end_frame,
        "ammo_detection_mode": "auto" if auto_ammo else "manual",
        "ammo_count_source": "template-ocr" if auto_ammo else "manual",
        "ammo_ocr_median_error": ammo_ocr_error,
        "detected_start_ammo": start_ammo,
        "detected_shot_count": shot_count,
        "ammo_roi": ammo_roi,
        "keyframes": keyframes,
        "shot_times_ms": [
            round((frame_index - keyframes[0]) * 1000.0 / fps)
            for frame_index in keyframes
        ],
        "interval_lengths": interval_lengths,
        "failed_steps_interpolated": failed_count,
        "detector": args.detector,
        "feature_scale": args.feature_scale,
        "coordinate_model": (
            "A_i maps outside-scope background from frame i-1 to i; "
            "C_i=C_(i-1)*inv(A_i); trajectory point p_i=C_i*reticle_i"
        ),
        "baseline_reticle": asdict(baseline_reticle),
        "minimum_reticle_confidence": min(
            result.confidence for result in reticle_by_frame.values()
        ),
        "reticle_mode": args.reticle_mode,
        "minimum_inlier_ratio": min(
            step.inlier_ratio for step in steps[1:] if step.status != "interpolated"
        ),
        "minimum_inliers": min(
            step.inliers for step in steps[1:] if step.status != "interpolated"
        ),
        "fov_degrees": args.fov_deg,
        "fov_axis": args.fov_axis,
        "scope_magnification": args.scope_magnification,
        "scoped_focal_length_px": scoped_focal_length_px,
        "estimated_recoil_pitch_range_deg": estimated_pitch_deg,
        "trajectory_corrections": trajectory_corrections,
    }
    if ammo_detection is not None:
        summary["ammo_auto_detection"] = {
            "decoded_frame_count": ammo_detection.decoded_frame_count,
            "estimated_cadence_frames": ammo_detection.cadence_frames,
            "event_threshold": ammo_detection.event_threshold,
            "baseline_change_score": ammo_detection.baseline_score,
            "ocr_zero_frame": ammo_zero_frame,
            "ocr_stable_zero_frame": ammo_stable_zero_frame,
            "cadence_duration_ratio": ammo_cadence_duration_ratio,
            "cadence_model": (
                "regular" if ammo_cadence_duration_ratio is not None and ammo_cadence_duration_ratio <= 1.25
                else "variable"
            ),
            "approximate_keyframes": ammo_detection.approximate_keyframes,
        }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("[4/4] 完成。输出:", flush=True)
    for name in (
        "keyframes_recoil.csv",
        "all_frames_motion.csv",
        "summary.json",
        "recoil_trajectory.png",
        "reticle_keyframes_contact_sheet.jpg",
        "feature_mask.png",
    ):
        print(f"      {output_dir / name}", flush=True)
    if auto_ammo:
        print(f"      {output_dir / 'ammo_detection.csv'}", flush=True)
    print(
        f"      自动估算最大 pitch: {estimated_pitch_deg:.4f}° "
        f"(FOV={args.fov_deg:g}° {args.fov_axis}, {args.scope_magnification:g}x)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, cv2.error) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2)
