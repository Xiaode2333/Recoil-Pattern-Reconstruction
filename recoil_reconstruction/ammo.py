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
    try:
        x0, y0, x1, y1 = roi
        segment_features: list[np.ndarray] = []
        display_features: list[np.ndarray] = []
        ocr_features: list[np.ndarray] = []
        while True:
            if len(segment_features) % 30 == 0:
                checkpoint("scan", len(segment_features), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            ok, frame = cap.read()
            if not ok:
                break
            crop = frame[y0:y1, x0:x1]
            segment_features.append(normalized_ammo_feature(crop))
            display_features.append(ammo_display_feature(crop))
            ocr_features.append(ammo_ocr_cells(crop))
    finally:
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
    """Return the first zero-like frame after a confirmed stable ``001`` state.

    Bright smoke or reload hands can make the translucent leading cell of 000
    resemble 100. Treat 000/100 as one zero-like run only after seeing 001;
    this also prevents an obscured non-empty magazine (for example 004 followed
    by reload animation) from being accepted as a completed magazine.
    """
    one_run = 0
    saw_stable_one = False
    run_start: int | None = None
    run_length = 0
    for frame_index in range(first_event_frame, len(ocr_features)):
        number, error = recognize_ammo_number(ocr_features[frame_index], templates)
        if number == 1 and error <= 0.08:
            one_run += 1
            if one_run >= 2:
                saw_stable_one = True
        elif not saw_stable_one:
            one_run = 0
        if saw_stable_one and number in {0, 100} and error <= 0.08:
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


def refine_regular_clock_viterbi(
    clock: Sequence[int],
    features: np.ndarray,
    frame_offset: int,
    radius: int = 2,
) -> list[int]:
    """Jointly align a regular shot clock to robust n->n-1 HUD transitions.

    The temporal-median change score supplies the image evidence. Dynamic
    programming prevents a single smoke/flash frame from moving one boundary
    independently and breaking the cyclic rate. The reviewed terminal anchor
    is never moved.
    """
    if len(clock) < 2:
        return list(clock)
    scores = _ammo_change_scores(features)
    baseline = float(np.median(scores))
    scale = float(np.median(np.abs(scores - baseline))) + 1e-6
    evidence = np.clip((scores - baseline) / scale, 0.0, 12.0)
    target = float(np.median(np.diff(np.asarray(clock, dtype=np.float64))))
    candidates: list[list[int]] = []
    for index, predicted in enumerate(clock):
        if index == len(clock) - 1:
            candidates.append([int(predicted)])
            continue
        local = int(predicted) - frame_offset
        choices = [
            frame_offset + value
            for value in range(max(0, local - radius), min(len(scores) - 1, local + radius) + 1)
        ]
        candidates.append(choices or [int(predicted)])

    costs: list[dict[int, float]] = []
    parents: list[dict[int, int]] = []
    for index, choices in enumerate(candidates):
        checkpoint("timing", index, len(candidates))
        row: dict[int, float] = {}
        parent_row: dict[int, int] = {}
        for choice in choices:
            local = choice - frame_offset
            emission = -float(evidence[local]) + 0.20 * (choice - clock[index]) ** 2
            if index == 0:
                row[choice] = emission
                continue
            best_parent = None
            best_cost = math.inf
            for previous, previous_cost in costs[-1].items():
                interval = choice - previous
                if interval <= 0:
                    continue
                transition = 1.25 * (interval - target) ** 2
                candidate_cost = previous_cost + transition + emission
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_parent = previous
            if best_parent is not None:
                row[choice] = best_cost
                parent_row[choice] = best_parent
        if not row:
            return list(clock)
        costs.append(row)
        parents.append(parent_row)
    current = min(costs[-1], key=costs[-1].get)
    result = [current]
    for index in range(len(clock) - 1, 0, -1):
        current = parents[index][current]
        result.append(current)
    result.reverse()
    return result


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
        checkpoint("timing", state, segment_count)
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
