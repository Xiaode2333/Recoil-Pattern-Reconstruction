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


def detect_finals_colored_reticle(frame: np.ndarray, color: str) -> ReticleResult:
    """Locate The Finals' red/green optic glyph and use its centre as POI.

    The glyph moves independently inside the optic during recoil.  A compact
    central chroma mask therefore tracks the actual sight indication instead
    of assuming that the screen centre is the bullet position.
    """
    height, width = frame.shape[:2]
    center_x, center_y = width / 2.0, height / 2.0
    scale = min(width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT)
    radius_x = max(90, int(round(180 * scale)))
    radius_y = max(70, int(round(140 * scale)))
    x0 = max(0, int(round(center_x)) - radius_x)
    x1 = min(width, int(round(center_x)) + radius_x + 1)
    y0 = max(0, int(round(center_y)) - radius_y)
    y1 = min(height, int(round(center_y)) + radius_y + 1)
    crop = frame[y0:y1, x0:x1]
    blue, green, red = cv2.split(crop)
    b16 = blue.astype(np.int16)
    g16 = green.astype(np.int16)
    r16 = red.astype(np.int16)
    if color == "green":
        chroma = g16 - np.maximum(r16, b16)
        mask = (green > 100) & (chroma > 22)
    else:
        chroma = r16 - np.maximum(g16, b16)
        mask = (red > 100) & (chroma > 22)
    mask_u8 = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    candidates: list[tuple[float, float, float, float]] = []
    for label in range(1, count):
        left, top, component_width, component_height, area = map(int, stats[label])
        if area < max(8, int(12 * scale * scale)) or area > int(5000 * scale * scale):
            continue
        if component_width < 4 or component_height < 3:
            continue
        component = labels == label
        ys, xs = np.nonzero(component)
        weights = np.maximum(chroma[component].astype(np.float64), 1.0)
        glyph_x = float(x0 + np.average(xs, weights=weights))
        glyph_y = float(y0 + np.average(ys, weights=weights))
        distance = math.hypot(glyph_x - center_x, glyph_y - center_y)
        if distance > max(30.0, 45.0 * scale):
            continue
        score = float(np.mean(chroma[component])) + 0.06 * area - 1.50 * distance
        candidates.append((score, glyph_x, glyph_y, float(np.mean(chroma[component]))))
    if not candidates:
        fail(f"The Finals {color} reticle detection failed")
    score, x, y, purity = max(candidates, key=lambda item: item[0])
    return ReticleResult(x, y, 0.0, score, purity, score - purity)


def detect_finals_front_sight(frame: np.ndarray) -> ReticleResult:
    """Track the top of the central iron-sight post used as the actual POI."""
    height, width = frame.shape[:2]
    center_x, center_y = width / 2.0, height / 2.0
    scale = min(width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT)
    half_width = max(70, int(round(105 * scale)))
    up = max(120, int(round(190 * scale)))
    down = max(130, int(round(190 * scale)))
    x0 = int(round(center_x)) - half_width
    y0 = int(round(center_y)) - up
    crop = frame[y0 : int(round(center_y)) + down, x0 : int(round(center_x)) + half_width]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    darkness = cv2.GaussianBlur(gray, (0, 0), max(5.0, 9.0 * scale)) - gray
    local_cx = half_width
    best: tuple[float, int, int] | None = None
    y_min = max(16, up - int(round(50 * scale)))
    y_max = min(darkness.shape[0] - int(round(48 * scale)), up + int(round(50 * scale)))
    x_radius = max(10, int(round(14 * scale)))
    stem_length = max(24, int(round(44 * scale)))
    stem_half = max(1, int(round(2 * scale)))
    side_near = max(6, int(round(8 * scale)))
    side_far = max(side_near + 2, int(round(16 * scale)))
    for y in range(y_min, y_max):
        for x in range(local_cx - x_radius, local_cx + x_radius + 1):
            stem = float(np.mean(darkness[y : y + stem_length, x - stem_half : x + stem_half + 1]))
            sides = 0.5 * (
                float(np.mean(darkness[y + 4 : y + stem_length, x - side_far : x - side_near]))
                + float(np.mean(darkness[y + 4 : y + stem_length, x + side_near : x + side_far]))
            )
            above = float(np.mean(darkness[y - 12 : y - 3, x - 3 : x + 4]))
            # The true post remains close to the optical axis; this rejects
            # circular sight guards and the tall M60 protective ears.
            score = stem - 0.38 * sides - 0.22 * max(0.0, above) - 0.50 * abs(x - local_cx)
            if best is None or score > best[0]:
                best = (score, x, y)
    if best is None or not math.isfinite(best[0]):
        fail("The Finals front-sight detection failed")
    score, x, y = best
    return ReticleResult(
        x=float(x0 + x), y=float(y0 + y), angle_deg=0.0,
        confidence=float(score), horizontal_score=float(score), vertical_score=float(score),
    )


def detect_reticle(
    frame: np.ndarray,
    mode: str,
    max_angle_deg: float,
    angle_step_deg: float,
) -> ReticleResult:
    if mode == "red-dot":
        return detect_reticle_red_dot(frame)
    if mode == "finals-red":
        return detect_finals_colored_reticle(frame, "red")
    if mode == "finals-green":
        return detect_finals_colored_reticle(frame, "green")
    if mode == "finals-front-sight":
        return detect_finals_front_sight(frame)
    return detect_reticle_crosshair(frame, max_angle_deg, angle_step_deg)


def interpolate_occluded_reticle(
    video: Path,
    frame_index: int,
    mode: str,
    max_angle: float,
    angle_step: float,
    radius: int = 3,
) -> ReticleResult:
    """Interpolate a muzzle-flash-occluded reticle from visible neighbors."""
    observations: dict[int, ReticleResult] = {}
    capture = cv2.VideoCapture(str(video))
    for offset in (*range(-1, -radius - 1, -1), *range(1, radius + 1)):
        sample_frame = frame_index + offset
        if sample_frame < 0:
            continue
        try:
            image = seek_and_read(capture, sample_frame)
            observations[sample_frame] = detect_reticle(
                image, mode, max_angle, angle_step
            )
        except RuntimeError:
            continue
    capture.release()
    before = max((value for value in observations if value < frame_index), default=None)
    after = min((value for value in observations if value > frame_index), default=None)
    if before is None or after is None:
        fail(f"关键帧 {frame_index} 的准星被遮挡，且前后帧不足以插值")
    left, right = observations[before], observations[after]
    weight = (frame_index - before) / (after - before)
    lerp = lambda a, b: float(a + weight * (b - a))
    return ReticleResult(
        x=lerp(left.x, right.x),
        y=lerp(left.y, right.y),
        angle_deg=lerp(left.angle_deg, right.angle_deg),
        confidence=max(0.01, min(left.confidence, right.confidence) * 0.5),
        horizontal_score=lerp(left.horizontal_score, right.horizontal_score),
        vertical_score=lerp(left.vertical_score, right.vertical_score),
    )
