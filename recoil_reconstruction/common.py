from __future__ import annotations
import argparse,csv,json,math,sys
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Iterable,Sequence
import cv2
import numpy as np
from contextvars import ContextVar
from dataclasses import field
from typing import Callable
REFERENCE_WIDTH=2560
REFERENCE_HEIGHT=1440


class AnalysisCancelled(RuntimeError):
    pass


@dataclass
class RunContext:
    progress: Callable[[str, int, int], None] | None = None
    cancelled: Callable[[], bool] | None = None


RUN_CONTEXT = ContextVar("reconstruction_context", default=RunContext())


def checkpoint(phase: str, current: int = 0, total: int = 0) -> None:
    context = RUN_CONTEXT.get()
    if context.cancelled and context.cancelled():
        raise AnalysisCancelled("Analysis cancelled")
    if context.progress:
        context.progress(phase, current, total)
from .models import *


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
