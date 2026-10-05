from __future__ import annotations
import argparse,csv,json,math,sys
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Iterable,Sequence
import cv2
import numpy as np
REFERENCE_WIDTH=2560
REFERENCE_HEIGHT=1440


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
