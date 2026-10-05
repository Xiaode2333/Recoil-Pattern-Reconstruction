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
