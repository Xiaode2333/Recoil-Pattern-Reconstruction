#!/usr/bin/env python3
"""Build HUD digit templates from a labeled ammunition countdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import analyze_recoil


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="optional reconstruction summary containing verified keyframes",
    )
    parser.add_argument("--start-ammo", type=int, default=50)
    parser.add_argument(
        "--ammo-roi", default="2348,1218,2464,1268"
    )
    parser.add_argument(
        "--output", type=Path, default=analyze_recoil.DEFAULT_AMMO_TEMPLATE_FILE
    )
    return parser.parse_args(argv)


def load_keyframes(args: argparse.Namespace, display_features: np.ndarray) -> list[int]:
    if args.summary is not None and args.summary.is_file():
        payload = json.loads(args.summary.read_text(encoding="utf-8"))
        keyframes = [int(value) for value in payload.get("keyframes", [])]
        if len(keyframes) == args.start_ammo:
            return keyframes
    detection = analyze_recoil.detect_ammo_sequence(
        display_features,
        min_segment=4,
        max_segment=22,
        threshold_mad=3.5,
        minimum_shots=5,
    )
    if len(detection.approximate_keyframes) != args.start_ammo:
        raise RuntimeError(
            "Calibration countdown did not produce the expected "
            f"{args.start_ammo} events: {len(detection.approximate_keyframes)}"
        )
    return analyze_recoil.refine_keyframes_from_display(
        detection.approximate_keyframes, display_features
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    video = args.video.resolve()
    width, height, _fps, _frame_count = analyze_recoil.read_video_info(video)
    roi = analyze_recoil.parse_scaled_roi(args.ammo_roi, width, height)
    _segments, display_features, ocr_features = analyze_recoil.extract_full_ammo_features(
        video, roi
    )
    keyframes = load_keyframes(args, display_features)
    samples: dict[int, list[np.ndarray]] = {digit: [] for digit in range(10)}
    start_ammo = int(args.start_ammo)
    for state in range(start_ammo, -1, -1):
        if state == start_ammo:
            frame = keyframes[0] - 3
        elif state == 0:
            frame = keyframes[-1] + 3
        else:
            shots_fired = start_ammo - state
            frame = (
                keyframes[shots_fired - 1] + keyframes[shots_fired]
            ) // 2
        text = f"{state:03d}"
        first_significant = 2 if state < 10 else 1 if state < 100 else 0
        for position in range(first_significant, 3):
            samples[int(text[position])].append(ocr_features[frame, position])
    missing = [digit for digit, values in samples.items() if not values]
    if missing:
        raise RuntimeError(f"Calibration did not observe digits: {missing}")
    templates = np.asarray(
        [np.mean(samples[digit], axis=0) for digit in range(10)],
        dtype=np.float32,
    )

    correct = 0
    total = 0
    for digit, values in samples.items():
        for cell in values:
            mse = np.mean(np.square(templates - cell[None, :, :]), axis=(1, 2))
            correct += int(int(np.argmin(mse)) == digit)
            total += 1
    accuracy = correct / max(1, total)
    if accuracy < 0.95:
        raise RuntimeError(f"Digit-template self-test accuracy is too low: {accuracy:.3%}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        templates=templates,
        start_ammo=start_ammo,
        roi=np.asarray(roi, dtype=np.int32),
        sample_counts=np.asarray([len(samples[digit]) for digit in range(10)]),
    )
    print(f"Wrote: {output}")
    print(f"Samples: {total}; training accuracy: {accuracy:.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
