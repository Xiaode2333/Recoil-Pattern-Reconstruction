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
from .ammo import *
from .tracking import *
from .reticle import *
from .output import *
PIPELINE_VERSION="delta-force-v9-temporal-ammo-viterbi"
DEFAULT_AMMO_TEMPLATE_FILE=Path(__file__).resolve().with_name("ammo_digit_templates.npz")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "检测弹药变化关键帧；在瞄准镜外做 SIFT/ORB Feature Matching + "
            "RANSAC；在镜内检测刻度线交点；重建后坐力轨迹。"
        )
    )
    parser.add_argument(
        "video",
        help="输入视频路径",
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
        "--terminal-keyframe",
        type=int,
        default=None,
        help="手动模式中经弹药/画面审核确认的最后一发关键帧",
    )
    parser.add_argument(
        "--reviewed-keyframes",
        default="",
        help="逗号分隔、逐帧审核过的关键帧；仅用于无法由 HUD 恢复的特殊连发录制",
    )
    parser.add_argument(
        "--trainer-tail-extrapolation-count",
        type=int,
        default=0,
        help="仅对确认未录到的尾部弹数，用此前可靠增量补齐 Trainer 轨迹；原始观测仍保留",
    )
    parser.add_argument(
        "--cadence-model",
        choices=("regular", "burst", "warmup"),
        default="regular",
        help="射速校验模型：普通稳定射速、连发/点射、或预热加速",
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
        choices=(
            "tick-lines",
            "red-dot",
            "finals-red",
            "finals-green",
            "finals-front-sight",
        ),
        default="tick-lines",
        help="准星检测模式；The Finals 可使用彩色准星或机械前准星模式",
    )
    parser.add_argument(
        "--output-dir", default="recoil_output", help="CSV/JSON/可视化输出目录"
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
        help="FOV 模型；默认使用 Recoil Trainer 的 4:3 参考横向 FOV",
    )
    parser.add_argument(
        "--scope-magnification",
        type=float,
        default=2.0,
        help="录制时瞄准镜倍率（默认 2x）",
    )
    parser.add_argument("--anchor-roi", default="", help="可选固定墙面参考 x0,y0,x1,y1，独立检查累积轨迹")
    return parser.parse_args(argv)


def audit_keyframe_cadence(keyframes: Sequence[int], model: str) -> dict[str, object]:
    """Validate cadence while preserving legitimate burst and warmup patterns."""
    intervals = [int(b - a) for a, b in zip(keyframes, keyframes[1:])]
    if not intervals:
        return {"model": model, "passed": True, "intervals": intervals, "violations": []}
    median = float(np.median(intervals))
    violations: list[dict[str, object]] = []
    if model == "regular":
        tolerance = max(2.0, median * 0.20)
        for index, value in enumerate(intervals, start=2):
            if abs(value - median) > tolerance:
                violations.append({"shot": index, "interval_frames": value, "reason": "unstable_regular_cadence"})
    elif model == "burst":
        # Trigger timing can vary between bursts; only reject intervals too
        # short to be a real cyclic shot or implausibly long for this capture.
        for index, value in enumerate(intervals, start=2):
            if value < 4 or value > max(12, median * 4.0):
                violations.append({"shot": index, "interval_frames": value, "reason": "invalid_burst_cadence"})
    else:
        for index, (previous, value) in enumerate(zip(intervals, intervals[1:]), start=3):
            if value > previous + max(3, round(previous * 0.35)) or value < max(2, previous * 0.45):
                violations.append({"shot": index, "interval_frames": value, "reason": "invalid_warmup_cadence_change"})
    return {
        "model": model,
        "passed": not violations,
        "median_interval_frames": median,
        "intervals": intervals,
        "violations": violations,
    }


def run_analysis(args: argparse.Namespace) -> int:
    checkpoint("scan")
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
    if auto_ammo and args.terminal_keyframe is not None:
        fail("terminal-keyframe 只能与完整的手动弹药参数一起使用")
    if auto_ammo and args.reviewed_keyframes:
        fail("reviewed-keyframes 只能用于完整手动模式")
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
        if args.terminal_keyframe is not None and not (
            start_frame < args.terminal_keyframe <= end_frame
        ):
            fail("terminal-keyframe 必须位于手动分析范围内且晚于 start-frame")
        print(
            f"[1/4] 视频 {width}x{height} @ {fps:.3f} fps；"
            f"手动提取弹药 ROI {ammo_roi} ...",
            flush=True,
        )
        ammo_features = extract_ammo_features(video, start_frame, end_frame, ammo_roi)
        cadence_weight = args.ammo_cadence_weight

    reviewed_keyframes = (
        [int(value.strip()) for value in args.reviewed_keyframes.split(",") if value.strip()]
        if args.reviewed_keyframes else []
    )
    if reviewed_keyframes:
        if len(reviewed_keyframes) != shot_count:
            fail(f"reviewed-keyframes 数量异常: {len(reviewed_keyframes)} != {shot_count}")
        if reviewed_keyframes != sorted(set(reviewed_keyframes)):
            fail("reviewed-keyframes 必须严格递增且不重复")
        if reviewed_keyframes[0] < start_frame or reviewed_keyframes[-1] > end_frame:
            fail("reviewed-keyframes 超出手动分析范围")
        if args.terminal_keyframe is not None and reviewed_keyframes[-1] != args.terminal_keyframe:
            fail("reviewed-keyframes 的末帧必须等于 terminal-keyframe")
        keyframes = reviewed_keyframes
        provisional = keyframes
    elif ammo_detection is not None:
        # Segment only the non-empty states and hard-lock the last shot to the
        # independently OCR-confirmed first stable 000 frame. Including a full
        # post-zero cadence in the DP lets its equal-duration penalty push the
        # terminal boundary into ADS-exit/reload animation.
        assert ammo_stable_zero_frame is not None
        pre_zero_features = full_ammo_features[
            start_frame:ammo_stable_zero_frame
        ]
        endpoints = segment_fixed_count_sequence(
            pre_zero_features,
            segment_count=shot_count,
            min_length=args.ammo_min_segment,
            max_length=max(args.ammo_max_segment, args.ammo_max_segment * 3),
            cadence_weight=cadence_weight,
        )
        provisional = [start_frame + endpoint for endpoint in endpoints]
        keyframes = refine_keyframes_from_display(
            provisional[:-1], display_features
        ) + [ammo_stable_zero_frame]
    elif args.terminal_keyframe is not None:
        terminal_keyframe = int(args.terminal_keyframe)
        pre_terminal_features = ammo_features[: terminal_keyframe - start_frame]
        endpoints = segment_fixed_count_sequence(
            pre_terminal_features,
            segment_count=shot_count,
            min_length=args.ammo_min_segment,
            max_length=max(args.ammo_max_segment, args.ammo_max_segment * 3),
            cadence_weight=cadence_weight,
        )
        provisional = [start_frame + endpoint for endpoint in endpoints]
        keyframes = refine_keyframes_from_display(
            provisional[:-1], ammo_features, radius=4
        ) + [terminal_keyframe]
    else:
        endpoints = segment_fixed_count_sequence(
            ammo_features,
            segment_count=shot_count + 1,
            min_length=args.ammo_min_segment,
            max_length=args.ammo_max_segment,
            cadence_weight=cadence_weight,
        )
        keyframes = [start_frame + endpoint for endpoint in endpoints[:-1]]
        # Fixed-count capture ranges still benefit from snapping each dynamic-
        # programming boundary to the strongest local HUD glyph transition.
        local_keyframes = [frame - start_frame for frame in keyframes]
        keyframes = [
            start_frame + frame
            for frame in refine_keyframes_from_display(
                local_keyframes, ammo_features, radius=4
            )
        ]
    if len(keyframes) != shot_count:
        fail(f"关键帧数量异常: {len(keyframes)} != {shot_count}")
    display_boundary_candidates = list(keyframes)
    keyframe_decoder = "fixed-state-segmentation"
    if args.cadence_model == "regular" and not reviewed_keyframes:
        stable_clock = [
            int(round(value))
            for value in np.linspace(keyframes[0], keyframes[-1], shot_count)
        ]
        decoder_features = display_features if ammo_detection is not None else ammo_features
        decoder_offset = 0 if ammo_detection is not None else start_frame
        decoded_keyframes = refine_regular_clock_viterbi(
            stable_clock, decoder_features, decoder_offset
        )
        decoded_audit = audit_keyframe_cadence(decoded_keyframes, args.cadence_model)
        if decoded_audit["passed"] and len(set(decoded_keyframes)) == shot_count:
            keyframes = decoded_keyframes
            keyframe_decoder = "cadence-constrained-temporal-median-viterbi"
    cadence_audit = audit_keyframe_cadence(keyframes, args.cadence_model)
    if not cadence_audit["passed"] and args.cadence_model == "regular" and not reviewed_keyframes:
        # HUD peak snapping can select smoke/reload edges. The joint-state DP
        # candidate is preferred when it alone restores an ordinary gun's
        # stable firing rhythm.
        locked_terminal = keyframes[-1]
        cadence_candidate = provisional[:-1] + [locked_terminal]
        candidate_audit = audit_keyframe_cadence(cadence_candidate, args.cadence_model)
        if candidate_audit["passed"]:
            keyframes = cadence_candidate
            cadence_audit = candidate_audit
        else:
            # An ordinary automatic weapon has a fixed cyclic rate. Anchor the
            # first DP transition and the independently reviewed final shot,
            # then quantize the intervening shot clock to video frames. This
            # prevents muzzle flash/HUD smoke from fabricating early/late shots.
            stable_clock = [
                int(round(value))
                for value in np.linspace(provisional[0], locked_terminal, shot_count)
            ]
            clock_audit = audit_keyframe_cadence(stable_clock, args.cadence_model)
            if clock_audit["passed"] and len(set(stable_clock)) == shot_count:
                keyframes = stable_clock
                cadence_audit = clock_audit
    if not cadence_audit["passed"]:
        fail(f"射速校验失败 ({args.cadence_model}): {cadence_audit['violations']}")
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
    try:
        first_frame = seek_and_read(cap, start_frame)
        from .quality import AnchorTracker
        anchor = AnchorTracker(first_frame, parse_scaled_roi(args.anchor_roi, width, height)) if args.anchor_roi else None
        anchor_rows = []
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
        reticle_interpolated_keyframes: list[int] = []
        for frame_index in range(start_frame + 1, end_frame + 1):
            checkpoint("motion", frame_index - start_frame, end_frame - start_frame)
            ok, frame = cap.read()
            if not ok:
                fail(f"运动分析时无法读取第 {frame_index} 帧")
            current_features = matcher.extract(frame)
            if anchor is not None:
                anchor_rows.append(anchor.track(frame, frame_index))
            steps.append(
                matcher.match(frame_index, previous_features, current_features)
            )
            previous_features = current_features
            if frame_index in keyframe_set:
                try:
                    reticle = detect_reticle(
                        frame,
                        args.reticle_mode,
                        args.reticle_max_angle,
                        args.reticle_angle_step,
                    )
                except RuntimeError:
                    reticle = interpolate_occluded_reticle(
                        video,
                        frame_index,
                        args.reticle_mode,
                        args.reticle_max_angle,
                        args.reticle_angle_step,
                    )
                    reticle_interpolated_keyframes.append(frame_index)
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
    finally:
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
    trajectory_corrections.extend(
        apply_reviewed_tail_extrapolation(
            key_rows, args.trainer_tail_extrapolation_count
        )
    )

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
        "pipeline_version": PIPELINE_VERSION,
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
        "cadence_audit": cadence_audit,
        "display_boundary_candidates": display_boundary_candidates,
        "terminal_keyframe_source": (
            "manual-reviewed-keyframes" if reviewed_keyframes
            else "manual-reviewed" if args.terminal_keyframe is not None
            else "ocr-stable-zero"
        ),
        "keyframe_decoder": keyframe_decoder,
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
        "reticle_interpolated_keyframes": reticle_interpolated_keyframes,
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
    summary["quality_schema_version"] = 1
    summary["anchor_roi"] = args.anchor_roi or None
    summary["anchor_track"] = anchor_rows
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


def main(args: argparse.Namespace | None = None, *, progress=None, cancelled=None) -> int:
    token = RUN_CONTEXT.set(RunContext(progress=progress, cancelled=cancelled))
    try:
        return run_analysis(args or parse_args())
    finally:
        RUN_CONTEXT.reset(token)
