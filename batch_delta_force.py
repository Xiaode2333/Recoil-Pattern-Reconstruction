#!/usr/bin/env python3
"""Batch reconstruct Delta Force recoil videos and import validated profiles.

Every video in D:\\DF is analyzed with automatic ammo-HUD range detection,
Feature Matching + RANSAC background motion, and reticle tick tracking.  The
result is converted to a bilingual Recoil Trainer Workshop-compatible JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = Path(r"D:\DF")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "delta_force_batch_output"
DEFAULT_TRAINER_ROOT = Path(r"C:\XiaodeDocuments\Programs\RecoilTrainer")
DEFAULT_STEAM_DATA_ROOT = Path(os.environ.get("LOCALAPPDATA", "")) / "RecoilTrainer"

# Filenames already use the desired weapon label except where the Chinese and
# English clients use different display names or the capture filename includes
# recording notes.
NAME_OVERRIDES: dict[str, tuple[str, str]] = {
    "RM277_x1_opx2": ("RM277", "RM277"),
    "MCX_LT": ("MCX LT", "MCX LT"),
    "勇士": ("Vityaz", "勇士"),
    "腾龙": ("Tenglong", "腾龙"),
    "野牛": ("Bizon", "野牛"),
}
ONE_X_WEAPONS = {"93R", "G18"}
AMPLITUDE_DIVISORS = {"QBZ95-1": 0.89}


class BatchError(RuntimeError):
    """Raised when the batch cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class WeaponJob:
    video: Path
    stem: str
    slug: str
    name_en: str
    name_zh_cn: str
    scope_magnification: float
    amplitude_divisor: float
    output_dir: Path


@dataclass(slots=True)
class JobResult:
    stem: str
    name_en: str
    name_zh_cn: str
    status: str
    scope_magnification: float
    amplitude_divisor: float
    output_dir: str
    profile_json: str = ""
    shots: int = 0
    duration_ms: int = 0
    pitch_deg: float = 0.0
    segments: int = 0
    failed_steps: int = 0
    min_inliers: int = 0
    min_inlier_ratio: float = 0.0
    min_reticle_confidence: float = 0.0
    analysis_start_frame: int = 0
    analysis_end_frame: int = 0
    error: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-create bilingual Delta Force Recoil Trainer profiles."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recoil-trainer-root", type=Path, default=DEFAULT_TRAINER_ROOT)
    parser.add_argument("--steam-data-root", type=Path, default=DEFAULT_STEAM_DATA_ROOT)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing output only when it still passes every quality check.",
    )
    parser.add_argument(
        "--import-steam",
        action="store_true",
        help="After every profile passes, back up and import them into Steam Recoil Trainer.",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Reuse validated analysis CSV/summary files and rebuild only profile JSON.",
    )
    return parser.parse_args(argv)


def _slug(stem: str) -> str:
    aliases = {"勇士": "vityaz", "腾龙": "tenglong", "野牛": "bizon"}
    raw = aliases.get(stem, stem).lower().replace("_x1_opx2", "")
    cleaned = "".join(char if char.isalnum() else "-" for char in raw)
    return "-".join(part for part in cleaned.split("-") if part)


def discover_jobs(video_dir: Path, output_root: Path) -> list[WeaponJob]:
    videos = sorted(video_dir.glob("*.mp4"), key=lambda path: path.name.casefold())
    if not videos:
        raise BatchError(f"No MP4 videos found in {video_dir}")
    jobs: list[WeaponJob] = []
    seen_slugs: set[str] = set()
    for video in videos:
        stem = video.stem
        name_en, name_zh = NAME_OVERRIDES.get(stem, (stem.replace("_", " "), stem.replace("_", " ")))
        slug = _slug(stem)
        if not slug or slug in seen_slugs:
            raise BatchError(f"Duplicate or invalid output slug for {video.name}: {slug!r}")
        seen_slugs.add(slug)
        jobs.append(
            WeaponJob(
                video=video.resolve(),
                stem=stem,
                slug=slug,
                name_en=name_en,
                name_zh_cn=name_zh,
                scope_magnification=1.0 if stem in ONE_X_WEAPONS else 2.0,
                amplitude_divisor=AMPLITUDE_DIVISORS.get(stem, 1.0),
                output_dir=(output_root / slug).resolve(),
            )
        )
    return jobs


def _run_logged(command: list[str], cwd: Path, log_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        env=environment,
    )
    log_path.write_text(
        "COMMAND\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT\n" + result.stdout
        + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        message = detail[-1] if detail else f"exit code {result.returncode}"
        raise BatchError(f"{log_path.stem} failed: {message}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BatchError(f"JSON root must be an object: {path}")
    return payload


def validate_job(job: WeaponJob) -> JobResult:
    summary_path = job.output_dir / "summary.json"
    csv_path = job.output_dir / "keyframes_recoil.csv"
    profile_path = job.output_dir / f"{job.slug}_recoiltrainer.json"
    summary = _read_json(summary_path)
    profile = _read_json(profile_path)
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise BatchError(f"Cannot read trajectory CSV {csv_path}: {exc}") from exc

    shots = int(summary.get("detected_shot_count", 0))
    failed_steps = int(summary.get("failed_steps_interpolated", 0))
    pitch = float(profile.get("recorded_recoil_pitch_range_deg", 0.0))
    points = profile.get("points", [])
    segments = profile.get("segments", [])
    expected_pitch = float(summary.get("estimated_recoil_pitch_range_deg", 0.0)) / job.amplitude_divisor
    errors: list[str] = []
    if summary.get("pipeline_version") != "delta-force-v6-joint-zero-boundary":
        errors.append("output was produced by an older reconstruction pipeline")
    if summary.get("ammo_detection_mode") != "auto":
        errors.append("ammo range was not detected automatically")
    if summary.get("ammo_count_source") != "template-ocr":
        errors.append("magazine size was not read by template OCR")
    if float(summary.get("ammo_ocr_median_error", 1.0)) > 0.08:
        errors.append(f"ammo OCR error too high: {summary.get('ammo_ocr_median_error')}")
    if not 5 <= shots <= 200:
        errors.append(f"implausible shot count {shots}")
    if len(rows) != shots or len(points) != shots:
        errors.append(f"shot count mismatch summary/csv/json: {shots}/{len(rows)}/{len(points)}")
    if len(summary.get("keyframes", [])) != shots:
        errors.append("keyframe count does not equal shot count")
    zero_boundary = summary.get("ammo_auto_detection", {}).get("ocr_zero_frame")
    stable_zero_boundary = summary.get("ammo_auto_detection", {}).get(
        "ocr_stable_zero_frame"
    )
    if zero_boundary is None or not summary.get("keyframes"):
        errors.append("missing OCR 1-to-0 boundary evidence")
    elif min(
        abs(int(summary["keyframes"][-1]) - int(zero_boundary)),
        abs(int(summary["keyframes"][-1]) - int(stable_zero_boundary))
        if stable_zero_boundary is not None
        else 10**9,
    ) > 3:
        errors.append(
            f"last keyframe is not aligned to OCR 1-to-0 boundary: "
            f"{summary['keyframes'][-1]} vs peak={zero_boundary}, stable={stable_zero_boundary}"
        )
    if failed_steps > max(2, math.ceil((int(summary.get("analysis_end_frame", 0)) - int(summary.get("analysis_start_frame", 0))) * 0.01)):
        errors.append(f"too many interpolated RANSAC steps: {failed_steps}")
    if int(summary.get("minimum_inliers", 0)) < 12:
        errors.append(f"minimum RANSAC inliers too low: {summary.get('minimum_inliers')}")
    if float(summary.get("minimum_inlier_ratio", 0.0)) < 0.25:
        errors.append(f"minimum RANSAC inlier ratio too low: {summary.get('minimum_inlier_ratio')}")
    if float(summary.get("minimum_reticle_confidence", 0.0)) < 5.0:
        errors.append(f"minimum reticle confidence too low: {summary.get('minimum_reticle_confidence')}")
    if not math.isfinite(pitch) or not 0.01 < pitch < 120.0:
        errors.append(f"implausible recorded pitch {pitch}")
    if not math.isclose(pitch, expected_pitch, rel_tol=1e-9, abs_tol=1e-9):
        errors.append(f"pitch/divisor mismatch {pitch} != {expected_pitch}")
    if float(summary.get("scope_magnification", 0.0)) != job.scope_magnification:
        errors.append("scope magnification mismatch")
    expected_reticle_mode = "red-dot" if job.scope_magnification == 1.0 else "tick-lines"
    if summary.get("reticle_mode") != expected_reticle_mode:
        errors.append("reticle detector mode mismatch")
    if profile.get("smoothing") != "spline" or float(profile.get("smoothing_strength", -1)) != 0.2:
        errors.append("profile is not spline smoothing 0.2")
    if not segments:
        errors.append("automatic segmentation produced no segments")
    localizations = profile.get("data", {}).get("localizations", {})
    for locale in ("en-US", "zh-CN"):
        item = localizations.get(locale, {})
        for field in ("game_name", "weapon_name", "workshop_title"):
            if not str(item.get(field, "")).strip():
                errors.append(f"missing {locale} {field}")
    expected_title_en = f"Delta Force · {job.name_en}"
    expected_title_zh = f"三角洲·{job.name_zh_cn}"
    if profile.get("name") != expected_title_en:
        errors.append("primary profile title mismatch")
    if profile.get("localized_names", {}).get("en-US") != expected_title_en:
        errors.append("en-US localized profile name mismatch")
    if profile.get("localized_names", {}).get("zh-CN") != expected_title_zh:
        errors.append("zh-CN localized profile name mismatch")
    expected_localizations = {
        "en-US": {
            "game_name": "Delta Force",
            "weapon_name": job.name_en,
            "workshop_title": f"Delta Force · {job.name_en} Recoil",
        },
        "zh-CN": {
            "game_name": "三角洲行动",
            "weapon_name": job.name_zh_cn,
            "workshop_title": f"三角洲·{job.name_zh_cn} 后坐力",
        },
    }
    for locale, expected in expected_localizations.items():
        actual = localizations.get(locale, {})
        for field, value in expected.items():
            if actual.get(field) != value:
                errors.append(f"{locale} {field} mismatch")
    summary_corrections = summary.get("trajectory_corrections", [])
    profile_reconstruction = profile.get("data", {}).get("reconstruction", {})
    if profile_reconstruction.get("trajectory_corrections", []) != summary_corrections:
        errors.append("trajectory correction audit metadata mismatch")
    if summary_corrections:
        required_trainer_columns = {
            "trainer_recoil_x_right_px",
            "trainer_recoil_y_up_px",
            "trainer_shot_delta_x_right_px",
            "trainer_shot_delta_y_up_px",
            "trajectory_correction",
        }
        if not rows or not required_trainer_columns.issubset(rows[-1]):
            errors.append("terminal repair is missing trainer coordinate columns")
        else:
            trainer_terminal_delta = math.hypot(
                float(rows[-1]["trainer_shot_delta_x_right_px"]),
                float(rows[-1]["trainer_shot_delta_y_up_px"]),
            )
            raw_terminal_delta = math.hypot(
                float(rows[-1]["shot_delta_x_right_px"]),
                float(rows[-1]["shot_delta_y_up_px"]),
            )
            if trainer_terminal_delta >= raw_terminal_delta:
                errors.append("terminal repair did not reduce the last-point outlier")
    if errors:
        raise BatchError("; ".join(errors))

    duration_ms = int(points[-1]["t_ms"])
    if duration_ms <= 0:
        raise BatchError(f"invalid firing duration {duration_ms} ms")
    return JobResult(
        stem=job.stem,
        name_en=job.name_en,
        name_zh_cn=job.name_zh_cn,
        status="passed",
        scope_magnification=job.scope_magnification,
        amplitude_divisor=job.amplitude_divisor,
        output_dir=str(job.output_dir),
        profile_json=str(profile_path),
        shots=shots,
        duration_ms=duration_ms,
        pitch_deg=pitch,
        segments=len(segments),
        failed_steps=failed_steps,
        min_inliers=int(summary.get("minimum_inliers", 0)),
        min_inlier_ratio=float(summary.get("minimum_inlier_ratio", 0.0)),
        min_reticle_confidence=float(summary.get("minimum_reticle_confidence", 0.0)),
        analysis_start_frame=int(summary.get("analysis_start_frame", 0)),
        analysis_end_frame=int(summary.get("analysis_end_frame", 0)),
    )


def process_job(job: WeaponJob, args: argparse.Namespace) -> JobResult:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    convert_only = bool(getattr(args, "convert_only", False))
    if args.resume and not convert_only:
        try:
            result = validate_job(job)
            preserved_profile_id = getattr(args, "preserved_profile_ids", {}).get(job.stem)
            if preserved_profile_id:
                profile_payload = json.loads(
                    Path(result.profile_json).read_text(encoding="utf-8")
                )
                if profile_payload.get("profile_id") != preserved_profile_id:
                    raise BatchError("profile ID changed; rebuild is required for in-place Steam update")
            return result
        except BatchError:
            pass

    if convert_only:
        summary_path = job.output_dir / "summary.json"
        csv_path = job.output_dir / "keyframes_recoil.csv"
        if not summary_path.is_file() or not csv_path.is_file():
            raise BatchError(f"convert-only analysis artifacts are missing for {job.stem}")
        if _read_json(summary_path).get("pipeline_version") != "delta-force-v6-joint-zero-boundary":
            raise BatchError(f"convert-only analysis is stale for {job.stem}")
    else:
        analyze_command = [
            sys.executable,
            str(SCRIPT_DIR / "analyze_recoil.py"),
            str(job.video),
            "--output-dir",
            str(job.output_dir),
            "--fov-deg",
            "104",
            "--fov-axis",
            "reference-horizontal",
            "--scope-magnification",
            str(job.scope_magnification),
            "--reticle-mode",
            "red-dot" if job.scope_magnification == 1.0 else "tick-lines",
            "--ammo-template-file",
            str(SCRIPT_DIR / "ammo_digit_templates.npz"),
        ]
        _run_logged(analyze_command, SCRIPT_DIR, job.output_dir / "analyze.log")

    csv_path = job.output_dir / "keyframes_recoil.csv"
    profile_path = job.output_dir / f"{job.slug}_recoiltrainer.json"
    tags = ",".join(
        [
            "game:delta-force",
            "Delta Force",
            f"weapon:{job.name_en}",
            "video-reconstruction",
            f"scope:{job.scope_magnification:g}x",
        ]
    )
    convert_command = [
        sys.executable,
        str(SCRIPT_DIR / "convert_to_recoiltrainer.py"),
        str(csv_path),
        "--output",
        str(profile_path),
        "--name",
        f"Delta Force · {job.name_en}",
        "--localized-name-en",
        f"Delta Force · {job.name_en}",
        "--localized-name-zh-cn",
        f"三角洲·{job.name_zh_cn}",
        "--weapon-name-en",
        job.name_en,
        "--weapon-name-zh-cn",
        job.name_zh_cn,
        "--game-name-en",
        "Delta Force",
        "--game-name-zh-cn",
        "三角洲行动",
        "--workshop-title-en",
        f"Delta Force · {job.name_en} Recoil",
        "--workshop-title-zh-cn",
        f"三角洲·{job.name_zh_cn} 后坐力",
        "--scope-magnification",
        str(job.scope_magnification),
        "--recoil-amplitude-divisor",
        str(job.amplitude_divisor),
        "--smoothing",
        "spline",
        "--smoothing-strength",
        "0.2",
        "--segmentation",
        "recoil-trainer",
        "--tags",
        tags,
        "--recoil-trainer-root",
        str(args.recoil_trainer_root.resolve()),
    ]
    preserved_profile_id = getattr(args, "preserved_profile_ids", {}).get(job.stem)
    if preserved_profile_id:
        convert_command.extend(["--profile-id", preserved_profile_id])
    _run_logged(convert_command, SCRIPT_DIR, job.output_dir / "convert.log")
    return validate_job(job)


def _trainer_is_running() -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq RecoilTrainer.exe", "/FO", "CSV", "/NH"],
        text=True,
        capture_output=True,
        check=False,
    )
    return "RecoilTrainer.exe" in result.stdout


def import_into_steam(
    results: list[JobResult], trainer_root: Path, steam_data_root: Path, output_root: Path
) -> dict[str, Any]:
    if _trainer_is_running():
        raise BatchError("RecoilTrainer.exe is running; close it before importing profiles")
    db_path = steam_data_root / "assets" / "data" / "app.db"
    profiles_dir = steam_data_root / "assets" / "profiles"
    if not db_path.is_file() or not profiles_dir.is_dir():
        raise BatchError(f"Steam Recoil Trainer data directory is incomplete: {steam_data_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = steam_data_root / "codex_backups" / f"delta_force_batch_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(db_path, backup_dir / "app.db")
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.is_file():
            shutil.copy2(sidecar, backup_dir / sidecar.name)
    shutil.copytree(profiles_dir, backup_dir / "profiles")

    sources = [str(Path(result.profile_json).resolve()) for result in results]
    poetry = shutil.which("poetry")
    if poetry is None:
        raise BatchError("Poetry is required to import through Recoil Trainer's repository")
    runner = (
        "import json,os;from pathlib import Path;"
        "request=json.loads(os.environ.pop('RECOIL_IMPORT_REQUEST'));"
        "from src.core.storage.profile_repo import ProfileRepository;"
        "repo=ProfileRepository(Path(request['db']),Path(request['profiles']));"
        "loaded=[repo.import_json_file(Path(item)) for item in request['sources']];"
        "assert all(loaded),'one or more profiles failed to import';"
        "print('RECOIL_IMPORTED='+json.dumps([item.profile_id for item in loaded]))"
    )
    environment = dict(os.environ)
    environment["RECOIL_IMPORT_REQUEST"] = json.dumps(
        {"db": str(db_path), "profiles": str(profiles_dir), "sources": sources},
        ensure_ascii=False,
    )
    command = [poetry, "run", "python", "-c", runner]
    run = subprocess.run(
        command,
        cwd=trainer_root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        env=environment,
    )
    if run.returncode != 0:
        raise BatchError("Steam profile repository import failed: " + (run.stderr or run.stdout).strip())
    marker = "RECOIL_IMPORTED="
    line = next((line for line in run.stdout.splitlines() if line.startswith(marker)), "")
    if not line:
        raise BatchError("Steam profile repository import returned no profile IDs")
    imported_ids = json.loads(line[len(marker) :])
    if len(imported_ids) != len(results):
        raise BatchError("Steam profile repository imported an unexpected profile count")

    # Keep the bilingual Workshop metadata in the on-disk JSON. ProfileRepository
    # has already registered the validated core fields in SQLite.
    for source, profile_id in zip(sources, imported_ids):
        shutil.copy2(source, profiles_dir / f"{profile_id}.json")

    with sqlite3.connect(db_path) as connection:
        placeholders = ",".join("?" for _ in imported_ids)
        count = connection.execute(
            f"SELECT COUNT(*) FROM profiles WHERE profile_id IN ({placeholders})",
            imported_ids,
        ).fetchone()[0]
    if count != len(imported_ids):
        raise BatchError(f"Post-import DB verification found {count}/{len(imported_ids)} profiles")

    report = {
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "steam_data_root": str(steam_data_root),
        "backup_dir": str(backup_dir),
        "profile_count": len(imported_ids),
        "profile_ids": imported_ids,
    }
    (output_root / "steam_import_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def write_manifest(output_root: Path, jobs: list[WeaponJob], results: list[JobResult]) -> Path:
    by_stem = {result.stem: result for result in results}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video_count": len(jobs),
        "passed_count": sum(result.status == "passed" for result in results),
        "failed_count": sum(result.status != "passed" for result in results),
        "settings": {
            "fov_degrees": 104.0,
            "fov_axis": "reference-horizontal",
            "one_x_weapons": sorted(ONE_X_WEAPONS),
            "default_scope_magnification": 2.0,
            "amplitude_divisors": AMPLITUDE_DIVISORS,
            "smoothing": "spline",
            "smoothing_strength": 0.2,
            "segmentation": "Recoil Trainer Auto Segment",
            "locales": ["en-US", "zh-CN"],
        },
        "profiles": [asdict(by_stem[job.stem]) for job in jobs if job.stem in by_stem],
    }
    path = output_root / "batch_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_preserved_profile_ids(output_root: Path) -> dict[str, str]:
    """Recover the IDs from the previous Steam import so reruns update in place."""
    manifest_path = output_root / "batch_manifest.json"
    report_path = output_root / "steam_import_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        profiles = manifest.get("profiles", [])
        profile_ids = report.get("profile_ids", [])
        if len(profiles) != len(profile_ids):
            return {}
        return {
            str(profile["stem"]): str(profile_id)
            for profile, profile_id in zip(profiles, profile_ids)
            if profile.get("stem") and profile_id
        }
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_workers < 1 or args.max_workers > 8:
        print("ERROR: --max-workers must be between 1 and 8", file=sys.stderr)
        return 2
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    args.preserved_profile_ids = load_preserved_profile_ids(output_root)
    template_path = SCRIPT_DIR / "ammo_digit_templates.npz"
    if not template_path.is_file():
        print("Calibrating Delta Force ammo HUD digit templates from AR57...", flush=True)
        try:
            _run_logged(
                [sys.executable, str(SCRIPT_DIR / "calibrate_ammo_templates.py")],
                SCRIPT_DIR,
                output_root / "ammo_template_calibration.log",
            )
        except BatchError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    try:
        jobs = discover_jobs(args.video_dir.resolve(), output_root)
    except BatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Discovered {len(jobs)} videos; workers={args.max_workers}; "
        f"preserved Steam IDs={len(args.preserved_profile_ids)}",
        flush=True,
    )
    results: list[JobResult] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures: dict[Future[JobResult], WeaponJob] = {
            executor.submit(process_job, job, args): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"PASS {job.stem}: {result.shots} shots, {result.pitch_deg:.3f} deg, "
                    f"{result.segments} segments, frames {result.analysis_start_frame}-{result.analysis_end_frame}",
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                result = JobResult(
                    stem=job.stem,
                    name_en=job.name_en,
                    name_zh_cn=job.name_zh_cn,
                    status="failed",
                    scope_magnification=job.scope_magnification,
                    amplitude_divisor=job.amplitude_divisor,
                    output_dir=str(job.output_dir),
                    error=str(exc),
                )
                results.append(result)
                print(f"FAIL {job.stem}: {exc}", flush=True)

    manifest = write_manifest(output_root, jobs, results)
    print(f"Manifest: {manifest}", flush=True)
    if failures:
        print(f"ERROR: {failures}/{len(jobs)} profiles failed; Steam import was not attempted", file=sys.stderr)
        return 1

    if args.import_steam:
        try:
            report = import_into_steam(
                sorted(results, key=lambda result: result.stem.casefold()),
                args.recoil_trainer_root.resolve(),
                args.steam_data_root.resolve(),
                output_root,
            )
        except BatchError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3
        print(
            f"Imported {report['profile_count']} profiles into Steam Recoil Trainer; "
            f"backup: {report['backup_dir']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
