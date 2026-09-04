#!/usr/bin/env python3
"""Run the reconstruction pipeline over a directory of videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(slots=True)
class Result:
    video: str
    output_dir: str
    status: str
    elapsed_seconds: float
    return_code: int
    error: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", type=Path, help="directory containing videos")
    parser.add_argument("--output-dir", type=Path, default=Path("batch_output"))
    parser.add_argument("--pattern", default="*.mp4", help="input glob (default: *.mp4)")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--analyzer-arg",
        action="append",
        default=[],
        help="argument forwarded to analyze_recoil.py; repeat as needed",
    )
    return parser.parse_args(argv)


def discover_videos(video_dir: Path, pattern: str) -> list[Path]:
    if not video_dir.is_dir():
        raise ValueError(f"Video directory does not exist: {video_dir}")
    videos = sorted(video_dir.glob(pattern), key=lambda path: path.name.casefold())
    if not videos:
        raise ValueError(f"No videos matching {pattern!r} in {video_dir}")
    return videos


def safe_stem(path: Path) -> str:
    cleaned = "".join(char if char.isalnum() else "-" for char in path.stem.lower())
    return "-".join(part for part in cleaned.split("-") if part) or "video"


def run_one(video: Path, output_root: Path, analyzer_args: list[str]) -> Result:
    output_dir = output_root / safe_stem(video)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "analyze_recoil.py"),
        str(video),
        "--output-dir",
        str(output_dir),
        *analyzer_args,
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    (output_dir / "analyze.log").write_text(
        completed.stdout + "\n" + completed.stderr, encoding="utf-8"
    )
    error = ""
    if completed.returncode:
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        error = lines[-1] if lines else f"exit code {completed.returncode}"
    return Result(
        video=video.name,
        output_dir=str(output_dir),
        status="ok" if completed.returncode == 0 else "failed",
        elapsed_seconds=round(elapsed, 3),
        return_code=completed.returncode,
        error=error,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_workers < 1:
        print("ERROR: --max-workers must be positive", file=sys.stderr)
        return 2
    try:
        videos = discover_videos(args.video_dir.resolve(), args.pattern)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        pending = {
            executor.submit(run_one, video, output_root, args.analyzer_arg): video
            for video in videos
        }
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(f"[{result.status}] {result.video} ({result.elapsed_seconds:.1f}s)")

    results.sort(key=lambda item: item.video.casefold())
    manifest = {
        "videos": len(results),
        "succeeded": sum(item.status == "ok" for item in results),
        "failed": sum(item.status != "ok" for item in results),
        "results": [asdict(item) for item in results],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if manifest["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
