#!/usr/bin/env python3
"""Record a CFR screen video and Windows Raw Input mouse data on one clock.

The recorder is intentionally passive: it captures desktop frames through DXGI
Desktop Duplication and observes Raw Input packets.  It never injects input,
hooks the game process, or reads game memory.

The encoded MP4 is constant-frame-rate.  Frame N represents
``session_start_qpc + N / fps``; if capture or encoding misses a slot, the last
available image is repeated.  ``video_frames.csv`` preserves the actual DXGI
presentation timestamps, so downstream analysis does not have to pretend that
every source frame arrived perfectly on time.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Iterable


if sys.platform != "win32":
    raise SystemExit("record_mouse_video.py only supports Windows.")


try:
    import dxcam
except ImportError as exc:  # pragma: no cover - exercised on an unprepared PC
    raise SystemExit(
        "Missing dependency 'dxcam'. Run: python -m pip install -r requirements.txt"
    ) from exc


NANOSECONDS_PER_SECOND = 1_000_000_000
MILLISECONDS_PER_SECOND = 1_000

WM_INPUT = 0x00FF
WM_HOTKEY = 0x0312
WM_DESTROY = 0x0002
WM_APP_START = 0x8001
WM_APP_STOP = 0x8002
WM_APP_QUIT = 0x8003

RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIDEV_INPUTSINK = 0x00000100

HOTKEY_RECORD_ID = 1
HOTKEY_QUIT_ID = 2

MOUSE_MOVE_ABSOLUTE = 0x0001
RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
RI_MOUSE_LEFT_BUTTON_UP = 0x0002
RI_MOUSE_RIGHT_BUTTON_DOWN = 0x0004
RI_MOUSE_RIGHT_BUTTON_UP = 0x0008
RI_MOUSE_MIDDLE_BUTTON_DOWN = 0x0010
RI_MOUSE_MIDDLE_BUTTON_UP = 0x0020
RI_MOUSE_BUTTON_4_DOWN = 0x0040
RI_MOUSE_BUTTON_4_UP = 0x0080
RI_MOUSE_BUTTON_5_DOWN = 0x0100
RI_MOUSE_BUTTON_5_UP = 0x0200
RI_MOUSE_WHEEL = 0x0400
RI_MOUSE_HWHEEL = 0x0800

VK_F1 = 0x70
VK_F24 = 0x87

CREATE_NO_WINDOW = 0x08000000

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT,
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
)


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class RAWMOUSEBUTTONS(ctypes.Structure):
    _fields_ = [
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData", wintypes.USHORT),
    ]


class RAWMOUSEBUTTONUNION(ctypes.Union):
    _anonymous_ = ("buttons",)
    _fields_ = [
        ("ulButtons", wintypes.ULONG),
        ("buttons", RAWMOUSEBUTTONS),
    ]


class RAWMOUSE(ctypes.Structure):
    _anonymous_ = ("button_union",)
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("button_union", RAWMOUSEBUTTONUNION),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class RAWINPUTDATA(ctypes.Union):
    _fields_ = [("mouse", RAWMOUSE)]


class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("data", RAWINPUTDATA)]


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HCURSOR),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


user32.DefWindowProcW.argtypes = [
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
]
user32.DefWindowProcW.restype = LRESULT
user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
user32.RegisterClassExW.restype = wintypes.ATOM
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
]
user32.CreateWindowExW.restype = wintypes.HWND
user32.DestroyWindow.argtypes = [wintypes.HWND]
user32.DestroyWindow.restype = wintypes.BOOL
user32.RegisterRawInputDevices.argtypes = [
    ctypes.POINTER(RAWINPUTDEVICE),
    wintypes.UINT,
    wintypes.UINT,
]
user32.RegisterRawInputDevices.restype = wintypes.BOOL
user32.GetRawInputData.argtypes = [
    wintypes.HANDLE,
    wintypes.UINT,
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.UINT),
    wintypes.UINT,
]
user32.GetRawInputData.restype = wintypes.UINT
user32.RegisterHotKey.argtypes = [
    wintypes.HWND,
    ctypes.c_int,
    wintypes.UINT,
    wintypes.UINT,
]
user32.RegisterHotKey.restype = wintypes.BOOL
user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
user32.UnregisterHotKey.restype = wintypes.BOOL
user32.PostMessageW.argtypes = [
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
]
user32.PostMessageW.restype = wintypes.BOOL
user32.GetMessageTime.restype = wintypes.LONG
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
kernel32.GetTickCount64.restype = ctypes.c_ulonglong
kernel32.GetCurrentThread.restype = wintypes.HANDLE
kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
kernel32.SetThreadPriority.restype = wintypes.BOOL


@dataclass(slots=True)
class MouseEvent:
    event_index: int
    qpc_ns: int
    session_time_ns: int
    message_qpc_estimate_ns: int
    queue_latency_us: float
    dx_counts: int
    dy_counts: int
    dt_us: float
    vx_counts_s: float
    vy_counts_s: float
    speed_counts_s: float
    movement_mode: str
    button_flags: int
    button_names: str
    wheel_delta: int
    left_button_down: bool
    device_handle: str


@dataclass(slots=True)
class VideoFrameRow:
    frame_index: int
    video_time_ns: int
    expected_qpc_ns: int
    source_qpc_ns: int
    effective_qpc_ns: int
    retrieved_qpc_ns: int
    frame_kind: str
    timestamp_source: str
    source_age_ms: float
    timeline_error_ms: float


def parse_hotkey(value: str) -> int:
    normalized = value.strip().lower()
    if not normalized.startswith("f") or not normalized[1:].isdigit():
        raise argparse.ArgumentTypeError("hotkey must be F1 through F24")
    number = int(normalized[1:])
    vk = VK_F1 + number - 1
    if number < 1 or vk > VK_F24:
        raise argparse.ArgumentTypeError("hotkey must be F1 through F24")
    return vk


def hotkey_name(vk: int) -> str:
    return f"F{vk - VK_F1 + 1}"


def parse_region(value: str) -> tuple[int, int, int, int]:
    try:
        parts = tuple(int(piece.strip()) for piece in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "region must be left,top,right,bottom"
        ) from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be left,top,right,bottom")
    left, top, right, bottom = parts
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("region coordinates are invalid")
    return left, top, right, bottom


def signed_short(value: int) -> int:
    return ctypes.c_short(value).value


def decode_buttons(flags: int) -> list[str]:
    names: list[str] = []
    mapping = (
        (RI_MOUSE_LEFT_BUTTON_DOWN, "left_down"),
        (RI_MOUSE_LEFT_BUTTON_UP, "left_up"),
        (RI_MOUSE_RIGHT_BUTTON_DOWN, "right_down"),
        (RI_MOUSE_RIGHT_BUTTON_UP, "right_up"),
        (RI_MOUSE_MIDDLE_BUTTON_DOWN, "middle_down"),
        (RI_MOUSE_MIDDLE_BUTTON_UP, "middle_up"),
        (RI_MOUSE_BUTTON_4_DOWN, "button4_down"),
        (RI_MOUSE_BUTTON_4_UP, "button4_up"),
        (RI_MOUSE_BUTTON_5_DOWN, "button5_down"),
        (RI_MOUSE_BUTTON_5_UP, "button5_up"),
        (RI_MOUSE_WHEEL, "wheel"),
        (RI_MOUSE_HWHEEL, "hwheel"),
    )
    for bit, name in mapping:
        if flags & bit:
            names.append(name)
    return names


def format_handle(handle: Any) -> str:
    value = ctypes.cast(handle, ctypes.c_void_p).value
    return "0x0" if value is None else f"0x{value:x}"


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def write_dataclass_csv(path: Path, rows: Iterable[Any], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            if "button_flags" in values:
                values["button_flags"] = f"0x{values['button_flags']:04x}"
            writer.writerow(values)


def aggregate_mouse_by_frame(
    events: list[MouseEvent],
    frame_count: int,
    fps: int,
) -> list[dict[str, Any]]:
    """Bin raw packets onto the exact CFR MP4 timeline."""
    rows: list[dict[str, Any]] = []
    bins: list[list[MouseEvent]] = [[] for _ in range(frame_count)]
    for event in events:
        index = (event.session_time_ns * fps) // NANOSECONDS_PER_SECOND
        if 0 <= index < frame_count:
            bins[int(index)].append(event)

    cumulative_x = 0
    cumulative_y = 0
    left_down = False
    for index, frame_events in enumerate(bins):
        dx = sum(event.dx_counts for event in frame_events)
        dy = sum(event.dy_counts for event in frame_events)
        cumulative_x += dx
        cumulative_y += dy
        for event in frame_events:
            left_down = event.left_button_down
        rows.append(
            {
                "frame_index": index,
                "video_time_ns": round(index * NANOSECONDS_PER_SECOND / fps),
                "video_time_s": index / fps,
                "raw_event_count": len(frame_events),
                "mouse_dx_counts": dx,
                "mouse_dy_counts": dy,
                "mouse_vx_counts_s": dx * fps,
                "mouse_vy_counts_s": dy * fps,
                "mouse_speed_counts_s": math.hypot(dx, dy) * fps,
                "cumulative_dx_counts": cumulative_x,
                "cumulative_dy_counts": cumulative_y,
                "left_button_down": left_down,
                "button_events": "|".join(
                    event.button_names for event in frame_events if event.button_names
                ),
            }
        )
    return rows


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ffmpeg_encoder_args(encoder: str, quality: int) -> list[str]:
    if encoder == "h264_nvenc":
        return [
            "-c:v",
            encoder,
            "-preset",
            "p3",
            "-tune",
            "ll",
            "-rc",
            "constqp",
            "-qp",
            str(quality),
            "-bf",
            "0",
            "-rc-lookahead",
            "0",
            "-delay",
            "0",
            "-zerolatency",
            "1",
        ]
    if encoder == "h264_amf":
        return [
            "-c:v",
            encoder,
            "-usage",
            "lowlatency",
            "-quality",
            "speed",
            "-rc",
            "cqp",
            "-qp_i",
            str(quality),
            "-qp_p",
            str(quality),
        ]
    if encoder == "h264_qsv":
        return [
            "-c:v",
            encoder,
            "-preset",
            "veryfast",
            "-global_quality",
            str(quality),
        ]
    if encoder == "libx264":
        return [
            "-c:v",
            encoder,
            "-preset",
            "ultrafast",
            "-crf",
            str(quality),
        ]
    raise ValueError(f"unsupported encoder: {encoder}")


def encoder_works(ffmpeg: str, encoder: str, quality: int) -> bool:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        # Current NVENC generations reject extremely small frame dimensions.
        # 256x256 is still a cheap probe and is accepted by the supported
        # hardware encoders.
        "color=size=256x256:rate=1:color=black",
        "-frames:v",
        "1",
        *ffmpeg_encoder_args(encoder, quality),
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
        timeout=15,
        check=False,
    )
    return result.returncode == 0


def select_encoder(ffmpeg: str, requested: str, quality: int) -> str:
    candidates = (
        [requested]
        if requested != "auto"
        else ["h264_nvenc", "h264_amf", "h264_qsv", "libx264"]
    )
    for encoder in candidates:
        if encoder_works(ffmpeg, encoder, quality):
            return encoder
    raise RuntimeError(
        f"FFmpeg could not initialize any requested H.264 encoder: {candidates}"
    )


class FrameEncoder:
    def __init__(
        self,
        *,
        camera: Any,
        ffmpeg: str,
        encoder: str,
        quality: int,
        output_dir: Path,
        fps: int,
        width: int,
        height: int,
        session_start_ns: int,
    ) -> None:
        self.camera = camera
        self.ffmpeg = ffmpeg
        self.encoder = encoder
        self.quality = quality
        self.output_dir = output_dir
        self.fps = fps
        self.width = width
        self.height = height
        self.session_start_ns = session_start_ns
        self.stop_requested = threading.Event()
        self.thread: threading.Thread | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any = None
        self.rows: list[VideoFrameRow] = []
        self.error: BaseException | None = None
        self.capture_packets = 0
        self.skipped_same_slot = 0
        self.initial_fill_frames = 0
        self.gap_fill_frames = 0
        self.tail_fill_frames = 0
        self.stop_qpc_ns: int | None = None
        self.last_frame: Any = None
        self.last_source_qpc_ns = 0
        self.command: list[str] = []

    @property
    def period_ns(self) -> float:
        return NANOSECONDS_PER_SECOND / self.fps

    def prepare(self) -> None:
        video_path = self.output_dir / "video.mp4"
        self.log_handle = (self.output_dir / "ffmpeg.log").open(
            "wb", buffering=0
        )
        self.command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            *ffmpeg_encoder_args(self.encoder, self.quality),
            "-g",
            str(self.fps),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.log_handle,
            creationflags=CREATE_NO_WINDOW,
        )

    def start(self) -> None:
        if self.process is None:
            raise RuntimeError("encoder was not prepared")
        self.thread = threading.Thread(
            target=self._run, name="SyncedVideoEncoder", daemon=True
        )
        self.thread.start()

    def request_stop(self, stop_qpc_ns: int) -> None:
        self.stop_qpc_ns = stop_qpc_ns
        self.stop_requested.set()

    def join(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError("video encoder did not stop within 30 seconds")
        if self.error is not None:
            raise RuntimeError("video encoder failed") from self.error

    def abort_before_start(self) -> None:
        if self.process is not None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()

    def _write_frame(
        self,
        frame: Any,
        *,
        source_qpc_ns: int,
        effective_qpc_ns: int,
        retrieved_qpc_ns: int,
        frame_kind: str,
        timestamp_source: str,
    ) -> None:
        assert self.process is not None and self.process.stdin is not None
        index = len(self.rows)
        expected_qpc_ns = round(
            self.session_start_ns + index * NANOSECONDS_PER_SECOND / self.fps
        )
        try:
            self.process.stdin.write(memoryview(frame).cast("B"))
        except BrokenPipeError as exc:
            raise RuntimeError(
                "FFmpeg stopped while encoding; inspect ffmpeg.log"
            ) from exc
        self.rows.append(
            VideoFrameRow(
                frame_index=index,
                video_time_ns=round(index * NANOSECONDS_PER_SECOND / self.fps),
                expected_qpc_ns=expected_qpc_ns,
                source_qpc_ns=source_qpc_ns,
                effective_qpc_ns=effective_qpc_ns,
                retrieved_qpc_ns=retrieved_qpc_ns,
                frame_kind=frame_kind,
                timestamp_source=timestamp_source,
                source_age_ms=(retrieved_qpc_ns - source_qpc_ns) / 1_000_000,
                timeline_error_ms=(effective_qpc_ns - expected_qpc_ns) / 1_000_000,
            )
        )

    def _consume_scheduled(self, frame: Any, source_time_s: float | None) -> None:
        retrieved_qpc_ns = time.perf_counter_ns()
        source_qpc_ns = (
            round(source_time_s * NANOSECONDS_PER_SECOND)
            if source_time_s is not None
            else 0
        )
        self.capture_packets += 1
        source_is_new = source_qpc_ns > self.last_source_qpc_ns
        source_is_plausible = (
            source_is_new
            and source_qpc_ns >= self.session_start_ns
            and source_qpc_ns <= retrieved_qpc_ns + round(self.period_ns * 2)
            and retrieved_qpc_ns - source_qpc_ns <= 250_000_000
        )
        self.last_source_qpc_ns = max(self.last_source_qpc_ns, source_qpc_ns)
        target_index = max(
            0,
            math.floor(
                (retrieved_qpc_ns - self.session_start_ns)
                * self.fps
                / NANOSECONDS_PER_SECOND
            ),
        )

        if self.last_frame is None:
            while len(self.rows) < target_index:
                self._write_frame(
                    frame,
                    source_qpc_ns=source_qpc_ns,
                    effective_qpc_ns=round(
                        self.session_start_ns
                        + len(self.rows) * NANOSECONDS_PER_SECOND / self.fps
                    ),
                    retrieved_qpc_ns=retrieved_qpc_ns,
                    frame_kind="initial_fill",
                    timestamp_source="cfr_schedule",
                )
                self.initial_fill_frames += 1
        elif target_index > len(self.rows):
            while len(self.rows) < target_index:
                self._write_frame(
                    self.last_frame,
                    source_qpc_ns=self.rows[-1].source_qpc_ns,
                    effective_qpc_ns=self.rows[-1].effective_qpc_ns,
                    retrieved_qpc_ns=retrieved_qpc_ns,
                    frame_kind="gap_fill",
                    timestamp_source="previous_frame",
                )
                self.gap_fill_frames += 1

        if target_index < len(self.rows):
            # A high-resolution wait should not wake before its CFR deadline.
            # If it does because of timer quantization, defer this sample rather
            # than creating a video frame ahead of the shared clock.
            self.skipped_same_slot += 1
            self.last_frame = frame
            return

        if source_is_plausible:
            effective_qpc_ns = source_qpc_ns
            timestamp_source = "dxgi_present"
            frame_kind = "captured"
        elif source_is_new:
            effective_qpc_ns = retrieved_qpc_ns
            timestamp_source = "retrieval_qpc"
            frame_kind = "captured"
        else:
            effective_qpc_ns = round(
                self.session_start_ns
                + target_index * NANOSECONDS_PER_SECOND / self.fps
            )
            timestamp_source = "cfr_schedule"
            frame_kind = "source_repeat"

        self._write_frame(
            frame,
            source_qpc_ns=source_qpc_ns,
            effective_qpc_ns=effective_qpc_ns,
            retrieved_qpc_ns=retrieved_qpc_ns,
            frame_kind=frame_kind,
            timestamp_source=timestamp_source,
        )
        self.last_frame = frame

    def _finish_tail(self) -> None:
        if self.last_frame is None or self.stop_qpc_ns is None:
            return
        target_count = max(
            1,
            round(
                (self.stop_qpc_ns - self.session_start_ns)
                * self.fps
                / NANOSECONDS_PER_SECOND
            ),
        )
        while len(self.rows) < target_count:
            self._write_frame(
                self.last_frame,
                source_qpc_ns=self.rows[-1].source_qpc_ns,
                effective_qpc_ns=self.rows[-1].effective_qpc_ns,
                retrieved_qpc_ns=self.stop_qpc_ns,
                frame_kind="tail_fill",
                timestamp_source="previous_frame",
            )
            self.tail_fill_frames += 1

    def _run(self) -> None:
        try:
            while not self.stop_requested.is_set():
                next_deadline_ns = round(
                    self.session_start_ns
                    + len(self.rows) * NANOSECONDS_PER_SECOND / self.fps
                )
                remaining_s = (next_deadline_ns - time.perf_counter_ns()) / (
                    NANOSECONDS_PER_SECOND
                )
                if remaining_s > 0:
                    # Python's Windows sleep uses a high-resolution waitable
                    # timer. threading.Event.wait can fall back to the coarser
                    # scheduler tick and collapse a 120 Hz loop toward 60 Hz.
                    time.sleep(remaining_s)
                    if self.stop_requested.is_set():
                        break
                frame = self.camera.grab(copy=True)
                if frame is None:
                    time.sleep(0.0005)
                    continue
                self._consume_scheduled(frame, self.camera.latest_frame_time)
            self._finish_tail()
        except BaseException as exc:
            self.error = exc
        finally:
            if self.process is not None and self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
            if self.process is not None:
                return_code = self.process.wait(timeout=30)
                if return_code != 0 and self.error is None:
                    self.error = RuntimeError(
                        f"FFmpeg exited with status {return_code}; inspect ffmpeg.log"
                    )
            if self.log_handle is not None:
                self.log_handle.close()


class RecordingSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = self._make_output_dir(args.output_root, args.label)
        self.camera: Any = None
        self.encoder_name = ""
        self.encoder: FrameEncoder | None = None
        self.session_start_ns: int | None = None
        self.session_stop_ns: int | None = None
        self.started_wallclock: str | None = None
        self.stopped_wallclock: str | None = None
        self.measurement_ready_ns: int | None = None
        self.mouse_events: list[MouseEvent] = []
        self.left_button_down = False
        self.last_mouse_qpc_ns: int | None = None
        self.active = False
        self.completed = False
        self.tick_anchor_qpc_ns = time.perf_counter_ns()
        self.tick_anchor_ms = int(kernel32.GetTickCount64())
        self.duration_thread: threading.Thread | None = None
        self.window_handle: wintypes.HWND | None = None
        self.ffmpeg = ""
        self.region: tuple[int, int, int, int] | None = args.region

    @staticmethod
    def _make_output_dir(root: Path, label: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_label = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in label.strip()
        ).strip("_") or "capture"
        path = root.resolve() / f"{timestamp}_{safe_label}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def prepare(self) -> None:
        ffmpeg_candidate = self.args.ffmpeg or shutil.which("ffmpeg")
        if not ffmpeg_candidate:
            raise RuntimeError("FFmpeg was not found on PATH; pass --ffmpeg PATH")
        self.ffmpeg = str(Path(ffmpeg_candidate).resolve())
        self.encoder_name = select_encoder(
            self.ffmpeg, self.args.encoder, self.args.quality
        )
        self.camera = dxcam.create(
            device_idx=self.args.device_index,
            output_idx=self.args.output_index,
            region=self.region,
            output_color="BGR",
            max_buffer_len=16,
            backend="dxgi",
        )
        if self.region is None:
            self.region = tuple(self.camera.region)
        left, top, right, bottom = self.region
        width = right - left
        height = bottom - top
        if width % 2 or height % 2:
            raise RuntimeError("H.264 capture width and height must both be even")
        print(f"Output directory: {self.output_dir}")
        print(
            f"Capture: {width}x{height} at {self.args.fps} FPS, "
            f"DXGI + {self.encoder_name}"
        )

    def attach_window(self, hwnd: wintypes.HWND) -> None:
        self.window_handle = hwnd

    def start(self) -> None:
        if self.active or self.completed:
            return
        if self.camera is None:
            raise RuntimeError("session was not prepared")
        self.session_start_ns = time.perf_counter_ns()
        self.started_wallclock = datetime.now().astimezone().isoformat()
        left, top, right, bottom = self.region or (0, 0, 0, 0)
        self.encoder = FrameEncoder(
            camera=self.camera,
            ffmpeg=self.ffmpeg,
            encoder=self.encoder_name,
            quality=self.args.quality,
            output_dir=self.output_dir,
            fps=self.args.fps,
            width=right - left,
            height=bottom - top,
            session_start_ns=self.session_start_ns,
        )
        self.encoder.prepare()
        self.active = True
        if self.args.settle_seconds == 0:
            self.measurement_ready_ns = self.session_start_ns
        self.camera.start(
            region=self.region,
            target_fps=self.args.fps,
            video_mode=True,
        )
        self.encoder.start()
        print(
            f"RECORDING '{self.args.label}' | "
            f"press {hotkey_name(self.args.hotkey)} to stop"
        )
        if self.args.settle_seconds > 0:
            print(
                f"Wait {self.args.settle_seconds:g} s for the second beep, "
                "then begin the measured movement/firing."
            )
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_OK)
        except Exception:
            pass
        if self.args.settle_seconds > 0:
            threading.Thread(
                target=self._settle_beep,
                name="RecordingSettleBeep",
                daemon=True,
            ).start()
        if self.args.duration is not None:
            self.duration_thread = threading.Thread(
                target=self._duration_wait,
                name="RecordingDuration",
                daemon=True,
            )
            self.duration_thread.start()

    def _settle_beep(self) -> None:
        time.sleep(self.args.settle_seconds)
        if not self.active:
            return
        self.measurement_ready_ns = time.perf_counter_ns()
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass

    def _duration_wait(self) -> None:
        time.sleep(self.args.duration)
        if self.active and self.window_handle:
            user32.PostMessageW(self.window_handle, WM_APP_STOP, 0, 0)

    def stop(self) -> None:
        if not self.active:
            return
        self.session_stop_ns = time.perf_counter_ns()
        self.stopped_wallclock = datetime.now().astimezone().isoformat()
        self.active = False
        assert self.encoder is not None
        self.encoder.request_stop(self.session_stop_ns)
        self.camera.stop()
        self.encoder.join()
        self._write_outputs()
        self.completed = True
        duration = (self.session_stop_ns - self.session_start_ns) / NANOSECONDS_PER_SECOND
        print(
            f"Saved {len(self.encoder.rows)} frames and {len(self.mouse_events)} "
            f"raw mouse packets ({duration:.3f} s)."
        )
        print(f"Session: {self.output_dir}")
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

    def close(self) -> None:
        if self.active:
            self.stop()
        elif not self.completed and self.encoder is not None:
            self.encoder.abort_before_start()
        if self.camera is not None:
            self.camera.release()

    def estimate_message_qpc_ns(self, message_tick32: int) -> int:
        now_tick64 = int(kernel32.GetTickCount64())
        base = now_tick64 & ~0xFFFFFFFF
        candidate = base | (message_tick32 & 0xFFFFFFFF)
        if candidate - now_tick64 > 0x80000000:
            candidate -= 0x100000000
        elif now_tick64 - candidate > 0x80000000:
            candidate += 0x100000000
        return self.tick_anchor_qpc_ns + (
            candidate - self.tick_anchor_ms
        ) * 1_000_000

    def add_mouse_packet(
        self,
        *,
        raw: RAWINPUT,
        processing_qpc_ns: int,
        message_tick32: int,
    ) -> None:
        if not self.active or self.session_start_ns is None:
            return
        mouse = raw.data.mouse
        flags = int(mouse.usButtonFlags)
        if flags & RI_MOUSE_LEFT_BUTTON_DOWN:
            self.left_button_down = True
        if flags & RI_MOUSE_LEFT_BUTTON_UP:
            self.left_button_down = False
        message_qpc_ns = self.estimate_message_qpc_ns(message_tick32)
        previous = self.last_mouse_qpc_ns
        dt_ns = processing_qpc_ns - previous if previous is not None else 0
        dt_s = dt_ns / NANOSECONDS_PER_SECOND
        dx = int(mouse.lLastX)
        dy = int(mouse.lLastY)
        vx = dx / dt_s if dt_s > 0 else 0.0
        vy = dy / dt_s if dt_s > 0 else 0.0
        wheel_delta = (
            signed_short(int(mouse.usButtonData))
            if flags & (RI_MOUSE_WHEEL | RI_MOUSE_HWHEEL)
            else 0
        )
        self.mouse_events.append(
            MouseEvent(
                event_index=len(self.mouse_events),
                qpc_ns=processing_qpc_ns,
                session_time_ns=processing_qpc_ns - self.session_start_ns,
                message_qpc_estimate_ns=message_qpc_ns,
                # GetMessageTime is expressed in milliseconds and may be
                # quantized to the Windows scheduler tick.  Clamp the small
                # negative mapping error instead of reporting impossible
                # negative queue latency.
                queue_latency_us=max(
                    0.0, (processing_qpc_ns - message_qpc_ns) / 1_000
                ),
                dx_counts=dx,
                dy_counts=dy,
                dt_us=dt_ns / 1_000,
                vx_counts_s=vx,
                vy_counts_s=vy,
                speed_counts_s=math.hypot(vx, vy),
                movement_mode=(
                    "absolute" if mouse.usFlags & MOUSE_MOVE_ABSOLUTE else "relative"
                ),
                button_flags=flags,
                button_names="|".join(decode_buttons(flags)),
                wheel_delta=wheel_delta,
                left_button_down=self.left_button_down,
                device_handle=format_handle(raw.header.hDevice),
            )
        )
        self.last_mouse_qpc_ns = processing_qpc_ns

    def _write_outputs(self) -> None:
        assert self.encoder is not None
        assert self.session_start_ns is not None
        assert self.session_stop_ns is not None
        mouse_fields = list(MouseEvent.__dataclass_fields__)
        video_fields = list(VideoFrameRow.__dataclass_fields__)
        write_dataclass_csv(
            self.output_dir / "mouse_events.csv", self.mouse_events, mouse_fields
        )
        write_dataclass_csv(
            self.output_dir / "video_frames.csv", self.encoder.rows, video_fields
        )
        per_frame = aggregate_mouse_by_frame(
            self.mouse_events, len(self.encoder.rows), self.args.fps
        )
        write_dict_csv(self.output_dir / "mouse_by_video_frame.csv", per_frame)

        duration_s = (
            self.session_stop_ns - self.session_start_ns
        ) / NANOSECONDS_PER_SECOND
        dt_values = [
            event.dt_us for event in self.mouse_events if event.dt_us > 0
        ]
        speed_values = [
            event.speed_counts_s
            for event in self.mouse_events
            if event.dx_counts or event.dy_counts
        ]
        latency_values = [
            max(0.0, event.queue_latency_us) for event in self.mouse_events
        ]
        captured_rows = [
            row for row in self.encoder.rows if row.frame_kind == "captured"
        ]
        source_repeat_rows = [
            row for row in self.encoder.rows if row.frame_kind == "source_repeat"
        ]
        source_age = [
            max(0.0, row.source_age_ms)
            for row in captured_rows
            if row.timestamp_source == "dxgi_present"
        ]
        warnings: list[str] = []
        fill_frames = self.encoder.gap_fill_frames + self.encoder.tail_fill_frames
        if self.encoder.rows and fill_frames / len(self.encoder.rows) > 0.05:
            warnings.append(
                "More than 5% of MP4 frames were timing fills; reduce capture "
                "resolution or encoder quality before measuring recoil."
            )
        if source_age and (percentile(source_age, 0.95) or 0) > 20:
            warnings.append(
                "DXGI presentation-to-consumer latency exceeded 20 ms at p95."
            )
        if not self.mouse_events:
            warnings.append("No Raw Input mouse packets were recorded.")
        if any(event.movement_mode == "absolute" for event in self.mouse_events):
            warnings.append(
                "Absolute mouse packets were observed; relative gaming-mouse "
                "counts are required for recoil calibration."
            )

        manifest = {
            "schema_version": 1,
            "label": self.args.label,
            "recording_complete": True,
            "started_wallclock": self.started_wallclock,
            "stopped_wallclock": self.stopped_wallclock,
            "session_start_qpc_ns": self.session_start_ns,
            "session_stop_qpc_ns": self.session_stop_ns,
            "duration_s": duration_s,
            "measurement_settle_seconds": self.args.settle_seconds,
            "measurement_ready_qpc_ns": self.measurement_ready_ns,
            "measurement_ready_video_time_s": (
                (self.measurement_ready_ns - self.session_start_ns)
                / NANOSECONDS_PER_SECOND
                if self.measurement_ready_ns is not None
                else None
            ),
            "measurement_ready_video_frame": (
                round(
                    (self.measurement_ready_ns - self.session_start_ns)
                    * self.args.fps
                    / NANOSECONDS_PER_SECOND
                )
                if self.measurement_ready_ns is not None
                else None
            ),
            "clock": {
                "name": "Windows QueryPerformanceCounter via Python perf_counter",
                "video_dxgi_timestamp_same_clock_domain": True,
                "mouse_primary_timestamp": "WM_INPUT processing QPC",
                "mouse_message_timestamp_note": (
                    "GetMessageTime is also mapped to QPC at 1 ms resolution; "
                    "queue_latency_us is diagnostic, not the primary timeline."
                ),
            },
            "capture": {
                "backend": "DXGI Desktop Duplication (dxcam)",
                "device_index": self.args.device_index,
                "output_index": self.args.output_index,
                "region_left_top_right_bottom": list(self.region or ()),
                "width": (self.region or (0, 0, 0, 0))[2]
                - (self.region or (0, 0, 0, 0))[0],
                "height": (self.region or (0, 0, 0, 0))[3]
                - (self.region or (0, 0, 0, 0))[1],
                "target_fps": self.args.fps,
                "video_timeline": (
                    "CFR frame N = session_start_qpc + N/target_fps"
                ),
                "encoded_frame_count": len(self.encoder.rows),
                "capture_packets_consumed": self.encoder.capture_packets,
                "captured_video_frames": len(captured_rows),
                "source_repeat_frames": len(source_repeat_rows),
                "initial_fill_frames": self.encoder.initial_fill_frames,
                "gap_fill_frames": self.encoder.gap_fill_frames,
                "tail_fill_frames": self.encoder.tail_fill_frames,
                "capture_packets_skipped_same_slot": self.encoder.skipped_same_slot,
                "effective_video_fps": (
                    len(self.encoder.rows) / duration_s if duration_s > 0 else 0
                ),
                "dxgi_source_age_ms": {
                    "median": statistics.median(source_age) if source_age else None,
                    "p95": percentile(source_age, 0.95),
                    "max": max(source_age) if source_age else None,
                },
            },
            "encoder": {
                "ffmpeg": self.ffmpeg,
                "codec": self.encoder_name,
                "quality": self.args.quality,
                "command": self.encoder.command,
            },
            "mouse": {
                "raw_packet_count": len(self.mouse_events),
                "relative_packet_count": sum(
                    event.movement_mode == "relative" for event in self.mouse_events
                ),
                "device_handles": sorted(
                    {event.device_handle for event in self.mouse_events}
                ),
                "total_dx_counts": sum(event.dx_counts for event in self.mouse_events),
                "total_dy_counts": sum(event.dy_counts for event in self.mouse_events),
                "packet_interval_us": {
                    "median": statistics.median(dt_values) if dt_values else None,
                    "p95": percentile(dt_values, 0.95),
                },
                "movement_speed_counts_s": {
                    "median": statistics.median(speed_values) if speed_values else None,
                    "p95": percentile(speed_values, 0.95),
                    "max": max(speed_values) if speed_values else None,
                },
                "message_queue_latency_us": {
                    "median": statistics.median(latency_values)
                    if latency_values
                    else None,
                    "p95": percentile(latency_values, 0.95),
                },
            },
            "files": {
                "video": "video.mp4",
                "video_frames": "video_frames.csv",
                "mouse_events": "mouse_events.csv",
                "mouse_by_video_frame": "mouse_by_video_frame.csv",
                "ffmpeg_log": "ffmpeg.log",
            },
            "warnings": warnings,
        }
        atomic_json(self.output_dir / "session.json", manifest)


class RawInputWindow:
    def __init__(self, session: RecordingSession, args: argparse.Namespace) -> None:
        self.session = session
        self.args = args
        self.hwnd: wintypes.HWND | None = None
        self.error: BaseException | None = None
        self.class_name = f"RecoilRawInputRecorder_{id(self):x}"
        self.hinstance = kernel32.GetModuleHandleW(None)
        self._wndproc_callback = WNDPROC(self._wndproc)

    def create(self) -> None:
        window_class = WNDCLASSEXW()
        window_class.cbSize = ctypes.sizeof(WNDCLASSEXW)
        window_class.lpfnWndProc = self._wndproc_callback
        window_class.hInstance = self.hinstance
        window_class.lpszClassName = self.class_name
        atom = user32.RegisterClassExW(ctypes.byref(window_class))
        if not atom:
            raise ctypes.WinError(ctypes.get_last_error())
        self.hwnd = user32.CreateWindowExW(
            0,
            self.class_name,
            "Recoil synchronized recorder",
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            self.hinstance,
            None,
        )
        if not self.hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        device = RAWINPUTDEVICE(
            usUsagePage=0x01,
            usUsage=0x02,
            dwFlags=RIDEV_INPUTSINK,
            hwndTarget=self.hwnd,
        )
        if not user32.RegisterRawInputDevices(
            ctypes.byref(device), 1, ctypes.sizeof(RAWINPUTDEVICE)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not user32.RegisterHotKey(
            self.hwnd, HOTKEY_RECORD_ID, 0, self.args.hotkey
        ):
            raise RuntimeError(
                f"Could not register {hotkey_name(self.args.hotkey)}; another "
                "application is already using it. Pass --hotkey F10 (or another key)."
            )
        if not user32.RegisterHotKey(
            self.hwnd, HOTKEY_QUIT_ID, 0, self.args.quit_hotkey
        ):
            raise RuntimeError(
                f"Could not register {hotkey_name(self.args.quit_hotkey)}; pass "
                "--quit-hotkey with another key."
            )
        self.session.attach_window(self.hwnd)

    def _read_raw_input(self, lparam: wintypes.LPARAM) -> RAWINPUT | None:
        size = wintypes.UINT(0)
        result = user32.GetRawInputData(
            wintypes.HANDLE(lparam),
            RID_INPUT,
            None,
            ctypes.byref(size),
            ctypes.sizeof(RAWINPUTHEADER),
        )
        if result == 0xFFFFFFFF or size.value < ctypes.sizeof(RAWINPUTHEADER):
            return None
        buffer = ctypes.create_string_buffer(size.value)
        result = user32.GetRawInputData(
            wintypes.HANDLE(lparam),
            RID_INPUT,
            buffer,
            ctypes.byref(size),
            ctypes.sizeof(RAWINPUTHEADER),
        )
        if result == 0xFFFFFFFF or result != size.value:
            return None
        # Own the bytes after this function returns; a ctypes pointer view would
        # otherwise depend on the lifetime of the local buffer.
        return RAWINPUT.from_buffer_copy(buffer)

    def _wndproc(
        self,
        hwnd: wintypes.HWND,
        message: int,
        wparam: wintypes.WPARAM,
        lparam: wintypes.LPARAM,
    ) -> int:
        try:
            if message == WM_INPUT:
                processing_qpc_ns = time.perf_counter_ns()
                message_tick32 = int(user32.GetMessageTime()) & 0xFFFFFFFF
                raw = self._read_raw_input(lparam)
                if raw is not None and raw.header.dwType == RIM_TYPEMOUSE:
                    self.session.add_mouse_packet(
                        raw=raw,
                        processing_qpc_ns=processing_qpc_ns,
                        message_tick32=message_tick32,
                    )
            elif message == WM_HOTKEY:
                if int(wparam) == HOTKEY_RECORD_ID:
                    if self.session.active:
                        self.session.stop()
                    elif not self.session.completed:
                        self.session.start()
                elif int(wparam) == HOTKEY_QUIT_ID:
                    if self.session.active:
                        self.session.stop()
                    user32.DestroyWindow(hwnd)
                return 0
            elif message == WM_APP_START:
                self.session.start()
                return 0
            elif message == WM_APP_STOP:
                self.session.stop()
                if self.args.auto_start:
                    user32.DestroyWindow(hwnd)
                return 0
            elif message == WM_APP_QUIT:
                user32.DestroyWindow(hwnd)
                return 0
            elif message == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
        except BaseException as exc:
            self.error = exc
            user32.DestroyWindow(hwnd)
            return 0
        return int(user32.DefWindowProcW(hwnd, message, wparam, lparam))

    def run(self) -> None:
        self.create()
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 1)
        assert self.hwnd is not None
        print(
            f"Ready. Press {hotkey_name(self.args.hotkey)} in the game to "
            f"start/stop; {hotkey_name(self.args.quit_hotkey)} exits."
        )
        if self.args.auto_start:
            user32.PostMessageW(self.hwnd, WM_APP_START, 0, 0)
        message = wintypes.MSG()
        while True:
            result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
            if result == 0:
                break
            if result == -1:
                raise ctypes.WinError(ctypes.get_last_error())
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
        user32.UnregisterHotKey(self.hwnd, HOTKEY_RECORD_ID)
        user32.UnregisterHotKey(self.hwnd, HOTKEY_QUIT_ID)
        if self.error is not None:
            raise self.error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record a 120 FPS CFR desktop video and Windows Raw Input mouse "
            "packets on the same QPC timeline."
        )
    )
    parser.add_argument(
        "--label",
        required=True,
        help="session label, for example no_fire or recoil",
    )
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument(
        "--region",
        type=parse_region,
        help="DXGI region as left,top,right,bottom; default is the full output",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=Path("synced_captures")
    )
    parser.add_argument(
        "--encoder",
        choices=("auto", "h264_nvenc", "h264_amf", "h264_qsv", "libx264"),
        default="auto",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=16,
        help="H.264 QP/quality value; lower is higher quality (default: 16)",
    )
    parser.add_argument("--ffmpeg", help="path to ffmpeg.exe")
    parser.add_argument("--hotkey", type=parse_hotkey, default=parse_hotkey("F8"))
    parser.add_argument(
        "--quit-hotkey", type=parse_hotkey, default=parse_hotkey("F9")
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="automatically stop this many seconds after recording starts",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="delay before the second beep that signals measurement start",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="start immediately; intended for automated smoke tests",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.fps < 1 or args.fps > 240:
        raise SystemExit("--fps must be between 1 and 240")
    if not 0 <= args.quality <= 51:
        raise SystemExit("--quality must be between 0 and 51")
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.settle_seconds < 0:
        raise SystemExit("--settle-seconds cannot be negative")
    if args.hotkey == args.quit_hotkey:
        raise SystemExit("--hotkey and --quit-hotkey must be different")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    session = RecordingSession(args)
    listener: RawInputWindow | None = None
    try:
        session.prepare()
        listener = RawInputWindow(session, args)
        listener.run()
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except BaseException as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            session.close()
        except BaseException as exc:
            print(f"ERROR while closing recorder: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
