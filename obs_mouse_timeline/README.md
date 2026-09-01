# OBS Mouse Timeline

Native Windows OBS plugin for synchronized recoil capture. OBS performs game
capture and NVENC recording; the plugin passively receives Windows Raw Input and
records a frame-clock sidecar without copying video pixels.

For every OBS recording the plugin writes three files beside the video:

- `<recording>.mouse.csv`: raw mouse counts, buttons, speed, the most recent OBS
  tick index, and both libobs clocks.
- `<recording>.frames.csv`: every OBS video tick with the recording output frame
  counter.
- `<recording>.mouse-session.json`: OBS/FPS/resolution metadata, start/stop
  markers, frame totals, dropped frames, and alignment semantics.

The Raw Input listener uses `RIDEV_INPUTSINK` only. It does not inject input,
install a keyboard/mouse hook, or access the game process.

## Build

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64
```

The project follows the official OBS plugin template and builds against the OBS
31.1 plugin SDK, whose C ABI is compatible with the locally installed OBS 32.2.2.

## Recording setup

1. Set OBS video FPS to 120 and output resolution to 2560x1440.
2. Add Rainbow Six Siege using Game Capture.
3. Select NVENC and record to MKV.
4. Assign an OBS **Start/Stop Recording** global hotkey; no CMD focus is needed.
5. Keep recording unpaused. Start, wait one second, perform the take, and stop.

The sidecars appear in the same directory as the MKV recording.
