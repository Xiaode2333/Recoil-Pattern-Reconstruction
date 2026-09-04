# Synchronized capture

Two capture paths are included.

## Standalone Windows recorder

`record_mouse_video.py` uses DXGI Desktop Duplication for video and Windows Raw
Input for mouse packets. Both streams share a QueryPerformanceCounter clock.
The encoded MP4 is a constant-rate timeline; missing source frames are filled
and disclosed rather than silently removing time.

```powershell
python .\record_mouse_video.py --label reference --fps 120
python .\record_mouse_video.py --label recoil --fps 120
```

Each session contains video, raw input events, per-frame timing, input aggregated
onto video frames, and a diagnostic JSON summary. Keep resolution, refresh rate,
input settings, field of view, and magnification unchanged between reference and
recoil captures.

## OBS sidecar plugin

`obs_mouse_timeline/` is a native Windows OBS plugin that records Raw Input and
frame-clock sidecars while OBS handles capture and encoding. See its README for
build instructions. `analyze_synced_recoil.py` consumes a reference/recoil pair:

```powershell
python .\analyze_synced_recoil.py `
  .\captures\reference.mp4 .\captures\recoil.mp4 `
  --reference-mode ads --shot-count 30 `
  --output-dir .\synced_result
```

Use a cropped region if capture diagnostics report excessive timing-fill frames.
Recordings with dropped-output evidence should not be used for formal results.
