# Recoil Pattern Reconstruction (Feature Matching + RANSAC)

## Synchronized 120 FPS Video + Raw Mouse Recording

`record_mouse_video.py` passively records the selected Windows display through
DXGI Desktop Duplication while also logging Windows Raw Input mouse packets. Both
streams use the same QueryPerformanceCounter clock. The MP4 is a constant 120
FPS timeline; timing gaps are filled with the previous image and disclosed in
`session.json` and `video_frames.csv` instead of silently shortening the video.

Install the dependencies once, then run the baseline recording:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\record_mouse_video.py --label no_fire
```

The recorder waits in the background. In Rainbow Six Siege, press `F8` to start.
The first beep confirms recording; wait for the second beep one second later so
the encoder is fully warm, then move the mouse through a useful mix of slow and
fast horizontal/vertical motions without firing. Press `F8` again to stop and
`F9` to exit the recorder. Run the firing take separately:

```powershell
.\.venv\Scripts\python.exe .\record_mouse_video.py --label recoil
```

For the firing take, start recording, begin firing, and manually pull the view
back into the usable pitch range. The left-button state is included in the Raw
Input log. Do not change DPI, Windows pointer settings, game sensitivity, FOV,
optic, resolution, or display mode between the two takes.

Each session is stored below `synced_captures/<timestamp>_<label>/`:

- `video.mp4`: hardware-encoded constant-120-FPS capture.
- `mouse_events.csv`: every Raw Input packet, QPC timestamp, raw X/Y counts,
  instantaneous speed, button transitions, and input device handle.
- `video_frames.csv`: the actual DXGI source timestamp for every encoded frame,
  including explicit `initial_fill`, `gap_fill`, and `tail_fill` rows.
- `mouse_by_video_frame.csv`: raw counts, velocity, cumulative movement, and left
  button state already aggregated onto each MP4 frame.
- `session.json`: common clock origin, capture/encoder diagnostics, achieved FPS,
  latency percentiles, and warnings.

The default capture is the entire primary display at 120 FPS. A cropped region
reduces bandwidth if diagnostics report more than 5% timing-fill frames:

```powershell
.\.venv\Scripts\python.exe .\record_mouse_video.py --label recoil `
  --region 0,0,1920,1080 --fps 120 --duration 15
```

Set the monitor to at least 120 Hz and keep the game's rendered frame rate at or
above 120 if you want 120 distinct source frames rather than correctly timed
duplicates.

Use borderless/windowed-fullscreen mode if exclusive fullscreen prevents Desktop
Duplication capture. This recorder only observes desktop frames and Raw Input; it
does not inject mouse input, hook the game, or access game memory.

## OBS Raw-Input Pair Analysis

`analyze_synced_recoil.py` combines the sidecars from the native
`obs_mouse_timeline` plugin with two OBS recordings. The no-fire reference maps
Raw Input counts to measured camera motion; the firing take subtracts that mouse
component from the observed view movement and emits a per-frame and per-shot
recoil trajectory.

For a formal result, record the reference while holding the same ADS optic used
by the firing take. Keep resolution, 120 FPS, FOV, sensitivity, DPI, aspect ratio,
operator stance, and optic unchanged:

```powershell
.\.venv\Scripts\python.exe .\analyze_synced_recoil.py `
  "D:\captures\m4_ads_ref.mp4" `
  "D:\captures\m4_fire.mp4" `
  --reference-mode ads `
  --shot-count 30 `
  --output-dir .\m4_synced_result
```

The script infers each video's `.mouse.csv`, `.frames.csv`, and
`.mouse-session.json` files, refuses recordings that report OBS output drops,
finds the longest left-button hold, detects the 30 ammunition changes, and
writes:

- `recoil_per_frame.csv`: observed view motion, predicted mouse contribution,
  and residual recoil on every video frame.
- `recoil_per_shot.csv`: the recovered trajectory and equivalent cumulative
  compensation in raw mouse counts.
- `summary.json`: synchronization, fit quality, shot cadence, and an explicit
  `formal_result_ready` quality gate.
- `recoil_trajectory.png`: right/up-positive review plot.

A hip-fire reference can be processed with the default `--reference-mode hip`
only as a feasibility check. It is deliberately marked non-formal because the
cross-FOV estimate is weaker than a same-ADS calibration.

By default, the script automatically scans an entire video and analyzes one complete magazine:

1. It scans the current ammo count in the lower-right corner of the video and uses periodic glyph changes to detect the firing frame range, fire rate, magazine capacity, and shot count. Dynamic programming then refines the segmentation, and the first frame where the count changes from `n` to `n-1` becomes a keyframe.
2. It extracts SIFT (or ORB) features only from the wall area outside the scope. The scope, weapon, HUD, and bright muzzle flashes are masked out.
3. Adjacent frames are matched with KNN feature matching and Lowe's ratio test. RANSAC rejects dynamic outliers such as muzzle flashes, and the inliers are fitted to a scale-free translation-and-rotation SE(2) transform.
4. The reticle is detected independently in every keyframe. For 2x optics, detection uses the intersection and small-angle rotation of the black horizontal and vertical tick marks. For the 1x optics used by the 93R and G18, it uses compact red-dot detection. Intense muzzle flashes do not directly determine the reticle position.
5. The detected tick-mark intersection in each keyframe is inverse-transformed into the coordinate system of the automatically detected start frame, producing the recoil trajectory.
6. The keyframes and video FPS are used to generate `shot_time_ms`, with the first shot at zero. The pixel trajectory, FOV, and optic magnification are used to estimate the maximum pitch angle.

The core equations are:

```text
A_i : background outside the scope in frame(i-1) -> frame(i)
C_i = C_(i-1) @ inverse(A_i) : frame(i) -> analysis_start_frame
p_i = C_i @ [reticle_x_i, reticle_y_i, 1]
```

The final point `p_i` therefore includes both the translation/rotation of the scene outside the scope and the reticle's within-frame jitter. The transform between keyframes is not treated directly as the point of impact.

## Installation and Usage

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\analyze_recoil.py "D:\DF\RM277_x1_opx2.mp4"
```

The video found on the development machine is `D:\DF\RM277_x1_opx2.mp4`, not `D:\DF\RM277\_x1\_opx2.mp4`. This is already the script's default input, so it can also be run directly:

```powershell
.\.venv\Scripts\python.exe .\analyze_recoil.py
```

For example, to analyze an AR57 video automatically and write the results to a separate directory:

```powershell
.\.venv\Scripts\python.exe .\analyze_recoil.py "D:\DF\AR57.mp4" `
  --output-dir .\recoil_output_ar57
```

Automatic mode assumes that the video contains one uninterrupted sequence from a full magazine down to `0`, so the number of consecutive ammo-count changes is treated as the magazine capacity. If the video contains only part of a magazine, all four manual parameters must be provided:

```powershell
.\.venv\Scripts\python.exe .\analyze_recoil.py VIDEO.mp4 `
  --start-frame 400 --end-frame 1000 --start-ammo 45 --shot-count 45
```

## Main Outputs

- `recoil_output/keyframes_recoil.csv`: Automatically detected ammo-change keyframes, `shot_time_ms`, on-screen reticle positions, unified coordinates, per-shot displacement, background components, reticle-jitter components, and rotation.
- `recoil_output/all_frames_motion.csv`: Match counts, RANSAC inliers, inlier ratio, reprojection error, translation, and rotation for every frame in the automatically detected firing interval.
- `recoil_output/ammo_detection.csv`: Per-frame ammo-glyph change scores, automatic thresholds, and coarse events for the entire video.
- `recoil_output/recoil_trajectory.png`: Recoil trajectory plot, with right/up as positive directions.
- `recoil_output/reticle_keyframes_contact_sheet.jpg`: Review sheet showing tick-mark intersection detection across 45 keyframes.
- `recoil_output/feature_mask.png`: The green area shows the region outside the scope that RANSAC actually uses.
- `recoil_output/summary.json`: Automatically detected magazine count, frame ranges, keyframes, shot times, minimum quality metrics, and the FOV/magnification-based pitch estimate.

The following CSV columns can be used directly:

- `recoil_x_right_px`, `recoil_y_up_px`: Cumulative recoil points relative to the analysis start frame.
- `shot_time_ms`: Shot time calculated from the ammo-change keyframes and video FPS, with the first shot at zero.
- `shot_delta_x_right_px`, `shot_delta_y_up_px`: Displacement of the current shot relative to the previous shot.
- `background_only_*`: The trajectory produced only by motion outside the scope, assuming the reticle remains fixed at the center of the frame.
- `reticle_jitter_contribution_*`: The additional component contributed by within-frame reticle movement.

## Calibration Parameters

The default ROI and masks are calibrated for the reference 2560x1440 video. If the ammo counter appears in a different location, use:

```powershell
.\.venv\Scripts\python.exe .\analyze_recoil.py VIDEO.mp4 --ammo-roi x0,y0,x1,y1
```

If motion quality is low in some frames, first inspect `status`, `inliers`, `inlier_ratio`, and `median_reprojection_error_px` in `all_frames_motion.csv`, then adjust:

```text
--feature-scale 0.75
--max-features 3500
--ratio-test 0.78
--ransac-threshold 2.5
```

The script does not silently accept frames where RANSAC fails. Failed steps are marked as `interpolated`, and their count is also recorded in `summary.json`.

The default pitch parameters match Recoil Trainer: the game's configured FOV is a 104-degree 4:3 reference horizontal FOV, and the optic is 2x. These values can be overridden:

```powershell
.\.venv\Scripts\python.exe .\analyze_recoil.py VIDEO.mp4 `
  --fov-deg 104 --fov-axis reference-horizontal --scope-magnification 2
```

The conversion uses pinhole projection, not a simple pixel-ratio-times-FOV calculation. `reference-horizontal` first reduces the ADS FOV according to the optic magnification, then calculates the vertical focal length using Recoil Trainer's 4:3 reference model.

## Converting to Recoil Trainer Profile JSON

`convert_to_recoiltrainer.py` reads the cumulative trajectory columns `recoil_x_right_px` and `recoil_y_up_px` and produces a Recoil Trainer `WeaponProfile` JSON file. Reticle jitter is preserved because these cumulative coordinates already include both scene motion outside the scope and within-frame tick-mark movement.

Convert the currently detected result with:

```powershell
.\.venv\Scripts\python.exe .\convert_to_recoiltrainer.py
```

The output is `recoil_output/rm277_x1_opx2_recoiltrainer.json`. The script:

- Maps the first detected shot to Profile `shot_index=0` without adding an artificial starting shot.
- Uses `shot_time_ms` from the detection stage whenever possible and verifies that `t_ms` starts at zero and is strictly increasing.
- Uses the same scale factor for both axes. By default, it normalizes the vertical span to 240 while preserving the actual horizontal movement ratio.
- Automatically writes the maximum pitch estimated from the 104-degree FOV and 2x optic in `summary.json` to `recorded_recoil_pitch_range_deg`.
- Writes `smoothing="spline"` and `smoothing_strength=0.2` by default.
- Calls Recoil Trainer's own smoothing and Auto Segment implementations to generate `segments` from changes in the smoothed trajectory's X direction.
- Uses the official `WeaponProfile` model in `C:\XiaodeDocuments\Programs\RecoilTrainer` for loading and round-trip validation.

Common customization parameters:

```powershell
.\.venv\Scripts\python.exe .\convert_to_recoiltrainer.py `
  .\recoil_output\keyframes_recoil.csv `
  --output .\recoil_output\rm277.json `
  --name "RM277 x1 OPX2" `
  --target-vertical-span 240
```

The detected data determines the trajectory shape, while `recorded_recoil_pitch_range_deg` determines the physical pitch angle represented by that vertical trajectory in the training range. If the exact total recoil angle is known from the game, use `--recorded-pitch-deg VALUE` to override the automatic estimate. For debugging without trainer-driven automatic segmentation, explicitly use `--segmentation single`.

## Batch-Processing Delta Force Videos and Importing into the Steam Version

`batch_delta_force.py` discovers every `.mp4` file under `D:\DF` and runs the complete analysis and conversion pipeline for each one. `--import-steam` writes to the Steam data directory only after every item passes the magazine OCR, keyframe-count, RANSAC, reticle-confidence, pitch, bilingual-field, smoothing, and segmentation checks:

```powershell
.\.venv\Scripts\python.exe .\batch_delta_force.py `
  --max-workers 2 --resume --import-steam
```

Batch-processing rules:

- Game names are stored as `Delta Force` and `三角洲行动`. Trainer list titles use `Delta Force · <weapon>` in English and `三角洲·<weapon>` in Chinese. The middle dot avoids appearing like a vertical `|` or `I` in card fonts. Workshop titles use `Delta Force · <weapon> Recoil` and `三角洲·<weapon> 后坐力`; the weapon field itself still stores only the weapon name.
- The 93R and G18 use 1x; all other weapons use 2x.
- The reconstructed pitch for the QBZ95-1 is divided by `0.89`; no amplitude correction is applied to other weapons.
- Every profile uses `spline` with a smoothing strength of `0.2` and calls Recoil Trainer's Auto Segment implementation.
- `--resume` reuses only outputs that still pass the current pipeline version and all quality checks.
- `--convert-only` reuses validated CSV and summary files and rebuilds only the JSON. This is useful when changing titles or other Profile metadata in bulk.
- Before importing, `RecoilTrainer.exe` must not be running. The database and complete profiles directory are automatically backed up to `%LOCALAPPDATA%\RecoilTrainer\codex_backups`.

Batch results are written to `delta_force_batch_output`, including:

- `batch_manifest.json`: Every video, magazine count, frame range, pitch, optic magnification, correction factor, quality metrics, and final JSON path.
- `steam_import_report.json`: The 43 profile IDs imported in this run, the Steam data directory, and the recoverable backup location.
- One subdirectory per weapon containing all intermediate analysis artifacts, review images, logs, and bilingual Recoil Trainer JSON files.

### Discontinuity Audit

Automatic ammo events are used only to locate the firing interval coarsely. The final `1 -> 0` boundary is determined by a stable empty-magazine frame plus joint dynamic programming across all ammo states. This prevents a missed weak change peak from causing the empty-magazine animation to be misinterpreted as the final few shots.

The CSV always preserves the original `recoil_*` and `shot_delta_*` columns produced by combining feature matching, RANSAC, and reticle detection. If the final point satisfies all the following conditions, the pipeline identifies it as an empty-magazine/weapon-drop animation after firing rather than a new recoil impulse:

- The vertical drop exceeds 80 px.
- The total displacement exceeds 100 px.
- The displacement also exceeds both five times the median previous per-shot displacement and `median + 8 x robust_sigma`.

In this case, only the `trainer_recoil_*` columns extrapolate the last shot using the robust median of recent reliable per-shot increments; the original observation is not overwritten. `summary.json` and `data.reconstruction.trajectory_corrections` in the final JSON record the original increment, replacement increment, thresholds, and reason. Mid-sequence recoil recovery during burst fire, such as the long interval between M16A4 three-round bursts, is not modified by this rule.

Batch reruns restore previously imported Profile IDs from the previous `batch_manifest.json` and `steam_import_report.json`, so Steam profiles are updated in place instead of creating duplicate profiles with the same name.
