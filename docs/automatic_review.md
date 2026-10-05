# Local reconstruction and frame review

The `recoil_reconstruction` package is the portable, no-AI analysis/review backend.
`analyze_recoil.py` remains the compatible CLI and helper import surface. Runtime analysis
uses OpenCV/NumPy, digit templates, temporal constraints and multi-feature RANSAC; it does
not require PaddleOCR, an AI model, an account or a remote analysis service.

The interactive recording dialog is implemented in
[RecoilTrainer](https://github.com/Xiaode2333/RecoilTrainer/tree/codex/reconstruction-review-08c79614).
Its layout gives the recording the main area, with a vertical trajectory sidebar and a small
ammo preview. This repository owns the backend; it does not bundle the Trainer's Qt/i18n UI.
The Trainer vendors this package and its template file using its synchronization script.

## Four improvements

- Known-geometry regression exercises translation, noise, occlusion, small pinhole rotation,
  synthetic video/reticle motion, cancellation and native-pixel ROI roundtrips.
- Optional `--anchor-roi` tracking compares with the original reference patch. Weak matches
  do not move its reference or replace the main RANSAC motion estimate.
- `ReviewSession` supports explicit frame/reticle correction and confirmation. Edits recompute
  relative geometry and shot times; the affected shot needs confirmation again.
- Inferred frames, independent HUD candidates and selected frames remain distinct. Export
  preserves raw observations, reviewed observations, reasons, corrections, calibration and
  a source fingerprint. Reviewed CLI keyframes are not overwritten by cadence fitting.

## CLI and API

Existing commands still work:

```powershell
python analyze_recoil.py recording.mp4 --reticle-mode red-dot --scope-magnification 1 --output-dir output
python analyze_recoil.py --help
```

For manual verification, use `--reviewed-keyframes` with reviewed source-frame indices and
the existing manual capture-range parameters. Frame indices start at zero. CLI ROI arguments
retain their reference/normalized coordinate convention. The service API accepts native-pixel
ROIs; `roi_argument` converts them without introducing rounding drift, including 4K captures.

```python
from recoil_reconstruction.service import AnalysisRequest, analyze

request = AnalysisRequest(
    "recording.mp4",
    reticle_mode="red-dot",
    magnification=1,
    ammo_roi=(2348, 1218, 2464, 1268),  # Native pixels; adjust for the recording.
)
session = analyze(request, "output")
for shot in session.pending:
    print(shot.shot, shot.inferred_frame, shot.observed_frame, shot.reasons)
# After inspecting the recording:
# session.change_frame(index, selected_frame, reticle_xy=(native_x, native_y))
# session.confirm_shot(index)
# Set the recording/calibration acknowledgements only after actually checking them.
# session.export_payload("Profile name")
```

`analyze` accepts progress and cooperative cancellation callbacks. Import/export gates check
pending review, capture/calibration acknowledgements, cadence, motion quality, finite geometry
and source changes. Export converts right/up recoil into right/down mouse compensation by
negating X only, with the same positive scale on both axes. It stores the source basename and
digest rather than an absolute local source path.

## Limits and evidence

Initial support is a stationary, full-magazine Delta Force recording with no mouse movement
or recoil compensation, readable ammo digits and textured surroundings. The service admits
30-240 FPS, up to 4K pixel count, 9000 frames and five minutes. It assumes constant-FPS timing;
dropped/duplicate frames are not reconstructed. Mouse input cannot be separated from recoil.
HUD timing is not muzzle-launch timing; SE2/FOV calibration is approximate. Quality gates are
evidence checks, not calibrated confidence probabilities or arbitrary-gameplay robustness.

The actual G18 rerun recovered 33 shots, with 25 conservatively flagged. Its automatic frame
sequence matched 26/33 independently annotated first visible HUD decrements exactly, with
all differences within one frame. The optional marker had no lost/disagreeing shot samples.
M4A1 recovered 60 shots, first frame 267 and last 800, with 56 review flags and 43 lost marker
samples; no independent per-shot M4A1 timing annotation exists. Marker loss is surfaced for
review and does not replace the primary tracking path. Two recordings do not establish
universal accuracy. The Trainer's real G18 analyze/review/save/reload smoke preserved the
selected HUD frame sequence, timestamps and provenance in isolated test data.

## Regression

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_reconstruction.py -q
```

The regression uses synthetic footage with known motion; it does not require game recordings
or the Trainer checkout. Physical player acceptance and installed-App DPI/template lookup
remain separate checks. Existing batch outputs, source media and unrelated local migrations
are outside this change.
