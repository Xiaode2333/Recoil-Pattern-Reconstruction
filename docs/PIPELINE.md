# Reconstruction pipeline

## Coordinate model

For adjacent frames, `A_i` maps background coordinates from frame `i-1` into
frame `i`. The accumulated inverse transform maps observations back to the
analysis-start frame:

```text
A_i : background(frame i-1) -> background(frame i)
C_i = C_(i-1) @ inverse(A_i)
p_i = C_i @ [reticle_x_i, reticle_y_i, 1]
```

The point `p_i` therefore includes both scene motion and motion of the detected
reticle inside the frame.

## Stages

1. Detect the firing interval from ammunition-display changes, or accept manual
   start/end frames for data without a compatible display.
2. Mask the optic, foreground object, HUD, and transient flashes.
3. Extract SIFT features (ORB is available as a fallback), perform KNN matching,
   and apply Lowe's ratio test.
4. Estimate a scale-free SE(2) step with RANSAC. Reject or interpolate steps that
   miss the configured inlier, reprojection-error, translation, or rotation gates.
5. Detect the reticle independently at each event keyframe.
6. Transform keyframe reticle locations into a common coordinate system and
   write the trajectory, diagnostics, and review figures.

## Quality gates

`all_frames_motion.csv` records match count, RANSAC inliers, inlier ratio,
reprojection error, translation, rotation, and interpolation status. The JSON
summary aggregates failed steps and reticle confidence. These diagnostics are
part of the output contract: a trajectory should not be treated as valid merely
because a plot was produced.

The default regions of interest are calibrated for 2560x1440 input and scale
with resolution. For a different layout, pass an explicit `--ammo-roi` and
review `feature_mask.png` before accepting a result.
