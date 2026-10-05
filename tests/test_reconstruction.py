"""Known geometry and review contracts; independent of any AI service."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from recoil_reconstruction.common import AnalysisCancelled, apply_matrix, parse_scaled_roi
from recoil_reconstruction.engine import main, parse_args
from recoil_reconstruction.quality import AnchorTracker, ShotEvidence, observed_boundary
from recoil_reconstruction.service import AnalysisRequest, ReviewSession, roi_argument
from recoil_reconstruction.tracking import OutsideFeatureMatcher


def session_fixture():
    request = AnalysisRequest("fixture.mp4", recording_conditions_confirmed=True, calibration_confirmed=True, magnification=1)
    poses = {}
    for f in range(30):
        p = np.eye(3); p[0,2] = .8*f; p[1,2] = -.9*f; poses[f] = p
    rows=[]
    for f in (5,11,17):
        world=apply_matrix(poses[f],(50,50))
        rows.append(dict(world_x_ref=world[0],world_y_ref=world[1],reticle_screen_x=50,reticle_screen_y=50,keyframe=f))
    evidence=[ShotEvidence(i+1,f,f,None,3-i,2-i,["timing_ambiguous"]) for i,f in enumerate((5,11,17))]
    motion={f:dict(status="ok",inlier_ratio=1,inliers=40) for f in poses}
    s=ReviewSession(request,dict(fps=120,video_width=100,video_height=100),copy.deepcopy(rows),motion,evidence,poses,copy.deepcopy(rows),"")
    s._rebuild_relative()
    return s


def matcher():
    return OutsideFeatureMatcher(640,360,"sift",1,2500,.75,2.5,12,.35,100,5)


def texture():
    rng=np.random.default_rng(723)
    gray=rng.integers(0,180,(360,640),dtype=np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(gray,(5,5),1.2),cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("occlusion",[False,True])
def test_known_translation_with_noise_and_occlusion(occlusion):
    original=texture(); expected=np.array([[1,0,8.25],[0,1,-5.5]],dtype=float)
    moved=cv2.warpAffine(original,expected,(640,360))
    if occlusion: moved[90:180,100:220]=0
    rng=np.random.default_rng(724)
    moved=np.clip(moved.astype(float)+rng.normal(0,2,moved.shape),0,255).astype(np.uint8)
    m=matcher(); result=m.match(1,m.extract(original),m.extract(moved))
    assert result.matrix is not None
    assert np.linalg.norm(apply_matrix(result.matrix,(320,180))-apply_matrix(np.vstack([expected,[0,0,1]]),(320,180)))<.5


def test_known_pinhole_rotation_small_angle():
    original=texture(); angle=np.deg2rad(2)
    k=np.array([[450,0,320],[0,450,180],[0,0,1.]])
    r=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
    h=k@r@np.linalg.inv(k)
    moved=cv2.warpPerspective(original,h,(640,360))
    m=matcher(); result=m.match(1,m.extract(original),m.extract(moved))
    gt=h@np.array([320,180,1]); gt=gt[:2]/gt[2]
    assert result.matrix is not None
    # SE2 is approximate under perspective; this is a bounded synthetic check.
    assert np.linalg.norm(apply_matrix(result.matrix,(320,180))-gt)<3


def test_anchor_rejects_occlusion_and_recovers_original_patch():
    original=texture(); tracker=AnchorTracker(original,(100,100,150,150))
    moved=cv2.warpAffine(original,np.array([[1.,0,7.5],[0,1,-4.25]]),(640,360))
    good=tracker.track(moved,1)
    assert good["valid"] and abs(good["x"]+7.5)<.5 and abs(good["y"]+4.25)<.5
    last=tracker.last.copy()
    assert not tracker.track(np.zeros_like(original),2)["valid"]
    assert np.array_equal(last,tracker.last)
    assert tracker.track(moved,3)["valid"]


def test_hud_observation_remains_distinct_from_inference():
    readings=[3]*8+[2]*8
    assert observed_boundary(readings,[.01]*16,3,2,9)==8
    assert observed_boundary(readings,[.2]*16,3,2,9) is None
    s=session_fixture(); s.evidence[1].observed_frame=10
    assert s.frames[1]==11 and s.evidence[1].observed_frame==10


def test_manual_review_recomputes_geometry_and_timing_and_preserves_raw():
    s=session_fixture(); raw=copy.deepcopy(s.raw_rows)
    for i in range(3): s.confirm_shot(i)
    s.change_frame(1,12,reticle_xy=(53,47))
    assert s.evidence[1].inferred_frame==11 and s.frames[1]==12
    assert "pending_review" in s.import_errors()
    assert s.working_rows[1]["world_x_ref"]==pytest.approx(62.6)
    assert s.working_rows[1]["shot_time_ms"]==58
    assert s.raw_rows==raw
    s.confirm_shot(1)
    payload=s.export_payload("Reviewed")
    points=payload["points"]; scale=payload["reconstruction"]["pixel_scale"]
    assert points[1]["x"]-points[0]["x"]==pytest.approx(-8.6*scale)
    assert points[1]["y"]-points[0]["y"]==pytest.approx(9.3*scale)
    assert payload["reconstruction"]["shots"][1]["correction"]["kind"]=="manual"
    assert json.loads(json.dumps(payload))["reconstruction"]==payload["reconstruction"]


def test_review_cannot_bypass_motion_quality_or_invalid_order():
    s=session_fixture()
    with pytest.raises(ValueError): s.change_frame(1,18,reticle_xy=(50,50))
    with pytest.raises(ValueError): s.export_payload("Unreviewed")
    for i in range(3): s.confirm_shot(i)
    for r in s.motion.values(): r["status"]="interpolated"
    assert "motion_quality" in s.import_errors()
    with pytest.raises(ValueError): s.export_payload("Still bad")


def test_cancellation_before_opening_video():
    with pytest.raises(AnalysisCancelled):
        main(parse_args(["not-present.mp4"]),cancelled=lambda:True)




@pytest.mark.parametrize("size",[(1280,720),(1920,1080),(2560,1440),(3840,2160)])
def test_ui_region_keeps_exact_native_pixel_boundaries(size):
    width,height=size
    roi=(113,37,741,128)
    assert parse_scaled_roi(roi_argument(roi,width,height),width,height)==roi




def test_source_change_requires_reanalysis(tmp_path):
    path=tmp_path/"clip.mp4"; path.write_bytes(b"original")
    s=session_fixture()
    from dataclasses import replace
    s.request=replace(s.request,video=str(path))
    stat=path.stat(); s.source_fingerprint=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    assert "source_changed" not in s.import_errors()
    path.write_bytes(b"different video")
    assert "source_changed" in s.import_errors()
    with pytest.raises(ValueError): s.change_frame(1,12,reticle_xy=(50,50))


def test_end_to_end_known_video_preserves_reviewed_frames_and_reticle_motion(tmp_path):
    import csv
    video=tmp_path/"known.avi"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),60,(640,360))
    assert writer.isOpened()
    base=texture()
    try:
        for f in range(25):
            image=cv2.warpAffine(base,np.array([[1.,0,.6*f],[0,1,-.5*f]]),(640,360))
            cv2.circle(image,(320+round(.1*f),180+round(.2*f)),3,(0,0,255),-1)
            writer.write(image)
    finally: writer.release()
    output=tmp_path/"output"
    args=parse_args([str(video),"--output-dir",str(output),"--start-frame","4","--end-frame","20",
                    "--start-ammo","3","--shot-count","3","--terminal-keyframe","17",
                    "--reviewed-keyframes","5,11,17","--reticle-mode","red-dot","--feature-scale","1",
                    "--ammo-roi","0.8,0.8,0.95,0.95"])
    main(args)
    with (output/"keyframes_recoil.csv").open(encoding="utf-8-sig") as stream: rows=list(csv.DictReader(stream))
    assert [int(r["keyframe"]) for r in rows]==[5,11,17]
    assert [int(r["shot_time_ms"]) for r in rows]==[0,100,200]
    # Known world point = reticle screen position minus generated background shift.
    gt=np.array([[320+round(.1*f)-.6*f,180+round(.2*f)+.5*f] for f in (5,11,17)])
    gt-=gt[0]; gt[:,1]*=-1
    measured=np.array([[float(r["recoil_x_right_px"]),float(r["recoil_y_up_px"])] for r in rows])
    measured-=measured[0]
    assert np.max(np.linalg.norm(measured-gt,axis=1))<.7


def test_compatibility_facade_preserves_cli_and_public_helpers():
    import analyze_recoil as legacy
    from recoil_reconstruction import engine, ammo, tracking
    assert legacy.main is engine.main
    assert legacy.parse_args is engine.parse_args
    assert legacy.recognize_ammo_number is ammo.recognize_ammo_number
    assert legacy.OutsideFeatureMatcher is tracking.OutsideFeatureMatcher
    assert engine.DEFAULT_AMMO_TEMPLATE_FILE.is_file()
    assert legacy.parse_args(["clip.mp4"]).video=="clip.mp4"
    with pytest.raises(SystemExit): legacy.parse_args([])
    with np.load(engine.DEFAULT_AMMO_TEMPLATE_FILE) as templates:
        assert "calibration_video" not in templates.files
