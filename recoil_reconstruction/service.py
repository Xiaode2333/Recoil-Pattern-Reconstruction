"""UI-independent automatic analysis, reversible review, and compensation export.

Pixel paths are observations under the reconstruction model, not ground-truth
bullet impacts. All player review records are retained separately from them.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
import argparse
import csv
import hashlib
import json
import math
import copy
import cv2
import numpy as np

from . import engine, ammo, reticle
from .common import AnalysisCancelled, RunContext, RUN_CONTEXT, apply_matrix, checkpoint, estimate_pitch_range_deg, read_video_info, seek_and_read
from .quality import ShotEvidence, observed_boundary, cadence_valid


@dataclass(frozen=True)
class AnalysisRequest:
    video: str
    reticle_mode: str = "tick-lines"
    ammo_roi: tuple[int,int,int,int] = (2348,1218,2464,1268)
    anchor_roi: tuple[int,int,int,int] | None = None
    fov_deg: float = 104.
    fov_axis: str = "reference-horizontal"
    magnification: float = 2.
    cadence: str = "regular"
    start_frame: int | None = None
    end_frame: int | None = None
    magazine: int | None = None
    terminal_frame: int | None = None
    recording_conditions_confirmed: bool = False
    calibration_confirmed: bool = False


@dataclass
class ReviewSession:
    request: AnalysisRequest
    summary: dict
    raw_rows: list[dict]
    motion: dict[int,dict]
    evidence: list[ShotEvidence]
    poses: dict[int,np.ndarray]
    working_rows: list[dict]
    output_dir: str
    manual_reticles: dict[int,tuple[float,float]] = field(default_factory=dict)
    revision: int = 0
    source_fingerprint: dict | None = None

    @property
    def pending(self):
        return [e for e in self.evidence if e.reasons and not e.reviewed]

    @property
    def frames(self):
        return [e.sample_frame for e in self.evidence]

    def import_errors(self):
        errors=[]
        if self.source_fingerprint:
            try:
                stat=Path(self.request.video).stat()
                if stat.st_size!=self.source_fingerprint["size"] or stat.st_mtime_ns!=self.source_fingerprint["mtime_ns"]:
                    errors.append("source_changed")
            except OSError: errors.append("source_changed")
        if self.pending: errors.append("pending_review")
        if not self.request.recording_conditions_confirmed: errors.append("recording_conditions")
        if not self.request.calibration_confirmed: errors.append("calibration")
        if not cadence_valid(self.frames,float(self.summary["fps"]),self.request.cadence): errors.append("cadence")
        statuses=[r.get("status") for r in self.motion.values()]
        if statuses.count("interpolated")>max(2,math.ceil(len(statuses)*.01)): errors.append("motion_quality")
        measured=[r for r in self.motion.values() if r.get("status") not in ("reference","interpolated")]
        weak=sum(float(r.get("inlier_ratio",0))<.35 or int(r.get("inliers",0))<12 for r in measured)
        if measured and weak>max(2,math.ceil(len(measured)*.05)) and "motion_quality" not in errors:
            errors.append("motion_quality")
        if len(self.working_rows)!=len(self.evidence) or len(self.evidence)<2: errors.append("shot_count")
        coords=np.array([[r["recoil_x_right_px"],r["recoil_y_up_px"]] for r in self.working_rows],dtype=float)
        if coords.size==0 or not np.isfinite(coords).all(): errors.append("coordinates")
        if len(coords)>1 and np.linalg.norm(np.ptp(coords,axis=0))<.5: errors.append("coordinates")
        return errors

    def confirm_shot(self, index):
        self.evidence[index].reviewed=True
        self.revision+=1

    def change_frame(self,index,frame,*,reticle_xy=None,frame_image=None):
        if "source_changed" in self.import_errors(): raise ValueError("source_changed")
        frame=int(frame)
        if frame not in self.poses: raise ValueError("Frame is outside analyzed motion range")
        if index>0 and frame<=self.frames[index-1] or index+1<len(self.frames) and frame>=self.frames[index+1]:
            raise ValueError("Shot frames must be strictly increasing")
        previous=self.evidence[index].sample_frame
        result=reticle_xy or (self.manual_reticles.get(index) if previous==frame else None)
        if result is None:
            if frame_image is None:
                cap=cv2.VideoCapture(self.request.video)
                try: image=seek_and_read(cap,frame)
                finally: cap.release()
            else:
                image=frame_image
                if image.shape[:2]!=(self.summary["video_height"],self.summary["video_width"]):
                    raise ValueError("Preview dimensions differ from analyzed video")
            detected=reticle.detect_reticle(image,self.request.reticle_mode,3.,.25)
            result=(detected.x,detected.y)
        if len(result)!=2 or not np.isfinite(result).all(): raise ValueError("Invalid reticle")
        w,h=self.summary["video_width"],self.summary["video_height"]
        if not 0<=result[0]<w or not 0<=result[1]<h: raise ValueError("Reticle outside frame")
        # Do not apply a clicked reticle from an earlier image to a new frame.
        if reticle_xy is not None: self.manual_reticles[index]=tuple(result)
        elif previous!=frame: self.manual_reticles.pop(index,None)
        row=copy.deepcopy(self.working_rows[index])
        world=apply_matrix(self.poses[frame],result)
        row.update(keyframe=frame,reticle_screen_x=float(result[0]),reticle_screen_y=float(result[1]),
                   world_x_ref=float(world[0]),world_y_ref=float(world[1]))
        self.working_rows[index]=row
        e=self.evidence[index]
        e.sample_frame=frame; e.reviewed=False
        if "manual_correction" not in e.reasons: e.reasons.append("manual_correction")
        e.correction={"previous_frame":previous,"selected_frame":frame,"reticle_xy":list(result),"kind":"manual"}
        self.revision+=1
        self._rebuild_relative()

    def _rebuild_relative(self):
        first=self.working_rows[0]
        x0,y0=float(first["world_x_ref"]),float(first["world_y_ref"])
        t0=self.frames[0]; fps=float(self.summary["fps"])
        for i,row in enumerate(self.working_rows):
            row["recoil_x_right_px"]=float(row["world_x_ref"])-x0
            row["recoil_y_up_px"]=-(float(row["world_y_ref"])-y0)
            row["shot_time_ms"]=round((self.frames[i]-t0)*1000/fps)
            row["time_sec"]=self.frames[i]/fps

    def export_payload(self,name):
        if not name.strip(): raise ValueError("name")
        errors=self.import_errors()
        if errors: raise ValueError(",".join(errors))
        coords=np.array([[r["recoil_x_right_px"],r["recoil_y_up_px"]] for r in self.working_rows])
        coords-=coords[0]
        span=float(np.ptp(coords[:,1])); scale=240/span if span>1 else 1.
        # Recoil is right/up; compensation is right/down. Negate only X.
        transformed=np.column_stack([-coords[:,0],coords[:,1]])*scale
        transformed-=transformed.min(axis=0); transformed+=20
        pitch,focal=estimate_pitch_range_deg(coords[:,1],int(self.summary["video_width"]),int(self.summary["video_height"]),
                    self.request.fov_deg,self.request.fov_axis,self.request.magnification)
        if not math.isfinite(pitch) or not .01<pitch<89: raise ValueError("calibration")
        points=[{"shot_index":i,"x":float(x),"y":float(y),"t_ms":int(self.working_rows[i]["shot_time_ms"])} for i,(x,y) in enumerate(transformed)]
        for i,p in enumerate(points):
            if not np.allclose([p["x"]-points[0]["x"],p["y"]-points[0]["y"]],[-coords[i,0]*scale,coords[i,1]*scale]):
                raise ValueError("coordinates")
        metadata={"schema_version":1,"algorithm":engine.PIPELINE_VERSION,"review_revision":self.revision,
              "profile_coordinate_convention":"mouse-compensation-screen-right-down-v1","pixel_scale":scale,
              "capture":{**asdict(self.request),"video":Path(self.request.video).name},"source_fingerprint":self.source_fingerprint,
              "fps":self.summary["fps"],"focal_px":focal,
              "display_boundary_candidates":self.summary.get("display_boundary_candidates",[]),
              "shots":[asdict(e) for e in self.evidence],"raw_observations":self.raw_rows,
              "reviewed_observations":[{k:r[k] for k in ("keyframe","shot_time_ms","world_x_ref","world_y_ref","reticle_screen_x","reticle_screen_y","recoil_x_right_px","recoil_y_up_px")} for r in self.working_rows],
              "limitations":["HUD timing is not muzzle-launch timing","Frame times assume constant FPS; dropped/duplicate frames are not reconstructed",
                             "SE2/pinhole calibration is approximate","Mouse contribution not separated"]}
        metadata=json.loads(json.dumps(metadata,ensure_ascii=False))
        # Store a source name and digest, without leaking a local absolute path.
        return {"profile_id":"","name":name.strip(),"points":points,"segments":[{"segment_id":"Seq1","start_shot":0,"end_shot":len(points)-1}],
             "smoothing":"spline","smoothing_strength":.2,"device_mode":"kbm",
             "recorded_recoil_pitch_range_deg":pitch,"reconstruction":metadata}


def analyze(request:AnalysisRequest,output_dir,*,progress=None,cancelled=None):
    source_stat=Path(request.video).stat()
    width,height,fps,frame_count=read_video_info(Path(request.video))
    validate_recording(width,height,fps,frame_count)
    if request.reticle_mode not in ("tick-lines","red-dot","finals-red","finals-green","finals-front-sight"):
        raise ValueError("Unknown reticle mode")
    for roi in (request.ammo_roi,request.anchor_roi):
        if roi and not (0<=roi[0]<roi[2]<=width and 0<=roi[1]<roi[3]<=height): raise ValueError("ROI outside frame")
    argv=[request.video,"--output-dir",str(output_dir),"--reticle-mode",request.reticle_mode,
          "--scope-magnification",str(request.magnification),"--fov-deg",str(request.fov_deg),"--fov-axis",request.fov_axis,
          "--cadence-model",request.cadence,"--ammo-min-segment",str(max(2,round(fps/30))),
          "--ammo-max-segment",str(max(4,round(fps*22/120)))]
    # Normalized CLI ROIs round back to the exact native pixels, including 4K.
    encode=lambda roi:roi_argument(roi,width,height)
    argv += ["--ammo-roi",encode(request.ammo_roi)]
    if request.anchor_roi: argv += ["--anchor-roi",encode(request.anchor_roi)]
    if request.magazine is not None:
        if request.start_frame is None or request.end_frame is None or request.terminal_frame is None: raise ValueError("Manual range needs a reviewed last shot")
        argv += ["--start-frame",str(request.start_frame),"--end-frame",str(request.end_frame),"--start-ammo",str(request.magazine),
                 "--shot-count",str(request.magazine),"--terminal-keyframe",str(request.terminal_frame)]
    engine.main(engine.parse_args(argv),progress=progress,cancelled=cancelled)
    root=Path(output_dir)
    summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    with (root/"keyframes_recoil.csv").open(encoding="utf-8-sig") as stream:
        rows=list(csv.DictReader(stream))
    with (root/"all_frames_motion.csv").open(encoding="utf-8-sig") as stream:
        motion_rows=list(csv.DictReader(stream))
    motion={int(r["frame"]):r for r in motion_rows}
    poses={int(r["frame"]):pose_from_row(r,width,height) for r in motion_rows}
    token=RUN_CONTEXT.set(RunContext(progress,cancelled))
    try:
        _,_,features=ammo.extract_full_ammo_features(Path(request.video),request.ammo_roi)
        templates=ammo.load_ammo_digit_templates(engine.DEFAULT_AMMO_TEMPLATE_FILE)
        readings=[]; errors=[]
        for i,cells in enumerate(features):
            if i%30==0: checkpoint("evidence",i,len(features))
            value,error=ammo.recognize_ammo_number(cells,templates)
            readings.append(value); errors.append(error)
    finally: RUN_CONTEXT.reset(token)
    evidence=[]
    for i,row in enumerate(rows):
        f=int(row["keyframe"]); before=int(row["ammo_before"]); after=int(row["ammo_after"])
        observed=observed_boundary(readings,errors,before,after,f)
        reasons=[]
        if observed is None: reasons.append("timing_ambiguous")
        elif observed!=f: reasons.append("timing_disagreement")
        if motion[f]["status"]=="interpolated": reasons.append("motion_interpolated")
        if f in summary["reticle_interpolated_keyframes"]: reasons.append("reticle_interpolated")
        if i>=len(rows)-8: reasons.append("tail_review")
        e=ShotEvidence(i+1,f,f,observed,before,after,reasons,original_reasons=tuple(reasons))
        evidence.append(e)
    session=ReviewSession(request,summary,rows,motion,evidence,poses,copy.deepcopy(rows),str(root))
    digest=hashlib.sha256()
    with Path(request.video).open("rb") as source:
        while chunk:=source.read(1024*1024):
            if cancelled and cancelled(): raise AnalysisCancelled("Analysis cancelled")
            digest.update(chunk)
    session.source_fingerprint={"size":source_stat.st_size,"mtime_ns":source_stat.st_mtime_ns,"sha256":digest.hexdigest()}
    if "source_changed" in session.import_errors(): raise ValueError("source_changed")
    session._rebuild_relative()
    # Independent digit templates can disagree during HUD animation. Expose
    # both candidates for review instead of silently moving the sample.
    anchors={int(r["frame"]):r for r in summary.get("anchor_track",[])}
    if anchors:
        anchor0=anchors.get(session.frames[0])
        bg0=poses[session.frames[0]][:2,2]
        for e in evidence:
            a=anchors.get(e.sample_frame)
            if not a or not a["valid"] or not anchor0 or not anchor0["valid"]:
                e.reasons.append("anchor_lost")
            else:
                center=np.array([width/2,height/2])
                bg=apply_matrix(poses[e.sample_frame],center)-apply_matrix(poses[session.frames[0]],center)
                expected=np.array([bg[0],-bg[1]])
                delta=np.array([a["x"]-anchor0["x"],a["y"]-anchor0["y"]])
                if np.linalg.norm(delta-expected)>max(40.,height*.025): e.reasons.append("anchor_disagreement")
            e.original_reasons=tuple(e.reasons)
    (root/"review_evidence.json").write_text(json.dumps([asdict(e) for e in evidence],ensure_ascii=False,indent=2),encoding="utf-8")
    if cancelled and cancelled(): raise AnalysisCancelled("Analysis cancelled")
    if progress: progress("complete",1,1)
    return session


def pose_from_row(row,width,height):
    angle=math.radians(float(row["rotation_to_reference_deg"]))
    rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    center=np.array([width/2,height/2])
    target=np.array([float(row["center_x_in_reference"]),float(row["center_y_in_reference"])])
    pose=np.eye(3); pose[:2,:2]=rotation; pose[:2,2]=target-rotation@center
    return pose


def validate_recording(width,height,fps,frame_count):
    if not (width>0 and height>0 and width*height<=3840*2160 and np.isfinite(fps)
            and 30<=fps<=240 and 0<frame_count<=9000 and frame_count/fps<=300):
        raise ValueError("Use a short 30–240 FPS recording, up to 4K, 9000 frames and five minutes")


def roi_argument(roi,width,height):
    return ",".join(repr(v/(width if i%2==0 else height)) for i,v in enumerate(roi))
