"""Independent fixed-reference checks and explicit per-shot evidence."""
from dataclasses import dataclass, field
import math
import cv2
import numpy as np


class AnchorTracker:
    """Compare each frame with the original patch; never update a weak match."""

    def __init__(self, frame, roi):
        x0,y0,x1,y1=roi
        self.template=cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).copy()
        if min(self.template.shape) < 8 or np.std(self.template) < 3:
            raise ValueError("Anchor requires a textured, stationary region")
        self.origin=np.array([(x0+x1)/2,(y0+y1)/2],dtype=float)
        self.last=self.origin.copy()

    def track(self, frame, index):
        h,w=self.template.shape
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        left=max(0,int(self.last[0]-w/2-250)); top=max(0,int(self.last[1]-h/2-250))
        right=min(gray.shape[1],int(self.last[0]+w/2+250)); bottom=min(gray.shape[0],int(self.last[1]+h/2+250))
        region=gray[top:bottom,left:right]
        if region.shape[0]<h or region.shape[1]<w:
            return {"frame":index,"confidence":0.,"valid":False,"x":0.,"y":0.}
        surface=cv2.matchTemplate(region,self.template,cv2.TM_CCOEFF_NORMED)
        _,score,_,location=cv2.minMaxLoc(surface)
        x,y=location
        def refine(a,b,c):
            d=float(a)-2*float(b)+float(c)
            return float(np.clip(.5*(float(a)-float(c))/d,-.5,.5)) if abs(d)>1e-9 else 0.
        ox=refine(surface[y,x-1],surface[y,x],surface[y,x+1]) if 0<x<surface.shape[1]-1 else 0.
        oy=refine(surface[y-1,x],surface[y,x],surface[y+1,x]) if 0<y<surface.shape[0]-1 else 0.
        point=np.array([left+x+w/2+ox,top+y+h/2+oy])
        valid=score>=.8
        if valid: self.last=point
        delta=point-self.origin
        return {"frame":index,"confidence":float(score),"valid":bool(valid),"x":float(-delta[0]),"y":float(delta[1])}


@dataclass
class ShotEvidence:
    shot: int
    inferred_frame: int
    sample_frame: int
    observed_frame: int | None
    ammo_before: int
    ammo_after: int
    reasons: list[str] = field(default_factory=list)
    reviewed: bool = False
    correction: dict | None = None
    original_reasons: tuple[str, ...] = ()


def observed_boundary(readings, errors, expected_before, expected_after, center, radius=4):
    """Trust a readable unit decrement, not a brightness peak or a clock alone."""
    candidates=[]
    for f in range(max(2,center-radius),min(len(readings)-2,center+radius+1)):
        if all(readings[j]==expected_before and errors[j]<.08 for j in (f-2,f-1)) and all(
            readings[j]==expected_after and errors[j]<.08 for j in (f,f+1)):
            candidates.append(f)
    return candidates[0] if len(candidates)==1 else None


def cadence_valid(frames, fps, model):
    if len(frames)<2 or any(b<=a for a,b in zip(frames,frames[1:])):
        return False
    gaps=np.diff(frames)
    if np.min(gaps)<max(1,math.floor(fps/30)):
        return False
    if model=="regular":
        return bool(np.max(np.abs(gaps-np.median(gaps)))<=max(2,math.ceil(np.median(gaps)*.2)))
    return True
