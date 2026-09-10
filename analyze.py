#!/usr/bin/env python3
"""Local basketball video analysis: RF-DETR + BoT-SORT, with conservative metadata.

Source frames are streamed. Track IDs are scoped to a shot; they are not player
identities. Team colors use a demo-specific blue/white uniform classifier.
"""
import argparse
from collections import Counter, defaultdict, deque
import csv
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent
for key, directory in {"MPLCONFIGDIR":"matplotlib", "HF_HOME":"huggingface", "TORCH_HOME":"torch", "RF_HOME":"rfdetr-cache", "YOLO_CONFIG_DIR":"ultralytics"}.items():
    os.environ.setdefault(key, str(ROOT / "work" / directory))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2
import numpy as np
import psutil
import torch
from trackers import BoTSORTTracker
from benchmark import load_detector, sync

TEAM_COLORS = {"blue":(255,155,65), "white":(140,238,195), "unknown":(175,175,175)}


def confirmed_tracks(detections):
    """Keep identified tracks; unconfirmed -1 rows do not share an identity."""
    if detections.tracker_id is None:
        return detections[:0]
    detections=detections[detections.tracker_id>=0]
    if len(set(detections.tracker_id.tolist()))!=len(detections):
        raise RuntimeError("Tracker returned duplicate confirmed IDs in one frame")
    return detections


def torso_crop(frame, box):
    x1,y1,x2,y2 = box
    w,h = x2-x1,y2-y1
    xa,ya = max(0,int(x1+w*.12)),max(0,int(y1+h*.14))
    xb,yb = min(frame.shape[1],int(x2-w*.12)),min(frame.shape[0],int(y1+h*.58))
    return frame[ya:yb,xa:xb]


def uniform_vote(crop):
    """Explicit blue/white demo prior; abstain if evidence is weak or mixed."""
    if crop.size < 300:
        return "unknown",0.0
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    blue = np.mean((hsv[:,:,0]>98)&(hsv[:,:,0]<137)&(hsv[:,:,1]>95)&(hsv[:,:,2]>45))
    white = np.mean((hsv[:,:,1]<65)&(hsv[:,:,2]>155))
    if blue>.16 and blue>white*1.15:
        return "blue",float(blue)
    if white>.26 and white>blue*1.4:
        return "white",float(white)
    return "unknown",0.0


def cut_score(previous, frame):
    gray = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(96,54))
    score = 0 if previous is None else float(np.mean(np.abs(gray.astype(float)-previous.astype(float)))/255)
    return gray,score


def close_to_ball(players, balls):
    """Conservative 2D proximity cue. This is not confirmed possession."""
    if len(balls)!=1 or not players:
        return None
    ball=balls[0]; bx=(ball[0]+ball[2])/2; by=(ball[1]+ball[3])/2
    scores=[]
    for p in players:
        x1,y1,x2,y2=p["box"]; w=x2-x1;h=y2-y1
        if x1-w*.20<=bx<=x2+w*.20 and y1+h*.2<=by<=y2+h*.1:
            scores.append((abs(bx-(x1+x2)/2)/max(w,1),p["track"]))
    scores.sort()
    if scores and (len(scores)==1 or scores[1][0]-scores[0][0]>.35):
        return scores[0][1]
    return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos",nargs="*",type=Path)
    ap.add_argument("--assets",type=Path,default=ROOT/"work/assets")
    ap.add_argument("--out",type=Path,default=ROOT/"outputs/demo")
    ap.add_argument("--scratch",type=Path,default=ROOT/"work/demo")
    ap.add_argument("--precision",choices=["float32","float16"],default="float32")
    ap.add_argument("--batch",type=int,default=1)
    ap.add_argument("--limit-seconds",type=float,default=0)
    ap.add_argument("--court",action="store_true")
    args=ap.parse_args()
    if args.batch<1:ap.error("--batch must be at least 1")
    if args.limit_seconds<0:ap.error("--limit-seconds cannot be negative")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal is unavailable. Run from a normal macOS terminal with GPU access.")
    def clock_order(path):
        match=re.search(r"-q(\d+)-(\d+)\.(\d+)-",path.name)
        return (int(match[1]),-int(match[2])*60-int(match[3])) if match else (99,path.name)
    videos=args.videos or sorted(args.assets.glob("*.mp4"),key=clock_order)
    if not videos: raise RuntimeError("No input videos found")
    args.out.mkdir(parents=True,exist_ok=True)
    scratch=args.scratch
    crops_dir=scratch/"crops";crops_dir.mkdir(parents=True,exist_ok=True)
    start_all=time.perf_counter()
    model=load_detector(args.assets/"ebard-rfdetr-nano-basketball.pth",precision=args.precision)
    court=None
    if args.court:
        from court import CourtMapper
        court=CourtMapper(args.assets/"yolo11n-basketball-court-keypoints.pt",device="mps")
    cap=cv2.VideoCapture(str(videos[0]));ok,first=cap.read();cap.release()
    if not ok:raise RuntimeError("Could not read first input frame")
    for _ in range(3):model.predict([cv2.cvtColor(first,cv2.COLOR_BGR2RGB)]*args.batch,threshold=.15,include_source_image=False)
    sync("mps")
    warmup=time.perf_counter()-start_all
    durations=defaultdict(float);track_stats={};records=[];clip_info=[]
    scene=0;total_frames=0;seconds_offset=0.0
    peak_rss=0;peak_mps=0;process=psutil.Process()
    processing_start=time.perf_counter()
    for clip_index,source in enumerate(videos):
        cap=cv2.VideoCapture(str(source));fps=float(cap.get(cv2.CAP_PROP_FPS));width=int(cap.get(3));height=int(cap.get(4));n=int(cap.get(7))
        if fps<=0: raise RuntimeError(f"Invalid frame rate: {source}")
        if args.limit_seconds:n=min(n,int(args.limit_seconds*fps))
        scene+=1;previous=None;last_cut=-1000
        def new_tracker():
            return BoTSORTTracker(frame_rate=fps,track_activation_threshold=.5,high_conf_det_threshold=.4,
                                  minimum_consecutive_frames=2,lost_track_buffer=15,cmc_downscale=4)
        tracker=new_tracker(); frame_index=0
        print(f"Analyzing {source.name}: {n} frames at {fps:g} fps",flush=True)
        while frame_index<n:
            frames=[];t=time.perf_counter()
            for _ in range(min(args.batch,n-frame_index)):
                ok,frame=cap.read()
                if not ok:break
                frames.append(frame)
            durations["decode"]+=time.perf_counter()-t
            if not frames:break
            t=time.perf_counter()
            preds=model.predict([cv2.cvtColor(f,cv2.COLOR_BGR2RGB) for f in frames],threshold=.15,include_source_image=False)
            sync("mps");durations["detector"]+=time.perf_counter()-t
            if not isinstance(preds,list):preds=[preds]
            for frame,dets in zip(frames,preds):
                t=time.perf_counter();timestamp=frame_index/fps
                previous,score=cut_score(previous,frame)
                if score>.27 and frame_index-last_cut>fps*.5:
                    tracker=new_tracker();scene+=1;last_cut=frame_index
                players=dets[dets.class_id==2].with_nms(threshold=.5)
                tracked=tracker.update(players,frame=frame,timestamp=timestamp)
                # BoT-SORT returns unconfirmed boxes with sentinel ID -1. Several
                # such boxes can coexist; they must never share an identity.
                tracked=confirmed_tracks(tracked)
                durations["tracking"]+=time.perf_counter()-t
                t=time.perf_counter()
                fit=None;xy=None
                if court:
                    fit=court.update(frame,timestamp,scene)
                    anchors=tracked.get_anchors_coordinates(__import__('supervision').Position.BOTTOM_CENTER)
                    xy=court.project(anchors,timestamp)
                durations["court"]+=time.perf_counter()-t
                t=time.perf_counter();items=[]
                for i,(box,tid,confidence) in enumerate(zip(tracked.xyxy,tracked.tracker_id,tracked.confidence)):
                    key=f"s{scene:02d}-t{int(tid):03d}"
                    stats=track_stats.setdefault(key,{"scene":scene,"first":seconds_offset+timestamp,"last":seconds_offset+timestamp,
                                                     "frames":0,"team_votes":{},"crops":[],"last_crop":-99})
                    stats["frames"]+=1;stats["last"]=seconds_offset+timestamp
                    crop=torso_crop(frame,box);team,weight=uniform_vote(crop)
                    stats["team_votes"][team]=stats["team_votes"].get(team,0)+weight
                    if timestamp-stats["last_crop"]>=.33 and crop.size>300:
                        quality=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
                        path=crops_dir/f"{key}-f{frame_index:05d}.jpg"
                        cv2.imwrite(str(path),crop,[cv2.IMWRITE_JPEG_QUALITY,95])
                        stats["crops"].append({"path":str(path),"quality":quality,"frame":frame_index,"time":timestamp})
                        stats["last_crop"]=timestamp
                    item={"track":key,"box":[round(float(v),2) for v in box],"confidence":round(float(confidence),4)}
                    if xy is not None and len(xy)>i and np.isfinite(xy[i]).all():item["court_xy"]=[round(float(v),5) for v in xy[i]]
                    items.append(item)
                balls=dets[(dets.class_id==0)&(dets.confidence>.35)].with_nms(threshold=.5)
                ball_boxes=[[round(float(v),2) for v in box] for box in balls.xyxy]
                row={"clip":clip_index,"frame":frame_index,"time":round(seconds_offset+timestamp,4),"scene":scene,
                     "players":items,"balls":ball_boxes,"near_ball":close_to_ball(items,ball_boxes)}
                if fit is not None:
                    row["court_fit"]=fit.as_dict()
                records.append(row);durations["metadata_and_crops"]+=time.perf_counter()-t
                frame_index+=1;total_frames+=1
                peak_rss=max(peak_rss,process.memory_info().rss);peak_mps=max(peak_mps,torch.mps.driver_allocated_memory())
                if total_frames%300==0:print(f"  {total_frames} frames; detector {total_frames/durations['detector']:.1f} fps; elapsed {time.perf_counter()-processing_start:.1f}s",flush=True)
        cap.release()
        clip_info.append({"path":str(source.resolve()),"frames":frame_index,"fps":fps,"width":width,"height":height,"duration":frame_index/fps})
        seconds_offset+=frame_index/fps
    analysis_wall=time.perf_counter()-processing_start
    for key,stats in track_stats.items():
        votes={k:v for k,v in stats["team_votes"].items() if k!="unknown"}
        ranked=sorted(votes.items(),key=lambda item:item[1],reverse=True)
        total=sum(votes.values())
        stats["team"]=ranked[0][0] if ranked and ranked[0][1]/max(total,.001)>.75 else "unknown"
        stats["team_vote_share"]=ranked[0][1]/total if ranked else 0
        stats["crops"]=sorted(stats["crops"],key=lambda item:item["quality"],reverse=True)[:8]
    manifest={"device":"Apple Silicon / Metal (MPS)","pipeline":"RF-DETR Nano E-BARD + BoT-SORT + blue/white uniform votes",
              "model_resolution":704,"precision":args.precision,"batch":args.batch,"clips":clip_info,"frames":total_frames,
              "video_seconds":seconds_offset,"scenes":scene,"tracks":track_stats,
              "timing":{"load_and_warmup_s":warmup,"analysis_wall_s":analysis_wall,"analysis_fps":total_frames/analysis_wall,
                        "detector_fps":total_frames/durations["detector"],"stages_s":dict(durations)},
              "memory":{"peak_process_rss_gb":peak_rss/1e9,"peak_mps_driver_gb":peak_mps/1e9,
                        "note":"RSS and Metal allocations may overlap in unified memory; do not add them."},
              "limitations":["Track IDs reset at shot boundaries; no appearance-based cross-play ReID.",
                              "Blue/white uniform rule is specific to these sample games.",
                              "Ball proximity is a 2D heuristic, not verified possession.",
                              "No ground-truth identity or detection accuracy benchmark was run."]}
    (args.out/"analysis.json").write_text(json.dumps(manifest,indent=2))
    with (args.out/"frames.jsonl").open("w") as handle:
        for record in records:handle.write(json.dumps(record,allow_nan=False)+"\n")
    with (args.out/"trajectories.csv").open("w",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["time_s","clip","scene","frame","track_id","team_color","confidence","foot_x_px","foot_y_px","court_x_normalized","court_y_normalized"])
        for row in records:
            for p in row["players"]:
                box=p["box"];courtxy=p.get("court_xy",[None,None])
                writer.writerow([row["time"],row["clip"],row["scene"],row["frame"],p["track"],track_stats[p["track"]]["team"],p["confidence"],(box[0]+box[2])/2,box[3],*courtxy])
    print(json.dumps({"frames":total_frames,"seconds":seconds_offset,"tracks":len(track_stats),"timing":manifest["timing"],"memory":manifest["memory"]},indent=2),flush=True)


if __name__=="__main__":main()
