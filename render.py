#!/usr/bin/env python3
"""Render saved analysis as an annotated H.264 video, without rerunning models."""
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from court import render_court

ROOT=Path(__file__).resolve().parent
COLORS={"blue":(255,155,65),"white":(140,238,195),"unknown":(175,175,175)}
BG=(21,16,12); TEXT=(241,240,231); MUTED=(159,151,141); GREEN=(161,245,122)


def text(image,label,x,y,size=.6,color=TEXT,thickness=1):
    cv2.putText(image,str(label),(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,thickness,cv2.LINE_AA)


def clock(seconds):
    return f"{int(seconds)//60:02d}:{seconds%60:05.2f}"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis",type=Path,default=ROOT/"outputs/demo/analysis.json")
    ap.add_argument("--jerseys",type=Path)
    ap.add_argument("--output",type=Path,default=ROOT/"outputs/basketball-demo.mp4")
    ap.add_argument("--title",default="BASKETBALL ANALYSIS")
    args=ap.parse_args()
    analysis=json.loads(args.analysis.read_text());tracks=analysis["tracks"]
    records=[json.loads(line) for line in args.analysis.with_name("frames.jsonl").read_text().splitlines()]
    jerseys={}
    if args.jerseys and args.jerseys.exists():
        jersey_data=json.loads(args.jerseys.read_text())
        jerseys=jersey_data.get("tracks",jersey_data.get("results",jersey_data))
    def jersey_for(key):
        item=jerseys.get(key,{})
        if not isinstance(item,dict):return None
        return item.get("candidate_number") or item.get("number") or item.get("consensus",{}).get("number")
    width,height,fps=1600,900,30
    args.output.parent.mkdir(parents=True,exist_ok=True)
    command=["ffmpeg","-y","-hide_banner","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{width}x{height}","-r",str(fps),"-i","-","-an","-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(args.output)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    start=time.perf_counter();trails=defaultdict(lambda:deque(maxlen=14));court_trails=defaultdict(lambda:deque(maxlen=32))
    last_seen={}
    previous_scene=None;current_clip=-1;cap=None;rendered=0;source_index=-1;frame=None
    sample_targets=set(np.linspace(0,len(records)-1,6,dtype=int).tolist());samples=[]
    by_clip=defaultdict(dict)
    for record in records:by_clip[record["clip"]][record["frame"]]=record
    try:
        global_index=0
        for clip_index,info in enumerate(analysis["clips"]):
            cap=cv2.VideoCapture(info["path"])
            # Resample each clip to 30fps while retaining exact source timestamps.
            wanted_frames=int(round(info["duration"]*fps))
            last_read=-1
            for out_index in range(wanted_frames):
                index=min(info["frames"]-1,int(out_index*info["fps"]/fps))
                while last_read<index:
                    ok,frame=cap.read();last_read+=1
                    if not ok:raise RuntimeError(f"Source ended before frame {index}: {info['path']}")
                row=by_clip[clip_index][index]
                if previous_scene!=row["scene"]:
                    trails.clear();court_trails.clear();last_seen.clear();previous_scene=row["scene"]
                canvas=np.full((height,width,3),BG,dtype=np.uint8)
                cv2.line(canvas,(24,66),(1576,66),(53,46,38),1)
                text(canvas,"COURTSIDE",24,42,.86,TEXT,2)
                text(canvas,args.title[:55],243,40,.5,MUTED)
                cv2.rectangle(canvas,(1420,20),(1576,48),(51,59,34),-1)
                text(canvas,"100% LOCAL",1439,40,.47,GREEN,1)
                view=cv2.resize(frame,(1200,675));sx=1200/info["width"];sy=675/info["height"]
                points=[];labels=[];colors=[]
                fit=row.get("court_fit",{});valid=fit.get("valid",False)
                if not valid:court_trails.clear()
                for player in row["players"]:
                    key=player["track"];team=tracks[key]["team"];color=COLORS[team];short=key.split('-')[-1].upper()
                    x1,y1,x2,y2=np.round(np.array(player["box"])*[sx,sy,sx,sy]).astype(int)
                    x1=max(0,x1);x2=min(1199,x2);y1=max(0,y1);y2=min(674,y2)
                    foot=(int((x1+x2)/2),int(y2))
                    if key in last_seen and (index-last_seen[key]>1 or (trails[key] and np.linalg.norm(np.array(foot)-trails[key][-1])>max(55,(y2-y1)*.65))):
                        trails[key].clear();court_trails.pop(short,None)
                    last_seen[key]=index;trails[key].append(foot)
                    if len(trails[key])>1:cv2.polylines(view,[np.array(trails[key],np.int32)],False,color,2,cv2.LINE_AA)
                    cv2.ellipse(view,foot,(max(10,(x2-x1)//2),5),0,0,360,color,2,cv2.LINE_AA)
                    corner=min(13,max(4,(x2-x1)//4))
                    for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                        cv2.line(view,(px,py),(px+corner*dx,py),color,2,cv2.LINE_AA)
                        cv2.line(view,(px,py),(px,py+corner*dy),color,2,cv2.LINE_AA)
                    short=key.split('-')[-1].upper();j=jersey_for(key)
                    label=f"{short}"+(f"  #{j}?" if j is not None else "")
                    (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.39,1)
                    lx=max(0,min(1199-tw-8,x1));ly=max(th+7,y1-4)
                    cv2.rectangle(view,(lx,ly-th-6),(lx+tw+8,ly+3),BG,-1)
                    text(view,label,lx+4,ly,.39,color)
                    if player.get("court_xy"):
                        points.append(player["court_xy"]);labels.append(short);colors.append(color)
                        if court_trails[short] and np.linalg.norm(np.array(player["court_xy"])-court_trails[short][-1])>.06:
                            court_trails[short].clear()
                        court_trails[short].append(player["court_xy"])
                    else:court_trails.pop(short,None)
                for b in row["balls"]:
                    x1,y1,x2,y2=np.array(b)*[sx,sy,sx,sy];cx=int((x1+x2)/2);cy=int((y1+y2)/2)
                    cv2.circle(view,(cx,cy),max(7,int((x2-x1)/2)+4),(68,229,255),2,cv2.LINE_AA)
                    cv2.line(view,(cx-14,cy),(cx-9,cy),(68,229,255),1)
                    cv2.line(view,(cx+9,cy),(cx+14,cy),(68,229,255),1)
                canvas[88:763,24:1224]=view
                text(canvas,f"PLAY {clip_index+1:02d} / {len(analysis['clips']):02d}",1246,108,.56,GREEN,1)
                text(canvas,"COURT POSITION",1246,144,.48,MUTED)
                court=render_court(np.asarray(points).reshape(-1,2),labels,colors,width=330,
                                   trails=dict(court_trails),status="Estimated floor positions" if valid else "Calibration unavailable")
                canvas[160:160+court.shape[0],1246:1576]=court
                cy=175+court.shape[0]
                if valid:
                    text(canvas,f"{fit.get('inliers',0)} court landmarks",1246,cy,.40,MUTED)
                else:text(canvas,"Map withheld when fit is unreliable",1246,cy,.35,MUTED)
                cy+=40;text(canvas,"PLAYERS IN VIEW",1246,cy,.48,MUTED)
                counts=CounterCompat(tracks[p["track"]]["team"] for p in row["players"])
                text(canvas,str(counts["white"]),1246,cy+42,1.0,COLORS["white"],2)
                text(canvas,"WHITE",1295,cy+40,.42,COLORS["white"])
                text(canvas,str(counts["blue"]),1400,cy+42,1.0,COLORS["blue"],2)
                text(canvas,"BLUE",1450,cy+40,.42,COLORS["blue"])
                cy+=85;text(canvas,"ACTIVE TRACKS",1246,cy,.48,MUTED)
                for i,p in enumerate(sorted(row["players"],key=lambda p:p['track'])[:11]):
                    key=p["track"];color=COLORS[tracks[key]["team"]];j=jersey_for(key)
                    text(canvas,key.split('-')[-1].upper(),1246,cy+25+i*23,.43,color)
                    text(canvas,f"Jersey {j}?" if j is not None else "Jersey unread",1343,cy+25+i*23,.39,MUTED)
                text(canvas,"Track IDs restart each play.",1246,809,.36,MUTED)
                text(canvas,"? = experimental jersey reading",1246,832,.36,MUTED)
                text(canvas,clock(row["time"]),24,811,.82,TEXT,2)
                text(canvas,f"/ {clock(analysis['video_seconds'])}",175,810,.50,MUTED)
                text(canvas,"APPLE SILICON",400,797,.43,MUTED)
                text(canvas,"Local Metal inference",400,825,.59,TEXT,1)
                stage_label="DETECTION + TRACKING"+(" + COURT" if "court_fit" in row else "")
                text(canvas,stage_label,805,797,.41,MUTED)
                text(canvas,f"{analysis['timing']['analysis_fps']:.1f} frames/sec measured",805,825,.61,GREEN,1)
                cv2.line(canvas,(24,850),(1576,850),(53,46,38),1)
                text(canvas,"Local video analysis  |  Local inference  |  No API calls",24,879,.40,MUTED)
                text(canvas,"Team colors and trajectories are estimates",1100,879,.39,MUTED)
                encoder.stdin.write(canvas.tobytes());rendered+=1
                if global_index in sample_targets:
                    sample_path=args.output.parent/f"preview-{len(samples)+1}.jpg"
                    cv2.imwrite(str(sample_path),canvas,[cv2.IMWRITE_JPEG_QUALITY,92]);samples.append(str(sample_path))
                global_index+=1
            cap.release();cap=None
            print(f"Rendered play {clip_index+1}: {rendered} output frames",flush=True)
    finally:
        if cap is not None:cap.release()
        if encoder.stdin:encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError("FFmpeg video encoding failed")
    result={"output":str(args.output),"frames":rendered,"fps":fps,"duration":rendered/fps,"render_and_encode_s":time.perf_counter()-start,"samples":samples}
    args.output.with_suffix('.render.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


class CounterCompat(defaultdict):
    def __init__(self,items):
        super().__init__(int)
        for item in items:self[item]+=1


if __name__=="__main__":main()
