#!/usr/bin/env python3
"""Benchmark the downloaded basketball detector on real frames, with GPU synchronization."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
for key, directory in {"MPLCONFIGDIR": "matplotlib", "RF_HOME": "rfdetr-cache", "HF_HOME": "huggingface", "TORCH_HOME": "torch"}.items():
    os.environ.setdefault(key, str(ROOT / "work" / directory))

import cv2
import numpy as np
import torch
from rfdetr import RFDETRNano


def load_detector(weights, device="mps", resolution=704, precision="float32", compile=False, batch_size=1):
    # This checkpoint's only extra pickle global is the stdlib Namespace container.
    with torch.serialization.safe_globals([argparse.Namespace]):
        model = RFDETRNano(pretrain_weights=str(weights), device=device, resolution=resolution)
    # E-BARD source categories are zero-based: basketball, hoop, player, referee.
    # The legacy checkpoint omits names and counts all four classifier outputs;
    # infer labels ourselves instead of trusting the library's default COCO names.
    model.inference(compile=compile, dtype=getattr(torch, precision), batch_size=batch_size)
    return model


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=ROOT / "work/assets")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/benchmark.json")
    args = ap.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal GPU unavailable to this process; run from a normal macOS terminal.")
    frames = []
    for source in sorted(args.assets.glob("*.mp4")):
        cap = cv2.VideoCapture(str(source))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in np.linspace(0, n - 1, 4, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    frames = frames[:32]
    if len(frames) < 8:
        raise RuntimeError("Need at least eight real sample frames.")
    output = {"device": "mps", "torch": torch.__version__, "sample_frames": len(frames),
              "input_shape": list(frames[0].shape), "timing": "predict including preprocessing and postprocessing; synchronized Metal; excludes decode and warmup",
              "runs": []}
    baseline = None
    for precision, batch, resolution in [("float32",1,704),("float32",4,704),("float16",1,704),("float16",4,704),("float32",1,512)]:
        print(f"Benchmark {precision} batch={batch} resolution={resolution}", flush=True)
        row = {"precision": precision, "batch": batch, "resolution": resolution}
        try:
            start = time.perf_counter()
            model = load_detector(args.assets / "ebard-rfdetr-nano-basketball.pth", resolution=resolution, precision=precision)
            for _ in range(3):
                model.predict(frames[:batch], threshold=.2, include_source_image=False)
            sync("mps")
            row["load_and_warmup_s"] = time.perf_counter() - start
            durations, predictions = [], []
            for i in range(0, len(frames), batch):
                sample = frames[i:i+batch]
                start = time.perf_counter()
                ds = model.predict(sample, threshold=.2, include_source_image=False)
                sync("mps")
                durations.append(time.perf_counter() - start)
                predictions.extend(ds if isinstance(ds, list) else [ds])
            row.update(frames=len(predictions), seconds=sum(durations), fps=len(predictions)/sum(durations),
                       batch_latency_ms_median=float(np.median(durations)*1000),
                       batch_latency_ms_p95=float(np.percentile(durations,95)*1000),
                       mps_allocated_gb=torch.mps.current_allocated_memory()/1e9,
                       mps_driver_gb=torch.mps.driver_allocated_memory()/1e9,
                       detections=sum(len(d) for d in predictions), players=sum(int(np.sum(d.class_id==2)) for d in predictions))
            if baseline is None:
                baseline = predictions
            else:
                # Agreement check, not an accuracy metric: compare player boxes to fp32 baseline.
                from supervision import box_iou_batch
                matched, total = 0, 0
                for reference, current in zip(baseline, predictions):
                    ref = reference[reference.class_id==2]
                    cur = current[current.class_id==2]
                    total += len(ref)
                    if len(ref) and len(cur):
                        iou = box_iou_batch(ref.xyxy, cur.xyxy)
                        matched += int((iou.max(axis=1) > .8).sum())
                row["baseline_player_box_recall_iou_0_8"] = matched/max(total,1)
            print(json.dumps(row), flush=True)
            del model
            torch.mps.empty_cache()
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(row["error"], flush=True)
        output["runs"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
