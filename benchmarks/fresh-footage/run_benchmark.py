#!/usr/bin/env python3
"""Repeated local basketball throughput benchmark; standard-library orchestrator.

Each core run uses a new process and fresh output/crop directories. A single
full pipeline consists of core run 1 plus one Qwen process plus one renderer.
Three core repeats are never added to the reported single-pipeline total.
This script does not download assets, call APIs, or publish anything.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time


APP_FILES = ("analyze.py", "benchmark.py", "court.py", "vlm_ocr.py", "render.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(command, *, env=None):
    return subprocess.check_output([str(part) for part in command], text=True, env=env).strip()


def describe(values):
    values = list(values)
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    # Linear interpolation, equivalent to NumPy's default percentile method.
    index = (len(ordered) - 1) * .95
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return {
        "n": len(values), "mean": statistics.mean(values),
        "min": min(values), "max": max(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "p95": ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower),
    }


def hardware():
    result = {"architecture": platform.machine(), "macos": platform.mac_ver()[0]}
    for label, key in (("chip", "machdep.cpu.brand_string"),
                       ("cpu_cores", "hw.physicalcpu"), ("memory_bytes", "hw.memsize")):
        try:
            value = capture(["sysctl", "-n", key])
            result[label] = int(value) if value.isdigit() else value
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        displays = json.loads(capture(["system_profiler", "SPDisplaysDataType", "-json"]))
        result["gpus"] = [{key: item[key] for key in ("sppci_model", "sppci_cores", "spdisplays_metal") if key in item}
                          for item in displays.get("SPDisplaysDataType", [])]
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        pass
    return result


def probe(path):
    data = json.loads(capture([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
        "-of", "json", path,
    ]))
    return {"video": data["streams"][0], "container": data.get("format", {})}


def timed_process(command, log, env):
    print(f"Starting {log.stem}", flush=True)
    start = time.perf_counter()
    with log.open("w") as handle:
        process = subprocess.run([str(part) for part in command], stdout=handle,
                                 stderr=subprocess.STDOUT, env=env, check=False)
    elapsed = time.perf_counter() - start
    print(f"Finished {log.stem}: {elapsed:.3f}s (exit {process.returncode})", flush=True)
    if process.returncode:
        raise RuntimeError(f"{log.stem} failed with exit {process.returncode}; inspect {log}")
    return elapsed


def core_summary(manifest, process_wall, output_dir):
    records = [json.loads(line) for line in (output_dir / "frames.jsonl").read_text().splitlines()]
    frames = manifest["frames"]
    if len(records) != frames:
        raise RuntimeError("Metadata row count differs from source frame count")
    observations = sum(len(row["players"]) for row in records)
    valid = sum(bool(row.get("court_fit", {}).get("valid")) for row in records)
    positions = sum("court_xy" in player for row in records for player in row["players"])
    timing = manifest["timing"]
    stages = timing["stages_s"]
    return {
        "frames": frames, "decoded_video_seconds": manifest["video_seconds"],
        "source_fps": [clip["fps"] for clip in manifest["clips"]],
        "process_wall_s": process_wall,
        "process_wall_ms_per_source_frame": process_wall * 1000 / frames,
        "load_and_warmup_s": timing["load_and_warmup_s"],
        "analysis_wall_s": timing["analysis_wall_s"],
        "analysis_ms_per_source_frame": timing["analysis_wall_s"] * 1000 / frames,
        "analysis_fps": frames / timing["analysis_wall_s"],
        "stages_s": stages,
        "stages_ms_per_source_frame": {key: value * 1000 / frames for key, value in stages.items()},
        "unassigned_analysis_loop_overhead_s": timing["analysis_wall_s"] - sum(stages.values()),
        "process_overhead_outside_analysis_and_warmup_s": process_wall - timing["analysis_wall_s"] - timing["load_and_warmup_s"],
        "scenes": manifest["scenes"], "track_fragments": len(manifest["tracks"]),
        "player_observations": observations,
        "valid_court_frames": valid, "valid_court_fraction": valid / frames,
        "in_court_player_positions": positions,
        "in_court_player_position_fraction": positions / observations if observations else None,
        "memory": manifest["memory"],
    }


def jersey_summary(data, process_wall, frames):
    tracks = list(data["tracks"].values())
    checks = [check for track in tracks for check in track.get("single_crop_checks", [])]
    sheet_seconds = [track["inference_seconds"] - sum(check["inference_seconds"] for check in track.get("single_crop_checks", []))
                     for track in tracks]
    check_seconds = [check["inference_seconds"] for check in checks]
    result = {
        "process_wall_s": process_wall,
        "process_wall_ms_amortized_per_source_frame": process_wall * 1000 / frames,
        "reported_internal_timing": data.get("timing"),
        "model_load_s": data["load_seconds"], "peak_memory_gb": data.get("peak_memory_gb"),
        "track_fragments_evaluated": len(tracks),
        "candidate_track_fragments": sum(track["number"] is not None for track in tracks),
        "abstained_track_fragments": sum(track["number"] is None for track in tracks),
        "abstention_reasons": dict(Counter(track.get("reason") for track in tracks if track["number"] is None)),
        "generation_calls": len(tracks) + len(checks),
        "contact_sheet_generation_calls": len(tracks),
        "single_crop_generation_calls": len(checks),
        "input_torso_panels_across_contact_sheets": sum(len(track["crops"]) for track in tracks),
        "contact_sheet_generation_seconds": describe(sheet_seconds),
        "single_crop_generation_seconds": describe(check_seconds),
        "all_generation_call_seconds": describe(sheet_seconds + check_seconds),
        "track_total_inference_seconds": describe(track["inference_seconds"] for track in tracks),
        "sum_generation_seconds": sum(sheet_seconds) + sum(check_seconds),
        "contact_sheet_generation_tokens": sum(track["generation_tokens"] for track in tracks),
        "contact_sheet_prompt_tokens": sum(track["prompt_tokens"] for track in tracks),
        "single_crop_token_counts": "Not recorded by current application; excluded from token totals.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--ocr-python", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--qwen-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Must not already exist")
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--label", default="Independent basketball footage")
    parser.add_argument("--source-url")
    parser.add_argument("--limit-seconds", type=float, default=60.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--core-only", action="store_true", help="Skip Qwen and rendering")
    args = parser.parse_args()
    if args.repeats < 1 or args.limit_seconds < 0:
        parser.error("repeats must be positive and limit-seconds nonnegative")
    # Do not resolve interpreter symlinks: the venv directory selects dependencies.
    for key in ("video", "app", "assets", "qwen_model", "out", "cache_root"):
        setattr(args, key, getattr(args, key).resolve())
    args.python = args.python.absolute()
    args.ocr_python = args.ocr_python.absolute()
    for path in [args.video, args.python, args.ocr_python, args.assets / "ebard-rfdetr-nano-basketball.pth",
                 args.assets / "yolo11n-basketball-court-keypoints.pt", args.qwen_model / "config.json"]:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / "app-snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for name in APP_FILES:
        shutil.copy2(args.app / name, snapshot / name)
        source_hashes[name] = sha256(snapshot / name)
    env = os.environ.copy()
    for key, directory in {"MPLCONFIGDIR": "matplotlib", "HF_HOME": "huggingface", "TORCH_HOME": "torch", "RF_HOME": "rfdetr-cache", "YOLO_CONFIG_DIR": "ultralytics"}.items():
        env[key] = str(args.cache_root / directory)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "YOLO_AUTOINSTALL": "false",
                "YOLO_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    try:
        git_revision = capture(["git", "-C", args.app, "rev-parse", "HEAD"])
    except subprocess.CalledProcessError:
        git_revision = None
    report = {
        "schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware(),
        "source": {"label": args.label, "url": args.source_url, "sha256": sha256(args.video), "probe": probe(args.video)},
        "application": {"git_revision": git_revision, "source_sha256": source_hashes},
        "config": {"batch": 4, "precision": "float16", "detector_resolution": 704,
                   "court_image_size": 960, "limit_seconds": args.limit_seconds, "repeats": args.repeats,
                   "device": "MPS", "jersey_model": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
                   "jersey_min_track_frames": 30, "jersey_limit_tracks": 0},
        "method": [
            "Each core run is a new Python process using a snapshot of the same application code, existing local weights, and a unique crop/output directory.",
            "RF-DETR uses three warmup batches before the timed analysis loop. Model files and OS caches can be warm; this is not a cold disk benchmark.",
            "The existing core analysis timer excludes imports, model loading/warmup, and final metadata writes. External process wall time includes them.",
            "Detector stage includes color conversion, preprocessing, predict, postprocessing, and explicit torch.mps.synchronize(). Court outputs are transferred to CPU before geometric fitting.",
            "Per-source-frame stage costs and full pipeline costs are throughput averages, not request latency. Batch size is four.",
            "Qwen processes selected torso contact sheets, plus up to two verification crops per candidate track; its per-call seconds are not full-frame Astra latency.",
            "End-to-end process total is core run 1 + one complete Qwen run + render/encode process; it excludes downloads/setup and separate verification.",
            "Core repeats are sequential. No model warmup or repeated core runs are added to the single-pipeline total beyond core run 1.",
            "Render resamples to 30fps; throughput denominators use actual analyzed source frame count, not rendered frame count.",
            "Detection, identity, jersey and calibration accuracy have no ground-truth labels here. Counts and coverage are not accuracy scores.",
            "Uniform classification is the existing blue/white demo heuristic, unsuitable for arbitrary team colors.",
        ],
        "core_runs": [],
    }
    dependency_script = "import importlib.metadata as m,json; names={names!r}; print(json.dumps({{n:m.version(n) for n in names}}))"
    report["dependencies"] = {
        "core": json.loads(capture([args.python, "-c", dependency_script.format(names=["torch", "torchvision", "rfdetr", "trackers", "ultralytics", "supervision", "opencv-python", "numpy"])], env=env)),
        "ocr": json.loads(capture([args.ocr_python, "-c", dependency_script.format(names=["mlx", "mlx-vlm", "transformers", "Pillow"])], env=env)),
    }
    def checkpoint():
        (args.out / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    checkpoint()
    workflow_start = time.perf_counter()
    for index in range(1, args.repeats + 1):
        directory = args.out / f"core-{index}"
        process_wall = timed_process([
            args.python, snapshot / "analyze.py", args.video, "--assets", args.assets,
            "--out", directory, "--scratch", directory / "scratch", "--batch", "4",
            "--precision", "float16", "--court", "--limit-seconds", str(args.limit_seconds),
        ], args.out / f"core-{index}.log", env)
        data = json.loads((directory / "analysis.json").read_text())
        summary = core_summary(data, process_wall, directory)
        summary["run"] = index
        report["core_runs"].append(summary)
        checkpoint()
    runs = report["core_runs"]
    frames = runs[0]["frames"]
    if frames <= 0 or any(run["frames"] != frames for run in runs):
        raise RuntimeError("Core runs did not analyze an equal positive source frame count")
    report["core_repeat_statistics"] = {
        "process_wall_s": describe(run["process_wall_s"] for run in runs),
        "analysis_wall_s": describe(run["analysis_wall_s"] for run in runs),
        "analysis_ms_per_source_frame": describe(run["analysis_ms_per_source_frame"] for run in runs),
        "analysis_fps": describe(run["analysis_fps"] for run in runs),
        "stages_s": {stage: describe(run["stages_s"][stage] for run in runs) for stage in runs[0]["stages_s"]},
        "stages_ms_per_source_frame": {stage: describe(run["stages_ms_per_source_frame"][stage] for run in runs) for stage in runs[0]["stages_s"]},
    }
    checkpoint()
    if not args.core_only:
        first_core = args.out / "core-1"
        jerseys_path = args.out / "jerseys.json"
        qwen_wall = timed_process([
            args.ocr_python, snapshot / "vlm_ocr.py", "--analysis", first_core / "analysis.json",
            "--model", args.qwen_model, "--output", jerseys_path,
            "--sheets-dir", args.out / "jersey-evidence",
        ], args.out / "qwen.log", env)
        report["jerseys"] = jersey_summary(json.loads(jerseys_path.read_text()), qwen_wall, frames)
        checkpoint()
        video_output = args.out / "annotated.mp4"
        render_wall = timed_process([
            args.python, snapshot / "render.py", "--analysis", first_core / "analysis.json",
            "--jerseys", jerseys_path, "--output", video_output, "--title", args.label,
        ], args.out / "render.log", env)
        rendered = json.loads(video_output.with_suffix(".render.json").read_text())
        report["render"] = {
            "process_wall_s": render_wall, "internal_render_and_encode_s": rendered["render_and_encode_s"],
            "frames": rendered["frames"], "fps": rendered["fps"], "duration_s": rendered["duration"],
            "process_wall_ms_amortized_per_source_frame": render_wall * 1000 / frames,
        }
        total = runs[0]["process_wall_s"] + qwen_wall + render_wall
        qwen_internal = report["jerseys"]["reported_internal_timing"]
        internal_total = (runs[0]["analysis_wall_s"] + qwen_internal["wall_seconds"] + rendered["render_and_encode_s"]
                          if qwen_internal is not None else None)
        report["single_full_pipeline"] = {
            "core_run_used": 1, "source_frames": frames, "decoded_video_seconds": runs[0]["decoded_video_seconds"],
            "full_process_wall_s": total,
            "full_process_wall_ms_amortized_per_source_frame": total * 1000 / frames,
            "source_frames_per_processing_second": frames / total,
            "processing_seconds_per_video_second": total / runs[0]["decoded_video_seconds"],
            "sum_existing_internal_timers_s": internal_total,
            "sum_existing_internal_timers_ms_amortized_per_source_frame": internal_total * 1000 / frames if internal_total is not None else None,
            "explanation": "The internal-timer sum matches the style of the earlier approximate130ms figure. Use full_process_wall for complete process startup-to-exit cost; both are amortized across source video frames, not per-VLM-frame latency.",
        }
        checkpoint()
        verify_wall = timed_process([
            "ffmpeg", "-hide_banner", "-v", "error", "-xerror", "-i", video_output,
            "-map", "0:v:0", "-f", "null", "-",
        ], args.out / "verify-render.log", env)
        output_probe = probe(video_output)
        if int(output_probe["video"]["nb_read_frames"]) != rendered["frames"]:
            raise RuntimeError("Rendered video decoded frame count differs from renderer metadata")
        report["verification"] = {
            "full_render_decode_exit_code": 0, "full_render_decode_s": verify_wall,
            "render_probe": output_probe, "render_sha256": sha256(video_output),
            "render_frames_match_metadata": True,
            "all_core_metadata_rows_match_source_frame_count": True,
        }
    report["workflow_wall_s_including_repeats_and_verification"] = time.perf_counter() - workflow_start
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    print(json.dumps({"core_statistics": report["core_repeat_statistics"],
                      "single_full_pipeline": report.get("single_full_pipeline"),
                      "jerseys": report.get("jerseys")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
