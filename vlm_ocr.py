#!/usr/bin/env python3
"""Local jersey-number candidates from torso crops with MLX Qwen3-VL.

Run in work/ocr-venv; model weights live in work/ocr-research. No cloud inference.

  python vlm_ocr.py --analysis outputs/demo/analysis.json --output jersey.json
  python vlm_ocr.py --manifest tracks.json --output jersey.json

Manifest format: {"track_id": ["torso1.png", "torso2.png", ...]}.
Analysis format: {"tracks": {"track_id": {"frames": 100, "crops": [
    {"path": "torso.png", "quality": 12.3, "frame": 100, "time": 3.3}]}}}.

Only existing within-shot tracks should be supplied. Crop votes are correlated
model readings, not independent identity evidence or calibrated probabilities.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
DEFAULT_MODEL = WORK / "ocr-research" / "qwen3-vl-8b-4bit"
MODEL_ID = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
MODEL_REVISION = "defcdea7cc7a4b0858fea563cbbce171d328e457"
LABELS = "ABCDEF"
NUMERAL = re.compile(r"[0-9]{1,2}\Z")


def make_contact_sheet(paths: list[str], destination: Path) -> list[str]:
    """Letter labels avoid introducing unrelated digits into the OCR image."""
    if not 1 <= len(paths) <= 6:
        raise ValueError("Provide one to six torso crops per track")
    cols = min(3, len(paths))
    rows = math.ceil(len(paths) / cols)
    cell_w, cell_h = 256, 336
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "#e7e7e7")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=26)
    labels = list(LABELS[:len(paths)])
    for index, (label, path) in enumerate(zip(labels, paths)):
        with Image.open(path) as source:
            crop = ImageOps.exif_transpose(source).convert("RGB")
            crop = ImageOps.contain(crop, (cell_w - 20, cell_h - 58), Image.Resampling.LANCZOS)
        x, y = (index % cols) * cell_w, (index // cols) * cell_h
        draw.rectangle((x + 3, y + 3, x + cell_w - 4, y + cell_h - 4), fill="white", outline="#aaaaaa")
        draw.text((x + 12, y + 10), label, font=font, fill="black")
        sheet.paste(crop, (x + (cell_w - crop.width) // 2, y + 48 + (cell_h - 58 - crop.height) // 2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return labels


def parse_readings(text: str, labels: list[str]) -> tuple[dict[str, str | None], str | None]:
    answer = text.strip()
    if answer.startswith("```json\n") and answer.endswith("```"):
        answer = answer[8:-3].strip()
    elif answer.startswith("```\n") and answer.endswith("```"):
        answer = answer[4:-3].strip()
    try:
        parsed = json.loads(answer)
        if not isinstance(parsed, dict) or set(parsed) != set(labels):
            raise ValueError("Expected exactly the panel letters as JSON keys")
        readings = {}
        for label in labels:
            value = parsed[label]
            if value is None or value == "unknown":
                readings[label] = None
            elif isinstance(value, str) and NUMERAL.fullmatch(value):
                readings[label] = value
            else:
                raise ValueError("Each value must be a one/two digit string or unknown")
        return readings, None
    except (json.JSONDecodeError, ValueError) as error:
        return dict.fromkeys(labels), str(error)


def summarize_votes(readings: list[str | None], min_votes: int = 3, min_share: float = .8) -> dict[str, Any]:
    counts = Counter(number for number in readings if number is not None)
    count = sum(counts.values())
    number, support = counts.most_common(1)[0] if counts else (None, 0)
    share = support / count if count else 0.0
    coverage = support / len(readings) if readings else 0.0
    accepted = support >= min_votes and share >= min_share and coverage >= .5
    return {
        "number": number if accepted else None,
        "status": "candidate" if accepted else "abstain",
        "reason": None if accepted else "insufficient_consensus",
        "supporting_crops": support,
        "readable_crops": count,
        "total_crops": len(readings),
        "vote_share": share,
        "support_fraction_of_all_crops": coverage,
        "votes": dict(counts),
        "confidence": None,
        "evidence_note": "Correlated visual model readings; not calibrated confidence or verified player identity.",
    }


class JerseyVLM:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        # Keep model and framework caches in the workspace. A local directory is
        # mandatory here so inference cannot silently trigger a large download.
        self.model_path = Path(model_path).resolve()
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"Local Qwen3-VL model missing: {self.model_path}")
        os.environ.setdefault("HF_HOME", str(WORK / "ocr-research" / "hf-cache"))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        started = time.perf_counter()
        self.model, self.processor = load(str(self.model_path), trust_remote_code=False)
        self.generate = generate
        self.apply_chat_template = apply_chat_template
        self.load_seconds = time.perf_counter() - started

    def _check_single_crop(self, path: str) -> dict[str, Any]:
        # A separate context prevents a clear panel from filling an obscured
        # panel in the contact sheet. Repeated views still are not independent
        # statistical evidence, and two agreeing reads may both be wrong.
        prompt = (
            "Read ONLY the printed basketball jersey number visible in this torso crop. "
            "Use visible digit shapes only. Do not identify the player, use a roster, "
            "or infer any hidden digit. Return unknown if the number is incomplete, "
            "blurred, hidden, or ambiguous. Respond with ONLY one or two digits, "
            "preserving 00, or the single word unknown."
        )
        formatted = self.apply_chat_template(self.processor, self.model.config, prompt, num_images=1)
        started = time.perf_counter()
        generated = self.generate(
            self.model, self.processor, formatted, image=[path],
            max_tokens=12, temperature=0.0, verbose=False,
        )
        raw = generated.text.strip()
        number = raw if NUMERAL.fullmatch(raw) else None
        return {"path": path, "number": number, "raw_answer": generated.text,
                "inference_seconds": time.perf_counter() - started}

    def recognize_track(self, paths: list[str], sheet_path: str | Path) -> dict[str, Any]:
        unique = list(dict.fromkeys(str(Path(path).resolve()) for path in paths))[:6]
        labels = make_contact_sheet(unique, Path(sheet_path))
        shape = ", ".join(f'"{label}": "number or unknown"' for label in labels)
        prompt = (
            "Read the printed basketball jersey NUMBER in each letter-labeled image panel. "
            "Each panel is a torso crop. Read each panel independently using only visible digits. "
            "Do not infer a number from a player's face, name, team, roster, another panel, "
            "or basketball knowledge. Ignore names, clothing brands, and the panel letters. "
            "Only report a one- or two-digit jersey number if its digits are clearly visible. "
            "Preserve 00 as 00. If the number is hidden, incomplete, blurred, or ambiguous, "
            "return unknown. Return ONLY a JSON object with exactly these keys and string values: {"
            + shape + "}."
        )
        formatted = self.apply_chat_template(self.processor, self.model.config, prompt, num_images=1)
        started = time.perf_counter()
        generated = self.generate(
            self.model, self.processor, formatted, image=[str(Path(sheet_path).resolve())],
            max_tokens=160, temperature=0.0, verbose=False,
        )
        elapsed = time.perf_counter() - started
        readings, parse_error = parse_readings(generated.text, labels)
        result = summarize_votes(list(readings.values()))
        if parse_error:
            result.update(reason="invalid_model_response", parse_error=parse_error)
        checks = []
        if result["number"] is not None:
            candidate = result["number"]
            matching_paths = [path for label, path in zip(labels, unique) if readings[label] == candidate]
            checks = [self._check_single_crop(path) for path in matching_paths[:2]]
            elapsed += sum(check["inference_seconds"] for check in checks)
            if len(checks) < 2 or any(check["number"] != candidate for check in checks):
                result.update(number=None, status="abstain", reason="single_crop_check_disagreement",
                              contact_sheet_candidate=candidate)
        result.update(
            crops=[{"panel": label, "path": path, "number": readings[label]}
                   for label, path in zip(labels, unique)],
            raw_answer=generated.text, contact_sheet=str(Path(sheet_path).resolve()),
            inference_seconds=elapsed,
            prompt_tokens=generated.prompt_tokens, generation_tokens=generated.generation_tokens,
            prompt_tokens_per_second=generated.prompt_tps,
            generation_tokens_per_second=generated.generation_tps,
            peak_memory_gb=generated.peak_memory,
            single_crop_checks=checks,
        )
        return result


def _resolve_crop_path(value: str, manifest: Path) -> str:
    path = Path(value)
    if path.is_absolute() or path.is_file():
        return str(path.resolve())
    return str((manifest.parent / path).resolve())


def tracks_from_analysis(path: Path, min_frames: int = 30) -> dict[str, list[str]]:
    data = json.loads(path.read_text())
    tracks = {}
    for key, track in data["tracks"].items():
        frames = track.get("frames", 0)
        frame_count = len(frames) if isinstance(frames, list) else int(frames)
        if frame_count < min_frames:
            continue
        crops = sorted(track.get("crops", []), key=lambda crop: crop.get("quality", 0), reverse=True)
        unique = {}
        for crop in crops:
            identity = crop.get("frame", crop["path"])
            if identity not in unique:
                unique[identity] = _resolve_crop_path(crop["path"], path)
        if unique:
            tracks[str(key)] = list(unique.values())[:6]
    return tracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--analysis", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sheets-dir", type=Path)
    parser.add_argument("--min-track-frames", type=int, default=30)
    parser.add_argument("--limit-tracks", type=int, default=0)
    args = parser.parse_args()
    if args.analysis:
        tracks = tracks_from_analysis(args.analysis, args.min_track_frames)
    else:
        raw = json.loads(args.manifest.read_text())
        tracks = {str(key): [_resolve_crop_path(path, args.manifest) for path in paths][:6]
                  for key, paths in raw.items() if paths}
    if args.limit_tracks:
        tracks = dict(list(tracks.items())[:args.limit_tracks])
    sheets = args.sheets_dir or args.output.parent / "jersey-sheets"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    engine = JerseyVLM(args.model)
    output: dict[str, Any] = {
        "model": MODEL_ID, "model_path": str(engine.model_path),
        "model_revision": MODEL_REVISION,
        "load_seconds": engine.load_seconds, "inference_location": "local Apple Silicon MLX",
        "method": "Correlated panel readings in one contact sheet per existing track; at least 3 agreeing crops, 80% of readable votes, and 50% of all crops, followed by two matching single-crop checks in separate contexts. Outputs remain unverified candidates.",
        "tracks": {},
    }
    for index, (track, paths) in enumerate(tracks.items(), 1):
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", track)
        sheet = sheets / f"{index:03d}-{safe_id}.jpg"
        result = engine.recognize_track(paths, sheet)
        output["tracks"][track] = result
        output["timing"] = {
            "wall_seconds": time.perf_counter() - run_started,
            "inference_seconds": sum(item["inference_seconds"] for item in output["tracks"].values()),
            "model_load_seconds": engine.load_seconds,
        }
        output["peak_memory_gb"] = max(item["peak_memory_gb"] for item in output["tracks"].values())
        # Checkpoint each track, preserving partial useful work if interrupted.
        checkpoint = args.output.with_suffix(args.output.suffix + ".checkpoint")
        checkpoint.write_text(json.dumps(output, indent=2) + "\n")
        checkpoint.replace(args.output)
        print(f"[{index}/{len(tracks)}] track {track}: {result['status']} {result['number']} "
              f"({result['supporting_crops']}/{result['total_crops']} crops), "
              f"{result['inference_seconds']:.2f}s", flush=True)
    if not tracks:
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped; completed tracks remain in the output checkpoint.", file=sys.stderr)
        raise SystemExit(130)
