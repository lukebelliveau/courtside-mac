#!/bin/zsh
# Analyze local videos; no arguments runs the separately downloaded tutorial demo.
set -euo pipefail
TASK_ROOT="${0:A:h}"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  print 'Usage: ./run.sh [video.mp4 ...]'
  print 'Run ./setup.sh first, then pass one or more local video paths.'
  print 'For the public tutorial demo: ./setup.sh --demo, then ./run.sh'
  print 'Results are written to outputs/ inside this repository.'
  exit 0
fi
# Resolve inputs before changing directories so caller-relative paths stay valid.
typeset -a TASK_VIDEOS
TASK_VIDEOS=()
for TASK_VIDEO in "$@"; do
  if [[ ! -f "$TASK_VIDEO" ]]; then
    print "Video does not exist: $TASK_VIDEO" >&2
    exit 1
  fi
  TASK_VIDEOS+=("${TASK_VIDEO:A}")
done
cd "$TASK_ROOT"
TASK_PY="$TASK_ROOT/work/.venv/bin/python"
TASK_OCR_PY="$TASK_ROOT/work/ocr-venv/bin/python"
if [[ ! -x "$TASK_PY" || ! -x "$TASK_OCR_PY" ]]; then
  print 'Environments missing. Run ./setup.sh first.' >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  print 'Missing ffmpeg. Install it with: brew install ffmpeg' >&2
  exit 1
fi
if [[ ! -f "$TASK_ROOT/work/ocr-research/qwen3-vl-8b-4bit/config.json" ]]; then
  print 'Jersey model missing. Run ./setup.sh first.' >&2
  exit 1
fi
export HF_HUB_OFFLINE=1
export YOLO_OFFLINE=1
export YOLO_AUTOINSTALL=false
TASK_LABEL='BASKETBALL ANALYSIS'
if (( ${#TASK_VIDEOS} > 0 )); then
  "$TASK_PY" download_assets.py --verify-only --models-only
else
  TASK_LABEL='CELTICS / KNICKS'
  "$TASK_PY" download_assets.py --verify-only
fi
"$TASK_PY" analyze.py --batch 4 --precision float16 --court "${TASK_VIDEOS[@]}"
"$TASK_OCR_PY" vlm_ocr.py --analysis "$TASK_ROOT/outputs/demo/analysis.json" --output "$TASK_ROOT/outputs/demo/jerseys.json" --sheets-dir "$TASK_ROOT/outputs/demo/jersey-evidence"
"$TASK_PY" render.py --jerseys "$TASK_ROOT/outputs/demo/jerseys.json" --title "$TASK_LABEL"
print "Video: $TASK_ROOT/outputs/basketball-demo.mp4"
