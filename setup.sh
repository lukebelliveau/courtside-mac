#!/bin/zsh
# Install the pinned local environments and model weights; --demo adds sample videos.
set -euo pipefail
TASK_ROOT="${0:A:h}"
cd "$TASK_ROOT"
TASK_DEMO=0
case "${1:-}" in
  --help|-h)
    print 'Usage: ./setup.sh [--demo]'
    print 'Default: install Python environments and download model weights (~6 GB).'
    print -- '--demo: also download the 10 public tutorial clips.'
    print 'Requires Apple Silicon macOS, uv, and ffmpeg (brew install uv ffmpeg).'
    exit 0 ;;
  --demo) TASK_DEMO=1 ;;
  '') ;;
  *) print 'Usage: ./setup.sh [--demo]' >&2; exit 2 ;;
esac
if (( $# > 1 )); then print 'Usage: ./setup.sh [--demo]' >&2; exit 2; fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  print 'This setup requires Apple Silicon macOS. Run a native arm64 terminal.' >&2
  exit 1
fi
for TASK_COMMAND in uv ffmpeg; do
  if ! command -v "$TASK_COMMAND" >/dev/null 2>&1; then
    print "Missing $TASK_COMMAND. Install prerequisites with: brew install uv ffmpeg" >&2
    exit 1
  fi
done
export UV_CACHE_DIR="$TASK_ROOT/work/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/work/python"
export HF_HOME="$TASK_ROOT/work/huggingface"
export HF_HUB_OFFLINE=0
mkdir -p "$TASK_ROOT/work"
uv python install 3.12.8
for TASK_ENV in .venv ocr-venv; do
  if [[ ! -x "$TASK_ROOT/work/$TASK_ENV/bin/python" ]]; then
    uv venv --python 3.12.8 "$TASK_ROOT/work/$TASK_ENV"
  fi
done
uv pip sync --python "$TASK_ROOT/work/.venv/bin/python" requirements.txt
uv pip sync --python "$TASK_ROOT/work/ocr-venv/bin/python" requirements-ocr.txt
if (( TASK_DEMO )); then
  "$TASK_ROOT/work/.venv/bin/python" download_assets.py
else
  "$TASK_ROOT/work/.venv/bin/python" download_assets.py --models-only
fi
"$TASK_ROOT/work/ocr-venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
from vlm_ocr import DEFAULT_MODEL, MODEL_ID, MODEL_REVISION

snapshot_download(
    repo_id=MODEL_ID,
    revision=MODEL_REVISION,
    local_dir=DEFAULT_MODEL,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
)
print(f"Local jersey model ready: {DEFAULT_MODEL}")
PY
print 'Setup complete. Run ./run.sh /path/to/video.mp4'
if (( TASK_DEMO )); then print 'Or run ./run.sh for the downloaded tutorial demo.'; fi
