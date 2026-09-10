# Model and footage provenance

Retrieved 10 September 2026. The benchmark used public basketball models and the original tutorial’s sample footage. Weights and footage are downloaded separately; they are not included in this source package. No account credentials or inference APIs were used to obtain them.

## Detection model

**E-BARD RF-DETR Nano**, trained and released by Gabriele Giudici.

- [Model card](https://huggingface.co/GabrieleGiudici/E-BARD-detection-models)
- Architecture: RF-DETR Nano, 4 basketball classes, trained at 704 × 704.
- Class IDs: `0=basketball`, `1=hoop`, `2=player`, `3=referee`. These are verified from [the author’s COCO conversion source](https://github.com/GabrieleGiudic/E-BARD/blob/main/detection/src/to_coco.py), which preserves zero-based category IDs.
- Model card license: **CC BY 4.0**.
- Revision: `3f4789c4431aa73269f60107a4ba0a5f86b7af8b`.
- [Pinned checkpoint download](https://huggingface.co/GabrieleGiudici/E-BARD-detection-models/resolve/3f4789c4431aa73269f60107a4ba0a5f86b7af8b/BODD_rf-detr-nano_0000/checkpoint_best_total.pth).
- Filename: `ebard-rfdetr-nano-basketball.pth`; 120811190 bytes.
- SHA-256: `759968ecf8f83663c85de3d881713e072f4a9dd0ea2f136b082878b1e4fe2400`.

The original tutorial’s `basketball-player-detection-3-ycjdo/4` resolves to YOLOv11s, despite RF-DETR wording in the notebook. [Official repository issue #431](https://github.com/roboflow/notebooks/issues/431) identifies `/13` as RF-DETR Medium. The original Roboflow project API requires an API key; this prototype therefore uses the separately released E-BARD checkpoint. It detects entities, but has no possession, shot-type, or jersey-number classes.

## Court landmark model

**YOLO11n basketball court keypoints**, released by Sameer Prasad Koppolu.

- [Model card](https://huggingface.co/koppolusameer/yolo11n-basketball-court-keypoints).
- Architecture: Ultralytics YOLO11n pose; 48 output keypoint slots.
- Model card license: **AGPL-3.0**.
- Revision: `6cb899251439982067a81baa225b33f45f335981`.
- [Pinned checkpoint download](https://huggingface.co/koppolusameer/yolo11n-basketball-court-keypoints/resolve/6cb899251439982067a81baa225b33f45f335981/best.pt).
- Filename: `yolo11n-basketball-court-keypoints.pt`; 7934664 bytes.
- SHA-256: `68e5faf5fb5388cf83477238abe22e9824a69fb33a4cf34f79ab61858f410064`.

The author’s [training dataset](https://huggingface.co/datasets/koppolusameer/basketball-keypoint-detection-yolov8) is a Roboflow basketball-court dataset export. Its `data.yaml` and Roboflow export note list **CC BY 4.0**; the Hugging Face dataset front page lists MIT. Preserve the upstream Roboflow attribution and treat that metadata discrepancy as unresolved.

Landmark indices are not interchangeable with the older notebook’s 33-vertex geometry. We downloaded 116 public test annotation files and visually inspected eight labeled image pairs to map model slots to the [Roboflow sports court geometry](https://github.com/roboflow/sports/blob/feat/basketball/sports/basketball/config.py). The prototype uses only the supported mapping. Unknown slots are excluded; floor projections below rims may be excluded from fitting. Some supplied clips are also represented in the court model’s dataset, so this is an integration and hardware demonstration, not an independent accuracy evaluation.

## Source footage

[Original notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb) → [public sample video folder](https://drive.google.com/drive/folders/1eDJYqQ77Fytz15tKGdJCMeYSgmoQ-2-H).

Ten Celtics–Knicks broadcast excerpts, 1920 × 1080, generally 30 fps. Total duration **57.43 seconds**, total download **74,831,222 bytes**. These are the tutorial’s public samples; no separate open license was identified for the underlying NBA broadcast footage.

| Clip | Duration | Public source |
|---|---:|---|
| boston-celtics-new-york-knicks-game-1-q1-01.54-01.48.mp4 | 6.38s | [Download](https://drive.google.com/uc?export=download&id=1GYxJGG8_OT5wlHxcYjUb62P2rjHeTg7p) |
| boston-celtics-new-york-knicks-game-1-q1-03.16-03.11.mp4 | 5.03s | [Download](https://drive.google.com/uc?export=download&id=1Gm-QSkfXngzEnkrQEdE1HsLPG3cU-KCV) |
| boston-celtics-new-york-knicks-game-1-q1-04.28-04.20.mp4 | 8.00s | [Download](https://drive.google.com/uc?export=download&id=1It3wz3eoGcjo6tI9OGUt3X69ZK6x5SNK) |
| boston-celtics-new-york-knicks-game-1-q1-04.44-04.39.mp4 | 4.84s | [Download](https://drive.google.com/uc?export=download&id=1hx7_rJAmgBjt6PgHH8wh49OwUh2Lhe9f) |
| boston-celtics-new-york-knicks-game-1-q1-05.13-05.09.mp4 | 3.97s | [Download](https://drive.google.com/uc?export=download&id=1fPEw_w51nNQeBHh_GYhRyThVg71V1trn) |
| boston-celtics-new-york-knicks-game-1-q1-06.00-05.54.mp4 | 6.10s | [Download](https://drive.google.com/uc?export=download&id=1zwnAE4jHVI0qH7ioDFnzX98XtOocAUbH) |
| boston-celtics-new-york-knicks-game-1-q1-07.41-07.34.mp4 | 7.04s | [Download](https://drive.google.com/uc?export=download&id=1wQMSO-C4jOm6BBIVUuJ_6Uqf0QLdbJHD) |
| boston-celtics-new-york-knicks-game-1-q2-08.09-08.03.mp4 | 5.57s | [Download](https://drive.google.com/uc?export=download&id=1w9e1zCZtXtOmi6C4m-1BJJzTpa3znkbB) |
| boston-celtics-new-york-knicks-game-1-q2-08.43-08.38.mp4 | 6.78s | [Download](https://drive.google.com/uc?export=download&id=1zaltcB_-j8Pzxh8rBuvAV2o0Ngf3IX0I) |
| boston-celtics-new-york-knicks-game-1-q2-10.36-10.32.mp4 | 3.71s | [Download](https://drive.google.com/uc?export=download&id=14DuRHherVUn-CmER3z9oOiroWfkbk7eP) |


The reusable asset manifest records each URL, model revision, byte count, and SHA-256 checksum. Regenerating the demo should verify checksums before loading model weights.

## Reproduce the downloads

Run `python3 download_assets.py` from this source directory, or `python3 download_assets.py --verify-only` to verify the existing files. The standard-library downloader reads `assets.json`, makes only unauthenticated HTTPS requests, and checks each download against its pinned SHA-256 and exact byte count. `--models-only` omits footage; `--assets PATH` selects another destination. Existing files with unexpected contents are preserved and reported. Failed partial downloads are also preserved; move an unwanted partial file with `trash` before retrying.

## Local jersey-number model

**Qwen3-VL-8B-Instruct, MLX 4-bit conversion**, released by `mlx-community` from the original Qwen model.

- [Pinned model repository](https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit/tree/defcdea7cc7a4b0858fea563cbbce171d328e457).
- [Original model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
- Model card license: **Apache-2.0**.
- Revision: `defcdea7cc7a4b0858fea563cbbce171d328e457`.
- Two safetensors shards total 5,760,665,246 bytes; the complete download is approximately 5.78 GB.
- Local directory: `work/ocr-research/qwen3-vl-8b-4bit`, measured as 5.4 GiB on disk.
- Runtime: Python 3.12.8, `mlx-vlm==0.7.0`, `mlx==0.32.2`, `mlx-metal==0.32.2`, `transformers==5.17.0`. The complete tested environment is pinned in `requirements-ocr.txt`.

This model is downloaded separately from the detector/court asset manifest. From the workspace root:

```bash
UV_CACHE_DIR=work/ocr-research/uv-cache uv venv --python 3.12 work/ocr-venv
UV_CACHE_DIR=work/ocr-research/uv-cache uv pip install --python work/ocr-venv/bin/python -r requirements-ocr.txt
HF_HOME=work/ocr-research/hf-cache work/ocr-venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    revision="defcdea7cc7a4b0858fea563cbbce171d328e457",
    local_dir="work/ocr-research/qwen3-vl-8b-4bit",
)
PY
```

`vlm_ocr.py` loads the local model with `trust_remote_code=False` and `HF_HUB_OFFLINE=1`; image inference runs on this Mac through MLX. It reads up to six torso crops per confirmed track, requires three agreeing readings, 80% agreement among readable crops, and support from at least half of all crops. Two additional single-crop calls, each without the sheet or an expected answer, must agree. Accepted numbers remain **candidates**, displayed with `?`; neither these votes nor the repeated calls establish a player's identity across camera cuts.
