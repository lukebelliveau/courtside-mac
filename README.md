# Courtside Mac

Local basketball video analysis on Apple Silicon: detect players and the ball, track players within a shot, project positions onto a court, and read tentative jersey numbers. All model inference runs on the Mac using PyTorch MPS and MLX.

Inspired by [Piotr Skalski's basketball analysis thread](https://x.com/skalskip92/status/2098072972011442316) and [Roboflow's basketball tutorial](https://github.com/roboflow/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb). This is an independent local adaptation using public pretrained checkpoints. It does not reproduce the original pipeline's complete identity or event analysis.

**Measured on an M3 Ultra:** 56.7 seconds of 1080p basketball processed in 53.46 seconds for detection, tracking and court mapping (**31.8 FPS**). Local jersey reading added about 151 seconds; rendering added 15.81 seconds. See [benchmark results](benchmarks/README.md) for scope, memory measurements and limitations.

## Quick start

Requires an Apple Silicon Mac, `uv`, and FFmpeg. Setup creates two isolated Python 3.12 environments and downloads pinned models. The jersey model is approximately 5.4 GiB; allow additional space for Python dependencies, footage and outputs. The tested machine had 512 GB RAM; a minimum supported RAM configuration has not been established.

Clone the repository, then run setup:

```sh
git clone https://github.com/lukebelliveau/courtside-mac.git
cd courtside-mac

# Install uv and FFmpeg first if needed, for example with Homebrew:
brew install uv ffmpeg

# Install dependencies and download models once.
./setup.sh

# Process a video saved on your Mac.
./run.sh /absolute/path/to/basketball-clip.mp4
```

To reproduce the tutorial sample run, opt into downloading its ten public clips:

```sh
./setup.sh --demo
./run.sh
```

The sample footage is downloaded from the original tutorial's public links. Broadcast footage and model weights are not included in this repository. See [asset provenance](ASSETS.md) for sources, checksums, revisions and licenses.

Setup requires internet access. Once the environments, models and input videos are on disk, `run.sh` can operate offline. A two-second end-to-end run passed under OS-enforced network denial; [the recorded result](benchmarks/offline-verification.json) includes the failed network probe and successful processing stages. This is a standalone command-line pipeline; no inference API key is required.

## Output

The default run writes:

- `outputs/basketball-demo.mp4`: annotated 1600 × 900, 30 FPS video, without audio.
- `outputs/demo/analysis.json`: tracks, source metadata, timings and memory readings.
- `outputs/demo/frames.jsonl` and `trajectories.csv`: per-frame observations and accepted court positions.
- `outputs/demo/jerseys.json` and `jersey-evidence/`: candidate readings, verification results and contact sheets.

Selected torso crops are saved in `work/demo/crops/`. Keep them to rerun jersey reading. Re-running the wrapper replaces its default output files; move an earlier result elsewhere if you want to retain it.

## Pipeline

| Stage | Implementation |
|---|---|
| Detection | Gabriele Giudici's E-BARD RF-DETR Nano; 704 × 704, MPS, float16, batch 4 |
| Tracking | BoT-SORT with camera-motion compensation; confirmed track IDs only |
| Uniforms | Blue/white torso-color votes accumulated within a track |
| Court mapping | Sameer Prasad Koppolu's YOLO11n court landmarks; 960-pixel input, mapped floor landmarks, RANSAC homography |
| Jersey candidates | Qwen3-VL 8B Instruct, MLX 4-bit; up to six crops and two separate crop checks |
| Rendering | OpenCV annotations and FFmpeg H.264 encoding |

The contribution here is the local integration: model loading and MPS batching, verified court landmark mapping, conservative geometry checks, crop selection and jersey verification, tracking safeguards, rendering, and measured offline execution. The pretrained models, inference libraries and BoT-SORT algorithm come from their respective authors. No models were trained for this project and no custom Metal kernels were written.

## What the labels mean

- Track IDs apply within a shot. Occlusion can split one player into multiple track fragments. The project does not identify player names or merge identities across plays.
- A jersey number followed by `?` is a model candidate. It requires three agreeing readings, at least 80% of readable votes, support from half of all crops, and two matching single-crop checks. Repeated calls to the same model are correlated evidence, not calibrated confidence. Labels can use later frames from the same track.
- Court positions are withheld when calibration is invalid, stale or projects outside the court. Bounding-box feet can move during jumps and occlusions.
- Uniform grouping currently assumes **blue and white uniforms**, as in the sample clips. Adapt `uniform_vote()` in `analyze.py` for other teams. Court geometry assumes an NBA floor.
- The exported `near_ball` field is a proximity heuristic. Possession classification, shot recognition and event analytics are not implemented.

The sample footage overlaps the court model's underlying dataset. These results demonstrate integration and hardware throughput; they are not an independent accuracy evaluation. Only the M3 Ultra configuration has been exercised. The current scripts target macOS MPS and MLX Metal; CUDA and managed Colab execution have not been adapted or tested.

## Development

```sh
work/.venv/bin/python -m unittest discover -s . -p 'test_*.py' -v

# Detector benchmark uses the sample clips downloaded by ./setup.sh --demo.
work/.venv/bin/python benchmark.py

# Run individual stages:
work/.venv/bin/python analyze.py --batch 4 --precision float16 --court /absolute/path/to/clip.mp4
work/ocr-venv/bin/python vlm_ocr.py \
  --analysis outputs/demo/analysis.json \
  --output outputs/demo/jerseys.json \
  --sheets-dir outputs/demo/jersey-evidence
work/.venv/bin/python render.py --jerseys outputs/demo/jerseys.json --title 'BASKETBALL ANALYSIS'
```

Run from a normal macOS terminal with GPU access. Restricted processes can report MPS unavailable even when PyTorch includes it. The scripts do not change system memory limits. Dependency sets are pinned separately in `requirements.txt` and `requirements-ocr.txt`.

## License and credits

Project source is released under [AGPL-3.0](LICENSE). Upstream code, pretrained weights and footage retain their respective licenses. The integrated court model and Ultralytics runtime use AGPL-3.0. See [third-party notices](THIRD_PARTY_NOTICES.md) and [asset provenance](ASSETS.md).

Credit to Piotr Skalski and Roboflow for the original workflow and court geometry; Gabriele Giudici for E-BARD; Sameer Prasad Koppolu for the court model; Qwen and mlx-community for the jersey model; and the RF-DETR, trackers, supervision, PyTorch, MLX, Ultralytics, OpenCV and FFmpeg maintainers.
