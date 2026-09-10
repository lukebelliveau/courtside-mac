# Independent-footage timing check — M3 Ultra

Measured 10 September 2026. The detector and core-processing timings were close to the original demonstration across three repeated runs on a different basketball game. The new clip also exposed failures in court calibration, uniform grouping and detection quality. This is evidence about throughput, not equivalent accuracy to Astra or a successful general-purpose basketball analysis system.

## Workload and configuration

- Source: [PlayOn's public basketball CV sample](https://github.com/playon/basketball-analysis-samples), Mayfield Trojans vs Centennial Hawks. A 60-second excerpt beginning at the requested 18-minute stream offset; first half free throws, then active play and transitions.
- Exactly **1,500 decoded source frames**, **1920 × 1080**, **25 FPS**, **60.000 seconds**. The source was stream-copied without rescaling or frame-rate conversion. An HLS timestamp discontinuity was corrected by FFmpeg; the saved clip decoded fully without errors.
- Hardware: Apple M3 Ultra, 32 CPU cores, 80 GPU cores, 512 GB unified memory, macOS 26.5.2. Normal desktop session; no thermal/performance warning was reported before the run.
- Detector: existing E-BARD RF-DETR Nano, 704 × 704, FP16, batch 4, PyTorch MPS. Court model: existing YOLO11n, 960-pixel input, default FP32. Jersey model: existing Qwen3-VL 8B Instruct, MLX 4-bit.
- No model or algorithm was changed for this clip. Source selection happened through visual inspection before model execution. Environments and weights were already installed. Model code hashes and dependency versions are in [measurements.json](measurements.json).

Exact source URL, revision, trim command and SHA-256 are in [source-provenance.json](source-provenance.json). The publisher documents downloading for CV prototyping; no separate redistribution license was found. This report includes links and measurements, not footage or crop images.

## Repeated core timing

Each repeat used a new process and fresh output/crop directories. The detector received three warm-up batches before the analysis timer started. Detector timing includes color conversion, preprocessing, model execution, postprocessing and explicit GPU synchronization. Court output transfers to CPU complete before geometry processing.

| Run | Detection, ms/source frame | Core processing, ms/source frame | Core analysis time | Entire core process |
|---|---:|---:|---:|---:|
| 1 | 13.513 | 29.602 | 44.402 s | 50.597 s |
| 2 | 13.442 | 29.115 | 43.673 s | 48.330 s |
| 3 | 13.357 | 28.957 | 43.436 s | 48.070 s |
| **Mean** | **13.437** | **29.225** | **43.837 s** | **48.999 s** |
| Sample standard deviation | 0.078 | 0.336 | 0.504 s | 1.390 s |

Core processing includes decoding, detection, tracking, court inference, crop extraction and metadata collection. Its internal timer excludes imports, model load/warm-up and final metadata serialization. Entire core-process timing includes those costs.

These are **batch-throughput averages** over all source frames. They are not single-frame request latencies, and three repetitions on one clip do not establish performance across all games or machines. Model files and OS caches may be warm.

## One execution of each complete stage

The table sums core run 1, one Qwen process and one render process. Core repeats 2 and 3 are excluded from this total. Because the repeats occurred between stages, this is a sum of measured stage-process wall times, not an uninterrupted stopwatch interval.

| Stage | Process start-to-exit time | Amortized per source frame |
|---|---:|---:|
| Core run 1, including imports and warm-up | 50.597 s | 33.732 ms |
| Local jersey analysis | 55.881 s | 37.254 ms |
| Rendering and H.264 encoding | 14.797 s | 9.865 ms |
| **Total** | **121.275 s** | **80.850 ms** |

Thus, this 60-second video required about **2 minutes 1 second** of processing for one execution of each stage, including process startup. Setup, downloads, separate repeat runs and verification are excluded. Rendering produced 1,800 frames at 30 FPS; the timing denominator remains the original **1,500 source frames**.

## What the jersey model actually processed

Qwen evaluated **44 track fragments**, not 1,500 full video frames. It made 44 contact-sheet calls containing 259 torso panels in total, plus 44 individual-crop verification calls. It retained 19 tentative jersey candidates and abstained on 25 tracks. Repeated track fragments can belong to the same player.

| Qwen operation | Calls | Mean generation-call time |
|---|---:|---:|
| Contact sheet containing up to six torso crops | 44 | 1.018 s |
| Individual-crop verification | 44 | 0.192 s |

The per-call timer wraps the local `generate()` call, including image processing and text generation. The 37.254 ms averaged over each source video frame is an allocation of the complete jersey-process cost; it is **not** Qwen full-frame inference latency. The unchanged eligibility threshold of 30 source frames means 1.2 seconds at this clip's 25 FPS, versus 1 second at the original clip's 30 FPS.

## Comparison with the original demonstration

| Measurement | Original 10-clip montage | New 60-second excerpt |
|---|---:|---:|
| Source frames | 1,701 | 1,500 |
| Detector throughput average | 13.99 ms/frame | 13.44 ms/frame, mean of 3 runs |
| Core-processing throughput average | 31.43 ms/frame | 29.22 ms/frame, mean of 3 runs |
| Qwen track fragments evaluated | 116 | 44 |
| Sum of internal stage timers / source frames | 129.47 ms/frame | 76.52 ms/frame, core run 1 |
| Complete stage-process times / source frames | Not recorded | 80.85 ms/frame, core run 1 |

The earlier approximate **130 ms** figure used internal stage timers and excluded process-startup overhead. It should not be called a precise end-to-end stopwatch measurement. It also depends on the number of track fragments and resulting Qwen calls; it is not a fixed cost per video frame. The new footage has one continuous scene and fewer eligible track fragments than the ten-clip montage.

## Quality and scope limitations

All three repeats produced the same per-frame metadata hash, 63 confirmed track fragments, and 11,052 player observations. Negative or duplicate confirmed IDs were absent. Both detector and court weight checksums matched the pinned manifest. The final video fully decoded, with its 1,800 output frames matching renderer metadata. Details: [validation.json](validation.json).

However, **court calibration was valid on 0 of 1,500 frames**. Court inference ran on every frame, but the mapper produced no court coordinates. The school court and camera view differ from the NBA setting the model and geometry assume. This cannot be reported as successful court mapping on unseen footage.

Visual checks also found a referee classified/tracked as a player and incorrect uniform grouping. The blue/white preset does not fit this game's teams. Four sampled accepted jersey candidates (33, 1, 50 and 34) were visually supported; this spot check does not establish jersey accuracy or identity continuity. Candidate counts and calibration coverage are not accuracy scores.

The system still has no cross-play player re-identification, confirmed player names or robust possession/event recognition. This new source differs from the original demo; its exclusion from every upstream model's training data has not been established.

## How to discuss the Astra comparison

[Skalski's reply](https://x.com/skalskip92/status/2098119733451031039) says **16 seconds to process a single frame**, not 14 milliseconds. Our batch-4 detector timing, selected-crop Qwen timing and amortized whole-pipeline timing measure different work. The models, inputs, outputs and supported capabilities differ. These results do not support a direct speedup factor or an accuracy-parity claim against Astra.

Suggested wording:

> Rechecked on a different 60s 1080p/25fps basketball clip, 3 runs on M3 Ultra: detector 13.4ms/frame; core processing 29.2ms/frame (batch 4). One pass of core + sparse local jersey OCR + rendering totaled 121s for 60s of video, including process startup.

> Those are throughput averages, not Astra-equivalent frame latency. Qwen evaluated 44 track fragments; we don't reproduce cross-play ReID or possession. The new school-court clip also exposed calibration and team-grouping failures, so this validates timing—not general accuracy.

## Reproduce

Use the repository's installed dependencies and weights from `setup.sh`. Save the source excerpt using the command in [source-provenance.json](source-provenance.json), then run the included [benchmark harness](run_benchmark.py). It requires explicit paths and creates a new output directory; `--help` lists every argument.

From the repository root, after saving this report folder as `benchmarks/fresh-footage`:

```sh
work/.venv/bin/python benchmarks/fresh-footage/run_benchmark.py \
  --video work/fresh-source.mp4 --app . \
  --python work/.venv/bin/python --ocr-python work/ocr-venv/bin/python \
  --assets work/assets --qwen-model work/ocr-research/qwen3-vl-8b-4bit \
  --cache-root work --out work/fresh-run \
  --label 'MAYFIELD / CENTENNIAL' --limit-seconds 60 --repeats 3 \
  --source-url https://github.com/playon/basketball-analysis-samples
```

No cloud inference or paid API calls are part of this benchmark.
