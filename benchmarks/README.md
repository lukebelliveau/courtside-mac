# Measured M3 Ultra run

Follow-up: [three repeated runs on a different basketball game](fresh-footage/README.md), including process-startup timing, sparse-Qwen call counts and observed failures on the new court. The original numbers below remain measurements of the original ten-clip workload.

Recorded on 10 September 2026 on a Mac Studio with M3 Ultra, 32 CPU cores, 80 GPU cores, 512 GB unified memory, macOS 26.5.2. These measurements come from the original prototype before source packaging.

Ten Celtics–Knicks tutorial clips supplied **1,701 decoded frames, 56.7 seconds of video**, at 1920 × 1080 and 30 FPS. Container durations including audio total 57.43 seconds. Source clips are linked in [ASSETS.md](../ASSETS.md) and are not distributed here.

| Stage | Measured time |
|---|---:|
| Core models: load and warm-up | 1.01 s |
| Decode | 0.76 s |
| RF-DETR detection | 23.80 s |
| BoT-SORT tracking | 11.61 s |
| Court inference and geometry | 15.42 s |
| Metadata and crop extraction | 1.66 s |
| **Core analysis wall time** | **53.46 s / 31.8 FPS** |
| Jersey model load | 0.62 s |
| Jersey inference and verification calls | 148.40 s |
| Complete jersey pass, including overhead | 150.97 s |
| Rendering and H.264 encoding | 15.81 s |

The component passes total approximately **3 minutes 41 seconds**, excluding setup and downloads. The 31.8 FPS figure is core batch-processing throughput; the complete pipeline, including jersey reading and rendering, took longer than playback. Detector throughput within the core pass was 71.5 FPS.

Core peak RSS was 1.23 GB; Metal driver allocation peaked at 1.40 GB. MLX reported a 6.75 GB peak during jersey reading. The counters have different accounting and can overlap in unified memory; do not add them. This run does not establish a minimum supported RAM configuration.

The run produced 131 confirmed track fragments and 16,264 player observations. Court calibration was valid on 1,237 of 1,701 frames (72.7%). Jersey reading evaluated 116 eligible tracks, retained 49 candidates and abstained on 67. These are output counts, not accuracy scores or unique-player counts.

## Detector tuning sample

Synchronized predictions on 32 real frames. Timing includes detector preprocessing and postprocessing; excludes decoding and warm-up. Agreement compares player boxes with the full-precision baseline, not human annotations.

| Precision | Batch | Resolution | FPS | Baseline boxes recovered at IoU > 0.8 |
|---|---:|---:|---:|---:|
| float32 | 1 | 704 | 32.1 | Baseline |
| float32 | 4 | 704 | 59.8 | 100.0% |
| float16 | 1 | 704 | 40.5 | 100.0% |
| float16 | 4 | 704 | 72.9 | 100.0% |
| float32 | 1 | 512 | 42.3 | 90.4% |

## Verification and limitations

Ten court geometry and actual BoT-SORT lifecycle regression tests passed. The saved full run was checked for negative IDs and duplicate confirmed IDs within a frame. Rendered frames and jersey evidence received visual spot checks; no labeled accuracy study was performed.

Separately, a two-second, 60-frame clip passed asset verification, detection, tracking, court mapping, local Qwen jersey reading, rendering and full video decode with the macOS sandbox policy `(version 1) (allow default) (deny network*)`. A network connection probe failed with `Operation not permitted`. This demonstrates offline operation with already-installed dependencies and downloaded assets; it does not claim an offline installation.

The benchmark clips overlap the court model's underlying dataset. Uniform grouping uses a blue/white preset. Court coordinates and jersey candidates remain estimates. Shot-scoped tracking does not establish identity across plays, and possession/event classification is not implemented.

Machine-readable records: [full-run summary](run-summary.json), [detector comparison](detector.json), [network-denied verification](offline-verification.json). Private local paths have been omitted; no source footage or crop images are included.

## Repository packaging check

After relocating the scripts into a standalone repository, the two-second end-to-end test passed again with networking denied. The packaged code used the existing installed interpreters and downloaded weights through explicit paths; a fresh dependency installation was not repeated. Nine source regression tests passed; the generated-output audit was skipped because the clean repository contains no saved full-run artifacts. Python compilation, shell syntax/help checks and the two model checksums passed. See [packaged-code verification](package-verification.json).
