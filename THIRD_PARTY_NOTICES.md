# Third-party notices

The source in this repository is distributed under AGPL-3.0. Third-party components retain their own licenses; this repository's license does not relicense model weights, datasets or broadcast footage.

## Original workflow and court geometry

This project builds on the workflow demonstrated by [Piotr Skalski](https://x.com/skalskip92/status/2098072972011442316) and the [Roboflow basketball notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb). It uses replacement pretrained checkpoints and local execution paths.

Court dimensions and vertex geometry in `court.py` derive from [Roboflow sports](https://github.com/roboflow/sports/blob/feat/basketball/sports/basketball/config.py). The model-slot mapping, fit validation and rendering integration were added for this prototype. The upstream [MIT license](https://github.com/roboflow/sports/blob/feat/basketball/LICENSE) is retained below.

```text
MIT License

Copyright (c) 2024 Roboflow

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Downloaded pretrained models

Weights are downloaded from their authors and are not bundled here. Downloads preserve the weights unchanged; local filenames may differ.

| Model | Credit | Upstream license |
|---|---|---|
| [E-BARD basketball RF-DETR Nano](https://huggingface.co/GabrieleGiudici/E-BARD-detection-models) | Gabriele Giudici; RF-DETR by Roboflow | CC BY 4.0 for the checkpoint |
| [YOLO11n basketball court keypoints](https://huggingface.co/koppolusameer/yolo11n-basketball-court-keypoints) | Sameer Prasad Koppolu; Ultralytics | AGPL-3.0 |
| [Qwen3-VL 8B Instruct, MLX 4-bit](https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit) | Qwen; conversion by mlx-community | Apache-2.0 |

The separately installed [Ultralytics runtime](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) is AGPL-3.0. Dependency distributions include their respective license notices. Exact package versions are recorded in the two requirements files. See [ASSETS.md](ASSETS.md) for model revisions, checksums, source dataset notes and provenance.

## Sample footage

Optional sample downloads use the original tutorial's public Google Drive links. No separate open license was identified for the underlying NBA broadcast footage. Video, screenshots, jersey crops, and training data are excluded from this source distribution. Use footage you have permission to process and share.
