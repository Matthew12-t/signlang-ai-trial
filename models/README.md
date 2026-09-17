# SignBart checkpoints

Checkpoints are intentionally not stored in Git.

Before using `SIGN_MODEL_BACKEND=signbart`, record the checkpoint source, license,
SHA-256 checksum, expected input shape, and the loader reference in the deployment
notes. The adapter accepts the upstream SignBart repository plus matching YAML/JSON
config through `SIGN_SIGNBART_REPO` and `SIGN_SIGNBART_CONFIG`. It loads the
upstream `SignBart(config)` class and `.pth` or `.safetensors` state dict. A project-specific loader
may still be supplied through `SIGN_SIGNBART_LOADER` when a deployment uses a
compatible wrapper.

The upstream repository documents MediaPipe Holistic skeleton input with shape
`(T, 75, 2)` and provides dataset-specific weights/configs. Do not mix a
checkpoint and config from different datasets. Record the chosen checkpoint's
source URL, license, SHA-256 checksum, dataset, config filename, vocabulary
mapping, and evaluation result here before demo release.

## Selected development artifact

- Model: `tinh2312/SignBart-WLASL-100`
- Source: `https://huggingface.co/tinh2312/SignBart-WLASL-100`
- Revision: `8f98ae70a36b61c9aeb23a2e7f1d6d618b017dad`
- Checkpoint: `model.safetensors`
- Checkpoint SHA-256: `A7F2F16AF2A07508469CC361653F855FB333AE93C7D979C05A70E3087E37A921`
- Runtime config: `../SignBart/configs/WLASL-100.yaml`
- Runtime config SHA-256: `7F9E761A840453C360716C60A52AFA40E6D5C7EB7504043228FA791F9D693431`
- Hugging Face config SHA-256 (reference only): `59F3E02C21D0B5C69254EE3A0E9EAC3F383F042D15D5482FD130924F44B0BDC0`
- Dataset/config: WLASL-100, 100 logits
- Upstream source revision: `TinhNguyen2312/SignBart@1b7d3d1a1df11ee9147464143ea002d2b8688970`
- Checkpoint license: not declared in the Hugging Face repository; do not
  redistribute it or treat it as production-cleared until the author confirms
  the license.
- Dataset restriction: WLASL is released for academic/computational use and
  excludes commercial use.

The Hugging Face `config.json` contains an `id2label` table that does not match
the canonical WLASL-100 ordering and is not used as the runtime config. The
service uses the upstream WLASL-100 YAML for architecture settings and the
first 100 glosses from the official WLASL v0.3 metadata as the logit mapping. The
vocabulary file retains all 100 indices for checkpoint compatibility and marks
20 meeting-relevant labels as enabled for the MVP. This mapping must be
validated with known WLASL clips before the model is used in the demo.

The original Google Drive link in the upstream README currently returns 404;
the Hugging Face repository owned by the same author is used instead.

## Local runtime smoke result

Validated on 2026-09-17 with Windows, Python 3.10, PyTorch 2.14.0+cu132,
MediaPipe 0.10.21, and an NVIDIA GeForce RTX 3060 Laptop GPU (6 GB):

- checkpoint load and CUDA warm-up: ready;
- model output dimension: 100 logits;
- warmed synthetic-skeleton model call: 18.12 ms;
- 24-frame, 2-second blank MP4 through decode, MediaPipe, model, and API:
  1,059 ms, correctly classified as `NO_SIGN`.

These numbers are development smoke measurements, not an accuracy benchmark.
Known labeled WLASL clips are still required to validate the logit-to-gloss
mapping and to calibrate confidence thresholds before the demo.

The default `mock` backend is deterministic and explicitly reports
`modelVersion=mock-sign-v0`; it is for contract and local UI development only.
