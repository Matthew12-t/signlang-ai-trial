# Docker deployment for Sign Language Service

This bundle runs the edge-local Sign Language Service on `127.0.0.1:8765`.
The browser talks directly to the container through the published loopback
port. The container does not call the Application Server, and video is only
held for the duration of a prediction request by the existing service code.

## Prerequisites

- Docker Desktop with the WSL2 backend on Windows.
- Docker Compose v2 with support for GPU device reservations.
- For real SignBart inference: a compatible NVIDIA driver, working NVIDIA
  container GPU passthrough, the matching checkpoint, the matching YAML, the
  upstream SignBart checkout, and the validated vocabulary mapping.
- The base image tag and PyTorch CUDA wheel must be reachable from the Docker
  build environment. Verify the tag before the first build:

```powershell
docker manifest inspect python:3.10.16-slim-bookworm
```

The repository's current development environment uses Python 3.10, MediaPipe
0.10.21, and PyTorch 2.7.1 with CUDA 11.8. The pinned Docker dependencies are
in `requirements-docker.txt`; the CUDA wheel is selected explicitly from the
official PyTorch cu118 index.

## Safe mock demo

Copying the example file is optional because the compose file already has the
same safe defaults. It contains no secret and leaves `INTERNAL_API_KEY` empty.

```powershell
Copy-Item .env.docker.example .env.docker
docker compose --env-file .env.docker.example config
docker compose --env-file .env.docker.example build
docker compose --env-file .env.docker.example up --detach sign
```

The host binding is intentionally fixed to `127.0.0.1:8765:8765`. Uvicorn
listens on `0.0.0.0` only inside the container so Docker can publish the port.

Verify liveness and mock readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health/live
Invoke-RestMethod http://127.0.0.1:8765/health/ready
docker compose --env-file .env.docker.example ps
docker compose --env-file .env.docker.example port sign 8765
```

The mock prediction must report `modelVersion=mock-sign-v0`; it is not real
model inference. A repeatable end-to-end smoke test generates a temporary MP4
inside the container and removes it after the run:

```powershell
.\scripts\docker-smoke.ps1
```

The smoke test checks build/start, liveness, readiness, a valid multipart
prediction, the mock model version, allowed and denied CORS preflight,
loopback-only port publishing, non-root execution, and that the image does not
contain `.env`, `.venv`, model checkpoints, or video fixtures.

## Real SignBart profile

The real profile is opt-in and is the only service that requests an NVIDIA GPU.
Do not start it with arbitrary artifacts. The checkpoint, YAML, upstream source,
and vocabulary must be a matched set; a mismatch must leave readiness at
`MODEL_NOT_READY`.

Set the host checkout path and start the profile:

```powershell
$env:SIGNBART_REPO_HOST_PATH = 'C:\Users\ASUS\.vscode\project\SignBart'
docker compose --env-file .env.docker.example --profile real config
docker compose --env-file .env.docker.example --profile real up --detach sign-real
```

The real service mounts these paths read-only:

- `./models` -> `/app/models`
- `./config` -> `/app/config`
- `$env:SIGNBART_REPO_HOST_PATH` -> `/opt/signbart`

The real profile uses `/app/models/SignBart-WLASL-100/model.safetensors`,
`/opt/signbart/configs/WLASL-100.yaml`, and
`/app/config/mvp-en-v1.json`. It does not download a checkpoint at startup.

Verify GPU visibility separately from model readiness:

```powershell
.\scripts\verify-gpu.ps1
docker compose --env-file .env.docker.example --profile real logs sign-real
Invoke-WebRequest http://127.0.0.1:8765/health/ready
.\scripts\docker-real-smoke.ps1
```

The GPU check must show `cuda_available: True`, a CUDA version, and at least one
device. That alone does not prove SignBart is ready; `/health/ready` must also
show vocabulary, model, device, and warm-up checks as ready. If artifacts are
missing or mismatched, an HTTP 503 with `MODEL_NOT_READY` is the correct result.

## Stop and cleanup

```powershell
docker compose --env-file .env.docker.example down --remove-orphans
docker image rm isyara-sign-service:local
```

The compose stop command removes containers and the network only. It does not
delete host-mounted models, configs, or the upstream checkout.

## Troubleshooting

- `docker manifest inspect` or `docker build` cannot reach the registry: fix
  Docker Desktop proxy/network access before treating the image as verified.
- Mock service starts but readiness is 503: check that `/app/config/mvp-en-v1.json`
  is mounted and that all three confidence thresholds are configured.
- Real service has `MODEL_NOT_READY`: inspect `/health/ready` and verify the
  checkpoint extension, checksum, YAML, upstream source, output dimension, and
  vocabulary mapping as one matched artifact set.
- GPU check fails: confirm Docker Desktop uses WSL2, the Windows NVIDIA driver
  supports CUDA containers, and the Docker GPU test works before debugging the
  application.
- Browser CORS fails: set `SIGN_ALLOWED_ORIGINS` to the exact frontend origin.
  Never use `*`; `SIGN_ALLOW_PRIVATE_NETWORK` should remain false until the
  browser PNA preflight has been explicitly tested.

## Python regression tests

Run the existing suite outside Docker with the repository's Python 3.10
environment as documented by the project:

```powershell
$Project = 'C:\Users\ASUS\.vscode\project\signlang-ai-trial'
$Py = "$Project\.venv\Scripts\python.exe"
& $Py -m pytest -q -p no:cacheprovider
```

The Docker build and smoke test are independent of the host virtualenv.
