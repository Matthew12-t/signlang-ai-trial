# Isyara AI Services

REST service untuk model AI Isyara. Implementasi yang tersedia saat ini:

- Speech-to-Text lokal menggunakan `Systran/faster-whisper-large-v3` dan CTranslate2.
- Text-to-Speech lokal menggunakan `openbmb/VoxCPM2`.
- Health/readiness check dan error contract yang konsisten.

Kontrak serta perilaku layanan mengacu pada `isyara-ai-services-design.md` dan
`isyara-application-server-design.md`.

## Persyaratan

- Python 3.10 atau lebih baru.
- NVIDIA CUDA 12 dan cuDNN 9 untuk deployment GPU Faster Whisper terbaru.
- VRAM yang cukup. Jalankan STT dan TTS sebagai deployment terpisah bila kedua
  model tidak dapat berada di GPU yang sama.

Kedua model dijalankan secara lokal. Hugging Face digunakan untuk mengunduh
model, bukan sebagai hosted inference provider.

## Instalasi

Instal API dan kedua runtime model:

```bash
python -m pip install -e ".[all]"
```

Untuk menjalankan test tanpa mengunduh model:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

Salin `.env.example` menjadi `.env`, export variabel yang diperlukan, kemudian:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001
```

FastAPI tidak membaca `.env` secara otomatis. Gunakan pengelola environment
deployment atau jalankan Uvicorn dengan `--env-file .env`.

Untuk deployment model terpisah:

```dotenv
# STT deployment
ENABLED_SERVICES=stt

# TTS deployment
ENABLED_SERVICES=tts
```

## Endpoint

```text
POST /v1/stt/transcriptions
POST /v1/tts/synthesize
GET  /health/live
GET  /health/ready
```

`POST /v1/tts/synthesize` mewajibkan header `Idempotency-Key`. Jika
`INTERNAL_API_KEY` dikonfigurasi, seluruh endpoint model juga mewajibkan header
`X-Internal-API-Key`.

Dokumentasi interaktif tersedia di `/docs`. Kontrak mesin ada di
`contracts/openapi.yaml`.
