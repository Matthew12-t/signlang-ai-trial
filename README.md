# Isyara AI Services

Scaffold layanan STT, TTS, gloss normalization, Conversation Recall, dan
vertical slice Sign Language Service untuk Isyara.

## Sign Language Service

Implementasi Sign Service berada di `src/sign` dan mengikuti kontrak
`contracts/isyara-sign-language-service-openapi.yaml` serta spesifikasi
`isyara-sign-language-service-design-spec.md`.

Service bersifat stateless dan menyediakan:

- `GET /health/live` untuk liveness proses;
- `GET /health/ready` untuk model, vocabulary, device, dan warm-up;
- `POST /v1/sign/predict` untuk satu klip isolated sign.

### Menjalankan vertical slice mock

Gunakan Python 3.10. MediaPipe 0.10.21 dipin karena menyediakan API Holistic
yang dipakai preprocessing upstream SignBart; MediaPipe 1.x tidak kompatibel
dengan adapter ini. Pasang dependency:

```powershell
$Project = "C:\Users\ASUS\.vscode\project\signlang-ai-trial"
$Py = "$Project\.venv\Scripts\python.exe"
Set-Location $Project
$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
& $Py -m pip install -r "$Project\requirements.txt"
```

Konfigurasi dibaca langsung dari environment process; file `.env` tidak dimuat
otomatis. Ekspor variabelnya pada shell yang sama dengan process server.
`SIGN_ALLOWED_ORIGINS` harus berisi origin frontend secara eksplisit. Jangan
gunakan `*`. `SIGN_ALLOW_PRIVATE_NETWORK` tetap `false` kecuali preflight PNA
browser demo telah diuji untuk origin tersebut.

Backend default adalah `mock`, yang deterministik dan sengaja mengembalikan
`modelVersion=mock-sign-v0`. Output tersebut bukan inferensi SignBart.

```powershell
$env:SIGN_HOST = "127.0.0.1"
$env:SIGN_PORT = "8765"
$env:SIGN_ALLOWED_ORIGINS = "http://localhost:5173"
$env:SIGN_ALLOW_PRIVATE_NETWORK = "false"
$env:SIGN_MODEL_BACKEND = "mock"
$env:SIGN_VOCABULARY_PATH = "./config/mvp-en-v1.json"
$env:SIGN_CONFIDENT_THRESHOLD = "0.80"
$env:SIGN_UNKNOWN_THRESHOLD = "0.40"
$env:SIGN_MIN_TOP1_TOP2_MARGIN = "0.20"
& $Py -m uvicorn src.sign.main:app --host 127.0.0.1 --port 8765
```

Readiness gagal secara eksplisit jika threshold, vocabulary, atau model belum
siap. Klip mock yang valid tetap melalui decoder dan preprocessing RGB, tetapi
hasilnya selalu ditandai `mock-sign-v0`.

### Menjalankan adapter SignBart nyata

Adapter nyata mengikuti implementasi upstream
[`TinhNguyen2312/SignBart`](https://github.com/TinhNguyen2312/SignBart): video
diubah menjadi skeleton MediaPipe Holistic `(T, 75, 2)`, dinormalisasi per
komponen body/left-hand/right-hand, lalu dikirim ke model dengan attention mask.
Pasang dependency tambahan:

```powershell
& $Py -m pip install -r "$Project\requirements-signbart.txt"
```

Untuk laptop NVIDIA/Windows, wheel PyTorch dari PyPI dapat terpasang sebagai
CPU-only. Pasang wheel CUDA resmi yang sesuai dengan driver mesin demo lalu
verifikasi `torch.cuda.is_available()` bernilai `True`. Konfigurasi laptop
pengembangan ini memakai wheel CUDA 11.8:

```powershell
& $Py -m pip install torch==2.7.1 `
  --index-url https://download.pytorch.org/whl/cu118 `
  --extra-index-url https://pypi.org/simple
```

Checkout upstream SignBart diletakkan sejajar dengan repository ini. Checkpoint
pengembangan WLASL-100 tersedia dari mirror Hugging Face milik author upstream;
gunakan pasangan artifact yang sudah didokumentasikan:

```powershell
$env:SIGN_HOST = "127.0.0.1"
$env:SIGN_PORT = "8765"
$env:SIGN_ALLOWED_ORIGINS = "http://localhost:5173"
$env:SIGN_ALLOW_PRIVATE_NETWORK = "false"
$env:SIGN_MODEL_BACKEND = "signbart"
$env:SIGN_MODEL_PATH = "$Project\models\SignBart-WLASL-100\model.safetensors"
$env:SIGN_MODEL_VERSION = "signbart-wlasl100-hf-8f98ae7"
$env:SIGN_VOCABULARY_PATH = "$Project\config\mvp-en-v1.json"
$env:SIGN_VOCABULARY_VERSION = "mvp-en-v1"
$env:SIGN_SIGNBART_REPO = "C:\Users\ASUS\.vscode\project\SignBart"
$env:SIGN_SIGNBART_CONFIG = "C:\Users\ASUS\.vscode\project\SignBart\configs\WLASL-100.yaml"
$env:SIGN_DEVICE = "cuda:0"
$env:SIGN_CONFIDENT_THRESHOLD = "0.80"
$env:SIGN_UNKNOWN_THRESHOLD = "0.40"
$env:SIGN_MIN_TOP1_TOP2_MARGIN = "0.20"
& $Py -m uvicorn src.sign.main:app --host 127.0.0.1 --port 8765
```

Konfigurasi model memakai `../SignBart/configs/WLASL-100.yaml`, bukan tabel
`id2label` pada config Hugging Face. `config/mvp-en-v1.json` mempertahankan
seluruh 100 indeks output checkpoint,
tetapi hanya mengekspos 20 label yang diaktifkan untuk MVP. Jangan menghapus atau
mengurutkan ulang indeks yang dinonaktifkan karena softmax dan output model tetap
berdimensi 100. Sumber, revision, checksum, catatan mapping, dan batasan validasi
artifact dicatat di `models/README.md`. Tanpa checkpoint, repo/config, PyTorch,
atau MediaPipe, readiness tetap `503 MODEL_NOT_READY`.

Sebelum mengaktifkan fitur sign, web client harus memanggil
`GET http://127.0.0.1:8765/health/ready`. Preflight browser dari origin yang
terdaftar hanya mengizinkan `Content-Type` dan `X-Request-ID`. Header
`Access-Control-Allow-Private-Network: true` hanya dikirim jika
`SIGN_ALLOW_PRIVATE_NETWORK=true` dan preflight meminta Private Network Access.

Contoh request:

```powershell
curl.exe -X POST http://127.0.0.1:8765/v1/sign/predict `
  -H "X-Request-ID: 00000000-0000-0000-0000-000000000001" `
  -F "video=@clip.webm;type=video/webm" `
  -F "topK=3" `
  -F "vocabularyVersion=mvp-en-v1"
```

Jika `INTERNAL_API_KEY` diisi, tambahkan header
`X-Internal-API-Key` pada request. Video hanya digunakan selama request dan
tidak disimpan oleh service secara default.

## Pengujian

```powershell
& $Py -m pip install -r "$Project\requirements-dev.txt"
& $Py -m pip check
& $Py -m pytest -q -p no:cacheprovider
```

Unit test mencakup frame sampling, confidence policy, dan konfigurasi.
Contract/smoke test menggunakan decoder dan model mock yang eksplisit sehingga
tidak memerlukan GPU atau checkpoint.

Modul STT, TTS, Gloss, dan Recall yang sudah ada tetap dipertahankan.
