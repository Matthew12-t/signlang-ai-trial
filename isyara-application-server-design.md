# Isyara Application Server — Design and API Contract

Status: Draft v0.2  
Target: MVP Hackathon IFEST 2026  
Implementasi acuan: Python REST API  
Audience: tim software engineering dan tim AI

## 1. Posisi Dokumen

Dokumen ini merupakan turunan dari `isyara-ai-services-design.md`. Jika terdapat perbedaan:

1. konteks proyek Isyara tetap menjadi sumber keputusan produk;
2. `isyara-ai-services-design.md` menjadi sumber kebenaran batas antarlayanan AI;
3. dokumen ini menjadi sumber kebenaran perilaku Application Server;
4. `isyara-application-server-openapi.yaml` menjadi sumber kebenaran kontrak HTTP publik yang dapat dibaca mesin;
5. `isyara-ai-services-openapi.yaml` menjadi sumber kebenaran kontrak HTTP internal menuju AI services.

Perubahan schema lintas-repo harus memperbarui kedua OpenAPI dalam pull request atau perubahan terkoordinasi yang sama.

## 2. Tanggung Jawab

Application Server adalah orchestration layer antara web client, penyimpanan session, dan AI services yang berjalan di sisi server. Sign Language Service merupakan pengecualian: layanan tersebut berjalan di laptop penanda dan dipanggil langsung oleh web client pada laptop yang sama.

Application Server bertanggung jawab atas:

- lifecycle session;
- lifecycle sign utterance;
- menerima dan memvalidasi laporan prediction dari Sign Language Service lokal;
- menyimpan prediction sementara dan token yang dikonfirmasi;
- meneruskan audio chunk ke STT Service;
- menyimpan hanya transcript STT final;
- memfinalisasi sign utterance melalui Gloss Service;
- memanggil TTS untuk teks final/terkonfirmasi;
- memilih bounded transcript context untuk Recall Service;
- memvalidasi referensi evidence Recall sekali lagi;
- normalisasi error, timeout, retry, request ID, dan idempotency;
- mengisolasi kegagalan setiap service.

Application Server tidak bertanggung jawab atas:

- inferensi SignBart;
- menerima atau meneruskan video sign;
- memverifikasi secara kriptografis bahwa laporan prediction benar-benar berasal dari model lokal pada MVP;
- inferensi STT/TTS;
- prompt dan inferensi Qwen;
- threshold confidence model;
- akses langsung ke Hugging Face;
- penyimpanan video atau audio mentah secara permanen.

## 3. Arsitektur

```mermaid
flowchart LR
    CAMERA[Kamera laptop penanda] --> WEB[Web Client penanda]
    WEB -->|video lokal, REST loopback| SIGN[Sign Service lokal\nSignBart + RTX 3060]
    SIGN -->|prediction| WEB
    WEB -->|prediction report, confirm, finalize| APP[Python Application Server]
    VIEWER[Web Client laptop lain] -->|REST /v1| APP
    APP --> DB[(Session and Transcript Store)]
    APP -->|REST internal| STT[STT Service]
    APP -->|REST internal| GLOSS[Gloss Service]
    APP -->|REST internal| TTS[TTS Service]
    APP -->|REST internal| RECALL[Recall Service]
```

Topologi demo minimum:

| Node | Proses wajib | Koneksi |
|---|---|---|
| Laptop A — penanda | browser + Sign Language Service + GPU | loopback ke Sign Service; jaringan ke Application Server |
| Laptop B — peserta | browser | jaringan ke Application Server; tidak mengakses Sign Language Service Laptop A |
| Host bersama | Application Server + database; AI services dapat berada di host server AI lain | dapat dijangkau kedua browser |

Application Server tidak ditempatkan pada laptop penanda sebagai dependency arsitektural. Kedua laptop menggunakan `APP_PUBLIC_BASE_URL` yang sama agar session, token terkonfirmasi, transcript final, dan hasil Recall memiliki sumber kebenaran bersama.

Web client mengenal dua origin: base URL Application Server bersama dan `http://127.0.0.1:8765` untuk Sign Language Service lokal. Hanya web client pada laptop penanda yang memanggil origin lokal tersebut. URL STT, TTS, Gloss, Recall, API key internal, token provider, dan prompt tidak boleh dikirim ke browser.

Video sign tidak pernah dikirim ke Application Server. Batas kepercayaan MVP berakhir pada web client: Application Server dapat memvalidasi bentuk dan konsistensi laporan prediction, tetapi belum dapat membuktikan bahwa prediction benar-benar dihasilkan oleh model lokal. Bukti prediction bertanda tangan atau registrasi perangkat berada di luar ruang lingkup demo.

Kontrak MVP tidak mewajibkan autentikasi pengguna karena targetnya demo lokal. CORS tetap dibatasi ke origin frontend. Autentikasi publik dapat ditambahkan kemudian tanpa mengubah kontrak internal antarlayanan.

## 4. Resource dan State

### 4.1 Session

```text
ACTIVE → ENDED
```

Session menyimpan:

- `sessionId`;
- `status`;
- `language`;
- `createdAt` dan `endedAt`;
- kumpulan transcript entry;
- utterance yang masih aktif atau selesai.

Hanya session `ACTIVE` yang dapat menerima audio, laporan sign prediction, token baru, atau pertanyaan Recall.

### 4.2 Sign utterance

```text
CAPTURING → FINALIZED
     └────→ CANCELLED
```

Aturan:

- maksimum satu utterance `CAPTURING` per session;
- prediction belum menjadi token;
- token hanya dibuat melalui tindakan konfirmasi pengguna;
- utterance tanpa token tidak dapat difinalisasi;
- utterance `FINALIZED` dan `CANCELLED` bersifat immutable;
- finalisasi yang berhasil membuat tepat satu transcript entry `source=sign`.

### 4.3 Prediction

Prediction yang dilaporkan web client disimpan sementara agar konfirmasi pengguna dapat diverifikasi.

```text
PENDING_CONFIRMATION → ACCEPTED
                    └→ REJECTED
                    └→ EXPIRED
```

`selectedLabel` harus merupakan label yang terdapat dalam `candidates`. Jika pengguna ingin mengulang sign, prediction lama ditolak dan klip baru dikirim.

### 4.4 Transcript entry

Transcript entry bersifat append-only selama session aktif.

```json
{
  "id": "tr_17",
  "sessionId": "ses_01J8Y6Z4Q1",
  "sequence": 17,
  "source": "speech",
  "speaker": "Participant",
  "text": "The submission deadline is Friday at five PM.",
  "startedAt": "2026-09-18T10:18:11Z",
  "endedAt": "2026-09-18T10:18:14Z",
  "isFinal": true
}
```

Hanya entry `isFinal=true` yang boleh dikirim ke Recall Service.

## 5. Endpoint Publik Frontend

### 5.1 Session

| Method dan path | Fungsi |
|---|---|
| `POST /v1/sessions` | membuat session aktif |
| `GET /v1/sessions/{sessionId}` | membaca status session |
| `DELETE /v1/sessions/{sessionId}` | mengakhiri session dan menghapus data sesuai kebijakan MVP |

### 5.2 Sign-to-Text

| Method dan path | Fungsi |
|---|---|
| `POST /v1/sessions/{sessionId}/utterances` | Start Sign |
| `GET /v1/sessions/{sessionId}/utterances/active` | membaca utterance aktif dan token terkonfirmasi untuk sinkronisasi antarlaptop |
| `POST /v1/utterances/{utteranceId}/predictions` | mendaftarkan hasil prediction dari Sign Language Service lokal |
| `POST /v1/utterances/{utteranceId}/tokens` | konfirmasi satu candidate sebagai kata |
| `POST /v1/utterances/{utteranceId}:finalize` | Stop Sign dan buat transcript entry |
| `DELETE /v1/utterances/{utteranceId}` | batalkan utterance aktif |

Prediction endpoint menerima `application/json`. Payload berasal dari response `POST http://127.0.0.1:8765/v1/sign/predict` yang telah dipetakan oleh web client:

```json
{
  "localRequestId": "c946f42d-474e-40f4-a6d9-b0eb75f05a9a",
  "predictionId": "pred_01J8Y7RK2F",
  "prediction": "REPEAT",
  "confidence": 0.86,
  "candidates": [
    {"label": "REPEAT", "confidence": 0.86},
    {"label": "AGAIN", "confidence": 0.09}
  ],
  "status": "CONFIDENT",
  "requiresConfirmation": false,
  "modelVersion": "signbart-mvp-v1",
  "vocabularyVersion": "mvp-en-v1",
  "latencyMs": 184,
  "capturedAt": "2026-09-18T10:15:22Z"
}
```

Application Server tidak memanggil Sign Language Service. Server memvalidasi bahwa utterance masih `CAPTURING`, `predictionId` belum pernah dipakai, `vocabularyVersion` sesuai dengan utterance, kandidat unik dan terurut, confidence berada pada rentang `0–1`, serta status konsisten dengan `prediction` dan `requiresConfirmation`. Server lalu menyimpan prediction dengan waktu penerimaan server. `localRequestId` hanya digunakan untuk korelasi; `X-Request-ID` pada request Application Server tetap merupakan ID request publik yang berbeda.

Hanya prediction berstatus `CONFIDENT` atau `AMBIGUOUS` yang dapat dikonfirmasi. `selectedLabel` harus ada di `candidates`. Status `UNKNOWN`, `NO_SIGN`, dan `INVALID_INPUT` tidak boleh menghasilkan token.

Untuk demo dua laptop, web client lain melakukan polling `GET /v1/sessions/{sessionId}/utterances/active` setiap 300–500 ms selama mode sign aktif. Endpoint tersebut hanya menampilkan token terkonfirmasi; prediction mentah tidak disebarkan ke peserta lain. WebSocket dapat ditambahkan kemudian tanpa mengubah state machine.

Konfirmasi token:

```json
{
  "predictionId": "pred_01J8Y7RK2F",
  "selectedLabel": "REPEAT"
}
```

Response:

```json
{
  "token": {
    "id": "tok_01J8Y7T1Z8",
    "predictionId": "pred_01J8Y7RK2F",
    "label": "REPEAT",
    "position": 2,
    "confirmedAt": "2026-09-18T10:15:23Z"
  },
  "utteranceStatus": "CAPTURING"
}
```

### 5.3 Speech-to-Text

| Method dan path | Fungsi |
|---|---|
| `POST /v1/sessions/{sessionId}/stt/chunks` | mengirim satu audio chunk |

Request multipart membawa:

```text
audio: binary
streamId: stt_01J8Y8B1CA
sequence: 4
commit: false
language: en
encoding: webm
startedAt: 2026-09-18T10:18:11Z
```

Aturan:

- kombinasi `streamId + sequence` unik dan idempotent;
- `commit=false` menghasilkan partial provisional dan tidak membuat transcript entry;
- `commit=true` menghasilkan final, melakukan deduplikasi overlap, lalu membuat transcript entry;
- sequence yang lebih lama tidak boleh menimpa revision yang lebih baru;
- detail buffering dan model dimiliki tim STT.

### 5.4 TTS

| Method dan path | Fungsi |
|---|---|
| `POST /v1/utterances/{utteranceId}:speak` | membuat audio dari utterance sign final |

Application Server mengambil teks final dari store; frontend tidak mengirim teks bebas pada endpoint ini. Hal ini mencegah prediction ambigu dibacakan sebelum konfirmasi.

Response berupa `audio/wav` atau `audio/flac`. Jika TTS gagal, transcript tetap tersedia dan response error tidak mengubah status utterance.

### 5.5 Transcript dan Recall

| Method dan path | Fungsi |
|---|---|
| `GET /v1/sessions/{sessionId}/transcript` | membaca transcript final berurutan |
| `POST /v1/sessions/{sessionId}/recall` | mengajukan pertanyaan berbasis transcript |

Frontend mengirim pertanyaan dan opsi rentang, bukan context mentah:

```json
{
  "query": "When is the deadline?",
  "range": {
    "mode": "recent",
    "lastSeconds": 300
  },
  "language": "en"
}
```

Application Server:

1. mengambil transcript final sesuai range;
2. mengurutkan berdasarkan `sequence`;
3. memotong entry tertua jika melewati batas konteks;
4. memanggil Recall Service dengan `contextEntries` eksplisit;
5. memastikan setiap evidence `entryId` ada pada context yang dikirim;
6. mengembalikan jawaban, evidence, dan `grounded` kepada frontend.

Jika context kosong, Application Server langsung mengembalikan `grounded=false` dengan `notFoundReason=EMPTY_CONTEXT` tanpa memanggil Recall Service.

## 6. Pemetaan ke Internal AI Services

| Public operation | Internal call | Data yang disimpan |
|---|---|---|
| laporkan prediction lokal | tidak ada | prediction sementara; bukan transcript |
| konfirmasi kata | tidak ada | confirmed token |
| finalisasi utterance | `POST /v1/gloss/normalize` | satu transcript entry `source=sign` |
| kirim audio chunk | `POST /v1/stt/transcriptions` | hanya final transcript entry |
| speak utterance | `POST /v1/tts/synthesize` | tidak menyimpan audio secara default |
| Recall | `POST /v1/recall/query` | opsional: pertanyaan dan jawaban; transcript tidak berubah |

## 7. Fungsi Aplikasi Python

Framework acuan dapat menggunakan FastAPI dan Pydantic, tetapi kontrak HTTP tidak bergantung pada framework tersebut.

```python
async def create_session(command: CreateSession) -> Session: ...

async def create_sign_utterance(
    session_id: str,
    command: CreateUtterance,
) -> SignUtterance: ...

async def register_sign_prediction(
    utterance_id: str,
    report: SignPredictionReport,
) -> SignPrediction: ...

async def confirm_sign_token(
    utterance_id: str,
    command: ConfirmSignToken,
) -> ConfirmedSignToken: ...

async def finalize_sign_utterance(
    utterance_id: str,
) -> FinalizedUtterance: ...

async def transcribe_stt_chunk(
    session_id: str,
    audio: UploadFile,
    command: STTChunkCommand,
) -> STTChunkResult: ...

async def synthesize_utterance(
    utterance_id: str,
) -> AudioStream: ...

async def answer_session_recall(
    session_id: str,
    command: RecallCommand,
) -> RecallAnswer: ...
```

Internal clients:

```python
class STTServiceClient(Protocol):
    async def transcribe(self, audio: UploadFile, options: STTOptions) -> Transcript: ...

class GlossServiceClient(Protocol):
    async def normalize(self, tokens: list[ConfirmedSignToken]) -> NormalizedUtterance: ...

class TTSServiceClient(Protocol):
    async def synthesize(self, text: str, options: TTSOptions) -> AudioStream: ...

class RecallServiceClient(Protocol):
    async def query(self, query: str, context: list[TranscriptEntry]) -> RecallAnswer: ...
```

## 8. Persistence MVP

Pilihan acuan: SQLite untuk demo lokal atau PostgreSQL jika deployment bersama sudah tersedia. Domain model tidak boleh bergantung pada vendor database.

Tabel/collection minimum:

- `sessions`;
- `sign_utterances`;
- `sign_predictions`;
- `confirmed_sign_tokens`;
- `transcript_entries`;
- `idempotency_records`.

Audio STT hanya hidup selama request berlangsung. Application Server tidak menerima video sign. Jika debugging audio diaktifkan, gunakan direktori sementara, TTL singkat, dan feature flag yang nilai default-nya `false`.

## 9. Transaksi dan Idempotency

- `POST /sessions`, registrasi prediction, token confirmation, finalization, TTS, dan Recall menerima `Idempotency-Key`.
- Request dengan key dan payload yang sama mengembalikan hasil sebelumnya.
- Key yang sama dengan payload berbeda menghasilkan `409 IDEMPOTENCY_CONFLICT`.
- Finalisasi utterance dan pembuatan transcript entry berlangsung dalam satu transaksi.
- Jika Gloss Service berhasil tetapi commit database gagal, retry finalisasi tidak boleh membuat entry ganda.
- `sequence` transcript dialokasikan oleh Application Server, bukan AI service.

## 10. Timeout dan Retry

| Internal call | Timeout total | Retry |
|---|---:|---:|
| STT partial | 6 detik | 0 |
| STT final | 10 detik | 1 |
| Gloss template | 1 detik | 0 |
| TTS | 12 detik | 1 |
| Recall | 15 detik | 1 |

Retry hanya untuk timeout, `429`, dan `5xx` yang dinyatakan retryable. Jangan retry validation error. Retry registrasi prediction dengan `Idempotency-Key` yang sama harus mengembalikan hasil sebelumnya tanpa membuat prediction ganda.

## 11. Error Publik

```json
{
  "error": {
    "code": "PREDICTION_REPORT_INVALID",
    "message": "The reported sign prediction is inconsistent.",
    "retryable": false,
    "requestId": "c946f42d-474e-40f4-a6d9-b0eb75f05a9a",
    "details": {}
  }
}
```

Application Server tidak membocorkan nama provider, stack trace, URL internal, atau credential. Error internal dinormalisasi:

| Internal condition | Public code |
|---|---|
| STT timeout/503 | `STT_SERVICE_UNAVAILABLE` |
| TTS timeout/503 | `TTS_SERVICE_UNAVAILABLE` |
| Recall timeout/503 | `RECALL_SERVICE_UNAVAILABLE` |
| prediction ID tidak dikenal/kedaluwarsa | `PREDICTION_NOT_AVAILABLE` |
| laporan prediction tidak konsisten | `PREDICTION_REPORT_INVALID` |
| versi vocabulary berbeda dari utterance | `VOCABULARY_VERSION_MISMATCH` |
| utterance sudah final | `UTTERANCE_ALREADY_FINALIZED` |
| session sudah berakhir | `SESSION_ENDED` |

## 12. Environment Variables

```dotenv
APP_HOST=0.0.0.0
APP_PORT=8000
DATABASE_URL=sqlite:///./isyara.db
CORS_ALLOWED_ORIGINS=http://localhost:5173
APP_PUBLIC_BASE_URL=http://APP_SERVER_HOST:8000
MAX_AUDIO_CHUNK_BYTES=10000000
RECALL_MAX_CONTEXT_CHARS=24000
SIGN_PREDICTION_MAX_CANDIDATES=5
SIGN_PREDICTION_TTL_SECONDS=120
SIGN_ALLOWED_VOCABULARY_VERSIONS=mvp-en-v1

STT_SERVICE_URL=http://127.0.0.1:8001
GLOSS_SERVICE_URL=http://127.0.0.1:8002
TTS_SERVICE_URL=http://127.0.0.1:8003
RECALL_SERVICE_URL=http://127.0.0.1:8004
INTERNAL_API_KEY=
```

Service URL harus dapat berbeda tanpa mengubah route publik.

## 13. Health dan Degraded Mode

```text
GET /health/live
GET /health/ready
```

Readiness Application Server menampilkan status dependency tanpa credential:

```json
{
  "status": "degraded",
  "service": "application-api",
  "dependencies": {
    "database": "ready",
    "stt": "unavailable",
    "tts": "ready",
    "gloss": "ready",
    "recall": "ready"
  }
}
```

`degraded` tetap dapat mengembalikan HTTP `200` untuk demo jika core page dan sebagian fitur dapat digunakan. `503` hanya ketika database atau dependency yang ditetapkan sebagai wajib tidak siap.

## 14. Observability

Setiap request publik memperoleh `X-Request-ID`; ID yang sama diteruskan ke service internal yang dipanggil Application Server. Catat:

- route dan status;
- latency total;
- latency setiap dependency;
- ukuran payload;
- session ID dan utterance ID yang telah di-hash atau diperlakukan sebagai identifier teknis;
- error code terstandardisasi.

Untuk laporan sign, catat `localRequestId`, `predictionId`, versi model/vocabulary, status, latency inferensi yang dilaporkan, dan latency registrasi. Nilai dari perangkat klien harus diberi label sebagai data yang dilaporkan klien, bukan metrik server tepercaya.

Jangan catat media mentah, secret, atau transcript lengkap secara default.

## 15. Acceptance Criteria

- Web client memanggil Application Server untuk state bersama dan Sign Language Service melalui loopback hanya pada laptop penanda.
- Application Server dapat dijalankan dengan mock AI clients berdasarkan kedua OpenAPI.
- Video sign tidak mencapai Application Server.
- Satu klip sign menghasilkan satu prediction lokal; laporan prediction tidak langsung menjadi transcript.
- Hanya candidate yang dikonfirmasi yang menjadi token.
- Laptop kedua dapat membaca token terkonfirmasi dari utterance aktif melalui REST.
- Finalisasi sign menghasilkan tepat satu transcript entry.
- Partial STT tidak disimpan.
- Recall hanya menerima transcript final yang dipilih Application Server.
- Evidence Recall divalidasi terhadap context yang benar-benar dikirim.
- Kegagalan STT/TTS/Recall tidak menghapus session atau transcript.
- URL dan credential internal tidak muncul pada response frontend.

## 16. Pengujian Kontrak Minimum

1. Buat session → buat utterance → laporkan prediction lokal → konfirmasi → finalisasi → transcript berisi satu entry sign.
2. Pastikan request registrasi prediction hanya berisi JSON dan tidak menerima video/multipart.
3. Prediction ambigu tidak masuk transcript sebelum konfirmasi.
4. Prediction yang ditolak tidak dapat dikonfirmasi kembali.
5. `UNKNOWN`, `NO_SIGN`, dan `INVALID_INPUT` tidak dapat dikonfirmasi sebagai token.
6. Laptop kedua membaca token terbaru melalui endpoint utterance aktif.
7. Partial STT tidak menambah transcript; commit menambah tepat satu entry.
8. TTS gagal, tetapi utterance final dan transcript tetap ada.
9. Recall tidak menemukan jawaban dan mengembalikan `grounded=false`.
10. Retry dengan `Idempotency-Key` yang sama tidak membuat data ganda.
11. Sign Language Service lokal mati dan web client penanda menampilkan error tanpa mengubah health Application Server.
