# API Xác thực giọng nói (Qwen3 Voice Verification)

FastAPI service that generates 2048-dimensional speaker embeddings and performs
voice verification, backed by
[marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B](https://huggingface.co/marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B)
(an ECAPA-TDNN encoder, ~12.2M parameters, extracted from Qwen3-TTS).

Two use cases:
- **1:1 comparison** — do two recordings belong to the same speaker (`POST /compare`).
- **1:N verification** — enroll speakers, then check a new recording against an
  enrolled identity (`POST /speakers/enroll`, `POST /speakers/{speaker_id}/verify`).

Decision is a cosine similarity on a *calibrated* embedding (raw embedding minus a
background mean vector, then L2-normalized); "same speaker" when
`similarity >= threshold` (default `0.70`).

Designed to run as an offline edge-inference service on a trusted internal
network — **no authentication is enforced** on any endpoint.

## Prerequisites

- Python 3.11
- `ffmpeg` on the host/image, only required to decode `mp3`/`m4a` uploads
  (`wav`/`flac`/`ogg` decode natively via `libsndfile`)
- A CUDA GPU is optional; the service auto-detects and falls back to CPU

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The first run downloads and caches the pinned model revision from the Hugging
Face Hub. For a fully offline/air-gapped deployment, pre-populate
`~/.cache/huggingface/hub/models--marksverdhei--Qwen3-Voice-Embedding-12Hz-1.7B`
and set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` in `.env`.

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MODEL_ID` | No | `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B` | HF model repo |
| `MODEL_REVISION` | No | `7577f61c42737fc8064bba773e2a18602df92803` | Pinned commit for reproducibility |
| `DEVICE` | No | `auto` | `auto` \| `cpu` \| `cuda` |
| `HF_TOKEN` | No | — | Only for private/gated models |
| `HOST` | No | `0.0.0.0` | Bind address |
| `PORT` | No | `6422` | Bind port |
| `MIN_AUDIO_SECONDS` | No | `1` | Minimum accepted recording length |
| `MAX_UPLOAD_MB` | No | `25` | Maximum accepted upload size |
| `DEFAULT_THRESHOLD` | No | `0.70` | Default similarity threshold |
| `SPEAKER_DATA_DIR` | No | `data/speakers` | Where enrolled embeddings are stored |
| `CALIBRATION_PATH` | No | `calibration/mean_embedding.npy` | Background mean vector used to center scores |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | No | `0` | Set both to `1` for a fully offline deployment once the model is cached |

## Run locally

```bash
uvicorn main:app --host 0.0.0.0 --port 6422
```

`--host 0.0.0.0` is required to reach the service from other machines on the
network; without it, it only listens on localhost.

Check readiness:

```bash
curl http://localhost:6422/health
# {"status": "ok", "model": "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B", "device": "cuda"}
```

Interactive docs (Swagger UI): `http://localhost:6422/docs`

## Run with Docker

```bash
docker compose up --build
```

## Calibration

`calibration/mean_embedding.npy` ships as a zero vector (identity calibration —
no centering applied). Before relying on the default `0.70` threshold in
production, recalibrate against real recordings from your deployment
(≥ 5 recordings, ideally 20+ distinct speakers), then restart the service:

```bash
python scripts/calibrate.py --input-dir /path/to/background_audio
```

Enrolled speaker embeddings live under `data/speakers/` (`{speaker_id}.npy`,
one row per enrolled sample) — back this directory up if needed.

## Endpoint reference

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | — | `{"status", "model", "device"}` |
| POST | `/compare` | `file_a`, `file_b` (File), `threshold` (Text, optional) | `{"similarity", "threshold", "same_speaker"}` |
| POST | `/speakers/enroll` | `speaker_id` (Text), `file` (File) | `{"speaker_id", "samples"}` |
| POST | `/speakers/{speaker_id}/verify` | `file` (File), `threshold` (Text, optional) | `{"speaker_id", "similarity", "threshold", "verified", "samples"}` |
| GET | `/speakers` | — | `[{"speaker_id", "samples"}, ...]` |
| DELETE | `/speakers/{speaker_id}` | — | `{"speaker_id", "deleted"}` |
| POST | `/embed` | `file` (File), `raw` (Text, optional: `true`/`false`, default `false`) | `{"embedding", "dim", "duration_seconds"}` |

`speaker_id` must be 1–64 characters: letters, digits, `.`, `_`, `-`.

### curl examples

```bash
curl http://localhost:6422/health

curl -X POST http://localhost:6422/compare \
  -F "file_a=@a.wav" -F "file_b=@b.wav" -F "threshold=0.72"

curl -X POST http://localhost:6422/speakers/enroll \
  -F "speaker_id=nguyenvana" -F "file=@sample1.wav"

curl -X POST http://localhost:6422/speakers/nguyenvana/verify \
  -F "file=@probe.wav"

curl http://localhost:6422/speakers

curl -X DELETE http://localhost:6422/speakers/nguyenvana

curl -X POST http://localhost:6422/embed \
  -F "file=@sample.wav" -F "raw=false"
```

## Error codes

| HTTP | Meaning | Resolution |
|---|---|---|
| 400 | Cannot decode audio / empty file | Check file format (native: wav/flac/ogg; mp3/m4a need ffmpeg) |
| 404 | Speaker not enrolled | Enroll first via `/speakers/enroll` |
| 413 | File exceeds `MAX_UPLOAD_MB` (25 MB default) | Trim the recording |
| 422 | Audio shorter than `MIN_AUDIO_SECONDS` (1s), invalid `speaker_id`, or invalid `threshold` | Use a ≥1s recording; use a valid `speaker_id`/`threshold` |
| 500 | Server error | Check the service's terminal/log output |

## Project layout

```
main.py                          FastAPI app entrypoint + lifespan model warm-up
routes/voice.py                  /embed, /compare, /speakers... endpoints
services/model.py                Model + processor loading, embedding extraction
services/calibration.py          Mean-centering + cosine similarity
services/audio.py                Upload validation & audio decoding
services/store.py                Enrolled-speaker persistence (data/speakers/)
calibration/mean_embedding.npy   Background mean vector used to center scores
scripts/calibrate.py             Recompute the calibration vector from real audio
schemas/voice.py                 Pydantic request/response models
```
