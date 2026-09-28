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
- `ffmpeg` on the host/image, required for every upload `libsndfile` cannot
  open: video (`mp4`/`mov`/`mkv`/`webm`, the audio track is used) and
  `m4a`/`aac`/`wma`. `wav`/`flac`/`ogg`/`mp3` decode in memory without it.
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
| `MAX_AUDIO_SECONDS` | No | `0` | Optional maximum recording length in seconds, `0` (default) means no cap (see Memory) |
| `MAX_UPLOAD_MB` | No | `25` | Maximum accepted upload size |
| `CPU_FALLBACK_ON_OOM` | No | `1` | After a CUDA OOM and one retry, run the request on a CPU copy of the model instead of failing |
| `PYTORCH_CUDA_ALLOC_CONF` | No | unset | Optional torch CUDA allocator config, passed through as is; the service does not set it |
| `LOG_LEVEL` | No | `INFO` | Python logging level |
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

The threshold only makes sense together with the calibration state. Measured
with `evals/threshold_eval.py` on VIVOS (19 test speakers, 380 clips of 3 to
10 s, CPU, `MODEL_REVISION` above):

| calibration | score range, same speaker | score range, different speakers | 0.70 | 0.96 | best cut (EER) |
|---|---|---|---|---|---|
| zero vector as shipped | 0.97 to 1.00 | 0.90 to 0.99 | accepts every pair | 0 % false reject, 38 % false accept | 0.98, 3.5 % |
| mean of 12 other speakers | 0.33 to 0.94 | -0.38 to 0.79 | 17.5 % false reject, 1.4 % false accept | rejects every pair | 0.63, 9.6 % |

So `0.70` (README, manual) assumes `scripts/calibrate.py` has been run, and
`0.96` (the code and `.env.example` default) is a raw-score threshold for the
shipped zero vector. Pick one pair and keep `DEFAULT_THRESHOLD` and the
calibration file in step.

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
| 400 | Cannot decode audio / empty file / video without an audio track | Check the file has an audio track (native: wav/flac/ogg/mp3; video and m4a need ffmpeg on the host) |
| 404 | Speaker not enrolled | Enroll first via `/speakers/enroll` |
| 413 | File exceeds `MAX_UPLOAD_MB` (25 MB default) | Trim the recording |
| 422 | Audio shorter than `MIN_AUDIO_SECONDS` (1s), longer than `MAX_AUDIO_SECONDS` when that cap is set, invalid `speaker_id`, or invalid `threshold` | Use a recording of at least 1s; use a valid `speaker_id`/`threshold` |
| 500 | Server error | Check the service's terminal/log output. If `detail` says "out of memory", retry later, send a shorter recording, or free GPU memory held by other processes; otherwise the response only names the exception type |

## Memory

Activation memory grows linearly with recording length: measured **7.6 MiB per
second of audio** (fp32, CPU; the GPU figure is the same order), so a 6-minute
song needs about 2.9 GB on top of the ~50 MB model. `/compare` embeds its two
files one after the other, so its peak is the longer file, not the sum.

When the GPU is shared with other jobs the free memory can drop to a few
hundred MiB at any moment. The service handles that as follows:

1. Inference is serialized (one forward pass at a time), so concurrent requests
   queue instead of multiplying peak memory.
2. On CUDA OOM the caching allocator is flushed and the forward pass retried once.
3. If it still fails and `CPU_FALLBACK_ON_OOM=1` (default), the request runs on
   a CPU copy of the model (loaded once from the local model cache, never
   deep-copied through the GPU, ~50 MB RAM; a 6-minute file takes ~2.6 s on CPU).
4. Before any CPU pass (fallback or `DEVICE=cpu`) the activation memory is
   estimated from the processor output (160 bytes per input byte, measured)
   and compared with the smaller of `MemAvailable` from `/proc/meminfo` and
   the process's cgroup v2 cap (`memory.max - memory.current`: a docker
   `mem_limit` or a systemd `MemoryMax=`, which the kernel enforces with
   SIGKILL). A recording that would not fit is refused as an out-of-memory
   500 instead of driving the host into swap and the kernel OOM killer.
   Skipped where neither file exists.
5. Otherwise the client gets **500** with a `detail` that starts with
   `Server out of memory on <device>` (the HDSD manual documents 500 as the
   only server error). A CPU (host RAM) exhaustion maps the same way, whether
   it hits the forward pass, the mel feature extraction (`torch.stft`) or the
   decoder (numpy `MemoryError` in `librosa.load`); none of them is reported
   as a 400.

Measured on 222 with `evals/eval_oom_forced.py` (CPU, the 378 s test song,
the process alone limited with `RLIMIT_AS`, four concurrent requests): every
request that cannot get memory is a 500 whose detail says out of memory, the
short requests in between still answer 200, and the same long file answers
200 once the limit is lifted. With a 3 GB cgroup cap and 80 GB host
`MemAvailable` the guard refuses the song in 2.2 s; before the cgroup read the
kernel killed the process (exit 137) and the client got an empty reply.

`MAX_AUDIO_SECONDS` is off by default (`0`), matching the manual where the 25 MB
upload size is the only limit. Set it to e.g. `600` to reject longer recordings
with a 422 before any inference runs; the decoder then stops reading one second
past the cap, so an over-long upload is not decoded in full first.

## Tests and evals

```bash
# Gate tests (no model, no GPU, ~3 s)
python -m unittest discover -s tests -v

# Memory-pressure eval against a running service (real model):
# N concurrent /compare calls with a long recording; every response must be
# 200 or an "out of memory" 500, never any other error, and at least
# --min-success of them 200.
python evals/eval_oom.py /path/to/long.mp3 --url http://localhost:6422 --concurrency 4 --rounds 2

# Forced host OOM (real model, starts its own CPU instance on --port, Linux
# only): warms up, caps the instance's address space with prlimit, and checks
# that a long file is an "out of memory" 500 while short files still get 200,
# under 4 concurrent requests too, and that the long file passes once the cap
# is lifted. Nothing outside that one process is limited.
python evals/eval_oom_forced.py --long /path/to/long.mp3 --short /path/to/short.wav --port 6423

# Threshold eval (real model, no service): false accept / false reject at each
# threshold over a folder with one sub-folder per speaker, with the shipped
# calibration file or with a background mean computed from --calibrate-from.
python evals/threshold_eval.py --audio-dir vivos/test/waves --thresholds 0.70 0.96
python evals/threshold_eval.py --audio-dir vivos/test/waves --calibrate-from vivos/train/waves
```

## Project layout

```
main.py                          FastAPI app entrypoint, lifespan model warm-up, error handlers
routes/voice.py                  /embed, /compare, /speakers... endpoints
services/model.py                Model + processor loading, embedding extraction, OOM retry/fallback
services/calibration.py          Mean-centering + cosine similarity
services/audio.py                Upload validation & audio decoding (length caps)
services/store.py                Enrolled-speaker persistence (data/speakers/)
calibration/mean_embedding.npy   Background mean vector used to center scores
scripts/calibrate.py             Recompute the calibration vector from real audio
schemas/voice.py                 Pydantic request/response models
tests/test_oom.py                Gate tests for OOM retry, CPU fallback, 500 mapping, optional length cap
evals/eval_oom.py                Memory-pressure eval against a live service
evals/eval_oom_forced.py         Forces a host OOM on a throwaway CPU instance (RLIMIT_AS) and checks the 500 path
evals/threshold_eval.py          False accept / false reject per threshold over a speaker-labelled folder
```
