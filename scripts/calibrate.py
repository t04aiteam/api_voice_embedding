#!/usr/bin/env python
"""Recompute calibration/mean_embedding.npy from a folder of background audio.

Usage:
    python scripts/calibrate.py --input-dir /path/to/background_audio

Recommended: >= 5 recordings, ideally 20+ distinct speakers, covering the
range of voices/recording conditions this deployment will see. Restart the
service after recalibrating so it picks up the new vector.
"""

import argparse
import os
import sys

import librosa
import numpy as np
from dotenv import load_dotenv

# Same settings as the server (MODEL_ID, MODEL_REVISION, DEVICE, ...), otherwise
# the calibration vector could come from a different model than the one serving.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services import model as model_service  # noqa: E402

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "..", "calibration", "mean_embedding.npy"
)


def iter_audio_files(input_dir: str):
    for root, _, files in sorted(os.walk(input_dir)):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                yield os.path.join(root, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory of background audio files (recursively scanned).",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-files",
        type=int,
        default=5,
        help="Minimum number of files required to proceed (default: 5).",
    )
    args = parser.parse_args()

    files = list(iter_audio_files(args.input_dir))
    if len(files) < args.min_files:
        parser.error(
            f"Found only {len(files)} audio files in {args.input_dir}; "
            f"at least {args.min_files} are required."
        )

    embeddings = []
    for path in files:
        waveform, sample_rate = librosa.load(path, sr=None, mono=True)
        embeddings.append(model_service.extract_embedding(waveform, sample_rate))
        print(f"embedded {path}")

    mean_embedding = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, mean_embedding)
    print(
        f"Saved calibration mean embedding ({mean_embedding.shape[0]}-d) "
        f"from {len(files)} files to {args.output}"
    )


if __name__ == "__main__":
    main()
