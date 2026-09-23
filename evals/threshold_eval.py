"""Score same-speaker and different-speaker pairs and compare decision thresholds.

Runs the service's own decode -> embed -> calibrate -> cosine chain (no HTTP)
over a speaker-labelled folder (one sub-folder per speaker, VIVOS layout) and
reports, for each candidate threshold, the false accept rate (different
speakers judged the same) and the false reject rate (same speaker judged
different), plus the equal-error-rate threshold for this data.

Usage (CPU is enough, 380 short files take about 20 s):
    DEVICE=cpu emb-env/bin/python evals/threshold_eval.py --audio-dir vivos/test/waves \
        --per-speaker 20 --thresholds 0.70 0.96 --json out.json
    # same, but scored the way scripts/calibrate.py would leave the service:
    ... --calibrate-from vivos/train/waves

Speaker folders: VIVOS (https://huggingface.co/datasets/AILAB-VNUHCM/vivos)
has one folder per speaker; any folder laid out that way works.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO, ".env"), override=False)

from services import audio as audio_service  # noqa: E402
from services import calibration as calibration_service  # noqa: E402
from services import model as model_service  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def collect(
    audio_dir: str, per_speaker: int, max_speakers: int
) -> dict[str, list[str]]:
    """Map speaker id (sub-folder name) to its first N audio files.

    Args:
        audio_dir: Folder with one sub-folder per speaker.
        per_speaker: Files kept per speaker.
        max_speakers: Speakers kept, 0 for all.

    Returns:
        Ordered mapping of speaker id to file paths.
    """
    speakers: dict[str, list[str]] = {}
    for name in sorted(os.listdir(audio_dir)):
        folder = os.path.join(audio_dir, name)
        if not os.path.isdir(folder):
            continue
        files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in AUDIO_EXT
        )
        if files:
            speakers[name] = files[:per_speaker]
        if max_speakers and len(speakers) >= max_speakers:
            break
    return speakers


def embed_file(path: str) -> np.ndarray:
    """Decode and embed one file exactly as the /compare route does, before centering.

    Args:
        path: Audio file path.

    Returns:
        Raw (uncalibrated) embedding.
    """
    with open(path, "rb") as f:
        raw = f.read()
    waveform, sample_rate, _ = audio_service._decode(raw)
    return model_service.extract_embedding(waveform, sample_rate)


def center(raw: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Apply the service's calibration step with an explicit background mean.

    Args:
        raw: Raw embedding.
        mean: Background mean vector (zeros = the identity calibration in the repo).

    Returns:
        Centered, L2-normalised embedding.
    """
    centered = raw - mean
    norm = np.linalg.norm(centered)
    if norm > 1e-8:
        centered = centered / norm
    return centered.astype(np.float32)


def rates(same: np.ndarray, diff: np.ndarray, thr: float) -> dict:
    """Compute accept/reject rates at one threshold.

    Args:
        same: Scores of same-speaker pairs.
        diff: Scores of different-speaker pairs.
        thr: Decision threshold (same_speaker = score >= thr).

    Returns:
        Rates and counts for this threshold.
    """
    fr = int((same < thr).sum())
    fa = int((diff >= thr).sum())
    return {
        "threshold": thr,
        "false_reject": fr,
        "false_reject_rate": round(fr / len(same), 4),
        "false_accept": fa,
        "false_accept_rate": round(fa / len(diff), 4),
    }


def eer(same: np.ndarray, diff: np.ndarray) -> tuple[float, float]:
    """Find the threshold where false accept and false reject rates cross.

    Args:
        same: Scores of same-speaker pairs.
        diff: Scores of different-speaker pairs.

    Returns:
        Threshold and the error rate at that threshold.
    """
    best = (1.0, 1.0)
    for thr in np.linspace(0.0, 1.0, 1001):
        frr = float((same < thr).mean())
        far = float((diff >= thr).mean())
        gap = abs(frr - far)
        if gap < best[1]:
            best = (float(thr), gap)
            if gap == 0:
                break
    thr = best[0]
    return round(thr, 3), round(
        float(((same < thr).mean() + (diff >= thr).mean()) / 2), 4
    )


def main() -> None:
    """Embed the files, score all pairs, print and save the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--per-speaker", type=int, default=10)
    parser.add_argument("--max-speakers", type=int, default=0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.70, 0.96])
    parser.add_argument(
        "--calibrate-from",
        default=None,
        help="Speaker folder used as background: its mean raw embedding replaces "
        "calibration/mean_embedding.npy (what scripts/calibrate.py would write).",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    for name in ("audio_dir", "calibrate_from", "json"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    os.chdir(REPO)  # CALIBRATION_PATH defaults to a repo-relative path

    mean = calibration_service.get_mean_embedding()
    calibration = (
        "repo file (all zeros = identity)" if not np.any(mean) else "repo file"
    )
    if args.calibrate_from:
        bg = collect(args.calibrate_from, args.per_speaker, args.max_speakers)
        bg_vecs = [embed_file(p) for files in bg.values() for p in files]
        mean = np.mean(np.stack(bg_vecs), axis=0)
        calibration = f"mean of {len(bg_vecs)} files from {len(bg)} speakers in {args.calibrate_from}"
    print(f"calibration: {calibration}")

    speakers = collect(args.audio_dir, args.per_speaker, args.max_speakers)
    n_files = sum(len(v) for v in speakers.values())
    print(
        f"{len(speakers)} speakers, {n_files} files, device={model_service.resolve_device()}"
    )

    t0 = time.time()
    embs: dict[str, list[np.ndarray]] = {}
    for spk, files in speakers.items():
        embs[spk] = [center(embed_file(p), mean) for p in files]
        print(
            f"  {spk}: {len(files)} files, {time.time() - t0:.0f}s elapsed", flush=True
        )

    same, diff = [], []
    for spk, vecs in embs.items():
        for a, b in itertools.combinations(vecs, 2):
            same.append(calibration_service.cosine_similarity(a, b))
    for (sa, va), (sb, vb) in itertools.combinations(embs.items(), 2):
        for a in va:
            for b in vb:
                diff.append(calibration_service.cosine_similarity(a, b))
    same_a, diff_a = np.array(same), np.array(diff)

    def pct(x: np.ndarray) -> dict:
        return {
            k: round(float(np.percentile(x, p)), 4)
            for k, p in (("min", 0), ("p5", 5), ("p50", 50), ("p95", 95), ("max", 100))
        }

    eer_thr, eer_rate = eer(same_a, diff_a)
    out = {
        "calibration": calibration,
        "speakers": len(speakers),
        "files": n_files,
        "same_pairs": len(same),
        "diff_pairs": len(diff),
        "same_scores": pct(same_a),
        "diff_scores": pct(diff_a),
        "per_threshold": [rates(same_a, diff_a, t) for t in args.thresholds],
        "eer_threshold": eer_thr,
        "eer": eer_rate,
        "embed_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
