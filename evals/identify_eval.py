"""Measure POST /speakers/identify (1:N, UC 68) end to end over HTTP.

Enrolls the first --enroll clips of every speaker in --gallery-dir, then sends
--queries further clips per gallery speaker (genuine queries) and --queries
clips per speaker of --impostor-dir, whose speakers are never enrolled (open
set: nobody in the database should be verified). Reports:

- rank-1: the top match is the right person (genuine queries);
- top-k hit: the right person is anywhere in the returned list;
- per threshold: genuine accept (top match right and verified), wrong accept
  (top match is another enrolled person and verified) and impostor false match
  (top match verified for someone who is not enrolled);
- the threshold where genuine misses equal impostor false matches.

Run it against a test instance with its own empty speaker store, never the
live one: it refuses to start if GET /speakers is not empty, and deletes what
it enrolled when it ends.

Usage (222, CPU test instance):
    cd ~/quant04/api_voice_embedding
    SPEAKER_DATA_DIR=/tmp/identify-eval/speakers DEVICE=cpu PORT=6423 \\
        emb-env/bin/python main.py &
    emb-env/bin/python evals/identify_eval.py --url http://127.0.0.1:6423 \\
        --gallery-dir /tmp/voice-oom/pool_test --impostor-dir /tmp/voice-oom/pool_train \\
        --json /tmp/identify-eval/identify.json

Exit code 1 when rank-1 is below --min-rank1.
"""

import argparse
import json
import os
import sys
import time

import requests

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mp4"}
PREFIX = "eval_"


def collect(audio_dir: str, per_speaker: int) -> dict[str, list[str]]:
    """Map each speaker sub-folder to its first N audio files, sorted.

    Args:
        audio_dir: Folder with one sub-folder per speaker.
        per_speaker: Files kept per speaker.

    Returns:
        Speaker folder name to file paths, in name order.
    """
    out = {}
    for name in sorted(os.listdir(audio_dir)):
        folder = os.path.join(audio_dir, name)
        if not os.path.isdir(folder):
            continue
        files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in AUDIO_EXT
        )
        out[name] = files[:per_speaker]
    return out


def rate(hits: int, total: int) -> float:
    """Fraction rounded for the report, 0 when there is nothing to count.

    Args:
        hits: Counted events.
        total: Trials.

    Returns:
        hits / total, rounded to 4 places.
    """
    return round(hits / total, 4) if total else 0.0


def at_threshold(genuine: list[dict], impostor: list[dict], t: float) -> dict:
    """Open-set outcomes when a top match counts as verified at ``t``.

    Args:
        genuine: Queries from enrolled speakers, each {"speaker", "matches"}.
        impostor: Queries from speakers never enrolled, same shape.
        t: Threshold on the top match's similarity.

    Returns:
        Genuine accept, wrong accept and impostor false match rates.
    """
    tops = [(q["speaker"], q["matches"][0]) for q in genuine if q["matches"]]
    right = sum(1 for s, m in tops if m["speaker_id"] == s and m["similarity"] >= t)
    wrong = sum(1 for s, m in tops if m["speaker_id"] != s and m["similarity"] >= t)
    false_match = sum(
        1 for q in impostor if q["matches"] and q["matches"][0]["similarity"] >= t
    )
    return {
        "threshold": t,
        "genuine_accept_rate": rate(right, len(genuine)),
        "wrong_accept_rate": rate(wrong, len(genuine)),
        "impostor_false_match_rate": rate(false_match, len(impostor)),
    }


def summarize(
    genuine: list[dict], impostor: list[dict], thresholds: list[float]
) -> dict:
    """Rank and threshold metrics over all queries.

    Args:
        genuine: Queries from enrolled speakers, each {"speaker", "matches"}.
        impostor: Queries from speakers never enrolled, same shape.
        thresholds: Thresholds to report.

    Returns:
        rank-1, top-k hit, per-threshold rates and the equal-error threshold.
    """
    rank1 = sum(
        1
        for q in genuine
        if q["matches"] and q["matches"][0]["speaker_id"] == q["speaker"]
    )
    topk = sum(
        1 for q in genuine if any(m["speaker_id"] == q["speaker"] for m in q["matches"])
    )
    # Equal error: genuine miss (1 - genuine accept) against impostor false match,
    # swept over every top-1 score seen.
    cuts = sorted(
        {q["matches"][0]["similarity"] for q in genuine + impostor if q["matches"]}
    )
    best = None
    for t in cuts:
        r = at_threshold(genuine, impostor, t)
        gap = abs((1 - r["genuine_accept_rate"]) - r["impostor_false_match_rate"])
        if best is None or gap < best[0]:
            best = (gap, t, r)
    eer = None
    if best:
        _, t, r = best
        eer = {
            "threshold": t,
            "genuine_miss_rate": round(1 - r["genuine_accept_rate"], 4),
            "impostor_false_match_rate": r["impostor_false_match_rate"],
        }
    return {
        "genuine_queries": len(genuine),
        "impostor_queries": len(impostor),
        "rank1_accuracy": rate(rank1, len(genuine)),
        "topk_hit_rate": rate(topk, len(genuine)),
        "thresholds": [at_threshold(genuine, impostor, t) for t in thresholds],
        "equal_error": eer,
    }


def post(url: str, path: str, data: dict) -> dict:
    """POST one file as multipart and return the JSON body, raising on non-200.

    Args:
        url: Full URL.
        path: File to upload.
        data: Extra form fields.

    Returns:
        The parsed JSON body.

    Raises:
        RuntimeError: When the service does not answer 200.
    """
    with open(path, "rb") as fh:
        r = requests.post(
            url, files={"file": (os.path.basename(path), fh)}, data=data, timeout=600
        )
    if r.status_code != 200:
        raise RuntimeError(
            f"{url} {os.path.basename(path)}: {r.status_code} {r.text[:200]}"
        )
    return r.json()


def main() -> None:
    """Enroll, query, clean up, report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="base URL of a test instance")
    ap.add_argument("--gallery-dir", required=True, help="speakers to enroll")
    ap.add_argument("--impostor-dir", required=True, help="speakers never enrolled")
    ap.add_argument("--enroll", type=int, default=3, help="clips enrolled per speaker")
    ap.add_argument("--queries", type=int, default=10, help="query clips per speaker")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.96, 0.97, 0.98, 0.99],
    )
    ap.add_argument("--min-rank1", type=float, default=0.90, help="pass bar")
    ap.add_argument("--json", required=True, help="report path")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    existing = requests.get(f"{base}/speakers", timeout=30).json()
    if existing:
        sys.exit(f"refusing: {base} already has {len(existing)} enrolled speakers")

    gallery = collect(args.gallery_dir, args.enroll + args.queries)
    impostors = collect(args.impostor_dir, args.queries)
    t0 = time.monotonic()
    genuine, impostor = [], []
    enrolled = []
    try:
        for spk, files in gallery.items():
            for path in files[: args.enroll]:
                post(f"{base}/speakers/enroll", path, {"speaker_id": PREFIX + spk})
            enrolled.append(PREFIX + spk)
        for label, pool, sink in (
            ("genuine", {s: f[args.enroll :] for s, f in gallery.items()}, genuine),
            ("impostor", impostors, impostor),
        ):
            for spk, files in pool.items():
                for path in files:
                    body = post(
                        f"{base}/speakers/identify",
                        path,
                        {"top_k": str(args.top_k), "threshold": "0"},
                    )
                    sink.append(
                        {
                            "speaker": PREFIX + spk,
                            "file": os.path.basename(path),
                            "matches": body["matches"],
                        }
                    )
            print(f"{label}: {len(sink)} queries", flush=True)
    finally:
        for sid in enrolled:
            requests.delete(f"{base}/speakers/{sid}", timeout=30)

    report = summarize(genuine, impostor, args.thresholds)
    report.update(
        {
            "url": base,
            "gallery_speakers": len(gallery),
            "impostor_speakers": len(impostors),
            "enroll_per_speaker": args.enroll,
            "seconds": round(time.monotonic() - t0, 1),
            "min_rank1": args.min_rank1,
            "passed": report["rank1_accuracy"] >= args.min_rank1,
            "queries": {"genuine": genuine, "impostor": impostor},
        }
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    shown = {k: v for k, v in report.items() if k != "queries"}
    print(json.dumps(shown, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
