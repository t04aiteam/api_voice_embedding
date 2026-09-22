#!/usr/bin/env python
"""Memory-pressure eval against a live service: long audio, concurrent requests.

Fires N concurrent POST /compare calls with the same long recording on both
sides and checks that the service degrades cleanly: every response must be
200 (embedded) or 503 (out of memory, retryable). A single 500, timeout or
connection error fails the eval, and so does a 200 rate below --min-success.

This exercises the real model, so it runs after deploy and on a schedule, not
as a pre-commit gate (tests/test_oom.py covers the logic deterministically).

Usage:
    python evals/eval_oom.py /path/to/long.mp3 --url http://localhost:6422 \
        --concurrency 4 --rounds 2 --min-success 0.5

Exit code 0 = pass, 1 = fail.
"""

import argparse
import concurrent.futures
import json
import sys
import time

import httpx


def one_request(url: str, audio: bytes, name: str, timeout: float) -> dict:
    """Send a single /compare request and summarise the outcome.

    Args:
        url: Service base URL.
        audio: Encoded audio file bytes, used for both sides.
        name: File name to send (drives the decoder's format detection).
        timeout: Per-request timeout in seconds.

    Returns:
        A dict with status (int or "error"), latency seconds and a detail string.
    """
    files = {"file_a": (name, audio), "file_b": (name, audio)}
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{url}/compare", files=files, timeout=timeout)
        latency = time.perf_counter() - t0
        try:
            detail = r.json()
        except ValueError:
            detail = r.text[:200]
        return {
            "status": r.status_code,
            "latency": latency,
            "retry_after": r.headers.get("Retry-After"),
            "detail": detail,
        }
    except Exception as exc:  # noqa: BLE001 - any transport failure is a hard fail
        return {
            "status": "error",
            "latency": time.perf_counter() - t0,
            "retry_after": None,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    """Run the eval and print a verdict.

    Returns:
        Process exit code, 0 on pass and 1 on fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio", help="Long recording (several minutes) to compare with itself."
    )
    parser.add_argument("--url", default="http://localhost:6422")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--min-success",
        type=float,
        default=0.5,
        help="Minimum fraction of requests that must return 200.",
    )
    parser.add_argument("--json", help="Write per-request results to this file.")
    args = parser.parse_args()

    with open(args.audio, "rb") as fh:
        audio = fh.read()
    name = args.audio.rsplit("/", 1)[-1]

    results = []
    for rnd in range(args.rounds):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futs = [
                pool.submit(one_request, args.url, audio, name, args.timeout)
                for _ in range(args.concurrency)
            ]
            for i, fut in enumerate(futs):
                res = fut.result()
                res["round"] = rnd
                res["slot"] = i
                results.append(res)
                print(
                    f"round={rnd} slot={i} status={res['status']} "
                    f"latency={res['latency']:.2f}s retry_after={res['retry_after']}",
                    flush=True,
                )

    total = len(results)
    ok = sum(1 for r in results if r["status"] == 200)
    busy = sum(1 for r in results if r["status"] == 503)
    bad = [r for r in results if r["status"] not in (200, 503)]
    success_rate = ok / total if total else 0.0
    verdict = "PASS" if not bad and success_rate >= args.min_success else "FAIL"

    print(
        f"\n{verdict}: {ok}/{total} ok, {busy} busy(503), {len(bad)} failed, "
        f"success_rate={success_rate:.2f} (min {args.min_success})"
    )
    for r in bad:
        print(
            f"  round={r['round']} slot={r['slot']} status={r['status']} detail={r['detail']}"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {"verdict": verdict, "results": results}, fh, indent=2, default=str
            )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
