#!/usr/bin/env python
"""Force a host out-of-memory on a throwaway CPU instance and check that the service survives it.

Starts ``main.py`` on a spare port with ``DEVICE=cpu``, warms it with a short
file, then lowers that one process's address-space limit (``RLIMIT_AS`` via
prlimit, so nothing else on the box is touched) to its current peak plus
``--headroom-mb``. A long recording then cannot get its activation memory:
torch's CPU allocator fails inside the forward pass and the service must
answer 500 with an "out of memory" detail, stay alive, and keep serving short
files. The limit is lifted at the end and the long file must succeed, which
proves the failure was the limit and not the file.

Steps and pass conditions (the server pid must be alive after every step):

1. N concurrent short, no limit  -> 200 (warm-up: creates the worker threads and
                                    their OpenMP teams before the limit goes on)
2. long file, limited            -> 500, detail contains "out of memory"
3. short file, still limited     -> 200 (the failed pass released its memory)
4. N concurrent long, limited    -> every one 500 out of memory, none dropped
5. long file, limit lifted       -> 200

Linux only (``/proc`` and ``resource.prlimit``). Runs the real model, so it is
a post-deploy / scheduled eval, not a gate test.

Usage (on a box with the model cached, CPU only):
    emb-env/bin/python evals/eval_oom_forced.py \
        --long /tmp/asr-vadbug/NhaEmOLungDoi.mp3 \
        --short /tmp/voice-oom/pool_test/VIVOSDEV01/VIVOSDEV01_R002.wav --port 6423

Exit code 0 = pass, 1 = fail.
"""

import argparse
import concurrent.futures
import json
import os
import resource
import subprocess
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_oom import is_oom, one_request  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def start_server(python: str, port: int, log_path: str) -> subprocess.Popen:
    """Launch main.py on CPU on the given port, logging to a file.

    Args:
        python: Interpreter to run main.py with (the service's venv python).
        port: Port to bind.
        log_path: File that receives the server's stdout and stderr.

    Returns:
        The server process.
    """
    env = {
        **os.environ,
        "DEVICE": "cpu",
        "PORT": str(port),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    log = open(log_path, "wb")
    return subprocess.Popen(
        [python, "main.py"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT
    )


def wait_health(url: str, timeout: float) -> None:
    """Block until /health answers 200 or the timeout passes.

    Args:
        url: Service base URL.
        timeout: Seconds to wait.

    Raises:
        TimeoutError: If the service does not become healthy in time.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError(f"{url}/health not up after {timeout:.0f}s")


def vm_peak_bytes(pid: int) -> int:
    """Read the process's peak virtual size from /proc.

    Args:
        pid: Process id.

    Returns:
        VmPeak in bytes.

    Raises:
        RuntimeError: If /proc/<pid>/status has no VmPeak line.
    """
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmPeak:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f"no VmPeak for pid {pid}")


def set_as_limit(pid: int, soft: int) -> None:
    """Set the process's soft RLIMIT_AS, leaving the hard limit unlimited so it can be lifted again.

    Args:
        pid: Process id (same user, so no capability is needed).
        soft: Soft address-space limit in bytes, or resource.RLIM_INFINITY.
    """
    resource.prlimit(pid, resource.RLIMIT_AS, (soft, resource.RLIM_INFINITY))


def alive(proc: subprocess.Popen) -> bool:
    """Tell whether the server process is still running.

    Args:
        proc: The server process.

    Returns:
        True while the process has not exited.
    """
    return proc.poll() is None


def outcome(res: dict) -> str:
    """Reduce one request result to a comparable label.

    Args:
        res: Result dict from eval_oom.one_request.

    Returns:
        "200", "oom" (500 whose detail says out of memory), or "<status>" otherwise.
    """
    if res["status"] == 200:
        return "200"
    if is_oom(res):
        return "oom"
    return str(res["status"])


def verdict(steps: list[dict]) -> list[str]:
    """List every way the run deviated from the expected outcomes.

    Args:
        steps: Dicts with name, expect ("200" or "oom"), outcomes (list of
            labels from outcome) and alive (bool, server still up afterwards).

    Returns:
        Failure messages; empty means PASS.
    """
    failures = []
    for s in steps:
        wrong = [o for o in s["outcomes"] if o != s["expect"]]
        if wrong:
            failures.append(f"{s['name']}: expected {s['expect']}, got {wrong}")
        if not s["alive"]:
            failures.append(f"{s['name']}: server died")
    return failures


def main() -> int:
    """Run the five steps against a fresh CPU instance and print a verdict.

    Returns:
        Process exit code, 0 on pass and 1 on fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long", required=True, help="Recording of several minutes.")
    parser.add_argument("--short", required=True, help="Recording of a few seconds.")
    parser.add_argument("--port", type=int, default=6423)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--headroom-mb", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--log",
        default=None,
        help="Server log path (default: <json>.server.log or ./oom_forced.server.log).",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    log_path = args.log or (
        f"{args.json}.server.log" if args.json else "oom_forced.server.log"
    )
    with open(args.long, "rb") as f:
        long_bytes = f.read()
    with open(args.short, "rb") as f:
        short_bytes = f.read()
    long_name = os.path.basename(args.long)
    short_name = os.path.basename(args.short)

    def post(data: bytes, name: str) -> dict:
        return one_request(url, data, name, args.timeout)

    def burst(data: bytes, name: str, n: int) -> list[dict]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            return [
                f.result() for f in [pool.submit(post, data, name) for _ in range(n)]
            ]

    proc = start_server(args.python, args.port, log_path)
    steps: list[dict] = []
    limit = None
    crash = None
    n = args.concurrency
    try:
        wait_health(url, 180)

        def step(name: str, expect: str, results: list[dict]) -> None:
            s = {
                "name": name,
                "expect": expect,
                "outcomes": [outcome(r) for r in results],
                "latency": [round(r["latency"], 2) for r in results],
                "detail": [r["detail"] for r in results],
                "alive": alive(proc),
            }
            steps.append(s)
            print(
                f"{name}: {s['outcomes']} in {s['latency']}s alive={s['alive']}",
                flush=True,
            )

        # Warm up with the same concurrency as step 4: each worker thread that
        # runs torch spawns its own OpenMP team (80 threads x 8 MiB stack of
        # address space on 222), and libgomp exits the whole process when a
        # thread cannot be created. That is a property of RLIMIT_AS, not of
        # the service, so the teams must exist before the limit goes on.
        step(
            f"1 {n} concurrent short, no limit",
            "200",
            burst(short_bytes, short_name, n),
        )

        peak = vm_peak_bytes(proc.pid)
        limit = peak + args.headroom_mb * 1024 * 1024
        set_as_limit(proc.pid, limit)
        print(
            f"RLIMIT_AS set to {limit >> 20} MiB (VmPeak {peak >> 20} MiB + {args.headroom_mb} MiB)"
        )

        step("2 long, limited", "oom", [post(long_bytes, long_name)])
        step("3 short, limited", "200", [post(short_bytes, short_name)])
        step(f"4 {n} concurrent long, limited", "oom", burst(long_bytes, long_name, n))

        set_as_limit(proc.pid, resource.RLIM_INFINITY)
        print("RLIMIT_AS lifted")
        step("5 long, limit lifted", "200", [post(long_bytes, long_name)])
    except (ProcessLookupError, TimeoutError, httpx.HTTPError) as exc:
        crash = f"{type(exc).__name__}: {exc}"
        print(f"aborted: {crash}", flush=True)
    finally:
        if alive(proc):
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()

    failures = verdict(steps)
    if crash:
        failures.append(crash)
    result = "PASS" if not failures else "FAIL"
    with open(log_path, errors="replace") as f:
        oom_log = [ln.rstrip() for ln in f if "out of memory" in ln.lower()]
    print(f"\n{result}")
    for msg in failures:
        print(f"  {msg}")
    print(f"server log lines mentioning out of memory: {len(oom_log)}")
    for ln in oom_log[:3]:
        print(f"  {ln}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "verdict": result,
                    "failures": failures,
                    "rlimit_as_bytes": limit,
                    "steps": steps,
                    "server_oom_log": oom_log,
                },
                fh,
                indent=2,
                default=str,
            )
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
