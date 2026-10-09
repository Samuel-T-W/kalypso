# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check multi-worker semantic queries against an in-process baseline.

Compare output uniqueness, row mapping, and operator output values. Optional
coordinator traces also verify exactly-once calls and row affinity within each
stage, and report per-worker call counts. The matrix command runs these checks
across worker counts and routing policies.

KALYPSO_COIN_FLIP_SEED makes responses deterministic functions of seed and prompt
while still sending every LLM call to a real worker. Capture the baseline with
the same seed so differences reveal routing or state bugs rather than model noise.
"""

import argparse
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def contains(value: Any, text: str) -> bool:
    if isinstance(value, str):
        return text in value
    children = value.values() if isinstance(value, dict) else value
    return isinstance(value, (list, dict)) and any(
        contains(child, text) for child in children
    )


def values(response: dict) -> dict:
    return {r["idx"]: [o["value"] for o in r["outputs"]] for r in response["results"]}


def check(
    payload: dict, baseline: dict, candidate: dict, trace_dir: str | Path | None = None
) -> tuple[bool, dict]:
    records = payload["records"]
    rows = candidate["results"]
    indices = [r["idx"] for r in rows]
    valid = all(type(i) is int and 0 <= i < len(records) for i in indices)
    checks = {"exactly once (output)": valid and len(set(indices)) == len(indices)}
    checks["row mapping"] = valid and all(
        contains(r["input_context"], records[r["idx"]][payload["text_column"]])
        for r in rows
    )
    before = values(baseline)
    after = values(candidate)
    differing = sorted(
        i
        for i in before.keys() | after.keys()
        if i not in before or i not in after or before[i] != after[i]
    )
    checks[f"agreement (differing idx: {differing})"] = not differing
    workers = Counter()
    if trace_dir is not None:
        calls = [
            json.loads(line)
            for line in (Path(trace_dir) / "requests.jsonl").read_text().splitlines()
        ]
        first_stage = min((c["stage"] for c in calls), default=None)
        row_calls = [c for c in calls if c["row"] is not None]
        counts = Counter()
        first = Counter()
        affinity = defaultdict(set)
        for call in calls:
            workers[call["worker_id"]] += 1
            row = call["row"]
            if row is None:
                continue
            stage = call["stage"]
            op_index = call["op_index"]
            counts[row, stage, op_index] += 1
            if stage == first_stage and op_index == 0:
                first[row] += 1
            affinity[row, stage].add(call["worker_id"])
        checks["exactly once (calls)"] = (
            first == Counter(range(len(records)))
            and all(n == 1 for n in counts.values())
            and all(
                type(c["row"]) is int
                and 0 <= c["row"] < len(records)
                and c["ok"] is True
                for c in row_calls
            )
        )
        checks["affinity"] = all(len(ids) == 1 for ids in affinity.values())
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if trace_dir is not None:
        print(f"Per-worker calls: {dict(workers)}")
    return all(checks.values()), dict(workers)


def run(endpoint: str, payload: dict, out: str | Path) -> tuple[dict, float]:
    started = time.monotonic()
    response = requests.post(endpoint, json=payload, timeout=600)
    response.raise_for_status()
    result = response.json()
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result, time.monotonic() - started


def matrix(args: argparse.Namespace, payload: dict) -> bool:
    baseline = read(args.baseline)
    summary = []
    launcher = Path(__file__).with_name("launch_multi_worker.sh")
    for n in args.workers:
        for policy in args.policies:
            directory = Path(args.out_dir).resolve() / f"N{n}_{policy}"
            command = [str(launcher), "-o", str(directory)]
            passed = False
            kept = "-"
            latency = "-"
            workers = {}
            try:
                if (directory / "trace" / "requests.jsonl").exists():
                    raise ValueError(f"Use a fresh trace directory: {directory}")
                subprocess.run(
                    command + ["-n", str(n), "-p", policy, "--detach"],
                    env={**os.environ, "KALYPSO_COIN_FLIP_SEED": str(args.seed)},
                    check=True,
                )
                result, elapsed = run(
                    "http://localhost:8100/v1/semantic/query",
                    payload,
                    directory / "response.json",
                )
                kept = len(result["results"])
                latency = f"{elapsed:.3f}s"
                passed, workers = check(payload, baseline, result, directory / "trace")
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                subprocess.SubprocessError,
                requests.RequestException,
            ) as exc:
                print(f"FAIL N={n} policy={policy}: {exc}")
            finally:
                try:
                    subprocess.run(command + ["--stop"], check=True)
                except (OSError, subprocess.SubprocessError) as exc:
                    print(f"FAIL cleanup N={n} policy={policy}: {exc}")
                    passed = False
            summary.append((n, policy, kept, latency, workers, passed))
    print("N | policy | rows kept | latency | per-worker calls | status")
    for n, policy, kept, latency, workers, passed in summary:
        print(
            f"{n} | {policy} | {kept} | {latency} | {workers} | "
            f"{'PASS' if passed else 'FAIL'}"
        )
    return all(row[-1] for row in summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check", "matrix"):
        sub = subs.add_parser(name)
        sub.add_argument("--payload", required=True)
        required = ["endpoint", "out"] if name == "run" else ["baseline"]
        for option in required:
            sub.add_argument(f"--{option}", required=True)
        if name == "check":
            sub.add_argument("--candidate", required=True)
            sub.add_argument("--trace-dir")
        if name == "matrix":
            sub.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
            sub.add_argument(
                "--policies", nargs="+", default=["round_robin", "least_outstanding"]
            )
            sub.add_argument("--seed", type=int, default=7)
            sub.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    try:
        payload = read(args.payload)
        if args.command == "run":
            run(args.endpoint, payload, args.out)
            return 0
        if args.command == "check":
            return int(
                not check(
                    payload, read(args.baseline), read(args.candidate), args.trace_dir
                )[0]
            )
        return int(not matrix(args, payload))
    except (OSError, ValueError, KeyError, TypeError, requests.RequestException) as exc:
        print(f"FAIL {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
