import argparse
import json
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from . import servers
from .config import Experiment, Run, expand, load_config
from .dataset import preflight, prepare_inputs
from .plans import REPO_ROOT
from .results import Result, append_result


def registered_policies() -> set[str]:
    from vllm.kalypso.workers.policy import POLICIES

    return set(POLICIES)


def source_commit(repo_root: Path) -> str:
    provenance = repo_root / "benchmark_multi_kalypso/source_commit.txt"
    if provenance.exists():
        return provenance.read_text().strip()
    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "describe",
                "--always",
                "--dirty",
                "--match",
                "",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def run_one(
    config: Experiment,
    run: Run,
    workers: list[servers.Server],
    run_dir: Path,
    repo_root: Path,
    result: Result,
    body: dict,
) -> None:
    coordinator = None
    try:
        (run_dir / "request.json").write_text(json.dumps(body, indent=2) + "\n")
        if config.reset_prefix_cache:
            servers.reset_prefix_caches(workers)
        coordinator = servers.start_coordinator(
            config, run.policy, workers, run_dir, repo_root
        )
        started = time.monotonic()
        try:
            response = servers.post_query(
                coordinator.url, body, config.timeouts.query_s
            )
        finally:
            result.completion_s = time.monotonic() - started
        (run_dir / "response.json").write_text(json.dumps(response, indent=2) + "\n")
        result.num_output_rows = response["num_output_rows"]
        result.status = "ok"
    except Exception as exc:
        result.status, result.error = "error", str(exc)
    finally:
        if coordinator is not None:
            servers.stop(coordinator)
    if result.status == "ok" and result.completion_s and result.rows is not None:
        result.rows_per_s = result.rows / result.completion_s


def run_host(
    config: Experiment,
    n_workers: int,
    gpus: list[str],
    out_dir: str | Path,
    repo_root: str | Path,
) -> list[Result]:
    runs = expand(config, n_workers)
    if len(gpus) != n_workers or len(set(gpus)) != n_workers:
        raise ValueError("Supply exactly N distinct GPU identifiers")
    out, repo = Path(out_dir).resolve(), Path(repo_root).resolve()
    counts = preflight(config, repo)
    shortages = [
        f"{name} {label}: have {have}, need {need}"
        for name, tables in counts.items()
        for label, (have, need) in tables.items()
        if have < need
    ]
    if shortages:
        raise ValueError("Data preflight failed: " + "; ".join(shortages))
    # Fresh directories prevent stale recorder rows and responses contaminating runs.
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    commit = source_commit(repo)
    (out / "source_commit.txt").write_text(commit + "\n")
    prepared = prepare_inputs(config, repo, out)
    policies = registered_policies()
    results = []
    workers = servers.start_workers(config, gpus, out, repo)
    try:
        for run in runs:
            run_dir = out / run.name
            run_dir.mkdir(parents=True)
            result = Result(
                commit,
                run.workload.name,
                run.workload.rows,
                None,
                n_workers,
                run.policy,
                run.rep,
            )
            if run.policy not in policies:
                result.status = "skipped_policy"
                result.error = f"Policy {run.policy!r} is not in POLICIES"
            else:
                body, result.rows, result.hypotheses = prepared[run.workload.name]
                run_one(config, run, workers, run_dir, repo, result, body)
            append_result(out, result)
            results.append(result)
            print(
                f"N={n_workers} {run.name}: {result.status} {result.error or ''}",
                flush=True,
            )
    finally:
        for worker in reversed(workers):
            servers.stop(worker)
    return results


def main() -> int:
    def interrupt(signum: int, _frame: object) -> None:
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt(f"Received {signal.Signals(signum).name}")

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, interrupt)
    parser = argparse.ArgumentParser(
        description="Run one N-worker M3 group on this host"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    gpus = [g.strip() for g in args.gpus.split(",")]
    results = run_host(config, args.n, gpus, args.out, REPO_ROOT)
    return int(any(r.status == "error" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
