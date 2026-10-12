# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "vllm/kalypso/benchmark/plot_worker_trace.py")


def test_two_workers_summary_and_png(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    calls = [
        dict(worker_id=0, start=10, finish=11, latency=1, ok=True,
             op="sem_filter", stage=0),
        dict(worker_id=0, start=11, finish=14, latency=3, ok=False,
             op="sem_map", stage=1),
        dict(worker_id=1, start=10, finish=12, latency=2, ok=True,
             op="sem_filter", stage=0),
    ]
    for directory in (first, second):
        (directory / "requests.jsonl").write_text(
            "\n".join(json.dumps(c) for c in reversed(calls)) + "\n")
    queue = [dict(t=t, worker_id=w, outstanding=2, admitted=a,
                  vllm_waiting=waiting, vllm_running=1)
             for t, w, a, waiting in [(12, 0, 1, 0), (10, 0, 2, 1),
                                      (10, 1, 1, None)]]
    (first / "queue.jsonl").write_text(
        "\n".join(json.dumps(q) for q in queue))
    out = tmp_path / "workers.png"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--trace-dir", str(first),
         "--trace-dir", str(second), "--out", str(out)],
        env={**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path)},
        check=True, capture_output=True, text=True)
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result.stdout.count("0 2 2.000 2.980 0.500") == 2
    assert result.stdout.count("1 1 2.000 2.000 0.500") == 2
    assert "first (all calls" in result.stdout
    assert "second (all calls" in result.stdout


def test_completion_bins_and_empty_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("plot_worker_trace", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = [{"finish": 10.5}, {"finish": 11}, {"finish": 11.9}, {"finish": 14}]
    bins = module.completion_bins(calls, 10, 5)
    assert bins == [1, 2, 0, 0, 1]
    assert sum(bins) == len(calls)
    count, p50, p99, rate = module.summarize([])
    assert count == rate == 0
    assert p50 != p50 and p99 != p99
    assert module.percentile([2], 0.99) == pytest.approx(2)
