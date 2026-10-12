# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from vllm.kalypso.execution.executor import CompletionResult, Text, apply_determinism
from vllm.kalypso.execution.instrumentation import (
    CALL_CONTEXT,
    RequestTracer,
    parse_metric,
)
from vllm.kalypso.execution.worker_pool import Worker


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def traced_worker(tmp_path, client):
    worker = Worker(2, "w2", 100, client)
    worker.tracer = RequestTracer(tmp_path)
    return worker


def test_traced_worker_writes_one_record(tmp_path):
    client = AsyncMock()
    client.execute.return_value = CompletionResult(
        "ok", "id", "stop", prompt_tokens=11, completion_tokens=3
    )
    worker = traced_worker(tmp_path, client)
    CALL_CONTEXT.set(
        {"row": 4, "stage": 1, "op": "SemFilter", "op_index": 0, "ready_at": 5.0}
    )

    asyncio.run(worker.execute(None, "prompt", 8))

    (record,) = read_lines(tmp_path / "requests.jsonl")
    assert record["worker_id"] == 2
    assert record["worker"] == "w2"
    assert (record["row"], record["stage"]) == (4, 1)
    assert (record["op"], record["op_index"], record["submit"]) == ("SemFilter", 0, 5.0)
    assert record["prompt_tokens"] == 11
    assert record["output_tokens"] == 3
    assert record["ok"] is True
    assert record["latency"] == pytest.approx(record["finish"] - record["start"])


def test_traced_worker_records_failure_and_reraises(tmp_path):
    client = AsyncMock()
    client.execute.side_effect = RuntimeError("boom")
    worker = traced_worker(tmp_path, client)
    CALL_CONTEXT.set(None)

    with pytest.raises(RuntimeError):
        asyncio.run(worker.execute(None, "prompt", 8))

    (record,) = read_lines(tmp_path / "requests.jsonl")
    assert record["ok"] is False
    assert record["row"] is None
    assert worker.outstanding == 0


def test_parse_metric_sums_label_sets():
    text = (
        "# HELP vllm:num_requests_waiting waiting\n"
        'vllm:num_requests_waiting{engine="0",model_name="m"} 2.0\n'
        'vllm:num_requests_waiting{engine="1",model_name="m"} 3.0\n'
        "vllm:num_requests_running 5\n"
    )
    assert parse_metric(text, "vllm:num_requests_waiting") == 5.0
    assert parse_metric(text, "vllm:num_requests_running") == 5.0
    assert parse_metric(text, "vllm:missing") is None


def test_determinism_keeps_token_counts(monkeypatch):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "1")
    result = CompletionResult("x", "id", "length", prompt_tokens=7, completion_tokens=2)

    flipped = apply_determinism("prompt", result, Text())

    assert (flipped.prompt_tokens, flipped.completion_tokens) == (7, 2)
