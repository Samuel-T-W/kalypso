# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request and per-worker queue tracing for multi-worker runs.

`requests.jsonl` has one record per LLM call made through a Worker:
  submit: when the row became ready for this stage (Task enqueue time), so
    `start - submit` is time spent in Kalypso admission plus earlier
    operators of the stage.
  start: when the call was dispatched to the worker client.
  finish: when the reply arrived. `latency = finish - start`.
All times are wall-clock `time.time()` seconds.
vLLM-internal queue time is not visible per request over HTTP without
changing vLLM; the `vllm_waiting` field of `queue.jsonl` covers queue depth.

`queue.jsonl` has one record per worker per sampler tick.
"""

import asyncio
import contextvars
import json
import re
import time
from pathlib import Path
from typing import Any

import aiohttp

# Set by Stage.run_task before each operator call:
# {"row", "stage", "op", "op_index", "ready_at"}.
CALL_CONTEXT: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "kalypso_call_context", default=None
)


class RequestTracer:
    def __init__(self, trace_dir: str | Path):
        directory = Path(trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._requests = open(directory / "requests.jsonl", "a", buffering=1)
        self._queue = open(directory / "queue.jsonl", "a", buffering=1)

    def record_request(self, record: dict[str, Any]) -> None:
        self._requests.write(json.dumps(record) + "\n")

    def record_queue(self, record: dict[str, Any]) -> None:
        self._queue.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._requests.close()
        self._queue.close()


def parse_metric(text: str, name: str) -> float | None:
    """Sum a Prometheus metric over its label sets; None when absent."""
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)", re.M)
    values = [float(match) for match in pattern.findall(text)]
    return sum(values) if values else None


async def _vllm_queue(session: aiohttp.ClientSession, base_url: str):
    try:
        async with session.get(f"{base_url}/metrics") as resp:
            text = await resp.text()
        return (
            parse_metric(text, "vllm:num_requests_waiting"),
            parse_metric(text, "vllm:num_requests_running"),
        )
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        return None, None


async def sample_queue_depth(
    pool: Any, manager: Any, tracer: RequestTracer, interval_s: float = 0.5
) -> None:
    timeout = aiohttp.ClientTimeout(total=max(interval_s, 1.0))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            for worker in pool.workers:
                waiting = running = None
                base_url = getattr(worker.client, "base_url", None)
                if base_url is not None:
                    waiting, running = await _vllm_queue(session, base_url)
                used, capacity = manager.worker_usage(worker.worker_id)
                tracer.record_queue(
                    {
                        "t": time.time(),
                        "worker_id": worker.worker_id,
                        "worker": worker.name,
                        "outstanding": worker.outstanding,
                        "admitted": manager.worker_inflight(worker.worker_id),
                        "kv_used": used,
                        "kv_capacity": capacity,
                        "vllm_waiting": waiting,
                        "vllm_running": running,
                    }
                )
            await asyncio.sleep(interval_s)
