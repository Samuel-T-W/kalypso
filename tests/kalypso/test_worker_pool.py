# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.controller.plan import SemanticPlan
from vllm.kalypso.execution.executor import CompletionResult
from vllm.kalypso.execution.worker_pool import LocalWorker, Worker, WorkerPool
from vllm.kalypso.query import Query
from vllm.kalypso.query_processor import QueryProcessor


@pytest.fixture
def three_workers():
    return [
        Worker(i, str(i), 100, AsyncMock(execute=AsyncMock(return_value=i)))
        for i in range(3)
    ]


def test_round_robin_candidates_and_affinity(three_workers):
    pool = WorkerPool(three_workers, "model")
    manager = Mock(num_workers=Mock(return_value=3))
    orders = [pool.candidate_workers(None, manager) for _ in range(6)]
    assert orders == [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ]
    assert pool.candidate_workers(None, manager) == [0, 1, 2]
    assert pool.candidate_workers(0, manager) == [0, 1, 2]
    assert pool.candidate_workers(2, manager) == [2, 1, 0]
    assert pool.candidate_workers(None, manager) == [1, 2, 0]


@pytest.mark.parametrize(
    "counts,expected", [([3, 1, 2], [1, 2, 0]), ([1, 1, 1], [0, 1, 2])]
)
def test_least_outstanding_candidates(three_workers, counts, expected):
    pool = WorkerPool(three_workers, "model", policy="least_outstanding")
    manager = Mock(
        num_workers=Mock(return_value=3),
        worker_inflight=Mock(side_effect=lambda worker_id: counts[worker_id]),
    )
    assert pool.candidate_workers(None, manager) == expected
    for preferred in range(3):
        candidates = pool.candidate_workers(preferred, manager)
        assert candidates[0] == preferred
        assert sorted(candidates) == [0, 1, 2]
    assert pool.candidate_workers(0, manager) == [0, 1, 2]
    counts[:] = [2, 3, 1]
    assert pool.candidate_workers(None, manager) == [2, 0, 1]


def test_invalid_policy(three_workers):
    with pytest.raises(ValueError, match="unknown"):
        WorkerPool(three_workers, "model", policy="unknown")


def test_round_robin_execute_uses_separate_cursor(three_workers):
    pool = WorkerPool(three_workers, "model")
    manager = Mock(num_workers=Mock(return_value=3))
    assert pool.candidate_workers(None, manager) == [0, 1, 2]

    async def run():
        return [
            await pool.execute(None, "hello", 8, pin=False, priority=7)
            for _ in range(6)
        ]

    assert asyncio.run(run()) == [0, 1, 2, 0, 1, 2]
    assert pool.candidate_workers(None, manager) == [1, 2, 0]
    for worker in three_workers:
        assert worker.completed == 2
        assert worker.outstanding == 0
        worker.client.execute.assert_awaited_with(None, "hello", 8, False, 7, None)


@pytest.mark.parametrize("counts,expected", [([3, 1, 2], 1), ([1, 1, 1], 0)])
def test_least_outstanding_execute(three_workers, counts, expected):
    pool = WorkerPool(three_workers, "model", policy="least_outstanding")
    for worker, count in zip(three_workers, counts):
        worker.outstanding = count
    assert asyncio.run(pool.execute(None, "hello", 8, False, 7)) == expected
    for worker, count in zip(three_workers, counts):
        assert worker.outstanding == count
        assert worker.completed == int(worker.worker_id == expected)


def test_local_passes_policy(monkeypatch):
    worker = Worker(0, "local", 100, AsyncMock())
    monkeypatch.setattr(
        "vllm.kalypso.execution.worker_pool.LocalWorker", Mock(return_value=worker)
    )
    pool = WorkerPool.local("model", 100, policy="least_outstanding")
    assert pool.policy == "least_outstanding"
    assert pool.workers == [worker]


@pytest.mark.parametrize("injected,budget", [(False, 100), (True, None), (True, 999)])
def test_query_rows_call_worker_pool(tmp_path, monkeypatch, injected, budget):
    init = Mock()
    monkeypatch.setattr(KVMemoryManager, "init", init)
    monkeypatch.setattr(
        KVMemoryManager, "get_instance", Mock(return_value=Mock(token_length=len))
    )
    if injected:
        pool = WorkerPool(
            [
                Worker(0, "first", 100, AsyncMock()),
                Worker(1, "second", 123, AsyncMock()),
            ],
            "model",
        )
        processor = QueryProcessor("model", budget=budget, pool=pool)
        init.assert_called_once_with("model", 223)
    else:
        processor = QueryProcessor("model", budget=budget)
        pool = processor.pool
        assert len(pool) == 1
        assert isinstance(pool.worker(0), LocalWorker)
        init.assert_called_once_with("model", 100)

    client = pool.worker(0).client
    client.execute = AsyncMock(
        return_value=CompletionResult("hello back", "request-1", "stop")
    )
    assert processor.executor is pool
    plan = SemanticPlan(processor.executor)
    assert plan.plan_executor.pipeline_executor.pool is pool
    path = tmp_path / "rows.csv"
    path.write_text("data\nhello\n")
    raw_request = object()
    rows = asyncio.run(processor.execute(raw_request, Query([], str(path))))
    assert len(rows) == 1
    assert rows[0].state.executor is pool
    result = asyncio.run(
        rows[0].state.executor.execute(raw_request, "hello", 8, False, 7)
    )
    assert (result.text, result.request_id, result.finish_reason) == (
        "hello back",
        "request-1",
        "stop",
    )
    client.execute.assert_awaited_once_with(raw_request, "hello", 8, False, 7, None)
    assert (pool.worker(0).outstanding, pool.worker(0).completed) == (0, 1)
    if injected:
        pool.worker(1).client.execute.assert_not_awaited()


def test_query_processor_requires_budget_without_pool():
    with pytest.raises(ValueError, match="budget is required"):
        QueryProcessor("model")


def test_plan_accepts_executor_without_pool(monkeypatch):
    monkeypatch.setattr(KVMemoryManager, "get_instance", Mock())
    plan = SemanticPlan(AsyncMock())
    assert plan.plan_executor.pipeline_executor.pool is None


@pytest.mark.parametrize("fails", [False, True])
def test_execute_counters(fails):
    async def execute(*args):
        assert worker.outstanding == 1
        if fails:
            raise RuntimeError("inference failed")
        return "unchanged"

    client = AsyncMock(execute=AsyncMock(side_effect=execute))
    worker = Worker(0, "test", 100, client)
    if fails:
        with pytest.raises(RuntimeError, match="inference failed"):
            asyncio.run(worker.execute(None, "hello", 8, True, 7))
    else:
        assert asyncio.run(worker.execute(None, "hello", 8, True, 7)) == "unchanged"
    assert (worker.outstanding, worker.completed) == (0, int(not fails))


@pytest.mark.parametrize(
    "case", ["success", "mismatch", "unreachable", "empty", "missing"]
)
def test_config_discovery(tmp_path, case):
    async def run():
        async def models(request):
            other = case == "mismatch" and request.match_info["id"] == "1"
            return web.json_response({"data": [{"id": "other" if other else "model"}]})

        async def dummy(request):
            assert request.match_info["id"] == "0", "Override must skip discovery"
            return web.json_response({"budget": 200})

        app = web.Application()
        app.router.add_get("/{id}/v1/models", models)
        app.router.add_post("/{id}/v1/semantic/dummy", dummy)
        async with TestServer(app) as server:
            base = str(server.make_url(""))
            entries = [
                {"url": f"{base}/0"},
                {"url": f"{base}/1", "kv_budget_bytes": 123},
            ]
            if case == "unreachable":
                await server.close()
            path = tmp_path / "workers.json"
            config = {"empty": [], "missing": [{}]}.get(case, entries)
            path.write_text(json.dumps({"workers": config}))
            if case in ("mismatch", "unreachable", "empty", "missing"):
                error = ValueError if case in ("empty", "missing") else RuntimeError
                pattern = "workers" if error is ValueError else f"{base}/0"
                with pytest.raises(error, match=pattern) as caught:
                    await WorkerPool.from_config(path)
                if case == "mismatch":
                    assert f"{base}/1" in str(caught.value)
            else:
                pool = await WorkerPool.from_config(path, policy="least_outstanding")
                assert pool.policy == "least_outstanding"
                assert [w.kv_budget for w in pool.workers] == [100, 123]
                assert pool.total_kv_budget() == 223
                await pool.close()

    asyncio.run(run())
