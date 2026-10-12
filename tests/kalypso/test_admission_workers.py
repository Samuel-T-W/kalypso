# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from unittest.mock import AsyncMock

import pytest

from vllm.kalypso import budget
from vllm.kalypso.context import ExecutionState, SemanticInput, SemContext
from vllm.kalypso.controller.stage import Stage, Task
from vllm.kalypso.execution.pipeline_execution import JoinTracker
from vllm.kalypso.execution.worker_pool import Worker, WorkerPool
from vllm.kalypso.sem_ops.joins import CartesianProduct, IndexedCartesianProduct


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.setattr(budget.AutoTokenizer, "from_pretrained", lambda _: None)
    monkeypatch.setattr(budget, "compute_bytes_per_token", lambda *_: 1)
    manager = budget.KVMemoryManager("mock", 100, None)
    manager.register_workers([40, 60])
    manager.register_stage(0, 1.0)
    monkeypatch.setattr(budget.KVMemoryManager, "get_instance", lambda: manager)
    stage = Stage(0, [])
    monkeypatch.setattr(stage, "estimate_budget", lambda _: 20)
    pool = WorkerPool(
        [
            Worker(i, str(i), capacity, AsyncMock())
            for i, capacity in enumerate([40, 60])
        ],
        "mock",
    )
    task = Task(SemContext(state=ExecutionState(executor=pool)), stage_index=0)
    return manager, stage, pool, task


def test_accept_routes_calls_and_release_uses_charged_worker(admission):
    manager, stage, pool, task = admission
    task.ctx.state.worker_id = 1
    pool.worker(1).client.execute.return_value = "hello back"

    async def run():
        assert await stage.accept(task, manager, pool) == 20
        assert task.ctx.state.worker_id == 1
        assert task.ctx.state.executor is pool.worker(1)
        assert manager.worker_usage(1) == (20, 60)
        for _ in range(2):
            assert (
                await task.ctx.state.executor.execute(None, "hello", 8) == "hello back"
            )
        assert pool.worker(1).client.execute.await_count == 2
        pool.worker(0).client.execute.assert_not_awaited()
        task.ctx.state.worker_id = 0
        reservation, worker_id = stage.detach_budget(task)
        assert (reservation, worker_id) == (20, 1)
        await stage.release_budget(reservation, worker_id, manager)
        assert manager.worker_usage(1) == (0, 60)
        assert manager.stage_usage(0) == (0, 100)
        assert stage.detach_budget(task) == (0, 0)

    asyncio.run(run())


@pytest.mark.parametrize("both_full", [False, True])
def test_accept_falls_back_or_waits_when_workers_full(admission, both_full):
    manager, stage, pool, task = admission
    task.ctx.state.worker_id = 0

    async def run():
        await manager.force_allocate_stage(0, 40, 0)
        if both_full:
            await manager.force_allocate_stage(0, 60, 1)
        assert await stage.accept(task, manager, pool) == (None if both_full else 20)
        if both_full:
            assert task.ctx.state.worker_id == 0
            assert task.ctx.state.executor is pool
            assert not stage.running_tasks
            assert manager.stage_usage(0) == (100, 100)
        else:
            assert task.ctx.state.worker_id == 1
            assert task.ctx.state.executor is pool.worker(1)
            assert manager.worker_usage(1) == (20, 60)

    asyncio.run(run())


@pytest.mark.parametrize("forced", [False, True])
def test_retained_reservation_keeps_worker_without_charging_twice(admission, forced):
    manager, stage, pool, task = admission

    async def run():
        task.ctx.state.worker_id = 1
        await stage.accept(task, manager, pool)
        reservation, worker_id = stage.detach_budget(task)
        retry = Task(task.ctx, 0, reserved_budget=reservation)
        accept = stage.force_accept if forced else stage.accept
        assert await accept(retry, manager, pool) == 20
        assert retry.reserved_budget == 0
        assert retry.ctx.state.worker_id == worker_id == 1
        assert retry.ctx.state.executor is pool.worker(1)
        assert manager.worker_usage(1) == (20, 60)
        assert manager.worker_inflight(1) == 1
        assert stage.detach_budget(retry) == (20, 1)

    asyncio.run(run())


@pytest.mark.parametrize("preferred,expected", [(None, 1), (0, 0)])
def test_force_accept_uses_most_free_unless_worker_assigned(
    admission, preferred, expected
):
    manager, stage, pool, task = admission
    task.ctx.state.worker_id = preferred

    async def run():
        assert await stage.force_accept(task, manager, pool) == 20
        assert task.ctx.state.worker_id == expected
        assert task.ctx.state.executor is pool.worker(expected)
        assert stage.detach_budget(task) == (20, expected)
        assert manager.worker_usage(expected)[0] == 20

    asyncio.run(run())


@pytest.mark.parametrize("indexed", [False, True])
def test_join_children_copy_parent_worker(admission, monkeypatch, indexed):
    _, _, pool, task = admission
    task.ctx.input = SemanticInput(data="left")
    task.ctx.state.worker_id = 1
    task.ctx.state.executor = pool.worker(1)
    if indexed:
        join = IndexedCartesianProduct()
        monkeypatch.setattr(
            join,
            "_query_retrieval",
            lambda _: [{"metadata": [0, "right"], "score": 1.0}],
        )
    else:
        join = CartesianProduct([SemContext(input=SemanticInput(data="right"))])
    (child,) = join(task.ctx)
    assert child.state.worker_id == 1
    assert child.state.executor is pool.worker(1)


def test_join_tracker_releases_original_worker(admission):
    manager, stage, pool, task = admission

    async def run():
        task.ctx.state.worker_id = 1
        await stage.accept(task, manager, pool)
        reservation, worker_id = stage.detach_budget(task)
        tracker = JoinTracker(task.ctx, 1, manager, 0, reservation, worker_id=worker_id)
        task.ctx.state.worker_id = 0
        await tracker.child_finished()
        assert manager.worker_usage(1) == (0, 60)
        assert manager.worker_inflight(1) == 0

    asyncio.run(run())
