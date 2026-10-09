# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio

import pytest

from vllm.kalypso import budget


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(budget.AutoTokenizer, "from_pretrained", lambda _: None)
    monkeypatch.setattr(budget, "compute_bytes_per_token", lambda *_: 1)
    return budget.KVMemoryManager("mock", 100, None)


def test_two_workers_admit_on_other_worker_and_release(manager):
    async def run():
        manager.register_workers([40, 60])
        manager.register_stage(0, 1.0)
        assert manager.num_workers() == 2
        assert manager.capacity() == 100
        assert manager.stage_usage(0) == (0, 100)
        assert manager.most_free_worker() == 1
        assert await manager.try_allocate_stage(0, 40, worker_id=0)
        assert not await manager.can_admit_stage(0, 10, worker_id=0)
        assert not await manager.try_allocate_stage(0, 10, worker_id=0)
        assert await manager.can_admit_stage(0, 20, worker_id=1)
        assert manager.worker_usage(1) == (0, 60)
        assert await manager.try_allocate_stage(0, 20, worker_id=1)
        assert manager.stage_usage(0) == (60, 100)
        assert manager.worker_usage(0) == (40, 40)
        assert manager.worker_usage(1) == (20, 60)
        assert manager.stage_inflight(0) == 2
        assert manager.worker_inflight(0) == 1
        assert manager.worker_inflight(1) == 1
        await manager.release_stage(0, 40, worker_id=0)
        assert manager.stage_usage(0) == (20, 100)
        assert manager.worker_usage(0) == (0, 40)
        assert manager.worker_inflight(0) == 0
        assert manager.stage_inflight(0) == 1
        assert manager.most_free_worker() == 0

    asyncio.run(run())


def test_one_worker_preserves_default_stage_limit(manager):
    async def run():
        manager.register_stage(0, 0.5)
        assert manager.num_workers() == 1
        assert manager.worker_usage(0) == (0, 100)
        assert await manager.try_allocate_stage(0, 50)
        assert not await manager.can_admit_stage(0, 1)
        assert not await manager.try_allocate_stage(0, 1)
        assert manager.stage_usage(0) == (50, 50)
        assert manager.worker_usage(0) == (50, 100)

    asyncio.run(run())


@pytest.mark.parametrize("fraction,worker_id", [(1.0, 0), (0.4, 1)])
def test_allocate_waits_for_both_levels_and_release_wakes(manager, fraction, worker_id):
    async def run():
        manager.register_workers([40, 60])
        manager.register_stage(0, fraction)
        await manager.allocate_stage(0, 40, worker_id=0)
        started = asyncio.Event()

        async def allocate():
            started.set()
            await manager.allocate_stage(0, 10, worker_id=worker_id)

        waiter = asyncio.create_task(allocate())
        try:
            await started.wait()
            assert not waiter.done()
            assert not await manager.can_admit_stage(0, 10, worker_id=worker_id)
            assert not await manager.try_allocate_stage(0, 10, worker_id=worker_id)
            await manager.release_stage(0, 40, worker_id=0)
            await asyncio.wait_for(waiter, timeout=1)
            assert manager.stage_usage(0) == (10, int(100 * fraction))
            assert manager.worker_usage(worker_id)[0] == 10
            assert manager.worker_inflight(worker_id) == 1
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    asyncio.run(run())


def test_force_allocation_and_release_clamp_both_levels(manager):
    async def run():
        manager.register_workers([40, 60])
        manager.register_stage(0, 0.5)
        await manager.force_allocate_stage(0, 80, worker_id=1)
        assert manager.stage_usage(0) == (80, 50)
        assert manager.worker_usage(1) == (80, 60)
        assert manager.worker_inflight(1) == 1
        await manager.release_stage(0, 100, worker_id=1)
        await manager.release_stage(0, 100, worker_id=1)
        assert manager.stage_usage(0) == (0, 50)
        assert manager.worker_usage(1) == (0, 60)
        assert manager.stage_inflight(0) == manager.worker_inflight(1) == 0

    asyncio.run(run())


def test_register_workers_replaces_lists_and_rejects_empty(manager):
    manager.register_workers([40, 60])
    manager.register_workers([30, 50, 70])
    assert manager.num_workers() == 3
    assert manager.capacity() == 150
    assert manager.worker_usage(2) == (0, 70)
    assert manager.worker_inflight(2) == 0
    with pytest.raises(ValueError, match="empty"):
        manager.register_workers([])
    assert manager.capacity() == 150



def test_oversized_row_runs_only_on_idle_worker(manager):
    async def run():
        manager.register_workers([10, 10])
        manager.register_stage(1, 1.0)
        assert await manager.try_allocate_stage(1, 15, 0)
        assert not await manager.try_allocate_stage(1, 1, 0)
        assert not await manager.try_allocate_stage(1, 15, 0)
        assert await manager.try_allocate_stage(1, 1, 1)

    asyncio.run(run())


def test_reset_workers_clears_leaked_usage(manager):
    async def run():
        manager.register_workers([10, 10])
        manager.register_stage(1, 1.0)
        assert await manager.try_allocate_stage(1, 8, 1)
        manager.reset_workers()
        assert manager.worker_usage(1) == (0, 10)
        assert manager.worker_inflight(1) == 0

    asyncio.run(run())
