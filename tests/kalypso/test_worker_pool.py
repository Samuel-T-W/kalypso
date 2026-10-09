# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from unittest.mock import AsyncMock

import pytest

from vllm.kalypso.execution.worker_pool import Worker


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
