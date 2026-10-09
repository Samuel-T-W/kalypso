# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.kalypso.execution.worker_pool import Worker, WorkerPool


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
                pool = await WorkerPool.from_config(path)
                assert [w.kv_budget for w in pool.workers] == [100, 123]
                assert pool.total_kv_budget() == 223
                await pool.close()

    asyncio.run(run())
