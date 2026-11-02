import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.kalypso.context import RETRY_TASK, ExecutionState, SemContext, SemanticInput
from vllm.kalypso.controller.map_estimator import MapRatioEstimator


@pytest.mark.parametrize("limit,expected", [("1024", 1024), (None, 8192)])
def test_map_retry_limit_preserves_estimated_first_attempt(monkeypatch, limit, expected):
    if limit is None:
        monkeypatch.delenv("KALYPSO_MAP_RETRY_MAX_TOKENS", raising=False)
    else:
        monkeypatch.setenv("KALYPSO_MAP_RETRY_MAX_TOKENS", limit)
    # Load a fresh class as a coordinator would, without changing shared imports.
    spec = importlib.util.spec_from_file_location(
        "vllm.kalypso.sem_ops._map_retry_test",
        Path(__file__).resolve().parents[1] / "vllm/kalypso/sem_ops/map.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SemMap.MAX_TOKEN_LIMIT == expected
    manager = NS(token_length=lambda text: 64, apply_chat_template=str)
    monkeypatch.setattr(module.KVMemoryManager, "get_instance", lambda: manager)
    estimator = NS(get_ratio=lambda position: 0.5, update=Mock())
    monkeypatch.setattr(MapRatioEstimator, "instance", lambda: estimator)
    execute = AsyncMock(side_effect=[
        NS(text="truncated", request_id="first", finish_reason="length"),
        NS(text="mapped", request_id="retry", finish_reason="length"),
    ])
    ctx = SemContext(
        input=SemanticInput(data=[{"role": "user", "content": "hello"}]),
        state=ExecutionState(executor=NS(execute=execute)),
    )
    op = module.SemMap("map", position=2)

    async def go():
        assert await op._run_single(ctx) is RETRY_TASK
        assert ctx.state.retry_max_tokens == expected
        assert await op._run_single(ctx) is ctx

    asyncio.run(go())
    assert [call.kwargs["max_tokens"] for call in execute.await_args_list] == [32, expected]
    assert ctx.state.retry_op_position == -1
    assert ctx.state.retry_max_tokens == 0
