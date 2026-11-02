import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.context import ExecutionState, SemContext, SemanticInput
from vllm.kalypso.controller.plan import PlanExecutor, SemanticPlan
from vllm.kalypso.instrumentation import current_request_meta, request_meta
from vllm.kalypso.sem_ops.base import BaseOp, OpBehavior
from vllm.kalypso.sem_ops.reducers import SemAgg, SemTopK
from vllm.kalypso.workers.policy import RoundRobinPolicy, RowAffinityPolicy
from vllm.kalypso.workers.router import CallRouter
from vllm.kalypso.workers.stub_pool import StubPool, Worker


@pytest.mark.parametrize("op_type,reply,expected_calls", [
    (SemAgg, "summary", 1), (SemTopK, "A", 2),
])
def test_blocking_llm_calls_inherit_operator_labels(monkeypatch, op_type, reply, expected_calls):
    manager = NS(
        token_length=lambda text: 1, apply_chat_template=str, bytes_per_token=1,
        can_admit=AsyncMock(return_value=True), allocate=AsyncMock(), release=AsyncMock(),
    )
    monkeypatch.setattr(KVMemoryManager, "get_instance", lambda: manager)
    labels = []

    async def llm_call(**kwargs):
        labels.append(current_request_meta())
        return NS(text=reply)

    async def go():
        ctxs = [SemContext(
            input=SemanticInput(data=text),
            state=ExecutionState(executor=NS(execute=llm_call), idx=i),
        ) for i, text in enumerate(["alpha", "beta", "gamma"])]
        op = op_type("summarize or rank", position=2, concurrency=2)
        if isinstance(op, SemTopK):
            op.k = 1
        with request_meta(operator="outer", stage=99, row_idx=7):
            out, stats = await PlanExecutor().execute(ctxs, [op])
            assert current_request_meta() == {"operator": "outer", "stage": 99, "row_idx": 7}
        assert len(out) == 1 and stats == []
        assert labels == [{
            "operator": op_type.__name__, "stage": 2, "position": 2, "row_idx": None,
        }] * expected_calls
        assert current_request_meta() == {}

    asyncio.run(go())


@pytest.mark.parametrize("op_type,reply", [(SemAgg, "summary"), (SemTopK, "A")])
@pytest.mark.parametrize("pin_req_id", [None, "live-pin"])
def test_blocking_calls_use_default_router_unless_row_is_pinned(
    monkeypatch, op_type, reply, pin_req_id,
):
    manager = NS(
        token_length=lambda text: 1, apply_chat_template=str, bytes_per_token=1,
        can_admit=AsyncMock(return_value=True), allocate=AsyncMock(), release=AsyncMock(),
    )
    monkeypatch.setattr(KVMemoryManager, "get_instance", lambda: manager)
    workers = [Worker(
        f"w{i}", NS(execute=AsyncMock(return_value=NS(text=reply))),
        NS(capacity=lambda: 100),
    ) for i in range(2)]
    pool = StubPool(workers)
    router = CallRouter(pool, RoundRobinPolicy(), lambda prompt, max_tokens: 1)
    ctxs = [SemContext(
        input=SemanticInput(data=text),
        state=ExecutionState(
            executor=workers[0].executor, worker=workers[0], pin_req_id=pin_req_id,
        ),
    ) for text in ["alpha", "beta", "gamma"]]
    op = op_type("summarize or rank", position=2, concurrency=2)
    if isinstance(op, SemAgg):
        op.max_tokens = 5  # Reduce alpha/beta, then reduce their summary with gamma.
    else:
        op.k = 1

    async def go():
        plan = SemanticPlan(router, pool=pool, policy=RowAffinityPolicy())
        out, stats = await plan.plan_executor.execute(ctxs, [op])
        assert out[0].input.data[0]["content"] == (
            'CONTEXT:\n  {\n    "text": summary\n  }\n' if op_type is SemAgg
            else 'CONTEXT:\n  {\n    "text": alpha\n  }\n'
        )
        assert len(out) == 1 and stats == []
        if pin_req_id is None:
            assert router.calls == {"w0": 1, "w1": 1}
            assert all(ctx.state.executor is router and ctx.state.worker is None for ctx in ctxs)
            assert [w.executor.execute.await_count for w in workers] == [1, 1]
        else:
            assert router.calls == {"w0": 0, "w1": 0}
            assert all(ctx.state.executor is workers[0].executor for ctx in ctxs)
            assert all(ctx.state.worker is workers[0] for ctx in ctxs)
            assert all(ctx.state.pin_req_id == "live-pin" for ctx in ctxs)
            assert [w.executor.execute.await_count for w in workers] == [2, 0]

    asyncio.run(go())


def test_blocking_metadata_restored_after_exception(monkeypatch):
    monkeypatch.setattr(KVMemoryManager, "get_instance", lambda: object())

    class FailingOp(BaseOp):
        async def __call__(self, ctxs):
            assert current_request_meta() == {
                "operator": "FailingOp", "stage": 4, "position": 4, "row_idx": None,
            }
            raise ValueError("boom")

    async def go():
        with pytest.raises(ValueError, match="boom"):
            await PlanExecutor().execute([], [FailingOp(OpBehavior.BLOCKING, position=4)])
        assert current_request_meta() == {}

    asyncio.run(go())
