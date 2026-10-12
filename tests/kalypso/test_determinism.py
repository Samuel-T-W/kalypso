# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import re
from collections import Counter
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.controller.plan import (
    DeterminismUnsupportedError,
    SemanticPlan,
    check_determinism,
)
from vllm.kalypso.execution.executor import (
    BOOLEAN,
    Bucket,
    Choice,
    CompletionResult,
    Text,
    apply_determinism,
)


def test_literal_seeded_replies(monkeypatch):
    result = CompletionResult("real reply", "req-1", "length")
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "2")
    flipped = apply_determinism("hello", result, BOOLEAN)
    assert flipped.text == "true"
    assert flipped.request_id == "req-1"
    assert flipped.finish_reason == "stop"
    assert apply_determinism("hello", result, Bucket()).text == "bucket 4"
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "1")
    assert apply_determinism("hello", result, BOOLEAN).text == "false"
    assert apply_determinism("hello", result, Text()).text == "79057cc0"
    assert result.text == "real reply"
    assert result.finish_reason == "length"


@pytest.mark.parametrize("seed", [None, ""])
def test_disabled_returns_same_result(monkeypatch, seed):
    if seed is None:
        monkeypatch.delenv("KALYPSO_COIN_FLIP_SEED", raising=False)
    else:
        monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", seed)
    result = CompletionResult("original\ntext", "req-2", "length")
    assert apply_determinism("hello", result, None) is result


def test_enabled_without_strategy_raises(monkeypatch):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "1")
    with pytest.raises(RuntimeError, match="no determinism strategy"):
        apply_determinism("hello", CompletionResult("x", "id", "stop"), None)


@pytest.mark.parametrize("prompt", ["hello", [{"role": "user", "content": "hello"}]])
def test_prompt_determines_reply_independent_of_call_order(monkeypatch, prompt):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "42")
    first = apply_determinism(
        prompt, CompletionResult("answer A", "req-A", "length"), Text()
    )
    apply_determinism(
        "intervening prompt", CompletionResult("other", "req-B", "stop"), Text()
    )
    second = apply_determinism(
        prompt, CompletionResult("answer C", "req-C", "stop"), Text()
    )
    assert first.text == second.text
    assert re.fullmatch(r"[0-9a-f]{8}", first.text)
    assert first.request_id == "req-A"
    assert second.request_id == "req-C"


def test_chat_dictionary_key_order_is_canonical(monkeypatch):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "-7")
    result = CompletionResult("real reply", "req-chat", "length")
    first = [{"role": "user", "content": "hello"}]
    reordered = [{"content": "hello", "role": "user"}]
    assert (
        apply_determinism(first, result, Text()).text
        == apply_determinism(reordered, result, Text()).text
    )
    different = [{"role": "user", "content": "goodbye"}]
    assert (
        apply_determinism(first, result, Text()).text
        != apply_determinism(different, result, Text()).text
    )


def test_choice_only_returns_options_and_uses_all_of_them(monkeypatch):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "9")
    classes = Choice(["positive", "negative", "neutral"])
    result = CompletionResult("x", "id", "stop")
    replies = Counter(
        apply_determinism(f"row {i}", result, classes).text for i in range(300)
    )
    assert set(replies) == {"positive", "negative", "neutral"}


def test_bucket_spreads_calls_about_one_tenth_each(monkeypatch):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "9")
    result = CompletionResult("x", "id", "stop")
    counts = Counter(
        apply_determinism(f"chunk {i}", result, Bucket()).text for i in range(2000)
    )
    assert set(counts) == {f"bucket {b}" for b in range(10)}
    assert all(150 <= n <= 250 for n in counts.values())


@pytest.mark.parametrize("bad", [lambda: Choice([]), lambda: Bucket(0)])
def test_strategy_rejects_empty_spaces(bad):
    with pytest.raises(ValueError):
        bad()


@pytest.mark.parametrize("client", ["local", "remote"])
@pytest.mark.parametrize("prompt", ["hello", [{"role": "user", "content": "hello"}]])
@pytest.mark.parametrize("seed", [None, "3"])
def test_execute_still_calls_backend(monkeypatch, client, prompt, seed):
    from vllm.kalypso.execution.vllm_executor import VLLMExecutor
    from vllm.kalypso.execution.worker_client import WorkerClient

    executor = (
        VLLMExecutor("model")
        if client == "local"
        else WorkerClient("http://localhost:8000", "model")
    )
    if seed is None:
        monkeypatch.delenv("KALYPSO_COIN_FLIP_SEED", raising=False)
    else:
        monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", seed)
    original = CompletionResult("real reply", "req-1", "length")
    backend = AsyncMock(return_value=original)
    method = "complete" if isinstance(prompt, str) else "chatcomplete"
    monkeypatch.setattr(executor, method, backend)
    request = object()
    result = asyncio.run(executor.execute(request, prompt, 32, False, 7, BOOLEAN))
    backend.assert_awaited_once_with(
        raw_request=request, prompt=prompt, max_tokens=32, pin=False, priority=7
    )
    if seed is None:
        assert result is original
    else:
        assert result.text in ("true", "false")
        assert result.request_id == "req-1"
        assert result.finish_reason == "stop"


@pytest.fixture
def parse(monkeypatch):
    manager = Mock(token_length=Mock(return_value=1))
    monkeypatch.setattr(KVMemoryManager, "get_instance", lambda: manager)
    plan = SemanticPlan(AsyncMock())
    return lambda logical: (logical, plan.parse_ops(logical))


SUPPORTED = [
    {"op": "sem_filter", "args": {"prompt": "p"}},
    {"op": "sem_map", "args": {"prompt": "p"}},
    {"op": "sem_classify", "args": {"classes": ["a", "b"]}},
    {"op": "join", "args": {"instruction": "p"}},
    {"op": "sem_agg", "args": {"instruction": "p"}},
    {"op": "sem_topk", "args": {"instruction": "p", "k": 2}},
]


@pytest.mark.parametrize("seed", [None, "1"])
def test_check_accepts_supported_operators(monkeypatch, parse, seed):
    if seed is None:
        monkeypatch.delenv("KALYPSO_COIN_FLIP_SEED", raising=False)
    else:
        monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", seed)
    logical, physical = parse(SUPPORTED)
    check_determinism(logical, physical)
    llm_ops = [op for op in physical if op.CALLS_LLM]
    assert len(llm_ops) == len(SUPPORTED)
    assert all(op.determinism is not None for op in llm_ops)


def test_check_names_unsupported_operators(monkeypatch, parse):
    logical, physical = parse(
        [
            {"op": "sem_filter", "args": {"prompt": "p"}},
            {"op": "sem_filter", "args": {"prompt": "p", "cascade": True}},
        ]
    )
    monkeypatch.delenv("KALYPSO_COIN_FLIP_SEED", raising=False)
    check_determinism(logical, physical)
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "1")
    with pytest.raises(DeterminismUnsupportedError) as exc:
        check_determinism(logical, physical)
    message = str(exc.value)
    assert "sem_filter at position 1 (CascadeOperator)" in message
    assert "position 0" not in message
