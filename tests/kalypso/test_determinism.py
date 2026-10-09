# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import re
from unittest.mock import AsyncMock

import pytest

from vllm.kalypso.execution.executor import CompletionResult, apply_determinism


def test_literal_seeded_replies(monkeypatch):
    result = CompletionResult("real reply", "req-1", "length")
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "1")
    flipped = apply_determinism("hello", result)
    assert flipped.text == "true 79057cc0"
    assert flipped.request_id == "req-1"
    assert flipped.finish_reason == "stop"
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "3")
    assert apply_determinism("hello", result).text == "false a8d582d6"
    assert result.text == "real reply"
    assert result.finish_reason == "length"


@pytest.mark.parametrize("seed", [None, ""])
def test_disabled_returns_same_result(monkeypatch, seed):
    if seed is None:
        monkeypatch.delenv("KALYPSO_COIN_FLIP_SEED", raising=False)
    else:
        monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", seed)
    result = CompletionResult("original\ntext", "req-2", "length")
    assert apply_determinism("hello", result) is result


@pytest.mark.parametrize("prompt", ["hello", [{"role": "user", "content": "hello"}]])
def test_prompt_determines_reply_independent_of_call_order(monkeypatch, prompt):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "42")
    first = apply_determinism(prompt, CompletionResult("answer A", "req-A", "length"))
    apply_determinism("intervening prompt", CompletionResult("other", "req-B", "stop"))
    second = apply_determinism(prompt, CompletionResult("answer C", "req-C", "stop"))
    assert first.text == second.text
    assert re.fullmatch(r"(?:true|false) [0-9a-f]{8}", first.text)
    assert first.request_id == "req-A"
    assert second.request_id == "req-C"


def test_chat_dictionary_key_order_is_canonical(monkeypatch):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "-7")
    result = CompletionResult("real reply", "req-chat", "length")
    first = [{"role": "user", "content": "hello"}]
    reordered = [{"content": "hello", "role": "user"}]
    assert (
        apply_determinism(first, result).text == apply_determinism(reordered, result).text
    )
    different = [{"role": "user", "content": "goodbye"}]
    assert (
        apply_determinism(first, result).text != apply_determinism(different, result).text
    )


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
    result = asyncio.run(executor.execute(request, prompt, 32, False, 7))
    backend.assert_awaited_once_with(
        raw_request=request, prompt=prompt, max_tokens=32, pin=False, priority=7
    )
    if seed is None:
        assert result is original
    else:
        assert re.fullmatch(r"(?:true|false) [0-9a-f]{8}", result.text)
        assert result.request_id == "req-1"
        assert result.finish_reason == "stop"
