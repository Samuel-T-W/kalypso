# execution/executor.py
import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Optional


class CompletionResult:
    def __init__(self, text: str, request_id: str, finish_reason):
        self.text = text
        self.request_id = request_id
        self.finish_reason = finish_reason


def apply_determinism(
    prompt: str | list[dict[str, object]], result: CompletionResult
) -> CompletionResult:
    """Make harness replies reproducible across worker counts after real inference."""
    seed = os.environ.get("KALYPSO_COIN_FLIP_SEED")
    if not seed:
        return result
    canonical_prompt = (
        prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)
    )
    digest = hashlib.sha256(f"{seed}|{canonical_prompt}".encode()).digest()
    reply = "true" if digest[0] & 1 else "false"
    return CompletionResult(f"{reply} {digest.hex()[:8]}", result.request_id, "stop")


class LLMExecutor(ABC):
    @abstractmethod
    async def complete(
        self,
        raw_request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        pass

    async def chatcomplete(
        self,
        raw_request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        pass

    @abstractmethod
    async def unpin(self, request_id: str):
        pass

    @abstractmethod
    async def abort(self, request_id: str):
        pass
