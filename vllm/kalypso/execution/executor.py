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


def determinism_seed() -> str | None:
    """Return the harness seed, or None when determinism is off."""
    return os.environ.get("KALYPSO_COIN_FLIP_SEED") or None


class Determinism(ABC):
    """How an operator's replies are replaced when determinism is on.

    Each operator declares one so the reply has the shape its parser expects.
    """

    @abstractmethod
    def reply(self, digest: bytes) -> str:
        pass


class Choice(Determinism):
    """Pick one option uniformly, for example true/false or a class name."""

    def __init__(self, options: list[str]):
        if not options:
            raise ValueError("Choice determinism needs at least one option")
        self.options = list(options)

    def reply(self, digest: bytes) -> str:
        return self.options[int.from_bytes(digest[:8], "big") % len(self.options)]


class Bucket(Determinism):
    """Hash each call into a fixed number of buckets.

    With the default of 10 buckets, each bucket holds about one tenth of the
    calls on average; skew is acceptable.
    """

    def __init__(self, num_buckets: int = 10):
        if num_buckets < 1:
            raise ValueError("Bucket determinism needs at least one bucket")
        self.num_buckets = num_buckets

    def reply(self, digest: bytes) -> str:
        return f"bucket {int.from_bytes(digest[:8], 'big') % self.num_buckets}"


class Text(Determinism):
    """Free text that is unique per prompt, for operators like sem_map."""

    def reply(self, digest: bytes) -> str:
        return digest.hex()[:8]


BOOLEAN = Choice(["true", "false"])


def apply_determinism(
    prompt: str | list[dict[str, object]],
    result: CompletionResult,
    determinism: Determinism | None,
) -> CompletionResult:
    """Make harness replies reproducible across worker counts after real inference."""
    seed = determinism_seed()
    if seed is None:
        return result
    if determinism is None:
        raise RuntimeError(
            "Determinism is on but this LLM call has no determinism strategy"
        )
    canonical_prompt = (
        prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)
    )
    digest = hashlib.sha256(f"{seed}|{canonical_prompt}".encode()).digest()
    return CompletionResult(determinism.reply(digest), result.request_id, "stop")


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
