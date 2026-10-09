# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp

from .executor import CompletionResult, LLMExecutor
from .worker_client import WorkerClient


class Worker(LLMExecutor):
    """One executor with an identity, planning budget, and call counters."""

    def __init__(self, worker_id: int, name: str, kv_budget: int, client: LLMExecutor):
        self.worker_id = worker_id
        self.name = name
        self.kv_budget = kv_budget
        self.client = client
        self.outstanding = 0
        self.completed = 0

    async def execute(
        self,
        raw_request: Any,
        prompt: Any,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0,
    ) -> CompletionResult:
        self.outstanding += 1
        try:
            result = await self.client.execute(
                raw_request, prompt, max_tokens, pin, priority
            )
            self.completed += 1
            return result
        finally:
            self.outstanding -= 1

    async def complete(self, *args: Any, **kwargs: Any) -> CompletionResult:
        return await self.client.complete(*args, **kwargs)

    async def chatcomplete(self, *args: Any, **kwargs: Any) -> CompletionResult:
        return await self.client.chatcomplete(*args, **kwargs)

    async def unpin(self, raw_request: Any, request_id: str) -> None:
        await self.client.unpin(raw_request, request_id)

    async def abort(self, raw_request: Any, request_id: str) -> None:
        await self.client.abort(raw_request, request_id)

    async def close(self) -> None:
        if hasattr(self.client, "close"):
            await self.client.close()


class LocalWorker(Worker):
    """A worker using the in-process vLLM serving handlers."""

    def __init__(self, worker_id: int, model: str, kv_budget: int):
        from .vllm_executor import VLLMExecutor

        super().__init__(worker_id, "local", kv_budget, VLLMExecutor(model=model))


class RemoteWorker(Worker):
    """A worker reached through its OpenAI-compatible HTTP server."""

    def __init__(self, worker_id: int, base_url: str, model: str, kv_budget: int):
        super().__init__(worker_id, base_url, kv_budget, WorkerClient(base_url, model))


class WorkerPool(LLMExecutor):
    """An ordered collection of workers serving one model."""

    def __init__(self, workers: list[Worker], model: str):
        self.workers = workers
        self.model = model

    def __len__(self) -> int:
        return len(self.workers)

    def worker(self, worker_id: int) -> Worker:
        return self.workers[worker_id]

    def total_kv_budget(self) -> int:
        return sum(worker.kv_budget for worker in self.workers)

    async def execute(self, *args: Any, **kwargs: Any) -> CompletionResult:
        return await self.workers[0].execute(*args, **kwargs)

    async def complete(self, *args: Any, **kwargs: Any) -> CompletionResult:
        return await self.workers[0].complete(*args, **kwargs)

    async def chatcomplete(self, *args: Any, **kwargs: Any) -> CompletionResult:
        return await self.workers[0].chatcomplete(*args, **kwargs)

    async def unpin(self, raw_request: Any, request_id: str) -> None:
        await self.workers[0].unpin(raw_request, request_id)

    async def abort(self, raw_request: Any, request_id: str) -> None:
        await self.workers[0].abort(raw_request, request_id)

    async def close(self) -> None:
        await asyncio.gather(
            *(worker.close() for worker in self.workers), return_exceptions=True
        )

    @classmethod
    async def from_config(
        cls, path: str | Path, kv_fraction: float = 0.5
    ) -> "WorkerPool":
        entries = json.loads(Path(path).read_text()).get("workers", [])
        if not entries or any(not entry.get("url") for entry in entries):
            raise ValueError("Config requires a nonempty workers list with a url each")
        workers: list[Worker] = []
        model = ""
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for index, entry in enumerate(entries):
                url = entry["url"].rstrip("/")
                try:
                    async with session.get(f"{url}/v1/models") as resp:
                        if resp.status != 200:
                            raise ValueError(f"models returned HTTP {resp.status}")
                        worker_model = (await resp.json())["data"][0]["id"]
                        if not isinstance(worker_model, str) or not worker_model:
                            raise ValueError("missing model id")
                    budget = entry.get("kv_budget_bytes")
                    if "kv_budget_bytes" not in entry:
                        async with session.post(f"{url}/v1/semantic/dummy") as resp:
                            resp.raise_for_status()
                            budget = int((await resp.json())["budget"] * kv_fraction)
                    if not isinstance(budget, int) or budget <= 0:
                        raise ValueError(f"invalid KV budget {budget!r}")
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    ValueError,
                    LookupError,
                    TypeError,
                ) as exc:
                    raise RuntimeError(f"Worker {url} startup failed: {exc}") from exc
                if workers and worker_model != model:
                    raise RuntimeError(
                        f"Model mismatch: {workers[0].name} ({model}) "
                        f"and {url} ({worker_model})"
                    )
                model = worker_model
                workers.append(RemoteWorker(index, url, model, budget))
        return cls(workers, model)

    @classmethod
    def local(cls, model: str, kv_budget: int) -> "WorkerPool":
        return cls([LocalWorker(0, model, kv_budget)], model)
