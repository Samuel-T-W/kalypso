# execution/worker_client.py
from typing import Any

import aiohttp

from .executor import CompletionResult, Determinism, LLMExecutor, apply_determinism


class WorkerClient(LLMExecutor):
    """The coordinator's HTTP client for one vLLM worker.

    A worker is a running `vllm serve` process.
    Mirrors VLLMExecutor's request fields, but does not register pins
    because physical pinning across workers is not supported.
    """

    LOG = False
    MAX_TOKENS_CAP = 16384
    ERROR_BODY_CHARS = 500

    @classmethod
    def _log(cls, message: str):
        if cls.LOG:
            print(message)

    def __init__(self, base_url: str, model: str, timeout_s: float = 3600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        # Created lazily so it is bound to the running event loop.
        # No connection limit: Kalypso admission control bounds concurrency.
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0),
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
            )
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def _sampling_fields(self, max_tokens, pin, priority) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": min(max_tokens, self.MAX_TOKENS_CAP),
            "temperature": 0.0,
            "seed": 42,
            "top_p": 1.0,
            "frequency_penalty": 0.5,
            "repetition_penalty": 1.3,
            "priority": priority,
            "vllm_xargs": {"pinned": pin},
        }

    def _build_chat_request(self, message, max_tokens, pin, priority):
        return {
            "messages": message,
            "chat_template_kwargs": {"enable_thinking": False},
            **self._sampling_fields(max_tokens, pin, priority),
        }

    def _build_request(self, prompt, max_tokens, pin, priority):
        return {
            "prompt": prompt,
            **self._sampling_fields(max_tokens, pin, priority),
        }

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        self._log(f"[worker-client] POST {url}")
        async with self._get_session().post(url, json=payload) as resp:
            if resp.status < 200 or resp.status >= 300:
                body = (await resp.text())[: self.ERROR_BODY_CHARS]
                raise RuntimeError(
                    f"Remote vLLM worker {url} returned HTTP {resp.status}: {body}"
                )
            return await resp.json()

    async def execute(
        self,
        raw_request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0,
        determinism: Determinism | None = None,
    ) -> CompletionResult:
        llm_func = self.complete if type(prompt) is str else self.chatcomplete
        result = await llm_func(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=max_tokens,
            pin=pin,
            priority=priority,
        )
        return apply_determinism(prompt, result, determinism)

    async def chatcomplete(
        self, raw_request, prompt, max_tokens: int, pin: bool = False, priority: int = 0
    ) -> CompletionResult:
        payload = self._build_chat_request(prompt, max_tokens, pin, priority)
        data = await self._post("/v1/chat/completions", payload)
        choice = data["choices"][0]
        return CompletionResult(
            text=choice["message"]["content"],
            request_id=data["id"],
            finish_reason=choice.get("finish_reason", ""),
        )

    async def complete(
        self, raw_request, prompt, max_tokens: int, pin: bool = False, priority: int = 0
    ) -> CompletionResult:
        payload = self._build_request(prompt, max_tokens, pin, priority)
        data = await self._post("/v1/completions", payload)
        choice = data["choices"][0]
        return CompletionResult(
            text=choice["text"],
            request_id=data["id"],
            finish_reason=choice.get("finish_reason", ""),
        )

    async def unpin(self, raw_request, request_id: str):
        raise NotImplementedError(
            "Physical pinning across remote workers is not supported"
        )

    async def abort(self, raw_request, request_id: str):
        raise NotImplementedError(
            "Physical pinning across remote workers is not supported"
        )
