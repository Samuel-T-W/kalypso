# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi.testclient import TestClient

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.coordinator import build_app, parse_args

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(autouse=True)
def offline_and_fresh_manager(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    KVMemoryManager._instance = None
    yield
    KVMemoryManager._instance = None


def fake_worker(calls: list[str]) -> web.Application:
    async def models(request):
        return web.json_response({"data": [{"id": MODEL}]})

    async def dummy(request):
        return web.json_response({"budget": 1_000_000_000})

    async def chat(request):
        calls.append("chat")
        return web.json_response(
            {
                "id": "chat-1",
                "choices": [
                    {
                        "message": {"content": "true"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
        )

    async def completions(request):
        calls.append("completion")
        return web.json_response(
            {
                "id": "cmpl-1",
                "choices": [{"text": "true", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
        )

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/semantic/dummy", dummy)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/completions", completions)
    return app


@pytest.fixture
def worker_urls():
    """Starts two fake workers on a background loop and yields their urls and calls."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    calls: list[list[str]] = [[], []]
    servers: list[TestServer] = []

    async def start():
        for worker_calls in calls:
            server = TestServer(fake_worker(worker_calls))
            await server.start_server()
            servers.append(server)

    async def stop():
        for server in servers:
            await server.close()

    asyncio.run_coroutine_threadsafe(start(), loop).result(timeout=10)
    yield [str(server.make_url("")).rstrip("/") for server in servers], calls
    asyncio.run_coroutine_threadsafe(stop(), loop).result(timeout=10)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=10)


def test_coordinator_serves_query_over_workers(worker_urls, tmp_path):
    urls, calls = worker_urls
    config = tmp_path / "workers.json"
    config.write_text(json.dumps({"workers": [{"url": url} for url in urls]}))
    args = parse_args(["--workers-config", str(config), "--policy", "round_robin"])

    with TestClient(build_app(args)) as client:
        health = client.get("/v1/semantic/healthz").json()
        assert health == {
            "status": "ok",
            "workers": 2,
            "model": MODEL,
            "policy": "round_robin",
        }

        body = {
            "ops": [{"op": "sem_filter", "args": {"prompt": "Is it a fruit?"}}],
            "records": ["apple", "pear", "plum", "fig"],
        }
        response = client.post("/v1/semantic/query", json=body)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["num_output_rows"] == 4
        assert payload["model_name"] == MODEL
        assert set(payload) >= {"request_id", "latency_sec", "results", "ops"}

        workers = client.get("/v1/semantic/workers").json()
        assert [w["id"] for w in workers] == [0, 1]
        assert all(
            set(w)
            == {"id", "name", "kv_budget", "outstanding", "completed", "admitted"}
            for w in workers
        )
        assert all(w["kv_budget"] > 0 and w["outstanding"] == 0 for w in workers)
        assert sum(w["completed"] for w in workers) == len(calls[0]) + len(calls[1])

    assert calls[0] and calls[1]


def test_query_requires_exactly_one_input(worker_urls, tmp_path):
    urls, _ = worker_urls
    config = tmp_path / "workers.json"
    config.write_text(json.dumps({"workers": [{"url": urls[0]}]}))
    args = parse_args(["--workers-config", str(config)])

    with TestClient(build_app(args)) as client:
        response = client.post(
            "/v1/semantic/query",
            json={"ops": [{"op": "sem_filter", "args": {"prompt": "x"}}]},
        )
        assert response.status_code == 400


def test_determinism_shapes_replies_and_rejects_unsupported_ops(
    worker_urls, tmp_path, monkeypatch
):
    urls, calls = worker_urls
    config = tmp_path / "workers.json"
    config.write_text(json.dumps({"workers": [{"url": url} for url in urls]}))
    args = parse_args(["--workers-config", str(config)])
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "7")
    records = [f"item {i}" for i in range(12)]

    with TestClient(build_app(args)) as client:
        response = client.post(
            "/v1/semantic/query",
            json={
                "ops": [
                    {"op": "sem_filter", "args": {"prompt": "Keep it?"}},
                    {"op": "sem_classify", "args": {"classes": ["red", "blue"]}},
                ],
                "records": records,
            },
        )
        assert response.status_code == 200, response.text
        rows = response.json()["results"]
        assert 0 < len(rows) < len(records)
        for row in rows:
            verdict, group = (o["value"] for o in row["outputs"])
            assert verdict == "true"
            assert group in ("red", "blue")

        before = len(calls[0]) + len(calls[1])
        response = client.post(
            "/v1/semantic/query",
            json={
                "ops": [
                    {"op": "sem_filter", "args": {"prompt": "x", "cascade": True}}
                ],
                "records": records,
            },
        )
        assert response.status_code == 400
        assert "sem_filter at position 0 (CascadeOperator)" in response.json()["detail"]
        assert len(calls[0]) + len(calls[1]) == before
