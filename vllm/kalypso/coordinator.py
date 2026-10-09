# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone Kalypso coordinator that serves queries over remote vLLM workers.

Run with:
    python -m vllm.kalypso.coordinator --workers-config workers.json --port 8003
"""

import argparse
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request

from vllm.kalypso import interface
from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.execution.worker_pool import WorkerPool
from vllm.kalypso.query_processor import QueryProcessor
from vllm.kalypso.serving import run_semantic_query


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers-config", required=True)
    parser.add_argument("--policy", choices=WorkerPool.POLICIES, default="round_robin")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--kv-fraction", type=float, default=0.5)
    return parser.parse_args(argv)


def build_app(args: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = await WorkerPool.from_config(
            args.workers_config, args.kv_fraction, args.policy
        )
        app.state.pool = pool
        app.state.query_processor = QueryProcessor(model_name=pool.model, pool=pool)
        if args.trace_dir:
            pool.enable_tracing(args.trace_dir)
            pool.start_queue_sampler(KVMemoryManager.get_instance())
        try:
            yield
        finally:
            await pool.close()

    app = FastAPI(lifespan=lifespan)
    # The budget manager is process-wide and reset per query.
    query_lock = asyncio.Lock()

    @app.post("/v1/semantic/query")
    async def semantic_query(
        sem_request: interface.SemanticQueryRequest, raw_request: Request
    ):
        async with query_lock:
            return await run_semantic_query(
                raw_request.app.state.query_processor, raw_request, sem_request
            )

    @app.get("/v1/semantic/healthz")
    async def healthz(raw_request: Request):
        pool = raw_request.app.state.pool
        return {
            "status": "ok",
            "workers": len(pool),
            "model": pool.model,
            "policy": pool.policy,
        }

    @app.get("/v1/semantic/workers")
    async def workers(raw_request: Request):
        pool = raw_request.app.state.pool
        manager = KVMemoryManager.get_instance()
        return [
            {
                "id": worker.worker_id,
                "name": worker.name,
                "kv_budget": worker.kv_budget,
                "outstanding": worker.outstanding,
                "completed": worker.completed,
                "admitted": manager.worker_inflight(worker.worker_id),
            }
            for worker in pool.workers
        ]

    return app


def main() -> None:
    args = parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
