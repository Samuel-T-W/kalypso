# Changes needed for multi-worker Kalypso: first-fit admission with worker affinity

This lists the minimum changes for Kalypso to run a semantic query across several vLLM instances.
Admission picks a worker for each row that has KV room on it, and keeps the row on that worker while it moves through a stage.
Work needed only for later policies, metrics, pipeline experiments, or pinning is left out.
Source facts were checked against this checkout on 2026-10-03.

## How it runs today

`vllm serve` starts an API server process (CPU) and an EngineCore process that owns the GPU, the scheduler, and the KV cache.
The API server creates Kalypso's `QueryProcessor` at startup, so Kalypso lives in the API server process.
Kalypso receives `POST /v1/semantic/query` from the benchmark client and calls the API server's `create_chat_completion` as a function for every operator call.

Kalypso's only imports from vLLM proper are in [vllm_executor.py](../vllm/kalypso/execution/vllm_executor.py).
The planner, operators, admission control, and budget logic are plain Python plus `transformers`.

## How a row moves today

- All rows are loaded into memory before execution and enqueued on the first stage's waiting queue ([pipeline_execution.py](../vllm/kalypso/execution/pipeline_execution.py), `execute_tasks`).
  Operators do not pull rows; a single admission loop pushes them.
- The waiting queue holds rows that have not been admitted.
  Nothing is computed for a queued row and it uses no KV.
  The queue also receives the previous stage's outputs, join fanout children, and retries.
- `launch_ready_tasks` peeks at the head of each stage's queue and calls `stage.accept`, which reserves the stage's estimated KV bytes for that row.
  If the head row does not fit, that stage stops admitting until a running row finishes and frees its reservation.
- The loop waits with `asyncio.wait(..., FIRST_COMPLETED)` on its own running rows, so a returned HTTP response is the event that frees space.
  No callback from vLLM is needed.
- A running row executes the stage's fused operators one at a time in `Stage.run_task`.
  Each operator adds its instruction to the row's data, makes one LLM call through `ctx.state.executor.execute(...)`, and waits for the reply before the next operator runs.
  A filter that rejects the row ends it there.
- The reservation is the peak token count across the stage's operators, not the sum, because each operator extends the same prompt prefix.
  It is held for the whole stage and released when the row leaves the stage; the next stage reserves again.

## Decisions

| Decision | Choice |
| --- | --- |
| Where the Kalypso coordinator runs | A standalone process on CPU, on the same GPU server as the workers, with no model loaded |
| How it reaches workers | One `vllm serve` per GPU group on its own localhost port, called over HTTP |
| Physical pinning | Off (`virtual_pinning=True`, already the default) |
| Models | All workers serve the same model |
| Who picks the worker | The admission loop, when it reserves KV for a row, not the operator and not the executor |
| KV accounting | One ledger per worker; a row is admitted only onto a worker with room |
| Stage shares | Kept global; admission checks both the stage share and the chosen worker's room |
| Worker choice | First fit, trying the row's previous worker first |

```text
benchmark --POST /v1/semantic/query--> coordinator (CPU, no model)
                                       plan, admission picks worker per row,
                                       operators call executor with that worker
                                     +--------------+--------------+
                                     v              v              v
                               vllm serve     vllm serve     vllm serve
                                 :8001          :8002          :8003
```

## Why round robin is not enough

A single summed budget lets admission accept a row when the worker its call lands on is already full.
That worker then evicts cached prefixes or preempts requests Kalypso did not plan for.
Round robin also sends a row's next operator call to a different worker, so the reservation no longer keeps that row's prefix cached.
Fixing both means the reservation and the call must use the same worker, so the worker has to be chosen where the reservation is made.

## Changes

### 1. Worker client

- A worker is a `vllm serve` process.
- New `WorkerClient` (in [worker_client.py](../vllm/kalypso/execution/worker_client.py)) is the Kalypso coordinator's client for one worker, with the same `execute(raw_request, prompt, max_tokens, pin, priority)` method as [VLLMExecutor](../vllm/kalypso/execution/vllm_executor.py).
- It posts to a worker's `/v1/chat/completions` (chat prompts) or `/v1/completions` (string prompts) with an async HTTP client and ignores `raw_request`.
- Send the same fields as `_build_chat_request` and `_build_request` (deterministic sampling, `priority`, `chat_template_kwargs`, the 16,384 `max_tokens` cap) as plain JSON.
- Parse `id`, `choices[0].message.content` or `text`, and `finish_reason` into the existing `CompletionResult`.
- Use one pooled client per worker and a request timeout long enough for the longest generation.
- `unpin` and `abort` can raise `NotImplementedError`: with pinning off no row gets a `pin_req_id`, so operators never call them.

### 2. Per-worker KV ledger

- [budget.py](../vllm/kalypso/budget.py): `KVMemoryManager` keeps its stage shares and adds per-worker capacity and used bytes.
- New `try_allocate(stage_id, budget, preferred)` returns a worker index or `None`.
  It succeeds only if the stage share has room and a worker has room.
  It tries `preferred` first, then the other workers in order (first fit).
- `release_stage` takes the worker index and frees the bytes on both the stage and that worker.
- `force_allocate_stage` picks the worker with the most free bytes.
- `capacity()` stays the sum of worker capacities, so `SemanticPlan.build` sizes stage shares as it does today.
- Each worker's capacity is its own `/v1/semantic/dummy` budget scaled by its tensor-parallel size and `0.5`, as the API server does today.

### 3. Worker assignment at admission

- [context.py](../vllm/kalypso/context/context.py): add `worker_id: int | None = None` to `ExecutionState`.
- [stage.py](../vllm/kalypso/controller/stage.py): `accept` and `force_accept` call the new allocator with `preferred=task.ctx.state.worker_id`, then set `task.ctx.state.worker_id` and `task.ctx.state.executor`.
- Store the worker with the reservation in `Stage.running_tasks`, so the release in [pipeline_execution.py](../vllm/kalypso/execution/pipeline_execution.py) frees the worker that was charged even if the row is reassigned later.
- A retry that keeps its budget (`retain_budget=True`) keeps its worker.
- [joins.py](../vllm/kalypso/sem_ops/joins.py): `CartesianProduct` builds a new `ExecutionState` for each child; copy `worker_id` from the parent so children prefer the parent's worker, which holds the left row's prefix.
- Blocking operators (`sem_agg`, `sem_topk`) gather all rows, so affinity ends there.

### 4. Routing through the row's executor

- Operators already call `ctx.state.executor.execute(...)`, so routing needs no operator change: when admission assigns a worker, it also sets `ctx.state.executor` to that worker's `WorkerClient`.
- New `WorkerPool` holds the `WorkerClient`s, counts requests per worker, and is itself an executor that round-robins.
  Rows start with the pool as their executor; it is used only by calls that skip admission, such as blocking operators (`sem_agg`, `sem_topk`), which today have no KV accounting either.
- It plugs in where `VLLMExecutor` is used today: `QueryProcessor` passes it to `SemanticPlan`, which attaches it to every row in [data/data.py](../vllm/kalypso/data/data.py).

### 5. Standalone Kalypso coordinator entry point

- New CPU-only FastAPI app serving `POST /v1/semantic/query` with the same request and response shape, so benchmark clients do not change.
- Move `_materialize_records_to_csv` and the result formatting from [api_server.py](../vllm/entrypoints/openai/api_server.py) into Kalypso, or copy them, so the coordinator can use them.
- Startup steps:
  1. Read the worker URLs from a CLI flag or config file.
  2. Check each worker with `/v1/models` and require the same model on all of them.
  3. Read each worker's per-GPU KV budget from `/v1/semantic/dummy` (workers run this fork) and scale it as in change 2.
  4. Create `QueryProcessor` with the model name, the per-worker budgets, and the `WorkerPool`.
- Do not start the stuck monitor: it calls engine-core utilities in process, and with pinning off its only action, unpinning, has nothing to do.

### 6. Let QueryProcessor take an executor and worker budgets

- [query_processor.py](../vllm/kalypso/query_processor.py) builds `VLLMExecutor(model=model_name)` itself; add optional executor and per-worker budget arguments so the coordinator can pass them while the in-process server keeps today's behavior as one worker.
- `_cleanup_query_pins` needs no change: with pinning off the registry is empty for every query and the method returns before touching the engine.

### 7. Launch script and correctness check

- Script that starts N workers, each on its own GPUs via `CUDA_VISIBLE_DEVICES` and its own port, with today's flags from `README.md` (`--enable-prefix-caching`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`), then starts the coordinator.
- Each worker loads a full model copy, so pick a model that fits the GPUs per worker (the README's 70B setup uses 4 GPUs per instance; a small model on 1 GPU per worker suits development).
- Point a benchmark client at the coordinator, for example `--endpoint http://localhost:<coordinator port>/v1/semantic/query`.
- Correctness: run the same query through the coordinator with 1 worker and with N workers and compare output rows and verdicts; temperature is 0 and `seed=42`, but batch differences can still change raw text slightly.
- Confirm every operator call for a row within one stage went to the same worker, and that per-worker request counts are roughly even.

## Not in this doc

Smarter worker choice than first fit, utilization and cache metrics, a materialize mode for pipeline experiments, physical pinning across workers, and support for concurrent queries are later steps.
