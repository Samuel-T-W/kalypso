# Changes needed for multi-worker Kalypso: basic round robin

This lists the minimum changes for Kalypso to run a semantic query across several vLLM instances with round-robin dispatch.
Work needed only for later policies, metrics, affinity, pipeline experiments, or pinning is left out.
Source facts were checked against this checkout on 2026-10-02.

## How it runs today

`vllm serve` starts an API server process (CPU) and an EngineCore process that owns the GPU, the scheduler, and the KV cache.
The API server creates Kalypso's `QueryProcessor` at startup, so Kalypso lives in the API server process.
Kalypso receives `POST /v1/semantic/query` from the benchmark client and calls the API server's `create_chat_completion` as a function for every operator call.

Kalypso's only imports from vLLM proper are in [vllm_executor.py](../vllm/kalypso/execution/vllm_executor.py).
The planner, operators, admission control, and budget logic are plain Python plus `transformers`.

## Decisions

| Decision | Choice |
| --- | --- |
| Where the coordinator runs | A standalone process on CPU, on the same GPU server as the workers, with no model loaded |
| How it reaches workers | One `vllm serve` per GPU group on its own localhost port, called over HTTP |
| Physical pinning | Off (`virtual_pinning=True`, already the default) |
| Models | All workers serve the same model |
| KV budget | Per-worker accounts are the target; for round robin, admission uses the sum of worker budgets |

```text
benchmark --POST /v1/semantic/query--> coordinator (CPU, no model)
                                       Kalypso plan and admission
                                       round-robin dispatch
                                     +--------------+--------------+
                                     v              v              v
                               vllm serve     vllm serve     vllm serve
                                 :8001          :8002          :8003
```

## Changes

### 1. Remote worker executor

- New `RemoteVLLMExecutor` with the same `execute(raw_request, prompt, max_tokens, pin, priority)` method as [VLLMExecutor](../vllm/kalypso/execution/vllm_executor.py), so operators call it unchanged.
- It posts to a worker's `/v1/chat/completions` (chat prompts) or `/v1/completions` (string prompts) with an async HTTP client and ignores `raw_request`.
- Send the same fields as `_build_chat_request` and `_build_request` (deterministic sampling, `priority`, `chat_template_kwargs`, the 16,384 `max_tokens` cap) as plain JSON.
- Parse `id`, `choices[0].message.content` or `text`, and `finish_reason` into the existing `CompletionResult`.
- Use one pooled client per worker and a request timeout long enough for the longest generation.
- `unpin` and `abort` can raise `NotImplementedError`: with pinning off no row gets a `pin_req_id`, so operators never call them.

### 2. Round-robin executor

- New `RoundRobinExecutor` that holds one `RemoteVLLMExecutor` per worker and sends each `execute` call to the next worker in turn.
- Count requests per worker so a run can confirm the load was spread.
- It plugs in where `VLLMExecutor` is used today: `QueryProcessor` passes it to `SemanticPlan`, which attaches it to every row in [data/data.py](../vllm/kalypso/data/data.py).
  No operator file changes.

### 3. Standalone coordinator entry point

- New CPU-only FastAPI app serving `POST /v1/semantic/query` with the same request and response shape, so benchmark clients do not change.
- Move `_materialize_records_to_csv` and the result formatting from [api_server.py](../vllm/entrypoints/openai/api_server.py) into Kalypso, or copy them, so the coordinator can use them.
- Startup steps:
  1. Read the worker URLs from a CLI flag or config file.
  2. Check each worker with `/v1/models` and require the same model on all of them.
  3. Read each worker's per-GPU KV budget from `/v1/semantic/dummy` (workers run this fork) and scale it as the API server does today: `* tensor_parallel_size * 0.5`, with the tensor-parallel size taken from the worker config.
  4. Create `QueryProcessor` with the model name, the summed budget, and the `RoundRobinExecutor`.
- Do not start the stuck monitor: it calls engine-core utilities in process, and with pinning off its only action, unpinning, has nothing to do.

### 4. Let QueryProcessor take an executor

- [query_processor.py](../vllm/kalypso/query_processor.py) builds `VLLMExecutor(model=model_name)` itself; add an optional executor argument so the coordinator can pass the round-robin executor while the in-process server keeps today's behavior.
- `_cleanup_query_pins` needs no change: with pinning off the registry is empty for every query and the method returns before touching the engine.

### 5. Launch script and correctness check

- Script that starts N workers, each on its own GPUs via `CUDA_VISIBLE_DEVICES` and its own port, with today's flags from `README.md` (`--enable-prefix-caching`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`), then starts the coordinator.
- Each worker loads a full model copy, so pick a model that fits the GPUs per worker (the README's 70B setup uses 4 GPUs per instance; a small model on 1 GPU per worker suits development).
- Point a benchmark client at the coordinator, for example `--endpoint http://localhost:<coordinator port>/v1/semantic/query`.
- Correctness: run the same query through the coordinator with 1 worker and with N workers and compare output rows and verdicts; temperature is 0 and `seed=42`, but batch differences can still change raw text slightly.
- Confirm the per-worker request counts are roughly equal.

## Not in this doc

Per-worker budget accounting, load-aware and cost-aware policies, row affinity, utilization and cache metrics, a materialize mode for pipeline experiments, physical pinning across workers, and support for concurrent queries are later steps.
