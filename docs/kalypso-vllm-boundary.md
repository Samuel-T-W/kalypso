# Kalypso and vLLM boundary

This is a source map for this checkout, not a proposed multi-engine design.
Read it for semantic queries, model selection, request routing, scheduling, KV memory, pinning, or multiple vLLM instances.
Recheck linked code before changing behavior because these are implementation facts, not stable APIs.

## Ownership and runtime path

Kalypso owns semantic query parsing, operator plans, row contexts, stage admission estimates, and its inference calls.
vLLM owns model loading, OpenAI serving handlers, tokenization of submitted requests, request scheduling, actual KV blocks, and generation.
The integration is currently **in process** and **bidirectional at the API server**: `vllm/entrypoints/openai/api_server.py` imports Kalypso and creates its `QueryProcessor`; Kalypso's `VLLMExecutor` calls functions in that same API server.
The lower vLLM engine does not parse semantic plans or know Kalypso's operators, but this fork includes pinning and scheduler utility hooks used by Kalypso.

```text
client -> POST /v1/semantic/query -> API server -> QueryProcessor
       -> SemanticPlan -> PlanExecutor -> semantic operators
       -> VLLMExecutor -> create_chat_completion/create_completion
       -> vLLM serving handler -> engine/scheduler/KV cache
                              <- response body + request ID
```

1. [The semantic request schema](../vllm/kalypso/interface.py) accepts `ops` and exactly one of `data_path` or `records`; `text_column` applies to inline row objects and `model_name` is declared.
2. [The API route](../vllm/entrypoints/openai/api_server.py) validates that choice, writes inline records to a temporary CSV, wraps `ops` and the path in `Query`, and uses `raw_request.app.state.query_processor`.
3. [The processor](../vllm/kalypso/query_processor.py) constructs a `SemanticPlan` per query with one stored `VLLMExecutor`; it cleans up pins owned by the raw request in `finally`.
4. [The plan](../vllm/kalypso/controller/plan.py) reads input rows, constructs physical operators and stages, registers stage budgets, then invokes `PlanExecutor`.
5. [The executor adapter](../vllm/kalypso/execution/vllm_executor.py) builds vLLM `ChatCompletionRequest` or `CompletionRequest`, calls the API handlers directly, decodes the JSON body, and returns `CompletionResult(text, request_id, finish_reason)`.
6. [The semantic route](../vllm/entrypoints/openai/api_server.py) converts output contexts into rows with `idx`, input context, right input context, and operator outputs.

`/v1/semantic/query_ref` is present in the API server but refers to absent or incompatible `QueryProcessor` members; it is not the operational path above.

## Cross-boundary interfaces

| Boundary | Current contract and source | Consequence |
| --- | --- | --- |
| Client to Kalypso | `SemanticQueryRequest` and `/v1/semantic/query` in [interface.py](../vllm/kalypso/interface.py) and [api_server.py](../vllm/entrypoints/openai/api_server.py) | `data_path` must be server-visible; inline `records` become CSV before planning. |
| Startup to Kalypso | API server reads `state.engine_client.vllm_config.model_config.model`, `tensor_parallel_size`, and `get_kv_cache_budget`; initializes one `QueryProcessor` in `app.state` | One process-wide model name, adapter, and KV budget manager are created at startup. |
| Kalypso to serving | `VLLMExecutor.execute` selects chat for non-`str` prompts and completion for `str`; builds protocol requests and passes the same FastAPI `raw_request` to `create_chat_completion` or `create_completion` | The adapter depends on in-process serving functions, their response body shape, and FastAPI app state. |
| Kalypso to engine | `raw_request.app.state.engine_client.abort(id)` and `engine_client.engine_core.call_utility_async(method, args)` | Pin cleanup and scheduler inspection depend on vLLM internals rather than a separate Kalypso service API. |
| vLLM extension | `vllm_xargs={"pinned": pin}` becomes sampling `extra_args`; [v1/request.py](../vllm/v1/request.py) sets `Request.pinned` | Pinning needs this fork's request, scheduler, KV manager, and block-pool changes. |
| Optional external services | [CascadeOperator](../vllm/kalypso/sem_ops/filters.py) calls a configured OpenAI-compatible vLLM server through LiteLLM; [IndexedCartesianProduct](../vllm/kalypso/sem_ops/joins.py) calls the separate ICP service | These are operator-specific helpers, not general routing of the main executor to multiple vLLM engines. |

The semantic request's `model_name` is **not used for routing** by `/v1/semantic/query`.
The response reports `processor.model_name`, and `VLLMExecutor.model` is fixed from the loaded model at startup.
The normal adapter sets deterministic generation defaults, `priority`, `vllm_xargs.pinned`, and caps `max_tokens` at 16,384 in [vllm_executor.py](../vllm/kalypso/execution/vllm_executor.py).
Its response parser expects a nonstreaming JSON body with `id`, `choices[0]`, and either `message.content` or `text`.
The adapter passes a priority value, but whether it changes scheduling depends on the vLLM server's scheduling policy.

### Which operators make inference calls

| Operator path | Main vLLM use |
| --- | --- |
| [SemFilter](../vllm/kalypso/sem_ops/filters.py) and join predicates | Submit a full chat prompt, parse TRUE/FALSE text, and optionally retain or release its request pin. |
| [SemMap](../vllm/kalypso/sem_ops/map.py) | Submit a chat prompt, learn an output-length ratio, and retry with more tokens after a `length` finish reason. |
| [SemClassify](../vllm/kalypso/sem_ops/classify.py) | Submit a data-only call and a classification call; it can pin the data call while using the second response as the class. |
| [SemAgg and SemTopK](../vllm/kalypso/sem_ops/reducers.py) | Submit aggregation chunks or pairwise comparisons through the row's stored executor. |
| [CascadeOperator](../vllm/kalypso/sem_ops/filters.py) | Call its configured external vLLM server, with an optional fallback to the main executor. |

[sem_ops/endpoint.py](../vllm/kalypso/sem_ops/endpoint.py) contains an older direct-completion helper with `pin_kv`; the active adapter uses `vllm_xargs.pinned` instead.

## Kalypso state that models or assumes vLLM

| Scope and owner | State or behavior | vLLM assumption |
| --- | --- | --- |
| Process: [QueryProcessor](../vllm/kalypso/query_processor.py) | `model_name`, `virtual_pinning`, one `VLLMExecutor`, stuck-monitor task and cooldown/counters | One loaded model and one engine client serve all normal semantic operators in this API process. |
| Process: [KVMemoryManager](../vllm/kalypso/budget.py) | Singleton tokenizer, `bytes_per_token`, `_capacity`, `_global_used`, stage capacities/usage/inflight, `asyncio.Condition` | Hugging Face model config/tokenizer and a single aggregate KV budget approximate the serving engine. This is admission accounting, not vLLM block ownership. |
| Process: [PinnedRequestRegistry](../vllm/kalypso/pin_registry.py) | Singleton mapping of vLLM request ID to kind, requested tokens, raw-request owner key, and pin time | Request IDs are enough to unpin on the same engine; the registry has no engine identity or actual block count. |
| Process: [MapRatioEstimator](../vllm/kalypso/controller/map_estimator.py) | Per-operator-position output/input token ratios; updated by `SemMap`, reset after successful plan execution | Future output budget estimates reflect completions from the loaded model. No model key separates ratios. |
| Query: [SemanticPlan](../vllm/kalypso/controller/plan.py) | Physical operators, stage IDs, stage capacity fractions, and `virtual_pinning` | All stage estimates spend the same KV budget. Default `virtual_pinning=True` clears operator pin/unpin flags; disabling it assigns them along tuple-independent chains. |
| Query: [PlanExecutor](../vllm/kalypso/execution/pipeline_execution.py) and [Stage](../vllm/kalypso/controller/stage.py) | Waiting/running tasks, per-task reserved bytes, stage admission and rebalancing, retries, fanout parent retention, priorities | Planned concurrency and task budgets approximate pressure on one vLLM scheduler/KV cache. |
| Row: [ExecutionState](../vllm/kalypso/context/context.py) | `raw_request`, `executor`, `pin_req_id`, `stage_id`, retry position/token limit, predicate and helper score | A row's next inference and any unpin/abort use its stored request and adapter, with only a request ID identifying a pin. |
| Row: `SemanticInput`/`SemContext` | Prompt messages, token length, joined right-side data, outputs | Prompt and estimate state are kept outside vLLM; vLLM receives a fresh request per operator call. |
| Operator: [semantic operators](../vllm/kalypso/sem_ops/) | Prompt templates, max token estimates, `pin`/`unpin` flags, output parsing, retry on `finish_reason="length"` | Operators depend on the model's chat template, generated text shape, request ID, and finish reason. |

[compute_bytes_per_token](../vllm/kalypso/budget.py) derives a per-token estimate from model layers, KV heads, head dimension, and configured dtype; it accounts for full-attention layers in hybrid models.
`KVMemoryManager` uses `AutoTokenizer` to estimate prompt lengths and render chat templates separately from the tokenizer used by vLLM's serving path.
At startup the API server computes Kalypso's capacity as `get_kv_cache_budget() * tensor_parallel_size * 0.5`.
The core utility returns `available_gpu_memory_for_kv_cache`; Kalypso does not read actual free blocks for each admission decision.
The API server comment explicitly says this budget does not reflect other requests' usage.
`register_stage` mutates the singleton's stage map when a plan is built, so the state is process-wide even though plans are per query.

## Pin and request lifecycle

1. Operators call `ctx.state.executor.execute(raw_request, prompt, max_tokens, pin, priority)`; [stage.py](../vllm/kalypso/controller/stage.py) derives priorities from stage and operator position.
2. For a pinned call, `VLLMExecutor` adds the returned vLLM request ID to `PinnedRequestRegistry` under `str(id(raw_request))`.
3. In vLLM, the request's `pinned` bit comes from sampling `extra_args`; [kv_cache_manager.py](../vllm/v1/core/kv_cache_manager.py) marks cached blocks pinned, and [scheduler.py](../vllm/v1/core/sched/scheduler.py) retains finished pinned requests' blocks.
4. A semantic operator or [JoinTracker](../vllm/kalypso/execution/pipeline_execution.py) may unpin a row's `pin_req_id`; `VLLMExecutor.unpin` calls engine-core `unpin_request` and removes the local registry entry.
5. On query exit, `QueryProcessor` calls engine-core `unpin_requests` for entries owned by that HTTP request, then removes them locally if the utility call succeeds.
6. Its background monitor asks engine-core for `get_scheduler_state`; after repeated stuck reports or RPC timeouts, it unpins all registry entries subject to a cooldown.

With the current default `virtual_pinning=True`, plan parsing clears pin/unpin flags; this lifecycle becomes active when callers select physical pinning or another path sets `pin=True`.

Engine-core `get_scheduler_state` reports running/waiting counts and KV usage; it calls the scheduler stuck when waiting requests exist and usage is at least 0.99.
The API also exposes `/v1/semantic/healthz`, `/v1/semantic/pinned`, `/v1/semantic/scheduler_state`, and `/v1/semantic/pinned/unpin` for status or manual cleanup in [api_server.py](../vllm/entrypoints/openai/api_server.py).
The registry is a local bookkeeping view; the engine KV manager is the source of truth for blocks.
The code does not tag registry entries, stage budgets, or row pins with an engine identity.

## Boundaries to revisit for multiple vLLM instances

- Choose whether the semantic API remains co-located with one engine or calls engines over a remote interface; the current adapter requires a FastAPI `Request` from its own API server.
- Route each operator call to a specific engine/model and carry that identity through `ExecutionState`, `CompletionResult`, pinned-request records, cleanup, and retries.
- Make model-specific tokenizer, chat template, bytes-per-token estimate, and output-ratio state correspond to the selected engine instead of one process-wide singleton.
- Define capacity and admission per engine from its actual configuration and competing load; the current half-budget estimate is shared across all semantic queries in one process.
- Send unpin, abort, and scheduler-state checks to the engine that created the request ID; the current monitor and registry assume one engine client.
- Decide how priority and pinning behave over a remote serving boundary; these hooks currently depend on this fork's `vllm_xargs` and engine-core utility methods.
- Keep the cascade helper separate in the design: its configured external vLLM endpoint does not participate in the main adapter's pin registry, budget manager, or engine cleanup.

Useful first files for a refresh: [API startup/routes](../vllm/entrypoints/openai/api_server.py), [processor](../vllm/kalypso/query_processor.py), [adapter](../vllm/kalypso/execution/vllm_executor.py), [budget](../vllm/kalypso/budget.py), [plan](../vllm/kalypso/controller/plan.py), [row state](../vllm/kalypso/context/context.py), [pin registry](../vllm/kalypso/pin_registry.py), [engine utilities](../vllm/v1/engine/core.py), and [KV cache manager](../vllm/v1/core/kv_cache_manager.py).
