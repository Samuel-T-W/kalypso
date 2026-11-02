# M3 sweep deviations and temporary changes

Changes made to get the M3 experiment running that differ from the Kalypso paper or from `main`.
Each entry says why it exists and what cleanup looks like.

## Experiment setup (differs from the paper)

| Change | Paper | Ours | Why | Cleanup |
|---|---|---|---|---|
| Model | Llama-3.3-70B-Instruct | Qwen3-8B, thinking disabled | Fits one 48 GB GPU per worker | Report as a deviation in M3 results |
| Hardware | 4x RTX Pro 6000, one model across all GPUs (tensor parallel) | 2x L40S, one model copy per GPU | Multi-worker scheduling needs separate workers | Report KV tokens per worker next to the paper's |
| Worker counts | n/a | N=2 only (`m3_sweep.yaml`) | Cost while the setup stabilizes | Restore `[2, 3, 4]` when ready |
| SemMap retry limit | 8192 tokens | 1024 via `map_retry_max_tokens` | Runaway generations decoded to 8192 tokens (~278s), dominating rep times | Drop once runaways are fixed at the source |
| FEVER execution | n/a | Pipelined (`blocking: False`) | Blocking mode skipped row placement | Keep; blocking was only for the exact-match check |

## Shared Kalypso code changed on this branch

| Change | Where | Why | Cleanup |
|---|---|---|---|
| `chat_template_kwargs: {enable_thinking: false}` on every chat call | `workers/http_executor.py` | Qwen3 otherwise writes long `<think>` blocks | Make it a per-model setting if other models need different template flags |
| Client keep-alive 4s, one retry on `ServerDisconnectedError` | `workers/http_executor.py` | Reused connections that vLLM had closed failed whole queries with 500s | Keep; consider upstreaming |
| `KALYPSO_MAP_RETRY_MAX_TOKENS` env var (default 8192) | `sem_ops/map.py` | Lets the sweep cap SemMap retries without changing the default | Remove with the retry cap |
| row_affinity default executor is a round-robin `CallRouter` | `server.py` | Rows never placed by `admit()` all went to `workers[0]` | Keep; review with Rukmini |
| Blocking operators drop unpinned rows' placed worker and use that router | `execution/pipeline_execution.py`, `controller/plan.py` | SemAgg followed the first row's worker; its summaries have no cache locality | Keep; review with Rukmini |
| `cost_aware` merged from unmerged PR #9 (`rukmini/cost-aware`) | merge commit on `samuel/experiment-sweep` | Needed in the sweep before #9 lands | Rebase onto `main` after #9 merges; drop the merge commit |

## Known issues left as is

- Sampling uses the paper's `frequency_penalty 0.5` and `repetition_penalty 1.3`, which likely cause the gibberish runaways on small models.
  Check `requests.jsonl` for SemMap calls with `finish_reason == "length"` at the retry limit.
- A SemMap retry that hits its limit is accepted as the answer, and its text is appended to the row's context for later operators.
- Calls from blocking operators are labelled `row_idx=None` even when they serve one row (`pipeline_execution.py`).
- Under row_affinity, the router's byte accounting is separate from `admit()`'s per-worker reservations.
- `cost_aware_smoke.yaml` runs Qwen3-8B on an A10 with `max_model_len: 16384`, because about 30k KV tokens fit.
- Rep times vary widely and reps are averaged, as in the paper; report medians or every rep.
