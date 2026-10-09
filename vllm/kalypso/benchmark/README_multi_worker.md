# Multi-worker testing (CPU)

Run a 20-row filter + map query across independent CPU vLLM workers and the Kalypso coordinator.
Install a CPU-compatible vLLM environment first; the launcher prefers the repository's `.venv`.
From the repository root:

```bash
KV_GIB=1 KALYPSO_COIN_FLIP_SEED=42 vllm/kalypso/benchmark/launch_multi_worker.sh -n 4 --detach
curl --fail -H 'Content-Type: application/json' --data-binary @vllm/kalypso/benchmark/sample_data/filter_map_20.json http://localhost:8100/v1/semantic/query
vllm/kalypso/benchmark/launch_multi_worker.sh --stop -o ./kalypso_run
```

Omit `--detach` to keep the launcher running; Ctrl-C stops its child processes.
Use `-m MODEL`, `-p POLICY`, `-b BASE_PORT`, `-c COORD_PORT`, and `-o OUT_DIR` to override defaults; append coordinator arguments after `--`.
Defaults are Qwen/Qwen2.5-0.5B-Instruct, round_robin, workers on 8101 onward, coordinator on 8100, and `./kalypso_run` for logs/config/traces.
`KALYPSO_COIN_FLIP_SEED` is inherited by workers and coordinator for reproducible coin-flip testing.
Each CPU worker needs about 1 GB for weights plus `KV_GIB` GiB for KV cache; four workers fit on a 16 GB laptop with `KV_GIB=1`.
The first launch may download model weights; readiness allows 600 seconds per worker and 300 seconds for the coordinator.
