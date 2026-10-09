#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
usage() {
    echo 'Usage: launch_multi_worker.sh -n N [-m MODEL] [-p POLICY] [-b BASE_PORT] [-c COORD_PORT] [-o OUT_DIR] [--detach] [-- extra coordinator args]'
    echo '       launch_multi_worker.sh --stop [-o OUT_DIR]'
}
die() { echo "$*" >&2; exit 1; }
n=0 model=Qwen/Qwen2.5-0.5B-Instruct policy=round_robin base=8100 coord=8100 out=./kalypso_run detach=0 stop=0
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --detach) detach=1; shift ;;
        --stop) stop=1; shift ;;
        --) shift; break ;;
        -n|-m|-p|-b|-c|-o)
            (($# >= 2)) || die "Missing value for $1"
            case "$1" in
                -n) n=$2 ;; -m) model=$2 ;; -p) policy=$2 ;;
                -b) base=$2 ;; -c) coord=$2 ;; -o) out=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done
if ((stop)); then
    [[ -f "$out/pids" ]] || die "No PID file: $out/pids"
    while read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]] || die "Invalid PID: $pid"
        kill "$pid" 2>/dev/null || true
    done < "$out/pids"
    rm -f "$out/pids"
    exit 0
fi
[[ "$n" =~ ^[1-9][0-9]*$ && "$base" =~ ^[1-9][0-9]*$ && "$coord" =~ ^[1-9][0-9]*$ ]] || die 'N and ports must be positive integers'
((n <= 65534 && base <= 65535 - n && coord <= 65535)) || die 'Ports must be at most 65535'
((coord <= base || coord > base + n)) || die 'Coordinator port overlaps worker ports'
[[ ! -e "$out/pids" ]] || die "Run --stop first: $out/pids already exists"
mkdir -p "$out"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python=python vllm=vllm
[[ ! -x .venv/bin/python ]] || python="$PWD/.venv/bin/python"
[[ ! -x .venv/bin/vllm ]] || vllm="$PWD/.venv/bin/vllm"
command -v curl >/dev/null || die 'curl is required'
threads=$("$python" -c 'import os,sys; print(max(1, (os.cpu_count() or 1) // int(sys.argv[1])))' "$n")
pids=()
cleanup() {
    if ((${#pids[@]})); then kill "${pids[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready() {
    local url=$1 timeout=$2 log=$3 pid=$4 deadline=$((SECONDS + $2)) code
    while ((SECONDS < deadline)); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$url") || code=000
        [[ "$code" != 200 ]] || return 0
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    tail -n 30 "$log" >&2
    die "Readiness failed (timeout ${timeout}s or process exited): $url"
}
for ((i=1; i<=n; i++)); do
    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE=${KV_GIB:-1} OMP_NUM_THREADS=$threads nohup "$vllm" serve "$model" --enable-prefix-caching --max-model-len 4096 --port "$((base+i))" > "$out/worker_$i.log" 2>&1 < /dev/null &
    pids+=("$!")
done
for ((i=1; i<=n; i++)); do ready "http://localhost:$((base+i))/v1/models" 600 "$out/worker_$i.log" "${pids[i-1]}"; done
"$python" -c 'import json,sys; json.dump({"workers": [{"url": f"http://localhost:{int(sys.argv[1])+i}"} for i in range(1,int(sys.argv[2])+1)]}, sys.stdout)' "$base" "$n" > "$out/workers.json"
nohup "$python" -m vllm.kalypso.coordinator --workers-config "$out/workers.json" --policy "$policy" --port "$coord" --trace-dir "$out/trace" "$@" > "$out/coordinator.log" 2>&1 < /dev/null &
pids+=("$!")
ready "http://localhost:$coord/v1/semantic/healthz" 300 "$out/coordinator.log" "$!"
echo "http://localhost:$coord/v1/semantic/query"
if ((detach)); then
    printf '%s\n' "${pids[@]}" > "$out/pids"
    trap - EXIT INT TERM
    exit 0
fi
wait
