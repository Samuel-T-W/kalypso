# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot coordinator traces. Counts include failed calls; percentiles interpolate.

Throughput uses each worker's first-start to last-finish span, including gaps.
Missing queue files or null vLLM waiting samples are allowed.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_trace(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(calls: list[dict]) -> tuple[int, float, float, float]:
    latencies = [call["latency"] for call in calls]
    span = (max(c["finish"] for c in calls) - min(c["start"] for c in calls)
            if calls else 0)
    return (len(calls), percentile(latencies, 0.5), percentile(latencies, 0.99),
            len(calls) / span if span > 0 else 0.0)


def completion_bins(calls: list[dict], origin: float, size: int) -> list[int]:
    counts = [0] * size
    for call in calls:
        counts[math.floor(call["finish"] - origin)] += 1
    return counts


def plot_traces(directories: list[Path], out: Path) -> None:
    traces = []
    for directory in directories:
        calls = read_trace(directory / "requests.jsonl")
        queue_path = directory / "queue.jsonl"
        queue = read_trace(queue_path) if queue_path.exists() else []
        traces.append((directory, calls, queue))
    workers = sorted({str(r["worker_id"]) for _, c, q in traces for r in c + q})
    colors = {w: plt.get_cmap("tab10")(i % 10) for i, w in enumerate(workers)}
    fig, axes = plt.subplots(len(traces), 4, figsize=(22, 5 * len(traces)),
                             squeeze=False, layout="constrained")
    for row, (directory, calls, queue) in zip(axes, traces):
        active = [w for w in workers if any(str(r["worker_id"]) == w
                                          for r in calls + queue)]
        origin = min((c["start"] for c in calls),
                     default=min((q["t"] for q in queue), default=0))
        size = math.floor(max((c["finish"] - origin for c in calls),
                              default=0)) + 1
        print(f"\n{directory.name} (all calls, including failures)")
        print("worker requests p50_s p99_s throughput_req_s")
        for position, worker in enumerate(active, 1):
            selected = [c for c in calls if str(c["worker_id"]) == worker]
            samples = sorted((q for q in queue if str(q["worker_id"]) == worker),
                             key=lambda q: q["t"])
            count, p50, p99, rate = summarize(selected)
            print(f"{worker} {count} {p50:.3f} {p99:.3f} {rate:.3f}")
            color = colors[worker]
            row[0].bar(position, count, color=color)
            if selected:
                row[1].boxplot([c["latency"] for c in selected],
                               positions=[position], widths=0.6, patch_artist=True,
                               boxprops={"facecolor": color, "alpha": 0.35},
                               medianprops={"color": color},
                               whiskerprops={"color": color},
                               capprops={"color": color},
                               flierprops={"markeredgecolor": color})
                row[1].plot([], [], color=color,
                            label=f"{worker}: p50={p50:.3f}, p99={p99:.3f} s")
            row[2].step(range(size + 1), completion_bins(selected, origin, size)
                        + [0], where="post", color=color, label=worker)
            row[3].plot([q["t"] - origin for q in samples],
                        [q["admitted"] for q in samples], color=color, marker=".",
                        label=f"{worker} admitted")
            waiting = [q for q in samples if q.get("vllm_waiting") is not None]
            if waiting:
                row[3].plot([q["t"] - origin for q in waiting],
                            [q["vllm_waiting"] for q in waiting], "--",
                            color=color, label=f"{worker} vllm_waiting")
        titles = ["Requests per worker", "Request latency",
                  "Throughput (1 s bins)", "Queue depth"]
        units = ["Requests (all calls)", "Latency (s)",
                 "Completed calls / s", "Requests"]
        for i, ax in enumerate(row):
            prefix = f"{directory.name}: " if len(traces) > 1 else ""
            ax.set_title(prefix + titles[i])
            ax.set_ylabel(units[i])
            ax.set_xlabel("Worker" if i < 2 else "Time since first start (s)")
            ax.grid(axis="y", alpha=0.25)
            if i < 2:
                ax.set_xticks(range(1, len(active) + 1), active)
            if i > 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize="small")
        row[0].set_ylim(bottom=0)
        row[2].set_ylim(bottom=0)
        row[3].set_ylim(bottom=0)
    fig.savefig(out, dpi=150, format="png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, action="extend", nargs="+",
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot_traces(args.trace_dir, args.out)


if __name__ == "__main__":
    main()
