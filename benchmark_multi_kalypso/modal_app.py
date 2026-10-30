"""Thin optional backend. Importing this module requires Modal."""

import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
VLLM_BASE = "2c0ee0fde8db447bd6e7d3ea0d49070661a243fa"
WHEEL = (
    f"https://wheels.vllm.ai/{VLLM_BASE}/"
    "vllm-0.13.0rc5.dev1%2Bg2c0ee0fde-cp38-abi3-manylinux_2_31_x86_64.whl"
)
BUILD_DEPS = ["setuptools>=77.0.3,<81", "setuptools-scm>=8", "packaging>=24.2"]
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl")
    # The upstream wheel brings matching torch and dependencies in a cached layer.
    .uv_pip_install(WHEEL, *BUILD_DEPS, "wheel", "jinja2", "faiss-cpu", "pyyaml")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_dir(
        REPO,
        "/repo",
        copy=True,
        ignore=[".git", "**/__pycache__", "**/*.so", "build", "kalypso_modal_run"],
    )
    # vLLM's standard source install: the fork's Python plus the wheel's kernels.
    .run_commands(
        "cd /repo && VLLM_USE_PRECOMPILED=1 "
        f"VLLM_PRECOMPILED_WHEEL_LOCATION={WHEEL} "
        "VLLM_VERSION_OVERRIDE=0.13.0rc5.dev1+kalypso "
        "python -m pip install --no-deps --no-build-isolation -e ."
    )
)
hf_cache = modal.Volume.from_name("kalypso-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("kalypso-runs", create_if_missing=True)
app = modal.App("kalypso-m3-sweep", image=image)


@app.function(timeout=24 * 3600, volumes={"/cache": hf_cache, "/runs": runs})
def run_n(config_text: str, n: int, commit: str, stamp: str) -> int:
    Path("/repo/benchmark_multi_kalypso/source_commit.txt").write_text(commit + "\n")
    config = Path("/tmp/config.yaml")
    config.write_text(config_text, encoding="utf-8")
    try:
        return subprocess.run(
            [
                "python",
                "-m",
                "benchmark_multi_kalypso.runner",
                "--config",
                str(config),
                "--n",
                str(n),
                "--gpus",
                ",".join(str(i) for i in range(n)),
                "--out",
                f"/runs/{stamp}/n{n}",
            ],
            cwd="/repo",
        ).returncode
    finally:
        runs.commit()


@app.local_entrypoint()
def main(config: str, only_n: int = 0):
    from benchmark_multi_kalypso.config import parse_config
    from benchmark_multi_kalypso.remote import jobs, merge_results

    config_text = Path(config).read_text(encoding="utf-8")
    planned = jobs(parse_config(config_text), only_n or None)
    # The image copies the working tree, so a dirty tree is what runs.
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "describe", "--always", "--dirty", "--match", ""],
        text=True,
    ).strip()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Spawn all N groups before waiting on any of them.
    calls = [
        (n, run_n.with_options(gpu=gpu).spawn(config_text, n, commit, stamp))
        for n, gpu in planned
    ]
    status = 0
    for n, call in calls:
        try:
            status = max(status, call.get())
        except Exception as exc:
            print(f"N={n} container failed: {exc}")
            status = 1
    # `modal volume get` creates the <stamp> folder inside the destination.
    runs_dir = REPO / "kalypso_modal_run"
    runs_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "kalypso-runs", stamp, str(runs_dir)], check=True
    )
    out = runs_dir / stamp
    merge_results(sorted(out.glob("n*")), out)
    print(f"Saved results to {out} (status {status})")
    if status:
        raise SystemExit(status)
