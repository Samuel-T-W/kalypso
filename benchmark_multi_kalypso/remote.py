import csv
from pathlib import Path
from typing import TYPE_CHECKING

from .results import Result, append_result

if TYPE_CHECKING:
    from .config import Experiment


def jobs(config: "Experiment", only_n: int | None = None) -> list[tuple[int, str]]:
    if only_n is not None and only_n not in config.n_workers:
        raise ValueError(f"N={only_n} is not in configured n_workers")
    return [
        (n, f"{config.hardware.gpu_type}:{n}")
        for n in config.n_workers
        if only_n is None or n == only_n
    ]


def merge_results(groups: list[Path], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.csv").exists():
        raise ValueError(f"Merged results already exist: {out / 'results.csv'}")
    for group in groups:
        path = group / "results.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                append_result(out, Result(**row))
