import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Result:
    commit: str
    workload: str
    requested_rows: int
    rows: int | None
    n_workers: int
    policy: str
    rep: int
    completion_s: float | None = None
    rows_per_s: float | None = None
    num_output_rows: int | None = None
    status: str = "error"
    error: str | None = None
    hypotheses: int | None = None


def append_result(out_dir: Path, result: Result) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    path = out_dir / "results.csv"
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
