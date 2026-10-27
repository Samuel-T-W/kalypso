# SPDX-License-Identifier: Apache-2.0
"""Compare every input row of two semantic query responses."""

import argparse
import csv
import html
import json
from pathlib import Path


def compare(payload: dict, old: dict, new: dict) -> dict:
    """Return ordered row events, summary counts, errors, and an overall ok flag.

    Dropped rows have no surviving history, so their first filter is marked as
    the drop and all other events are unknown ("-").
    """
    ops = [op["op"] for op in payload["ops"]]
    filters = [i for i, op in enumerate(ops) if "filter" in op.lower()]
    records = payload["records"]
    errors = []
    invalid_rows = set()

    def index(response, side):
        indexed = {}
        for row in response["results"]:
            idx = row["idx"]
            if type(idx) is not int or not 0 <= idx < len(records):
                errors.append(f"{side}: idx out of range: {idx!r}")
            elif idx in indexed:
                errors.append(f"{side}: duplicate idx {idx}")
                invalid_rows.add(idx)
            else:
                indexed[idx] = row
        return indexed

    before, after = index(old, "old"), index(new, "new")

    def events(indexed, idx, side):
        if idx not in indexed:
            result = ["-"] * len(ops)
            if filters:
                result[filters[0]] = (
                    "dropped" if len(filters) == 1 else "dropped (filter)"
                )
            else:
                errors.append(f"{side}: missing idx {idx} without a filter")
                invalid_rows.add(idx)
            return result
        outputs = indexed[idx]["outputs"]
        if len(outputs) != len(ops):
            errors.append(
                f"{side}: idx {idx} has {len(outputs)} outputs for {len(ops)} ops"
            )
            invalid_rows.add(idx)
        result = []
        for position in range(len(ops)):
            if position >= len(outputs):
                result.append("-")
            elif position in filters:
                result.append("kept")
            else:
                value = outputs[position]["value"]
                result.append(
                    (value if isinstance(value, str) else json.dumps(value)).strip()
                )
        return result

    rows = []
    for idx, record in enumerate(records):
        old_events, new_events = events(before, idx, "old"), events(after, idx, "new")
        rows.append(
            dict(
                idx=idx,
                text=record[payload["text_column"]],
                old=old_events,
                new=new_events,
                same=old_events == new_events and idx not in invalid_rows,
            )
        )
    summary = {
        "rows total": len(records),
        "kept by both": len(before.keys() & after.keys()),
        "dropped by both": len(records) - len(before.keys() | after.keys()),
        "kept only by old": len(before.keys() - after.keys()),
        "kept only by new": len(after.keys() - before.keys()),
        "kept by both but some value differs": sum(
            row["old"] != row["new"]
            for row in rows
            if row["idx"] in before and row["idx"] in after
        ),
        "errors": len(errors),
    }
    return dict(
        rows=rows,
        summary=summary,
        errors=errors,
        ok=not errors and all(row["same"] for row in rows),
    )


def write_report(payload: dict, report: dict, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{i + 1}_{op['op']}" for i, op in enumerate(payload["ops"])]
    with (out_dir / "row_diff.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["idx", "text"]
            + [f"{side}_{label}" for side in ("old", "new") for label in labels]
            + ["same"]
        )
        for row in report["rows"]:
            writer.writerow(
                [row["idx"], row["text"], *row["old"], *row["new"], row["same"]]
            )

    def cell(value):
        return (
            html.escape(str(value))
            .replace("|", "&#124;")
            .replace("\r\n", "\n")
            .replace("\n", "<br>")
        )

    def table(rows):
        headers = (
            ["idx", "text"]
            + [f"{side} {label}" for label in labels for side in ("old", "new")]
            + ["same"]
        )
        lines = [
            "| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            values = [row["idx"], row["text"]]
            values += [event for pair in zip(row["old"], row["new"]) for event in pair]
            lines.append("| " + " | ".join(map(cell, [*values, row["same"]])) + " |")
        return "\n".join(lines)

    summary = "\n".join(f"{name}: {count}" for name, count in report["summary"].items())
    sections = ["# Row diff"]
    commits = out_dir / "commits.txt"
    if commits.exists():
        sections.append("```text\n" + commits.read_text().strip() + "\n```")
    sections += ["## Summary", "\n".join("- " + line for line in summary.splitlines())]
    if report["errors"]:
        sections += ["## Errors", "\n".join("- " + cell(e) for e in report["errors"])]
    sections += [
        "## Differing rows",
        table([r for r in report["rows"] if not r["same"]]),
        "## All rows",
        table(report["rows"]),
    ]
    (out_dir / "row_diff.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("payload", "old", "new", "out-dir"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        payload, old, new = [
            json.loads(path.read_text()) for path in (args.payload, args.old, args.new)
        ]
        report = compare(payload, old, new)
        print(write_report(payload, report, args.out_dir))
        return int(not report["ok"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
