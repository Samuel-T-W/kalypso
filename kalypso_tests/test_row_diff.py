# SPDX-License-Identifier: Apache-2.0
import csv
import json

import pytest

from vllm.kalypso.benchmark.row_diff import compare, main


def test_identical_filter_map_rows_have_literal_events():
    payload = {
        "text_column": "text",
        "records": [{"text": "apple"}, {"text": "spam"}],
        "ops": [{"op": "sem_filter"}, {"op": "sem_map"}],
    }
    response = {
        "results": [
            {
                "idx": 0,
                "outputs": [
                    {"op": "SemFilter", "value": "true"},
                    {"op": "SemMap", "value": " fruit \n"},
                ],
            }
        ]
    }
    report = compare(payload, response, response)
    assert report["rows"] == [
        {
            "idx": 0,
            "text": "apple",
            "old": ["kept", "fruit"],
            "new": ["kept", "fruit"],
            "same": True,
        },
        {
            "idx": 1,
            "text": "spam",
            "old": ["dropped", "-"],
            "new": ["dropped", "-"],
            "same": True,
        },
    ]
    assert report["summary"] == {
        "rows total": 2,
        "kept by both": 1,
        "dropped by both": 1,
        "kept only by old": 0,
        "kept only by new": 0,
        "kept by both but some value differs": 0,
        "errors": 0,
    }
    assert report["ok"] and not report["errors"]


@pytest.mark.parametrize("kept_side", ["old", "new"])
def test_row_kept_by_only_one_side(kept_side):
    payload = {
        "records": [{"text": "hello"}],
        "text_column": "text",
        "ops": [{"op": "sem_filter"}],
    }
    responses = {"old": {"results": []}, "new": {"results": []}}
    responses[kept_side] = {"results": [{"idx": 0, "outputs": [{"value": "true"}]}]}
    report = compare(payload, **responses)
    assert report["rows"][0][kept_side] == ["kept"]
    assert report["summary"][f"kept only by {kept_side}"] == 1
    assert not report["ok"] and not report["rows"][0]["same"]


def test_repeated_maps_keep_operator_order_and_compare_values():
    payload = {
        "records": [{"text": "apple"}],
        "text_column": "text",
        "ops": [{"op": "sem_map"}, {"op": "sem_map"}],
    }
    old = {"results": [{"idx": 0, "outputs": [{"value": "fruit"}, {"value": "red"}]}]}
    new = {"results": [{"idx": 0, "outputs": [{"value": "red"}, {"value": "fruit"}]}]}
    report = compare(payload, old, new)
    assert report["rows"][0]["old"] == ["fruit", "red"]
    assert report["rows"][0]["new"] == ["red", "fruit"]
    assert report["summary"]["kept by both but some value differs"] == 1
    assert not report["ok"]


@pytest.mark.parametrize("bad_side", ["old", "new"])
def test_duplicate_idx_is_an_error_even_when_events_match(bad_side):
    payload = {"records": [{"text": "hello"}], "text_column": "text", "ops": []}
    row = {"idx": 0, "outputs": []}
    responses = {"old": {"results": [row]}, "new": {"results": [row]}}
    responses[bad_side] = {"results": [row, row]}
    report = compare(payload, **responses)
    assert report["errors"] == [f"{bad_side}: duplicate idx 0"]
    assert report["summary"]["errors"] == 1
    assert not report["ok"] and not report["rows"][0]["same"]


@pytest.mark.parametrize("idx", [-1, 1, True, "0"])
def test_invalid_idx_is_reported_without_losing_input_rows(idx):
    payload = {
        "records": [{"text": "spam"}],
        "text_column": "text",
        "ops": [{"op": "sem_filter"}],
    }
    report = compare(payload, {"results": [{"idx": idx}]}, {"results": []})
    assert len(report["rows"]) == 1
    assert report["errors"] == [f"old: idx out of range: {idx!r}"]
    assert not report["ok"]


def test_dropped_row_with_multiple_filters_has_no_invented_history():
    payload = {
        "records": [{"text": "spam"}],
        "text_column": "text",
        "ops": [{"op": "sem_map"}, {"op": "sem_filter"}, {"op": "sem_filter"}],
    }
    report = compare(payload, {"results": []}, {"results": []})
    assert report["rows"][0]["old"] == ["-", "dropped (filter)", "-"]
    assert report["ok"]


@pytest.mark.parametrize("different", [False, True])
def test_cli_writes_reports_commits_and_exit_status(tmp_path, capsys, different):
    payload = {
        "records": [{"text": "a|b\nc"}],
        "text_column": "text",
        "ops": [{"op": "sem_map"}],
    }
    old = {"results": [{"idx": 0, "outputs": [{"value": "fruit"}]}]}
    new = {
        "results": [
            {"idx": 0, "outputs": [{"value": "pear" if different else "fruit"}]}
        ]
    }
    argv = []
    for name, data in (("payload", payload), ("old", old), ("new", new)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        argv.extend([f"--{name}", str(path)])
    (tmp_path / "commits.txt").write_text("Old commit: abc\nNew commit: def\n")
    assert main([*argv, "--out-dir", str(tmp_path)]) == int(different)
    with (tmp_path / "row_diff.csv").open(newline="") as stream:
        assert list(csv.reader(stream)) == [
            ["idx", "text", "old_1_sem_map", "new_1_sem_map", "same"],
            [
                "0",
                "a|b\nc",
                "fruit",
                "pear" if different else "fruit",
                str(not different),
            ],
        ]
    markdown = (tmp_path / "row_diff.md").read_text()
    assert "Old commit: abc\nNew commit: def" in markdown
    assert "a&#124;b<br>c" in markdown
    differing, full = markdown.split("## Differing rows\n\n")[1].split("## All rows")
    assert ("| 0 |" in differing) == different
    assert "| 0 |" in full
    assert "errors: 0" in capsys.readouterr().out
