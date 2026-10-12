# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from copy import deepcopy

import pytest

from vllm.kalypso.benchmark.correctness_harness import check


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "duplicate",
        "invalid",
        "mapping",
        "agreement",
        "missing",
        "repeat",
        "affinity",
        "failed",
        "empty",
        "first_index",
        "trace_invalid",
    ],
)
@pytest.mark.parametrize("stage", [1, 5])
def test_response_and_trace_checks(tmp_path, capsys, fault, stage):
    payload = {
        "records": [{"text": "apple"}, {"text": "pear"}],
        "text_column": "text",
        "ops": [{"op": "sem_filter"}],
    }
    baseline = {
        "results": [
            {
                "idx": 0,
                "input_context": [{"content": {"text": "apple"}}],
                "outputs": [{"value": "fruit"}],
            },
            {"idx": 1, "input_context": "pear", "outputs": [{"value": "green"}]},
        ]
    }
    candidate = deepcopy(baseline)
    candidate["results"].reverse()
    calls = [
        {
            "row": 0,
            "stage": stage,
            "op": "SemFilter",
            "op_index": 0,
            "worker_id": 0,
            "ok": True,
        },
        {
            "row": 1,
            "stage": stage,
            "op": "SemFilter",
            "op_index": 0,
            "worker_id": 1,
            "ok": True,
        },
        {
            "row": 0,
            "stage": stage,
            "op": "SemFilter",
            "op_index": 1,
            "worker_id": 0,
            "ok": True,
        },
    ]
    if fault == "duplicate":
        candidate["results"].append(deepcopy(candidate["results"][0]))
    if fault == "invalid":
        candidate["results"][0]["idx"] = -1
    if fault == "mapping":
        candidate["results"][0]["input_context"] = "apple"
    if fault == "agreement":
        candidate["results"][0]["outputs"][0]["value"] = "wrong"
    if fault == "missing":
        calls.pop(1)
    if fault == "repeat":
        calls.append(deepcopy(calls[0]))
    if fault == "affinity":
        calls[2]["worker_id"] = 1
    if fault == "failed":
        calls[0]["ok"] = False
    if fault == "empty":
        calls.clear()
    if fault == "first_index":
        calls[1]["op_index"] = 2
    if fault == "trace_invalid":
        calls[2]["row"] = -1
    for call in calls:
        call["start"] = 1.0
        call["finish"] = 2.0
    (tmp_path / "requests.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in calls)
    )
    passed, workers = check(payload, baseline, candidate, tmp_path)
    assert passed == (fault is None)
    output = capsys.readouterr().out
    if fault is None:
        assert workers == {0: 2, 1: 1}
        assert output.count("PASS") == 5
    else:
        expected_check = {
            "duplicate": "exactly once (output)",
            "invalid": "exactly once (output)",
            "mapping": "row mapping",
            "agreement": "agreement (differing idx: [1])",
            "affinity": "affinity",
        }.get(fault, "exactly once (calls)")
        assert f"FAIL {expected_check}" in output


def test_blocking_calls_are_counted_but_excluded_from_row_checks(tmp_path, capsys):
    payload = {"records": [{"text": "apple"}], "text_column": "text"}
    response = {
        "results": [
            {"idx": 0, "input_context": "apple", "outputs": [{"value": "fruit"}]}
        ]
    }
    calls = [
        {
            "row": 0,
            "stage": 5,
            "op": "SemFilter",
            "op_index": 0,
            "worker_id": 0,
            "ok": True,
            "start": 1.0,
            "finish": 2.0,
        },
        {
            "row": None,
            "stage": 6,
            "op": "SemAgg",
            "op_index": 0,
            "worker_id": 1,
            "ok": True,
            "start": 2.0,
            "finish": 3.0,
        },
        {
            "row": None,
            "stage": 6,
            "op": "SemAgg",
            "op_index": 0,
            "worker_id": 2,
            "ok": False,
            "start": 3.0,
            "finish": 4.0,
        },
    ]
    (tmp_path / "requests.jsonl").write_text(
        "".join(json.dumps(call) + "\n" for call in reversed(calls))
    )
    assert check(payload, response, response, tmp_path) == (True, {0: 1, 1: 1, 2: 1})
    assert "FAIL" not in capsys.readouterr().out
