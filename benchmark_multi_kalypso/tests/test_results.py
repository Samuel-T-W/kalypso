import csv

from benchmark_multi_kalypso.results import Result, append_result


def test_rows_are_appended_under_one_header(tmp_path):
    result = Result(
        "abc",
        "fever",
        10,
        10,
        2,
        "round_robin",
        1,
        completion_s=2.0,
        rows_per_s=5.0,
        status="ok",
    )
    append_result(tmp_path, result)
    append_result(tmp_path, result)
    with (tmp_path / "results.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 2
    assert csv_rows[0]["rows_per_s"] == "5.0"
