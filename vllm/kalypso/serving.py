# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request handling shared by the in-process API server and the coordinator."""

import asyncio
import uuid

from fastapi import HTTPException, Request

from vllm.kalypso import interface, query
from vllm.kalypso.query_processor import QueryProcessor


def materialize_records_to_csv(records: list, text_column: str | None = None) -> str:
    """Writes inline records to a temp CSV so the existing path-based data
    readers (vllm.kalypso.data.data._csv_reader) can consume them unchanged,
    columns intact - that reader picks the row's text from a "data" column,
    a "Text"/"Sentences" pair, a plain "Text" column, or else joins every
    column, in that order, so preserving columns keeps all of those working.
    A plain cell value (str/int/float, no column name) is treated as a
    {"data": value} row. Records must be all row objects or all bare cells,
    never mixed - the reader picks the "data" column whenever it's merely
    *present* in the row (not merely non-empty), so a mixed list would give
    every dict row an empty "data" cell and incorrectly read as blank
    instead of falling through to its real Text/Sentences columns.

    text_column picks which of a row object's columns is the text to run ops
    over, instead of relying on that fixed guessing order - its value is
    copied into a "data" column (which the reader always tries first), the
    original columns kept alongside for ops that reference them by name."""
    import csv
    import os
    import tempfile

    is_dict_row = [isinstance(record, dict) for record in records]
    if any(is_dict_row) and not all(is_dict_row):
        raise HTTPException(
            status_code=400,
            detail="records must be all row objects or all plain values, not a mix of both.",
        )

    if not all(is_dict_row):
        if text_column is not None:
            raise HTTPException(
                status_code=400,
                detail="text_column requires row-object records, not plain values.",
            )
        records = [{"data": str(record)} for record in records]
    elif text_column is not None:
        missing = [i for i, record in enumerate(records) if text_column not in record]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f'text_column "{text_column}" missing from record(s) at index {missing[:5]}.',
            )
        records = [{**record, "data": record[text_column]} for record in records]

    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["data"]

    fd, path = tempfile.mkstemp(prefix="kalypso_records_", suffix=".csv")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore", restval=""
        )
        writer.writeheader()
        writer.writerows(records)
    return path


async def run_semantic_query(
    processor: QueryProcessor,
    raw_request: Request,
    sem_request: interface.SemanticQueryRequest,
) -> dict:
    if bool(sem_request.data_path) == bool(sem_request.records):
        raise HTTPException(
            status_code=400,
            detail="Exactly one of data_path or records must be provided.",
        )

    data_path = sem_request.data_path or materialize_records_to_csv(
        sem_request.records, sem_request.text_column
    )

    started = asyncio.get_running_loop().time()
    print("=== Semantic Execute Request ===")
    print("Query:")
    print(sem_request.ops)
    print("data_path:")
    print(data_path)
    print("================================")
    _query = query.Query(sem_request.ops, data_path)
    print("query called")

    out_ctxs = await processor.execute(raw_request, _query)

    results = []
    for ctx in out_ctxs:
        outputs = []
        for item in ctx.output:
            if isinstance(item, dict) and len(item) == 1:
                op_name, value = next(iter(item.items()))
                outputs.append({"op": op_name, "value": value})
            else:
                outputs.append({"op": "unknown", "value": item})

        results.append(
            {
                "idx": getattr(ctx.state, "idx", None),
                "input_context": getattr(ctx.input, "data", None),
                "right_input_context": getattr(ctx.input, "right_data", None),
                "outputs": outputs,
            }
        )

    elapsed = asyncio.get_running_loop().time() - started
    return {
        "request_id": uuid.uuid4().hex,
        "model_name": processor.model_name,
        "data_path": sem_request.data_path,
        "ops": sem_request.ops,
        "num_output_rows": len(out_ctxs),
        "latency_sec": round(elapsed, 3),
        "results": results,
    }
