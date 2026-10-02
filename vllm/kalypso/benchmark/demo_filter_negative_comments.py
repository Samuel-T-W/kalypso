#!/usr/bin/env python3
"""Run a Kalypso semantic filter over synthetic social-media comments.

Requires a running vLLM server from this repo with the semantic API enabled
(for example on Mac CPU or on AWS GPU). The client sends an explicit operator
list (`ops`); Kalypso does not call Lotus or any external planner.

Example (after `vllm serve ... --enable-prefix-caching --port 8003`):

    python demo_filter_negative_comments.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --port 8003
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass

import requests

# Rows kept by sem_filter must *not* contain the token "false" in the model reply.
FILTER_PROMPT = textwrap.dedent(
    """
    You are moderating comments on a social media post.

    Decide whether the comment is acceptable to keep visible: friendly, neutral,
    or constructive praise. Answer false if the comment is insulting, harassing,
    body-shaming, or otherwise clearly negative toward the person in the post.

    Reply with only the word true or false.
    """
).strip()

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PORT = 8003
DEFAULT_ROUTE = "/v1/semantic/query"


@dataclass(frozen=True)
class CommentRow:
    comment_id: int
    text: str
    is_negative: bool


# Twenty synthetic Facebook-style comments; five are negative (body-shaming).
SYNTHETIC_COMMENTS: tuple[CommentRow, ...] = (
    CommentRow(1, "Your outfit looks amazing in this photo!", False),
    CommentRow(2, "Love the color you picked for that jacket.", False),
    CommentRow(3, "You look so confident here, great pic.", False),
    CommentRow(4, "Wow you got fat since your last post.", True),
    CommentRow(5, "Sending this to my stylist for inspo.", False),
    CommentRow(6, "Honestly you look huge in that dress.", True),
    CommentRow(7, "Your smile totally made my day.", False),
    CommentRow(8, "Did you gain like fifty pounds?", True),
    CommentRow(9, "Where did you get those shoes?", False),
    CommentRow(10, "This angle is really flattering on you.", False),
    CommentRow(11, "No offense but you look bloated.", True),
    CommentRow(12, "Congrats on the new job, you deserve it!", False),
    CommentRow(13, "The lighting in this shot is perfect.", False),
    CommentRow(14, "Maybe skip tight clothes until you lose weight.", True),
    CommentRow(15, "Your hair looks great like that.", False),
    CommentRow(16, "So happy for you and your family.", False),
    CommentRow(17, "Following for more content like this.", False),
    CommentRow(18, "That cafe in the background looks fun.", False),
    CommentRow(19, "You always post the best travel tips.", False),
    CommentRow(20, "Have a wonderful weekend!", False),
)


def build_payload(model_name: str) -> dict:
    return {
        "model_name": model_name,
        "text_column": "text",
        "records": [
            {"id": row.comment_id, "text": row.text}
            for row in SYNTHETIC_COMMENTS
        ],
        "ops": [
            {
                "op": "sem_filter",
                "args": {
                    "prompt": FILTER_PROMPT,
                    "max_tokens": 8,
                },
            }
        ],
    }


def parse_args() -> tuple[str, str, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name served by vLLM (must match the running server).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port of the vLLM OpenAI server hosting /v1/semantic/query.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Full semantic query URL; overrides --port when set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the JSON payload and exit without calling the server.",
    )
    args = parser.parse_args()
    endpoint = args.endpoint or f"http://localhost:{args.port}{DEFAULT_ROUTE}"
    return args.model_name, endpoint, args.dry_run


def _filter_verdict(result_row: dict) -> str | None:
    for item in result_row.get("outputs") or []:
        if item.get("op") == "SemFilter":
            return str(item.get("value", "")).strip()
    return None


def print_report(response: dict) -> None:
    # Kalypso sets idx to the row's position in the loaded table (0-based).
    kept_positions: set[int] = set()
    results_by_idx: dict[int, dict] = {}
    for item in response.get("results") or []:
        idx = item.get("idx")
        if idx is None:
            continue
        pos = int(idx)
        kept_positions.add(pos)
        results_by_idx[pos] = item

    print("\n=== Kalypso response summary ===")
    print(f"model_name:       {response.get('model_name')}")
    print(f"latency_sec:      {response.get('latency_sec')}")
    print(f"input_rows:       {len(SYNTHETIC_COMMENTS)}")
    print(f"output_rows:      {response.get('num_output_rows')}")
    print(f"expected_negative: {sum(r.is_negative for r in SYNTHETIC_COMMENTS)}")

    print("\n=== Per-row (ground truth is local; server does not see labels) ===")
    print(f"{'id':>3}  {'kept':5}  {'expect_neg':10}  {'verdict':8}  comment")
    for pos, row in enumerate(SYNTHETIC_COMMENTS):
        kept = pos in kept_positions
        verdict = ""
        if pos in results_by_idx:
            verdict = _filter_verdict(results_by_idx[pos]) or ""
        if not kept and not verdict:
            verdict = "(dropped)"
        ok = kept != row.is_negative
        flag = "ok" if ok else "MISMATCH"
        snippet = row.text if len(row.text) <= 52 else row.text[:49] + "..."
        print(
            f"{row.comment_id:3d}  {str(kept):5}  {str(row.is_negative):10}  "
            f"{verdict:8}  {snippet}  [{flag}]"
        )


def main() -> int:
    model_name, endpoint, dry_run = parse_args()
    payload = build_payload(model_name)

    print("=== Query plan (ops) sent to Kalypso ===")
    print(json.dumps(payload["ops"], indent=2))
    print(f"\nEndpoint: {endpoint}")
    print(f"Records:  {len(payload['records'])} comments\n")

    if dry_run:
        print("=== Full request body (dry run) ===")
        print(json.dumps(payload, indent=2))
        return 0

    try:
        http = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=3600,
        )
        http.raise_for_status()
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        if getattr(exc, "response", None) is not None:
            print(exc.response.text, file=sys.stderr)
        return 1

    response = http.json()
    print_report(response)

    print("\n=== Raw JSON (truncated results) ===")
    preview = dict(response)
    if len(preview.get("results") or []) > 3:
        preview["results"] = preview["results"][:3] + ["..."]
    print(json.dumps(preview, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
