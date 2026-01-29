#!/usr/bin/env python3
"""
Convert Vidur agent trace jsonl (one call per line) into the expected aggregated JSON format.

Input (.jsonl):
  - Each line is a dict describing one call (Map_Worker / Reduce_Master / ...)
  - Calls that belong to the same top-level agent request share the same `top_level_request_id`

Output (.json):
  - A JSON array, where each element is:
      {
        "agent_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        "type": "mapreduce",
        "summary": [ {call fields...}, ... ]
      }
  - The summary list contains only the fields used in `data/MapReduceTrace/expected_format.json`.
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


EXPECTED_CALL_FIELDS: Tuple[str, ...] = (
    "call_id",
    "parent_id",
    "node_name",
    "inputs",
    "outputs",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "model_name",
)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from e


def _slim_call(rec: Dict[str, Any]) -> Dict[str, Any]:
    slim: Dict[str, Any] = {}
    for k in EXPECTED_CALL_FIELDS:
        if k in rec:
            slim[k] = rec[k]
    return slim


def _reorder_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep original order within (maps) and within (non-maps), but ensure Map_Worker entries come first.
    This matches the "maps then reduce" expectation without relying on file order.
    """
    maps: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for c in calls:
        if c.get("node_name") == "Map_Worker":
            maps.append(c)
        else:
            others.append(c)
    return maps + others


def convert(input_path: Path) -> List[Dict[str, Any]]:
    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

    for rec in _iter_jsonl(input_path):
        top_id = rec.get("top_level_request_id")
        if not top_id:
            # Fallback: group by parent_id (best-effort). This keeps the script usable on partial logs.
            top_id = rec.get("parent_id")
        if not top_id:
            raise ValueError("Record missing both `top_level_request_id` and `parent_id`.")

        if top_id not in groups:
            groups[top_id] = []
        groups[top_id].append(_slim_call(rec))

    out: List[Dict[str, Any]] = []
    for _top_id, calls in groups.items():
        out.append(
            {
                "agent_id": str(uuid.uuid4()),
                "type": "mapreduce",
                "summary": _reorder_calls(calls),
            }
        )

    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert agent trace jsonl to expected aggregated JSON format."
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input .jsonl (agent trace).",
    )
    p.add_argument(
        "--output",
        required=False,
        type=Path,
        help="Path to output .json (aggregated). Defaults to <input>.converted.json",
    )
    args = p.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path: Path = args.output or input_path.with_suffix(input_path.suffix + ".converted.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = convert(input_path=input_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(data)} agent request(s) to: {output_path}")


if __name__ == "__main__":
    main()


