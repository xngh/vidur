#!/usr/bin/env python3
"""
Summarize total_result.json into a flat table (CSV + Markdown).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple


def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                flat[f"{key}.{sub_key}"] = sub_val
        else:
            flat[key] = value
    return flat


def _load_total_result(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _build_rows(payload: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    variant_keys = payload.get("variant_keys", [])
    results = payload.get("results", [])

    rows: List[Dict[str, Any]] = []
    metric_keys: List[str] = []

    for item in results:
        params = item.get("params", {})
        result = item.get("result") or {}
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        flat_metrics = _flatten_metrics(metrics)

        # collect metric keys in stable order
        for k in flat_metrics.keys():
            if k not in metric_keys:
                metric_keys.append(k)

        row = {
            "run_name": item.get("run_name"),
            "run_dir": item.get("run_dir"),
            "result_path": item.get("result_path"),
            "status": "ok" if result else "missing",
        }
        for k in variant_keys:
            row[k] = params.get(k)
        for k, v in flat_metrics.items():
            row[k] = v
        rows.append(row)

    headers = ["run_name", "run_dir", "result_path", "status"] + list(variant_keys) + metric_keys
    return headers, rows


def _write_csv(path: str, headers: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: str, headers: List[str], rows: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize total_result.json to table.")
    parser.add_argument("total_result", help="Path to total_result.json")
    parser.add_argument("--output-csv", default=None, help="CSV output path")
    parser.add_argument("--output-md", default=None, help="Markdown output path")
    args = parser.parse_args()

    payload = _load_total_result(args.total_result)
    headers, rows = _build_rows(payload)

    base_dir = os.path.dirname(args.total_result)
    csv_path = args.output_csv or os.path.join(base_dir, "total_result_table.csv")
    md_path = args.output_md or os.path.join(base_dir, "total_result_table.md")

    _write_csv(csv_path, headers, rows)
    _write_markdown(md_path, headers, rows)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
