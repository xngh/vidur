#!/usr/bin/env python3
"""
Batch experiment runner for vidur.

Edit BASE_ARGS and VARIANTS below to customize experiments.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List


# ---- Editable section: base args + variants ----
BASE_ARGS: Dict[str, object] = {
    "replica_config_device": "a100",
    "replica_config_model_name": "meta-llama/Meta-Llama-3-8B",
    "cluster_config_replica_configs": json.dumps(
        [
            {
                "device": "h100",
                "network_device": "h100_pairwise_nvlink",
                "model_name": "meta-llama/Llama-2-7b-hf",
                "count": 2,
            },
            {
                "device": "a100",
                "network_device": "a100_pairwise_nvlink",
                "model_name": "meta-llama/Llama-2-7b-hf",
                "count": 2,
            },
        ]
    ),
    "replica_config_tensor_parallel_size": 1,
    "replica_config_num_pipeline_stages": 1,
    "share_g_p_t_request_generator_config_max_tokens": 16384,
    "random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size": 16384,
    "random_forrest_execution_time_predictor_config_prediction_max_batch_size": 512,
    "random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request": 16384,
    "local_replica_scheduler_config_batch_size_cap": 512,
}

# Only these four are expected to vary most often; extend freely.
# 对这几个参数的列表里的属性，做笛卡尔积后进行批量实验.
VARIANTS: Dict[str, List[object]] = {
    # e.g. ["parrot", "round_robin", "sharp"]
    "global_scheduler_config_type": ["sharp","parrot"],
    # e.g. ["unified", "synthetic"]
    "request_generator_config_type": ["mapreduce"],
    # e.g. ["local", "slo"]
    "replica_scheduler_config_type": ["local"],
    # e.g. [1, 2, 5, 10]
    "poisson_request_interval_generator_config_qps": [5,6,7],
}

# Optional: put all runs under a subfolder to keep outputs together.
RUNS_OUTPUT_ROOT = "simulator_output/batch_runs"


# ---- Runner implementation ----
def build_experiments(base_args: Dict[str, object], variants: Dict[str, List[object]]):
    keys = list(variants.keys())
    values_product = itertools.product(*(variants[k] for k in keys))
    for values in values_product:
        params = dict(base_args)
        params.update(dict(zip(keys, values)))
        yield params


def _sanitize(value: object) -> str:
    text = str(value)
    return (
        text.replace("/", "_")
        .replace(" ", "")
        .replace(":", "-")
        .replace(".", "p")
    )


def build_run_name(params: Dict[str, object], keys: Iterable[str]) -> str:
    parts = [f"{k}={_sanitize(params[k])}" for k in keys]
    return "__".join(parts)


def params_to_args(params: Dict[str, object]) -> List[str]:
    args: List[str] = []
    for key, value in params.items():
        flag = f"--{key}"
        if value is None:
            args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args


def run_one(
    params: Dict[str, object],
    run_idx: int,
    total: int,
    dry_run: bool,
):
    cmd = [sys.executable, "-m", "vidur.main"] + params_to_args(params)
    print(f"[{run_idx}/{total}] {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _find_latest_result_json(run_base_dir: str) -> str | None:
    if not os.path.isdir(run_base_dir):
        return None
    candidates = []
    for root, _, files in os.walk(run_base_dir):
        if "result.json" in files:
            result_path = os.path.join(root, "result.json")
            candidates.append(result_path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def write_total_result_json(
    batch_dir: str,
    experiments: List[Dict[str, object]],
    variant_keys: List[str],
) -> None:
    results = []
    for exp in experiments:
        run_name = build_run_name(exp, variant_keys)
        run_base_dir = os.path.join(batch_dir, run_name)
        result_path = _find_latest_result_json(run_base_dir)
        if result_path and os.path.exists(result_path):
            with open(result_path, "r") as f:
                result_payload = json.load(f)
        else:
            result_payload = None

        results.append(
            {
                "run_name": run_name,
                "run_dir": run_base_dir,
                "params": {k: exp.get(k) for k in variant_keys},
                "result_path": result_path,
                "result": result_payload,
            }
        )

    payload = {
        "batch_dir": batch_dir,
        "variant_keys": variant_keys,
        "results": results,
    }
    os.makedirs(batch_dir, exist_ok=True)
    total_path = os.path.join(batch_dir, "total_result.json")
    with open(total_path, "w") as f:
        json.dump(payload, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run vidur experiments.")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit #experiments (0=all).")
    parser.add_argument(
        "--batch-name",
        type=str,
        default=None,
        help="Optional subfolder name under batch_runs (default: timestamp).",
    )
    args = parser.parse_args()

    experiments = list(build_experiments(BASE_ARGS, VARIANTS))
    if args.limit and args.limit > 0:
        experiments = experiments[: args.limit]

    # add per-run output_dir to keep runs grouped
    variant_keys = list(VARIANTS.keys())
    batch_name = args.batch_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = os.path.join(RUNS_OUTPUT_ROOT, batch_name)
    for exp in experiments:
        run_name = build_run_name(exp, variant_keys)
        exp["metrics_config_output_dir"] = os.path.join(batch_dir, run_name)

    total = len(experiments)
    if total == 0:
        print("No experiments to run. Check VARIANTS values.")
        return

    if args.max_parallel <= 1:
        for idx, exp in enumerate(experiments, start=1):
            run_one(exp, idx, total, args.dry_run)
        if not args.dry_run:
            write_total_result_json(batch_dir, experiments, variant_keys)
        return

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [
            executor.submit(run_one, exp, idx, total, args.dry_run)
            for idx, exp in enumerate(experiments, start=1)
        ]
        for future in as_completed(futures):
            future.result()
    if not args.dry_run:
        write_total_result_json(batch_dir, experiments, variant_keys)


if __name__ == "__main__":
    main()
