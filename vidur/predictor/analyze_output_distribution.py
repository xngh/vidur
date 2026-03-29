import argparse
import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np

from vidur.predictor.prepare_training_data import iter_json_array


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _word_count(text: str) -> int:
    return len(str(text).split())


def collect_mapreduce_output_lengths(
    path: str, max_samples: Optional[int]
) -> List[int]:
    lengths: List[int] = []
    count = 0
    for workflow in iter_json_array(path):
        summary = workflow.get("summary", [])
        for call in summary:
            node_name = call.get("node_name", "")
            if node_name not in ("Map_Worker", "Reduce_Master"):
                continue
            if not call.get("inputs") or not call.get("outputs"):
                continue
            output_len = call.get("completion_tokens")
            if not isinstance(output_len, int):
                output_len = _word_count(call.get("outputs", ""))
            lengths.append(int(output_len))
            count += 1
            if max_samples is not None and count >= max_samples:
                return lengths
    return lengths


def collect_sharegpt_output_lengths(
    path: str, max_samples: Optional[int]
) -> List[int]:
    lengths: List[int] = []
    count = 0
    for record in iter_json_array(path):
        conversations = record.get("conversations", [])
        for i in range(0, len(conversations) - 1, 2):
            turn_in = conversations[i]
            turn_out = conversations[i + 1]
            if turn_in.get("from") != "human" or turn_out.get("from") != "gpt":
                continue
            output = turn_out.get("value", "")
            if not output:
                continue
            output_len = _word_count(output)
            lengths.append(int(output_len))
            count += 1
            if max_samples is not None and count >= max_samples:
                return lengths
    return lengths


def summarize(name: str, lengths: List[int], percentiles: Iterable[int]) -> None:
    if not lengths:
        logger.warning("%s: no samples found", name)
        return
    arr = np.array(lengths, dtype=np.float64)
    stats = {
        "count": len(arr),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
    }
    logger.info(
        "%s | count=%d | min=%d | max=%d | mean=%.2f | std=%.2f | p50=%.2f",
        name,
        stats["count"],
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["std"],
        stats["p50"],
    )
    pct_values = np.percentile(arr, list(percentiles)).tolist()
    pct_pairs = ", ".join(
        f"p{p}={v:.2f}" for p, v in zip(percentiles, pct_values)
    )
    logger.info("%s percentiles: %s", name, pct_pairs)


def suggest_buckets(
    lengths: List[int], num_buckets: int
) -> Tuple[List[int], List[float]]:
    if not lengths:
        return [], []
    if num_buckets < 2:
        raise ValueError("num_buckets must be >= 2")
    arr = np.array(lengths, dtype=np.float64)
    quantiles = [i / num_buckets for i in range(1, num_buckets)]
    boundaries = np.quantile(arr, quantiles).tolist()
    # Round to ints and enforce non-decreasing boundaries
    int_bounds: List[int] = []
    last = -1
    for b in boundaries:
        v = int(round(b))
        if v < last:
            v = last
        int_bounds.append(v)
        last = v
    return int_bounds, boundaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze output length distribution and suggest bucket boundaries."
    )
    parser.add_argument(
        "--mapreduce_path",
        default="/home/linchx/vidur/data/map_reduce/MapReduceTraceNew.json",
    )
    parser.add_argument(
        "--sharegpt_path",
        default="/home/linchx/vidur/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    parser.add_argument("--max_mapreduce", type=int, default=20000)
    parser.add_argument("--max_sharegpt", type=int, default=20000)
    parser.add_argument("--num_buckets", type=int, default=6)
    parser.add_argument(
        "--percentiles",
        type=int,
        nargs="+",
        default=[50, 60, 70, 80, 90, 95, 99],
    )
    args = parser.parse_args()

    max_map = None if args.max_mapreduce is None else args.max_mapreduce
    max_sg = None if args.max_sharegpt is None else args.max_sharegpt

    mapreduce_lengths = collect_mapreduce_output_lengths(
        args.mapreduce_path, max_map
    )
    sharegpt_lengths = collect_sharegpt_output_lengths(
        args.sharegpt_path, max_sg
    )
    combined = mapreduce_lengths + sharegpt_lengths

    summarize("mapreduce", mapreduce_lengths, args.percentiles)
    summarize("sharegpt", sharegpt_lengths, args.percentiles)
    summarize("combined", combined, args.percentiles)

    int_bounds, float_bounds = suggest_buckets(combined, args.num_buckets)
    if int_bounds:
        quantiles = [i / args.num_buckets for i in range(1, args.num_buckets)]
        q_str = ", ".join(f"q{int(q*100)}={v:.2f}" for q, v in zip(quantiles, float_bounds))
        logger.info("Suggested %d-bucket boundaries (quantiles): %s", args.num_buckets, q_str)
        logger.info("Suggested bucket boundaries (ints): %s", ",".join(map(str, int_bounds)))
    else:
        logger.warning("No samples; cannot suggest bucket boundaries.")


if __name__ == "__main__":
    main()
