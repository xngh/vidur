#!/usr/bin/env python3
"""
从仿真导出的 request_metrics.csv 统计 prefill / decode token 长度，并绘制 CDF。

默认读取 data/trace_cdf_data/ 下三个 agent 的运行结果 CSV，使用列：
  request_num_prefill_tokens, request_num_decode_tokens
（与 Vidur 导出的 request_metrics.csv 表头一致。）

可选 --truncate-pct：对每一行先取 max(prefill, decode)，再对该序列算 P 分位阈值，
只保留「行内 max <= 阈值」的样本；prefill/decode 两条 CDF 共用同一批行。
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass
plt.rc("font", family="Noto Sans CJK JP")
plt.rcParams["axes.unicode_minus"] = False

COL_PREFILL = "request_num_prefill_tokens"
COL_DECODE = "request_num_decode_tokens"


def truncate_by_max_prefill_decode(
    prefill: List[int],
    decode: List[int],
    percentile: float,
) -> Tuple[List[int], List[int]]:
    """对每行取 max(prefill, decode)，用该序列的 P 分位数作阈值，保留行内 max <= 阈值的行。"""
    pf = np.asarray(prefill, dtype=np.float64)
    dc = np.asarray(decode, dtype=np.float64)
    if pf.size == 0:
        return [], []
    joint = np.maximum(pf, dc)
    if percentile >= 100:
        return pf.astype(np.int64).tolist(), dc.astype(np.int64).tolist()
    thr = float(np.percentile(joint, percentile))
    mask = joint <= thr
    return pf[mask].astype(np.int64).tolist(), dc[mask].astype(np.int64).tolist()


def cdf_xy(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])
    x = np.sort(values.astype(np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def load_prefill_decode_from_csv(
    path: str,
    max_rows: Optional[int],
) -> Tuple[List[int], List[int]]:
    df = pd.read_csv(path)
    if COL_PREFILL not in df.columns or COL_DECODE not in df.columns:
        raise ValueError(
            f"{path} 缺少列 {COL_PREFILL!r} 或 {COL_DECODE!r}，实际列: {list(df.columns)}"
        )
    df = df[[COL_PREFILL, COL_DECODE]].dropna()
    if max_rows is not None:
        df = df.head(max_rows)
    prefill = df[COL_PREFILL].astype(np.int64).tolist()
    decode = df[COL_DECODE].astype(np.int64).tolist()
    return prefill, decode


def plot_one(
    title: str,
    prefill: List[int],
    decode: List[int],
    out_path: str,
    xlabel: str,
) -> None:
    pf = np.asarray(prefill, dtype=np.int64)
    dc = np.asarray(decode, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 5))
    if pf.size:
        x1, y1 = cdf_xy(pf)
        ax.plot(x1, y1, label="Prefill tokens", linewidth=2)
    if dc.size:
        x2, y2 = cdf_xy(dc)
        ax.plot(x2, y2, label="Decode tokens", linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("累积分布")
    # ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.35)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 request_metrics.csv 绘制 prefill/decode token 的 CDF"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "trace_cdf_data"),
        help="存放各 agent 的 request_metrics.csv 的目录",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "trace_cdf_plots"),
        help="输出图片目录",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="每个 CSV 最多使用前 N 行（默认用全部有效行）",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="pdf",
        help="输出图片格式",
    )
    parser.add_argument(
        "--truncate-pct",
        type=float,
        default=None,
        metavar="P",
        help="截断长尾：每行 max(prefill,decode) 的 P 分位为阈值，去掉行内 max 超过该阈值的行（如 99）。不设则全量",
    )
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)

    datasets = [
        ("ShareGPT V3", "sharegpt_request_metrics.csv"),
        ("MapReduce Trace", "mapreduce_request_metrics.csv"),
        ("Code Agent Trace", "coder_request_metrics.csv"),
    ]

    xlabel_base = "Token数"
    safe = lambda s: re.sub(r"[^\w\-]+", "_", s).strip("_")

    tp = args.truncate_pct
    if tp is not None and not (0 < tp <= 100):
        raise SystemExit("--truncate-pct 须在 (0, 100] 内，例如 99")

    for name, csv_name in datasets:
        path = os.path.join(data_dir, csv_name)
        if not os.path.isfile(path):
            print(f"跳过（文件不存在）: {path}")
            continue
        print(f"处理 {name} …")
        try:
            prefill, decode = load_prefill_decode_from_csv(path, args.max_rows)
        except ValueError as e:
            print(f"  错误: {e}")
            continue
        n0 = len(prefill)
        if tp is not None:
            prefill, decode = truncate_by_max_prefill_decode(prefill, decode, tp)
            print(
                f"  截断 P{tp:g}（按行 max(prefill,decode)）：原始 {n0} 行 → 保留 {len(prefill)} 行"
            )
        else:
            print(f"  样本数: {len(prefill)}")

        xlabel = xlabel_base
        title_suffix = "（CSV）"
        if tp is not None:
            title_suffix = f"（CSV，截断至 P{tp:g}）"
            # xlabel = f"{xlabel_base}\n（去掉 max(prefill,decode) > P{tp:g} 分位阈值的行）"

        ext = args.format
        if tp is not None:
            tp_tag = str(int(tp)) if float(tp) == int(float(tp)) else str(tp).replace(".", "_")
            suffix = f"_p{tp_tag}"
        else:
            suffix = ""
        out_png = os.path.join(
            out_dir, f"cdf_prefill_decode_2_{safe(name)}{suffix}.{ext}"
        )
        plot_one(
            f"{name}：Prefill 与 Decode token 长度累积分布{title_suffix}",
            prefill,
            decode,
            out_png,
            xlabel,
        )
        print(f"  已保存: {out_png}")

    print("完成。")


if __name__ == "__main__":
    main()
