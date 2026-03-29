#!/usr/bin/env python3
"""
统计三个 trace 数据集中「每个 workflow 包含的 request 数量」，并画在一张图上的经验 CDF。

request 计数口径与 plot_trace_prefill_decode_cdf.py 一致：
- MapReduce：每个 workflow 的 summary 列表长度（每次 LLM 调用计 1）。
- Code Agent：每个 workflow 的 conversations 条目数。
- ShareGPT：conversations 中 human→gpt 配对数。

大 JSON 以流式解析；--max-workflows 表示每个数据集最多处理的 workflow 条数（从文件开头顺序取）。

长尾分布：默认横轴为对数刻度（--x-scale log），并采用阶梯状 ECDF（离散整数更合适）。
也可用 --x-scale linear 配合 --x-clip-pct 只显示主区间。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Generator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass
plt.rc("font", family="Noto Sans CJK JP")
plt.rcParams["axes.unicode_minus"] = False


def iter_json_array_objects(path: str) -> Generator[dict, None, None]:
    """从形如 [ {...}, {...}, ... ] 的 JSON 文件中逐个 yield 顶层对象。"""
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return
            if ch == "[":
                break
        buf = ""
        while True:
            while True:
                if not buf:
                    chunk = f.read(65536)
                    if not chunk:
                        return
                    buf += chunk
                buf = buf.lstrip()
                if not buf:
                    continue
                if buf[0] == "]":
                    return
                if buf[0] == ",":
                    buf = buf[1:].lstrip()
                    continue
                break
            while True:
                try:
                    obj, idx = decoder.raw_decode(buf)
                    yield obj
                    buf = buf[idx:]
                    break
                except json.JSONDecodeError:
                    chunk = f.read(65536)
                    if not chunk:
                        raise RuntimeError(f"无法完整解析 JSON: {path}")
                    buf += chunk


def cdf_xy(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])
    x = np.sort(values.astype(np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def count_requests_mapreduce(wf: dict) -> int:
    return len(wf.get("summary", []))


def count_requests_code_agent(wf: dict) -> int:
    return len(wf.get("conversations", []))


def count_requests_sharegpt(wf: dict) -> int:
    conv = wf.get("conversations", [])
    n = 0
    i = 0
    while i + 1 < len(conv):
        a, b = conv[i], conv[i + 1]
        fa = (a.get("from") or "").lower()
        fb = (b.get("from") or "").lower()
        if fa == "human" and fb == "gpt":
            n += 1
            i += 2
        else:
            i += 1
    return n


def collect_counts(
    path: str,
    counter: Callable[[dict], int],
    max_workflows: Optional[int],
) -> List[int]:
    out: List[int] = []
    for k, wf in enumerate(iter_json_array_objects(path)):
        if max_workflows is not None and k >= max_workflows:
            break
        out.append(counter(wf))
    return out


def _prepare_x_for_plot(x: np.ndarray, log_scale: bool) -> np.ndarray:
    """对数轴时避免出现 0 或负数。"""
    if not log_scale:
        return x
    return np.maximum(x, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="各 workflow 内 request 数量的 CDF（三数据集同图）"
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "data", "trace_cdf_plots", "workflow_request_count_cdf.pdf"
        ),
        help="输出图片路径（支持 .png / .pdf）",
    )
    parser.add_argument(
        "--max-workflows",
        type=int,
        default=None,
        help="每个数据集最多处理的 workflow 数量（默认不限制）",
    )
    parser.add_argument(
        "--x-scale",
        choices=("linear", "log"),
        default="log",
        help="横轴刻度：log 可压缩长尾（默认）；linear 为线性轴",
    )
    parser.add_argument(
        "--x-clip-pct",
        type=float,
        default=None,
        metavar="PCT",
        help="仅在线性轴下生效：横轴上限取「合并后样本」的 PCT 分位数（如 99），便于看主体区间",
    )
    parser.add_argument(
        "--smooth-line",
        action="store_true",
        help="用折线代替阶梯线（默认阶梯更符合离散计数的 ECDF）",
    )
    args = parser.parse_args()
    out_path = os.path.abspath(args.out)
    log_x = args.x_scale == "log"

    base = os.path.join(os.path.dirname(__file__), "..", "data")
    datasets = [
        ("Chatbot Agent数据集", os.path.join(base, "sharegpt", "ShareGPT_V3_unfiltered_cleaned_split.json"), count_requests_sharegpt),
        ("MapReduce Agent数据集", os.path.join(base, "map_reduce", "MapReduceTraceNew.json"), count_requests_mapreduce),
        ("Code Agent数据集", os.path.join(base, "code_agent_traces", "CodeAgentTrace_Diversified_v3.json"), count_requests_code_agent),
    ]

    series: List[Tuple[str, np.ndarray]] = []
    for label, path, counter in datasets:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            print(f"跳过（文件不存在）: {path}")
            continue
        counts = collect_counts(path, counter, args.max_workflows)
        arr = np.asarray(counts, dtype=np.float64)
        print(f"{label}: workflow 数 = {len(counts)}")
        if arr.size == 0:
            continue
        series.append((label, arr))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = ("#2563eb", "#ea580c", "#16a34a")

    xmax_clip: Optional[float] = None
    if not log_x and args.x_clip_pct is not None and series:
        merged = np.concatenate([a for _, a in series])
        xmax_clip = float(np.percentile(merged, args.x_clip_pct))

    for (label, arr), color in zip(series, colors):
        x, y = cdf_xy(arr)
        xp = _prepare_x_for_plot(x, log_x)
        draw = ax.plot if args.smooth_line else ax.step
        if args.smooth_line:
            draw(xp, y, label=label, linewidth=2.4, color=color, alpha=0.92, solid_capstyle="round")
        else:
            draw(xp, y, where="post", label=label, linewidth=2.4, color=color, alpha=0.92)

    if log_x:
        ax.set_xscale("log")
        ax.set_xlabel("单个 workflow 内的 request 数量（对数横轴）")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title(
            "各数据集：每个 workflow 中 request 条数的累积分布\n（对数横轴，压缩长尾）"
        )
    else:
        ax.set_xlabel("请求数量")
        if xmax_clip is not None:
            ax.set_xlim(left=0, right=xmax_clip)
            # ax.set_title(
            #     "各数据集：每个 workflow 中 request 条数的累积分布\n"
            #     f"（线性轴，横轴上限为合并样本的 {args.x_clip_pct:g}% 分位）"
            # )
        else:
            ax.set_title("各数据集：每个 workflow 中 request 条数的累积分布")

    ax.set_ylabel("累积分布")
    ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.95)
    ax.grid(True, which="major", alpha=0.45)
    if log_x:
        ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
