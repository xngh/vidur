#!/usr/bin/env python3
"""
为三个 agent trace 数据集绘制 prefill / decode 长度的 CDF 图（matplotlib）。

- MapReduceTraceNew.json: 使用每条记录自带的 prompt_tokens / completion_tokens。
- CodeAgentTrace_Diversified_v3.json 与 ShareGPT JSON: 使用每条「对话轮」的
  prompt 文本长度与生成文本长度；默认用 tiktoken cl100k_base 计 token（与常见 LLM
  口径一致）。若未安装 tiktoken，则回退为按空白分词后的词数（近似）。

大 JSON 数组文件以流式解析，避免一次性读入内存。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Callable, Generator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

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


def make_token_counter() -> Callable[[str], int]:
    """返回计数字符串长度的函数。

    - 有 tiktoken：BPE token 数（cl100k_base），与常见 LLM 口径一致。
    - 无 tiktoken：用 ``len(text.split())`` 仅作粗近似——**不是**「一词一 token」：
      英文里多词常合并为更少 BPE；中文常无空格，整段可能被当成极少段。
    """
    enc = None

    def count_tokens(text: str) -> int:
        nonlocal enc
        if text is None:
            return 0
        if not text:
            return 0
        try:
            import tiktoken  # type: ignore

            if enc is None:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return len(text.split())

    return count_tokens


def cdf_xy(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """经验 CDF：x 为排序后的取值，y 为 [1/n, ..., 1]。"""
    if values.size == 0:
        return np.array([]), np.array([])
    x = np.sort(values.astype(np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def extract_map_reduce(path: str, max_requests: Optional[int]) -> Tuple[List[int], List[int]]:
    prefill: List[int] = []
    decode: List[int] = []
    n = 0
    for wf in iter_json_array_objects(path):
        for step in wf.get("summary", []):
            pt = step.get("prompt_tokens")
            ct = step.get("completion_tokens")
            if pt is None or ct is None:
                continue
            prefill.append(int(pt))
            decode.append(int(ct))
            n += 1
            if max_requests is not None and n >= max_requests:
                return prefill, decode
    return prefill, decode


def extract_code_agent(path: str, count_tokens: Callable[[str], int], max_requests: Optional[int]) -> Tuple[List[int], List[int]]:
    prefill: List[int] = []
    decode: List[int] = []
    n = 0
    for item in iter_json_array_objects(path):
        for turn in item.get("conversations", []):
            inp = turn.get("input", "") or ""
            out = turn.get("output", "") or ""
            prefill.append(count_tokens(inp))
            decode.append(count_tokens(out))
            n += 1
            if max_requests is not None and n >= max_requests:
                return prefill, decode
    return prefill, decode


def extract_sharegpt(path: str, count_tokens: Callable[[str], int], max_requests: Optional[int]) -> Tuple[List[int], List[int]]:
    """ShareGPT 风格：human / gpt 交替，每对为一轮请求。"""
    prefill: List[int] = []
    decode: List[int] = []
    n = 0
    for item in iter_json_array_objects(path):
        conv = item.get("conversations", [])
        i = 0
        while i + 1 < len(conv):
            a, b = conv[i], conv[i + 1]
            fa = (a.get("from") or "").lower()
            fb = (b.get("from") or "").lower()
            if fa == "human" and fb == "gpt":
                inp = a.get("value", "") or ""
                out = b.get("value", "") or ""
                prefill.append(count_tokens(inp))
                decode.append(count_tokens(out))
                n += 1
                i += 2
                if max_requests is not None and n >= max_requests:
                    return prefill, decode
            else:
                i += 1
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
        ax.plot(x1, y1, label="Prefill token 数", linewidth=2)
    if dc.size:
        x2, y2 = cdf_xy(dc)
        ax.plot(x2, y2, label="Decode token 数", linewidth=2)

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
    parser = argparse.ArgumentParser(description="Trace 数据集 prefill/decode 长度 CDF")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "trace_cdf_plots"),
        help="输出图片目录",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="每种数据源最多采样的请求条数（调试用；默认不截断）",
    )
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)

    count_tokens = make_token_counter()
    try:
        import tiktoken  # noqa: F401

        text_xlabel = "Token数"
    except ImportError:
        text_xlabel = "Token数"

    datasets = [
        (
            "ShareGPT V3",
            os.path.join(os.path.dirname(__file__), "..", "data", "sharegpt", "ShareGPT_V3_unfiltered_cleaned_split.json"),
            lambda p: extract_sharegpt(p, count_tokens, args.max_requests),
            text_xlabel,
        ),
        (
            "MapReduce Trace",
            os.path.join(os.path.dirname(__file__), "..", "data", "map_reduce", "MapReduceTraceNew.json"),
            lambda p: extract_map_reduce(p, args.max_requests),
            "长度（token 数：数据集中的 prompt_tokens / completion_tokens）",
        ),
        (
            "Code Agent Trace",
            os.path.join(os.path.dirname(__file__), "..", "data", "code_agent_traces", "CodeAgentTrace_Diversified_v3.json"),
            lambda p: extract_code_agent(p, count_tokens, args.max_requests),
            text_xlabel,
        ),
    ]

    safe = lambda s: re.sub(r"[^\w\-]+", "_", s).strip("_")

    for name, path, extractor, xlabel in datasets:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            print(f"跳过（文件不存在）: {path}")
            continue
        print(f"处理 {name} …")
        prefill, decode = extractor(path)
        print(f"  样本数: prefill={len(prefill)}, decode={len(decode)}")
        out_pdf = os.path.join(out_dir, f"cdf_prefill_decode_{safe(name)}.pdf")
        plot_one(f"{name}：Prefill 与 Decode 长度累积分布", prefill, decode, out_pdf, xlabel)
        print(f"  已保存: {out_pdf}")

    print("完成。")


if __name__ == "__main__":
    main()
