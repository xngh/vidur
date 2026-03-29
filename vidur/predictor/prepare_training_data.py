import argparse
import json
from typing import Dict, Iterator, List, Optional, Tuple

'''
生成的训练数据集字段含义：
text：模型输入文本，即请求的 Prompt（对 MapReduce 是 inputs，对 ShareGPT 是 human 轮的文本）。
task_type：任务类型编码，用于结构嵌入层（0=Map_Worker，1=Reduce_Master，2=ShareGPT/General）。
prompt_len：prompt 长度（优先用 prompt_tokens；没有则用词数 split()），用于学习“输入-输出比例”关系。
output_len：输出长度（近似标签的“真实值”）。MapReduce 优先取 completion_tokens，ShareGPT 用 gpt 回复的词数（split()）。
label：输出长度对应的桶区间编号（根据 --buckets 的边界，默认是 0..3 四类）。
source：样本来源标识，mapreduce 或 sharegpt，便于统计和分析分布差异。
node_name：原始任务节点名称，MapReduce 会保留 Map_Worker/Reduce_Master，ShareGPT 固定为 ShareGPT，用于可视化或后续分析。
'''
def iter_json_array(path: str) -> Iterator[Dict]:
    """
    Stream a JSON array from disk without loading the full file into memory.
    This supports large datasets like ShareGPT and MapReduce traces.
    """
    decoder = json.JSONDecoder()
    buffer = ""

    with open(path, "r", encoding="utf-8") as f:
        # Find the array start "["
        while True:
            chunk = f.read(65536)
            if not chunk:
                return
            buffer += chunk
            buffer = buffer.lstrip()
            if buffer.startswith("["):
                buffer = buffer[1:]
                break
            # Keep only a small tail if we haven't found '[' yet
            buffer = buffer[-1024:]

        # Decode objects one by one
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return

            try:
                obj, idx = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = f.read(65536)
                if not chunk:
                    raise
                buffer += chunk
                continue

            yield obj
            buffer = buffer[idx:].lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]


def parse_buckets(buckets_str: str) -> List[int]:
    buckets = [int(x.strip()) for x in buckets_str.split(",") if x.strip()]
    if not buckets:
        raise ValueError("buckets is empty")
    return sorted(buckets)


def bucketize(length: int, boundaries: List[int]) -> int:
    for i, bound in enumerate(boundaries):
        if length <= bound:
            return i
    return len(boundaries)


def extract_mapreduce_samples(
    path: str,
    max_samples: Optional[int],
    boundaries: List[int],
) -> Iterator[Dict]:
    count = 0
    for workflow in iter_json_array(path):
        summary = workflow.get("summary", [])
        for call in summary:
            node_name = call.get("node_name", "")
            if not call.get("inputs") or not call.get("outputs"):
                continue

            if node_name == "Map_Worker":
                task_type = 0
            elif node_name == "Reduce_Master":
                task_type = 1
            else:
                # Skip other nodes for now
                continue

            # Prefer completion_tokens when available; fallback to word count.
            output_len = call.get("completion_tokens")
            if not isinstance(output_len, int):
                output_len = len(str(call.get("outputs", "")).split())

            # Prefer prompt_tokens when available; fallback to word count.
            prompt_len = call.get("prompt_tokens")
            if not isinstance(prompt_len, int):
                prompt_len = len(str(call.get("inputs", "")).split())

            sample = {
                "text": call.get("inputs", ""),
                "task_type": task_type,
                "prompt_len": int(prompt_len),
                "output_len": int(output_len),
                "label": bucketize(int(output_len), boundaries),
                "source": "mapreduce",
                "node_name": node_name,
            }
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def extract_sharegpt_samples(
    path: str,
    max_samples: Optional[int],
    boundaries: List[int],
) -> Iterator[Dict]:
    count = 0
    for record in iter_json_array(path):
        conversations = record.get("conversations", [])
        # Use consecutive human->gpt pairs as samples
        for i in range(0, len(conversations) - 1, 2):
            turn_in = conversations[i]
            turn_out = conversations[i + 1]
            if turn_in.get("from") != "human" or turn_out.get("from") != "gpt":
                continue

            prompt = turn_in.get("value", "")
            output = turn_out.get("value", "")
            if not prompt or not output:
                continue

            output_len = len(str(output).split())
            prompt_len = len(str(prompt).split())
            sample = {
                "text": prompt,
                "task_type": 2,  # ShareGPT / General chat
                "prompt_len": int(prompt_len),
                "output_len": int(output_len),
                "label": bucketize(int(output_len), boundaries),
                "source": "sharegpt",
                "node_name": "ShareGPT",
            }
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def write_jsonl(samples: Iterator[Dict], output_path: str) -> int:
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for WorkloadProfiler.")
    parser.add_argument(
        "--mapreduce_path",
        default="/home/linchx/vidur/data/map_reduce/MapReduceTraceNew.json",
        help="Path to MapReduce trace JSON.",
    )
    parser.add_argument(
        "--sharegpt_path",
        default="/home/linchx/vidur/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
        help="Path to ShareGPT JSON.",
    )
    parser.add_argument(
        "--output_path",
        default="/home/linchx/vidur/data/processed_traces/workload_profiler_train.jsonl",
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--buckets",
        # default="50,100,200,350",
        default="40,120,250,340",
        help="Comma-separated bucket boundaries. Example: 50,200,500",
    )
    parser.add_argument(
        "--max_mapreduce",
        type=int,
        default=20000,
        help="Max MapReduce samples to extract (None for all).",
    )
    parser.add_argument(
        "--max_sharegpt",
        type=int,
        default=20000,
        help="Max ShareGPT samples to extract (None for all).",
    )

    args = parser.parse_args()
    boundaries = parse_buckets(args.buckets)

    # Stream and merge samples from two sources
    def merged_samples() -> Iterator[Dict]:
        if args.max_mapreduce == 0:
            pass
        else:
            max_map = None if args.max_mapreduce is None else args.max_mapreduce
            for s in extract_mapreduce_samples(args.mapreduce_path, max_map, boundaries):
                yield s

        if args.max_sharegpt == 0:
            pass
        else:
            max_sg = None if args.max_sharegpt is None else args.max_sharegpt
            for s in extract_sharegpt_samples(args.sharegpt_path, max_sg, boundaries):
                yield s

    total = write_jsonl(merged_samples(), args.output_path)
    print(f"Wrote {total} samples to {args.output_path}")


if __name__ == "__main__":
    main()
