from vidur.config import MapReduceRequestGeneratorConfig, PoissonRequestIntervalGeneratorConfig
from vidur.request_generator.map_reduce_request_generator import MapReduceRequestGenerator
from pprint import pprint
from typing import Any, Dict, List


def _shorten(s: Any, n: int = 160) -> Any:
    if not isinstance(s, str):
        return s
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"...(len={len(s)})"


def _dump_unified_request(req: Any, idx: int) -> None:
    data: Dict[str, Any] = {
        "index": idx,
        "type": type(req).__name__,
        "workflow_id": getattr(req, "workflow_id", None),
        "arrive_at": getattr(req, "arrive_at", None),
        "workflow_status": getattr(req, "workflow_status", None),
        "total_steps": getattr(req, "total_steps", None),
        "current_step_index": getattr(req, "current_step_index", None),
        "step_names": getattr(req, "step_names", None),
        "workflow_config_len": len(getattr(req, "workflow_config", []) or []),
        "active_requests_len": len(getattr(req, "active_requests", []) or []),
        "deadline": getattr(req, "deadline", None),
    }
    pprint(data, width=120, sort_dicts=False)


def _dump_full_request(fr: Any, idx: int) -> None:
    # FullRequest 有 to_dict()，但里面可能包含很长的 token_id 列表，这里只打印摘要信息
    to_dict = getattr(fr, "to_dict", None)
    d = to_dict() if callable(to_dict) else (fr.__dict__ if hasattr(fr, "__dict__") else {"repr": repr(fr)})

    summary = {
        "index": idx,
        "type": type(fr).__name__,
        "req_id": getattr(fr, "req_id", None),
        "id": d.get("id", None),
        "arrived_at": d.get("arrived_at", getattr(fr, "arrived_at", None)),
        "num_prefill_tokens": d.get("num_prefill_tokens", getattr(fr, "num_prefill_tokens", None)),
        "num_decode_tokens": d.get("num_decode_tokens", getattr(fr, "num_decode_tokens", None)),
        "input_str": _shorten(getattr(fr, "input_str", d.get("input_str", ""))),
        "output_str": _shorten(getattr(fr, "output_str", d.get("output_str", ""))),
        "input_token_ids_len": len(getattr(fr, "input_token_ids", []) or []),
        "output_token_ids_len": len(getattr(fr, "output_token_ids", []) or []),
        "generated_token_ids_len": len(getattr(fr, "generated_token_ids", []) or []),
        "block_table": getattr(fr, "block_table", d.get("block_table", None)),
        "is_parallelizable": getattr(fr, "is_parallelizable", None),
    }
    pprint(summary, width=120, sort_dicts=False)


def test_load():
    interval_cfg = PoissonRequestIntervalGeneratorConfig(qps=0.5)

    config = MapReduceRequestGeneratorConfig(
        trace_file="data/map_reduce/expected_format.json",
        interval_generator_config=interval_cfg,  # 传入嵌套配置
    )
    print(config.trace_file)
    generator = MapReduceRequestGenerator(config)

    data_len = len(generator.trace_data)
    print(f"Data length: {data_len}")

    # sample
    print(generator.trace_data)
    first_row = generator.trace_data[0]

    first_conversation = first_row["summary"]
    # print(f"content: {first_conversation}")

    generator.generate_unified_request(0, first_row)

def test_init_requests():
    interval_cfg = PoissonRequestIntervalGeneratorConfig(qps=0.5)

    config = MapReduceRequestGeneratorConfig(
        trace_file="data/map_reduce/expected_format.json",
        interval_generator_config=interval_cfg,  # 传入嵌套配置
    )
    generator = MapReduceRequestGenerator(config)

    requests = generator.generate_requests()
    print(f"requests 总数: {len(requests)}")
    max_show = 10
    for i, r in enumerate(requests[:max_show]):
        print(f"\n=== UnifiedRequest[{i}] ===")
        _dump_unified_request(r, i)

    full_requests_list = requests[0].get_next_requests(0, "")
    print(f"\nrequests[0].get_next_requests(...) 返回 FullRequest 数: {len(full_requests_list)}")
    for i, fr in enumerate(full_requests_list[:max_show]):
        print(f"\n--- FullRequest[{i}] ---")
        _dump_full_request(fr, i)

    # # 关键校验：MapReduce workflow 应该在 Map 阶段之后还能发射 Reduce 阶段的 request
    # # 这里模拟 map 阶段全部完成，然后触发下一阶段
    # app = requests[0]
    # for r in list(app.active_requests):
    #     app.update_on_request_finish(r, 0.1)
    # reduce_reqs = app.get_next_requests(0.2, app.context_information)
    # print(f"\nMap 阶段完成后，get_next_requests(...) 返回 Reduce FullRequest 数: {len(reduce_reqs)}")
    # assert len(reduce_reqs) > 0, "Reduce request 未生成（疑似被过滤或未触发）"

    # assert len(full_requests_list) == 1

if __name__ == "__main__":
    test_init_requests()