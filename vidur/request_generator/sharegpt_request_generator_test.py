

from vidur.request_generator.sharegpt_request_generator import ShareGPTRequestGenerator
from vidur.config import ShareGPTRequestGeneratorConfig

# 在 /vidur 路径下执行:
# python -m vidur.request_generator.sharegpt_request_generator_test


if __name__ == "__main__":
    config = ShareGPTRequestGeneratorConfig(
        {
            "trace_file": "data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
            "qps": 0.5,
        }
    )

    generator = ShareGPTRequestGenerator(config)

    # 仅取前 2 条样本，生成 UnifiedRequest 并打印含 deadline 的摘要
    time_cursor = generator.time
    head_rows = generator.trace_df.head(2)
    for _, row in head_rows.iterrows():
        time_cursor += generator.poisson_request_interval_generator.get_next_inter_request_time()
        req = generator.generate_unified_request(time_cursor, row)
        print(
            f"[request] id={req.workflow_id} arrive_at={req.arrive_at:.3f} "
            f"deadline={req.deadline:.3f} steps={len(req.workflow_config)}"
        )