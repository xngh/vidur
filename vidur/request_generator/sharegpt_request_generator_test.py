

from vidur.request_generator.sharegpt_request_generator import ShareGPTRequestGenerator
from vidur.config import ShareGPTRequestGeneratorConfig

# 在 /vidur路径下，执行python -m vidur.request_generator.sharegpt_request_generator_test


if __name__ == "__main__":
    config = ShareGPTRequestGeneratorConfig({
        "trace_file": "data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
        "qps": 0.5,
    })

    ShareGPTRequestGenerator(config).test_load()

    ShareGPTRequestGenerator(config).test_generate_requests()