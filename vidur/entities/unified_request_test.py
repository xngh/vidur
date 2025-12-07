
import test
from vidur.request_generator.sharegpt_request_generator import ShareGPTRequestGenerator
from vidur.config import ShareGPTRequestGeneratorConfig
from vidur.entities.unified_request import UnifiedRequest
from vidur.entities.full_request import FullRequest

# /vidur   python -m vidur.entities.unified_request_test

def test_unified_request_config(req: UnifiedRequest):

    assert req.total_steps != 0 

    for node in req.workflow_config:
        assert node["step"] != ""
        assert node["input_str"] != ""
        assert node["output_str"] != ""
        assert node["num_decode_tokens"] != 0
        assert node["num_prefill_tokens"] != 0

def test_get_next_request(req: UnifiedRequest):
    child = req.get_next_requests(0)

    assert type(child[0]) == FullRequest 
    assert len(child) == 1


def test_initialize_step_names(req: UnifiedRequest):
    req._initialize_step_names()

    assert req.step_names != []

    print(f"req's step_names: {req.step_names}")


if __name__ == "__main__":
    config = ShareGPTRequestGeneratorConfig({
        "trace_file": "data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
        "qps": 0.5,
    })

    g = ShareGPTRequestGenerator(config)

    reqs = g.generate_requests()
    req = reqs[0]

    test_unified_request_config(req)

    test_get_next_request(req)

    test_initialize_step_names(req)


