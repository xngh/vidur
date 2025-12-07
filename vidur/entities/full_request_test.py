from vidur.request_generator.sharegpt_request_generator import ShareGPTRequestGenerator
from vidur.config import ShareGPTRequestGeneratorConfig
from vidur.entities.unified_request import UnifiedRequest
from vidur.entities.full_request import FullRequest


# python -m vidur.entities.full_request_test
def test_tokenizer(req: FullRequest):
    print(req.input_token_ids)
    print(req.input_str)
    
    assert len(req.input_token_ids) == req.num_prefill_tokens

if __name__ == "__main__":
    config = ShareGPTRequestGeneratorConfig({
        "trace_file": "data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
        "qps": 0.5,
    })

    g = ShareGPTRequestGenerator(config)

    reqs = g.generate_requests()
    req = reqs[0]

    req.get_next_requests(0) # get next full requests

    r = req.active_requests[0]
    test_tokenizer(r)

