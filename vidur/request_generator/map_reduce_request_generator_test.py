from torch.fx.experimental.unification.unification_tools import first

from vidur.config import MapReduceRequestGeneratorConfig, PoissonRequestIntervalGeneratorConfig
from vidur.request_generator.map_reduce_request_generator import MapReduceRequestGenerator


def test_load():
    interval_cfg = PoissonRequestIntervalGeneratorConfig(qps=0.5)

    config = MapReduceRequestGeneratorConfig(
        trace_file="data/map_reduce/test.json",
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
        trace_file="data/map_reduce/test.json",
        interval_generator_config=interval_cfg,  # 传入嵌套配置
    )
    generator = MapReduceRequestGenerator(config)

    requests = generator.generate_requests()
    print(requests)

    full_requests_list = requests[0].get_next_requests(0, "")

    assert len(full_requests_list) == 1

if __name__ == "__main__":
    test_init_requests()