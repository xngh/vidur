from vidur.request_generator.map_reduce_request_generator import MapReduceRequestGenerator
from vidur.request_generator.code_agent_request_generator import CodeAgentRequestGenerator
from vidur.request_generator.mix_agent_request_generator import MixAgentRequestGenerator
from vidur.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from vidur.request_generator.trace_replay_request_generator import (
    TraceReplayRequestGenerator,
)
from vidur.request_generator.sharegpt_request_generator import (
    ShareGPTRequestGenerator,
)
from vidur.types import RequestGeneratorType
from vidur.utils.base_registry import BaseRegistry


class RequestGeneratorRegistry(BaseRegistry):
    pass


RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.TRACE_REPLAY, TraceReplayRequestGenerator
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.UNIFIED, ShareGPTRequestGenerator
)

RequestGeneratorRegistry.register(
    RequestGeneratorType.MAPREDUCE, MapReduceRequestGenerator
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.CODE, CodeAgentRequestGenerator
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.MIXED, MixAgentRequestGenerator
)