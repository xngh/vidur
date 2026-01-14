import logging
import json
import random
from typing import List, Dict

import pandas as pd

from vidur.config import MapReduceRequestGeneratorConfig
from vidur.entities.unified_request import UnifiedRequest
from vidur.request_generator.base_request_generator import BaseRequestGenerator
from vidur.request_generator.poisson_request_interval_generator import PoissonRequestIntervalGenerator

logger = logging.getLogger(__name__)


class MapReduceRequestGenerator(BaseRequestGenerator):
    def __init__(self, config: MapReduceRequestGeneratorConfig):
        # 1. 这里的格式是标准 JSON，直接加载为列表
        with open(config.trace_file, 'r', encoding='utf-8') as f:
            self.trace_data = json.load(f)

        self.poisson_request_interval_generator = PoissonRequestIntervalGenerator(
            config.interval_generator_config
        )
        self.time = config.start_time
        self.max_tokens = config.max_tokens

        # NOTE: 以下分位点为示例占位值，可根据最新仿真数据替换
        # 单位: 秒；假设为 workflow 级 e2e runtime 的 P50/P70/P90
        self._p50_runtime = 3.0
        self._p70_runtime = 5.0
        self._p90_runtime = 8.0

    def _sample_deadline_budget(self) -> float:
        """
        按照给定分段概率与安全系数生成相对 deadline（秒）。
        分段概率: [0.6, 0.3, 0.1] -> [P50,P70], [P70,P90], [P90, 1.5*P90]
        安全系数: 乘以 U(1.05, 1.20)
        """
        p = random.random()
        if p < 0.6:
            base_budget = random.uniform(self._p50_runtime, self._p70_runtime)
        elif p < 0.9:
            base_budget = random.uniform(self._p70_runtime, self._p90_runtime)
        else:
            upper = 1.5 * self._p90_runtime
            base_budget = random.uniform(self._p90_runtime, upper)

        safety = random.uniform(1.05, 1.20)
        return base_budget * safety

    def generate_unified_request(self, arrive_at, workflow_data: Dict) -> UnifiedRequest:
        workflow_id = workflow_data["agent_id"]

        workflow_steps_config = []
        step_count = 0
        former_step_name = (
            workflow_data["summary"][0]["node_name"] if len(workflow_data["summary"]) else None
        )
        for node in workflow_data["summary"]:
            if node["node_name"] != former_step_name:
                step_count += 1
                former_step_name = node["node_name"]
            step_config = {
                "step": workflow_id + "_" + str(step_count),
                # 保留节点类型信息，方便后续在 UnifiedRequest.get_next_requests() 中做定向处理
                "node_name": node["node_name"],
                "input_str": node["inputs"],
                "output_str": node["outputs"],
            }
            workflow_steps_config.append(step_config)

        if not workflow_steps_config:
            return None

        # 4. 实例化 UnifiedRequest，格式与之前定义的 class 完全一致
        return UnifiedRequest(
            workflow_id=workflow_id,
            arrive_at=arrive_at,
            workflow_config=workflow_steps_config,
            max_token_for_request=self.max_tokens,
            deadline=arrive_at + self._sample_deadline_budget()
        )

    def generate_requests(self) -> List[UnifiedRequest]:
        requests = []

        for workflow_item in self.trace_data:
            # 获取下一个 Poisson 到达间隔并累加时间
            inter_request_time = self.poisson_request_interval_generator.get_next_inter_request_time()
            self.time += inter_request_time

            request = self.generate_unified_request(self.time, workflow_item)
            if request:
                requests.append(request)

            # 限制最大生成请求数，防止内存溢出
            if len(requests) >= 8000:
                break

        return requests