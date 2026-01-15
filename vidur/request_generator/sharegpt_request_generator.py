import logging
import random
from typing import List, Dict

import pandas as pd

from vidur.config import ShareGPTRequestGeneratorConfig
from vidur.entities.unified_request import UnifiedRequest
from vidur.request_generator.base_request_generator import BaseRequestGenerator
from vidur.request_generator.poisson_request_interval_generator import PoissonRequestIntervalGenerator

logger = logging.getLogger(__name__)

class ShareGPTRequestGenerator(BaseRequestGenerator):
    def __init__(self, config: ShareGPTRequestGeneratorConfig):
        self.trace_df = pd.read_json(config.trace_file)
        self.poisson_request_interval_generator = PoissonRequestIntervalGenerator(
            config.interval_generator_config
        )
        self.time = config.start_time
        self.max_tokens = config.max_tokens

        # NOTE: 以下分位点为示例占位值，可根据最新仿真数据替换
        # 单位: 秒；假设为 ShareGPT 对话型请求的 e2e runtime P50/P70/P90
        self._p50_runtime = 1.0
        self._p70_runtime = 2.0
        self._p90_runtime = 4.0
        
         #设置一个随机数种子，保证deadline生成是可复现的。
        seed = getattr(config, "seed", None) or config.get("seed", None)
        if seed is not None:
            random.seed(seed)
        else:
            seed = 42 
            random.seed(seed)

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

    # TODO: 后一个request，是否需要把前一个request的input和output拼接起来作为history
    def generate_unified_request(self, arrive_at, row) -> UnifiedRequest:
        id = row["id"]
        conversations = row["conversations"]

        workflow_steps_config: List[List[Dict]] = []
        
        # 遍历对话，每次取2个
        for i in range(0, len(conversations), 2):
            if i + 1 >= len(conversations):
                continue
                
            turn1 = conversations[i]
            turn2 = conversations[i + 1]
            
            input_str = turn1["value"]
            output_str = turn2["value"]
            
            # 对于ShareGPT格式，我们假设是纯串行工作流。
            # 每个 "human/gpt" 对 都是一个单独的阶段 (Stage)，
            # 每个阶段只包含一个任务 (Task)。
            step_config = {
                "step": id + "_" + str(i // 2), 
                "input_str": input_str,
                "output_str": output_str,
            }
            workflow_steps_config.append(step_config)
            
            if not workflow_steps_config:
                continue

        request = UnifiedRequest(
            workflow_id = id,
            workflow_config=workflow_steps_config,
            arrive_at = arrive_at,
            max_token_for_request = self.max_tokens,
            # deadline 为绝对时间：到达时间 + 抽样得到的相对预算
            deadline = arrive_at + self._sample_deadline_budget(),
        )
        return request

    def generate_requests(self) -> List[UnifiedRequest]:
        requests = []
        for index, row in self.trace_df.iterrows():
            next_time = self.poisson_request_interval_generator.get_next_inter_request_time()
            self.time += next_time
            request = self.generate_unified_request(self.time, row)
            requests.append(request)

            if len(requests) >= 3000:
                break

        return requests



    # ----------- Test methods ------------
    def print_single_request(self, request: UnifiedRequest):
        print(f"arrive_at: {request.arrive_at}")

    def test_load(self):
        data_len = len(self.trace_df)
        print(f"Data length: {data_len}")

        # sample
        first_row = self.trace_df.iloc[0]
        print(first_row)

        first_conversation = first_row["conversations"]
        print(f"type: {type(first_conversation)}")      # List
        print(f"type: {type(first_conversation[0])}")   # Dict
        #print(f"content: {first_conversation}")

        self.generate_unified_request(self.time, first_row)
    

    def test_generate_requests(self):
        requests = self.generate_requests()
        print(len(requests))
    

