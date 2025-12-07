import logging
from typing import List, Dict
from urllib import request

import pandas as pd

from vidur.config import ShareGPTRequestGeneratorConfig
from vidur.entities import Request
from vidur.entities.unified_request import UnifiedRequest
from vidur.entities.full_request import FullRequest
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
            arrive_at = arrive_at
        )
        return request

    def generate_requests(self) -> List[UnifiedRequest]:
        requests = []
        for index, row in self.trace_df.iterrows():
            next_time = self.poisson_request_interval_generator.get_next_inter_request_time()
            self.time += next_time
            request = self.generate_unified_request(self.time, row)
            requests.append(request)
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
    

