import logging
import json
import random
from typing import List, Dict

from vidur.config import CodeAgentRequestGeneratorConfig
from vidur.entities.unified_request import UnifiedRequest
from vidur.request_generator.base_request_generator import BaseRequestGenerator
from vidur.request_generator.poisson_request_interval_generator import PoissonRequestIntervalGenerator

logger = logging.getLogger(__name__)


class CodeAgentRequestGenerator(BaseRequestGenerator):
    def __init__(self, config: CodeAgentRequestGeneratorConfig):
        with open(config.trace_file, 'r', encoding='utf-8') as f:
            self.trace_data = json.load(f)

        self.poisson_request_interval_generator = PoissonRequestIntervalGenerator(
            config.interval_generator_config
        )
        self.time = config.start_time
        self.max_tokens = config.max_tokens

        # code agent workflow e2e runtime 分位点（秒），可根据实测数据替换
        self._p50_runtime = 20.0
        self._p70_runtime = 30.0
        self._p90_runtime = 60.0

        seed = getattr(config, "seed", None)
        if seed is None:
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

    def generate_unified_request(self, arrive_at, app_data: Dict) -> UnifiedRequest:
        # 2. 从 trace 数据中提取 ID
        workflow_id = str(app_data.get("id", "unknown"))

        workflow_steps_config = []
        conversations = app_data.get("conversations", [])

        # 3. 遍历 conversations 生成 steps
        # 这里假设列表中的每一个元素都是 workflow 的一个顺序步骤
        step_count = 0
        last_stage_name = ""
        for _, conv in enumerate(conversations):
            role = conv.get("from")
            step_config = {
                # 生成唯一的步骤名，例如 "45_0", "45_1"
                # 不同的worker可以并行，因此step_count可以是一样的
                "step": workflow_id + "_" + str(step_count),
                "input_str": conv.get("input", ""),
                "output_str": conv.get("output", ""),
            }
            if last_stage_name != "worker" or role != "worker":
                step_count += 1
            last_stage_name = role
            workflow_steps_config.append(step_config)

        if not workflow_steps_config:
            return None

        if len(workflow_steps_config) >= 50:
            workflow_steps_config = workflow_steps_config[:50]

        # 4. 实例化 UnifiedRequest
        return UnifiedRequest(
            workflow_id=workflow_id,
            arrive_at=arrive_at,
            workflow_config=workflow_steps_config,
            max_token_for_request=self.max_tokens,
            agent_type="code",
            deadline=arrive_at + self._sample_deadline_budget(),
        )

    def generate_requests(self) -> List[UnifiedRequest]:
        requests = []

        for app_item in self.trace_data:
            # 获取下一个 Poisson 到达间隔并累加时间
            inter_request_time = self.poisson_request_interval_generator.get_next_inter_request_time()
            self.time += inter_request_time

            request = self.generate_unified_request(self.time, app_item)
            if request:
                requests.append(request)

            # 保持与参考代码一致的硬编码限制，或者可以移入 config
            if len(requests) >= 200:
                break

        return requests