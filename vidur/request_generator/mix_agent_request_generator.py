import logging
import json
import random
from typing import List, Dict

import pandas as pd

from vidur.config import MixAgentRequestGeneratorConfig
from vidur.entities.unified_request import UnifiedRequest
from vidur.request_generator.base_request_generator import BaseRequestGenerator
from vidur.request_generator.poisson_request_interval_generator import PoissonRequestIntervalGenerator

logger = logging.getLogger(__name__)


class MixAgentRequestGenerator(BaseRequestGenerator):
    """
    从 code、sharegpt、mapreduce 三个数据集中按比例采样，组成异构混合 trace。

    参数:
        N (num_workflows)   — 总共生成 N 个 workflow
        code_ratio          — code 占 N 的比例
        sharegpt_ratio      — sharegpt 占 N 的比例
        mapreduce_ratio     — 1 - code_ratio - sharegpt_ratio
    """

    def __init__(self, config: MixAgentRequestGeneratorConfig):
        # ---- 加载三个数据集 ----
        # sharegpt: JSON 数组, 用 pd 读取
        self.sharegpt_df = pd.read_json(config.sharegpt_trace_file)

        # mapreduce: JSON 数组
        with open(config.map_reduce_trace_file, 'r', encoding='utf-8') as f:
            self.mapreduce_data = json.load(f)

        # code: JSON 数组
        with open(config.code_trace_file, 'r', encoding='utf-8') as f:
            self.code_data = json.load(f)

        self.poisson_request_interval_generator = PoissonRequestIntervalGenerator(
            config.interval_generator_config
        )
        self.time = config.start_time
        self.max_tokens = config.max_tokens

        self.num_workflows = config.num_workflows
        self.code_ratio = config.code_ratio
        self.sharegpt_ratio = config.sharegpt_ratio
        self.mapreduce_ratio = round(1.0 - config.code_ratio - config.sharegpt_ratio, 6)

        if self.mapreduce_ratio < 0:
            raise ValueError(
                f"code_ratio ({config.code_ratio}) + sharegpt_ratio ({config.sharegpt_ratio}) > 1.0"
            )

        # 各 agent 类型的 e2e runtime 分位点（秒），可根据实测数据替换
        # 格式: agent_type -> (p50, p70, p90)
        self._slo_percentiles: Dict[str, tuple] = {
            "code":      (10.0, 20.0,  40.0),
            "chat":      ( 5.0, 10.0,  20.0),
            "mapreduce": ( 5.0,  8.0,  12.0),
        }

        seed = getattr(config, "seed", None)
        self._deadline_rng = random.Random(seed if seed is not None else 42)

    # ---------- 从各数据集生成单个 UnifiedRequest ----------

    def _generate_sharegpt_request(self, row) -> UnifiedRequest:
        """复用 ShareGPTRequestGenerator 的逻辑。"""
        wid = str(row["id"])
        conversations = row["conversations"]
        steps: List[Dict] = []

        for i in range(0, len(conversations), 2):
            if i + 1 >= len(conversations):
                continue
            steps.append({
                "step": wid + "_" + str(i // 2),
                "input_str": conversations[i]["value"],
                "output_str": conversations[i + 1]["value"],
            })

        if not steps:
            return None

        return UnifiedRequest(
            workflow_id=wid,
            arrive_at=0,  # 后续统一赋值
            workflow_config=steps,
            max_token_for_request=self.max_tokens,
            agent_type="chat",
        )

    def _generate_mapreduce_request(self, workflow_data: Dict) -> UnifiedRequest:
        """复用 MapReduceRequestGenerator 的逻辑。"""
        wid = workflow_data["agent_id"]
        steps: List[Dict] = []
        step_count = 0
        former = workflow_data["summary"][0]["node_name"] if workflow_data["summary"] else None

        for node in workflow_data["summary"]:
            if node["node_name"] != former:
                step_count += 1
            steps.append({
                "step": wid + "_" + str(step_count),
                "input_str": node["inputs"],
                "output_str": node["outputs"],
            })
            former = node["node_name"]

        if not steps:
            return None

        return UnifiedRequest(
            workflow_id=wid,
            arrive_at=0,
            workflow_config=steps,
            max_token_for_request=self.max_tokens,
            agent_type="mapreduce",
        )

    def _generate_code_request(self, app_data: Dict) -> UnifiedRequest:
        """复用 CodeAgentRequestGenerator 的逻辑。"""
        wid = str(app_data.get("id", "unknown"))
        conversations = app_data.get("conversations", [])
        steps: List[Dict] = []
        step_count = 0
        last_stage_name = ""

        for conv in conversations:
            role = conv.get("from")
            steps.append({
                "step": wid + "_" + str(step_count),
                "input_str": conv.get("input", ""),
                "output_str": conv.get("output", ""),
            })
            if last_stage_name != "worker" or role != "worker":
                step_count += 1
            last_stage_name = role

        if not steps:
            return None

        if len(steps) >= 50:
            steps = steps[:50]

        return UnifiedRequest(
            workflow_id=wid,
            arrive_at=0,
            workflow_config=steps,
            max_token_for_request=self.max_tokens,
            agent_type="code",
        )

    # ---------- SLO / deadline ----------

    def _sample_deadline_budget(self, agent_type: str) -> float:
        """
        根据 agent 类型对应的 runtime 分位点生成相对 deadline（秒）。

        分段概率: [0.6, 0.3, 0.1] -> [P50,P70], [P70,P90], [P90, 1.5×P90]
        安全系数: 乘以 U(1.05, 1.20)
        """
        p50, p70, p90 = self._slo_percentiles.get(
            agent_type, self._slo_percentiles["chat"]
        )
        rng = self._deadline_rng
        p = rng.random()
        if p < 0.6:
            base = rng.uniform(p50, p70)
        elif p < 0.9:
            base = rng.uniform(p70, p90)
        else:
            base = rng.uniform(p90, 1.5 * p90)

        return base * rng.uniform(1.05, 1.20)

    # ---------- 采样辅助 ----------

    @staticmethod
    def _sample_indices(pool_size: int, n: int, rng: random.Random) -> List[int]:
        """从 pool_size 条数据中采样 n 个索引（允许循环取用）。"""
        if pool_size == 0:
            return []
        if n <= pool_size:
            return rng.sample(range(pool_size), n)
        # pool 不够时循环取用
        indices = list(range(pool_size)) * (n // pool_size)
        indices += rng.sample(range(pool_size), n % pool_size)
        rng.shuffle(indices)
        return indices

    # ---------- 主入口 ----------

    def generate_requests(self) -> List[UnifiedRequest]:
        N = self.num_workflows
        n_code = round(N * self.code_ratio)
        n_share = round(N * self.sharegpt_ratio)
        n_map = N - n_code - n_share

        rng = random.Random(42)

        requests: List[UnifiedRequest] = []

        # ---- code ----
        code_indices = self._sample_indices(len(self.code_data), n_code, rng)
        for idx in code_indices:
            req = self._generate_code_request(self.code_data[idx])
            if req:
                requests.append(req)

        # ---- sharegpt ----
        share_indices = self._sample_indices(len(self.sharegpt_df), n_share, rng)
        for idx in share_indices:
            req = self._generate_sharegpt_request(self.sharegpt_df.iloc[idx])
            if req:
                requests.append(req)

        # ---- mapreduce ----
        map_indices = self._sample_indices(len(self.mapreduce_data), n_map, rng)
        for idx in map_indices:
            req = self._generate_mapreduce_request(self.mapreduce_data[idx])
            if req:
                requests.append(req)

        # ---- shuffle & 分配到达时间 ----
        rng.shuffle(requests)

        for req in requests:
            inter_request_time = self.poisson_request_interval_generator.get_next_inter_request_time()
            self.time += inter_request_time
            req.arrive_at = self.time
            req.deadline = self.time + self._sample_deadline_budget(req.agent_type)

        logger.info(
            f"MixAgentRequestGenerator: generated {len(requests)} requests "
            f"(code={n_code}, sharegpt={n_share}, mapreduce={n_map})"
        )
        return requests
