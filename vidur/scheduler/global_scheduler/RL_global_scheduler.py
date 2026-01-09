from datetime import datetime
from typing import List, Tuple, Dict

import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from vidur.config import SimulationConfig, RlConfig
from vidur.entities import Request, Replica, Batch
from vidur.entities.fake_request import FakeRequest
from vidur.entities.full_request import FullRequest
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

from vidur.scheduler.replica_scheduler.local_replica_scheduler import LocalReplicaScheduler
from vidur.scheduler.rl.algorithms.DQN import DQN
from vidur.scheduler.rl.algorithms.DoubleDQN import DoubleDQN
from vidur.scheduler.rl.algorithms.PPO import PPO
from vidur.scheduler.utils.content_manager import ContentManager
from vidur.scheduler.rl.buffer import ReplayBuffer
from math import ceil
from collections import defaultdict
from vidur.logger import init_logger
import pandas as pd

logger = init_logger(__name__)
CURRENT_TIME= datetime.now().strftime("%Y%m%d_%H%M%S")

def convert_to_np(state: Dict):
    """
        将包含嵌套列表、标量、NumPy数组或PyTorch张量的state字典
        扁平化并转换为一个维度为 (1, X) 的NumPy数组。

        Args:
            state (dict): 从 get_state() 方法获取的嵌套字典状态。

        Returns:
            np.ndarray: 扁平化后的 (1, X) 维度 NumPy 数组。
        """

    # 辅助函数：将任何可迭代或标量转换为扁平列表
    def flatten(data):
        flat_list = []
        for item in data:
            if isinstance(item, (list, tuple)):
                # 如果是嵌套列表，递归调用
                flat_list.extend(flatten(item))
            else:
                # 否则，添加标量
                flat_list.append(item)
        return flat_list

    # 1. 初始化最终的扁平列表
    final_flat_list = []

    # 2. 按照您代码中的结构顺序处理 state 的三个主要部分

    # 2.1. request_state (字典中的字典/列表)
    request_state = state.get("request_state", {})
    if isinstance(request_state, dict):
        # 遍历 request_state 字典中的所有值
        for key in sorted(request_state.keys()):  # 排序保证特征顺序一致性
            final_flat_list.extend(flatten(request_state[key]))
    else:
        # 如果 request_state 本身是列表或标量
        final_flat_list.extend(flatten(request_state))

    # 2.2. global_state (通常是列表或标量)
    global_state = state.get("global_state")
    final_flat_list.extend(flatten(global_state))

    # 2.3. local_state (嵌套列表/特征矩阵)
    # 这是一个包含多个 replica 状态的列表，每个状态本身是一个列表
    local_state = state.get("local_state", [])
    final_flat_list.extend(flatten(local_state))

    # 3. 转换为 (1, X) 的 NumPy 数组
    if not final_flat_list:
        raise ValueError("state is empty")

    return np.array(final_flat_list, dtype=np.float32).reshape(1, -1)


def create_req(request: FullRequest) -> FakeRequest:
    return FakeRequest(
        request.arrived_at,
        request.num_prefill_tokens,
        request.num_decode_tokens,
        request.num_processed_tokens,
    )

def plot_rl_metric(x_label: str, y_label: str, data: List, metric_type: str, output_dir: str):
    """
    使用 matplotlib 绘制强化学习训练指标
    metric_type: 'policy_loss', 'value_loss', 或 'returns'
    window_size: 平滑窗口大小，设为1则不平滑
    """

    output_dir = os.path.join(output_dir, CURRENT_TIME)

    iterations = np.arange(len(data))
    data = np.array(data)

    plt.figure(figsize=(10, 5))

    plt.plot(iterations, data, alpha=0.3, color='blue', label='Raw')

    # 画平滑数据（深色粗线）
    # plt.plot(smooth_iterations, smooth_data, color='blue', linewidth=2, label=f'Smoothed (win={window_size})')

    plt.title(f'Training {metric_type.replace("_", " ").title()}', fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{metric_type}_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 释放内存
    print(f"图表已保存至: {save_path}")

def plot_rl_metricv2(x_label: str, y_label: str, data: List, metric_type: str, output_dir: str):
    """
    优化后的 RL 指标绘图函数
    1. 增加移动平均平滑线
    2. 降低原始数据透明度，减少视觉干扰
    3. 自动调整 y 轴范围
    """
    if not data:
        return

    output_dir = os.path.join(output_dir, CURRENT_TIME)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 数据准备
    data_series = pd.Series(data)
    # 根据数据量自动调整窗口大小（例如总量的 5%）
    window_size = max(int(len(data) * 0.05), 5)
    smoothed_data = data_series.rolling(window=window_size, min_periods=1).mean()
    std_data = data_series.rolling(window=window_size, min_periods=1).std()

    iterations = np.arange(len(data))

    plt.figure(figsize=(12, 6))

    # 2. 绘制原始数据（极低透明度，作为背景阴影）
    plt.plot(iterations, data, alpha=0.15, color='#1f77b4', label='Raw Episode Return')

    # 3. 绘制平滑后的趋势线（深色，加粗）
    plt.plot(iterations, smoothed_data, color='#1f77b4', linewidth=2, label=f'Moving Average (win={window_size})')

    # 4. (可选) 绘制标准差阴影区域，展示波动的稳定性
    # plt.fill_between(iterations, smoothed_data - std_data, smoothed_data + std_data, color='#1f77b4', alpha=0.1)

    # 5. 美化图表
    plt.title(f'RL Training {metric_type.replace("_", " ").title()}', fontsize=15, fontweight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')

    # 如果是 Return，固定 y 轴在 0-1 之间（因为你做了归一化）
    if metric_type.lower() == 'return':
        plt.ylim(-0.05, 1.05)

    save_path = os.path.join(output_dir, f"{metric_type.lower()}_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

class RLGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        super().__init__(config, replicas)

        self.content_manager: ContentManager

        self.buffer = ReplayBuffer(config.rl_config.buffer_size)
        self.agent = self.init_agent(config.rl_config, len(replicas), self.buffer)
        self.state = None

        self.batch_size = config.rl_config.batch_size
        self.minimal_size = config.rl_config.minimal_size

        # plot training progress
        self.update_count = 0
        self.episode_count = 0
        self.returns = []
        self.policy_loss = []
        self.value_loss = []

        # global state
        self.virtual_pending_queue = [[] for i in range(self._num_replicas)] # 用于记录当前调度的任务的pending情况
        self.bin_width = config.cluster_config.replica_scheduler_config.bin_width
        self.max_tokens_per_request = config.cluster_config.replica_scheduler_config.max_tokens_per_request
        self.chunk_size = config.cluster_config.replica_scheduler_config.chunk_size

        if config.rl_config.load_model_path is not None:
            self.agent.load_model(config.rl_config.load_model_path)

    def init_agent(self, config: RlConfig, num_replicas: int, buffer: ReplayBuffer):
        if config.algorithm == 'dqn':
            return DQN(buffer, 168, hidden_dim=config.hidden_dim, action_dim=num_replicas,
                       gamma=config.gamma, learning_rate=config.learning_rate, target_update=config.target_update_freq, batch_size=config.batch_size,
                       minimal_size=config.minimal_size, epsilon=config.epsilon)
        elif config.algorithm == 'ppo':
            return PPO(buffer, 168, hidden_dim=config.hidden_dim, action_dim=num_replicas,
                       actor_lr=config.actor_lr, critic_lr=config.critic_lr, gamma=config.gamma, lmbda=config.lmbda,
                       batch_size=config.batch_size, epochs=config.epochs, eps=config.eps, ent_coef=config.ent_coef)
        elif config.algorithm == 'double_dqn':
            return DoubleDQN(buffer, 168, hidden_dim=config.hidden_dim, action_dim=num_replicas,
                       gamma=config.gamma, learning_rate=config.learning_rate, target_update=config.target_update_freq, batch_size=config.batch_size,
                       minimal_size=config.minimal_size, epsilon=config.epsilon)
        else:
            return None



    def reset(self):
        self.state = None

        self.update_count = 0
        self.episode_count = 0
        self.returns = []
        self.policy_loss = []
        self.value_loss = []

        self.virtual_pending_queue = [[] for i in range(self._num_replicas)]

    # 这里要考虑action 与 replica_id 的关系  允不允许不调度这个request，把它放后面调度？暂时认为都调度
    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()
        self.virtual_pending_queue = [[] for i in range(self._num_replicas)]  # reset

        request_mapping = []
        episode_return = 0
        num_request = 0
        while self._request_queue:
            self.state = self.get_state()
            np_state = convert_to_np(self.state)
            action = self.agent.take_action(np_state)

            request = self._request_queue.pop(0)
            next_state, reward, done = self.step(action, request)
            request_mapping.append((action, request))

            logger.debug(f"Schedule request {request.id} to replica {action}, reward: {reward}.")
            np_next_state = convert_to_np(next_state)
            self.buffer.add(np_state, action, reward, np_next_state, done, None, None)

            episode_return += reward
            num_request += 1
            loss = self.agent.train()

            if isinstance(self.agent, DQN) and loss is not None:
                self.policy_loss.append(loss)
                self.update_count += 1
            elif isinstance(self.agent, DoubleDQN) and loss is not None:
                self.policy_loss.append(loss)
                self.update_count += 1
            elif isinstance(self.agent, PPO) and loss[0] is not None:
                self.policy_loss.append(loss[0])
                self.value_loss.append(loss[1])
                self.update_count += 1

        self.episode_count += 1
        self.returns.append(episode_return / (num_request + 1e-10) )
        if len(self.returns) % 1000 == 0:
            plot_rl_metricv2('episodes', 'return', self.returns, 'Return', 'data/result')
        # TODO 100 改成一个参数
        if (self.update_count + 1) % 10 == 0:
            plot_rl_metricv2('update_times', 'policy_loss', self.policy_loss, 'policy_loss', 'data/result')
            plot_rl_metricv2('update_times', 'value_loss', self.value_loss, 'value_loss', 'data/result')
        if self.episode_count % self._config.rl_config.save_model_steps == 0:
            self.agent.save_model('data/result/' + CURRENT_TIME + '/model')
        return request_mapping

    def get_global_scheduler_state(self):
        global_scheduler_state = []

        tokens_to_process = []
        for request in self._request_queue:
            tokens_to_process.append(request.num_prefill_tokens + request.num_decode_tokens)
        tokens_distribution = self.cal_token_distribution(tokens_to_process)
        global_scheduler_state.extend(tokens_distribution)
        return global_scheduler_state

    def get_state(self):
        '''
            state包括：
            1. request的state:
                a. request在agent中的位次，即agent从图的角度看执行进度
                b. request的prefill和decode的分布
            2. global scheduler 的 pending queue的任务概况，最好是用attention算位置信息
            3. 所有local scheduler的state
        '''
        request = self._request_queue[0]

        state = {}
        # request-level state (include agent-level)
        request_state = request.get_state()
        state["request_state"] = request_state

        # global-scheduler-level state
        global_state = self.get_global_scheduler_state()
        state["global_state"] = global_state

        # local-scheduler-level state
        num_replica = len(self._replica_schedulers)
        local_state = []
        for id, replica_scheduler in self._replica_schedulers.items():
            assert isinstance(replica_scheduler, LocalReplicaScheduler), "scheduler type error."

            one_hot_id = [0 for i in range(num_replica)]
            one_hot_id[id] = 1

            replica_state = replica_scheduler.get_replica_scheduler_state(request)
            replica_state = one_hot_id + replica_state

            local_state.append(replica_state)

        state["local_state"] = local_state
        return state

    def get_reward(self, action, request):
        rewards = []

        for i in range(self._num_replicas):
            rewards.append(self.cal_reward(i, request))

        return (rewards[action] - min(rewards)) / (max(rewards) - min(rewards) + 1e-8) # * request.relative_position

    def cal_reward(self, action, request: FullRequest):
        # 奖励怎么考虑呢？
        # 我觉得最重要的一点就是要刻画 被选择的这个local_engine相比于其他engine的收益
        # -- a. request到这个engine他的kv复用的长度相比其他engine的kv复用长度 [归一化后 - 平均值]
        replica_scheduler = self._replica_schedulers[action]
        assert isinstance(replica_scheduler, LocalReplicaScheduler), "scheduler type error."
        reuse_kv, _ = replica_scheduler.tree_cache.match_prefix(request.input_token_ids)
        reward_of_kv = len(reuse_kv) / request.num_prefill_tokens

        # evict block's penalty
        num_required_blocks = (request.num_prefill_tokens - len(
            reuse_kv)) // self._config.cluster_config.replica_scheduler_config.block_size
        num_blocks_left = replica_scheduler.block_manager.num_total_blocks - replica_scheduler.block_manager.num_used_blocks \
                          - num_required_blocks
        num_blocks_evict = max(0, replica_scheduler._watermark_blocks - num_blocks_left)
        reward_of_evict = num_blocks_evict

        # 用执行的速度来代替request的请求，越大越好
        # --a。 如果当前batch还未满，能够容纳request的全部prefill token，则用batch的处理时间来作为reward  TPOT
        # --b。 如果当前batch已满，计算排队时间作为reward  Waiting time

        # 直接参与batch
        replica_scheduler = self._replica_schedulers[action]

        # 计算current batch，要么已经组成batch，要么在preempted_queue中
        replica_stage_scheduler = replica_scheduler.get_replica_stage_scheduler(0)  # assume no parallelism
        current_batch = replica_stage_scheduler.get_current_batch()
        num_tokens = current_batch.num_tokens if current_batch is not None else []
        requests = current_batch.requests if current_batch is not None else []

        if hasattr(replica_scheduler, "_preempted_requests") and len(requests) == 0:
            requests = replica_scheduler._preempted_requests
            num_tokens = [req.num_prefill_tokens - req.num_generated_tokens for req in requests if
                          req.is_prefill_completed is False]

        prefill_token = min(request.num_prefill_tokens - len(reuse_kv), self.chunk_size - sum(num_tokens))
        assert 0 <= prefill_token <= request.num_prefill_tokens

        # 如果要排队，当前request在local_replica_scheduler里已经加入virtual_pending_queue了
        pending_queue = self.get_replica_scheduler(action).get_pending_requests() + self.virtual_pending_queue[action][:-1]

        reward_of_time = 0.0
        if prefill_token > 0 and len(pending_queue) == 0:  # request可以立刻执行
            fake_reqs = [create_req(req) for req in requests] + [create_req(request)]
            for fake_req, req in zip(fake_reqs, requests):
                fake_req._is_prefill_complete = req.is_prefill_complete
            fake_num_tokens = num_tokens + [prefill_token]
            fake_batch = Batch(action, fake_reqs, fake_num_tokens)
            predict_time = replica_stage_scheduler.execution_time_predictor.get_execution_time(
                fake_batch,
                replica_stage_scheduler.stage_id,
            )
            reward_of_time -= predict_time.total_time
        else:
            num_pending_tokens = sum([req.num_prefill_tokens for req in pending_queue])
            num_predict_batches = ceil(num_pending_tokens / self._config.cluster_config.replica_scheduler_config.chunk_size)

            current_batch = Batch(action, requests, num_tokens)
            predict_time = replica_stage_scheduler.execution_time_predictor.get_execution_time(
                current_batch,
                replica_stage_scheduler.stage_id,
            )
            reward_of_time -= predict_time.total_time * num_predict_batches
        return 5 * reward_of_time
        if true:
            pass
        else:
            # 排队时间
            # 简单用 waiting queue的所有的request的prefill token，去算需要多少个chunk，轮到当前request
            fake_reqs = [create_req(req) for req in requests] + [create_req(request)]
            for fake_req, req in zip(fake_reqs, requests):
                fake_req._is_prefill_complete = req.is_prefill_complete
            fake_num_tokens = num_tokens

            waiting_time = 0.0
            pending_requests = [create_req(req) for req in replica_scheduler._request_queue]

            while len(pending_requests) > 0:

                if self.chunk_size - sum(fake_num_tokens) > 0:
                    req = pending_requests.pop(0)
                    prefill_token = min(req.num_prefill_tokens,
                                        self.chunk_size - sum(num_tokens))

                    fake_reqs.append(req)
                    fake_num_tokens.append(prefill_token)

                fake_batch = Batch(action, fake_reqs, fake_num_tokens)

                predict_time = replica_stage_scheduler.execution_time_predictor.get_execution_time(
                    fake_batch,
                    replica_stage_scheduler.stage_id,
                )

                waiting_time += predict_time.total_time

                for req, token_len in zip(fake_reqs, fake_num_tokens):
                    if req.is_prefill_complete:
                        continue

                    req._num_prefill_tokens -= token_len

                    if req.num_prefill_tokens <= 0:
                        fake_reqs.remove(req)
                        fake_num_tokens.remove(token_len)

            reward_of_time -= waiting_time

        logger.debug(
            f"reward_of_kv: {reward_of_kv}, reward_of_evict: {reward_of_evict}, reward_of_time: {reward_of_time}")

        return 0.1 * reward_of_kv + 10 * reward_of_time + 0.0001 * reward_of_evict


    def step(self, action, request: FullRequest):
        target_scheduler = self.get_replica_scheduler(action)
        assert isinstance(target_scheduler, LocalReplicaScheduler), "scheduler type error."

        next_state = self.state

        num_replica = len(self._replica_schedulers)
        one_hot_id = [0 for i in range(num_replica)]
        one_hot_id[action] = 1

        next_state["local_state"][action] = one_hot_id + target_scheduler.step(request, self.virtual_pending_queue[action])

        next_request = self._request_queue[0] if len(self._request_queue) > 0 else None
        if next_request is not None:
            next_state["request_state"] = self._request_queue[0].get_state()
            done = False
        else:
            next_state["request_state"] = [0 for _ in range(len(self.state["request_state"]))]
            done = True

        # reward = self.cal_reward(action, request)
        reward = self.get_reward(action, request)

        return next_state, reward, done

    def cal_token_distribution(self, num_tokens: List[int]) -> List[int]:
        num_bins = ceil(self.max_tokens_per_request / self.bin_width)
        num_requests = len(num_tokens)

        # 使用 defaultdict 初始化分布，这样我们只需要处理有计数的 bin，
        # 后续再用一个循环确保所有 bin 都存在。
        token_distribution_counts = defaultdict(int)

        for remaining_tokens in num_tokens:
            # 计算 bin 的起始值，例如 150/100 * 100 = 100
            # remaining_tokens 已经被限制在 [1, max_token_per_request] 范围内
            bin_start = (remaining_tokens - 1) // self.bin_width * self.bin_width  # 确保剩余 1 token 落在 bin 0

            # 统计计数
            token_distribution_counts[bin_start] += 1

        # 5. 生成固定长度且有序的最终分布列表
        token_distribution = []
        for i in range(num_bins):
            bin_start = i * self.bin_width

            # 如果 bin_start 已经超过了 max_token_per_request，则停止
            if bin_start >= self.max_tokens_per_request:
                break

            # 获取计数，如果该 bin 没有请求，则计数值为 0
            count = token_distribution_counts[bin_start] / num_requests if num_requests > 0 else 0
            token_distribution.append(count)


        return token_distribution