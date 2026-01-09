import os

import numpy as np
import torch
import torch.nn.functional as F

from vidur.logger import init_logger
from vidur.scheduler.rl.network import PolicyNet, ValueNet

logger = init_logger(__name__)

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class PPO:
    def __init__(self,
                 buffer,
                 feature_dim,
                 hidden_dim,
                 action_dim,
                 actor_lr,
                 critic_lr,
                 gamma=0.99,
                 lmbda=0.95,
                 epochs=5,
                 eps=0.2,
                 ent_coef=0.01,
                 batch_size=256,
                 device = 'cuda'):
        self.action_dim = action_dim
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.buffer = buffer

        self.actor = PolicyNet(feature_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(feature_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.ent_coef = ent_coef
        self.batch_size = batch_size
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    #def update(self, transition_dict):
    #    states = torch.tensor(transition_dict['states'],
    #                          dtype=torch.float).squeeze(1).to(self.device)
    #    actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
    #        self.device)
    #    rewards = torch.tensor(transition_dict['rewards'],
    #                           dtype=torch.float).view(-1, 1).to(self.device)
    #    next_states = torch.tensor(transition_dict['next_states'],
    #                               dtype=torch.float).squeeze(1).to(self.device)
    #    dones = torch.tensor(transition_dict['dones'],
    #                         dtype=torch.float).view(-1, 1).to(self.device)
    #    td_target = rewards + self.gamma * self.critic(next_states) * (1 -
    #                                                                   dones)
    #    td_delta = td_target - self.critic(states)
    #    advantage = compute_advantage(self.gamma, self.lmbda,
    #                                    td_delta.cpu()).to(self.device)
    #    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    #    old_log_probs = torch.log(self.actor(states).gather(1,
    #                                                        actions)).detach()
#
    #    last_actor_loss = 0.0
    #    last_critic_loss = 0.0
    #    for _ in range(self.epochs):
    #        log_probs = torch.log(self.actor(states).gather(1, actions))
    #        ratio = torch.exp(log_probs - old_log_probs)
    #        surr1 = ratio * advantage
    #        surr2 = torch.clamp(ratio, 1 - self.eps,
    #                            1 + self.eps) * advantage  # 截断
    #        actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
    #        critic_loss = torch.mean(
    #            F.mse_loss(self.critic(states), td_target.detach()))
#
    #        self.actor_optimizer.zero_grad()
    #        self.critic_optimizer.zero_grad()
    #        actor_loss.backward()
    #        critic_loss.backward()
#
    #        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
    #        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
#
    #        last_actor_loss = actor_loss.item()
    #        last_critic_loss = critic_loss.item()
    #        self.actor_optimizer.step()
    #        self.critic_optimizer.step()
#
    #    return last_actor_loss, last_critic_loss

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).squeeze(1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).squeeze(1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 1. 计算 TD Target 和 Advantage (在旧策略上计算一次)
        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            old_log_probs = torch.log(self.actor(states).gather(1, actions))

        total_actor_loss = 0
        total_critic_loss = 0

        # 2. 开始 Epochs 迭代
        for _ in range(self.epochs):
            # 3. 核心改进：将全量数据打乱，分批次 (Minibatch) 训练
            indices = np.arange(states.size(0))
            np.random.shuffle(indices)

            for start in range(0, states.size(0), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                # 提取 Batch 数据
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_advantage = advantage[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_td_target = td_target[idx]

                # 计算当前策略
                log_probs = torch.log(self.actor(mb_states).gather(1, mb_actions)).view(-1, 1)
                ratio = torch.exp(log_probs - mb_old_log_probs)

                # PPO 损失计算
                surr1 = ratio * mb_advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_advantage

                dist = torch.distributions.Categorical(self.actor(mb_states))
                entropy = dist.entropy().mean()

                actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.ent_coef * entropy

                # Critic 损失
                critic_loss = torch.mean(F.mse_loss(self.critic(mb_states), mb_td_target))

                # 更新网络
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # 核心改进：梯度裁剪，防止模型跑飞
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        # 返回平均 Loss 用于绘图
        num_updates = self.epochs * (states.size(0) // self.batch_size + 1)
        return total_actor_loss / num_updates, total_critic_loss / num_updates

    def train(self):
        if self.buffer.size() < self.batch_size:
            return None, None

        b_s, b_a, b_r, b_ns, b_d, _, _ = self.buffer.sample(self.buffer.size())

        transition_dict = {
            'states': b_s,
            'actions': b_a,
            'next_states': b_ns,
            'rewards': b_r,
            'dones': b_d
        }

        actor_loss, critic_loss = self.update(transition_dict)
        self.buffer.clear()

        return actor_loss, critic_loss

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, 'model.pth')
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, file_path)

    def load_model(self, path):
        file_path = os.path.join(path, 'model.pth')
        checkpoint = torch.load(file_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])