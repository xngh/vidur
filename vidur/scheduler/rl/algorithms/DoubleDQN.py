import os

import numpy as np

from vidur.logger import init_logger
from vidur.scheduler.rl.network import PolicyNet, Qnet
import torch
import torch.optim as optim
import torch.nn.functional as F

logger = init_logger(__name__)

class DoubleDQN:
    def __init__(self,
                 buffer,
                 feature_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate = 1e-3,
                 gamma = 0.98,
                 epsilon = 0.01,
                 target_update = 10,
                 batch_size = 64,
                 minimal_size = 500,
                 device = "cuda"):
        self.action_dim = action_dim
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.q_net = Qnet(feature_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(feature_dim, hidden_dim, action_dim).to(device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.998
        self.tau = 0.01
        self.target_update = target_update
        self.count = 0
        self.batch_size = batch_size
        self.minimal_size = minimal_size

        self.buffer = buffer


    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            state = torch.from_numpy(state).float().to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def soft_update(self):
        """
        软更新：target_params = tau * local_params + (1 - tau) * target_params
        """
        for target_param, local_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states']).float().squeeze(1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states']).float().squeeze(1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        #logger.debug(f"states size: {states.size()}")
        #logger.debug(f"actions size: {actions.size()}")
        #logger.debug(f"network output size: {self.q_net(states).size()}")
        q_values = self.q_net(states).gather(1, actions)

        # 1. 用当前的 q_net 选出下一个状态下最好的动作下标
        best_actions = self.q_net(next_states).argmax(dim=1).view(-1, 1)
        # 2. 用 target_q_net 评估这个动作的 Q 值
        max_next_q_values = self.target_q_net(next_states).gather(1, best_actions)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # loss = torch.mean(F.mse_loss(q_values, q_targets))
        loss = F.smooth_l1_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        logger.info("DQN Policy loss is %.3f" % loss.item())
        return loss.item()
        #print(f"q_targets: {q_targets}")
        #print(f"q_values: {q_values}")

    def train(self):
        if self.buffer.size() > self.minimal_size:
            b_s, b_a, b_r, b_ns, b_d, _, _ = self.buffer.sample(self.batch_size)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d
            }

            loss = self.update(transition_dict)
            return loss
        else:
            return None

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, 'model.pth')
        torch.save({
            'q_state_dict': self.q_net.state_dict(),
            'target_q_state_dict': self.target_q_net.state_dict()
        }, file_path)

    def load_model(self, path):
        file_path = os.path.join(path, 'model.pth')
        checkpoint = torch.load(file_path)
        self.q_net.load_state_dict(checkpoint['q_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_state_dict'])