import numpy as np

from vidur.logger import init_logger
from vidur.scheduler.rl.network import PolicyNet
import torch
import torch.optim as optim
import torch.nn.functional as F

logger = init_logger(__name__)

class DQN:
    def __init__(self, feature_dim, hidden_dim, action_dim,
                 learning_rate = 1e-3, gamma = 0.98, epsilon = 0.01, target_update = 10, device = "cuda"):
        self.action_dim = action_dim
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.q_net = PolicyNet(feature_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = PolicyNet(feature_dim, hidden_dim, action_dim).to(device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

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
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

        logger.info("DQN Policy loss is %.3f" % loss.item())
        return loss.item()
        #print(f"q_targets: {q_targets}")
        #print(f"q_values: {q_values}")