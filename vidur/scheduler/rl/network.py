import torch
import random
import numpy as np
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

ZERO_PADDING = 0

class PolicyNet(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=feature_dim, out_features=hidden_dim)
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = torch.nn.Linear(in_features=hidden_dim, out_features=action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=512, out_features=hidden_dim)
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = torch.nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)