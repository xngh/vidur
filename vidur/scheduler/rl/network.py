import torch
import random
import numpy as np
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

ZERO_PADDING = 0

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
        self.fc1 = torch.nn.Linear(in_features=feature_dim, out_features=hidden_dim)
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = torch.nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CrossAttentionQNet(torch.nn.Module):
    def __init__(self, req_dim: int, engine_dim: int, action_dim, hidden_dim=256):
        super(CrossAttentionQNet, self).__init__()

        self.req_feature_dim = req_dim
        self.engine_feature_dim = engine_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim  # 嵌入维度

        self.req_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.req_feature_dim, self.hidden_dim),
            torch.nn.ReLU()
        )

        self.engine_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.engine_feature_dim, self.hidden_dim),
            torch.nn.ReLU()
        )

        # Multi-Head Cross Attention
        # 用 Request 作为 Query，Engine 状态作为 Key 和 Value
        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=action_dim, batch_first=True)

        # Q-Value 输出层
        # 输入：原始特征 + Attention 后的增强特征
        state_dim = self.req_feature_dim + self.engine_feature_dim * self.action_dim
        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(state_dim + self.hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, state_dim)
        batch_size = x.size(0)

        # 提取 Request 特征
        req_feat = x[:, :self.req_feature_dim]  # (batch, req_dim)

        # 提取 Engine 特征并重塑为序列形式 (batch, num_engines, engine_dim)
        engine_feats = x[:, self.req_feature_dim:].view(batch_size, self.action_dim, -1)

        # Query: (batch, 1, d_model)
        query = self.req_encoder(req_feat).unsqueeze(1)
        # Key & Value: (batch, num_engines, d_model)
        key_value = self.engine_encoder(engine_feats)

        # --- Cross Attention ---
        # attn_output: (batch, 1, d_model) 代表 Request 对所有 Engine 的加权感知
        attn_output, attn_weights = self.cross_attn(query, key_value, key_value)
        attn_output = attn_output.squeeze(1)  # (batch, d_model)

        # 拼接全局原始特征和局部注意力特征
        out_input = torch.cat([x, attn_output], dim=1)
        q_values = self.fc_out(out_input)

        return q_values