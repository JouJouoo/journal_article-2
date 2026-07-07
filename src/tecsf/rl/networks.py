from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class RecurrentGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        recurrent_dim: int,
        use_recurrence: bool = True,
        log_std_init: float = -3.0,
        log_std_min: float = -4.0,
        log_std_max: float = -2.5,
    ):
        super().__init__()
        self.use_recurrence = use_recurrence
        self.recurrent_dim = recurrent_dim
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden_dim, recurrent_dim)
        actor_in = recurrent_dim if use_recurrence else hidden_dim
        self.mean = layer_init(nn.Linear(actor_in, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def initial_hidden(self, num_agents: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(num_agents, self.recurrent_dim, device=device)

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[Normal, torch.Tensor]:
        encoded = self.encoder(obs)
        if self.use_recurrence:
            next_hidden = self.gru(encoded, hidden)
            features = next_hidden
        else:
            next_hidden = hidden
            features = encoded
        mean = torch.tanh(self.mean(features))
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std), next_hidden


class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim: int, hidden_dim: int, value_heads: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, value_heads), std=1.0),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)


class CreditAssignmentNetwork(nn.Module):
    def __init__(self, agent_feature_dim: int, hidden_dim: int):
        super().__init__()
        pair_dim = 2 * agent_feature_dim + 1
        self.net = nn.Sequential(
            layer_init(nn.Linear(pair_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=0.01),
        )

    def forward(
        self,
        agent_features: torch.Tensor,
        trade_relation: torch.Tensor,
    ) -> torch.Tensor:
        t_steps, n_agents, feature_dim = agent_features.shape
        source = agent_features.unsqueeze(2).expand(t_steps, n_agents, n_agents, feature_dim)
        target = agent_features.unsqueeze(1).expand(t_steps, n_agents, n_agents, feature_dim)
        edge = trade_relation.unsqueeze(-1)
        pair_features = torch.cat([source, target, edge], dim=-1)
        scores = self.net(pair_features).squeeze(-1)
        return torch.softmax(scores, dim=-1)


class AdvantageGate(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(1, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 2), std=0.01),
        )

    def forward(self, asset_state: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(asset_state.unsqueeze(-1)), dim=-1)


class ClipGate(nn.Module):
    def __init__(self, hidden_dim: int, clip_min: float, clip_max: float):
        super().__init__()
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.net = nn.Sequential(
            layer_init(nn.Linear(1, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=0.01),
        )

    def forward(self, asset_state: torch.Tensor) -> torch.Tensor:
        span = max(self.clip_max - self.clip_min, 0.0)
        value = self.clip_min + span * torch.sigmoid(
            self.net(asset_state.unsqueeze(-1)).squeeze(-1)
        )
        return value
