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
    ):
        super().__init__()
        self.use_recurrence = use_recurrence
        self.recurrent_dim = recurrent_dim
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden_dim, recurrent_dim)
        actor_in = recurrent_dim if use_recurrence else hidden_dim
        self.mean = layer_init(nn.Linear(actor_in, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

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
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std), next_hidden


class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)
