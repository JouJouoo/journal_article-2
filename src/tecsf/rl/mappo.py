from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tecsf.config import ExperimentConfig, load_config
from tecsf.device import resolve_device
from tecsf.env import EnergyCarbonEnv
from tecsf.metrics import summarize_episode, write_json
from tecsf.rl.buffer import RolloutBuffer
from tecsf.rl.networks import CentralizedCritic, RecurrentGaussianActor
from tecsf.variants import get_variant


@dataclass
class TrainingResult:
    variant: str
    output_dir: str
    checkpoint_path: str
    metrics_path: str
    episode_metrics: list[dict[str, float]]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for step in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[step] = last_gae
        next_value = values[step]
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def _collect_episode(
    env: EnergyCarbonEnv,
    actor: RecurrentGaussianActor,
    critic: CentralizedCritic,
    device: torch.device,
    use_heuristic_policy: bool = False,
) -> RolloutBuffer:
    buffer = RolloutBuffer()
    obs, global_state, _ = env.reset()
    hidden = actor.initial_hidden(env.num_agents, device)
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        state_t = torch.as_tensor(global_state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            hidden_before = hidden.clone()
            if use_heuristic_policy:
                action_t = torch.as_tensor(
                    env.deterministic_action(), dtype=torch.float32, device=device
                )
                next_hidden = hidden
                log_prob_t = torch.zeros(env.num_agents, dtype=torch.float32, device=device)
            else:
                dist, next_hidden = actor(obs_t, hidden)
                action_t = torch.clamp(dist.sample(), -1.0, 1.0)
                log_prob_t = dist.log_prob(action_t).sum(dim=-1)
            value_t = critic(state_t).squeeze(0)
        result = env.step(action_t.cpu().numpy())
        buffer.add(
            observation=obs,
            global_state=global_state,
            hidden_state=hidden_before.cpu().numpy(),
            action=action_t.cpu().numpy(),
            log_prob=log_prob_t.cpu().numpy(),
            reward=result.reward,
            done=result.terminated or result.truncated,
            value=float(value_t.cpu().item()),
            info=result.info,
        )
        obs = result.observation
        global_state = result.global_state
        hidden = next_hidden.detach()
        done = result.terminated or result.truncated
    return buffer


def _update_policy(
    config: ExperimentConfig,
    actor: RecurrentGaussianActor,
    critic: CentralizedCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    rollout: RolloutBuffer,
    device: torch.device,
) -> dict[str, float]:
    data = rollout.arrays()
    advantages, returns = _compute_gae(
        data["rewards"],
        data["dones"],
        data["values"],
        config.rl.gamma,
        config.rl.gae_lambda,
    )
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    t_steps, n_agents, obs_dim = data["observations"].shape
    action_dim = data["actions"].shape[-1]

    obs = torch.as_tensor(data["observations"].reshape(t_steps * n_agents, obs_dim), device=device)
    hidden = torch.as_tensor(
        data["hidden_states"].reshape(t_steps * n_agents, -1), device=device
    )
    actions = torch.as_tensor(
        data["actions"].reshape(t_steps * n_agents, action_dim), device=device
    )
    old_log_probs = torch.as_tensor(
        data["log_probs"].reshape(t_steps * n_agents), device=device
    )
    adv_actor = torch.as_tensor(
        np.repeat(adv_norm[:, None], n_agents, axis=1).reshape(t_steps * n_agents),
        device=device,
    )
    states = torch.as_tensor(data["global_states"], device=device)
    returns_t = torch.as_tensor(returns, device=device)

    actor_loss_value = 0.0
    critic_loss_value = 0.0
    entropy_value = 0.0
    for _ in range(config.rl.update_epochs):
        dist, _ = actor(obs, hidden)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratio * adv_actor
        clipped = torch.clamp(
            ratio, 1.0 - config.rl.clip_eps, 1.0 + config.rl.clip_eps
        ) * adv_actor
        actor_loss = -torch.min(unclipped, clipped).mean() - config.rl.entropy_coef * entropy

        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), config.rl.max_grad_norm)
        actor_opt.step()

        values = critic(states)
        critic_loss = config.rl.value_coef * torch.mean((values - returns_t) ** 2)
        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), config.rl.max_grad_norm)
        critic_opt.step()

        actor_loss_value = float(actor_loss.detach().cpu().item())
        critic_loss_value = float(critic_loss.detach().cpu().item())
        entropy_value = float(entropy.detach().cpu().item())

    return {
        "actor_loss": actor_loss_value,
        "critic_loss": critic_loss_value,
        "entropy": entropy_value,
    }


def train(
    config: ExperimentConfig | str | Path = "configs/default.yaml",
    variant: str = "tecsf",
    output_dir: str | Path = "outputs/train",
    episodes: int | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> TrainingResult:
    cfg = load_config(config) if isinstance(config, (str, Path)) else copy.deepcopy(config)
    torch_device = resolve_device(device or cfg.rl.device)
    cfg.rl.device = str(torch_device)
    variant_spec = get_variant(variant)
    run_seed = cfg.rl.seed if seed is None else seed
    cfg.rl.seed = run_seed
    _set_seed(run_seed)
    device_obj = torch_device
    env = EnergyCarbonEnv(cfg, variant=variant_spec)
    actor = RecurrentGaussianActor(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=cfg.rl.hidden_dim,
        recurrent_dim=cfg.rl.recurrent_dim,
        use_recurrence=variant_spec.use_recurrence,
    ).to(device_obj)
    critic = CentralizedCritic(env.global_state_dim, cfg.rl.hidden_dim).to(device_obj)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.rl.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.rl.critic_lr)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    episode_count = cfg.rl.episodes if episodes is None else episodes
    episode_metrics: list[dict[str, float]] = []
    for episode in range(episode_count):
        rollout = _collect_episode(
            env,
            actor,
            critic,
            device_obj,
            use_heuristic_policy=not variant_spec.learning,
        )
        update_metrics = (
            _update_policy(cfg, actor, critic, actor_opt, critic_opt, rollout, device_obj)
            if variant_spec.learning
            else {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        )
        summary = summarize_episode(rollout.infos, rollout.per_agent_rewards)
        summary.update(update_metrics)
        summary["episode"] = float(episode)
        episode_metrics.append(summary)

    checkpoint_path = output / f"{variant}_checkpoint.pt"
    torch.save(
        {
            "variant": variant,
            "config": asdict(cfg),
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "episode_metrics": episode_metrics,
        },
        checkpoint_path,
    )
    metrics_path = output / f"{variant}_metrics.json"
    write_json(metrics_path, episode_metrics)
    return TrainingResult(
        variant=variant,
        output_dir=str(output),
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        episode_metrics=episode_metrics,
    )


def evaluate_policy(
    checkpoint_path: str | Path,
    config: ExperimentConfig | str | Path = "configs/default.yaml",
    episodes: int = 3,
    device: str | None = None,
    seed_start: int | None = None,
) -> dict[str, Any]:
    cfg = load_config(config) if isinstance(config, (str, Path)) else copy.deepcopy(config)
    torch_device = resolve_device(device or cfg.rl.device)
    cfg.rl.device = str(torch_device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch_device,
        weights_only=True,
    )
    variant = checkpoint.get("variant", "tecsf")
    variant_spec = get_variant(variant)
    device_obj = torch_device
    env = EnergyCarbonEnv(cfg, variant=variant_spec)
    actor = RecurrentGaussianActor(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=cfg.rl.hidden_dim,
        recurrent_dim=cfg.rl.recurrent_dim,
        use_recurrence=variant_spec.use_recurrence,
    ).to(device_obj)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    summaries = []
    first_seed = cfg.scenario.seed if seed_start is None else seed_start
    for ep in range(episodes):
        obs, global_state, _ = env.reset(seed=first_seed + ep)
        hidden = actor.initial_hidden(env.num_agents, device_obj)
        infos: list[dict] = []
        rewards: list[np.ndarray] = []
        done = False
        while not done:
            if not variant_spec.learning:
                action_np = env.deterministic_action()
            else:
                with torch.no_grad():
                    dist, hidden = actor(
                        torch.as_tensor(obs, dtype=torch.float32, device=device_obj), hidden
                    )
                    action = torch.clamp(dist.mean, -1.0, 1.0)
                action_np = action.cpu().numpy()
            result = env.step(action_np)
            infos.append(result.info)
            rewards.append(result.reward)
            obs = result.observation
            global_state = result.global_state
            done = result.terminated or result.truncated
        summaries.append(summarize_episode(infos, rewards))
    return {
        "variant": variant,
        "episodes": episodes,
        "device": str(torch_device),
        "seed_start": first_seed,
        "metrics": summaries,
    }
