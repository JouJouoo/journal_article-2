from __future__ import annotations

import copy
import sys
import time
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
from tecsf.rl.networks import (
    AdvantageGate,
    CentralizedCritic,
    ClipGate,
    CreditAssignmentNetwork,
    RecurrentGaussianActor,
)
from tecsf.variants import VariantSpec, get_variant


@dataclass
class TrainingResult:
    variant: str
    output_dir: str
    checkpoint_path: str
    best_checkpoint_path: str
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


def _compute_gae_per_agent(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    reward_arr = np.asarray(rewards, dtype=np.float32)
    value_arr = np.asarray(values, dtype=np.float32)
    if value_arr.ndim == 1:
        value_arr = np.repeat(value_arr[:, None], reward_arr.shape[1], axis=1)
    advantages = np.zeros_like(reward_arr, dtype=np.float32)
    last_gae = np.zeros(reward_arr.shape[1], dtype=np.float32)
    next_value = np.zeros(reward_arr.shape[1], dtype=np.float32)
    for step in reversed(range(reward_arr.shape[0])):
        nonterminal = 1.0 - float(dones[step])
        delta = reward_arr[step] + gamma * next_value * nonterminal - value_arr[step]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[step] = last_gae
        next_value = value_arr[step]
    returns = advantages + value_arr
    return advantages.astype(np.float32), returns.astype(np.float32)


def _normalize_advantage(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return ((arr - arr.mean()) / (arr.std() + 1e-8)).astype(np.float32)


def _lr_factor(config: ExperimentConfig, episode: int) -> float:
    decay_episodes = max(int(config.rl.lr_decay_episodes), 0)
    min_factor = float(config.rl.min_lr_factor)
    if decay_episodes <= 0:
        return 1.0
    progress = min(max(float(episode) / float(decay_episodes), 0.0), 1.0)
    return max(min_factor, 1.0 - progress * (1.0 - min_factor))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _inactive_update_metrics() -> dict[str, float]:
    return {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "action_prior_loss": 0.0,
        "action_prior_coef": 0.0,
        "approx_kl": 0.0,
        "policy_update_accepted": 0.0,
    }


def _should_restore_best(
    config: ExperimentConfig,
    episode: int,
    best_episode: int | None,
    updates_frozen: bool,
) -> bool:
    if updates_frozen or not bool(config.rl.restore_best_on_plateau):
        return False
    if best_episode is None:
        return False
    patience = max(int(config.rl.early_stop_patience), 0)
    warmup = max(int(config.rl.early_stop_warmup_episodes), 0)
    if patience <= 0 or episode < warmup:
        return False
    return episode - best_episode >= patience


def _dual_reward_components(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    reward_eco = np.asarray(data["per_agent_reward_eco"], dtype=np.float32)
    reward_coin = np.asarray(data["per_agent_reward_coin"], dtype=np.float32)
    total_reward = np.asarray(data["per_agent_rewards"], dtype=np.float32)
    penalty_reward = total_reward - reward_eco - reward_coin
    return reward_eco + penalty_reward, reward_coin


def _collect_episode(
    env: EnergyCarbonEnv,
    actor: RecurrentGaussianActor,
    critic: CentralizedCritic,
    device: torch.device,
    use_heuristic_policy: bool = False,
) -> RolloutBuffer:
    buffer = RolloutBuffer()
    buffer.set_variant(env.variant)
    obs, global_state, _ = env.reset()
    hidden = actor.initial_hidden(env.num_agents, device)
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        state_t = torch.as_tensor(global_state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            hidden_before = hidden.clone()
            action_prior = env.greedy_feasible_action()
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
            value=value_t.cpu().numpy(),
            info=result.info,
            action_prior=action_prior,
        )
        obs = result.observation
        global_state = result.global_state
        hidden = next_hidden.detach()
        done = result.terminated or result.truncated
    return buffer


def _update_policy(
    config: ExperimentConfig,
    variant_spec: VariantSpec,
    actor: RecurrentGaussianActor,
    critic: CentralizedCritic,
    credit_net: CreditAssignmentNetwork,
    advantage_gate: AdvantageGate,
    clip_gate: ClipGate,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    rollout: RolloutBuffer,
    device: torch.device,
    episode: int,
) -> dict[str, float]:
    data = rollout.arrays()
    critic_values = data["values"]
    if critic_values.ndim == 1:
        critic_values = critic_values[:, None]
    if variant_spec.use_dual_advantage:
        reward_eco, reward_coin = _dual_reward_components(data)
        adv_eco, returns_eco = _compute_gae_per_agent(
            reward_eco,
            data["dones"],
            critic_values[:, 0],
            config.rl.gamma,
            config.rl.gae_lambda,
        )
        adv_coin, returns_coin = _compute_gae_per_agent(
            reward_coin,
            data["dones"],
            critic_values[:, 1],
            config.rl.gamma,
            config.rl.gae_lambda,
        )
        adv_eco = _normalize_advantage(adv_eco)
        adv_coin = _normalize_advantage(adv_coin)
        critic_targets = np.stack(
            [returns_eco.mean(axis=1), returns_coin.mean(axis=1)],
            axis=1,
        ).astype(np.float32)
    else:
        adv_total, returns_total = _compute_gae_per_agent(
            data["per_agent_rewards"],
            data["dones"],
            critic_values[:, 0],
            config.rl.gamma,
            config.rl.gae_lambda,
        )
        adv_total = _normalize_advantage(adv_total)
        critic_targets = np.zeros((returns_total.shape[0], 2), dtype=np.float32)
        critic_targets[:, 0] = returns_total.mean(axis=1)
    t_steps, n_agents, obs_dim = data["observations"].shape
    action_dim = data["actions"].shape[-1]

    obs_by_step = torch.as_tensor(data["observations"], device=device)
    obs = torch.as_tensor(data["observations"].reshape(t_steps * n_agents, obs_dim), device=device)
    hidden = torch.as_tensor(
        data["hidden_states"].reshape(t_steps * n_agents, -1), device=device
    )
    actions = torch.as_tensor(
        data["actions"].reshape(t_steps * n_agents, action_dim), device=device
    )
    action_priors = torch.as_tensor(
        data["action_priors"].reshape(t_steps * n_agents, action_dim), device=device
    )
    old_log_probs = torch.as_tensor(
        data["log_probs"].reshape(t_steps * n_agents), device=device
    )
    states = torch.as_tensor(data["global_states"], device=device)
    targets_t = torch.as_tensor(critic_targets, device=device)
    asset_states = torch.as_tensor(data["asset_states"], device=device)
    asset_flat = asset_states.reshape(t_steps * n_agents)
    trade_relations = torch.as_tensor(data["trade_relations"], device=device)
    adv_eco_t = (
        torch.as_tensor(adv_eco, device=device)
        if variant_spec.use_dual_advantage
        else None
    )
    adv_coin_t = (
        torch.as_tensor(adv_coin, device=device)
        if variant_spec.use_dual_advantage
        else None
    )
    adv_total_t = (
        None
        if variant_spec.use_dual_advantage
        else torch.as_tensor(adv_total, device=device)
    )

    actor_loss_value = 0.0
    critic_loss_value = 0.0
    entropy_value = 0.0
    action_prior_loss_value = 0.0
    approx_kl_value = 0.0
    policy_update_accepted = 1.0
    action_prior_coef = 0.0
    if variant_spec.name in {"tecsf", "lc_mappo"}:
        decay = max(int(config.rl.action_prior_decay_episodes), 0)
        if decay == 0:
            action_prior_coef = float(config.rl.action_prior_coef)
        else:
            remaining = max(0.0, 1.0 - float(episode) / float(decay))
            action_prior_coef = float(config.rl.action_prior_coef) * remaining
    for _ in range(config.rl.update_epochs):
        if variant_spec.use_dual_advantage:
            assert adv_eco_t is not None
            assert adv_coin_t is not None
            if variant_spec.use_credit_assignment:
                agent_features = torch.cat(
                    [obs_by_step, asset_states.unsqueeze(-1)],
                    dim=-1,
                )
                credit_weights = credit_net(agent_features, trade_relations)
                assigned_coin_adv = torch.sum(
                    credit_weights * adv_coin_t.unsqueeze(1),
                    dim=-1,
                )
            else:
                assigned_coin_adv = adv_coin_t
            if variant_spec.use_asset_utility:
                gate_weights = advantage_gate(asset_states)
                actor_advantage = (
                    gate_weights[..., 0] * adv_eco_t
                    + gate_weights[..., 1] * assigned_coin_adv
                )
            else:
                actor_advantage = adv_eco_t
            adv_actor = actor_advantage.reshape(t_steps * n_agents)
        else:
            assert adv_total_t is not None
            adv_actor = adv_total_t.reshape(t_steps * n_agents)
        adv_actor = (adv_actor - adv_actor.mean()) / (
            adv_actor.std(unbiased=False) + 1e-8
        )

        dist, _ = actor(obs, hidden)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        ratio = torch.exp(new_log_probs - old_log_probs)
        if variant_spec.use_adaptive_clip:
            clip_eps = clip_gate(asset_flat)
        else:
            clip_eps = torch.full_like(ratio, float(config.rl.clip_eps))
        unclipped = ratio * adv_actor
        clipped_ratio = torch.max(
            torch.min(ratio, 1.0 + clip_eps),
            1.0 - clip_eps,
        )
        clipped = clipped_ratio * adv_actor
        action_prior_loss = torch.mean((dist.mean - action_priors) ** 2)
        actor_loss = (
            -torch.min(unclipped, clipped).mean()
            - config.rl.entropy_coef * entropy
            + action_prior_coef * action_prior_loss
        )

        target_kl = float(config.rl.target_kl)
        actor_state = None
        actor_opt_state = None
        if target_kl > 0.0:
            actor_state = {
                "actor": copy.deepcopy(actor.state_dict()),
                "credit_net": copy.deepcopy(credit_net.state_dict()),
                "advantage_gate": copy.deepcopy(advantage_gate.state_dict()),
                "clip_gate": copy.deepcopy(clip_gate.state_dict()),
            }
            actor_opt_state = copy.deepcopy(actor_opt.state_dict())
        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), config.rl.max_grad_norm)
        actor_opt.step()
        if target_kl > 0.0:
            with torch.no_grad():
                checked_dist, _ = actor(obs, hidden)
                checked_log_probs = checked_dist.log_prob(actions).sum(dim=-1)
                log_ratio = checked_log_probs - old_log_probs
                approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
            approx_kl_value = float(approx_kl.detach().cpu().item())
            if approx_kl_value > target_kl:
                assert actor_state is not None
                assert actor_opt_state is not None
                actor.load_state_dict(actor_state["actor"])
                credit_net.load_state_dict(actor_state["credit_net"])
                advantage_gate.load_state_dict(actor_state["advantage_gate"])
                clip_gate.load_state_dict(actor_state["clip_gate"])
                actor_opt.load_state_dict(actor_opt_state)
                policy_update_accepted = 0.0

        values = critic(states)
        if variant_spec.use_dual_advantage:
            value_error = (values - targets_t) ** 2
        else:
            value_error = (values[:, 0] - targets_t[:, 0]) ** 2
        critic_loss = config.rl.value_coef * torch.mean(value_error)
        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), config.rl.max_grad_norm)
        critic_opt.step()

        actor_loss_value = float(actor_loss.detach().cpu().item())
        critic_loss_value = float(critic_loss.detach().cpu().item())
        entropy_value = float(entropy.detach().cpu().item())
        action_prior_loss_value = float(action_prior_loss.detach().cpu().item())

    return {
        "actor_loss": actor_loss_value,
        "critic_loss": critic_loss_value,
        "entropy": entropy_value,
        "action_prior_loss": action_prior_loss_value,
        "action_prior_coef": float(action_prior_coef),
        "approx_kl": approx_kl_value,
        "policy_update_accepted": policy_update_accepted,
    }


def _checkpoint_payload(
    variant: str,
    cfg: ExperimentConfig,
    state_dicts: dict[str, Any],
    episode_metrics: list[dict[str, float]],
    best_episode: int | None = None,
    best_score: float | None = None,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "config": asdict(cfg),
        **state_dicts,
        "episode_metrics": episode_metrics,
        "best_episode": best_episode,
        "best_score": best_score,
    }


def _model_state_dicts(
    actor: RecurrentGaussianActor,
    critic: CentralizedCritic,
    credit_net: CreditAssignmentNetwork,
    advantage_gate: AdvantageGate,
    clip_gate: ClipGate,
) -> dict[str, Any]:
    return {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "credit_assignment_state_dict": credit_net.state_dict(),
        "advantage_gate_state_dict": advantage_gate.state_dict(),
        "clip_gate_state_dict": clip_gate.state_dict(),
    }


def _load_model_state_dicts(
    state_dicts: dict[str, Any],
    actor: RecurrentGaussianActor,
    critic: CentralizedCritic,
    credit_net: CreditAssignmentNetwork,
    advantage_gate: AdvantageGate,
    clip_gate: ClipGate,
) -> None:
    actor.load_state_dict(state_dicts["actor_state_dict"])
    critic.load_state_dict(state_dicts["critic_state_dict"])
    credit_net.load_state_dict(state_dicts["credit_assignment_state_dict"])
    advantage_gate.load_state_dict(state_dicts["advantage_gate_state_dict"])
    clip_gate.load_state_dict(state_dicts["clip_gate_state_dict"])


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
        log_std_init=cfg.rl.log_std_init,
        log_std_min=cfg.rl.log_std_min,
        log_std_max=cfg.rl.log_std_max,
    ).to(device_obj)
    critic = CentralizedCritic(env.global_state_dim, cfg.rl.hidden_dim).to(device_obj)
    credit_net = CreditAssignmentNetwork(
        agent_feature_dim=env.observation_dim + 1,
        hidden_dim=cfg.rl.hidden_dim,
    ).to(device_obj)
    advantage_gate = AdvantageGate(cfg.rl.hidden_dim).to(device_obj)
    clip_gate = ClipGate(
        cfg.rl.hidden_dim,
        clip_min=cfg.rl.clip_eps_min,
        clip_max=cfg.rl.clip_eps_max,
    ).to(device_obj)
    actor_params = (
        list(actor.parameters())
        + list(credit_net.parameters())
        + list(advantage_gate.parameters())
        + list(clip_gate.parameters())
    )
    actor_opt = torch.optim.Adam(actor_params, lr=cfg.rl.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.rl.critic_lr)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    episode_count = cfg.rl.episodes if episodes is None else episodes
    episode_metrics: list[dict[str, float]] = []
    best_score = float("-inf")
    best_episode: int | None = None
    best_state_dicts: dict[str, Any] | None = None
    best_window = max(int(cfg.rl.best_checkpoint_window), 1)
    best_min_delta = max(float(cfg.rl.early_stop_min_delta), 0.0)
    updates_frozen = False
    restored_best_episode: int | None = None
    _progress_log_every = max(episode_count // 100, 1)
    _progress_t0 = time.time()
    for episode in range(episode_count):
        lr_factor = _lr_factor(cfg, episode)
        actor_lr = float(cfg.rl.actor_lr) * lr_factor
        critic_lr = float(cfg.rl.critic_lr) * lr_factor
        _set_optimizer_lr(actor_opt, actor_lr)
        _set_optimizer_lr(critic_opt, critic_lr)
        rollout = _collect_episode(
            env,
            actor,
            critic,
            device_obj,
            use_heuristic_policy=not variant_spec.learning,
        )
        skip_update = updates_frozen or lr_factor <= 0.0
        update_metrics = (
            _update_policy(
                cfg,
                variant_spec,
                actor,
                critic,
                credit_net,
                advantage_gate,
                clip_gate,
                actor_opt,
                critic_opt,
                rollout,
                device_obj,
                episode,
            )
            if variant_spec.learning and not skip_update
            else _inactive_update_metrics()
        )
        summary = summarize_episode(rollout.infos, rollout.per_agent_rewards)
        summary.update(update_metrics)
        summary["episode"] = float(episode)
        summary["actor_lr"] = actor_lr
        summary["critic_lr"] = critic_lr
        summary["lr_factor"] = lr_factor
        summary["updates_frozen"] = float(updates_frozen or lr_factor <= 0.0)
        summary["early_stop_triggered"] = 0.0
        episode_metrics.append(summary)
        if (episode + 1) % _progress_log_every == 0 or episode == episode_count - 1:
            elapsed = time.time() - _progress_t0
            rate = (episode + 1) / elapsed if elapsed > 0 else 0.0
            eta = (episode_count - episode - 1) / rate if rate > 0 else 0.0
            print(
                f"[{variant}] episode {episode + 1}/{episode_count} "
                f"reward={summary['mean_reward']:.4f} "
                f"elapsed={elapsed:.0f}s rate={rate:.1f}ep/s eta={eta:.0f}s",
                flush=True,
            )
        recent = episode_metrics[-best_window:]
        rolling_score = float(np.mean([m["mean_reward"] for m in recent]))
        if rolling_score > best_score + best_min_delta:
            best_score = rolling_score
            best_episode = episode
            best_state_dicts = copy.deepcopy(
                _model_state_dicts(
                    actor, critic, credit_net, advantage_gate, clip_gate
                )
            )
        if _should_restore_best(cfg, episode, best_episode, updates_frozen):
            if best_state_dicts is not None:
                _load_model_state_dicts(
                    best_state_dicts,
                    actor,
                    critic,
                    credit_net,
                    advantage_gate,
                    clip_gate,
                )
            updates_frozen = True
            restored_best_episode = best_episode
            summary["early_stop_triggered"] = 1.0

    checkpoint_path = output / f"{variant}_checkpoint.pt"
    best_checkpoint_path = output / f"{variant}_best_checkpoint.pt"
    final_state_dicts = _model_state_dicts(
        actor, critic, credit_net, advantage_gate, clip_gate
    )
    payload = _checkpoint_payload(
        variant,
        cfg,
        final_state_dicts,
        episode_metrics,
        best_episode=best_episode,
        best_score=best_score if np.isfinite(best_score) else None,
    )
    payload["best_checkpoint_path"] = str(best_checkpoint_path)
    payload["restored_best_episode"] = restored_best_episode
    best_payload = _checkpoint_payload(
        variant,
        cfg,
        best_state_dicts or final_state_dicts,
        episode_metrics,
        best_episode=best_episode,
        best_score=best_score if np.isfinite(best_score) else None,
    )
    best_payload["best_checkpoint_path"] = str(best_checkpoint_path)
    best_payload["restored_best_episode"] = restored_best_episode
    torch.save(payload, checkpoint_path)
    torch.save(best_payload, best_checkpoint_path)
    metrics_path = output / f"{variant}_metrics.json"
    write_json(metrics_path, episode_metrics)
    return TrainingResult(
        variant=variant,
        output_dir=str(output),
        checkpoint_path=str(checkpoint_path),
        best_checkpoint_path=str(best_checkpoint_path),
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
        log_std_init=cfg.rl.log_std_init,
        log_std_min=cfg.rl.log_std_min,
        log_std_max=cfg.rl.log_std_max,
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
