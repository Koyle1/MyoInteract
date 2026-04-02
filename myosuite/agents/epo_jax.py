"""JAX-style evolutionary policy optimization for continuous-control MyoSuite tasks.

This implementation adapts the high-level EPO recipe to MyoInteract's Gym/MuJoCo
environments:
  - latent-conditioned actor and critic
  - PPO-style on-policy updates in JAX/Flax/Optax
  - evolutionary updates over a population of latent "genes"

The environment stepping remains on the host side because the standard MyoSuite
tasks exposed here are Python Gym environments rather than fully JAX-native MJX
environments.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pickle
import time
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

try:
    import flax
    from flax import linen as nn
    import jax
    import jax.numpy as jnp
    import optax
except ImportError as exc:
    raise ImportError(
        "EPO requires the JAX stack (jax, flax, optax). "
        "Install the project dependencies from MyoInteract/pyproject.toml."
    ) from exc

from omegaconf import OmegaConf

import myosuite  # noqa: F401  # Needed to register MyoSuite environments.
from myosuite.utils import gym


LOG_TWO_PI = math.log(2.0 * math.pi)
EPS = 1e-6


@dataclasses.dataclass
class RolloutBatch:
    observations: np.ndarray
    latents: np.ndarray
    actions: np.ndarray
    raw_actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    segment_returns: np.ndarray
    latent_indices: np.ndarray


@dataclasses.dataclass
class LatentPool:
    latents: np.ndarray
    fitness: np.ndarray

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.fitness))

    @property
    def best_latent(self) -> np.ndarray:
        return self.latents[self.best_index]


@flax.struct.dataclass
class TrainState:
    params: Any
    opt_state: optax.OptState
    step: jnp.ndarray


class FiLMBlock(nn.Module):
    features: int
    activation: Any = nn.swish

    @nn.compact
    def __call__(self, x: jax.Array, latent: jax.Array) -> jax.Array:
        x = nn.Dense(
            self.features,
            kernel_init=nn.initializers.orthogonal(math.sqrt(2.0)),
            bias_init=nn.initializers.zeros,
        )(x)
        scale = nn.Dense(
            self.features,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(latent)
        shift = nn.Dense(
            self.features,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(latent)
        return self.activation(x * (1.0 + scale) + shift)


class LatentActorCritic(nn.Module):
    action_dim: int
    hidden_sizes: Sequence[int]
    latent_hidden_size: int

    def _tower(self, prefix: str, obs: jax.Array, latent: jax.Array) -> jax.Array:
        h = obs
        z = nn.Dense(
            self.latent_hidden_size,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
            name=f"{prefix}_latent_proj",
        )(latent)
        z = nn.tanh(z)
        for index, width in enumerate(self.hidden_sizes):
            h = FiLMBlock(width, name=f"{prefix}_block_{index}")(h, z)
        return h

    @nn.compact
    def __call__(self, obs: jax.Array, latent: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        actor_hidden = self._tower("actor", obs, latent)
        critic_hidden = self._tower("critic", obs, latent)

        means = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
            name="policy_mean",
        )(actor_hidden)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        log_std = jnp.clip(log_std, -5.0, 2.0)
        log_std = jnp.broadcast_to(log_std, means.shape)

        values = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
            name="value_head",
        )(critic_hidden)
        return means, log_std, jnp.squeeze(values, axis=-1)


def train_loop(job_data) -> None:
    config = _build_config(job_data)
    envs = [_make_env(config["env"], config["seed"] + idx) for idx in range(config["n_env"])]
    eval_envs = [
        _make_env(config["env"], config["seed"] + 10_000 + idx)
        for idx in range(config["n_eval_env"])
    ]

    try:
        _train(job_data, config, envs, eval_envs)
    finally:
        for env in envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass


def _train(job_data, config: Dict[str, Any], envs: Sequence[Any], eval_envs: Sequence[Any]) -> None:
    sample_env = envs[0]
    obs_dim = int(np.prod(sample_env.observation_space.shape))
    action_dim = int(np.prod(sample_env.action_space.shape))
    action_low = np.asarray(sample_env.action_space.low, dtype=np.float32).reshape(action_dim)
    action_high = np.asarray(sample_env.action_space.high, dtype=np.float32).reshape(action_dim)
    action_center = (action_high + action_low) * 0.5
    action_scale = (action_high - action_low) * 0.5

    model = LatentActorCritic(
        action_dim=action_dim,
        hidden_sizes=tuple(config["hidden_sizes"]),
        latent_hidden_size=config["latent_hidden_size"],
    )

    rng = jax.random.PRNGKey(config["seed"])
    rng, init_key, latent_key = jax.random.split(rng, 3)
    initial_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    initial_latent = jnp.zeros((1, config["latent_dim"]), dtype=jnp.float32)
    params = model.init(init_key, initial_obs, initial_latent)["params"]
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(config["learning_rate"]),
    )
    train_state = TrainState(
        params=params,
        opt_state=optimizer.init(params),
        step=jnp.array(0, dtype=jnp.int32),
    )

    latent_pool = LatentPool(
        latents=np.asarray(
            jax.random.normal(latent_key, (config["pool_size"], config["latent_dim"]))
            * config["latent_scale"],
            dtype=np.float32,
        ),
        fitness=np.zeros(config["pool_size"], dtype=np.float32),
    )
    np_rng = np.random.default_rng(config["seed"])

    sample_action, deterministic_action, value_fn, update_step = _build_jitted_functions(
        model=model,
        optimizer=optimizer,
        action_center=jnp.asarray(action_center),
        action_scale=jnp.asarray(action_scale),
        clip_epsilon=config["clip_epsilon"],
        value_coef=config["value_coef"],
        entropy_coef=config["entropy_coef"],
    )

    observations = np.stack(
        [_flatten_observation(_reset_env(env, seed=config["seed"] + idx)) for idx, env in enumerate(envs)],
        axis=0,
    )

    total_steps = 0
    updates = max(
        1,
        config["total_timesteps"] // max(1, config["n_env"] * config["rollout_length"]),
    )
    log_dir = os.path.abspath(f"results_epo_{config['env']}")
    os.makedirs(log_dir, exist_ok=True)
    metrics_path = os.path.join(log_dir, "metrics.jsonl")

    start_time = time.time()
    last_eval_steps = 0
    last_save_steps = 0

    for update_index in range(1, updates + 1):
        rng, rollout_key = jax.random.split(rng)
        latent_indices = np_rng.integers(0, config["pool_size"], size=config["n_env"])
        rollout_latents = latent_pool.latents[latent_indices]

        batch, observations, rollout_metrics = _collect_rollout(
            envs=envs,
            observations=observations,
            latents=rollout_latents,
            latent_indices=latent_indices,
            params=train_state.params,
            sample_action=sample_action,
            value_fn=value_fn,
            rng=rollout_key,
            rollout_length=config["rollout_length"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
        )
        total_steps += config["n_env"] * config["rollout_length"]

        batch_metrics = _update_model(
            train_state=train_state,
            batch=batch,
            update_step=update_step,
            np_rng=np_rng,
            minibatch_size=config["minibatch_size"],
            update_epochs=config["update_epochs"],
        )
        train_state = batch_metrics.pop("train_state")

        if update_index >= config["evolution_warmup_updates"]:
            latent_pool = evolve_latent_pool(
                latent_pool=latent_pool,
                latent_indices=batch.latent_indices,
                segment_returns=batch.segment_returns,
                elite_fraction=config["elite_fraction"],
                mutation_std=config["mutation_std"],
                mutation_clip=config["mutation_clip"],
                crossover_rate=config["crossover_rate"],
                fitness_ema=config["fitness_ema"],
                np_rng=np_rng,
            )

        best_latent = latent_pool.best_latent
        metrics = {
            "update": update_index,
            "env_steps": total_steps,
            "rollout/mean_reward": float(np.mean(batch.rewards)),
            "rollout/mean_segment_return": float(np.mean(batch.segment_returns)),
            "rollout/done_fraction": float(np.mean(batch.dones)),
            "ppo/policy_loss": batch_metrics["policy_loss"],
            "ppo/value_loss": batch_metrics["value_loss"],
            "ppo/entropy": batch_metrics["entropy"],
            "ppo/grad_norm": batch_metrics["grad_norm"],
            "evo/best_fitness": float(np.max(latent_pool.fitness)),
            "evo/mean_fitness": float(np.mean(latent_pool.fitness)),
            "evo/best_latent_norm": float(np.linalg.norm(best_latent)),
            "time/elapsed_sec": float(time.time() - start_time),
        }
        metrics.update(rollout_metrics)

        if update_index == 1 or update_index % config["log_interval"] == 0:
            print(_format_metrics(metrics))
            with open(metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")

        if total_steps - last_eval_steps >= config["eval_freq"]:
            eval_metrics = evaluate_policy(
                envs=eval_envs,
                params=train_state.params,
                latent=best_latent,
                deterministic_action=deterministic_action,
                eval_episodes=config["eval_episodes"],
            )
            print(_format_metrics({"eval/env_steps": total_steps, **eval_metrics}))
            last_eval_steps = total_steps

        if total_steps - last_save_steps >= config["save_freq"] or update_index == updates:
            _save_checkpoint(
                log_dir=log_dir,
                total_steps=total_steps,
                train_state=train_state,
                latent_pool=latent_pool,
                config=job_data,
            )
            last_save_steps = total_steps

    print(
        "Finished EPO training for "
        f"{config['env']} in {time.time() - start_time:.1f}s "
        f"after {total_steps} environment steps."
    )


def _build_jitted_functions(
    model: LatentActorCritic,
    optimizer: optax.GradientTransformation,
    action_center: jax.Array,
    action_scale: jax.Array,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
):
    def apply_model(params, obs, latents):
        return model.apply({"params": params}, obs, latents)

    @jax.jit
    def sample_action(params, obs, latents, rng):
        means, log_std, values = apply_model(params, obs, latents)
        noise = jax.random.normal(rng, means.shape)
        raw_actions = means + jnp.exp(log_std) * noise
        squashed = jnp.tanh(raw_actions)
        actions = action_center + action_scale * squashed
        log_probs = _squashed_gaussian_log_prob(raw_actions, means, log_std)
        return actions, raw_actions, log_probs, values

    @jax.jit
    def deterministic_action(params, obs, latents):
        means, _, values = apply_model(params, obs, latents)
        squashed = jnp.tanh(means)
        actions = action_center + action_scale * squashed
        return actions, values

    @jax.jit
    def value_fn(params, obs, latents):
        _, _, values = apply_model(params, obs, latents)
        return values

    @jax.jit
    def update_step(
        train_state: TrainState,
        observations: jax.Array,
        latents: jax.Array,
        raw_actions: jax.Array,
        old_log_probs: jax.Array,
        old_values: jax.Array,
        advantages: jax.Array,
        returns: jax.Array,
    ) -> Tuple[TrainState, Mapping[str, jax.Array]]:
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def loss_fn(params):
            means, log_std, values = apply_model(params, observations, latents)
            log_probs = _squashed_gaussian_log_prob(raw_actions, means, log_std)
            log_ratio = log_probs - old_log_probs
            ratio = jnp.exp(log_ratio)
            clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            policy_loss = -jnp.mean(
                jnp.minimum(ratio * normalized_advantages, clipped_ratio * normalized_advantages)
            )

            value_pred_clipped = old_values + jnp.clip(values - old_values, -clip_epsilon, clip_epsilon)
            value_losses = jnp.square(values - returns)
            value_losses_clipped = jnp.square(value_pred_clipped - returns)
            value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))

            entropy = jnp.mean(_gaussian_entropy(log_std))
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
            }
            return loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(train_state.params)
        updates, new_opt_state = optimizer.update(grads, train_state.opt_state, train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)
        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            step=train_state.step + 1,
        )
        metrics = dict(metrics)
        metrics["loss"] = loss
        metrics["grad_norm"] = optax.global_norm(grads)
        return new_state, metrics

    return sample_action, deterministic_action, value_fn, update_step


def _collect_rollout(
    envs: Sequence[Any],
    observations: np.ndarray,
    latents: np.ndarray,
    latent_indices: np.ndarray,
    params: Any,
    sample_action,
    value_fn,
    rng: jax.Array,
    rollout_length: int,
    gamma: float,
    gae_lambda: float,
) -> Tuple[RolloutBatch, np.ndarray, Dict[str, float]]:
    num_envs, obs_dim = observations.shape
    action_dim = int(np.prod(envs[0].action_space.shape))
    latent_dim = latents.shape[-1]

    obs_buf = np.zeros((rollout_length, num_envs, obs_dim), dtype=np.float32)
    latent_buf = np.zeros((rollout_length, num_envs, latent_dim), dtype=np.float32)
    action_buf = np.zeros((rollout_length, num_envs, action_dim), dtype=np.float32)
    raw_action_buf = np.zeros((rollout_length, num_envs, action_dim), dtype=np.float32)
    reward_buf = np.zeros((rollout_length, num_envs), dtype=np.float32)
    done_buf = np.zeros((rollout_length, num_envs), dtype=np.float32)
    value_buf = np.zeros((rollout_length, num_envs), dtype=np.float32)
    log_prob_buf = np.zeros((rollout_length, num_envs), dtype=np.float32)
    segment_returns = np.zeros(num_envs, dtype=np.float32)

    current_obs = observations.copy()
    current_latents = latents.astype(np.float32, copy=True)
    episode_resets = 0

    for step in range(rollout_length):
        obs_buf[step] = current_obs
        latent_buf[step] = current_latents
        rng, sample_key = jax.random.split(rng)
        actions, raw_actions, log_probs, values = sample_action(
            params,
            jnp.asarray(current_obs),
            jnp.asarray(current_latents),
            sample_key,
        )
        actions = np.asarray(jax.device_get(actions), dtype=np.float32)
        raw_actions = np.asarray(jax.device_get(raw_actions), dtype=np.float32)
        log_probs = np.asarray(jax.device_get(log_probs), dtype=np.float32)
        values = np.asarray(jax.device_get(values), dtype=np.float32)

        action_buf[step] = actions
        raw_action_buf[step] = raw_actions
        log_prob_buf[step] = log_probs
        value_buf[step] = values

        next_obs = np.zeros_like(current_obs)
        for env_index, env in enumerate(envs):
            obs, reward, done, _ = _step_env(env, actions[env_index])
            segment_returns[env_index] += reward
            reward_buf[step, env_index] = reward
            done_buf[step, env_index] = float(done)
            if done:
                episode_resets += 1
                obs = _reset_env(env)
            next_obs[env_index] = _flatten_observation(obs)
        current_obs = next_obs

    last_values = np.asarray(
        jax.device_get(value_fn(params, jnp.asarray(current_obs), jnp.asarray(current_latents))),
        dtype=np.float32,
    )
    advantages, returns = _compute_gae(
        rewards=reward_buf,
        values=value_buf,
        dones=done_buf,
        last_values=last_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    batch = RolloutBatch(
        observations=obs_buf,
        latents=latent_buf,
        actions=action_buf,
        raw_actions=raw_action_buf,
        rewards=reward_buf,
        dones=done_buf,
        values=value_buf,
        log_probs=log_prob_buf,
        advantages=advantages,
        returns=returns,
        segment_returns=segment_returns,
        latent_indices=np.asarray(latent_indices, dtype=np.int32),
    )
    metrics = {
        "rollout/episode_resets": float(episode_resets),
    }
    return batch, current_obs, metrics


def _update_model(
    train_state: TrainState,
    batch: RolloutBatch,
    update_step,
    np_rng: np.random.Generator,
    minibatch_size: int,
    update_epochs: int,
) -> Dict[str, Any]:
    flat = {
        "observations": batch.observations.reshape((-1, batch.observations.shape[-1])),
        "latents": batch.latents.reshape((-1, batch.latents.shape[-1])),
        "raw_actions": batch.raw_actions.reshape((-1, batch.raw_actions.shape[-1])),
        "log_probs": batch.log_probs.reshape(-1),
        "values": batch.values.reshape(-1),
        "advantages": batch.advantages.reshape(-1),
        "returns": batch.returns.reshape(-1),
    }

    num_items = flat["observations"].shape[0]
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "grad_norm": 0.0,
    }
    metric_steps = 0

    for _ in range(update_epochs):
        permutation = np_rng.permutation(num_items)
        for start in range(0, num_items, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            train_state, batch_metrics = update_step(
                train_state,
                jnp.asarray(flat["observations"][indices]),
                jnp.asarray(flat["latents"][indices]),
                jnp.asarray(flat["raw_actions"][indices]),
                jnp.asarray(flat["log_probs"][indices]),
                jnp.asarray(flat["values"][indices]),
                jnp.asarray(flat["advantages"][indices]),
                jnp.asarray(flat["returns"][indices]),
            )
            batch_metrics = jax.device_get(batch_metrics)
            for key in metrics:
                metrics[key] += float(batch_metrics[key])
            metric_steps += 1

    for key in metrics:
        metrics[key] /= max(1, metric_steps)
    metrics["train_state"] = train_state
    return metrics


def evolve_latent_pool(
    latent_pool: LatentPool,
    latent_indices: np.ndarray,
    segment_returns: np.ndarray,
    elite_fraction: float,
    mutation_std: float,
    mutation_clip: float,
    crossover_rate: float,
    fitness_ema: float,
    np_rng: np.random.Generator,
) -> LatentPool:
    latents = latent_pool.latents.copy()
    fitness = latent_pool.fitness.copy()
    counts = np.bincount(latent_indices, minlength=latents.shape[0]).astype(np.float32)
    sums = np.bincount(
        latent_indices,
        weights=np.asarray(segment_returns, dtype=np.float32),
        minlength=latents.shape[0],
    ).astype(np.float32)
    observed = counts > 0
    if np.any(observed):
        batch_scores = np.divide(sums, np.maximum(counts, 1.0))
        fitness[observed] = fitness_ema * fitness[observed] + (1.0 - fitness_ema) * batch_scores[observed]

    elite_count = max(1, int(round(latents.shape[0] * elite_fraction)))
    elite_indices = np.argsort(fitness)[-elite_count:][::-1]

    new_latents = np.zeros_like(latents)
    new_fitness = np.zeros_like(fitness)
    new_latents[:elite_count] = latents[elite_indices]
    new_fitness[:elite_count] = fitness[elite_indices]

    for child_index in range(elite_count, latents.shape[0]):
        parent_a, parent_b = np_rng.choice(elite_indices, size=2, replace=True)
        crossover_mask = np_rng.random(latents.shape[1]) < crossover_rate
        child = np.where(crossover_mask, latents[parent_a], latents[parent_b])
        child += np_rng.normal(loc=0.0, scale=mutation_std, size=child.shape)
        child = np.clip(child, -mutation_clip, mutation_clip)
        new_latents[child_index] = child.astype(np.float32)
        new_fitness[child_index] = 0.5 * (fitness[parent_a] + fitness[parent_b])

    return LatentPool(latents=new_latents, fitness=new_fitness)


def evaluate_policy(
    envs: Sequence[Any],
    params: Any,
    latent: np.ndarray,
    deterministic_action,
    eval_episodes: int,
) -> Dict[str, float]:
    returns = []
    lengths = []
    latent_batch = np.repeat(latent[None, :], repeats=1, axis=0).astype(np.float32)

    episodes_per_env = max(1, int(math.ceil(eval_episodes / max(1, len(envs)))))
    for env in envs:
        for _ in range(episodes_per_env):
            obs = _flatten_observation(_reset_env(env))
            done = False
            total_reward = 0.0
            total_length = 0
            while not done:
                actions, _ = deterministic_action(
                    params,
                    jnp.asarray(obs[None, :]),
                    jnp.asarray(latent_batch),
                )
                action = np.asarray(jax.device_get(actions[0]), dtype=np.float32)
                obs, reward, done, _ = _step_env(env, action)
                obs = _flatten_observation(obs)
                total_reward += reward
                total_length += 1
            returns.append(total_reward)
            lengths.append(total_length)
            if len(returns) >= eval_episodes:
                return {
                    "eval/mean_return": float(np.mean(returns)),
                    "eval/std_return": float(np.std(returns)),
                    "eval/mean_length": float(np.mean(lengths)),
                }

    return {
        "eval/mean_return": float(np.mean(returns) if returns else 0.0),
        "eval/std_return": float(np.std(returns) if returns else 0.0),
        "eval/mean_length": float(np.mean(lengths) if lengths else 0.0),
    }


def _build_config(job_data) -> Dict[str, Any]:
    config = {
        "env": str(getattr(job_data, "env")),
        "seed": int(getattr(job_data, "seed", 0)),
        "n_env": int(getattr(job_data, "n_env", 8)),
        "n_eval_env": int(getattr(job_data, "n_eval_env", 2)),
        "total_timesteps": int(getattr(job_data, "total_timesteps", 1_000_000)),
        "rollout_length": int(getattr(job_data, "rollout_length", 256)),
        "update_epochs": int(getattr(job_data, "update_epochs", 4)),
        "minibatch_size": int(getattr(job_data, "minibatch_size", 1024)),
        "learning_rate": float(getattr(job_data, "learning_rate", 3e-4)),
        "gamma": float(getattr(job_data, "gamma", 0.99)),
        "gae_lambda": float(getattr(job_data, "gae_lambda", 0.95)),
        "clip_epsilon": float(getattr(job_data, "clip_epsilon", 0.2)),
        "entropy_coef": float(getattr(job_data, "entropy_coef", 1e-3)),
        "value_coef": float(getattr(job_data, "value_coef", 0.5)),
        "max_grad_norm": float(getattr(job_data, "max_grad_norm", 1.0)),
        "hidden_sizes": _as_sequence(getattr(job_data, "hidden_sizes", [256, 256])),
        "latent_hidden_size": int(getattr(job_data, "latent_hidden_size", 128)),
        "latent_dim": int(getattr(job_data, "latent_dim", 32)),
        "pool_size": int(getattr(job_data, "pool_size", 64)),
        "latent_scale": float(getattr(job_data, "latent_scale", 1.0)),
        "elite_fraction": float(getattr(job_data, "elite_fraction", 0.25)),
        "mutation_std": float(getattr(job_data, "mutation_std", 0.15)),
        "mutation_clip": float(getattr(job_data, "mutation_clip", 3.0)),
        "crossover_rate": float(getattr(job_data, "crossover_rate", 0.5)),
        "fitness_ema": float(getattr(job_data, "fitness_ema", 0.5)),
        "evolution_warmup_updates": int(getattr(job_data, "evolution_warmup_updates", 1)),
        "eval_episodes": int(getattr(job_data, "eval_episodes", 5)),
        "eval_freq": int(getattr(job_data, "eval_freq", 200_000)),
        "save_freq": int(getattr(job_data, "save_freq", 500_000)),
        "log_interval": int(getattr(job_data, "log_interval", 10)),
    }

    overrides = _as_plain_dict(job_data, "alg_hyper_params")
    config.update({key: overrides[key] for key in overrides if key in config})
    if "hidden_sizes" in overrides:
        config["hidden_sizes"] = _as_sequence(overrides["hidden_sizes"])
    return config


def _as_plain_dict(job_data, key: str) -> Dict[str, Any]:
    if hasattr(job_data, "get"):
        value = job_data.get(key, None)
    else:
        value = getattr(job_data, key, None)
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_sequence(value: Any) -> Sequence[int]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return [256, 256]
        return [int(chunk.strip()) for chunk in value.split(",") if chunk.strip()]
    if isinstance(value, Iterable):
        return [int(item) for item in value]
    return [int(value)]


def _make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    _reset_env(env, seed=seed)
    return env


def _reset_env(env, seed: int | None = None) -> np.ndarray:
    if seed is None:
        reset_out = env.reset()
    else:
        try:
            reset_out = env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
            reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out
    return np.asarray(obs, dtype=np.float32)


def _step_env(env, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    step_out = env.step(np.asarray(action, dtype=np.float32))
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = step_out
        done = bool(done)
    return np.asarray(obs, dtype=np.float32), float(reward), done, dict(info)


def _flatten_observation(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in range(rewards.shape[0] - 1, -1, -1):
        if step == rewards.shape[0] - 1:
            next_values = last_values
        else:
            next_values = values[step + 1]
        not_done = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * not_done - values[step]
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    return advantages, returns


def _gaussian_entropy(log_std: jax.Array) -> jax.Array:
    return jnp.sum(log_std + 0.5 * (1.0 + LOG_TWO_PI), axis=-1)


def _squashed_gaussian_log_prob(raw_actions: jax.Array, means: jax.Array, log_std: jax.Array) -> jax.Array:
    inv_std = jnp.exp(-log_std)
    centered = (raw_actions - means) * inv_std
    gaussian_log_prob = -0.5 * jnp.sum(centered * centered + 2.0 * log_std + LOG_TWO_PI, axis=-1)
    correction = jnp.sum(jnp.log(1.0 - jnp.tanh(raw_actions) ** 2 + EPS), axis=-1)
    return gaussian_log_prob - correction


def _save_checkpoint(
    log_dir: str,
    total_steps: int,
    train_state: TrainState,
    latent_pool: LatentPool,
    config,
) -> None:
    config_payload = config
    if OmegaConf.is_config(config):
        config_payload = OmegaConf.to_container(config, resolve=True)
    payload = {
        "env_steps": int(total_steps),
        "params": jax.device_get(train_state.params),
        "step": int(np.asarray(jax.device_get(train_state.step))),
        "latent_pool": latent_pool.latents,
        "latent_fitness": latent_pool.fitness,
        "config": config_payload,
    }
    latest_path = os.path.join(log_dir, "epo_latest.pkl")
    step_path = os.path.join(log_dir, f"epo_step_{int(total_steps):012d}.pkl")
    for path in (latest_path, step_path):
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _format_metrics(metrics: Mapping[str, Any]) -> str:
    ordered = []
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, (int, np.integer)):
            ordered.append(f"{key}={int(value)}")
        else:
            ordered.append(f"{key}={float(value):.4f}")
    return " | ".join(ordered)
