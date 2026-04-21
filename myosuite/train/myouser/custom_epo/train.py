"""JAX EPO training for MJX-backed MyoSuite environments.

This trainer keeps the environment interaction fully in JAX/MJX while using a
latent-conditioned actor-critic and an evolutionary latent pool between PPO
style policy updates.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from absl import logging
from brax import envs
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import PRNGKey
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from orbax import checkpoint as ocp
import optax


LOG_TWO_PI = math.log(2.0 * math.pi)
EPS = 1e-6


@flax.struct.dataclass
class LatentPool:
  latents: jax.Array
  fitness: jax.Array


@flax.struct.dataclass
class RolloutBatch:
  observations: jax.Array
  latents: jax.Array
  raw_actions: jax.Array
  log_probs: jax.Array
  values: jax.Array
  rewards: jax.Array
  dones: jax.Array
  advantages: jax.Array
  returns: jax.Array
  segment_returns: jax.Array


@flax.struct.dataclass
class TrainingState:
  params: Any
  opt_state: optax.OptState
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: jax.Array
  updates: jax.Array


class FiLMBlock(nn.Module):
  features: int
  activation: Callable[[jax.Array], jax.Array] = nn.swish

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
  actor_hidden_sizes: Sequence[int]
  critic_hidden_sizes: Sequence[int]
  latent_hidden_size: int

  def _tower(
      self,
      prefix: str,
      obs: jax.Array,
      latent: jax.Array,
      hidden_sizes: Sequence[int],
  ) -> jax.Array:
    hidden = obs
    latent_projection = nn.Dense(
        self.latent_hidden_size,
        kernel_init=nn.initializers.orthogonal(1.0),
        bias_init=nn.initializers.zeros,
        name=f"{prefix}_latent_projection",
    )(latent)
    latent_projection = nn.tanh(latent_projection)
    for index, width in enumerate(hidden_sizes):
      hidden = FiLMBlock(width, name=f"{prefix}_film_block_{index}")(
          hidden, latent_projection
      )
    return hidden

  @nn.compact
  def __call__(
      self, obs: jax.Array, latent: jax.Array
  ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    actor_hidden = self._tower("actor", obs, latent, self.actor_hidden_sizes)
    critic_hidden = self._tower("critic", obs, latent, self.critic_hidden_sizes)

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


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    wrap_env: bool = True,
    madrona_backend: bool = False,
    augment_pixels: bool = False,
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[Callable[..., Any]] = None,
    learning_rate: float = 3e-4,
    entropy_cost: float = 1e-3,
    discounting: float = 0.99,
    unroll_length: int = 32,
    batch_size: int = 256,
    num_minibatches: int = 1,
    num_updates_per_batch: int = 4,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = True,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.2,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = 1.0,
    normalize_advantage: bool = True,
    network_factory: Any = None,
    seed: int = 0,
    num_evals: int = 0,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    progress_fn: Callable[[int, Dict[str, float]], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    num_checkpoints: int = 1,
    save_checkpoint_path: Optional[str] = None,
    policy_params_fn_checkpoints: Callable[..., None] = lambda *args: None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    algorithm: str = "EPO",
    latent_dim: int = 32,
    latent_hidden_size: int = 128,
    pool_size: int = 64,
    latent_scale: float = 1.0,
    elite_fraction: float = 0.25,
    mutation_std: float = 0.15,
    mutation_clip: float = 3.0,
    crossover_rate: float = 0.5,
    fitness_ema: float = 0.5,
    evolution_warmup_updates: int = 1,
    value_coef: float = 0.5,
):
  del (
      max_devices_per_host,
      num_minibatches,
      num_resets_per_eval,
      eval_env,
      num_eval_envs,
      deterministic_eval,
      restore_value_fn,
      algorithm,
  )
  if episode_length is None:
    raise ValueError("episode_length must be specified for EPO training")
  if madrona_backend or augment_pixels:
    raise NotImplementedError("EPO currently supports non-vision MJX training only")

  env = _maybe_wrap_env(
      environment,
      wrap_env=wrap_env,
      num_envs=num_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      wrap_env_fn=wrap_env_fn,
      randomization_fn=randomization_fn,
  )
  reset_fn = jax.jit(env.reset)
  key = jax.random.PRNGKey(seed)
  key, reset_key = jax.random.split(key)
  env_state = reset_fn(jax.random.split(reset_key, num_envs))
  if "proprioception" not in env_state.obs:
    raise ValueError("EPO expects a `proprioception` observation key")

  metric_keys = tuple(
      key
      for key in ("success_rate", "target_area_dynamic_width_scale")
      if key in env_state.info.get("episode_metrics", {})
  )

  actor_hidden_sizes = tuple(
      _extract_hidden_sizes(
          network_factory,
          "policy_hidden_layer_sizes",
          default=(128, 128, 128, 128),
      )
  )
  critic_hidden_sizes = tuple(
      _extract_hidden_sizes(
          network_factory,
          "value_hidden_layer_sizes",
          default=(256, 256, 256, 256, 256),
      )
  )

  observation_size = env_state.obs["proprioception"].shape[-1]
  model = LatentActorCritic(
      action_dim=environment.action_size,
      actor_hidden_sizes=actor_hidden_sizes,
      critic_hidden_sizes=critic_hidden_sizes,
      latent_hidden_size=latent_hidden_size,
  )
  optimizer = optax.adam(learning_rate=learning_rate)
  if max_grad_norm is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=learning_rate),
    )

  obs_spec = specs.Array((observation_size,), jnp.dtype("float32"))
  normalizer_params = running_statistics.init_state(obs_spec)

  key, params_key, latent_key = jax.random.split(key, 3)
  dummy_obs = jnp.zeros((1, observation_size), dtype=jnp.float32)
  dummy_latent = jnp.zeros((1, latent_dim), dtype=jnp.float32)
  init_params = model.init(params_key, dummy_obs, dummy_latent)["params"]
  train_state = TrainingState(
      params=init_params,
      opt_state=optimizer.init(init_params),
      normalizer_params=normalizer_params,
      env_steps=jnp.array(0, dtype=jnp.int32),
      updates=jnp.array(0, dtype=jnp.int32),
  )
  latent_pool = LatentPool(
      latents=jax.random.normal(latent_key, (pool_size, latent_dim), dtype=jnp.float32)
      * latent_scale,
      fitness=jnp.zeros((pool_size,), dtype=jnp.float32),
  )

  if restore_checkpoint_path is not None:
    restored = ocp.PyTreeCheckpointer().restore(restore_checkpoint_path)
    train_state, latent_pool = _restore_state(
        restored, train_state=train_state, latent_pool=latent_pool
    )
  if restore_params is not None:
    train_state, latent_pool = _restore_state(
        restore_params, train_state=train_state, latent_pool=latent_pool
    )

  sample_action, deterministic_action, update_step, collect_rollout = (
      _build_jitted_functions(
          env=env,
          model=model,
          optimizer=optimizer,
          rollout_length=unroll_length,
          normalize_observations=normalize_observations,
          reward_scaling=reward_scaling,
          clip_epsilon=clipping_epsilon,
          value_coef=value_coef,
          entropy_coef=entropy_cost,
          gae_lambda=gae_lambda,
          discounting=discounting,
          normalize_advantage=normalize_advantage,
          metric_keys=metric_keys,
      )
  )
  make_policy = _make_inference_fn_builder(
      model=model,
      deterministic_action=deterministic_action,
      sample_action=sample_action,
      normalize_observations=normalize_observations,
  )

  if num_timesteps == 0:
    params = _policy_state(train_state, latent_pool)
    return make_policy, params, {}

  np_rng = np.random.default_rng(seed)
  steps_per_update = int(num_envs * unroll_length * action_repeat)
  log_interval_steps = max(steps_per_update, int(training_metrics_steps or steps_per_update))
  checkpoint_interval = (
      max(steps_per_update, int(math.ceil(num_timesteps / max(num_checkpoints, 1))))
      if num_checkpoints > 0
      else None
  )
  eval_interval = (
      max(steps_per_update, int(math.ceil(num_timesteps / max(num_evals, 1))))
      if num_evals > 0
      else None
  )
  last_log_steps = int(jax.device_get(train_state.env_steps))
  last_checkpoint_steps = int(jax.device_get(train_state.env_steps))
  last_eval_steps = int(jax.device_get(train_state.env_steps))
  metrics = {}
  training_start = time.time()

  if num_evals > 0:
    policy_params_fn(
        int(jax.device_get(train_state.env_steps)),
        make_policy,
        _policy_state(train_state, latent_pool),
    )

  while int(jax.device_get(train_state.env_steps)) < num_timesteps:
    key, latent_key, rollout_key, evolve_key = jax.random.split(key, 4)
    latent_indices = jax.random.randint(
        latent_key, (num_envs,), minval=0, maxval=pool_size
    )
    rollout_latents = latent_pool.latents[latent_indices]
    env_state, batch, rollout_metrics = collect_rollout(
        train_state,
        env_state,
        rollout_latents,
        rollout_key,
    )
    train_state, batch_metrics = _update_model(
        train_state=train_state,
        batch=batch,
        update_step=update_step,
        np_rng=np_rng,
        minibatch_size=max(1, int(batch_size)),
        update_epochs=max(1, int(num_updates_per_batch)),
        normalize_observations=normalize_observations,
    )
    train_state = train_state.replace(
        env_steps=train_state.env_steps + steps_per_update,
        updates=train_state.updates + 1,
    )

    if int(jax.device_get(train_state.updates)) >= evolution_warmup_updates:
      latent_pool = evolve_latent_pool(
          latent_pool=latent_pool,
          latent_indices=latent_indices,
          segment_returns=batch.segment_returns,
          elite_fraction=elite_fraction,
          mutation_std=mutation_std,
          mutation_clip=mutation_clip,
          crossover_rate=crossover_rate,
          fitness_ema=fitness_ema,
          rng=evolve_key,
      )

    current_steps = int(jax.device_get(train_state.env_steps))
    should_log = (
        current_steps - last_log_steps >= log_interval_steps
        or current_steps >= num_timesteps
    )
    if should_log:
      metrics = {
          "rollout/mean_reward": float(jax.device_get(jnp.mean(batch.rewards))),
          "rollout/mean_segment_return": float(
              jax.device_get(jnp.mean(batch.segment_returns))
          ),
          "rollout/done_fraction": float(jax.device_get(jnp.mean(batch.dones))),
          "ppo/policy_loss": float(jax.device_get(batch_metrics["policy_loss"])),
          "ppo/value_loss": float(jax.device_get(batch_metrics["value_loss"])),
          "ppo/entropy": float(jax.device_get(batch_metrics["entropy"])),
          "ppo/grad_norm": float(jax.device_get(batch_metrics["grad_norm"])),
          "evo/best_fitness": float(jax.device_get(jnp.max(latent_pool.fitness))),
          "evo/mean_fitness": float(jax.device_get(jnp.mean(latent_pool.fitness))),
          "evo/best_latent_norm": float(
              jax.device_get(
                  jnp.linalg.norm(latent_pool.latents[jnp.argmax(latent_pool.fitness)])
              )
          ),
          "time/elapsed_sec": float(time.time() - training_start),
          **_metrics_to_python(rollout_metrics),
      }
      progress_fn(current_steps, metrics)
      last_log_steps = current_steps

    if (
        checkpoint_interval is not None
        and current_steps - last_checkpoint_steps >= checkpoint_interval
    ):
      params = _policy_state(train_state, latent_pool)
      policy_params_fn_checkpoints(current_steps, make_policy, params)
      if save_checkpoint_path is not None:
        _save_checkpoint(save_checkpoint_path, current_steps, params)
      last_checkpoint_steps = current_steps

    if eval_interval is not None and current_steps - last_eval_steps >= eval_interval:
      policy_params_fn(current_steps, make_policy, _policy_state(train_state, latent_pool))
      last_eval_steps = current_steps

  params = _policy_state(train_state, latent_pool)
  if checkpoint_interval is not None and last_checkpoint_steps != int(
      jax.device_get(train_state.env_steps)
  ):
    policy_params_fn_checkpoints(
        int(jax.device_get(train_state.env_steps)), make_policy, params
    )
    if save_checkpoint_path is not None:
      _save_checkpoint(save_checkpoint_path, int(jax.device_get(train_state.env_steps)), params)

  logging.info("Finished EPO training after %s env steps", train_state.env_steps)
  return make_policy, params, metrics


def evolve_latent_pool(
    latent_pool: LatentPool,
    latent_indices: jax.Array,
    segment_returns: jax.Array,
    elite_fraction: float,
    mutation_std: float,
    mutation_clip: float,
    crossover_rate: float,
    fitness_ema: float,
    rng: PRNGKey,
) -> LatentPool:
  pool_size, latent_dim = latent_pool.latents.shape
  elite_count = max(1, int(round(pool_size * elite_fraction)))
  counts = jnp.bincount(latent_indices, length=pool_size).astype(jnp.float32)
  sums = jnp.bincount(
      latent_indices,
      weights=jnp.asarray(segment_returns, dtype=jnp.float32),
      length=pool_size,
  ).astype(jnp.float32)
  observed = counts > 0
  batch_scores = sums / jnp.maximum(counts, 1.0)
  fitness = jnp.where(
      observed,
      fitness_ema * latent_pool.fitness + (1.0 - fitness_ema) * batch_scores,
      latent_pool.fitness,
  )

  elite_values, elite_indices = jax.lax.top_k(fitness, elite_count)
  elite_latents = latent_pool.latents[elite_indices]

  child_count = pool_size - elite_count
  if child_count <= 0:
    return LatentPool(latents=elite_latents, fitness=elite_values)

  key_a, key_b, key_mask, key_noise = jax.random.split(rng, 4)
  parent_a_indices = jax.random.randint(
      key_a, (child_count,), minval=0, maxval=elite_count
  )
  parent_b_indices = jax.random.randint(
      key_b, (child_count,), minval=0, maxval=elite_count
  )
  parent_a = elite_latents[parent_a_indices]
  parent_b = elite_latents[parent_b_indices]
  crossover_mask = jax.random.uniform(
      key_mask, (child_count, latent_dim)
  ) < crossover_rate
  children = jnp.where(crossover_mask, parent_a, parent_b)
  children = children + mutation_std * jax.random.normal(
      key_noise, children.shape, dtype=children.dtype
  )
  children = jnp.clip(children, -mutation_clip, mutation_clip)
  child_fitness = 0.5 * (
      elite_values[parent_a_indices] + elite_values[parent_b_indices]
  )
  return LatentPool(
      latents=jnp.concatenate([elite_latents, children], axis=0),
      fitness=jnp.concatenate([elite_values, child_fitness], axis=0),
  )


def _build_jitted_functions(
    env: envs.Env,
    model: LatentActorCritic,
    optimizer: optax.GradientTransformation,
    rollout_length: int,
    normalize_observations: bool,
    reward_scaling: float,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    gae_lambda: float,
    discounting: float,
    normalize_advantage: bool,
    metric_keys: Sequence[str],
):
  def preprocess_observation(obs, normalizer_params):
    if normalize_observations:
      return running_statistics.normalize(obs, normalizer_params)
    return obs

  def apply_model(params, obs, latents):
    return model.apply({"params": params}, obs, latents)

  @jax.jit
  def sample_action(params, obs, latents, rng):
    means, log_std, values = apply_model(params, obs, latents)
    noise = jax.random.normal(rng, means.shape, dtype=means.dtype)
    raw_actions = means + jnp.exp(log_std) * noise
    actions = jnp.tanh(raw_actions)
    log_probs = _squashed_gaussian_log_prob(raw_actions, means, log_std)
    return actions, raw_actions, log_probs, values

  @jax.jit
  def deterministic_action(params, obs, latents):
    means, _, values = apply_model(params, obs, latents)
    return jnp.tanh(means), values

  @jax.jit
  def update_step(
      train_state: TrainingState,
      observations: jax.Array,
      latents: jax.Array,
      raw_actions: jax.Array,
      old_log_probs: jax.Array,
      old_values: jax.Array,
      advantages: jax.Array,
      returns: jax.Array,
  ) -> Tuple[TrainingState, Mapping[str, jax.Array]]:
    if normalize_advantage:
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def loss_fn(params):
      means, log_std, values = apply_model(params, observations, latents)
      log_probs = _squashed_gaussian_log_prob(raw_actions, means, log_std)
      ratio = jnp.exp(log_probs - old_log_probs)
      clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
      policy_loss = -jnp.mean(
          jnp.minimum(ratio * advantages, clipped_ratio * advantages)
      )

      value_pred_clipped = old_values + jnp.clip(
          values - old_values, -clip_epsilon, clip_epsilon
      )
      value_losses = jnp.square(values - returns)
      value_losses_clipped = jnp.square(value_pred_clipped - returns)
      value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))

      entropy = jnp.mean(_gaussian_entropy(log_std))
      total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
      metrics = {
          "policy_loss": policy_loss,
          "value_loss": value_loss,
          "entropy": entropy,
      }
      return total_loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(train_state.params)
    updates, new_opt_state = optimizer.update(
        grads, train_state.opt_state, train_state.params
    )
    new_params = optax.apply_updates(train_state.params, updates)
    new_state = train_state.replace(params=new_params, opt_state=new_opt_state)
    metrics = dict(metrics)
    metrics["loss"] = loss
    metrics["grad_norm"] = optax.global_norm(grads)
    return new_state, metrics

  @jax.jit
  def collect_rollout(
      train_state: TrainingState,
      env_state,
      rollout_latents: jax.Array,
      rng: PRNGKey,
  ):
    def step_fn(carry, _):
      state, key = carry
      obs = state.obs["proprioception"]
      proc_obs = preprocess_observation(obs, train_state.normalizer_params)
      key, action_key = jax.random.split(key)
      actions, raw_actions, log_probs, values = sample_action(
          train_state.params, proc_obs, rollout_latents, action_key
      )
      next_state = env.step(state, actions)
      data = {
          "observations": obs,
          "latents": rollout_latents,
          "raw_actions": raw_actions,
          "log_probs": log_probs,
          "values": values,
          "rewards": next_state.reward * reward_scaling,
          "raw_rewards": next_state.reward,
          "dones": next_state.done.astype(jnp.float32),
          "episode_sum_reward": next_state.info["episode_metrics"]["sum_reward"],
          "episode_length": next_state.info["episode_metrics"]["length"],
      }
      for metric_key in metric_keys:
        data[metric_key] = next_state.info["episode_metrics"][metric_key]
      return (next_state, key), data

    (next_env_state, _), rollout = jax.lax.scan(
        step_fn, (env_state, rng), (), length=rollout_length
    )
    final_obs = preprocess_observation(
        next_env_state.obs["proprioception"], train_state.normalizer_params
    )
    _, _, last_values = apply_model(train_state.params, final_obs, rollout_latents)
    advantages, returns = _compute_gae(
        rewards=rollout["rewards"],
        values=rollout["values"],
        dones=rollout["dones"],
        last_values=last_values,
        gamma=discounting,
        gae_lambda=gae_lambda,
    )
    batch = RolloutBatch(
        observations=rollout["observations"],
        latents=rollout["latents"],
        raw_actions=rollout["raw_actions"],
        log_probs=rollout["log_probs"],
        values=rollout["values"],
        rewards=rollout["raw_rewards"],
        dones=rollout["dones"],
        advantages=advantages,
        returns=returns,
        segment_returns=jnp.sum(rollout["raw_rewards"], axis=0),
    )
    rollout_metrics = _summarize_rollout_metrics(rollout, metric_keys)
    return next_env_state, batch, rollout_metrics

  return (
      sample_action,
      deterministic_action,
      update_step,
      collect_rollout,
  )


def _update_model(
    train_state: TrainingState,
    batch: RolloutBatch,
    update_step,
    np_rng: np.random.Generator,
    minibatch_size: int,
    update_epochs: int,
    normalize_observations: bool,
) -> Tuple[TrainingState, Dict[str, float]]:
  flat_observations = batch.observations.reshape((-1, batch.observations.shape[-1]))
  flat = {
      "observations": flat_observations,
      "latents": batch.latents.reshape((-1, batch.latents.shape[-1])),
      "raw_actions": batch.raw_actions.reshape((-1, batch.raw_actions.shape[-1])),
      "log_probs": batch.log_probs.reshape((-1,)),
      "values": batch.values.reshape((-1,)),
      "advantages": batch.advantages.reshape((-1,)),
      "returns": batch.returns.reshape((-1,)),
  }
  if normalize_observations:
    normalized_observations = running_statistics.normalize(
        flat_observations, train_state.normalizer_params
    )
    new_normalizer_params = running_statistics.update(
        train_state.normalizer_params, flat_observations
    )
  else:
    normalized_observations = flat_observations
    new_normalizer_params = train_state.normalizer_params
  flat["observations"] = normalized_observations

  num_items = int(flat["observations"].shape[0])
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
          flat["observations"][indices],
          flat["latents"][indices],
          flat["raw_actions"][indices],
          flat["log_probs"][indices],
          flat["values"][indices],
          flat["advantages"][indices],
          flat["returns"][indices],
      )
      batch_metrics = jax.device_get(batch_metrics)
      for key in metrics:
        metrics[key] += float(batch_metrics[key])
      metric_steps += 1

  train_state = train_state.replace(normalizer_params=new_normalizer_params)
  for key in metrics:
    metrics[key] /= max(1, metric_steps)
  return train_state, metrics


def _maybe_wrap_env(
    environment: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: int,
    action_repeat: int,
    wrap_env_fn: Optional[Callable[[Any], Any]],
    randomization_fn: Optional[Callable[..., Any]],
):
  del num_envs
  if not wrap_env:
    return environment
  wrap_for_training = wrap_env_fn or envs.training.wrap
  return wrap_for_training(
      environment,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=randomization_fn,
  )


def _compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_values: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jax.Array, jax.Array]:
  def step_fn(carry, inputs):
    next_value, next_advantage = carry
    reward, value, done = inputs
    not_done = 1.0 - done
    delta = reward + gamma * next_value * not_done - value
    advantage = delta + gamma * gae_lambda * not_done * next_advantage
    return (value, advantage), advantage

  (_, _), advantages = jax.lax.scan(
      step_fn,
      (last_values, jnp.zeros_like(last_values)),
      (rewards, values, dones),
      reverse=True,
  )
  returns = advantages + values
  return advantages, returns


def _gaussian_entropy(log_std: jax.Array) -> jax.Array:
  return jnp.sum(log_std + 0.5 * (1.0 + LOG_TWO_PI), axis=-1)


def _squashed_gaussian_log_prob(
    raw_actions: jax.Array, means: jax.Array, log_std: jax.Array
) -> jax.Array:
  inv_std = jnp.exp(-log_std)
  centered = (raw_actions - means) * inv_std
  gaussian_log_prob = -0.5 * jnp.sum(
      centered * centered + 2.0 * log_std + LOG_TWO_PI, axis=-1
  )
  correction = jnp.sum(
      jnp.log(1.0 - jnp.tanh(raw_actions) ** 2 + EPS), axis=-1
  )
  return gaussian_log_prob - correction


def _extract_hidden_sizes(
    network_factory: Any, field_name: str, default: Sequence[int]
) -> Sequence[int]:
  if network_factory is None:
    return tuple(default)
  if isinstance(network_factory, Mapping) and field_name in network_factory:
    return tuple(int(x) for x in network_factory[field_name])
  if hasattr(network_factory, field_name):
    return tuple(int(x) for x in getattr(network_factory, field_name))
  return tuple(default)


def _policy_state(
    train_state: TrainingState, latent_pool: LatentPool
) -> Dict[str, Any]:
  best_index = jnp.argmax(latent_pool.fitness)
  return {
      "normalizer_params": train_state.normalizer_params,
      "network_params": train_state.params,
      "latent_pool": {
          "latents": latent_pool.latents,
          "fitness": latent_pool.fitness,
      },
      "best_latent": latent_pool.latents[best_index],
      "env_steps": train_state.env_steps,
      "updates": train_state.updates,
  }


def _make_inference_fn_builder(
    model: LatentActorCritic,
    deterministic_action,
    sample_action,
    normalize_observations: bool,
):
  def preprocess_observation(obs, normalizer_params):
    if normalize_observations:
      return running_statistics.normalize(obs, normalizer_params)
    return obs

  def make_policy(params, deterministic: bool = False):
    normalizer_params = params["normalizer_params"]
    network_params = params["network_params"]
    best_latent = jnp.asarray(params["best_latent"], dtype=jnp.float32)

    def policy(observation, key):
      obs = observation["proprioception"]
      proc_obs = preprocess_observation(obs, normalizer_params)
      latent = jnp.broadcast_to(best_latent, (proc_obs.shape[0], best_latent.shape[-1]))
      if deterministic:
        actions, _ = deterministic_action(network_params, proc_obs, latent)
      else:
        if getattr(key, "ndim", 0) > 1:
          means, log_std, _ = model.apply(
              {"params": network_params}, proc_obs, latent
          )
          noise = jax.vmap(
              lambda sample_key: jax.random.normal(
                  sample_key, (means.shape[-1],), dtype=means.dtype
              )
          )(key)
          raw_actions = means + jnp.exp(log_std) * noise
          actions = jnp.tanh(raw_actions)
        else:
          actions, _, _, _ = sample_action(network_params, proc_obs, latent, key)
      return actions, {}

    return policy

  return make_policy


def _restore_state(restored, train_state: TrainingState, latent_pool: LatentPool):
  if isinstance(restored, tuple):
    train_state = train_state.replace(
        normalizer_params=restored[0],
        params=restored[1],
    )
    return train_state, latent_pool
  latent_payload = restored.get("latent_pool", {})
  if "normalizer_params" in restored:
    train_state = train_state.replace(
        normalizer_params=restored["normalizer_params"],
        params=restored["network_params"],
        env_steps=jnp.asarray(restored.get("env_steps", train_state.env_steps)),
        updates=jnp.asarray(restored.get("updates", train_state.updates)),
    )
  elif "params" in restored:
    train_state = train_state.replace(
        normalizer_params=restored["params"][0],
        params=restored["params"][1],
        env_steps=jnp.asarray(restored.get("env_steps", train_state.env_steps)),
        updates=jnp.asarray(restored.get("updates", train_state.updates)),
    )
    latent_payload = restored.get("latent_pool", latent_payload)
  latent_pool = LatentPool(
      latents=jnp.asarray(latent_payload.get("latents", latent_pool.latents)),
      fitness=jnp.asarray(latent_payload.get("fitness", latent_pool.fitness)),
  )
  return train_state, latent_pool


def _summarize_rollout_metrics(
    rollout: Mapping[str, jax.Array], metric_keys: Sequence[str]
) -> Dict[str, jax.Array]:
  done_mask = rollout["dones"]
  done_count = jnp.maximum(jnp.sum(done_mask), 1.0)
  metrics = {
      "episode/sum_reward": jnp.sum(rollout["episode_sum_reward"] * done_mask)
      / done_count,
      "episode/length": jnp.sum(rollout["episode_length"] * done_mask) / done_count,
      # Live partial-episode progress for runs where no rollout chunk reaches done yet.
      "episode/current_length_mean": jnp.mean(rollout["episode_length"][-1]),
  }
  for metric_key in metric_keys:
    metrics[f"episode/{metric_key}"] = (
        jnp.sum(rollout[metric_key] * done_mask) / done_count
    )
  return metrics


def _metrics_to_python(metrics: Mapping[str, Any]) -> Dict[str, float]:
  return {key: float(np.asarray(jax.device_get(value))) for key, value in metrics.items()}


def _save_checkpoint(save_checkpoint_path: str, env_steps: int, params: Mapping[str, Any]):
  checkpointer = ocp.PyTreeCheckpointer()
  checkpointer.save(f"{save_checkpoint_path}/{env_steps}", params, force=True)
