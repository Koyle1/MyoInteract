import functools
from dataclasses import asdict

import elements
import embodied
import jax
import numpy as np
from ml_collections import config_dict

from myosuite.envs.myo.myouser.base import get_default_config
from myosuite.envs.myo.myouser.myouser_universal import (
    LIST_CONFIGS,
    UniversalEnvConfig,
    MyoUserUniversal,
)


class MyoUniversal(embodied.Env):

  def __init__(self, task='default', obs_key='vector', act_key='action', seed=0):
    self._obs_key = obs_key
    self._act_key = act_key
    self._seed = int(seed)
    self._done = True
    self._step = 0

    preset = self._parse_task(task)
    config = self._make_config(preset)
    self._env = MyoUserUniversal(config)
    self._episode_length = int(
        self._env._config.task_config.max_duration / self._env._config.ctrl_dt
    )

    self._jit_reset = jax.jit(self._env.reset)
    self._jit_step = jax.jit(self._env.step)
    self._rng = jax.random.PRNGKey(self._seed)
    self._state = self._jit_reset(self._next_rng())

  @property
  def env(self):
    return self._env

  @functools.cached_property
  def obs_space(self):
    obs = np.asarray(jax.device_get(self._state.obs), dtype=np.float32)
    return {
        self._obs_key: elements.Space(np.float32, obs.shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    low = np.zeros((self._env.action_size,), dtype=np.float32)
    high = np.ones((self._env.action_size,), dtype=np.float32)
    return {
        self._act_key: elements.Space(np.float32, low.shape, low, high),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      self._step = 0
      self._state = self._jit_reset(self._next_rng())
      return self._obs(
          self._state.obs, 0.0, is_first=True, is_terminal=False, is_last=False
      )

    env_action = np.asarray(action[self._act_key], dtype=np.float32)
    self._state = self._jit_step(self._state, env_action)
    self._step += 1

    terminated = bool(np.asarray(jax.device_get(self._state.done)))
    truncated = bool(self._episode_length and self._step >= self._episode_length)
    self._done = terminated or truncated

    return self._obs(
        self._state.obs,
        self._state.reward,
        is_first=False,
        is_last=self._done,
        is_terminal=terminated,
    )

  def close(self):
    pass

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False
  ):
    obs = np.asarray(jax.device_get(obs), dtype=np.float32)
    return {
        self._obs_key: obs,
        'reward': np.float32(np.asarray(jax.device_get(reward), dtype=np.float32)),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
    }

  def _next_rng(self):
    self._rng, reset_rng = jax.random.split(self._rng)
    return reset_rng

  @staticmethod
  def _parse_task(task):
    if task == 'universal':
      return 'default'
    if task.startswith('universal_'):
      return task[len('universal_'):]
    return task

  @staticmethod
  def _preset_ctor(name):
    presets = {preset_name: ctor for _, preset_name, ctor in LIST_CONFIGS}
    if name not in presets:
      available = ', '.join(sorted(presets))
      raise KeyError(f'Unknown universal preset {name!r}. Available: {available}')
    return presets[name]

  @classmethod
  def _make_config(cls, preset):
    config = UniversalEnvConfig()
    config.task_config.targets = cls._preset_ctor(preset)()
    full = get_default_config()
    full.model_path = config.model_path
    full.ctrl_dt = config.ctrl_dt
    full.sim_dt = config.sim_dt
    full.eval_mode = config.eval_mode
    full.muscle_config = config_dict.create(**asdict(config.muscle_config))
    full.task_config = config_dict.create(**asdict(config.task_config))
    return full
