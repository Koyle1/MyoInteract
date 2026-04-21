import numpy as np


class EvolutionaryLatentPool:

  def __init__(
      self,
      latent_dim,
      population_size,
      elite_fraction,
      init_std,
      mutation_std,
      seed,
  ):
    self.latent_dim = int(latent_dim)
    self.population_size = int(population_size)
    self.elite_fraction = float(elite_fraction)
    self.init_std = float(init_std)
    self.mutation_std = float(mutation_std)
    self.rng = np.random.default_rng(int(seed))

    self.population = np.zeros(
        (self.population_size, self.latent_dim), np.float32)
    self.fitness_sum = np.zeros(self.population_size, np.float32)
    self.fitness_count = np.zeros(self.population_size, np.int32)
    self.active_count = np.zeros(self.population_size, np.int32)
    self.next_slot = 0
    self.episodes = 0
    self.last_return = 0.0
    self.last_length = 0.0

  def assign(self, count, exploit=False):
    count = int(count)
    indices = np.zeros(count, np.int32)
    latents = np.zeros((count, self.latent_dim), np.float32)
    for i in range(count):
      index, latent = self._assign_one(exploit=exploit)
      indices[i] = index
      latents[i] = latent
      if not exploit:
        self.active_count[index] += 1
    return indices, latents

  def complete(self, indices, returns, lengths):
    if len(indices) == 0:
      return
    indices = np.asarray(indices, np.int32)
    returns = np.asarray(returns, np.float32)
    lengths = np.asarray(lengths, np.float32)
    valid = (indices >= 0)
    if not valid.any():
      return
    indices = indices[valid]
    returns = returns[valid]
    lengths = lengths[valid]
    np.add.at(self.fitness_sum, indices, returns)
    np.add.at(self.fitness_count, indices, 1)
    np.add.at(self.active_count, indices, -1)
    self.active_count = np.maximum(self.active_count, 0)
    self.episodes += len(indices)
    self.last_return = float(returns.mean())
    self.last_length = float(lengths.mean())

  def metrics(self):
    scores = self._scores()
    mask = np.isfinite(scores)
    best = float(scores[mask].max()) if mask.any() else 0.0
    mean = float(scores[mask].mean()) if mask.any() else 0.0
    return {
        'evo/best_fitness': np.float32(best),
        'evo/mean_fitness': np.float32(mean),
        'evo/evaluated': np.float32(mask.sum()),
        'evo/filled': np.float32(self.next_slot),
        'evo/active': np.float32((self.active_count > 0).sum()),
        'evo/last_episode_return': np.float32(self.last_return),
        'evo/last_episode_length': np.float32(self.last_length),
    }

  def save(self):
    return {
        'population': self.population.copy(),
        'fitness_sum': self.fitness_sum.copy(),
        'fitness_count': self.fitness_count.copy(),
        'active_count': self.active_count.copy(),
        'next_slot': int(self.next_slot),
        'episodes': int(self.episodes),
        'last_return': float(self.last_return),
        'last_length': float(self.last_length),
        'rng_state': self.rng.bit_generator.state,
    }

  def load(self, data):
    self.population[...] = data['population']
    self.fitness_sum[...] = data['fitness_sum']
    self.fitness_count[...] = data['fitness_count']
    self.active_count.fill(0)
    self.next_slot = int(data['next_slot'])
    self.episodes = int(data['episodes'])
    self.last_return = float(data.get('last_return', 0.0))
    self.last_length = float(data.get('last_length', 0.0))
    self.rng.bit_generator.state = data['rng_state']

  def best(self):
    index = self._best_index()
    return index, self.population[index].copy()

  def _assign_one(self, exploit=False):
    if self.next_slot == 0:
      self.population[0] = self._sample_noise(self.init_std)
      self.next_slot = 1
    if exploit:
      index = self._best_index()
      return index, self.population[index].copy()
    if self.next_slot < self.population_size:
      index = self.next_slot
      self.population[index] = self._sample_noise(self.init_std)
      self.next_slot += 1
      self.fitness_sum[index] = 0.0
      self.fitness_count[index] = 0
      return index, self.population[index].copy()

    inactive = np.flatnonzero(self.active_count == 0)
    if len(inactive) == 0:
      index = self._best_index()
      return index, self.population[index].copy()

    elite = self._elite_indices()
    replace = np.array(
        [idx for idx in inactive.tolist() if idx not in set(elite.tolist())],
        np.int32,
    )
    if len(replace) == 0:
      replace = inactive
    index = int(self.rng.choice(replace))
    parent = int(self.rng.choice(elite))
    self.population[index] = (
        self.population[parent] + self._sample_noise(self.mutation_std))
    self.fitness_sum[index] = 0.0
    self.fitness_count[index] = 0
    return index, self.population[index].copy()

  def _best_index(self):
    scores = self._scores()
    if np.isfinite(scores).any():
      return int(np.nanargmax(scores))
    return 0

  def _elite_indices(self):
    filled = max(self.next_slot, 1)
    scores = self._scores()[:filled]
    elite_size = max(1, int(np.ceil(filled * self.elite_fraction)))
    if np.isfinite(scores).any():
      order = np.argsort(scores)
      return order[-elite_size:].astype(np.int32)
    return np.arange(min(filled, elite_size), dtype=np.int32)

  def _scores(self):
    scores = np.full(self.population_size, -np.inf, np.float32)
    mask = self.fitness_count > 0
    scores[mask] = self.fitness_sum[mask] / self.fitness_count[mask]
    return scores

  def _sample_noise(self, scale):
    noise = self.rng.standard_normal(self.latent_dim)
    return np.asarray(scale * noise, np.float32)
