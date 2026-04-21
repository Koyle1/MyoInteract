import importlib.util
import os
import sys
import unittest

import numpy as np


REPO_ROOT = "/Users/koyle/project/benchmark/MyoInteract"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MODULE_PATH = os.path.join(REPO_ROOT, "myosuite", "agents", "epo_jax.py")
MJX_MODULE_PATH = os.path.join(
    REPO_ROOT, "myosuite", "train", "myouser", "custom_epo", "train.py"
)


def _deps_available():
    required = ("jax", "flax", "optax", "omegaconf", "flatten_dict")
    return all(importlib.util.find_spec(name) is not None for name in required)


def _mjx_deps_available():
    required = ("jax", "flax", "optax", "brax", "orbax")
    return all(importlib.util.find_spec(name) is not None for name in required)


def _load_epo_module():
    spec = importlib.util.spec_from_file_location("epo_jax_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_mjx_epo_module():
    spec = importlib.util.spec_from_file_location("epo_mjx_test_module", MJX_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(_deps_available(), "EPO JAX dependencies are not installed")
class TestEPOJax(unittest.TestCase):
    def test_evolve_latent_pool_preserves_best_elite(self):
        epo_jax = _load_epo_module()
        LatentPool = epo_jax.LatentPool
        evolve_latent_pool = epo_jax.evolve_latent_pool

        pool = LatentPool(
            latents=np.array(
                [
                    [5.0, 5.0],
                    [1.0, 1.0],
                    [-1.0, -1.0],
                    [0.5, -0.5],
                ],
                dtype=np.float32,
            ),
            fitness=np.array([0.1, 5.0, 1.0, 2.0], dtype=np.float32),
        )
        next_pool = evolve_latent_pool(
            latent_pool=pool,
            latent_indices=np.array([1, 1, 3, 2], dtype=np.int32),
            segment_returns=np.array([10.0, 8.0, 1.0, -2.0], dtype=np.float32),
            elite_fraction=0.25,
            mutation_std=0.0,
            mutation_clip=10.0,
            crossover_rate=1.0,
            fitness_ema=0.0,
            np_rng=np.random.default_rng(0),
        )

        np.testing.assert_allclose(next_pool.latents[0], np.array([1.0, 1.0], dtype=np.float32))
        self.assertGreaterEqual(next_pool.fitness[0], np.max(next_pool.fitness[1:]))

    def test_hidden_size_string_is_parsed(self):
        epo_jax = _load_epo_module()

        self.assertEqual(epo_jax._as_sequence("64, 128,256"), [64, 128, 256])


@unittest.skipUnless(_mjx_deps_available(), "MJX EPO JAX dependencies are not installed")
class TestEPOJaxMjx(unittest.TestCase):
    def test_mjx_evolve_latent_pool_preserves_best_elite(self):
        epo_jax = _load_mjx_epo_module()
        LatentPool = epo_jax.LatentPool
        evolve_latent_pool = epo_jax.evolve_latent_pool

        pool = LatentPool(
            latents=np.array(
                [
                    [5.0, 5.0],
                    [1.0, 1.0],
                    [-1.0, -1.0],
                    [0.5, -0.5],
                ],
                dtype=np.float32,
            ),
            fitness=np.array([0.1, 5.0, 1.0, 2.0], dtype=np.float32),
        )
        next_pool = evolve_latent_pool(
            latent_pool=pool,
            latent_indices=np.array([1, 1, 3, 2], dtype=np.int32),
            segment_returns=np.array([10.0, 8.0, 1.0, -2.0], dtype=np.float32),
            elite_fraction=0.25,
            mutation_std=0.0,
            mutation_clip=10.0,
            crossover_rate=1.0,
            fitness_ema=0.0,
            rng=epo_jax.jax.random.PRNGKey(0),
        )

        np.testing.assert_allclose(
            np.asarray(next_pool.latents[0]), np.array([1.0, 1.0], dtype=np.float32)
        )
        self.assertGreaterEqual(
            float(np.asarray(next_pool.fitness[0])),
            float(np.max(np.asarray(next_pool.fitness[1:]))),
        )

    def test_restore_state_accepts_legacy_tuple(self):
        epo_jax = _load_mjx_epo_module()

        train_state = epo_jax.TrainingState(
            params={"weights": np.array([0.0], dtype=np.float32)},
            opt_state=(),
            normalizer_params=np.array([0.0], dtype=np.float32),
            env_steps=np.array(0, dtype=np.int32),
            updates=np.array(0, dtype=np.int32),
        )
        latent_pool = epo_jax.LatentPool(
            latents=np.zeros((2, 2), dtype=np.float32),
            fitness=np.zeros((2,), dtype=np.float32),
        )
        restored_state, restored_pool = epo_jax._restore_state(
            (np.array([1.0], dtype=np.float32), {"weights": np.array([2.0], dtype=np.float32)}),
            train_state=train_state,
            latent_pool=latent_pool,
        )

        np.testing.assert_allclose(restored_state.normalizer_params, np.array([1.0], dtype=np.float32))
        np.testing.assert_allclose(restored_state.params["weights"], np.array([2.0], dtype=np.float32))
        np.testing.assert_allclose(restored_pool.latents, latent_pool.latents)


if __name__ == "__main__":
    unittest.main()
