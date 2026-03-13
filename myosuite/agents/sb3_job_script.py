""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Cameron Berg (cameronberg@fb.com), Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

"""
This is a job script for running SB3 on myosuite tasks.
"""

import os
import sys
import json
import myosuite
from omegaconf import OmegaConf

IS_WnB_enabled = False
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    IS_WnB_enabled = True
except ImportError as e:
    pass 

def train_loop(job_data) -> None:
    algo = job_data.algorithm
    if algo == 'Dreamer':
        train_loop_dreamer(job_data)
        return

    # Lazy SB3 imports so Dreamer can run without stable-baselines3 installed.
    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecNormalize
    from in_callbacks import InfoCallback, FallbackCheckpoint, EvalCallback

    config = {
            "policy_type": job_data.policy,
            "total_timesteps": job_data.total_timesteps,
            "env_name": job_data.env,
    }
    if IS_WnB_enabled:
        run = wandb.init(
            project="sb3_hand",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        tensorboard_log = f"wandb/{run.id}"
    else:
        tensorboard_log = None
    
    log = configure(f'results_{job_data.env}')
    # Create the vectorized environment and normalize ob
    env = make_vec_env(job_data.env, n_envs=job_data.n_env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    eval_env = make_vec_env(job_data.env, n_envs=job_data.n_eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    if algo == 'PPO':
        # Load activation function from config
        policy_kwargs = OmegaConf.to_container(job_data.policy_kwargs, resolve=True)

        model = PPO(job_data.policy, env,  verbose=1,
                    learning_rate=job_data.learning_rate,
                    batch_size=job_data.batch_size,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tensorboard_log,
                    gamma=job_data.gamma, **job_data.alg_hyper_params)
    elif algo == 'SAC':
        model = SAC(job_data.policy, env,
                    learning_rate=job_data.learning_rate,
                    buffer_size=job_data.buffer_size,
                    learning_starts=job_data.learning_starts, 
                    batch_size=job_data.batch_size, 
                    tau=job_data.tau, 
                    tensorboard_log=tensorboard_log,
                    gamma=job_data.gamma, **job_data.alg_hyper_params)
    else:
        raise ValueError(f"Unsupported algorithm '{algo}'")

    if job_data.job_name =="checkpoint.pt":
        foldername = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"baseline_SB3/myoChal24/{job_data.env}")
        file_path = os.path.join(foldername, job_data.job_name)
        if os.path.isfile(file_path):
            print("Loading weights from checkpoint")
            model.policy.load_state_dict(torch.load(file_path))
        else:
            raise FileNotFoundError(f"No file found at the specified path: {file_path}. See https://github.com/MyoHub/myosuite/blob/dev/myosuite/agents/README.md to download one.")
    else:
        print("No checkpoint loaded, training starts.")

    if IS_WnB_enabled:
        callback = [WandbCallback(
                model_save_path=f"models/{run.id}",
                verbose=2,
            )]
    else:
        callback = []

    callback += [EvalCallback(job_data.eval_freq, eval_env)]
    callback += [InfoCallback()]
    callback += [FallbackCheckpoint(job_data.restore_checkpoint_freq)]
    callback += [CheckpointCallback(save_freq=job_data.save_freq, save_path=f'logs/',
                                            name_prefix='rl_models')]

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
    )

    model.set_logger(log)

    model.save(f"{job_data.env}_"+algo+"_model")
    env.save(f'{job_data.env}_'+algo+'_env')

    if IS_WnB_enabled:
        run.finish()


def train_loop_dreamer(job_data) -> None:
    """Run Dreamer training through the vendored DreamerV3 runner."""
    agents_dir = os.path.dirname(os.path.realpath(__file__))
    dreamer_dir = os.path.join(agents_dir, "dreamer")
    for path in (agents_dir, dreamer_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        from dreamer import main as dreamer_main
    except ImportError as exc:
        raise ImportError(
            "Dreamer dependencies are missing. Install the Dreamer stack "
            "(jax, chex, optax, ninjax, portal, elements/playground) to run "
            "algorithm=Dreamer."
        ) from exc

    logdir = f"results_dreamer_{job_data.env}"
    args = [
        "--configs", "defaults",
        "--script", "train",
        "--task", f"myo_{job_data.env}",
        "--logdir", logdir,
        "--seed", str(int(job_data.seed)),
        "--batch_size", str(int(getattr(job_data, "batch_size", 16))),
        "--batch_length", str(int(getattr(job_data, "batch_length", 64))),
        "--run.steps", str(int(getattr(job_data, "total_timesteps", 15_000_000))),
        "--run.envs", str(int(getattr(job_data, "n_env", 8))),
        "--run.eval_envs", str(int(getattr(job_data, "n_eval_env", 2))),
        "--run.train_ratio", str(float(getattr(job_data, "train_ratio", 32))),
        "--replay.size", str(float(getattr(job_data, "replay_size", 5e6))),
    ]

    jax_platform = getattr(job_data, "jax_platform", None)
    if jax_platform:
        args.extend(["--jax.platform", str(jax_platform)])
    jax_dtype = getattr(job_data, "jax_compute_dtype", None)
    if jax_dtype:
        args.extend(["--jax.compute_dtype", str(jax_dtype)])

    _append_nested_flags(args, "agent.opt", _as_plain_dict(job_data, "dreamer_opt"))
    _append_nested_flags(args, "agent.dyn", _as_plain_dict(job_data, "dreamer_dyn"))
    _append_nested_flags(args, "agent.loss_scales", _as_plain_dict(job_data, "loss_scales"))
    args = [str(arg) for arg in args if arg is not None]

    print("Launching Dreamer with args:")
    print(" ".join(args))
    dreamer_main.main(args)


def _as_plain_dict(job_data, key):
    if hasattr(job_data, "get"):
        value = job_data.get(key, None)
    else:
        value = getattr(job_data, key, None)
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return value
    return {}


def _append_nested_flags(args, prefix, mapping):
    for key, value in mapping.items():
        if value is None:
            continue
        flag = f"{prefix}.{key}"
        if isinstance(value, dict):
            _append_nested_flags(args, flag, value)
        else:
            args.extend([f"--{flag}", _format_flag_value(value)])


def _format_flag_value(value):
    if isinstance(value, bool):
        # elements.Flags expects Python-style bool tokens.
        return "True" if value else "False"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return str(value)
