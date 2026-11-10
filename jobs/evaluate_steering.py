#!/usr/bin/env python3
"""Collect balanced steering trials; --dry-run needs only Python's standard library.

The registered task must implement prepare_eval_rollout(rng, steering_ids,
path_length), returning five, and eval_reset(rng, eval_id), selecting that
condition. Every state.info must contain steering_id, path_length, path_width,
ee_pos, and success. Geometry metadata must describe the actual corridor.
"""

import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs=5, type=float, default=[5, 10, 15, 20, 30])
    parser.add_argument("--trials-per-id", type=int, default=20)
    parser.add_argument("--path-length", type=float, default=0.20, help="Metres")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-name")
    parser.add_argument("--registration-module", help="Import to register your steering task")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in args.ids) or len(set(args.ids)) != 5:
        parser.error("--ids must contain five distinct positive finite numbers")
    if not math.isfinite(args.path_length) or args.path_length <= 0:
        parser.error("--path-length must be positive and finite")
    if args.trials_per_id < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Trial count and --max-steps must be positive")
    if not 0 <= args.seed <= 2**32 - 5 * args.trials_per_id:
        parser.error("Trial seeds must fit in unsigned 32-bit integers")
    if not args.dry_run and args.checkpoint is None:
        parser.error("--checkpoint is required unless --dry-run is used")
    return args


def make_schedule(args):
    return [dict(trial_id=f"trial_{i + 1:06d}", condition_index=i % 5,
                 repetition=i // 5 + 1, steering_id=args.ids[i % 5],
                 path_length=args.path_length,
                 path_width=args.path_length / args.ids[i % 5], seed=args.seed + i)
            for i in range(5 * args.trials_per_id)]


def write_job(args, schedule):
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "schedule.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(schedule[0]))
        writer.writeheader()
        writer.writerows(schedule)
    job = dict(vars(args), total_trials=len(schedule), units="metres",
               timing="duration_s is reset-to-terminal/truncation simulation duration")
    (output / "job.json").write_text(json.dumps(job, indent=2, default=str) + "\n")
    return output


def load_task(args, output):
    # Import simulation dependencies only for real evaluation.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import myosuite  # noqa: F401; registers built-in environments
    from etils import epath
    from ml_collections import ConfigDict
    from myosuite.train.utils.train import train_or_load_checkpoint
    from myosuite.envs.myo.myouser.base import MyoUserBase

    if args.registration_module:
        importlib.import_module(args.registration_module)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    config_path = args.config.expanduser().resolve() if args.config else (
        checkpoint / "config.json" if checkpoint.name == "checkpoints"
        else checkpoint.parent / "config.json")
    config = ConfigDict(json.loads(config_path.read_text()))
    if config.env.get("vision", {}).get("enabled", False):
        raise ValueError("This job supports non-vision steering evaluation only")
    env_name = args.env_name or config.env.env_name
    env, make_policy, params = train_or_load_checkpoint(
        env_name, config, eval_mode=True, logdir=epath.Path(str(output)),
        checkpoint_path=str(checkpoint), seed=config.run.seed)
    if getattr(env, "vision", False):
        raise ValueError("This job supports non-vision steering evaluation only")
    for method in ("prepare_eval_rollout", "eval_reset"):
        if getattr(type(env), method, None) is getattr(MyoUserBase, method):
            raise ValueError(f"Task must override {method} to configure steering conditions")
    n_conditions = env.prepare_eval_rollout(
        jax.random.PRNGKey(args.seed), steering_ids=jax.numpy.array(args.ids),
        path_length=args.path_length)
    if n_conditions != 5:
        raise ValueError(f"Task must prepare exactly five conditions, got {n_conditions}")
    max_steps = args.max_steps or math.ceil(
        float(config.env.task_config.max_duration) / float(config.env.ctrl_dt))
    if max_steps < 1:
        raise ValueError("Evaluation horizon must contain at least one control step")
    return env, make_policy(params, deterministic=True), max_steps


def collect(args, schedule, output):
    import jax
    import numpy as np

    env, policy, max_steps = load_task(args, output)
    reset = jax.jit(jax.vmap(env.eval_reset))
    step = jax.jit(jax.vmap(env.step))
    infer = jax.jit(jax.vmap(policy))
    required_info = ("steering_id", "path_length", "path_width", "ee_pos", "success")
    columns = ("qpos", "qvel", "ee_pos", "time", "reward", "done", "success")

    def snapshot(state):
        missing = set(required_info) - set(state.info)
        if missing:
            raise ValueError(f"Task state.info is missing steering fields: {sorted(missing)}")
        return jax.device_get(dict(qpos=state.data.qpos, qvel=state.data.qvel,
                                  ee_pos=state.info["ee_pos"], time=state.data.time,
                                  reward=state.reward, done=state.done,
                                  success=state.info["success"]))

    def check_geometry(state, expected):
        for name in ("steering_id", "path_length", "path_width"):
            actual = np.asarray(jax.device_get(state.info[name]))
            values = np.asarray([row[name] for row in expected])
            if actual.shape != (5,) or not np.allclose(actual, values, rtol=1e-5, atol=1e-7):
                raise ValueError(f"Task {name} does not match requested conditions: {actual}")
        if np.asarray(jax.device_get(state.info["ee_pos"])).shape not in ((5, 2), (5, 3)):
            raise ValueError("Task ee_pos must be a position vector for each of five trials")
        for name in ("success",):
            if np.asarray(jax.device_get(state.info[name])).shape != (5,):
                raise ValueError(f"Task {name} must be scalar for each trial")

    trajectories = output / "trajectories"
    trajectories.mkdir()
    counts = {str(value): dict(trials=0, success=0, failure=0, timeout=0) for value in args.ids}
    with (output / "results.jsonl").open("x") as manifest:
        for offset in range(0, len(schedule), 5):
            batch = schedule[offset:offset + 5]
            keys = jax.numpy.stack([jax.random.PRNGKey(row["seed"]) for row in batch])
            reset_keys, policy_keys = jax.vmap(lambda key: jax.random.split(key, 2))(keys).transpose(1, 0, 2)
            state = reset(reset_keys, jax.numpy.arange(5, dtype=jax.numpy.int32))
            first = snapshot(state)
            check_geometry(state, batch)
            records = [{name: [np.asarray(first[name][i])] for name in columns} for i in range(5)]
            actions = [[] for _ in batch]
            active = ~np.asarray(first["done"], dtype=bool)
            for step_index in range(max_steps):
                if not active.any():
                    break
                action_keys = jax.vmap(lambda key: jax.random.fold_in(key, step_index))(policy_keys)
                controls, _ = infer(state.obs, action_keys)
                state = step(state, controls)
                current = snapshot(state)
                control_values = np.asarray(jax.device_get(controls))
                for i in np.flatnonzero(active):
                    actions[i].append(control_values[i])
                    for name in columns:
                        records[i][name].append(np.asarray(current[name][i]))
                active &= ~np.asarray(current["done"], dtype=bool)
            for i, row in enumerate(batch):
                arrays = {name: np.asarray(values) for name, values in records[i].items()}
                arrays["time"] = arrays["time"] - arrays["time"][0]
                arrays["action"] = np.asarray(actions[i])
                timed_out = not bool(arrays["done"][-1])
                succeeded = bool(arrays["success"][-1]) and not timed_out
                status = "timeout" if timed_out else "success" if succeeded else "failure"
                relative_path = f"trajectories/{row['trial_id']}.npz"
                np.savez_compressed(output / relative_path, **arrays)
                result = dict(row, trajectory=relative_path, steps=len(actions[i]),
                              duration_s=float(arrays["time"][-1]), success=succeeded,
                              timeout=timed_out, status=status)
                manifest.write(json.dumps(result) + "\n")
                counts[str(row["steering_id"])]["trials"] += 1
                counts[str(row["steering_id"])][status] += 1
            manifest.flush()
            print(f"Collected {offset + 5}/{len(schedule)} trials", flush=True)
    summary = dict(total_trials=sum(c["trials"] for c in counts.values()), by_id=counts)
    if any(c["trials"] != args.trials_per_id for c in counts.values()):
        raise RuntimeError("Collected trial counts do not match the requested balanced schedule")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    args = parse_args()
    schedule = make_schedule(args)
    output = write_job(args, schedule)
    if args.dry_run:
        print(f"Scheduled {len(schedule)} trials: {args.trials_per_id} per ID; {output}")
    else:
        summary = collect(args, schedule, output)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
