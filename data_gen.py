import argparse
import os
import random
from typing import Any, Tuple

import numpy as np
import crafter


try:
    import stable_baselines3 as sb3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecTransposeImage

    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=True):

        return [self.action_space.sample()], None


class HeuristicAgent:
    """
    Blind (image-only) heuristic for Crafter:
    - Early steps: favor movement + 'interact'
    - Mid game: bias toward 'use'/'craft' occasionally
    - Periodic 'noop' to avoid getting stuck
    """

    def __init__(self, action_space_size=17):
        self.n = action_space_size
        self.step = 0

    def predict(self, observation, deterministic=True):
        self.step += 1

        p = np.full(self.n, 0.02, dtype=np.float64)

        if self.step < 150:

            p[1:5] += 0.08
            p[5] += 0.20
        elif self.step < 600:

            p[1:5] += 0.06
            p[5] += 0.12
            p[11] += 0.07
            p[12] += 0.05
            p[14] += 0.05
        else:

            p[1:5] += 0.04
            p[5] += 0.10
            p[11] += 0.08
            p[12] += 0.06
            p[14] += 0.06
            p[15] += 0.04

        if self.step % 120 == 0:
            p[6] += 0.25
        if self.step % 200 == 0:
            p[0] += 0.15

        p /= p.sum()
        a = np.random.choice(self.n, p=p)
        return [int(a)], None


def make_agent(
    kind: str, env, model_path: str, ppo_train_steps: int, ppo_n_envs: int, seed: int
):
    if kind == "random":
        return RandomAgent(env.action_space)
    if kind == "heuristic":
        return HeuristicAgent(env.action_space.n)
    if kind == "ppo":
        if not SB3_AVAILABLE:
            raise RuntimeError(
                "stable_baselines3 not installed but agent=ppo was requested."
            )

        if (
            model_path
            and os.path.exists(model_path)
            and (ppo_train_steps is None or ppo_train_steps <= 0)
        ):
            print(f"[PPO] Loading model from: {model_path}")
            return sb3.PPO.load(model_path)

        print(
            f"[PPO] Training new model for {ppo_train_steps} timesteps with {ppo_n_envs} envs..."
        )

        def _env_fn():
            e = crafter.Env()
            return e

        vec = make_vec_env(_env_fn, n_envs=max(1, ppo_n_envs), seed=seed)
        vec = VecTransposeImage(vec)

        model = sb3.PPO("CnnPolicy", vec, verbose=1)
        model.learn(total_timesteps=int(ppo_train_steps))
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            print(f"[PPO] Saved model to: {model_path}")
        return model
    raise ValueError(f"Unknown agent: {kind}")


def reset_env(env, seed=None) -> Any:
    try:

        obs, _info = env.reset(seed=seed)
    except TypeError:

        if seed is not None:
            try:
                env.seed(seed)
            except Exception:
                pass
        obs = env.reset()
    return obs


def step_env(env, action) -> Tuple[Any, float, bool, dict]:

    try:

        obs, r, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        return obs, float(r), done, info
    except ValueError:

        obs, r, done, info = env.step(action)
        return obs, float(r), bool(done), info


def collect_episode(env, agent, max_steps: int) -> dict:
    obs = reset_env(env)
    done = False
    t = 0

    obs_buf = []
    act_buf = []
    rew_buf = []
    done_buf = []

    while not done and t < max_steps:

        frame = np.asarray(obs, dtype=np.uint8)
        obs_buf.append(frame)

        action, _ = agent.predict(obs, deterministic=False)

        if isinstance(action, (list, np.ndarray)):
            action = int(action[0])

        obs, reward, done, _info = step_env(env, action)

        act_buf.append(np.int32(action))
        rew_buf.append(np.float32(reward))
        done_buf.append(np.bool_(done))

        t += 1

    return {
        "observations": np.stack(obs_buf).astype(np.uint8),
        "actions": np.asarray(act_buf, dtype=np.int32),
        "rewards": np.asarray(rew_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=bool),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Crafter data generator (random | ppo | heuristic)"
    )
    parser.add_argument(
        "--agent", choices=["random", "ppo", "heuristic"], default="random"
    )
    parser.add_argument("--outdir", default="data/crafter_episodes")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ppo_model_path", default="checkpoints/ppo_crafter.zip")
    parser.add_argument(
        "--ppo_train_steps",
        type=int,
        default=0,
        help="If > 0, train PPO for this many steps. If 0, try to load from --ppo_model_path.",
    )
    parser.add_argument("--ppo_n_envs", type=int, default=8)

    parser.add_argument(
        "--record_stats",
        action="store_true",
        help="Use crafter.Recorder(save_stats=True)",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Use crafter.Recorder(save_video=True)",
    )
    parser.add_argument("--record_dir", default="logdir/crafter_recorder")

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except Exception:
        pass

    os.makedirs(os.path.join(args.outdir, args.agent) , exist_ok=True)

    ppo_model = None
    if args.agent == "ppo":
        ppo_model = make_agent(
            "ppo",
            env=crafter.Env(),
            model_path=args.ppo_model_path,
            ppo_train_steps=args.ppo_train_steps,
            ppo_n_envs=args.ppo_n_envs,
            seed=args.seed,
        )

    base_env = crafter.Env()
    if args.record_stats or args.record_video:
        base_env = crafter.Recorder(
            base_env,
            directory=args.record_dir,
            save_stats=True if args.record_stats else False,
            save_episode=False,
            save_video=True if args.record_video else False,
        )

    if args.agent == "ppo":

        class PPOActor:
            def __init__(self, model):
                self.model = model

            def predict(self, observation, deterministic=True):

                action, _ = self.model.predict(observation, deterministic=deterministic)
                return [int(action)], None

        agent = PPOActor(ppo_model)
    elif args.agent == "heuristic":
        agent = HeuristicAgent(base_env.action_space.n)
    else:
        agent = RandomAgent(base_env.action_space)

    print(f"Agent: {args.agent}")
    print(
        f"Collecting {args.num_episodes} episodes to {args.outdir} (max_steps={args.max_steps})"
    )

    for ep in range(args.num_episodes):

        ep_seed = args.seed + ep
        try:
            base_env.action_space.seed(ep_seed)
        except Exception:
            pass

        data = collect_episode(base_env, agent, args.max_steps)

        fname = f"episode_{ep:04d}.npz"
        fpath = os.path.join(args.outdir, args.agent, fname)
        np.savez_compressed(
            fpath,
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            dones=data["dones"],
        )
        print(
            f"[{ep+1}/{args.num_episodes}] saved {len(data['actions'])} steps -> {fname}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
