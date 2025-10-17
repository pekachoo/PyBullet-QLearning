import time, argparse, os
import numpy as np
import gymnasium as gym
import pufferlib.emulation as pe
from stable_baselines3 import PPO

def make_env():
    # standard Gymnasium CartPole
    return gym.make("CartPole-v1")

def evaluate(model, n_episodes=5):
    # evaluate on a plain (non-Puffer) env to show interchangeability
    env = make_env()
    returns = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        ep_ret = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
        returns.append(ep_ret)
    env.close()
    return np.mean(returns), np.std(returns)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16, help="parallel envs")
    parser.add_argument("--timesteps", type=int, default=200_000, help="train steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Wrap your envs with Puffer. Swap to pe.Gymnasium(make_env) for single-env.
    env = pe.Vectorized(make_env, num_envs=args.num_envs, seed=args.seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        # CartPole is simple—defaults are fine. Feel free to tweak n_steps for throughput.
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    dt = time.time() - t0
    steps_per_sec = args.timesteps / max(dt, 1e-9)

    avg_ret, std_ret = evaluate(model, n_episodes=5)

    save_path = "ppo_cartpole_puffer.zip"
    model.save(save_path)

    print("\n=== RESULTS ===")
    print(f"Parallel envs          : {args.num_envs}")
    print(f"Total timesteps        : {args.timesteps}")
    print(f"Wall time (s)          : {dt:.2f}")
    print(f"Throughput (steps/sec) : {steps_per_sec:,.0f}")
    print(f"Eval return (mean±std) : {avg_ret:.1f} ± {std_ret:.1f}")
    print(f"Model saved to         : {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()
