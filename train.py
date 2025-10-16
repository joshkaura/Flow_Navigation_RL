# train.py — training / evaluation
from __future__ import annotations
import os, json
from typing import Dict, Callable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from env import EnvGen, FlowEnv
from agents import TabularQ, PPO

# ---------------------------
# IO helpers
# ---------------------------
def ensure_dirs(outdir: str = "results"):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

# ---------------------------
# Training loops
# ---------------------------
def train_tabular(env: FlowEnv, episodes: int = 300, seed: int = 0) -> Dict:
    
    #Tabular Q-learning on a fixed env.
    #Exploration: ε-greedy with exponential decay.

    np.random.seed(seed)
    agent = TabularQ(env.num_nodes, env.num_actions)

    rewards, steps, success, coverage = [], [], [], []
    for ep in tqdm(range(episodes), desc="Tabular"):
        obs = env.reset(seed + ep)
        done = truncated = False
        total, t = 0.0, 0
        while not (done or truncated):
            a = agent.act(env.s, explore=True)    # use internal state id
            obs2, r, done, truncated, info = env.step(a)
            # simple node-id update (tabular baseline)
            agent.update(env.s, a, r, env.s, done)
            total += r; t += 1; obs = obs2
        agent.step_epsilon()
        rewards.append(total); steps.append(t)
        success.append(1.0 if info["is_goal"] else 0.0)
        coverage.append(info["coverage"])
    return {"rewards": rewards, "steps": steps, "success": success, "coverage": coverage}

def train_ppo_single(
    env: FlowEnv,
    episodes: int = 300,
    seed: int = 1,
    recurrent: bool = False,
    update_every: int = 10,
    ent_start: float = 0.02,
    ent_end: float = 0.005,
    epochs: int = 4,
    batch_size: int = 64,
    clip_eps: float = 0.2,
) -> Dict:

    #PPO on a single env (ablation).
    #Exploration: entropy-regularised policy with linear annealing.

    np.random.seed(seed)
    agent = PPO(
        obs_dim=env.obs_dim, n_actions=env.num_actions,
        recurrent=recurrent, epochs=epochs, batch_size=batch_size,
        clip_eps=clip_eps, ent_coef=ent_start
    )

    rewards, steps, success, coverage = [], [], [], []
    for ep in tqdm(range(episodes), desc=f"PPO(single, rec={recurrent})"):
        # Anneal entropy from ent_start to ent_end
        frac = ep / max(1, episodes - 1)
        agent.ent_coef = ent_start + (ent_end - ent_start) * frac

        s = env.reset(seed + ep)
        agent.reset_hidden()
        done = truncated = False
        total, t = 0.0, 0
        while not (done or truncated):
            a, lp, v = agent.act(s, explore=True)
            s, r, done, truncated, info = env.step(a)
            agent.store(s, a, r, done, lp, v)
            total += r; t += 1

        if (ep + 1) % update_every == 0:
            agent.update()

        rewards.append(total); steps.append(t)
        success.append(1.0 if info["is_goal"] else 0.0)
        coverage.append(info["coverage"])

    agent.update()
    return {"rewards": rewards, "steps": steps, "success": success, "coverage": coverage, "agent": agent}

def train_ppo_distribution(
    make_env: Callable[[], FlowEnv],
    episodes: int = 1800,
    seed: int = 2,
    recurrent: bool = False,
    update_every: int = 20,
    ent_start: float = 0.05,
    ent_end: float = 0.005,
) -> Dict:
    '''
    PPO over a *distribution* of envs (domain randomization) with a simple curriculum:
      Phase A (warm-up): easier Simple graphs
      Phase B (main):   regular Simple graphs

    Training-only shaping is enabled to densify learning signal (off in evaluation).
    '''
    np.random.seed(seed)
    probe = make_env()
    agent = PPO(
        obs_dim=probe.obs_dim, n_actions=probe.num_actions,
        recurrent=recurrent, epochs=6, batch_size=128, clip_eps=0.15,
        ent_coef=ent_start
    )

    def easy_simple() -> FlowEnv:
        # slightly fewer popups/failures, longer horizon
        return FlowEnv(8, 6, popup_p=0.02, fail_p=0.08, dead_p=0.08, max_steps=60)

    def regular_simple() -> FlowEnv:
        return make_env()

    rewards, steps, success, coverage = [], [], [], []
    for ep in tqdm(range(episodes), desc=f"PPO(dist, rec={recurrent})"):
        # Anneal entropy
        frac = ep / max(1, episodes - 1)
        agent.ent_coef = ent_start + (ent_end - ent_start) * frac

        # Curriculum: easy → regular
        env = easy_simple() if ep < 600 else regular_simple()

        # Training-only shaping (safe)
        env.potential_shaping = True
        env.potential_coef = 0.35

        s = env.reset(seed + ep)
        agent.reset_hidden()
        done = truncated = False
        total, t = 0.0, 0
        while not (done or truncated):
            a, lp, v = agent.act(s, explore=True)
            s, r, done, truncated, info = env.step(a)
            agent.store(s, a, r, done, lp, v)
            total += r; t += 1

        if (ep + 1) % update_every == 0:
            agent.update()

        rewards.append(total); steps.append(t)
        success.append(1.0 if info["is_goal"] else 0.0)
        coverage.append(info["coverage"])

    agent.update()
    return {"rewards": rewards, "steps": steps, "success": success, "coverage": coverage, "agent": agent}

# ---------------------------
# Evaluation
# ---------------------------
def evaluate(agent: PPO, make_env: Callable[[], FlowEnv], episodes: int = 200, seed: int = 3000) -> Dict:

    #Zero-shot evaluation on fresh envs with shaping OFF (honest test conditions).

    rewards, length, succ, fail, cover = [], [], 0, 0, []
    for i in range(episodes):
        env = make_env()
        s = env.reset(seed + i)
        agent.reset_hidden()
        done = truncated = False
        total, t = 0.0, 0
        while not (done or truncated):
            a, _, _ = agent.act(s, explore=False)
            s, r, done, truncated, info = env.step(a)
            total += r; t += 1
        rewards.append(total); length.append(t)
        succ += 1 if info["is_goal"] else 0
        fail += 1 if info["is_failure"] else 0
        cover.append(info["coverage"])
    return {
        "success_rate": succ / episodes,
        "failure_rate": fail / episodes,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_length": float(np.mean(length)),
        "avg_coverage": float(np.mean(cover)),
    }

# ---------------------------
# Plotting
# ---------------------------
def plot_ablation(results: Dict, outdir: str = "results"):
    plt.figure(figsize=(12, 8))

    # Success
    plt.subplot(2, 2, 1)
    for name, res in results.items():
        plt.plot(np.array(res["success"]), label=name)
    plt.title("Learning: Success"); plt.xlabel("Episode"); plt.ylabel("Success")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8)

    # Reward
    plt.subplot(2, 2, 2)
    for name, res in results.items():
        plt.plot(np.array(res["rewards"]), label=name)
    plt.title("Learning: Reward"); plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    # Steps
    plt.subplot(2, 2, 3)
    for name, res in results.items():
        plt.plot(np.array(res["steps"]), label=name)
    plt.title("Episode Length"); plt.xlabel("Episode"); plt.ylabel("Steps")
    plt.grid(True, alpha=0.3)

    # Coverage
    plt.subplot(2, 2, 4)
    for name, res in results.items():
        plt.plot(np.array(res["coverage"]), label=name)
    plt.title("Coverage"); plt.xlabel("Episode"); plt.ylabel("Coverage")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "ablation.png"), dpi=160)
    plt.show()
    plt.close()

def plot_generalisation(gen: Dict, outdir: str = "results"):
    names = list(gen.keys())
    x = np.arange(len(names))

    plt.figure(figsize=(12, 4))
    # Success
    plt.subplot(1, 2, 1)
    s = [gen[n]["success_rate"] for n in names]
    plt.bar(x, s)
    plt.xticks(x, names, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.title("generalisation: Success")
    plt.grid(True, axis="y", alpha=0.3)

    # Reward
    plt.subplot(1, 2, 2)
    r = [gen[n]["avg_reward"] for n in names]
    plt.bar(x, r)
    plt.xticks(x, names, rotation=15, ha="right")
    plt.title("generalisation: Avg Reward")
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "generalisation.png"), dpi=160)
    plt.show()
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    ensure_dirs()
    gen = EnvGen(42)

    print("\n[1/3] Ablation on fixed MEDIUM env")
    env_med = gen.medium()
    env_med.potential_shaping = True  # training-only shaping ON in ablation (OK)
    env_med.potential_coef = 0.2

    tab = train_tabular(env_med, episodes=300, seed=0)
    ppo_ff = train_ppo_single(env_med, episodes=300, seed=1, recurrent=False)
    ppo_lstm = train_ppo_single(env_med, episodes=300, seed=2, recurrent=True)

    ablation = {
        "Tabular (ε-greedy)": tab,
        "PPO-FF (entropy)": {k: v for k, v in ppo_ff.items() if k != "agent"},
        "PPO-LSTM (entropy)": {k: v for k, v in ppo_lstm.items() if k != "agent"},
    }
    with open("results/ablation_results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "agent"} for k, v in ablation.items()}, f, indent=2)
    plot_ablation(ablation)

    print("\n[2/3] PPO over SIMPLE distribution (simple curriculum)")
    ppo_ff_dist = train_ppo_distribution(gen.simple, episodes=1800, seed=10, recurrent=False)
    ppo_lstm_dist = train_ppo_distribution(gen.simple, episodes=1800, seed=11, recurrent=True)

    print("\n[3/3] generalisation (no shaping)")
    tests = [
        ("Simple (in-distribution)", gen.simple),
        ("Medium (moderate shift)", gen.medium),
        ("Hard (larger shift)", gen.hard),
        ("Shifted (deeper + popups)", gen.shifted),
    ]
    generalisation = {}
    for name, maker in tests:
        generalisation[name] = {
            "PPO-FF": evaluate(ppo_ff_dist["agent"], maker, episodes=200, seed=3000),
            "PPO-LSTM": evaluate(ppo_lstm_dist["agent"], maker, episodes=200, seed=4000),
        }

    with open("results/generalisation_results.json", "w") as f:
        json.dump(generalisation, f, indent=2)

    # Plot the LSTM variant for a simple bar chart
    plot_generalisation({k: v["PPO-LSTM"] for k, v in generalisation.items()})

    print("\nSaved:")
    print("  - results/plots/ablation.png")
    print("  - results/plots/generalisation.png")
    print("  - results/ablation_results.json")
    print("  - results/generalisation_results.json")
    print("\nThis version intentionally avoids advanced extras (no curiosity, no medium-mix),")
    print("keeps sequence-aware PPO-LSTM, and remains easy to extend to QA later.")

if __name__ == "__main__":
    main()