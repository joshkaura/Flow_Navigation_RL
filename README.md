# Minimal RL for Flow Navigation (Tabular + PPO-FF + PPO-LSTM)

This repo contains an RL setup for learning to navigate a graph-like “flow” under partial observability. It’s written in 3 files:  
- `env.py` – environment (graph world, popups, optional training-only shaping)  
- `agents.py` – Tabular Q, PPO-FF, PPO-LSTM (sequence-aware)  
- `train.py` – training loops, evaluation, plots, JSON logging

---

## Environment — spec & design choices

**World:** small directed graph (8–20 nodes) with a mild forward bias.  
**Partial observability:** random “popup” steps ignore the chosen action.  
**Terminals:** one goal node (+1) and occasional failure nodes (−1).  
**Base rewards:** step penalty (−0.02) + first-visit bonus (+0.10).  
**Training-only shaping (safe):** potential‐based term `φ(s′) − φ(s)` with `φ` from inverse shortest-path distance to goal. Shaping is on in training, **off** in evaluation.  
**Observation (stationary across graphs):** 13-D = 10 local binary features + `[visited_frac, step_frac, φ]`. No node one-hot → better generalization across randomly generated graphs.

**Distributions for generalisation:**  
- *Simple* (in-distribution)  
- *Medium* (moderate shift)  
- *Hard* (larger shift)  
- *Shifted* (deeper + more popups)

---

## Algorithms & rationale

- **Tabular Q-learning (ε-greedy):** transparent baseline using node id as state.  
- **PPO-FF (entropy-regularized):** on-policy baseline with entropy annealed over episodes  
- **PPO-LSTM (sequence-aware):** PPO with LSTM, trained on **full episodes** to handle partial observability (popups).

**Curriculum:** start on an “easy Simple” generator (fewer popups/failures), then train on regular Simple. Stabilizes value/advantage estimates.

---

##  Setup, commands & expected runtimes

- Run everything (ablation + distribution training + generalization using `python train3.py`.

- Expected runtime here for training and evaluation runs here is < 60 seconds.

## Results

Results can be found in the *results* folder:

- **ablation_results.json**: rewards, steps, success, coverage for each of the 3 agents
- **generalisation_results**: PPO metrics when evaluated under distribution shift on unseen graphs
- **plots** folder contains figures showing results graphically

## To be Added/ Tested

- Research into the addition of curiosity to help deal with popups

- Diagnostics: save a few success/failure rollouts (nodes visited, actions, popups) to explain behavior and failure modes.
