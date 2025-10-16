# env.py — simple graph world with partial observability + optional (training-only) safe shaping
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from collections import deque
import numpy as np
import random

# ---------------------------
# Seeding utility
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

# ---------------------------
# Node definition
# ---------------------------
@dataclass
class Node:
    local_feats: np.ndarray  # 10 binary indicators (local hints)
    is_goal: bool
    is_failure: bool
    is_deadend: bool

# ---------------------------
# Environment
# ---------------------------
class FlowEnv:
    '''
    Minimal directed-graph MDP

    Partial observability: random “popup” steps ignore the chosen action.

    Observation (stationary across graphs):
      [10 local bits] + [visited_frac, step_frac, phi]
        - visited_frac : |visited| / num_nodes
        - step_frac    : t / max_steps
        - phi          : normalized inverse shortest distance to goal (0..1)

    Rewards:
      +1 goal, -1 failure, small step penalty per move, +first-visit bonus
      + (optional, training-only) shaping: coef * (phi(s') - phi(s))
    '''

    def __init__(
        self,
        num_nodes: int = 8,
        num_actions: int = 6,
        popup_p: float = 0.10,
        fail_p: float = 0.10,
        dead_p: float = 0.10,
        max_steps: int = 50,
        goal_node: Optional[int] = None,
        potential_shaping: bool = False,
        potential_coef: float = 0.2,
    ):
        self.num_nodes = int(num_nodes)
        self.num_actions = int(num_actions)
        self.popup_p = float(popup_p)
        self.fail_p = float(fail_p)
        self.dead_p = float(dead_p)
        self.max_steps = int(max_steps)

        # Training-only shaping toggles
        self.potential_shaping = bool(potential_shaping)
        self.potential_coef = float(potential_coef)

        # Build a random graph instance, then get potentials
        self._build_graph(goal_node)
        self._compute_potential()

        # Observation: 10 local + 3 scalars
        self.obs_dim = 10 + 3

        # Episode state
        self.reset()

    # ---------------------------
    # Graph generation
    # ---------------------------
    def _build_graph(self, goal_node: Optional[int]):
        self.goal = self.num_nodes - 1 if goal_node is None else int(goal_node)

        # Create nodes with local features and random failure/dead-end flags
        self.nodes: Dict[int, Node] = {}
        for i in range(self.num_nodes):
            feats = np.zeros(10, dtype=np.float32)
            k = np.random.randint(1, 4)
            feats[np.random.choice(10, size=k, replace=False)] = 1
            self.nodes[i] = Node(
                local_feats=feats,
                is_goal=(i == self.goal),
                is_failure=(i != self.goal and np.random.rand() < self.fail_p),
                is_deadend=(i != self.goal and np.random.rand() < self.dead_p),
            )

        # Transitions: mild forward bias, goals/dead-ends self-loop
        self.trans: Dict[int, Dict[int, int]] = {i: {} for i in range(self.num_nodes)}
        for i in range(self.num_nodes):
            if self.nodes[i].is_goal or self.nodes[i].is_deadend:
                # Self-loop on all actions
                for a in range(self.num_actions):
                    self.trans[i][a] = i
                continue

            for a in range(self.num_actions):
                if i < self.num_nodes - 1 and random.random() > 0.30:
                    nxt = random.choice(range(i + 1, min(self.num_nodes, i + 4)))
                else:
                    nxt = i
                self.trans[i][a] = nxt

        # Ensure at least one forward edge for non-goal nodes
        for i in range(self.num_nodes - 1):
            if all(self.trans[i][a] == i for a in range(self.num_actions)):
                self.trans[i][random.randrange(self.num_actions)] = i + 1

    def _compute_potential(self):
        # Build adjacency from transitions
        adj = {i: set() for i in range(self.num_nodes)}
        for i in range(self.num_nodes):
            for a in range(self.num_actions):
                adj[i].add(self.trans[i][a])

        # Reverse BFS from goal to compute distances => potential phi
        rev = {i: set() for i in range(self.num_nodes)}
        for u in range(self.num_nodes):
            for v in adj[u]:
                rev[v].add(u)

        INF = 10**9
        dist = [INF] * self.num_nodes
        dist[self.goal] = 0
        q = deque([self.goal])
        while q:
            v = q.popleft()
            for u in rev[v]:
                if dist[u] > dist[v] + 1:
                    dist[u] = dist[v] + 1
                    q.append(u)

        finite = [d for d in dist if d < INF]
        mx = max(finite) if finite and max(finite) > 0 else 1
        self.phi = np.array([0 if d >= INF else 1.0 - d / mx for d in dist], dtype=np.float32)

    # ---------------------------
    # Episode API
    # ---------------------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            set_seed(seed)
        self.s = 0
        self.t = 0
        self.visited = {0}
        return self._obs()

    def _obs(self):
        visited_frac = len(self.visited) / float(self.num_nodes)
        step_frac = self.t / float(self.max_steps)
        return np.concatenate([
            self.nodes[self.s].local_feats,
            np.array([visited_frac, step_frac, self.phi[self.s]], dtype=np.float32)
        ], axis=0).astype(np.float32)

    def step(self, action: int):
        self.t += 1

        # Random popup: ignore intended action
        if random.random() < self.popup_p:
            obs = self._obs()
            done = False
            truncated = self.t >= self.max_steps
            info = self._info(False, False)
            return obs, -0.5, done, truncated, info

        nxt = self.trans[self.s][action]

        # Potential-based shaping (training only)
        shaping = self.potential_coef * (self.phi[nxt] - self.phi[self.s]) if self.potential_shaping else 0.0

        # Base reward: small step penalty + first-visit bonus
        step_penalty = -0.02
        first_visit = 0.0 if nxt in self.visited else 0.10

        self.s = nxt
        self.visited.add(self.s)

        node = self.nodes[self.s]
        done = False
        truncated = self.t >= self.max_steps
        reward = step_penalty + first_visit + shaping

        if node.is_goal:
            reward = 1.0
            done = True
        elif node.is_failure:
            reward = -1.0
            done = True

        return self._obs(), float(reward), bool(done), bool(truncated), self._info(node.is_goal, node.is_failure)

    def _info(self, is_goal: bool, is_failure: bool):
        return {"node": self.s, "is_goal": is_goal, "is_failure": is_failure, "coverage": len(self.visited) / self.num_nodes}

# ---------------------------
# Generators for splits
# ---------------------------
class EnvGen:
    """Families of graphs for train/test; change only a few knobs to create shifts."""
    def __init__(self, seed: int = 42):
        set_seed(seed)

    def simple(self) -> FlowEnv:
        return FlowEnv(8, 6, popup_p=0.05, fail_p=0.10, dead_p=0.10, max_steps=50)

    def medium(self) -> FlowEnv:
        return FlowEnv(12, 6, popup_p=0.10, fail_p=0.12, dead_p=0.12, max_steps=60)

    def hard(self) -> FlowEnv:
        return FlowEnv(16, 6, popup_p=0.15, fail_p=0.15, dead_p=0.15, max_steps=70)

    def shifted(self) -> FlowEnv:
        return FlowEnv(20, 6, popup_p=0.25, fail_p=0.20, dead_p=0.20, max_steps=90)