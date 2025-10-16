# agents.py — Tabular (epsilon-greedy) and PPO (Feed-Forward & LSTM)
from __future__ import annotations
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================
# Tabular Q-learning (epsilon-greedy exploration)
# ===========================================
class TabularQ:
    def __init__(self, n_states: int, n_actions: int,
                 lr: float = 0.2, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 0.995):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.eps = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)
        self.n_actions = int(n_actions)

    def act(self, state_id: int, explore: bool = True) -> int:
        if explore and (np.random.rand() < self.eps):
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state_id]))

    def update(self, s: int, a: int, r: float, s2: int, done: bool):
        # One-step Q-learning update
        best_next = np.argmax(self.Q[s2])
        target = r if done else r + self.gamma * self.Q[s2, best_next]
        self.Q[s, a] += self.lr * (target - self.Q[s, a])

    def step_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

# ================
# PPO networks
# ================
class _FF(nn.Module):
    """Small feed-forward actor-critic."""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        x = torch.nan_to_num(x, nan=0.0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value

class _LSTM(nn.Module):
    #LSTM actor-critic that returns per-timestep logits/values for sequences.
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.emb = nn.Linear(obs_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x: torch.Tensor, h=None):
        '''
        x: [B, T, obs_dim]
        returns:
          logits: [B, T, n_actions]
          values: [B, T]
          hidden: final (h, c)
        '''
        x = torch.nan_to_num(x, nan=0.0)
        z = F.relu(self.emb(x))
        out, h = self.lstm(z, h)      # [B, T, H]
        logits = self.pi(out)         # [B, T, A]
        values = self.v(out).squeeze(-1)  # [B, T]
        return logits, values, h

    def init_hidden(self, batch_size: int):
        h = torch.zeros(1, batch_size, self.hidden)
        c = torch.zeros(1, batch_size, self.hidden)
        return (h, c)

# =====================================================================================
# PPO (with feed-forward or LSTM policy)
# - Entropy coefficient is annealed outside (in the trainer) to provide exploration.
# - LSTM trains on full episodes
# =====================================================================================
class PPO:
    def __init__(self, obs_dim: int, n_actions: int,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, ent_coef: float = 0.01, val_coef: float = 0.5,
                 epochs: int = 4, batch_size: int = 64,
                 recurrent: bool = False):
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.clip_eps = float(clip_eps)
        self.ent_coef = float(ent_coef)
        self.val_coef = float(val_coef)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.recurrent = bool(recurrent)

        self.net = _LSTM(obs_dim, n_actions) if recurrent else _FF(obs_dim, n_actions)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.hidden = None  # only used if recurrent=True

        # FF buffers (flat)
        self.S: List[np.ndarray] = []
        self.A: List[int] = []
        self.R: List[float] = []
        self.D: List[bool] = []
        self.LP: List[float] = []
        self.V: List[float] = []

        # LSTM buffers (episode lists)
        self.E_S: List[List[np.ndarray]] = []
        self.E_A: List[List[int]] = []
        self.E_R: List[List[float]] = []
        self.E_D: List[List[bool]] = []
        self.E_LP: List[List[float]] = []
        self.E_V: List[List[float]] = []

        # currently accumulating episode (LSTM mode)
        self._cs: List[np.ndarray] = []
        self._ca: List[int] = []
        self._cr: List[float] = []
        self._cd: List[bool] = []
        self._clp: List[float] = []
        self._cv: List[float] = []

    # ---------- Acting ----------
    def reset_hidden(self):
        if self.recurrent:
            self.hidden = None

    def act(self, obs: np.ndarray, explore: bool = True):
        if self.recurrent:
            # Single-step inference while maintaining hidden state
            x = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(1)  # [1,1,obs]
            if self.hidden is None:
                self.hidden = self.net.init_hidden(1)
            logits_seq, values_seq, self.hidden = self.net(x, self.hidden)  # [1,1,A], [1,1]
            logits = logits_seq[:, -1, :]
            value = values_seq[:, -1]
        else:
            x = torch.from_numpy(obs).float().unsqueeze(0)
            logits, value = self.net(x)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if explore else torch.argmax(logits, dim=-1)
        logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.squeeze().item())

    # ---------- Storage ----------
    def store(self, s, a, r, done, logp, v):
        if not self.recurrent:
            self.S.append(s); self.A.append(a); self.R.append(r); self.D.append(done); self.LP.append(logp); self.V.append(v)
        else:
            self._cs.append(s); self._ca.append(a); self._cr.append(r); self._cd.append(done); self._clp.append(logp); self._cv.append(v)
            if done:
                # push the finished episode to the sequence buffers
                self.E_S.append(self._cs); self._cs = []
                self.E_A.append(self._ca); self._ca = []
                self.E_R.append(self._cr); self._cr = []
                self.E_D.append(self._cd); self._cd = []
                self.E_LP.append(self._clp); self._clp = []
                self.E_V.append(self._cv); self._cv = []

    # ---------- GAE ----------
    def _gae_flat(self):
        vals = np.asarray(self.V, dtype=np.float32)
        adv, ret, gae = [], [], 0.0
        for t in reversed(range(len(self.R))):
            v = vals[t]
            next_v = 0.0 if t == len(self.R) - 1 else vals[t + 1]
            delta = self.R[t] + self.gamma * (1 - int(self.D[t])) * next_v - v
            gae = delta + self.gamma * self.lam * (1 - int(self.D[t])) * gae
            adv.insert(0, gae)
            ret.insert(0, gae + v)
        ADV = torch.tensor(adv, dtype=torch.float32)
        R = torch.tensor(ret, dtype=torch.float32)
        if ADV.numel() > 1:
            std = ADV.std(unbiased=False)
            ADV = (ADV - ADV.mean()) / (std + 1e-8) if torch.isfinite(std) and std > 1e-6 else (ADV - ADV.mean())
        return R, ADV

    def _gae_seq(self, vals: np.ndarray, rews: np.ndarray, dones: np.ndarray):
        T = len(rews)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            v = vals[t]
            next_v = 0.0 if t == T - 1 else vals[t + 1]
            delta = rews[t] + self.gamma * (1 - int(dones[t])) * next_v - v
            gae = delta + self.gamma * self.lam * (1 - int(dones[t])) * gae
            adv[t] = gae
        ADV = torch.tensor(adv, dtype=torch.float32)
        R = torch.tensor(adv + vals, dtype=torch.float32)
        if ADV.numel() > 1:
            std = ADV.std(unbiased=False)
            ADV = (ADV - ADV.mean()) / (std + 1e-8) if torch.isfinite(std) and std > 1e-6 else (ADV - ADV.mean())
        return R, ADV

    # ---------- Update ----------
    def update(self):
        if not self.recurrent:
            if len(self.S) < 2:
                self._clear_flat()
                return

            S = torch.tensor(np.asarray(self.S, dtype=np.float32))
            A = torch.tensor(self.A, dtype=torch.long)
            OL = torch.tensor(self.LP, dtype=torch.float32)
            R, ADV = self._gae_flat()

            for _ in range(self.epochs):
                idx = np.arange(len(S)); np.random.shuffle(idx)
                for st in range(0, len(S), self.batch_size):
                    mb = idx[st:st + self.batch_size]
                    s, a, old_lp, ret, adv = S[mb], A[mb], OL[mb], R[mb], ADV[mb]

                    logits, v = self.net(s)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_lp = dist.log_prob(a)
                    ratio = torch.exp(torch.clamp(new_lp - old_lp, -10.0, 10.0))

                    # PPO clipped 
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                    pi_loss = -torch.min(surr1, surr2).mean()
                    v_loss = F.mse_loss(v, ret)
                    ent = dist.entropy().mean()

                    loss = pi_loss + self.val_coef * v_loss - self.ent_coef * ent
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.opt.step()

            self._clear_flat()
            return

        # Recurrent: optimise per-episode sequences
        if len(self.E_S) == 0:
            return

        for _ in range(self.epochs):
            order = np.arange(len(self.E_S)); np.random.shuffle(order)
            for i in order:
                S = torch.tensor(np.asarray(self.E_S[i], dtype=np.float32)).unsqueeze(0)  # [1, T, obs]
                A = torch.tensor(self.E_A[i], dtype=torch.long).unsqueeze(0)              # [1, T]
                OL = torch.tensor(self.E_LP[i], dtype=torch.float32).unsqueeze(0)        # [1, T]

                vals = np.asarray(self.E_V[i], dtype=np.float32)
                rews = np.asarray(self.E_R[i], dtype=np.float32)
                dones = np.asarray(self.E_D[i], dtype=np.float32)

                R, ADV = self._gae_seq(vals, rews, dones)
                R = R.unsqueeze(0)      # [1, T]
                ADV = ADV.unsqueeze(0)  # [1, T]

                logits, V, _ = self.net(S)  # logits: [1, T, A], V: [1, T]
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(A.squeeze(0)).unsqueeze(0)  # [1, T]
                ratio = torch.exp(torch.clamp(new_lp - OL, -10.0, 10.0))

                # PPO clipped
                surr1 = ratio * ADV
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * ADV
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(V, R)
                ent = dist.entropy().mean()

                loss = pi_loss + self.val_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

        self._clear_seq()

    # ---------- Buffer clears ----------
    def _clear_flat(self):
        self.S.clear(); self.A.clear(); self.R.clear(); self.D.clear(); self.LP.clear(); self.V.clear()

    def _clear_seq(self):
        self.E_S.clear(); self.E_A.clear(); self.E_R.clear(); self.E_D.clear(); self.E_LP.clear(); self.E_V.clear()