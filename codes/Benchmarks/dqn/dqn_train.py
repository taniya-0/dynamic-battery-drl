import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import json

from environment import Env
from dqn_model import get_state, get_local_obs, QNetwork, ReplayBuffer, AverageRewardBaseline


def train_idqn():

    # ── Environment setup - Input environment parameters here ──────────────────────────────────────────────────────
    ENV_KWARGS = dict(
        n_row=16, n_aisle=8, n_cp=2, n_block=4,
        arrival_rate=0.6, capacity=10,
        max_battery=100, min_battery=0, min_battery_limit=15,
        depletion_rate=1, recharge_rate=2,
        velocity=1,
        MAX_STEPS=14400,
        seed=42,
    )
    env = Env(**ENV_KWARGS)

	# ── Dimensions ───────────────────────────────────────────────────────────
	N_AGENTS = ENV_KWARGS['n_block']
	LOCAL_DIM = 6 + 1 + 2 + (N_AGENTS * 4)
	ACTION_DIM = 8
    

    # ── Hyperparameters ───────────────────────────────────────────────────────────
    EPOCHS = 15000
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 128
    BUFFER_SIZE = 200000
    UPDATES_PER_EPISODE = 200
    TARGET_UPDATE_FREQ = 50
    EPSILON_START = 0.5
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.9995

    device = torch.device("cpu")

    # Networks per agent
    q_networks = [QNetwork(LOCAL_DIM, ACTION_DIM, 512).to(device) for _ in range(N_AGENTS)]
    target_networks = [QNetwork(LOCAL_DIM, ACTION_DIM, 512).to(device) for _ in range(N_AGENTS)]
    optimizers = [optim.Adam(q.parameters(), lr=LEARNING_RATE) for q in q_networks]

    # Initialize targets
    for i in range(N_AGENTS):
        target_networks[i].load_state_dict(q_networks[i].state_dict())

    # Replay buffers and baselines per agent
    buffers = [ReplayBuffer(BUFFER_SIZE) for _ in range(N_AGENTS)]
    r_bars = [AverageRewardBaseline(alpha=0.01) for _ in range(N_AGENTS)]

    # ─── Training loop ───────────────────────────────────────────────────────────
    best_completion = 0
    for ep in range(EPOCHS):
        epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** ep))

        obs, _ = env.reset()
        episode_rewards = [[] for _ in range(N_AGENTS)]

        # Collect episode
        while env.global_time_step < ENV_KWARGS['MAX_STEPS']:
            local_obs_arr = np.array([get_local_obs(obs, env, i) for i in range(1, N_AGENTS+1)])
            allowed = env.allowed_action()

            actions = []
            for i in range(N_AGENTS):
                if np.random.rand() < epsilon:
                    # Epsilon-greedy: random masked action
                    valid_actions = np.where(allowed[i])[0]
                    if len(valid_actions) == 0: action = 7
                    else: action = np.random.choice(valid_actions)
                else:
                    # Greedy: argmax over masked Q-values
                    with torch.no_grad():
                        obs_t = torch.tensor(local_obs_arr[i], dtype=torch.float32).unsqueeze(0)
                        q_vals = q_networks[i](obs_t).squeeze(0)
                        q_vals[~torch.tensor(allowed[i], dtype=torch.bool)] = -1e9
                        action = q_vals.argmax().item()
                actions.append(action)

            orders_before = {n: env.completion_info[n][0] for n in range(1, N_AGENTS+1)}
            next_obs, reward, done, info = env.step(actions)

            next_local_obs_arr = np.array([get_local_obs(next_obs, env, i) for i in range(1, N_AGENTS+1)])
            agent_was_free = [allowed[i].sum() > 1 for i in range(N_AGENTS)]
            for i in range(N_AGENTS):
                buffers[i].add(local_obs_arr[i], actions[i], reward[i], next_local_obs_arr[i])
                episode_rewards[i].append(reward[i])

            obs = next_obs

        # --- Update Q-networks  ------------------------------------------------------------------------#
        if len(buffers[0]) >= BATCH_SIZE:
            for _ in range(UPDATES_PER_EPISODE):
                for i in range(N_AGENTS):
                    obs_batch, act_batch, rew_batch, next_obs_batch = buffers[i].sample(BATCH_SIZE)

                    obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                    act_t = torch.tensor(act_batch, dtype=torch.long, device=device)
                    rew_t = torch.tensor(rew_batch, dtype=torch.float32, device=device)
                    next_obs_t = torch.tensor(next_obs_batch, dtype=torch.float32, device=device)

                    q_vals = q_networks[i](obs_t)
                    q_current = q_vals.gather(1, act_t.unsqueeze(1)).squeeze(1)

                    # Target Q-values
                    with torch.no_grad():
                        q_next = target_networks[i](next_obs_t).max(dim=1)[0]
                        # Differential TD target: r - r_bar + max Q(s',a')
                        q_target = rew_t - r_bars[i].get() + q_next

                    # Loss and update
                    loss = nn.MSELoss()(q_current, q_target)
                    optimizers[i].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_networks[i].parameters(), 0.5)
                    optimizers[i].step()

        # --- Update target networks -----------------------------------------------------------------#
        if ep % TARGET_UPDATE_FREQ == 0 and ep > 0:
            for i in range(N_AGENTS):
                target_networks[i].load_state_dict(q_networks[i].state_dict())

        # --- Update average reward baselines -----------------------------------------------------------------#
        for i in range(N_AGENTS):
            r_bars[i].update(episode_rewards[i])

        # Logging
        total_completed = sum(env.completion_info[n][0] for n in range(1, N_AGENTS+1))
        total_placed = sum(env.total_orders_placed)
        completion_pct = 100 * total_completed / max(total_placed, 1)
        if completion_pct > best_completion:
            best_completion = completion_pct
            torch.save({
                'epoch': ep,
                'completion': completion_pct,
                **{f'q_net_{i}': q_networks[i].state_dict() for i in range(N_AGENTS)},
                **{f'r_bar_{i}': r_bars[i].r_bar for i in range(N_AGENTS)}
            }, 'dqn_best.pt')
            print(f"✓ New best DQN: {completion_pct:.1f}%")

        if ep % 5 == 0:
            avg_reward = np.mean([np.sum(episode_rewards[i]) for i in range(N_AGENTS)])

            print(f"Ep {ep:4d} | Comp {completion_pct:5.1f}% | "
                  f"Reward {avg_reward:7.0f} | ε {epsilon:.3f} | "
                  f"r̄_avg {np.mean([r.get() for r in r_bars]):.2f}")

        # Save checkpoint
        if ep % 100 == 0 and ep > 0:
            torch.save({
                'epoch': ep,
                **{f'q_net_{i}': q_networks[i].state_dict() for i in range(N_AGENTS)},
                **{f'r_bar_{i}': r_bars[i].r_bar for i in range(N_AGENTS)}
            }, 'dqn_checkpoint.pt')

if __name__ == "__main__":
    train_idqn()