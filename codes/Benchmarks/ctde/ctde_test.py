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
from ctde_model import get_state, get_local_obs, get_global_obs, Actor, CentralizedCritic, RolloutBuffer, AverageRewardBaseline, compute_differential_gae, ppo_update

ENV_KWARGS = dict(
	n_row=24, n_aisle=8, n_cp=2, n_block=6,
	arrival_rate=0.75, capacity=10,
	max_battery=100, min_battery=0, min_battery_limit=20,
	depletion_rate=1, recharge_rate=2,
	velocity=1,
	MAX_STEPS=28800,
	seed=42,
)
env = Env(**ENV_KWARGS)

LOCAL_DIM = 7  
ACTION_DIM = 8  
N_AGENTS = ENV_KWARGS['n_block']
GLOBAL_DIM = LOCAL_DIM*N_AGENTS + ENV_KWARGS['n_cp']

STEPS = ENV_KWARGS['MAX_STEPS']

# ── Networks ─────────────────────────────────────────────────────────────
device = torch.device("cpu")
print(f"Using device: {device}")

actor  = SharedActor(LOCAL_DIM, ACTION_DIM, hidden=256).to(device)
critic = CentralizedCritic(GLOBAL_DIM, N_AGENTS, hidden=512).to(device)
actor_opt  = optim.Adam(actor.parameters(), lr=3e-4)
critic_opt = optim.Adam(critic.parameters(), lr=3e-4)

r_bars = [AverageRewardBaseline(alpha=0.01) for _ in range(N_AGENTS)]
	
CKPT_PATH_BEST = "best_model.pt"
ckpt_b = torch.load(CKPT_PATH_BEST, map_location=device, weights_only=False)
ckpt_b.pop('tracking_data', None)
torch.save(ckpt_b, CKPT_PATH_BEST)
actor.load_state_dict(ckpt_b['actor'])
actor.eval()

# ─────────────────────────────────────────────
#  Evaluation loop
# ─────────────────────────────────────────────

n_test_ep = 10


episode_rewards_arr   = []
episode_rewards = []
orders_completed      = []
orders_completed1     = []
orders_completed2     = []
orders_completed3     = []
orders_completed4     = []
orders_completed5     = []
orders_completed6     = []
episode_lengths       = []
time_travelled1   = []
time_travelled2   = []
time_travelled3   = []
time_travelled4   = []
time_travelled5   = []
time_travelled6   = []
total_orders_placed   = []
total_orders_placed1  = []
total_orders_placed2  = []
total_orders_placed3  = []
total_orders_placed4  = []
total_orders_placed5  = []
total_orders_placed6  = []
time_spent_charging1  = []
time_spent_charging2  = []
time_spent_charging3  = []
time_spent_charging4  = []
time_spent_charging5  = []
time_spent_charging6  = []
travel_time_pick1     = []
travel_time_pick2     = []
travel_time_pick3     = []
travel_time_pick4     = []
travel_time_pick5     = []
travel_time_pick6     = []
depot_travel_tracker1 = []
depot_travel_tracker2 = []
depot_travel_tracker3 = []
depot_travel_tracker4 = []
depot_travel_tracker5 = []
depot_travel_tracker6 = []
depot_counter_cap_tracker = {}
depot_counter_num_tracker = {}
wait_in_queue_tracker1 = []
wait_in_queue_tracker2 = []
wait_in_queue_tracker3 = []
wait_in_queue_tracker4 = []
wait_in_queue_tracker5 = []
wait_in_queue_tracker6 = []

for ep in range(n_test_ep):

    action_counter_tracker = {1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]}
    obs, info = env.reset()
    done         = False
    episode_rewards = 0.0
    steps        = 0

    cap = []
    bat = []
    stopcharge = []
    cp_state_tracker = {n: [] for n in range(1, 2 + 1)}
    cp_queue_tracker = {n: [] for n in range(1, 2 + 1)}
    ctiq1 = []
    ctiq2 = []
    which_cp = {1:[0,0], 2:[0,0], 3:[0,0], 4:[0,0]}
    rewards = []
    orders_compl = []
    while env.global_time_step < ENV_KWARGS['MAX_STEPS']:
        local_obs = np.stack([get_local_obs(obs, env, i) for i in range(1, N_AGENTS+1)])  # [n_agents, local_dim]
        allowed = env.allowed_action()

        with torch.no_grad():
            logits = actor(torch.tensor(local_obs, dtype=torch.float32))  # [n_agents, action_dim]

        actions = []
        for i in range(N_AGENTS):
            agent_logits = logits[i].masked_fill(~torch.tensor(allowed[i], dtype=torch.bool), -1e9)
            action = agent_logits.argmax().item()
            actions.append(action)

        orders_before = {n: env.completion_info[n][0] for n in range(1, N_AGENTS + 1)}
        next_obs, reward, done, info = env.step(actions)

        episode_rewards += np.sum(reward)
        obs = next_obs
        steps += 1

        rewards.append(reward)
        orders_compl_count = 0
        for i in range(1,env.n_agent+1):
          orders_compl_count += env.completion_info[i][0]
        orders_compl.append(orders_compl_count/np.sum(env.total_orders_placed))


    if n_test_ep > 1:
        episode_rewards_arr.append(episode_rewards)

        total_completed = sum(
            env.completion_info[n][0] for n in range(1, env.n_agent + 1)
        )
        orders_completed.append(total_completed)
        orders_completed1.append(env.completion_info[1][0])
        orders_completed2.append(env.completion_info[2][0])
        orders_completed3.append(env.completion_info[3][0])
        orders_completed4.append(env.completion_info[4][0])
        orders_completed5.append(env.completion_info[5][0])
        orders_completed6.append(env.completion_info[6][0])


        episode_lengths.append(steps)

        time_travelled1.append(env.time_travelled[0])
        time_travelled2.append(env.time_travelled[1])
        time_travelled3.append(env.time_travelled[2])
        time_travelled4.append(env.time_travelled[3])
        time_travelled5.append(env.time_travelled[4])
        time_travelled6.append(env.time_travelled[5])

        total_orders_placed1.append(env.total_orders_placed[0])
        total_orders_placed2.append(env.total_orders_placed[1])
        total_orders_placed3.append(env.total_orders_placed[2])
        total_orders_placed4.append(env.total_orders_placed[3])
        total_orders_placed5.append(env.total_orders_placed[4])
        total_orders_placed6.append(env.total_orders_placed[5])

        time_spent_charging1.append(env.completion_info[1][2])
        time_spent_charging2.append(env.completion_info[2][2])
        time_spent_charging3.append(env.completion_info[3][2])
        time_spent_charging4.append(env.completion_info[4][2])
        time_spent_charging5.append(env.completion_info[5][2])
        time_spent_charging6.append(env.completion_info[6][2])


        wait_in_queue_tracker1.append(env.wait_in_q[1])
        wait_in_queue_tracker2.append(env.wait_in_q[2])
        wait_in_queue_tracker3.append(env.wait_in_q[3])
        wait_in_queue_tracker4.append(env.wait_in_q[4])
        wait_in_queue_tracker5.append(env.wait_in_q[5])
        wait_in_queue_tracker6.append(env.wait_in_q[6])


results = {"mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_orders_completed":          np.mean(orders_completed),
        "mean_orders_completed1":         np.mean(orders_completed1),
        "mean_orders_completed2":         np.mean(orders_completed2),
        "mean_orders_completed3":         np.mean(orders_completed3),
        "mean_orders_completed4":         np.mean(orders_completed4),
        "mean_orders_completed5":         np.mean(orders_completed5),
        "mean_orders_completed6":         np.mean(orders_completed6),

        # episode length
        "mean_episode_length":            np.mean(episode_lengths),

        # orders placed
        "mean_total_orders_placed1":      np.mean(total_orders_placed1),
        "mean_total_orders_placed2":      np.mean(total_orders_placed2),
        "mean_total_orders_placed3":      np.mean(total_orders_placed3),
        "mean_total_orders_placed4":      np.mean(total_orders_placed4),
        "mean_total_orders_placed5":      np.mean(total_orders_placed5),
        "mean_total_orders_placed6":      np.mean(total_orders_placed6),

        # distance travelled
        "mean_time_travelled1":       np.mean(time_travelled1),
        "mean_time_travelled2":       np.mean(time_travelled2),
        "mean_time_travelled3":       np.mean(time_travelled3),
        "mean_time_travelled4":       np.mean(time_travelled4),
        "mean_time_travelled5":       np.mean(time_travelled5),
        "mean_time_travelled6":       np.mean(time_travelled6),

        # time spent charging
        "mean_time_spent_charging1":      np.mean(time_spent_charging1),
        "mean_time_spent_charging2":      np.mean(time_spent_charging2),
        "mean_time_spent_charging3":      np.mean(time_spent_charging3),
        "mean_time_spent_charging4":      np.mean(time_spent_charging4),
        "mean_time_spent_charging5":      np.mean(time_spent_charging5),
        "mean_time_spent_charging6":      np.mean(time_spent_charging6),

        "wait_in_queue1":                 np.mean(wait_in_queue_tracker1),
        "wait_in_queue2":                 np.mean(wait_in_queue_tracker2),
        "wait_in_queue3":                 np.mean(wait_in_queue_tracker3),
        "wait_in_queue4":                 np.mean(wait_in_queue_tracker4),
        "wait_in_queue5":                 np.mean(wait_in_queue_tracker5),
        "wait_in_queue6":                 np.mean(wait_in_queue_tracker6),

        "order_completion_rate_overall":  100*np.sum(orders_completed) / (np.sum(total_orders_placed1) + np.sum(total_orders_placed2) + np.sum(total_orders_placed3) + np.sum(total_orders_placed4) + np.sum(total_orders_placed5) + np.sum(total_orders_placed6)),
        "order_completion_rate1":         100*np.mean(orders_completed1) / np.mean(total_orders_placed1),
        "order_completion_rate2":         100*np.mean(orders_completed2) / np.mean(total_orders_placed2),
        "order_completion_rate3":         100*np.mean(orders_completed3) / np.mean(total_orders_placed3),
        "order_completion_rate4":         100*np.mean(orders_completed4) / np.mean(total_orders_placed4),
        "order_completion_rate5":         100*np.mean(orders_completed5) / np.mean(total_orders_placed5),
        "order_completion_rate6":         100*np.mean(orders_completed6) / np.mean(total_orders_placed6),
        }

for k, v in results.items():
  if "raw" not in k:                  # skip raw lists, print scalars only
      print(f"  {k}: {v:.3f}")