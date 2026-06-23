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


# ── Environment setup - Input environment parameters here ──────────────────────────────────────────────────────
ENV_KWARGS = dict(
	n_row=16, n_aisle=8, n_cp=2, n_block=4,
	arrival_rate=0.6, capacity=10,
	max_battery=100, min_battery=0, min_battery_limit=15,
	depletion_rate=1, recharge_rate=2,
	velocity=1,
	MAX_STEPS=28800,
	seed=42,
)
env = Env(**ENV_KWARGS)

# ── Dimensions ───────────────────────────────────────────────────────────
N_AGENTS = ENV_KWARGS['n_block']
LOCAL_DIM = 6 + 1 + 2 + (N_AGENTS * 4)
ACTION_DIM = 8
	
# Networks per agent
q_networks = [QNetwork(LOCAL_DIM, ACTION_DIM, 512).to(device) for _ in range(N_AGENTS)]
target_networks = [QNetwork(LOCAL_DIM, ACTION_DIM, 512).to(device) for _ in range(N_AGENTS)]

CKPT_PATH = 'dqn_best.pt'
print(f"[Loading DQN checkpoint: {CKPT_PATH}]")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
for i in range(N_AGENTS):
    q_networks[i].load_state_dict(ckpt[f'q_net_{i}'])
    q_networks[i].eval()

n_test_ep = 10

# Tracking arrays
episode_rewards_arr = []
orders_completed = []
orders_completed1 = []
orders_completed2 = []
orders_completed3 = []
orders_completed4 = []
episode_lengths = []
time_travelled1 = []
time_travelled2 = []
time_travelled3 = []
time_travelled4 = []
total_orders_placed = []
total_orders_placed1 = []
total_orders_placed2 = []
total_orders_placed3 = []
total_orders_placed4 = []
time_spent_charging1 = []
time_spent_charging2 = []
time_spent_charging3 = []
time_spent_charging4 = []
depot_counter_cap_tracker = {}
depot_counter_num_tracker = {}
depot_travel_tracker1 = []
depot_travel_tracker2 = []
depot_travel_tracker3 = []
depot_travel_tracker4 = []
travel_time_pick1 = []
travel_time_pick2 = []
travel_time_pick3 = []
travel_time_pick4 = []
total_travel_time1 = []
total_travel_time2 = []
total_travel_time3 = []
total_travel_time4 = []
wait_in_queue1 = []
wait_in_queue2 = []
wait_in_queue3 = []
wait_in_queue4 = []

for ep in range(n_test_ep):
    obs, info = env.reset()
    episode_rewards = 0.0

    while env.global_time_step < ENV_KWARGS['MAX_STEPS']:
        local_obs_arr = np.array([get_local_obs(obs, env, i) for i in range(1, N_AGENTS + 1)])
        allowed = env.allowed_action()

        # Greedy Q-value selection
        actions = []
        with torch.no_grad():
            for i in range(N_AGENTS):
                obs_t = torch.tensor(local_obs_arr[i], dtype=torch.float32).unsqueeze(0)
                q_vals = q_networks[i](obs_t).squeeze(0)
                # Mask invalid actions
                q_vals_masked = q_vals.clone()
                q_vals_masked[~torch.tensor(allowed[i], dtype=torch.bool)] = -1e9

                # Greedy action
                action = q_vals_masked.argmax().item()
                actions.append(action)

        orders_before = {n: env.completion_info[n][0] for n in range(1, N_AGENTS + 1)}
        next_obs, reward, done, info = env.step(actions)

        episode_rewards += np.sum(reward)
        obs = next_obs

    # Record metrics
    episode_rewards_arr.append(episode_rewards)
    orders_completed.append(env.completion_info[1][0]+env.completion_info[2][0]+env.completion_info[3][0]+env.completion_info[4][0])
    orders_completed1.append(env.completion_info[1][0])
    orders_completed2.append(env.completion_info[2][0])
    orders_completed3.append(env.completion_info[3][0])
    orders_completed4.append(env.completion_info[4][0])
    time_travelled1.append(env.time_travelled[0])
    time_travelled2.append(env.time_travelled[1])
    time_travelled3.append(env.time_travelled[2])
    time_travelled4.append(env.time_travelled[3])
    total_orders_placed.append(env.total_orders_placed[0]+env.total_orders_placed[1]+env.total_orders_placed[2]+env.total_orders_placed[3])
    total_orders_placed1.append(env.total_orders_placed[0])
    total_orders_placed2.append(env.total_orders_placed[1])
    total_orders_placed3.append(env.total_orders_placed[2])
    total_orders_placed4.append(env.total_orders_placed[3])
    time_spent_charging1.append(env.completion_info[1][2])
    time_spent_charging2.append(env.completion_info[2][2])
    time_spent_charging3.append(env.completion_info[3][2])
    time_spent_charging4.append(env.completion_info[4][2])
    travel_time_pick1.append(env.travel_time_pick[1])
    travel_time_pick2.append(env.travel_time_pick[2])
    travel_time_pick3.append(env.travel_time_pick[3])
    travel_time_pick4.append(env.travel_time_pick[4])


    depot_counter_cap_tracker[ep], depot_counter_num_tracker[ep] = {},{}
    for i in range(1,env.n_agent+1):
        depot_counter_cap_tracker[ep][i] = env.depot_counter[i][0]
        depot_counter_num_tracker[ep][i] = env.depot_counter[i][1]

    depot_travel_tracker1.append(env.travel_time_depot[1])
    depot_travel_tracker2.append(env.travel_time_depot[2])
    depot_travel_tracker3.append(env.travel_time_depot[3])
    depot_travel_tracker4.append(env.travel_time_depot[4])
    total_travel_time1.append(env.total_travel_time[1])
    total_travel_time2.append(env.total_travel_time[2])
    total_travel_time3.append(env.total_travel_time[3])
    total_travel_time4.append(env.total_travel_time[4])
    wait_in_queue1.append(env.wait_in_q[1])
    wait_in_queue2.append(env.wait_in_q[2])
    wait_in_queue3.append(env.wait_in_q[3])
    wait_in_queue4.append(env.wait_in_q[4])


result_dict = {
    "mean_reward": np.mean(episode_rewards_arr),
    "std_reward": np.std(episode_rewards_arr),
    "sum_orders_completed": np.sum(orders_completed),
    "sum_orders_completed1": np.sum(orders_completed1),
    "sum_orders_completed2": np.sum(orders_completed2),
    "sum_orders_completed3": np.sum(orders_completed3),
    "sum_orders_completed4": np.sum(orders_completed4),
    "sum_total_orders_placed": np.sum(total_orders_placed),
    "sum_total_orders_placed1": np.sum(total_orders_placed1),
    "sum_total_orders_placed2": np.sum(total_orders_placed2),
    "sum_total_orders_placed3": np.sum(total_orders_placed3),
    "sum_total_orders_placed4": np.sum(total_orders_placed4),
    "order_completed":  np.sum(orders_completed)/np.sum(total_orders_placed),
    "order_completed1":  np.sum(orders_completed1)/np.sum(total_orders_placed1),
    "order_completed2":  np.sum(orders_completed2)/np.sum(total_orders_placed2),
    "order_completed3":  np.sum(orders_completed3)/np.sum(total_orders_placed3),
    "order_completed4":  np.sum(orders_completed4)/np.sum(total_orders_placed4),
    "time_spent_charging": (np.sum(time_spent_charging1)+np.sum(time_spent_charging2)+np.sum(time_spent_charging3)+np.sum(time_spent_charging4)),
    "time_spent_charging1": np.sum(time_spent_charging1),
    "time_spent_charging2": np.sum(time_spent_charging2),
    "time_spent_charging3": np.sum(time_spent_charging3),
    "time_spent_charging4": np.sum(time_spent_charging4),
    "total_travel_time1": np.sum(total_travel_time1),
    "total_travel_time2": np.sum(total_travel_time2),
    "total_travel_time3": np.sum(total_travel_time3),
    "total_travel_time4": np.sum(total_travel_time4),
    "wait_in_q1": np.sum(wait_in_queue1),
    "wait_in_q2": np.sum(wait_in_queue2),
    "wait_in_q3": np.sum(wait_in_queue3),
    "wait_in_q4": np.sum(wait_in_queue4)
}

with open("dqn_testing.log", "w") as f:
    f.write("____DQN_TESTING____\n")
    for k, v in result_dict.items():
        f.write(f"  {k}: {v:.3f}\n")

print("DQN Testing Complete. Results:")
for k, v in result_dict.items():
    print(f"  {k}: {v:.3f}")