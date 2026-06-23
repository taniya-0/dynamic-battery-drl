###--------------------------------------------------- MAIN TESTING LOOP --------------------------------------------------- ###
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
from ippo_model import get_state, get_local_obs, ActorNetwork, LocalCritic, RolloutBuffer, AverageRewardBaseline, compute_differential_gae, ippo_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# ── Dimensions ───────────────────────────────────────────────────────────
N_AGENTS = ENV_KWARGS['n_block']
LOCAL_DIM = 6 + 1 + 2 + (N_AGENTS * 4)
ACTION_DIM = 8

# ── Networks ─────────────────────────────────────────────────────────────
device = torch.device("cpu")
print(f"Using device: {device}")

actor = ActorNetwork(LOCAL_DIM, ACTION_DIM, N_AGENTS, hidden_dim=512)
critics = [LocalCritic(LOCAL_DIM, hidden_dim=256) for _ in range(N_AGENTS)]

CKPT_PATH = 'best_model.pt'
print(f"[Resume] Loading: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only = False)
actor.load_state_dict(ckpt['actor'])
actor.eval()

# ─────────────────────────────────────────────
#  Evaluation loop
# ─────────────────────────────────────────────

n_test_ep = 10

# Adjust according to number of agents
episode_rewards_arr   = []
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
    obs, info = env.reset()
    done         = False
    episode_rewards = 0.0
    steps        = 0

    cap = []
    bat = []
    stopcharge = []
    cp_state_tracker = {n: [] for n in range(1, 2 + 1)}
    cp_queue_tracker = {n: [] for n in range(1, 2 + 1)}
    which_cp = {1:[0,0], 2:[0,0], 3:[0,0], 4:[0,0], 5:[0,0], 6:[0,0]}
    rewards = []
    orders_compl = []
    while env.global_time_step < ENV_KWARGS['MAX_STEPS']:
        local_obs_arr = np.array([get_local_obs(obs, env, i) for i in range(1, N_AGENTS + 1)])
        allowed = env.allowed_action()

        # Get deterministic actions (greedy policy)
        actions = []
        with torch.no_grad():
            for i in range(N_AGENTS):
                obs_tensor = torch.tensor(local_obs_arr[i], dtype=torch.float32).to(device)
                agent_id_tensor = torch.tensor(i, dtype=torch.long).to(device)
                mask_tensor = torch.tensor(allowed[i], dtype=torch.bool).to(device)

                # Get logits (no sampling, just greedy action)
                logits = actor(obs_tensor.unsqueeze(0), agent_id_tensor.unsqueeze(0)).squeeze(0)
                logits = logits.masked_fill(~mask_tensor, -1e9)

                # Greedy: take argmax action
                action = logits.argmax().item()

                # dist = Categorical(logits=logits)
                # action = dist.sample()
                actions.append(action)


        orders_before = {n: env.completion_info[n][0] for n in range(1, N_AGENTS + 1)}
        next_obs, reward, done, info = env.step(actions)

        episode_rewards += np.sum(reward)
        obs = next_obs
        steps += 1

        for i in range(len(actions)):
          if actions[i] == 4: cap.append(env.agent_state[i+1][2])
          if actions[i] in (1,2) and env.agent_state[i+1][4] == 0:
            bat.append(env.agent_state[i+1][3])
            which_cp[i+1][actions[i]-1] += 1
          if actions[i] == 3 :stopcharge.append(env.agent_state[i+1][3])
        for i in range(1,3):
          cp_state_tracker[i].append(list(env.cp_state[i]))
          cp_queue_tracker[i].append(env.cp_queue_len[i-1])
        rewards.append(reward)
        orders_compl_count = 0
        for i in range(1,env.n_agent+1):
          orders_compl_count += env.completion_info[i][0]
        orders_compl.append(orders_compl_count/np.sum(env.total_orders_placed))

    # Adjust according to number of agents
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

        travel_time_pick1.append(env.travel_time_pick[1])
        travel_time_pick2.append(env.travel_time_pick[2])
        travel_time_pick3.append(env.travel_time_pick[3])
        travel_time_pick4.append(env.travel_time_pick[4])
        travel_time_pick5.append(env.travel_time_pick[5])
        travel_time_pick6.append(env.travel_time_pick[6])


        depot_counter_cap_tracker[ep] = {}
        depot_counter_num_tracker[ep] = {}
        for k in range(1, env.n_agent + 1):
            depot_counter_cap_tracker[ep][k] = env.depot_counter[i][0]
            depot_counter_num_tracker[ep][k] = env.depot_counter[i][1]

        depot_travel_tracker1.append(env.travel_time_depot[1])
        depot_travel_tracker2.append(env.travel_time_depot[2])
        depot_travel_tracker3.append(env.travel_time_depot[3])
        depot_travel_tracker4.append(env.travel_time_depot[4])
        depot_travel_tracker5.append(env.travel_time_depot[5])
        depot_travel_tracker6.append(env.travel_time_depot[6])

        wait_in_queue_tracker1.append(env.wait_in_q[1])
        wait_in_queue_tracker2.append(env.wait_in_q[2])
        wait_in_queue_tracker3.append(env.wait_in_q[3])
        wait_in_queue_tracker4.append(env.wait_in_q[4])
        wait_in_queue_tracker5.append(env.wait_in_q[5])
        wait_in_queue_tracker6.append(env.wait_in_q[6])

# Adjust according to number of agents
if n_test_ep > 1:
    result_dict = {
        "mean_reward":                    np.mean(episode_rewards_arr),
        "std_reward":                     np.std(episode_rewards_arr),
        "sum_reward":                     np.sum(episode_rewards_arr),

        # orders completed
        "mean_orders_completed":          np.mean(orders_completed),
        "mean_orders_completed1":         np.mean(orders_completed1),
        "mean_orders_completed2":         np.mean(orders_completed2),
        "mean_orders_completed3":         np.mean(orders_completed3),
        "mean_orders_completed4":         np.mean(orders_completed4),
        "mean_orders_completed5":         np.mean(orders_completed5),
        "mean_orders_completed6":         np.mean(orders_completed6),
        "sum_orders_completed":           np.sum(orders_completed),
        "sum_orders_completed1":          np.sum(orders_completed1),
        "sum_orders_completed2":          np.sum(orders_completed2),
        "sum_orders_completed3":          np.sum(orders_completed3),
        "sum_orders_completed4":          np.sum(orders_completed4),
        "sum_orders_completed5":          np.sum(orders_completed5),
        "sum_orders_completed6":          np.sum(orders_completed6),

        # episode length
        "mean_episode_length":            np.mean(episode_lengths),

        # distance travelled
        "mean_time_travelled1":       np.mean(time_travelled1),
        "mean_time_travelled2":       np.mean(time_travelled2),
        "mean_time_travelled3":       np.mean(time_travelled3),
        "mean_time_travelled4":       np.mean(time_travelled4),
        "mean_time_travelled5":       np.mean(time_travelled5),
        "mean_time_travelled6":       np.mean(time_travelled6),

        # orders placed
        "mean_total_orders_placed1":      np.mean(total_orders_placed1),
        "mean_total_orders_placed2":      np.mean(total_orders_placed2),
        "mean_total_orders_placed3":      np.mean(total_orders_placed3),
        "mean_total_orders_placed4":      np.mean(total_orders_placed4),
        "mean_total_orders_placed5":      np.mean(total_orders_placed5),
        "mean_total_orders_placed6":      np.mean(total_orders_placed6),

        "sum_total_orders_placed1":       np.sum(total_orders_placed1),
        "sum_total_orders_placed2":       np.sum(total_orders_placed2),
        "sum_total_orders_placed3":       np.sum(total_orders_placed3),
        "sum_total_orders_placed4":       np.sum(total_orders_placed4),
        "sum_total_orders_placed5":       np.sum(total_orders_placed5),
        "sum_total_orders_placed6":       np.sum(total_orders_placed6),

        # depot travel time
        "mean_depot_travel1":             np.mean(depot_travel_tracker1),
        "mean_depot_travel2":             np.mean(depot_travel_tracker2),
        "mean_depot_travel3":             np.mean(depot_travel_tracker3),
        "mean_depot_travel4":             np.mean(depot_travel_tracker4),
        "mean_depot_travel5":             np.mean(depot_travel_tracker5),
        "mean_depot_travel6":             np.mean(depot_travel_tracker6),


        "order_completion_rate_overall":  100*np.sum(orders_completed) / (np.sum(total_orders_placed1) + np.sum(total_orders_placed2) + np.sum(total_orders_placed3) + np.sum(total_orders_placed4) + np.sum(total_orders_placed5) + np.sum(total_orders_placed6)),
        "order_completion_rate1":         100*np.mean(orders_completed1) / np.mean(total_orders_placed1),
        "order_completion_rate2":         100*np.mean(orders_completed2) / np.mean(total_orders_placed2),
        "order_completion_rate3":         100*np.mean(orders_completed3) / np.mean(total_orders_placed3),
        "order_completion_rate4":         100*np.mean(orders_completed4) / np.mean(total_orders_placed4),
        "order_completion_rate5":         100*np.mean(orders_completed5) / np.mean(total_orders_placed5),
        "order_completion_rate6":         00*np.mean(orders_completed6) / np.mean(total_orders_placed6),

        # time spent charging
        "mean_time_spent_charging1":      np.mean(time_spent_charging1),
        "mean_time_spent_charging2":      np.mean(time_spent_charging2),
        "mean_time_spent_charging3":      np.mean(time_spent_charging3),
        "mean_time_spent_charging4":      np.mean(time_spent_charging4),
        "mean_time_spent_charging5":      np.mean(time_spent_charging5),
        "mean_time_spent_charging6":      np.mean(time_spent_charging6),

        # travel time picking
        "mean_travel_time_pick1":         np.mean(travel_time_pick1),
        "mean_travel_time_pick2":         np.mean(travel_time_pick2),
        "mean_travel_time_pick3":         np.mean(travel_time_pick3),
        "mean_travel_time_pick4":         np.mean(travel_time_pick4),
        "mean_travel_time_pick5":         np.mean(travel_time_pick5),
        "mean_travel_time_pick6":         np.mean(travel_time_pick6),

        "wait_in_queue1":                 np.mean(wait_in_queue_tracker1),
        "wait_in_queue2":                 np.mean(wait_in_queue_tracker2),
        "wait_in_queue3":                 np.mean(wait_in_queue_tracker3),
        "wait_in_queue4":                 np.mean(wait_in_queue_tracker4),
        "wait_in_queue5":                 np.mean(wait_in_queue_tracker5),
        "wait_in_queue6":                 np.mean(wait_in_queue_tracker6),
        }
    with open("training.log", "a") as f:
        f.write("\n____TESTING__DETERMINISTIC___\n")
        for k, v in result_dict.items():
            if "raw" not in k:                  # skip raw lists, print scalars only
                line = f"  {k}: {v:.3f}\n"
                f.write(line)