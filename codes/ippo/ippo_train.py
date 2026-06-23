###--------------------------------------------------- MAIN TRAINING LOOP --------------------------------------------------- ###
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
from ippo_model import get_state, get_local_obs, ActorNetwork, LocalCritic, RolloutBuffer, AverageRewardBaseline, compute_differential_gae, compute_discounted_gae, ippo_update
                                                                                                               # compute_discounted_gae is used only to compare the difference in performance
if __name__ == "__main__":

    # ── Environment setup - Input environment parameters here ──────────────────────────────────────────────────────
    
    # n_row: Number of storage locations within a storage rack vertically in the environment (Y- axis range)
    # n_aisle: Number of aisles totally in the environment (X-axis range)
    # n_cp: Number of charging stations (CS) or charging point (CP)
    # n_block: Number of blocks / agents
    # arrival_rate: Order arrival rate that follows a poisson process
    # capacity: Maximum carrying capacity of any agent
    # max_battery: Maximum battery level (unit)
    # min_battery: Minimum battery level physically possible, always zero
    # min_battery_limit: Minimum battery threshold (unit)
    # depletion_rate: Battery depletion rate (unit/sec)
    # recharge_rate: Battery re-charge rate (unit/sec)
    # velocity: Velocity of the agent (unit/sec)
    # MAX_STEPS: Training time T of the simulation

    ENV_KWARGS = dict(
        n_row=24, n_aisle=8, n_cp=2, n_block=6,
        arrival_rate=0.75, capacity=10,
        max_battery=100, min_battery=0, min_battery_limit=20,
        depletion_rate=1, recharge_rate=2,
        velocity=1,
        MAX_STEPS=14400,
        seed=42,
    )
    env = Env(**ENV_KWARGS)

    # ── Dimensions ──────────────────────────────────────────────────────
    N_AGENTS = ENV_KWARGS['n_block']
    LOCAL_DIM = 6 + 1 + 2 + (N_AGENTS * 4)  # As per def get_local_obs; distances_capacity_batery(6) + service mode(1) + cp_queue(2) + other_agents(6*4)
    ACTION_DIM = 8   # [pick, cp1, cp2, stop_charging_index, go_to_depot_index, wait_in_queue, keep_charging, travelling]


    # ── Hyperparameters ──────────────────────────────────────────────────────
    EPOCHS = 10000
    STEPS = ENV_KWARGS['MAX_STEPS']
    LAM = 0.98
    CLIP_EPS = 0.208
    N_ACTOR_EPOCHS = 2
    N_CRITIC_EPOCHS = 3
    MINI_BATCH = 512

    # ── Networks ─────────────────────────────────────────────────────────────
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Shared actor + independent critics
    actor = ActorNetwork(LOCAL_DIM, ACTION_DIM, N_AGENTS, hidden_dim=512)
    critics = [LocalCritic(LOCAL_DIM, hidden_dim=256) for _ in range(N_AGENTS)]

    actor_opt = optim.Adam(actor.parameters(), lr=1.2e-4)
    critic_opts = [optim.Adam(c.parameters(), lr=4.9e-4) for c in critics]

    # Average reward baselines per agent
    r_bars = [AverageRewardBaseline(alpha=0.0117) for _ in range(N_AGENTS)]

    # ── Tracking ────────────────────────────────────────────────────────────────
    start_ep = 0
    output = {
        'epoch': [], 'reward': [], 'completion_pct': [],
        'actor_loss': [], 'critic_loss': [], 'entropy': [], 'r_bar_avg': [], 'completion_pct1':[], 'completion_pct2':[], 'completion_pct3':[], 'completion_pct4':[], 'completion_pct5':[], 'completion_pct6':[]
    }

    tracking_data = {'epoch': [], 'actor_losses': [],'critic_losses': [],'entropies': []}

    # ── Checkpoint resume ────────────────────────────────────────────────────
    CKPT_PATH = "ippo.pt"
    CKPT_PATH_BEST = "best_model.pt"
    if os.path.exists(CKPT_PATH):
        print(f"[Resume] Loading: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only = False)
        actor.load_state_dict(ckpt['actor'])
        for i in range(N_AGENTS):
            critics[i].load_state_dict(ckpt[f'critic_{i}'])
            r_bars[i].r_bar = ckpt[f'r_bar_{i}']
        actor_opt.load_state_dict(ckpt['actor_opt'])
        for i in range(N_AGENTS):
            critic_opts[i].load_state_dict(ckpt[f'critic_opt_{i}'])
        start_ep = ckpt['epoch'] + 1
        output = ckpt['output']
        tracking_data = ckpt.get('tracking_data', tracking_data)
        print(f"[Resume] From episode {start_ep}")
    else:
        print("[Fresh start]")

    # ── Training loop ────────────────────────────────────────────────────────
    start_time = time.time()

    best_compl_rate = 0
    for ep in range(start_ep, EPOCHS):
        entropy_coef = max(0.01, 0.164 * (1 - ep / 7000))
        obs, _ = env.reset()
        buffer = RolloutBuffer()
        ep_reward = 0.0


        # Collect episode
        print_ = 0
        while env.global_time_step < STEPS:
            local_obs_arr = np.array([get_local_obs(obs, env, i) for i in range(1, N_AGENTS+1)])
            allowed = env.allowed_action()

            actions = []
            log_probs = []
            for i in range(N_AGENTS):
                a, lp, _ = actor.get_action(
                    torch.tensor(local_obs_arr[i], dtype=torch.float32).to(device),
                    torch.tensor(i, dtype=torch.long).to(device),
                    torch.tensor(allowed[i], dtype=torch.bool).to(device),
                )
                actions.append(a)
                log_probs.append(lp)


            orders_before = {n: env.completion_info[n][0] for n in range(1, N_AGENTS+1)}
            next_obs, reward, done, info = env.step(actions)
            ep_reward += float(np.sum(reward))

            buffer.add(
                local_obs=local_obs_arr,
                actions=np.array(actions),
                log_probs=torch.stack(log_probs),
                reward=reward.copy(),
                action_masks=allowed.astype(bool),
            )

            obs = next_obs

        # PPO update
        track_this_ep = (ep % 10 == 0)
        losses = ippo_update(
            actor, critics, actor_opt, critic_opts,
            buffer, r_bars, n_agents=N_AGENTS, device=device,
            lam=LAM, clip_eps=CLIP_EPS, entropy_coef=entropy_coef,
            n_actor_epochs=N_ACTOR_EPOCHS, n_critic_epochs=N_CRITIC_EPOCHS,
            mini_batch_size=MINI_BATCH, track_metrics=track_this_ep,
        )

        # Logging every 50 episodes
        total_completed = sum(env.completion_info[n][0] for n in range(1, N_AGENTS+1))
        total_placed = sum(env.total_orders_placed)
        completion_pct = 100 * total_completed / max(total_placed, 1)
        completion_pct1 = 100 * env.completion_info[1][0] / max(env.total_orders_placed[0], 1)
        completion_pct2 = 100 * env.completion_info[2][0] / max(env.total_orders_placed[1], 1)
        completion_pct3 = 100 * env.completion_info[3][0] / max(env.total_orders_placed[2], 1)
        completion_pct4 = 100 * env.completion_info[4][0] / max(env.total_orders_placed[3], 1)
        completion_pct5 = 100 * env.completion_info[5][0] / max(env.total_orders_placed[4], 1)  # Adjust according to number of agents
        completion_pct6 = 100 * env.completion_info[6][0] / max(env.total_orders_placed[5], 1)  # Adjust according to number of agents
        if completion_pct > best_compl_rate:
            best_compl_rate = completion_pct
            torch.save({
                'epoch': ep,
                'completion': completion_pct,
                'actor': actor.state_dict(),
                **{f'critic_{i}': critics[i].state_dict() for i in range(N_AGENTS)}
            }, 'best_model.pt')
            with open("training.log", "a") as f:
                f.write(f"✓ New best: {completion_pct:.1f}\n")

        if ep % 50 == 0:

            output['epoch'].append(ep)
            output['reward'].append(ep_reward)
            output['completion_pct'].append(completion_pct)
            output['actor_loss'].append(losses.get('actor_loss', 0))
            output['critic_loss'].append(losses.get('critic_loss', 0))
            output['entropy'].append(losses.get('entropy', 0))
            output['r_bar_avg'].append(losses.get('r_bar_avg', 0))
            output['completion_pct1'].append(completion_pct1)
            output['completion_pct2'].append(completion_pct2)
            output['completion_pct3'].append(completion_pct3)
            output['completion_pct4'].append(completion_pct4)
            output['completion_pct5'].append(completion_pct5)  # Adjust according to number of agents
            output['completion_pct6'].append(completion_pct6)  # Adjust according to number of agents

            # Store tracking data
            if 'tracking' in losses and losses['tracking']:
                tracking_data['epoch'].append(ep)
                track = losses['tracking']
                for key in tracking_data.keys():
                    if key != 'epoch' and key in track:
                        tracking_data[key].append(track[key])

            elapsed = time.time() - start_time
            with open("training.log", "a") as f:
                f.write(f"Ep {ep:4d} | Reward {ep_reward:8.0f} | Comp {completion_pct:5.1f}% | "
                  f"ActL {losses.get('actor_loss',0):.3f} | CritL {losses.get('critic_loss',0):.3f} | "
                  f"Ent {losses.get('entropy',0):.3f} | r̄ {losses.get('r_bar_avg',0):.1f} | {elapsed:.0f}s\n")

        # Checkpointing every 100 episodes
        if ep % 50 == 0 and ep > 0:
            ckpt_dict = {
                'epoch': ep,
                'actor': actor.state_dict(),
                'actor_opt': actor_opt.state_dict(),
                'output': output,
                'tracking_data': tracking_data,
            }
            for i in range(N_AGENTS):
                ckpt_dict[f'critic_{i}'] = critics[i].state_dict()
                ckpt_dict[f'critic_opt_{i}'] = critic_opts[i].state_dict()
                ckpt_dict[f'r_bar_{i}'] = r_bars[i].r_bar
            torch.save(ckpt_dict, CKPT_PATH)

            # Also save tracking as JSON
            with open('ippo_tracking.json', 'w') as f:
                tracking_json = {}
                for k, v in tracking_data.items():
                    if isinstance(v, list) and len(v) > 0:
                        if isinstance(v[0], np.ndarray):
                            tracking_json[k] = [arr.tolist() for arr in v]
                        else:
                            tracking_json[k] = v
                    else:
                        tracking_json[k] = v
                json.dump(tracking_json, f)

            print(f"✓ Checkpoint saved")

    print(f"\nDone. Total time: {time.time() - start_time:.0f}s")
    with open("training.log", "a") as f:
        f.write(f"\nDone. Total time: {time.time() - start_time:.0f}s")

with open("training.log", "a") as f:
        f.write(f"\nDone. Total time: {time.time() - start_time:.0f}s")