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

if __name__ == "__main__":

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

    LOCAL_DIM = 7  
    ACTION_DIM = 8  
    N_AGENTS = ENV_KWARGS['n_block']
    GLOBAL_DIM = LOCAL_DIM*N_AGENTS + ENV_KWARGS['n_cp']

    # ── Hyperparameters ──────────────────────────────────────────────────────
    EPOCHS = 10000
    STEPS = ENV_KWARGS['MAX_STEPS']
    LAM = 0.95
    CLIP_EPS = 0.2          # Higher for stable local critics
    N_ACTOR_EPOCHS = 3      # More epochs OK with stable critics
    N_CRITIC_EPOCHS = 5     # Fewer - local critics converge fast
    MINI_BATCH = 512        # Larger batches for stability

    # ── Networks ─────────────────────────────────────────────────────────────
    device = torch.device("cpu")
    print(f"Using device: {device}")

    actor  = SharedActor(LOCAL_DIM, ACTION_DIM, hidden=256).to(device)
    critic = CentralizedCritic(GLOBAL_DIM, N_AGENTS, hidden=512).to(device)
    actor_opt  = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=3e-4)

    r_bars = [AverageRewardBaseline(alpha=0.01) for _ in range(N_AGENTS)]

    # ── State ────────────────────────────────────────────────────────────────
    start_ep = 0
    output = {
        'epoch': [], 'reward': [], 'completion_pct': [],
        'actor_loss': [], 'critic_loss': [], 'entropy': [], 'r_bar_avg': []
    }

    tracking_data = {
        'epoch': [],
        'advantages_raw': [],
        'advantages_norm': [],
        'returns': [],
        'values': [],
        'old_log_probs': [],
        'new_log_probs': [],
        'policy_ratios': [],
        'actor_losses': [],
        'critic_losses': [],
        'entropies': [],
        'actor_grads': [],
        'critic_grads': [],
        'kl_divergence': [],
    }

    # ── Checkpoint resume ────────────────────────────────────────────────────
    CKPT_PATH = "ppo.pt"
    CKPT_PATH_BEST = "best_model.pt"
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt['actor'])
        for i in range(N_AGENTS):
            critic.load_state_dict(ckpt[f'critic'])
            critic_opt.load_state_dict(ckpt[f'critic_opt'])
            r_bars[i].r_bar = ckpt[f'r_bar_{i}']
        actor_opt.load_state_dict(ckpt['actor_opt'])
        start_ep = ckpt['epoch'] + 1
        output = ckpt['output']
        tracking_data = ckpt['tracking_data']
        print(f"[Resume] From episode {start_ep}")
    else:
        print("[Fresh start]")

    # ── Training loop ────────────────────────────────────────────────────────
    start_time = time.time()

    best_compl_rate = 0
    for ep in range(start_ep, EPOCHS):
        entropy_coef = max(0.005, 0.1 * (1 - ep / 1500))

        obs, _ = env.reset()
        buffer = RolloutBuffer()
        ep_reward = 0.0


        # Collect episode
        while env.global_time_step < STEPS:
            global_obs = get_global_obs(obs, env)
            local_obs  = np.stack([get_local_obs(obs, env, i) for i in range(1, N_AGENTS+1)])
            allowed = env.allowed_action()

            with torch.no_grad():
                logits = actor(torch.tensor(local_obs, dtype=torch.float32))

            actions, log_probs = [], []
            for i in range(N_AGENTS):
                agent_logits = logits[i].masked_fill(~torch.tensor(allowed[i], dtype=torch.bool), -1e9)
                dist = Categorical(logits=agent_logits)
                a = dist.sample()
                actions.append(a.item())
                log_probs.append(dist.log_prob(a))

            orders_before = {n: env.completion_info[n][0] for n in range(1, N_AGENTS+1)}
            next_obs, reward, done, info = env.step(actions)


            ep_reward += float(np.sum(reward))
            buffer.add(global_obs, actions, torch.stack(log_probs), reward, allowed)
            obs = next_obs

        # PPO update
        track_this_ep = (ep % 10 == 0)
        losses = ppo_update(
            actor, critic, actor_opt, critic_opt,
            buffer, r_bars, n_agents=N_AGENTS, device=device,
            lam=LAM, clip_eps=CLIP_EPS, entropy_coef=entropy_coef,
            n_actor_epochs=N_ACTOR_EPOCHS, n_critic_epochs=N_CRITIC_EPOCHS,
            mini_batch_size=MINI_BATCH, track_metrics=track_this_ep,
        )

        # Logging every 5 episodes
        total_completed = sum(env.completion_info[n][0] for n in range(1, N_AGENTS+1))
        total_placed = sum(env.total_orders_placed)
        completion_pct = 100 * total_completed / max(total_placed, 1)
        if completion_pct > best_compl_rate:  # Changed: was int()
            best_compl_rate = completion_pct
            ckpt_dict_best = {
                'epoch': ep,
                'actor': actor.state_dict(),
                'actor_opt': actor_opt.state_dict(),
                'output': output,
                'tracking_data': tracking_data,
            }
            for i in range(N_AGENTS):
                ckpt_dict_best[f'critic'] = critic.state_dict()
                ckpt_dict_best[f'critic_opt'] = critic_opt.state_dict()
                ckpt_dict_best[f'r_bar_{i}'] = r_bars[i].r_bar
            torch.save(ckpt_dict_best, CKPT_PATH_BEST)
            print(f"✓ New best checkpoint saved at best completion rate {best_compl_rate}")

        if ep % 5 == 0:

            output['epoch'].append(ep)
            output['reward'].append(ep_reward)
            output['completion_pct'].append(completion_pct)
            output['actor_loss'].append(losses.get('actor_loss', 0))
            output['critic_loss'].append(losses.get('critic_loss', 0))
            output['entropy'].append(losses.get('entropy', 0))
            output['r_bar_avg'].append(losses.get('r_bar_avg', 0))

            # Store tracking data
            if 'tracking' in losses and losses['tracking']:
                tracking_data['epoch'].append(ep)
                track = losses['tracking']
                for key in tracking_data.keys():
                    if key != 'epoch' and key in track:
                        tracking_data[key].append(track[key])

            elapsed = time.time() - start_time
            print(f"Ep {ep:4d} | Reward {ep_reward:8.0f} | Comp {completion_pct:5.1f}% | "
                  f"ActL {losses.get('actor_loss',0):.3f} | CritL {losses.get('critic_loss',0):.3f} | "
                  f"Ent {losses.get('entropy',0):.3f} | r̄ {losses.get('r_bar_avg',0):.1f} | {elapsed:.0f}s")

        # Checkpointing every 100 episodes
        if ep % 50 == 0 and ep > 0:
            ckpt_dict = {
                'epoch': ep,
                'actor': actor.state_dict(),
                'actor_opt': actor_opt.state_dict(),
                'output': output,
                'tracking_data': tracking_data,
            }
            ckpt_dict[f'critic'] = critic.state_dict()
            ckpt_dict[f'critic_opt'] = critic_opt.state_dict()
            ckpt_dict[f'r_bar_{i}'] = r_bars[i].r_bar

            torch.save(ckpt_dict, CKPT_PATH)

            # Also save tracking as JSON for easy plotting
            with open('ppo_tracking.json', 'w') as f:
                json.dump(convert_json_safe(tracking_data), f)

            print(f"✓ Checkpoint saved")

    print(f"\nDone. Total time: {time.time() - start_time:.0f}s")