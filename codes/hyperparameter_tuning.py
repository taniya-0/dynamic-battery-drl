import logging
from datetime import datetime
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
import optuna
from multiprocessing import freeze_support
from optuna.storages import JournalStorage
import os
import torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from environment import Env
from ippo_model import get_state, get_local_obs, ActorNetwork, LocalCritic, RolloutBuffer, AverageRewardBaseline, compute_differential_gae, compute_discounted_gae, ippo_update
             
####-----------------------------HYPERPAREMETER TUNING USING OPTUNA----------------------------------####			 
def log_callback(study, trial):
    """Called after every trial completes or is pruned."""
    elapsed = (datetime.now() - study_start_time).seconds
    status = "PRUNED" if trial.state == optuna.trial.TrialState.PRUNED else "DONE"

    with open("optuna_progress.log", "a") as f:
        f.write(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Trial {trial.number:3d} | {status:6s} | "
            f"Value: {trial.value if trial.value else 'N/A':>6} | "
            f"Best so far: {study.best_value:.2f}% | "
            f"Params: {trial.params}\n"
        )
		

def objective(trial):
    lr_actor      = trial.suggest_float("lr_actor",    1e-4, 5e-4, log=True)
    lr_critic     = trial.suggest_float("lr_critic",   1e-4, 1e-3, log=True)
    clip_eps      = trial.suggest_float("clip_eps",    0.1,  0.3)
    lam           = trial.suggest_float("lam",         0.9,  0.99)
    alpha_rbar    = trial.suggest_float("alpha_rbar",  0.01, 0.1, log=True)
    n_actor_ep    = trial.suggest_int  ("n_actor_ep",  1,    4)
    n_critic_ep   = trial.suggest_int  ("n_critic_ep", 1,    10)
    entropy_start = trial.suggest_float("entropy_start", 0.05, 0.2)

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

    N_AGENTS = ENV_KWARGS['n_block']
    LOCAL_DIM = 6 + 1 + 2 + (N_AGENTS * 4)  # As per def get_local_obs; distances_capacity_batery(6) + service mode(1) + cp_queue(2) + other_agents(6*4)
    ACTION_DIM = 8   # [pick, cp1, cp2, stop_charging_index, go_to_depot_index, wait_in_queue, keep_charging, travelling]

    EPOCHS     = 2500        # reduced; enough to distinguish good vs bad configs
    STEPS      = ENV_KWARGS['MAX_STEPS']
    REPORT_EVERY = 100       # how often to report to Optuna for pruning

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    actor       = ActorNetwork(LOCAL_DIM, ACTION_DIM, N_AGENTS, hidden_dim=512).to(device)
    critics     = [LocalCritic(LOCAL_DIM, hidden_dim=256).to(device) for _ in range(N_AGENTS)]
    actor_opt   = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_opts = [optim.Adam(c.parameters(),    lr=lr_critic) for c in critics]
    r_bars      = [AverageRewardBaseline(alpha=alpha_rbar) for _ in range(N_AGENTS)]


    best_compl_rate  = 0.0
    recent_completions = []  # rolling window for smoother pruning signal

    for ep in range(EPOCHS):
        entropy_coef = max(0.01, entropy_start * (1 - ep / 7000))

        obs, _ = env.reset()
        buffer  = RolloutBuffer()

        while env.global_time_step < STEPS:
            local_obs_arr = np.array([get_local_obs(obs, env, i) for i in range(1, N_AGENTS+1)])
            allowed = env.allowed_action()

            actions, log_probs = [], []
            for i in range(N_AGENTS):
                a, lp, _ = actor.get_action(
                    torch.tensor(local_obs_arr[i], dtype=torch.float32).to(device),
                    torch.tensor(i, dtype=torch.long).to(device),
                    torch.tensor(allowed[i], dtype=torch.bool).to(device),
                )
                actions.append(a)
                log_probs.append(lp)

            orders_before = {n: env.orders_completed[n][0] for n in range(1, N_AGENTS+1)}
            next_obs, reward, done, info = env.step(actions)

            for n in range(1, N_AGENTS+1):
                newly = env.orders_completed[n][0] - orders_before[n]
                if newly > 0:
                    reward[n-1] += newly * 4.3

            buffer.add(
                local_obs=local_obs_arr,
                actions=np.array(actions),
                log_probs=torch.stack(log_probs),
                reward=reward.copy(),
                action_masks=allowed.astype(bool),
            )
            obs = next_obs

        losses = ippo_update(
            actor, critics, actor_opt, critic_opts,
            buffer, r_bars, n_agents=N_AGENTS, device=device,
            lam=lam, clip_eps=clip_eps, entropy_coef=entropy_coef,
            n_actor_epochs=n_actor_ep, n_critic_epochs=n_critic_ep,
            mini_batch_size=512, track_metrics=False,  
        )

        total_completed = sum(env.orders_completed[n][0] for n in range(1, N_AGENTS+1))
        total_placed    = sum(env.total_orders_placed)
        completion_pct  = 100 * total_completed / max(total_placed, 1)
        best_compl_rate = max(best_compl_rate, completion_pct)

        # ── Report every N episodes so pruner can kill bad trials early ──────
        if ep % REPORT_EVERY == 0 and ep > 0:
            recent_completions.append(completion_pct)
            smoothed = np.mean(recent_completions[-5:])  # smooth last 5 reports
            trial.report(smoothed, step=ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_compl_rate


if __name__ == "__main__":
    freeze_support()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    #storage = JournalStorage(JournalFileBackend("optuna_journal.log"))
    storage = optuna.storages.RDBStorage(
        "sqlite:///optuna_ippo.db",
        engine_kwargs={"connect_args": {"timeout": 30}}
    )
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=500,
            interval_steps=100,
        ),
        storage=storage,
        study_name="ippo_tuning",
        load_if_exists=True,   # ← resumes automatically from DB
    )

    # Log how many trials already done before resuming
    study_start_time = datetime.now()
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned    = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    with open("optuna_progress.log", "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Resumed at {study_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Trials already done: {completed} complete, {pruned} pruned\n")
        if completed > 0:
            f.write(f"Best so far: {study.best_value:.2f}% with {study.best_params}\n")
        f.write(f"{'='*60}\n")

    study.optimize(
        objective,
        n_trials=50,
        n_jobs=6,               # 4 parallel trials
        callbacks=[log_callback]
    )

    print("Best params:", study.best_params)
    print("Best value:",  study.best_value)
    with open("optuna_progress.log", "a") as f:
        f.write(f"\nFINAL BEST: {study.best_value:.2f}%\n")
        f.write(f"FINAL PARAMS: {study.best_params}\n")
    with open("results.log", "w") as f:
        f.write(f"Best params: {study.best_params}\n")
        f.write(f"Best value:  {study.best_value}\n")