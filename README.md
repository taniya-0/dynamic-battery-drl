# Deep Reinforcement Learning for Dynamic Battery Management of Autonomous Order Pickers

## Overview

This repository contains the code developed for dynamic battery management for Autonomous Mobile Robots (AMRs) operating in warehouse environments. A PPO-based Deep Reinforcement Learning framework optimizes dynamic battery charging for warehouse AMRs, learning optimal station selection and duration. The model outperforms benchmarks by 5-6% in order completion, reduces charging time, and ensures robust, coordinated performance. The repository also includes benchmark implementations used for comparison, including Independent PPO (IPPO), Deep Q-Networks (DQN), Centralized Training with Decentralized Execution (CTDE), a replication of the Bischoff et al. in "Reinforcement Learning for AMR Charging Decisions: The Impact of Reward and Action Space Design" (2025) strategy, and a heuristic charging approach.

---

## Repository Structure

### Core Environment

* `codes/environment.py`

  Main warehouse simulation environment used by:

  * IPPO
  * DQN
  * CTDE
  * Hyperparameter tuning experiments

* `codes/hyperparameter_tuning.py`

  Hyperparameter optimization using Optuna.

---

## Models Used

### Main Model: Independent PPO (IPPO)

Directory: `codes/ippo/`

Contains:

* `ippo_model.py` – PPO network architecture
* `ippo_train.py` – Training script
* `ippo_test.py` – Evaluation script

### Deep Q-Network (DQN)

Directory: `codes/Benchmarks/dqn/`

Contains:

* `dqn_model.py` – DQN architecture
* `dqn_train.py` – Training script
* `dqn_test.py` – Evaluation script

### CTDE (Centralized Training, Decentralized Execution)

Directory: `codes/Benchmarks/ctde/`

Contains:

* `ctde_model.py` – CTDE architecture
* `ctde_train.py` – Training script
* `ctde_test.py` – Evaluation script

### Bischoff Strategy

File: `codes/Benchmarks/bischoff.py`

Implementation of the charging strategy proposed by Bischoff et al., used as a benchmark in the experimental study.

### Heuristic Strategy

File: `codes/Benchmarks/heuristic_strategy.py`

Rule-based charging policy used as a benchmark.

---

## Environment Notes

The files

* `ippo_train.py`
* `dqn_train.py`
* `ctde_train.py`
* `hyperparameter_tuning.py`

all use the common warehouse environment defined in:

```text
codes/environment.py
```

The Bischoff replication and heuristic benchmark use customized environment implementations that are defined directly within their respective source files.

---

## Citation

If you use this repository in your research, please cite the associated thesis or publication.
