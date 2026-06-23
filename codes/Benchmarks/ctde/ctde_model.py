def get_state(obs, env, n):
    next_order = env.next_order_to_pick(n)
    dist_order = (env.func_distance(
        (env.agent_state[n][0], env.agent_state[n][1]),
        (next_order[1], next_order[2])) / env.max_distance
        if next_order else -1)
    dist_cp1 = env.func_distance(
        (env.agent_state[n][0], env.agent_state[n][1]),
        env.charger_location[0]) / env.max_distance
    dist_cp2 = env.func_distance(
        (env.agent_state[n][0], env.agent_state[n][1]),
        env.charger_location[1]) / env.max_distance
    dist_depot = env.func_distance(
        (env.agent_state[n][0], env.agent_state[n][1]),
        env.depot_location) / env.max_distance
    at_cp = env.agent_state[n][4] / env.n_cp
    dist_cap_bat = np.array([
        dist_order, dist_cp1, dist_cp2, dist_depot,
        env.agent_state[n][2] / env.capacity,
        env.agent_state[n][3] / env.max_battery,
    ], dtype=np.float32)
    return dist_cap_bat, np.array([at_cp])


def get_local_obs(obs, env, n):
    dist_cap_bat, at_cp = get_state(obs, env, n)
    agent_id = np.zeros(env.n_agent, dtype=np.float32)
    agent_id[n - 1] = 1.0
    return np.concatenate([dist_cap_bat, at_cp])

def get_global_obs(obs, env):
    all_local = np.concatenate([get_local_obs(obs, env, i) for i in range(1, env.n_agent+1)])
    return np.concatenate([all_local, obs['cp_queue_len'].astype(np.float32)])

class Actor(nn.Module):
    def __init__(self, local_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, local_obs):
        return self.net(local_obs) 


class CentralizedCritic(nn.Module):
    def __init__(self, global_dim, n_agents, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents)
        )
    def forward(self, global_obs):
        return self.net(global_obs)


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.global_obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.action_masks = []

    def add(self, global_obs, actions, log_probs, reward, action_masks):
        self.global_obs.append(global_obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.action_masks.append(action_masks)

    def __len__(self):
        return len(self.rewards)

class AverageRewardBaseline:
    def __init__(self, alpha=0.01):
        self.r_bar = 0.0
        self.alpha = alpha

    def update(self, rewards):
        self.r_bar = (1 - self.alpha) * self.r_bar + self.alpha * rewards.mean()

    def get(self):
        return self.r_bar



def compute_differential_gae(rewards, values, next_value, r_bar, lam=0.95):
    T = len(rewards)
    device = values.device
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    values_ext = torch.cat([values, next_value.unsqueeze(0)])

    advantages = torch.zeros(T, device=device)
    gae = 0.0

    for t in reversed(range(T)):
        # Differential TD error: r_t - r̄ + V(s') - V(s)
        delta = rewards_t[t] - r_bar + values_ext[t+1] - values_ext[t]
        gae = delta + lam * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def ppo_update(actor, critics, actor_opt, critic_opts, buffer, r_bars, n_agents,
               device, lam, clip_eps, entropy_coef, n_actor_epochs, n_critic_epochs,
               mini_batch_size, track_metrics=False):
    T = len(buffer)
    if T == 0:
        return {}

    track = {} if not track_metrics else {
        'advantages_raw': [], 'advantages_norm': [], 'returns': [], 'values': [],
        'old_log_probs': [], 'new_log_probs': [], 'policy_ratios': [],
        'actor_losses': [], 'critic_losses': [], 'entropies': [],
        'actor_grads': [], 'critic_grads': [], 'kl_divergence': []
    }

    global_obs_t = torch.tensor(np.stack(buffer.global_obs), dtype=torch.float32, device=device)
    actions_t = torch.tensor(np.stack(buffer.actions), dtype=torch.long, device=device)
    old_log_probs_t = torch.stack(buffer.log_probs, dim=0).detach()
    rewards_np = np.stack(buffer.rewards, axis=0)  # [T, n_agents]
    masks_t = torch.tensor(np.stack(buffer.action_masks), dtype=torch.bool, device=device)


    LOCAL_DIM = 7
    local_obs_all = [global_obs_t[:, i*LOCAL_DIM:(i+1)*LOCAL_DIM] for i in range(n_agents)]

    with torch.no_grad():
        values_all = critic(global_obs_t)            # [T, n_agents]

    advantages_all, returns_all = [], []
    for agent_id in range(n_agents):
        adv, ret = compute_differential_gae(
            rewards_np[:, agent_id],
            values_all[:, agent_id],
            values_all[-1, agent_id],
            r_bars[agent_id].get(), lam
        )
        r_bars[agent_id].update(torch.tensor(rewards_np[:, agent_id]))
        advantages_all.append(adv); returns_all.append(ret)

    advantages_t = torch.stack(advantages_all, dim=1)   # [T, n_agents]
    returns_t    = torch.stack(returns_all, dim=1)


    adv_flat = advantages_t.reshape(-1)
    adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    advantages_t = adv_norm.view(T, n_agents)


    # Critic training - train each local critic separately
    total_critic_loss = 0
    for _ in range(n_critic_epochs):
        values_pred = critic(global_obs_t)                       # [T, n_agents]
        critic_loss = F.huber_loss(values_pred, returns_t.detach(), delta=1.0)
        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic_opt.step()
        total_critic_loss += critic_loss.item()

    # Actor training - centralized
    total_actor_loss = 0
    total_entropy = 0
    n_updates = 0

    for _ in range(n_actor_epochs):
        actor_loss_sum, entropy_sum, kl_sum = 0, 0, 0
        for i in range(n_agents):
            agent_logits = actor(local_obs_all[i]).masked_fill(~masks_t[:, i, :], -1e9)
            dist = Categorical(logits=agent_logits)
            new_lp = dist.log_prob(actions_t[:, i])
            ratio = torch.exp(new_lp - old_log_probs_t[:, i])
            surr1 = ratio * advantages_t[:, i]
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages_t[:, i]
            actor_loss_sum += -torch.min(surr1, surr2).mean()
            entropy_sum    += dist.entropy().mean()
            kl_sum         += (old_log_probs_t[:, i] - new_lp.detach()).mean().item()
        actor_loss = actor_loss_sum / n_agents - entropy_coef * entropy_sum / n_agents
        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        actor_opt.step()

        total_actor_loss += actor_loss.item()
        total_entropy += (entropy_sum / n_agents).item()
        n_updates += 1

    result = {
        'actor_loss': total_actor_loss / n_updates,
        'critic_loss': total_critic_loss / (n_critic_epochs * n_agents),
        'entropy': total_entropy / n_updates,
        'r_bar_avg': np.mean([r.get() for r in r_bars])
    }

    if track_metrics:
        result['tracking'] = track

    return result