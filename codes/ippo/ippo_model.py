###--------------------------------------------------- GETIING OBSERVATIONS FOR NETWORK TRAINING ---------------------------------------------------###

# Get local observations of agent n
def get_state(obs, env, n):
    # Agent n's own observations
    next_order = env.next_order_to_pick(n)
    dist_order = (env.func_distance((env.agent_state[n][0], env.agent_state[n][1]),(next_order[1], next_order[2])) / env.max_distance if next_order else -1)
    dist_cp1 = env.func_distance((env.agent_state[n][0], env.agent_state[n][1]),env.charger_location[0]) / env.max_distance
    dist_cp2 = env.func_distance((env.agent_state[n][0], env.agent_state[n][1]),env.charger_location[1]) / env.max_distance
    dist_depot = env.func_distance((env.agent_state[n][0], env.agent_state[n][1]),env.depot_location) / env.max_distance
    at_cp = env.agent_state[n][4] / env.n_cp
    dist_cap_bat = np.array([
        dist_order, dist_cp1, dist_cp2, dist_depot,
        env.agent_state[n][2] / env.capacity,
        env.agent_state[n][3] / env.max_battery,
    ], dtype=np.float32)

    # Getting information of other agents
    agent_id = n
    my_block = n
    my_pos = [env.agent_state[agent_id][0], env.agent_state[agent_id][1]]
    other_features = []
    for other_id in range(1, env.n_agent+1):
        if other_id == agent_id:other_features.extend([0, 0, 0, 0])  # Pad with zeros for self
        else:
            other_pos = [env.agent_state[other_id][0], env.agent_state[other_id][1]]
            my_pos = [env.agent_state[agent_id][0], env.agent_state[agent_id][1]]

            other_features.extend([
                env.agent_state[other_id][3] / 100,           # battery
                env.agent_state[other_id][2] / env.capacity,  # capacity
                env.agent_state[other_id][4]/ env.n_cp,       # service mode
                env.func_distance(my_pos, other_pos) / env.max_distance
            ])

    return dist_cap_bat, np.array([at_cp]), np.array(other_features)


def get_local_obs(obs, env, n):
    dist_cap_bat, at_cp, other_features = get_state(obs, env, n)
    cp_queue = obs['cp_queue_len'].astype(np.float32)
    return np.concatenate([dist_cap_bat, at_cp, cp_queue, other_features])


###--------------------------------------------------- NETWORK ARCHITECTURE ---------------------------------------------------###

# Shared actor network: Takes local observations + one hot encoded agent id
class ActorNetwork(nn.Module):
    def __init__(self, local_obs_dim, action_dim, n_agents, hidden_dim=512):
        super().__init__()
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim + n_agents, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, obs, agent_id):
        agent_onehot = F.one_hot(agent_id, num_classes=self.n_agents).float().to(obs.device)
        return self.net(torch.cat([obs, agent_onehot], dim=-1))

    def get_action(self, obs, agent_id, action_mask):
        logits = self.forward(obs.unsqueeze(0), agent_id.unsqueeze(0)).squeeze(0)
        logits = logits.masked_fill(~action_mask, -1e9)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate(self, obs_batch, agent_ids, actions, action_masks):
        logits = self.forward(obs_batch, agent_ids)
        all_masked = ~action_masks.any(dim=-1)
        safe_masks = action_masks.clone()
        safe_masks[all_masked, 0] = True
        logits = logits.masked_fill(~safe_masks, -1e9)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions).masked_fill(all_masked, 0.0)
        entropies = dist.entropy().masked_fill(all_masked, 0.0)
        return log_probs, entropies

# Individual critic network for each agent, takes local obs for each agent
class LocalCritic(nn.Module):
    def __init__(self, local_obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim), 
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, local_obs):
        return self.net(local_obs).squeeze(-1)


###--------------------------------------------------- ROLLOUT BUFFER ---------------------------------------------------###
class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.local_obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.action_masks = []

    def add(self, local_obs, actions, log_probs, reward, action_masks):
        self.local_obs.append(local_obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.action_masks.append(action_masks)

    def __len__(self):
        return len(self.rewards)


###--------------------------------------------------- AVERAGE REWARD BASELINE ---------------------------------------------------###
class AverageRewardBaseline:
    def __init__(self, alpha=0.01):
        self.r_bar = 0.0
        self.alpha = alpha

    # Updated running average r̄
    def update(self, rewards):
        self.r_bar = (1 - self.alpha) * self.r_bar + self.alpha * rewards.mean()

    def get(self):
        return self.r_bar


###--------------------------------------------------- DIFFERENTIAL GAE (Average Reward Formulation) ------------------------------###

def compute_differential_gae(rewards, values, next_value, r_bar, lam=0.95):
    """
    Args:
        rewards: [T] array of rewards
        values: [T] tensor of value predictions
        next_value: scalar next value (for bootstrapping)
        r_bar: scalar average reward baseline
        lam: GAE lambda parameter

    Returns:
        advantages: [T] tensor
        returns: [T] tensor (advantages + values)
    """
    T = len(rewards)
    device = values.device
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    values_ext = torch.cat([values, next_value.unsqueeze(0)])

    advantages = torch.zeros(T, device=device)
    gae = 0.0

    for t in reversed(range(T)):
        # Differential TD error: r_t - r̄ + V(s_{t+1}) - V(s_t)
        delta = rewards_t[t] - r_bar + values_ext[t+1] - values_ext[t]
        # GAE calculation
        gae = delta + lam * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns

###--------------------------------------------------- DISCOUNTED GAE (Discounted Reward Formulation) ------------------------------###

def compute_discounted_gae(rewards, values, next_value, r_bar, lam=0.95):
    T = len(rewards)
    device = values.device
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    values_ext = torch.cat([values, next_value.unsqueeze(0)])

    advantages = torch.zeros(T, device=device)
    gae = 0.0

    for t in reversed(range(T)):
        # GAMMA USED HERE as 0.99
        delta = rewards_t[t] + 0.99 * values_ext[t+1] - values_ext[t]
        gae = delta + 0.99 * lam * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns

###--------------------------------------------------- MAIN IPPO TRAINING AND UPDATE LOOP ------------------------------###

def ippo_update(actor, critics, actor_opt, critic_opts, buffer, r_bars, n_agents,
                device, lam, clip_eps, entropy_coef, n_actor_epochs, n_critic_epochs,
                mini_batch_size, track_metrics=False):
    T = len(buffer)
    if T == 0:
        return {}

    if track_metrics:
        track = {'actor_losses': [],'critic_losses': [],'entropies': []}
    else:
        track = {}

    local_obs_t = torch.tensor(np.stack(buffer.local_obs), dtype=torch.float32, device=device)  # [T, n_agents, local_dim]
    actions_t = torch.tensor(np.stack(buffer.actions), dtype=torch.long, device=device)  # [T, n_agents]
    old_log_probs_t = torch.stack(buffer.log_probs, dim=0).detach().to(device)  # [T, n_agents]
    rewards_np = np.stack(buffer.rewards, axis=0)  # [T, n_agents]
    masks_t = torch.tensor(np.stack(buffer.action_masks), dtype=torch.bool, device=device)  # [T, n_agents, action_dim]

    # Compute advantages per agent using differential GAE
    advantages_all = []
    returns_all = []
    values_all = []

    for agent_id in range(n_agents):
        agent_local_obs = local_obs_t[:, agent_id, :]  # [T, local_dim]
        agent_rewards = rewards_np[:, agent_id]  # [T]

        with torch.no_grad():
            values = critics[agent_id](agent_local_obs)  # [T]
            next_value = values[-1]  # Last value as bootstrap

            advantages, returns = compute_differential_gae(
                agent_rewards, values, next_value, r_bars[agent_id].get(), lam
            )

        # Update average reward baseline
        r_bars[agent_id].update(torch.tensor(agent_rewards))

        advantages_all.append(advantages)
        returns_all.append(returns)
        values_all.append(values)

    advantages_t = torch.stack(advantages_all, dim=1)  # [T, n_agents]
    returns_t = torch.stack(returns_all, dim=1)  # [T, n_agents]

    # Normalize advantages per agent
    advantages_normalized = advantages_t.clone()
    for agent_id in range(n_agents):
        adv = advantages_t[:, agent_id]
        advantages_normalized[:, agent_id] = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Flatten for mini-batch training
    local_obs_flat = local_obs_t.view(T * n_agents, -1)
    actions_flat = actions_t.view(T * n_agents)
    old_lp_flat = old_log_probs_t.view(T * n_agents)
    masks_flat = masks_t.view(T * n_agents, -1)
    adv_flat = advantages_normalized.reshape(T * n_agents)
    returns_flat = returns_t.reshape(T * n_agents)
    agent_ids_flat = torch.arange(n_agents, device=device).unsqueeze(0).expand(T, -1).reshape(T * n_agents)

    dataset_size = T * n_agents
    indices = np.arange(dataset_size)

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0
    n_actor_updates = 0
    n_critic_updates = 0

    # Critic training
    for _ in range(n_critic_epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, mini_batch_size):
            mb_idx = indices[start:start + mini_batch_size]
            if len(mb_idx) == 0:
                continue

            mb_obs = local_obs_flat[mb_idx]
            mb_aids = agent_ids_flat[mb_idx]
            mb_returns = returns_flat[mb_idx]

            
            for agent_id in range(n_agents):
                agent_mask = (mb_aids == agent_id) # Update each critic independently for every agent
                if agent_mask.sum() == 0:
                    continue

                agent_obs = mb_obs[agent_mask]
                agent_returns = mb_returns[agent_mask]

                values_pred = critics[agent_id](agent_obs)
                critic_loss = F.huber_loss(values_pred, agent_returns.detach(), delta=1.0)

                critic_opts[agent_id].zero_grad()
                critic_loss.backward()

                if track_metrics:
                    grad_norm = sum(p.grad.norm().item()**2 for p in critics[agent_id].parameters() if p.grad is not None)**0.5
                    track['critic_losses'].append(critic_loss.item())

                nn.utils.clip_grad_norm_(critics[agent_id].parameters(), 0.5)
                critic_opts[agent_id].step()

                total_critic_loss += critic_loss.item()
                n_critic_updates += 1

    # Actor training
    for _ in range(n_actor_epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, mini_batch_size):
            mb_idx = indices[start:start + mini_batch_size]
            if len(mb_idx) == 0:
                continue

            mb_obs = local_obs_flat[mb_idx]
            mb_acts = actions_flat[mb_idx]
            mb_aids = agent_ids_flat[mb_idx]
            mb_old_lp = old_lp_flat[mb_idx]
            mb_masks = masks_flat[mb_idx]
            mb_adv = adv_flat[mb_idx]

            valid_mask = mb_masks.any(dim=-1)
            new_log_probs, entropies = actor.evaluate(mb_obs, mb_aids, mb_acts, mb_masks)

            ratio = torch.exp(new_log_probs - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv

            n_valid = valid_mask.sum().clamp(min=1)
            actor_loss = (
                -torch.min(surr1, surr2)[valid_mask].sum() / n_valid
                - entropy_coef * entropies[valid_mask].sum() / n_valid
            )

            actor_opt.zero_grad()
            actor_loss.backward()

            if track_metrics:
                grad_norm = sum(p.grad.norm().item()**2 for p in actor.parameters() if p.grad is not None)**0.5
                track['actor_losses'].append(actor_loss.item())
                track['entropies'].append(entropies.mean().item())
                kl = (mb_old_lp - new_log_probs).mean().item()

            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_opt.step()

            total_actor_loss += actor_loss.item()
            total_entropy += entropies.mean().item()
            n_actor_updates += 1

    n_actor_updates = max(n_actor_updates, 1)
    n_critic_updates = max(n_critic_updates, 1)

    result = {
        'actor_loss': total_actor_loss / n_actor_updates,
        'critic_loss': total_critic_loss / n_critic_updates,
        'entropy': total_entropy / n_actor_updates,
        'r_bar_avg': np.mean([r.get() for r in r_bars]),
    }

    if track_metrics:
        result['tracking'] = track

    return result