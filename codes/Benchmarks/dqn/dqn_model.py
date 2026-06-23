
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

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        return self.net(obs)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs):
        self.buffer.append((obs, action, reward, next_obs))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs = zip(*batch)
        return (np.array(obs), np.array(actions),
                np.array(rewards), np.array(next_obs))

    def __len__(self):
        return len(self.buffer)

class AverageRewardBaseline:
    def __init__(self, alpha=0.01):
        self.r_bar = 0.0
        self.alpha = alpha

    def update(self, episode_rewards):
        mean_reward = np.mean(episode_rewards)
        self.r_bar = (1 - self.alpha) * self.r_bar + self.alpha * mean_reward

    def get(self):
        return self.r_bar