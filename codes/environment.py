###--------------------------------------------------- MAIN ENVIRONMENT CLASS ---------------------------------------------------###

class Env(gym.Env):
    def __init__(self, n_row, n_aisle, n_cp, n_block, arrival_rate, capacity,
                 max_battery, min_battery, min_battery_limit, depletion_rate,
                 recharge_rate, velocity, MAX_STEPS, seed):
        self.n_row = n_row                                # Number of storage locations within a storage rack vertically in the environment (Y- axis range)
        self.n_aisle = n_aisle                            # Number of aisles totally in the environment (X-axis range)
        self.aisles_per_block = n_aisle // 2              # Number of aisles per block
        self.rows_per_block   = n_row // (n_block // 2)   # Number of storage locations within a storage rack
        self.n_cp = n_cp                                  # Number of charging stations (CS) or charging point (CP)
        self.n_block = n_block                            # Number of blocks
        self.n_agent = n_block                            # Number of agents with their individual blocks
        self.arrival_rate = arrival_rate                  # Order arrival rate that follows a poisson process
        self.capacity = capacity                          # Maximum carrying capacity of any agent
        self.max_battery = 100                            # Maximum battery level (unit)
        self.min_battery = 0                              # Minimum battery level physically possible, always zero
        self.min_battery_limit = min_battery_limit        # Minimum battery threshold (unit)
        self.depletion_rate = depletion_rate              # Battery depletion rate (unit/sec)
        self.recharge_rate = recharge_rate                # Battery re-charge rate (unit/sec)
        self.velocity = velocity                          # Velocity of the agent (unit/sec)
        self.max_distance = (self.n_aisle + self.n_row)*2 # Hyperparameter used for normalizing distances
        self.MAX_STEPS = MAX_STEPS                        # Training time T of the simulation
        self.block_border = {}                            # Co-ordinates that define every blocks corners
        for i in range(1, self.n_agent + 1):
            if i == 1:
                minx, miny = 0, 0
                maxx, maxy = minx + self.aisles_per_block - 1, miny + self.rows_per_block - 1
                self.block_border[i] = {'minx':minx,'maxx':maxx,'miny':miny,'maxy':maxy}
            elif i == 2:
                minx, miny = self.block_border[i-1]['maxx'] + 1, 0
                maxx, maxy = minx + self.aisles_per_block - 1, miny + self.rows_per_block - 1
                self.block_border[i] = {'minx':minx,'maxx':maxx,'miny':miny,'maxy':maxy}
            elif i > 2:
                minx, miny = self.block_border[i-2]['minx'], self.block_border[i-2]['maxy'] + 1
                maxx, maxy = minx + self.aisles_per_block - 1, miny + self.rows_per_block - 1
                self.block_border[i] = {'minx':minx,'maxx':maxx,'miny':miny,'maxy':maxy}

        self.depot_location = (-1, -1)                                                           # Depot located at bottom left corner
        self.charger_location = [[(n_aisle-1)/2, -1], [(n_aisle-1)/2, n_row]]                    # Charging Stations at the mid point of top and bottom boundaries

        self.agent_state = {}                                                                    # Dictionary to store information regarding agents location, capacity, battery and service mode. Refer def reset
        self.remaining_time_for_action = {n: 0 for n in range(1, self.n_agent + 1)}              # Dictionary that stores for every agent how much time is remaining in travel while going from one point to another
        self.total_trip_time = [0 for i in range(self.n_agent)]                                  # Total time required to travel from one point to another
        self.prev_location = {n: [(n_aisle-1)/2, (n_row-1)/2] for n in range(1, self.n_agent+1)} # Store previous location of agent
        self.next_location = {n: [] for n in range(1, self.n_agent + 1)}                         # Store next location of agent
        self.travel_to_depot_ind = {n: False for n in range(1, self.n_agent+1)}                  # Tells if an agent is in travel to or from depot
        self.returning_from_cp_ind = {n: False for n in range(1, self.n_agent+1)}                # Tells if an agent is travelling from CS to its previous position

        self.cp_state = {m: [] for m in range(1, self.n_cp+1)}                                   # List of agents in queue at CS
        self.cp_queue_len = [0 for m in range(self.n_cp)]                                        # Length of the queue at CS
        self.battery_just_before_charging = {n: 100 for n in range(1, self.n_agent+1)}           # Battery of each agent just before they started charging

        self.orders = []                                                                         # List that stores information of all arriving orders. Refer def new order
        self.total_orders_placed = [0 for i in range(self.n_agent)]                              # Total number of orders placed per agent
        self.orders_remaining = {n: 0 for n in range(1, self.n_agent + 1)}                       # Total number of orders remaining per agent

        
        self.global_time_step = 0                                                                # Time step counter
        self.completion_info = {n: [0,0,0] for n in range(1, self.n_agent+1)}                    # Stores [number of orders completed, time spent in simulation, time spent charing] for every agent
        self.time_travelled = [0 for i in range(self.n_agent)]                                   # Tracker for time spent in travel
        self.depot_counter = {n: [[], []] for n in range(1, self.n_agent+1)}                     # Tracks the [capacity level, battery level] for every agent when it decides to go to depot
        self.travel_time_depot = {n: 0 for n in range(1, self.n_agent+1)}                        # Time spent traveling to and from depot
        self.travel_time_pick  = {n: 0 for n in range(1, self.n_agent+1)}                        # Time spent traveling to pick order
        self.travel_time_cp    = {n: 0 for n in range(1, self.n_agent+1)}                        # Time spent traveling to and from CS
        self.wait_in_q         = {n: 0 for n in range(1, self.n_agent+1)}                        # Time spent waiting in queue at CS
        self.idle_time    = {n: 0 for n in range(1, self.n_agent+1)}                             # Time spent idle
        self.charging_time = {n: 0 for n in range(1, self.n_agent+1)}                            # Time spent charging
        self.seed = seed
        self.reset()

    def observe_env(self):
        agent_state_array = np.zeros((self.n_agent, 5), dtype=np.float32)
        for n in range(1, self.n_agent+1):
            sh, sv, c, sb, scp = (self.agent_state[n][0], self.agent_state[n][1],self.agent_state[n][2], self.agent_state[n][3],self.agent_state[n][4])
            # sh: x-cordinate, sv- y cordinate, c: carryigng capacity, sb: battery level, scp: service mode
            agent_state_array[n-1, :] = [sh, sv, c, sb, scp]
        cp_queue_len_array = np.array([len(self.cp_state[m]) for m in range(1, self.n_cp+1)], dtype=np.float32)
        return {
            'agent_state': agent_state_array,
            'cp_queue_len': cp_queue_len_array,
            'global_time_step': float(self.global_time_step),
            'allowed_action': self.allowed_action()
        }

    def reset(self):
        self.agent_state = {} # At the start of the simulation agents are located at center of warehouse with full capacity and battery
        for n in range(1, self.n_agent+1):
            self.agent_state[n] = [(self.n_aisle-1)/2, (self.n_row-1)/2, self.capacity, self.max_battery, 0]
        self.remaining_time_for_action = {n: 0 for n in range(1, self.n_agent + 1)}
        self.agent_todo_action = {n: 0 for n in range(1, self.n_agent + 1)}
        self.prev_location = {n: [(self.n_aisle-1)/2, (self.n_row-1)/2] for n in range(1, self.n_agent+1)}
        self.next_location = {n: [] for n in range(1, self.n_agent + 1)}
        self.travel_to_depot_ind = {n: False for n in range(1, self.n_agent+1)}
        self.arrival_rate = self.arrival_rate

        self.cp_state = {m: [] for m in range(1, self.n_cp+1)}
        self.cp_queue_len = [0 for m in range(self.n_cp)]
        self.battery_just_before_charging = {n: 100 for n in range(1, self.n_agent+1)}

        self.orders = []
        self.total_orders_placed = [0 for i in range(self.n_agent)]
        self.orders_remaining = {n: 0 for n in range(1, self.n_agent + 1)}

        self.total_trip_time = [0 for i in range(self.n_agent)]

        # OB: Order is for which block,  (OH, OV): x and y cordinates of the order, OK: items in the order, set as always equal to 1, ind_completed: tells if an order is picked, id_: order index
        for n in range(1, self.n_agent+1):
            OB = n
            minx, maxx = self.block_border[OB]['minx'], self.block_border[OB]['maxx']
            miny, maxy = self.block_border[OB]['miny'], self.block_border[OB]['maxy']
            OH = np.random.randint(minx, maxx+1)
            OV = np.random.randint(miny, maxy+1)
            OK, ind_completed, id_ = 1, 0, len(self.orders)+1
            self.orders.append([OB, OH, OV, OK, ind_completed, id_])
            self.next_location[n] = [self.orders[n-1][1], self.orders[n-1][2]]
            self.total_orders_placed[n-1] = 1
            self.orders_remaining[n] = 1

        self.time_travelled = [0 for i in range(self.n_agent)]
        self.travel_time_depot = {n: 0 for n in range(1, self.n_agent+1)}
        self.travel_time_pick  = {n: 0 for n in range(1, self.n_agent+1)}
        self.travel_time_cp    = {n: 0 for n in range(1, self.n_agent+1)}
        self.wait_in_q         = {n: 0 for n in range(1, self.n_agent+1)}
        self.idle_time     = {n: 0 for n in range(1, self.n_agent+1)}
        self.charging_time = {n: 0 for n in range(1, self.n_agent+1)}

        self.global_time_step = 0
        self.completion_info = {n: [0,0,0] for n in range(1, self.n_agent+1)}
        self.returning_from_cp_ind = {n: False for n in range(1, self.n_agent+1)}
        self.depot_counter = {n: [[], []] for n in range(1, self.n_agent+1)}
        return self.observe_env(), {'global_time_step': self.global_time_step}

    ###--------------------------------------------------- HELPER FUNCTIONS ---------------------------------------------------###
    
    # Generates and assigns new orders to agents
    def new_order(self, t):
        rng = np.random.default_rng(seed=None)
        k = rng.poisson(self.arrival_rate * t)
        # OB: Order is for which block,  (OH, OV): x and y cordinates of the order, OK: items in the order, set as always equal to 1, ind_completed: tells if an order is picked, id_: order index
        while k > 0:  
            OB = np.random.randint(1, self.n_block+1)
            minx, maxx = self.block_border[OB]['minx'], self.block_border[OB]['maxx']
            miny, maxy = self.block_border[OB]['miny'], self.block_border[OB]['maxy']
            OH, OV, OK = rng.integers(minx, maxx+1), rng.integers(miny, maxy+1), 1
            id_ = len(self.orders)+1
            ind_completed = 0
            self.orders.append([OB, OH, OV, OK, ind_completed, id_])
            k -= 1
            self.total_orders_placed[OB-1] += 1
            self.orders_remaining[OB] += 1
    # Calculates euclidean distance between two points a and b
    def func_distance(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

    # Calculates time to travel between two points  and b
    def func_travel_time(self, a, b):
        distance = self.func_distance(a, b)
        return max(1, distance / self.velocity)

    # Checks if there is sufficient battery to time to travel from origin to destination for agent n
    def check_battery_sufficiency(self, n, destination):
        origin = [self.agent_state[n][0], self.agent_state[n][1]]
        distance = self.func_distance(origin, destination)
        time_ = max(1, int(np.ceil(distance / self.velocity)))
        battery_needed = min(100, self.depletion_rate * time_)
        return (self.agent_state[n][3] - battery_needed) >= 0

    # Checks of there is sufficient carrying capacity for agent n to carry number of items (num_of_items)
    def check_capacity_sufficiency(self, n, num_of_items):
        return (self.agent_state[n][2] - num_of_items) >= 0

    # Get imfo regarding which is the next order to pick for agent n
    def next_order_to_pick(self, n):
        for order in self.orders:
            if order[0] == n and order[4] == 0:
                return order
        return None

    ###--------------------------------------------------- ALLOWED ACTION MASK ---------------------------------------------------###
    def allowed_action(self):
        allowed_action_info = np.zeros((self.n_agent, self.n_cp +3 + 3), dtype=np.int8)
        # go_pick is action 0
        # go to CS1 is action 1
        # go to CS2 is action 2
        stop_charging_index = self.n_cp + 1 # action 3
        go_to_depot_index   = self.n_cp + 2 # action 4
        wait_in_queue       = self.n_cp + 3 # action 5
        keep_charging       = self.n_cp + 4 # action 6
        travelling          = self.n_cp + 5 # action 7

        for n in range(1, self.n_agent+1):
            # If in travel, only action 7 allowed
            if self.remaining_time_for_action[n] > 0:
                allowed_action_info[n-1, 7] = 1
                continue
            else:
                next_order = self.next_order_to_pick(n)
                if self.agent_state[n][4] == 0:
                    if next_order is None:
                        # If no next order, no action allowed
                        allowed_action_info[n-1, :] = 0
                    elif next_order:
                        bat_next = min(100, self.depletion_rate * self.func_travel_time(
                            (self.agent_state[n][0], self.agent_state[n][1]),
                            (next_order[1], next_order[2])))
                        bat_depot = min(100, self.depletion_rate * 2 * self.func_travel_time(
                            (self.agent_state[n][0], self.agent_state[n][1]),
                            self.depot_location))
                        
                        # If no enough battery, only going to CS allowed
                        if self.agent_state[n][3] < (self.min_battery_limit + bat_next):
                            for i in range(1, self.n_cp+1):
                                if self.check_battery_sufficiency(n, self.charger_location[i-1]):
                                    allowed_action_info[n-1, i] = 1
                        elif self.agent_state[n][3] >= (self.min_battery_limit + bat_next):
                            if not self.check_capacity_sufficiency(n, next_order[3]):
                                # If no enough capacity, and no enough battery to go to depot, then only going to CS allowed
                                # elif no enough capacity but enough battery present, then only go to depot is allowed
                                if self.agent_state[n][3] < (self.min_battery_limit + bat_depot):
                                    for i in range(1, self.n_cp+1):
                                        if self.check_battery_sufficiency(n, self.charger_location[i-1]):
                                            allowed_action_info[n-1, i] = 1
                                else:
                                    allowed_action_info[n-1, go_to_depot_index] = 1
                            elif self.check_capacity_sufficiency(n, next_order[3]):
                                # If sufficient capacity and capacity not full, then agent can pick, go to CS or go to depot
                                allowed_action_info[n-1, 0] = 1
                                for i in range(1, self.n_cp+1):
                                    if self.check_battery_sufficiency(n, self.charger_location[i-1]):
                                        allowed_action_info[n-1, i] = 1
                                if (self.agent_state[n][3] >= (self.min_battery_limit + bat_depot) and
                                        self.agent_state[n][2] < self.capacity):
                                    allowed_action_info[n-1, go_to_depot_index] = 1
                                    
                elif self.agent_state[n][4] != 0: # If agent at CS;
                    cp_id = self.agent_state[n][4]
                    bat_return = min(100, self.depletion_rate * self.func_travel_time(
                        self.charger_location[cp_id-1],
                        (self.prev_location[n][0], self.prev_location[n][1])))
                    # If agent first in queue and has enough charge to return then it can choose to continue charging or stop charging
                    # If agent first in queue and already charged to full battery level then stop charging
                    # If agent first in queue but does not have enough charge to return then keep charging
                    # Elif agent not forst in queue then wait in queue
                    if self.cp_state[cp_id][0] == n:
                        if self.agent_state[n][3] == self.max_battery:
                            allowed_action_info[n-1, stop_charging_index] = 1
                        elif self.agent_state[n][3] <= max(bat_return + self.min_battery_limit,self.battery_just_before_charging[n]):
                            allowed_action_info[n-1, keep_charging] = 1
                        else:
                            allowed_action_info[n-1, stop_charging_index] = 1
                            allowed_action_info[n-1, keep_charging] = 1
                    else:
                        allowed_action_info[n-1, wait_in_queue] = 1
        return allowed_action_info

    ###--------------------------------------------------- ENVIRONMENTS MAIN STEP FUNCTION ---------------------------------------------------###
    def step(self, action_array):
        stop_charging_index = self.n_cp + 1
        go_to_depot_index   = self.n_cp + 2
        wait_in_queue       = self.n_cp + 3
        keep_charging       = self.n_cp + 4
        travelling          = self.n_cp + 5
        action_array = np.asarray(action_array)
        REWARD = np.zeros(self.n_agent, dtype=np.float32)

        for n in range(1, self.n_agent+1):
            action = action_array[n-1]

            if self.remaining_time_for_action[n] > 0:
                REWARD[n-1] += -1
                continue

            if self.agent_state[n][4] != 0:
                cp_id = self.agent_state[n][4]
                bat_return = min(100, self.depletion_rate * self.func_travel_time(self.charger_location[cp_id-1],(self.prev_location[n][0], self.prev_location[n][1])))

                if self.cp_state[cp_id][0] == n and action == stop_charging_index:
                    self.cp_state[self.agent_state[n][4]].pop(0)
                    self.cp_queue_len[self.agent_state[n][4]-1] -= 1
                    self.agent_state[n][4] = 0
                    self.next_location[n] = list([self.prev_location[n][0], self.prev_location[n][1]])
                    self.prev_location[n] = list(self.charger_location[cp_id-1])
                    self.remaining_time_for_action[n] = self.func_travel_time(self.prev_location[n], self.next_location[n])
                    self.returning_from_cp_ind[n] = True
                    self.time_travelled[n-1] += self.remaining_time_for_action[n]
                    self.total_trip_time[n-1] = self.remaining_time_for_action[n]
                    battery_gained = (self.agent_state[n][3] - self.battery_just_before_charging[n])
                    REWARD[n-1] += -1

                elif self.cp_state[cp_id][0] == n and action == keep_charging:
                    self.agent_state[n][3] = min(self.max_battery, self.agent_state[n][3] + self.recharge_rate)
                    self.completion_info[n][1] += 1
                    self.completion_info[n][2] += 1
                    REWARD[n-1] += -1
                    self.charging_time[n] += 1

                elif self.cp_state[cp_id][0] != n and action == wait_in_queue:
                    self.completion_info[n][1] += 1
                    REWARD[n-1] += -1
                    self.wait_in_q[n] += 1
                continue

            elif self.agent_state[n][4] == 0:
                bat_depot = min(100, self.depletion_rate * 2 * self.func_travel_time((self.agent_state[n][0], self.agent_state[n][1]), self.depot_location))
                next_order = self.next_order_to_pick(n)

                if next_order is None:
                    self.idle_time[n] += 1
                    if action in range(1, self.n_cp + 1):
                        self.remaining_time_for_action[n] = self.func_travel_time((self.agent_state[n][0], self.agent_state[n][1]),self.charger_location[action-1])
                        self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
                        self.next_location[n] = list(self.charger_location[action-1])
                        self.battery_just_before_charging[n] = self.agent_state[n][3]
                        self.time_travelled[n-1] += self.remaining_time_for_action[n]
                        self.total_trip_time[n-1] = self.remaining_time_for_action[n]

                        REWARD[n-1] += -1
                    elif (action == go_to_depot_index and
                          self.agent_state[n][3] >= (self.min_battery_limit + bat_depot) and
                          self.agent_state[n][2] < self.capacity):
                        self.remaining_time_for_action[n] = 2 * self.func_travel_time(
                            (self.agent_state[n][0], self.agent_state[n][1]), self.depot_location)
                        self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
                        self.next_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
                        self.travel_to_depot_ind[n] = True
                        self.time_travelled[n-1] += self.remaining_time_for_action[n]
                        self.total_trip_time[n-1] = self.remaining_time_for_action[n]
                        self.depot_counter[n][0].append(self.agent_state[n][2])
                        self.depot_counter[n][1].append(self.orders_remaining[n])
                        self.travel_time_depot[n] += 2 * self.func_travel_time(
                            (self.agent_state[n][0], self.agent_state[n][1]), self.depot_location)
                        REWARD[n-1] += -1
                    continue

                elif next_order:
                    bat_next = min(100, self.depletion_rate * self.func_travel_time(
                        (self.agent_state[n][0], self.agent_state[n][1]),
                        (next_order[1], next_order[2])))

                    if self.agent_state[n][3] < (self.min_battery_limit + bat_next) and action in range(1, self.n_cp + 1):
                        self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
                        self.next_location[n] = list(self.charger_location[action-1])
                        self.remaining_time_for_action[n] = self.func_travel_time(
                            self.prev_location[n], self.next_location[n])
                        self.battery_just_before_charging[n] = self.agent_state[n][3]
                        self.time_travelled[n-1] += self.remaining_time_for_action[n]
                        self.total_trip_time[n-1] = self.remaining_time_for_action[n]
                        REWARD[n-1] += -1
                        self.travel_time_cp[n] += self.remaining_time_for_action[n]

                    elif self.agent_state[n][3] >= (self.min_battery_limit + bat_next):
                        if action == 0 and self.check_capacity_sufficiency(n, next_order[3]):
                            self.remaining_time_for_action[n] = self.func_travel_time(
                                (self.agent_state[n][0], self.agent_state[n][1]),
                                (next_order[1], next_order[2]))
                            self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
                            self.next_location[n] = list([next_order[1], next_order[2]])
                            self.time_travelled[n-1] += self.remaining_time_for_action[n]
                            self.total_trip_time[n-1] = self.remaining_time_for_action[n]
                            REWARD[n-1] += 20
                            self.travel_time_pick[n] += self.remaining_time_for_action[n]

                        elif action in range(1, self.n_cp + 1):
                            self.remaining_time_for_action[n] = self.func_travel_time(
                                (self.agent_state[n][0], self.agent_state[n][1]),
                                self.charger_location[action-1])
                            self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
                            self.next_location[n] = list(self.charger_location[action-1])
                            self.battery_just_before_charging[n] = self.agent_state[n][3]
                            self.time_travelled[n-1] += self.remaining_time_for_action[n]
                            self.total_trip_time[n-1] = self.remaining_time_for_action[n]
                            REWARD[n-1] += -1
                            self.travel_time_cp[n] += self.remaining_time_for_action[n]

                        elif (action == go_to_depot_index and
                              self.agent_state[n][3] >= (self.min_battery_limit + bat_depot) and
                              self.agent_state[n][2] < self.capacity):
                            self.remaining_time_for_action[n] = 2 * self.func_travel_time(
                                (self.agent_state[n][0], self.agent_state[n][1]), self.depot_location)
                            self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
                            self.next_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
                            self.travel_to_depot_ind[n] = True
                            self.time_travelled[n-1] += self.remaining_time_for_action[n]
                            self.total_trip_time[n-1] = self.remaining_time_for_action[n]
                            self.depot_counter[n][0].append(self.agent_state[n][2])
                            self.depot_counter[n][1].append(self.orders_remaining[n])
                            REWARD[n-1] +=-1
                            self.travel_time_depot[n] += self.remaining_time_for_action[n]
                    continue

        for n in range(1, self.n_agent+1):
            if self.remaining_time_for_action[n] > 0:
                self.completion_info[n][1] += 1
                self.remaining_time_for_action[n] = max(0, self.remaining_time_for_action[n] - 1)
                prev_x, prev_y = self.agent_state[n][0], self.agent_state[n][1]
                if self.travel_to_depot_ind[n] == True and self.remaining_time_for_action[n] >= self.total_trip_time[n-1]/2:
                    time_spent = self.total_trip_time[n-1] - self.remaining_time_for_action[n]
                    half_trip_time = self.total_trip_time[n-1]/2
                    elapsed = time_spent / max(1,half_trip_time)
                    self.agent_state[n][0] = round(self.prev_location[n][0] + elapsed * (self.depot_location[0] - self.prev_location[n][0]),3)
                    self.agent_state[n][1] = round(self.prev_location[n][1] + elapsed * (self.depot_location[1] - self.prev_location[n][1]),3)
                elif self.travel_to_depot_ind[n] == True and self.remaining_time_for_action[n] < self.total_trip_time[n-1]/2:
                    time_spent = self.total_trip_time[n-1] - self.remaining_time_for_action[n]
                    half_trip_time = self.total_trip_time[n-1]/2
                    elapsed = (time_spent - half_trip_time) / max(1,self.total_trip_time[n-1]/2)
                    self.agent_state[n][0] = round(self.depot_location[0] + elapsed * (self.next_location[n][0] - self.depot_location[0]),3)
                    self.agent_state[n][1] = round(self.depot_location[1] + elapsed * (self.next_location[n][1] - self.depot_location[1]),3)
                else:
                    elapsed = (self.total_trip_time[n-1] - self.remaining_time_for_action[n])/ max(1,self.total_trip_time[n-1])
                    self.agent_state[n][0] = round(self.prev_location[n][0] + elapsed * (self.next_location[n][0] - self.prev_location[n][0]),3)
                    self.agent_state[n][1] = round(self.prev_location[n][1] + elapsed * (self.next_location[n][1] - self.prev_location[n][1]),3)

                distance_moved = self.func_distance((self.agent_state[n][0], self.agent_state[n][1]),(prev_x, prev_y))
                self.agent_state[n][3] = max(0, self.agent_state[n][3] - self.depletion_rate * distance_moved)

                if self.remaining_time_for_action[n] == 0:
                    at_cp = any(
                        np.isclose(self.agent_state[n][0],self.charger_location[m-1][0],atol=1e-3) and
                        np.isclose(self.agent_state[n][1],self.charger_location[m-1][1],atol=1e-3)
                        for m in range(1, self.n_cp+1))

                    if not at_cp: self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
                    self.agent_state[n][0] = self.next_location[n][0]
                    self.agent_state[n][1] = self.next_location[n][1]
                    self.next_location[n] = None

                    if not at_cp:
                        if not self.travel_to_depot_ind[n] and not self.returning_from_cp_ind[n]:
                            for i, o in enumerate(self.orders):
                                if (o[0] == n and o[4] == 0 and
                                        np.isclose(o[1],self.agent_state[n][0],atol=1e-3) and
                                        np.isclose(o[2],self.agent_state[n][1],atol=1e-3)):
                                    o[4] = 1
                                    self.orders_remaining[n] -= 1
                                    self.agent_state[n][2] -= o[3]
                                    self.completion_info[n][0] += 1
                                    self.orders.pop(i)
                                    break
                        elif self.travel_to_depot_ind[n]:
                            self.agent_state[n][2] = self.capacity
                            self.travel_to_depot_ind[n] = False
                        elif self.returning_from_cp_ind[n]:
                            self.returning_from_cp_ind[n] = False
                    else:
                        for m in range(1, self.n_cp+1):
                            if (np.isclose(self.agent_state[n][0],self.charger_location[m-1][0],atol=1e-3) and
                                    np.isclose(self.agent_state[n][1],self.charger_location[m-1][1],atol=1e-3)):
                                self.agent_state[n][4] = m
                                self.cp_state[m].append(n)
                                self.cp_queue_len[m-1] += 1
                                if action_array[n-1] != stop_charging_index and self.cp_state[m][0] == n:
                                    self.completion_info[n][2] += 1

        self.global_time_step += 1
        self.new_order(1)
        info = {
            "agent_state": self.agent_state,
            "action": action_array,
            "reward": REWARD,
            "allowed_action": self.allowed_action(),
            "remaining_time_for_action": self.remaining_time_for_action,
            "orders_completed": self.completion_info,
            "cp_queue_len": self.cp_queue_len,
            "global_time": self.global_time_step
        }
        return self.observe_env(), REWARD, False, info