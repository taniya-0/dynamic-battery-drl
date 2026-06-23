import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

from IPython.core.magic import MAGIC_NO_VAR_EXPAND_ATTR

"""
There are 2 parts to this code, all are in this .py file.
PART 1: Main environment class
PART 2: Evaluation
"""
###------------------------------------------------PART 1: MAIN ENV CLASS SUITED FOR HEURISTIC-------------------------###
class Env():
    def __init__(self, n_row, n_aisle, n_cp, n_block, arrival_rate, capacity, max_battery, min_battery, min_battery_limit, depletion_rate, recharge_rate, velocity, MAX_STEPS, interrupt_enabled, seed):
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
		self.interrupt_enabled = interrupt_enabled
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
		
		self.interrupt_enabled = interrupt_enabled

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

    def interrupt(self):
        agent_to_release = None
        count_of_agent_at_cp = 0
        for i in range(1, self.n_agent+1):
          if self.agent_state[i][4] != 0: count_of_agent_at_cp += 1

        if count_of_agent_at_cp == 0: return None
        else:
          for n in range(1, self.n_agent+1):
            if self.remaining_time_for_action[n] == 0 and self.agent_state[n][4] != 0 and self.agent_state[n][3] > 50: # conditions to decide which agent is to be released
              agent_to_release = n
              break
        return agent_to_release

    ###--------------------------------------------------- ALLOWED ACTION MASK - SELECTS A SINGLE FIXED ACTION --------------------------------------------------###
    def select_fixed_action(self):
        allowed_action_info = np.zeros((self.n_agent, self.n_cp + 3), dtype=np.int8)
        stop_charging_index = self.n_cp + 1
        go_to_depot_index = self.n_cp + 2
        if self.interrupt_enabled: interrupted_agent = self.interrupt()
        else: interrupted_agent = None


        for n in range(1, self.n_agent+1):
            if self.remaining_time_for_action[n] > 0:
                allowed_action_info[n-1, :] = 0
                continue
            else:
                next_order = self.next_order_to_pick(n)

                if self.agent_state[n][4] == 0:
                    battery_needed_to_go_to_depot = min(100,self.depletion_rate * 2*self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location))
                    if next_order == None: allowed_action_info[n-1, :] = 0
                    elif next_order:
                        battery_needed_to_go_to_next_order = min(100,self.depletion_rate * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),(next_order[1],next_order[2])))
                        # Always choose nearest CS for charging
						if self.agent_state[n][3] < (self.min_battery_limit + battery_needed_to_go_to_next_order):
                            distances_to_cp = [self.func_distance((self.agent_state[n][0],self.agent_state[n][1]),self.charger_location[i-1]) for i in range(1,self.n_cp+1)]
                            nearest_cp_id = np.argmin(distances_to_cp) + 1
                            allowed_action_info[n-1, nearest_cp_id] = 1

                        # Next order and sufficient battery to go to next order
                        elif self.agent_state[n][3] >= (self.min_battery_limit + battery_needed_to_go_to_next_order):
                          if self.check_capacity_sufficiency(n, next_order[3]) == False:
                              if (self.agent_state[n][3] < (self.min_battery_limit + battery_needed_to_go_to_depot)):
                                  distances_to_cp = [self.func_distance((self.agent_state[n][0],self.agent_state[n][1]),self.charger_location[i-1]) for i in range(1,self.n_cp+1)]
                                  nearest_cp_id = np.argmin(distances_to_cp) + 1
                                  allowed_action_info[n-1, nearest_cp_id] = 1

                              else: allowed_action_info[n-1, go_to_depot_index] = 1
                          # Enough capacity available
                          elif self.check_capacity_sufficiency(n, next_order[3]) == True:
                              allowed_action_info[n-1, 0] = 1


                # AT CP
                elif self.agent_state[n][4] != 0:
                    cp_id = self.agent_state[n][4]

                    # If first in queue
                    battery_needed_to_return = min(100,self.depletion_rate * self.func_travel_time((self.charger_location[cp_id-1]),(self.prev_location[n][0], self.prev_location[n][1])))
					
					### FOR HIGH - LOW STRATEGY USE fraction_of_orders_remaining as battery_high
					### FOR FIXED THRESHOLD STRATEGY, FIX THE UPPER BATTERY LIMIT
                    fraction_of_orders_remaining = self.num_of_orders_remaining(n)/self.total_orders_placed[n-1]
                    battery_high = fraction_of_orders_remaining * 100
					### OR
                    battery_high = 85

                    if self.cp_state[cp_id][0] == n:
                        if self.agent_state[n][3] == self.max_battery: allowed_action_info[n-1, stop_charging_index] = 1
                        elif self.agent_state[n][3] <= (battery_needed_to_return + self.min_battery_limit): allowed_action_info[n-1, cp_id] = 1 # In this case action = cp_id means continue charging
                        elif (self.agent_state[n][3] >= battery_high) and (self.agent_state[n][3] >= battery_needed_to_return + self.min_battery_limit): allowed_action_info[n-1, stop_charging_index] = 1
                        elif (n == interrupted_agent) and (self.agent_state[n][3] >= battery_needed_to_return + self.min_battery_limit): allowed_action_info[n-1, stop_charging_index] = 1
                        else: allowed_action_info[n-1, cp_id] = 1 # In this case action = cp_id means continue charging
                    else: allowed_action_info[n-1, cp_id] = 1

        return allowed_action_info

    def step(self,action_array): #action_array = [a,b,c,d] where a.b.c.d are the actions for each of the 4 agents

      stop_charging_index = self.n_cp + 1
      go_to_depot_index = self.n_cp + 2
      action_array = np.asarray(action_array)
      REWARD = 0

      for n in range(1,self.n_agent+1):
        action = action_array[n-1]

        if self.remaining_time_for_action[n] > 0:
          REWARD += - 1/(self.n_agent/self.n_cp)
          continue

        if self.agent_state[n][4] != 0:
          fraction_of_orders_remaining = self.num_of_orders_remaining(n)/self.total_orders_placed[n-1]
          cp_id = self.agent_state[n][4]
          battery_needed_to_return = min(100,self.depletion_rate * self.func_travel_time((self.charger_location[cp_id-1]),(self.prev_location[n][0], self.prev_location[n][1])))
          if self.cp_state[cp_id][0] == n and action == stop_charging_index: 
            self.action_counter[n][action] += 1
            self.agent_charging_ind[n] = False
            self.cp_state[self.agent_state[n][4]].pop(0)
            self.cp_queue_len[self.agent_state[n][4]-1] -= 1
            self.agent_state[n][4] = 0
            self.next_location[n] = [self.prev_location[n][0], self.prev_location[n][1]]
            self.prev_location[n] = self.charger_location[cp_id-1]
            self.remaining_time_for_action[n] = self.func_travel_time(self.prev_location[n],self.next_location[n])
            self.returning_from_cp_ind[n] = True
            self.time_travelled[n-1] += self.remaining_time_for_action[n]
            self.total_trip_time[n-1] = self.remaining_time_for_action[n]
            fraction_of_orders_remaining = min(1,self.orders_remaining[n]/(self.capacity * self.n_agent))

            REWARD += (self.agent_state[n][3] - fraction_of_orders_remaining * 100)/self.depletion_rate
            self.total_travel_time[n] += self.remaining_time_for_action[n]
            self.travel_time_cp[n] += self.remaining_time_for_action[n]

          elif action in [i for i in range(1,self.n_cp + 1)]:
            if self.cp_state[cp_id][0] == n: # Continue charging
              self.agent_state[n][3] = min(self.max_battery, self.agent_state[n][3] + self.recharge_rate)
              self.orders_completed[n][1] += 1
              self.orders_completed[n][2] += 1

              REWARD += - 1/(self.n_agent/self.n_cp)
              self.charging_time[n] += 1
            else:
              self.orders_completed[n][1] += 1

              REWARD += - 1/(self.n_agent/self.n_cp)
              self.wait_in_q[n] += 1
          continue

        elif self.agent_state[n][4] == 0:
          battery_needed_to_go_to_depot = min(100,self.depletion_rate * 2*self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location))
          next_order = self.next_order_to_pick(n)
          if next_order == None:
            self.idle_time[n] += 1
            if action in [i for i in range(1,self.n_cp + 1)]: # Go to CP
              self.action_counter[n][action] += 1
              self.remaining_time_for_action[n] = self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.charger_location[action-1])
              self.prev_location[n] = [self.agent_state[n][0], self.agent_state[n][1]]
              self.next_location[n] = self.charger_location[action-1]
              self.battery_just_before_charging[n] = self.agent_state[n][3] - self.depletion_rate * self.remaining_time_for_action[n]
              self.time_travelled[n-1] += self.remaining_time_for_action[n]
              self.total_trip_time[n-1] = self.remaining_time_for_action[n]

              REWARD += - 1 * (self.charging_time_for_agents_already_in_queue(action)/ (self.cp_queue_len[action-1]+1)) - self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.charger_location[action-1])
			  
            elif action == go_to_depot_index and (self.agent_state[n][3] >= (self.min_battery_limit + battery_needed_to_go_to_depot)) and (self.agent_state[n][2] < self.capacity): # Go to depot
              self.action_counter[n][action] += 1
              self.remaining_time_for_action[n] = 2 * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location)
              self.prev_location[n] = [self.agent_state[n][0], self.agent_state[n][1]]
              self.next_location[n] = [self.agent_state[n][0], self.agent_state[n][1]]
              self.travel_to_depot_ind[n] = True
              self.time_travelled[n-1] += self.remaining_time_for_action[n]
              self.depot_counter[n][0].append(self.agent_state[n][2])
              self.depot_counter[n][1].append(self.orders_remaining[n])
              self.travel_time_depot[n] +=  2 * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location)
              self.total_trip_time[n-1] = self.remaining_time_for_action[n]

              REWARD += + (self.capacity - self.agent_state[n][2]) * self.avg_time_per_order - 2 * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location)
              
            continue

          elif next_order:
            battery_needed_to_go_to_next_order = min(100,self.depletion_rate * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),(next_order[1],next_order[2])))
            if self.agent_state[n][3] < (self.min_battery_limit + battery_needed_to_go_to_next_order) and action in [i for i in range(1,self.n_cp + 1)]:
              self.action_counter[n][action] += 1
              self.prev_location[n] = [self.agent_state[n][0], self.agent_state[n][1]]
              self.next_location[n] = self.charger_location[action-1]
              self.remaining_time_for_action[n] = self.func_travel_time(self.prev_location[n],self.next_location[n])
              self.battery_just_before_charging[n] = self.agent_state[n][3] - self.depletion_rate * self.remaining_time_for_action[n]
              self.time_travelled[n-1] += self.remaining_time_for_action[n]
              self.total_trip_time[n-1] = self.remaining_time_for_action[n]

              REWARD += -1* (self.min_battery_limit + battery_needed_to_go_to_next_order - self.agent_state[n][3])/self.depletion_rate
              self.total_travel_time[n] += self.remaining_time_for_action[n]
              self.travel_time_cp[n] += self.remaining_time_for_action[n]
			  
            elif self.agent_state[n][3] >= (self.min_battery_limit + battery_needed_to_go_to_next_order):
              if action == 0 and self.check_capacity_sufficiency(n, next_order[3]) == True: # Go to Order
                self.action_counter[n][action] += 1
                self.remaining_time_for_action[n] = self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),(next_order[1],next_order[2]))
                self.prev_location[n] = [self.agent_state[n][0], self.agent_state[n][1]]
                self.next_location[n] = [next_order[1], next_order[2]]
                self.time_travelled[n-1] += self.remaining_time_for_action[n]

                REWARD += self.orders_remaining[n] * self.avg_time_per_order - self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),(next_order[1],next_order[2])) + self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location)
                self.time_per_trip[n].append(self.remaining_time_for_action[n])
                self.total_travel_time[n] += self.remaining_time_for_action[n]
                self.travel_time_pick[n] += self.remaining_time_for_action[n]
                self.total_trip_time[n-1] = self.remaining_time_for_action[n]
				
              elif action in [i for i in range(1,self.n_cp + 1)]:
                self.action_counter[n][action] += 1
                self.remaining_time_for_action[n] = self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.charger_location[action-1])
                self.prev_location[n] = [self.agent_state[n][0], self.agent_state[n][1]]
                self.next_location[n] = self.charger_location[action-1]
                self.battery_just_before_charging[n] = self.agent_state[n][3] - self.depletion_rate * self.remaining_time_for_action[n]
                self.time_travelled[n-1] += self.remaining_time_for_action[n]
                self.total_trip_time[n-1] = self.remaining_time_for_action[n]

                REWARD += - 1 * self.charging_time_for_agents_already_in_queue(action)/ (self.cp_queue_len[action-1]+1) - self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.charger_location[action-1])
                self.total_travel_time[n] += self.remaining_time_for_action[n]
                self.travel_time_cp[n] += self.remaining_time_for_action[n]
                #print(f"Reward for agent {n} to CHOOSE CHARGING{n}: {- 1 * self.charging_time_for_agents_already_in_queue(action)/ (self.cp_queue_len[action-1]+1)} REM_TIME: {self.remaining_time_for_action[n]} ACTION: {action} STATE: {self.agent_state[n]} FIRST_IN_QUEUE:{self.cp_state[action] == []}")

              elif action == go_to_depot_index and (self.agent_state[n][3] >= (self.min_battery_limit + battery_needed_to_go_to_depot)) and (self.agent_state[n][2] < self.capacity):
                self.action_counter[n][action] += 1
                self.remaining_time_for_action[n] = 2 * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location)
                self.prev_location[n] = [self.agent_state[n][0], self.agent_state[n][1]]
                self.next_location[n] = [self.agent_state[n][0], self.agent_state[n][1]]
                self.travel_to_depot_ind[n] = True
                self.time_travelled[n-1] += self.remaining_time_for_action[n]
                self.depot_counter[n][0].append(self.agent_state[n][2])
                self.depot_counter[n][1].append(self.orders_remaining[n])
                self.total_trip_time[n-1] = self.remaining_time_for_action[n]

                REWARD += + (self.capacity - self.agent_state[n][2]) * self.avg_time_per_order - 2 * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location)
                self.total_travel_time[n] += self.remaining_time_for_action[n]
                self.travel_time_depot[n] += self.remaining_time_for_action[n]
                #print(f"Reward for agent {n} to GO TO DEPOT: {+ (self.capacity - self.agent_state[n][2])} REM_TIME: {self.remaining_time_for_action[n]} ACTION: {action} STATE: {self.agent_state[n]}")
            continue

      ###___Change location of agents or if they are in motion, reduce their time by 1 unit___###
      for n in range(1, self.n_agent+1):
          if self.remaining_time_for_action[n] > 0:
              self.orders_completed[n][1] += 1
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
                                  self.orders_completed[n][0] += 1
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
                                  self.orders_completed[n][2] += 1

      self.global_time_step += 1
      self.new_order(1)
      info = {
          "agent_state": self.agent_state,
          "action": action_array,
          "reward": REWARD,
          "remaining_time_for_action": self.remaining_time_for_action,
          "next_location": self.next_location,
          "prev_location": self.prev_location,
          "orders_completed": self.orders_completed,
          "cp_queue_len": self.cp_queue_len,
          "cp_state": self.cp_state,
          "global_time": self.global_time_step}

      return self.observe_env(), REWARD, False, info
	  
###-------------------------------------------------------PART 2: EVALUATION FUNCTION ---------------------------------------------------###
def evaluate_(
    env,
    n_episodes,
    max_steps,
    device="cpu",
    render=False
):

    episode_rewards = []
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

    for ep in range(n_episodes):
        #print("Before all episodes:", test_env.travel_time_depot)
        obs, info = env.reset()
        #print("Reset travel depot:", env.orders)
        done = False
        total_reward = 0
        steps = 0

        while steps < max_steps:

            allowed_actions = env.select_fixed_action()
            actions = []
            for arr in allowed_actions:
                if sum(arr) !=0: a = int(np.where(arr == 1)[0])
                else: a = -1
                actions.append(a)

            obs, reward, done, info = env.step(actions)

            total_reward += reward
            steps += 1

            if render:
                env.render()


        episode_rewards.append(total_reward)
        orders_completed.append(env.orders_completed[1][0]+env.orders_completed[2][0]+env.orders_completed[3][0]+env.orders_completed[4][0])
        orders_completed1.append(env.orders_completed[1][0])
        orders_completed2.append(env.orders_completed[2][0])
        orders_completed3.append(env.orders_completed[3][0])
        orders_completed4.append(env.orders_completed[4][0])
        episode_lengths.append(steps)
        time_travelled1.append(env.time_travelled[0])
        time_travelled2.append(env.time_travelled[1])
        time_travelled3.append(env.time_travelled[2])
        time_travelled4.append(env.time_travelled[3])
        total_orders_placed.append(env.total_orders_placed[0]+env.total_orders_placed[1]+env.total_orders_placed[2]+env.total_orders_placed[3])
        total_orders_placed1.append(env.total_orders_placed[0])
        total_orders_placed2.append(env.total_orders_placed[1])
        total_orders_placed3.append(env.total_orders_placed[2])
        total_orders_placed4.append(env.total_orders_placed[3])
        time_spent_charging1.append(env.orders_completed[1][2])
        time_spent_charging2.append(env.orders_completed[2][2])
        time_spent_charging3.append(env.orders_completed[3][2])
        time_spent_charging4.append(env.orders_completed[4][2])
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

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "order_completed":  100*np.sum(orders_completed)/np.sum(total_orders_placed),
        "order_completed1":  100*np.sum(orders_completed1)/np.sum(total_orders_placed1),
        "order_completed2":  100*np.sum(orders_completed2)/np.sum(total_orders_placed2),
        "order_completed3":  100*np.sum(orders_completed3)/np.sum(total_orders_placed3),
        "order_completed4":  100*np.sum(orders_completed4)/np.sum(total_orders_placed4),
        "time_spent_charging1": np.mean(time_spent_charging1),
        "time_spent_charging2": np.mean(time_spent_charging2),
        "time_spent_charging3": np.mean(time_spent_charging3),
        "time_spent_charging4": np.mean(time_spent_charging4),
        "total_travel_time1": np.mean(total_travel_time1),
        "total_travel_time2": np.mean(total_travel_time2),
        "total_travel_time3": np.mean(total_travel_time3),
        "total_travel_time4": np.mean(total_travel_time4),
        "wait_in_q1": np.mean(wait_in_queue1),
        "wait_in_q2": np.mean(wait_in_queue2),
        "wait_in_q3": np.mean(wait_in_queue3),
        "wait_in_q4": np.mean(wait_in_queue4)

        }

###------------------------------------------------------- MAIN FUNCTION ---------------------------------------------------###

###  Set upper battery threshold to the variable battery_high in def select fixed action
###  Set lower battery threshold to the variable min_battery_limit when calling Env class below (test_env)

MAX_STEPS = 14400*2
test_env = Env(n_row=16,n_aisle=8,n_cp=2,n_block=4,arrival_rate=0.5,capacity=10,max_battery=100,min_battery=0,min_battery_limit=15,depletion_rate=1,recharge_rate=2,velocity=1,MAX_STEPS=MAX_STEPS,interrupt_enabled = False, seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Evaluate
results = evaluate_
    env=test_env,
    n_episodes=10,
    max_steps=MAX_STEPS,
    device=device
)

for k, v in results.items():
    if "raw" not in k:
        print(f"{k}: {v:.3f}")