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
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces
#-----importating stable baseline related packages (in built PPO) ----#
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from IPython.core.magic import MAGIC_NO_VAR_EXPAND_ATTR
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import time

"""
There are 4 parts to this code, all are in this .py file.
PART 1: Main environment class
PART 2: Agent wrapper, this is a single agent PPO. This part controls the environment progression
PART 3: Main training part
PART 4: Evaluation
"""
###---------------------------------------------------PART 1: MAIN ENVIRONMENT CLASS ---------------------------------------------------###
class Env(gym.Env):
    def __init__(self, n_row, n_aisle, n_cp, n_block, arrival_rate, capacity, max_battery, min_battery, min_battery_limit, depletion_rate, recharge_rate, velocity, charging_type, reward_type, interrupt_enabled, seed, MAX_STEPS):
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

        ### Specific to Bischoff paper
		self.curr_agent = None                          # Agent that is making a decision at current time step
		self.reward_type = reward_type                  # Choosing between different reward types, refer def reward_func
		self.interrupt_enabled = interrupt_enabled      # ENabling or disabling interrupt strategy, refer def interrupt
		


		self.agent_state = {} #State of agent (aisle, row, capacity, battery, at cp or not)
		self.remaining_time_for_action = {n: 0 for n in range(1, self.n_agent + 1)} #Time taken by each agent to complete an action
		self.agent_todo_action = {n: 0 for n in range(1, self.n_agent + 1)} #Curr action of each agent
		self.agent_charging_ind = {n:False for n in range(1,self.n_agent+1)} #Indicator for agent if it is currently charging or not
		self.prev_location = {n:[(n_aisle-1)/2,(n_row-1)/2] for n in range(1,self.n_agent+1)} # Prev loc of agent
		self.next_location = {n: [] for n in range(1, self.n_agent + 1)} # Next location of agent
		self.travel_to_depot_ind = {n:False for n in range(1,self.n_agent+1)} #Indicator for if agent has to trvael to depot or not
		self.returning_from_cp_ind = {n:False for n in range(1,self.n_agent+1)}

		self.cp_state = {m:[] for m in range(1,self.n_cp+1)} #State of CP [list of what are the agents in queue]
		self.cp_queue_len = [0 for m in range(self.n_cp)] #CP queue length
		self.battery_just_before_charging = {n:100 for n in range(1,self.n_agent+1)}
		self.time_tobe_free = {m:0 for m in range(1,self.n_cp+1)} #Stores info about how much time to be free for every CP

		self.orders = [] #Stores information of incoming orders
		self.total_orders_placed = [0 for i in range(self.n_agent)] #Stores information of total orders

		self.global_time_step = 0 #Counter for time globally
		self.orders_completed = {n:[0,0,0] for n in range(1,self.n_agent+1)} #Counter for orders completed by N agents. It has 3 components per agent: no.of orders completed, total time taken to complete, time spent in charging
		self.charging_counter = {'forced_depot':[0,0,0,0],'forced_order':[0,0,0,0],'chose':[0,0,0,0]}
		self.distance_travelled = [0 for i in range(self.n_agent)]
		self.depot_counter = {n:[[],[]] for n in range(1,self.n_agent+1)}
		self.travel_time_pick, self.travel_time_depot, self.travel_time, self.travel_depot, self.travel_cp, self.charging_time, self.wait_q, self.pick_time = {n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)}
		self.action_tracker = {n:[] for n in range(1,self.n_agent+1)}
		self.battery_just_before_charging = {n:0 for n in range(1,self.n_agent+1)}
		self.battery_charge_until_level = {n:0 for n in range(1,self.n_agent+1)}
		self.service_time_total = {n:0 for n in range(1,self.n_agent+1)}

		self.total_trip_time = [0 for i in range(self.n_agent)]

		### ACTIONS
		if charging_type == 'full': num_actions = 11  # Meaning choose from [0,10,20,30,40,50,60,70,80,90,100] - 11 items
		elif charging_type == 'binary': num_actions = 2 # Meaning choose from [0,100] - 2 items
		self.action_space = spaces.Discrete(num_actions)

		self.seed = seed
		self.reset()

	def observe_env(self): #function needed during training to look at the curret env in order to get the very first observation immediately, before the agent takes even one action.
		agent_requiring_decision = self.curr_agent
		n_depleted_agvs = 0
		for i in range(1,self.n_agent+1):
		  if self.agent_state[i][4] != 0: n_depleted_agvs += 1
		n_depleted_agvs = n_depleted_agvs/self.n_agent

		n_free_agv = 0 #idle and available.
		for i in range(1,self.n_agent+1):
		  if self.agent_state[i][4] == 0 and self.remaining_time_for_action[i] == 0: n_free_agv += 1
		n_free_agv = n_free_agv/self.n_agent

		n_working_agvs = 0
		for i in range(1,self.n_agent+1):
		  if self.agent_state[i][4] == 0 and self.remaining_time_for_action[i] > 0: n_working_agvs += 1
		n_working_agvs = n_working_agvs/self.n_agent

		tot_battery_working = 0 #Average battery of working AGVs.
		for i in range(1,self.n_agent+1):
		  if self.agent_state[i][4] == 0 and self.remaining_time_for_action[i] > 0: tot_battery_working += self.agent_state[i][3]
		if n_working_agvs != 0: avg_battery_working = (tot_battery_working/(n_working_agvs * self.n_agent))/100
		else: avg_battery_working = 0

		battery_cs1 = 0 # Battery level of AGV currently charging at charging station 1. #HERE4 - CHane if increase in numbber of CP or robots
		if self.cp_state[1] != []:
		  j = self.cp_state[1][0]
		  battery_cs1 = self.agent_state[j][3]/100
		battery_cs2 = 0
		if self.cp_state[2] != []:
		  j = self.cp_state[2][0]
		  battery_cs2 = self.agent_state[j][3]/100

		queue_len_cs1 = len(self.cp_state[1])/self.n_agent
		queue_len_cs2 = len(self.cp_state[2])/self.n_agent

		if agent_requiring_decision != None:
		  battery_a = self.agent_state[agent_requiring_decision][3]/100
		  dist_cp1a = self.func_distance((self.agent_state[agent_requiring_decision][0],self.agent_state[agent_requiring_decision][1]),self.charger_location[0]) / self.max_distance
		  dist_cp2a = self.func_distance((self.agent_state[agent_requiring_decision][0],self.agent_state[agent_requiring_decision][1]),self.charger_location[1]) / self.max_distance
		else:
		  battery_a = 1
		  dist_cp1a = 1
		  dist_cp2a = 1

		time = self.global_time_step
		if agent_requiring_decision != None: ratio_of_orders_completed = self.orders_completed[agent_requiring_decision][0]/self.total_orders_placed[agent_requiring_decision-1]
		else: ratio_of_orders_completed = 0

		obs = np.array([n_depleted_agvs, n_free_agv, n_working_agvs, avg_battery_working, battery_cs1, battery_cs2,queue_len_cs1, queue_len_cs2,battery_a, dist_cp1a,dist_cp2a, time], dtype=np.float32)
		return obs

	def reset(self):
		self.curr_agent = None

		self.agent_state = {} #State of agent (aisle, row, capacity, battery, at cp or not)
		for n in range(1,self.n_agent+1):
		  self.agent_state[n] = [(self.n_aisle-1)/2,(self.n_row-1)/2,self.capacity,self.max_battery,-1] #Initial state of agent
		self.remaining_time_for_action = {n: 0 for n in range(1, self.n_agent + 1)} #Time taken by each agent to complete an action
		self.agent_todo_action = {n: 0 for n in range(1, self.n_agent + 1)} #Curr action of each agent
		self.agent_charging_ind = {n:False for n in range(1,self.n_agent+1)} #Indicator for agent if it is currently charging or not
		self.prev_location = {n:[(self.n_aisle-1)/2,(self.n_row-1)/2] for n in range(1,self.n_agent+1)}
		self.next_location = {n: [] for n in range(1, self.n_agent + 1)}
		self.travel_to_depot_ind = {n:False for n in range(1,self.n_agent+1)}

		self.cp_state = {m:[] for m in range(1,self.n_cp+1)}
		self.cp_queue_len = [0 for m in range(self.n_cp)] #State of CP [list of what are the agents in queue]
		self.time_tobe_free = {m:0 for m in range(1,self.n_cp+1)}
		self.battery_just_before_charging = {n:100 for n in range(1,self.n_agent+1)}

		self.orders = [] #Stores information of incoming orders
		self.total_orders_placed = [0 for i in range(self.n_agent)]  #Stores information of total orders
		for n in range(1,self.n_agent+1):
		  OB = n
		  minx,maxx,miny,maxy = self.block_border[OB]['minx'],self.block_border[OB]['maxx'],self.block_border[OB]['miny'],self.block_border[OB]['maxy']
		  OH,OV,OK,ind_completed,id = np.random.randint(minx,maxx+1),np.random.randint(miny,maxy+1),1,0,len(self.orders)+1
		  arrival_time,completion_time = self.global_time_step, -1
		  self.orders.append([OB,OH,OV,OK,ind_completed,id,arrival_time,completion_time])
		  self.next_location[n] = [self.orders[n-1][1], self.orders[n-1][2]]
		  self.total_orders_placed[n-1] = 1
		# self.action_counter = {n:[0,0,0,0,0] for n in range(1,self.n_agent+1)}
		self.charging_counter = {'forced_depot':[0,0,0,0],'forced_order':[0,0,0,0],'chose':[0,0,0,0]}
		self.distance_travelled = [0 for i in range(self.n_agent)]
		self.travel_time_pick, self.travel_time_depot, self.travel_time, self.travel_depot, self.travel_cp, self.charging_time, self.wait_q, self.pick_time = {n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)},{n:0 for n in range(1, self.n_agent+1)}
		self.service_time_total = {n:0 for n in range(1,self.n_agent+1)}
		self.total_trip_time = [0 for i in range(self.n_agent)]
		self.global_time_step = 0 #Counter for time globally
		self.orders_completed = {n:[0,0,0] for n in range(1,self.n_agent+1)} #Counter for orders completed by N agents. It has 3 components per agent: no.of orders completed, total time taken to complete, time spent in charging
		return self.observe_env(), {'global_time_step': self.global_time_step}

	def new_order(self,t):
		rng = np.random.default_rng(seed=None)
		k = rng.poisson(self.arrival_rate*t)
		while k > 0:
		  OB = np.random.randint(1,self.n_block+1)
		  minx,maxx,miny,maxy = self.block_border[OB]['minx'],self.block_border[OB]['maxx'],self.block_border[OB]['miny'],self.block_border[OB]['maxy']
		  OH,OV,OK = rng.integers(minx,maxx+1),rng.integers(miny,maxy+1),1 #HERE1 - be careful if capacity = 1
		  ind_completed = 0
		  id = len(self.orders)+1
		  arrival_time,completion_time = self.global_time_step, -1
		  order_info = [OB,OH,OV,OK,ind_completed,id,arrival_time,completion_time]
		  self.orders.append(order_info)
		  k -= 1
		  self.total_orders_placed[OB-1] += 1

	def func_distance(self, a, b):
		return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

	def func_travel_time(self, a, b):
		distance = self.func_distance(a, b)
		return max(1, distance / self.velocity)


	def check_battery_sufficiency(self,n,destination):
		origin = [self.agent_state[n][0],self.agent_state[n][1]]
		distance = self.func_distance(origin,destination)
		time = max(1, int(np.ceil(distance / self.velocity)))
		battery_needed = min(100,self.depletion_rate * time)
		return (self.agent_state[n][3] - battery_needed) >= 0

	def check_capacity_sufficiency(self,n,num_of_items):
		if (self.agent_state[n][2] - num_of_items) >= 0: return True
		else: return False

	def next_order_to_pick(self, n):
		go_to_order = None
		for order in self.orders:
		  if order[0] == n and order[4] == 0:
			go_to_order = order
			break
		return go_to_order

	def num_of_orders_remaining(self,n):
		num_of_orders = 0
		for order in self.orders:
		  if order[0] == n and order[4] == 0:
			num_of_orders += 1
		return num_of_orders


	def fixed_cp_selection_rule(self,n):  #Always returns nearest CS
		distances_to_cp = [self.func_distance((self.agent_state[n][0],self.agent_state[n][1]),self.charger_location[i-1]) for i in range(1,self.n_cp+1)]
		nearest_cp_id = np.argmin(distances_to_cp) + 1
		if self.cp_state[nearest_cp_id] == []: return nearest_cp_id
		else:
		  cp_with_shortest_queue = nearest_cp_id
		  shortest_queue_len = len(self.cp_state[cp_with_shortest_queue])
		  for cp_id in range(1,self.n_cp+1):
			if self.cp_state[cp_id] != [] and len(self.cp_state[cp_id]) < shortest_queue_len:
			  cp_with_shortest_queue = cp_id
		  return cp_with_shortest_queue

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

	def reward_func(self,n,type_,action):
		st_avg = -1 * (self.service_time_total[n]+self.remaining_time_for_action[n])/(self.orders_completed[n][0]+1)
		if type_ == 'service_time_based': return -1 * st_avg
		elif type_ == 'queue_based': return -1 * self.num_of_orders_remaining(n)
		elif type_ == 'composite':
		  q = self.num_of_orders_remaining(n)
		  q_ind = 0
		  for i in range(1,self.n_agent+1):
			if self.num_of_orders_remaining(i) > 0: q_ind = 1
		  num_of_free_robots = 0 # robots that are idle - at block, not travelling, no next order
		  for i in range(1,self.n_agent+1):
			if self.remaining_time_for_action[i] == 0 and self.agent_state[i][4] == 0 and self.next_order_to_pick(i) == None: num_of_free_robots += 1
		  return -1 * (q + st_avg) + q_ind * num_of_free_robots/self.n_agent
		elif type_ == 'shaped':
		  r = 0
		  instructed_to_charge = False
		  if action in range(1,11): instructed_to_charge = True
		  q = self.num_of_orders_remaining(n)
		  num_of_free_cp = 0
		  for i in range(1,self.n_cp+1):
			if self.cp_state[i] == []: num_of_free_cp += 1
		  if q == 0 and instructed_to_charge == True: r += 1
		  if num_of_free_cp > 0 and instructed_to_charge == True: r += 1
		  if num_of_free_cp == 0 and instructed_to_charge == True: r -= 1
		  if q > 0 and instructed_to_charge == True: r -= 1
		  return -1 * q + r

	def allowed_action(self, n):
		# 0 - continue working (no charge)
		# 10, 20 ....,90, 100 - go to CP and charge till 10, 20 ..., 90, 100
		# charging type = full then actions are 0,10,20...100 (total 11 actions)
		# charging type = binary then actions 0,100 (total 2 actions)

		if self.charging_type == 'full': allowed_action_info = np.zeros(11, dtype=bool)
		elif self.charging_type == 'binary': allowed_action_info = np.zeros(2, dtype=bool)
		stop_charging_index = self.n_cp + 1
		go_to_depot_index = self.n_cp + 2


		# Travelling
		if self.remaining_time_for_action[n] > 0:
			allowed_action_info[:] = False

		# Not travelling
		elif self.remaining_time_for_action[n] == 0 and self.agent_state[n][4] == 0:
			next_order = self.next_order_to_pick(n)

			if next_order == None:
				### Ensuring that agents dont decide to choose a battery thresholw below its current battery level
				if self.charging_type == 'full':
					minimum_level = 10 * math.ceil(max(self.min_battery_limit,self.agent_state[n][3])/ 10)
					allowed_action_info[minimum_level//10:] = True
				elif self.charging_type == 'binary': allowed_action_info[1] = True

			if next_order:
				battery_needed_to_go_to_next_order = min(100,self.depletion_rate * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),(next_order[1],next_order[2])))
				battery_needed_to_go_to_depot = min(100,self.depletion_rate * 2 * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location))

				if self.agent_state[n][3] < (self.min_battery_limit + battery_needed_to_go_to_next_order):  # Go to CP
					cp_to_go_to = self.fixed_cp_selection_rule(n)
					battery_needed_to_go_to_cp_and_return = min(100,self.depletion_rate * 2 * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.charger_location[cp_to_go_to-1]))
					if self.charging_type == 'full':
						minimum_level = 10 * math.ceil(max((self.min_battery_limit + battery_needed_to_go_to_cp_and_return),self.agent_state[n][3])/ 10)
						allowed_action_info[minimum_level//10:] = True
					elif self.charging_type == 'binary': allowed_action_info[1] = True

				elif self.agent_state[n][3] >= (self.min_battery_limit + battery_needed_to_go_to_next_order) and self.check_capacity_sufficiency(n, next_order[3]):  # Can pick or charge
					allowed_action_info[0] = True # Becaue it goes to pick up
					minimum_level = 10 * math.ceil(max((self.min_battery_limit + battery_needed_to_go_to_next_order),self.agent_state[n][3])/ 10)
					if (self.agent_state[n][3] != 100): allowed_action_info[minimum_level//10:] = True

				elif self.agent_state[n][3] >= (self.min_battery_limit + battery_needed_to_go_to_next_order) and self.check_capacity_sufficiency(n, next_order[3]) == False:
					if self.agent_state[n][3] < self.min_battery_limit + battery_needed_to_go_to_depot:
						if self.charging_type == 'full':
							minimum_level = 10 * math.ceil(max((self.min_battery_limit + battery_needed_to_go_to_depot),self.agent_state[n][3])/ 10)
							allowed_action_info[minimum_level//10:] = True
						elif self.charging_type == 'binary': allowed_action_info[1] = True

		elif self.remaining_time_for_action[n] == 0 and self.agent_state[n][4] != 0:
			# In Bischoffs work no decision is made while at CS
			allowed_action_info[:] = False

		return allowed_action_info

	def apply_step(self,agent,action): #action_array = [a,b,c,d] where a.b.c.d are the actions for each of the 4 agents

		stop_charging_index = self.n_cp + 1
		go_to_depot_index = self.n_cp + 2

		# Action based on Interrupt
		if self.interrupt_enabled == True: agent_to_be_interrupted = self.interrupt()
		else: agent_to_be_interrupted = None

		REWARD = 0

		# 1. If it is currently travelling
		for n in range(agent,agent+1):
		    if self.remaining_time_for_action[n] > 0:
				REWARD += 0
				continue

		    if self.remaining_time_for_action[n] == 0:
				if self.agent_state[n][4] != 0:
					cp_id = self.agent_state[n][4]
					if (self.cp_state[cp_id][0] == n and self.agent_state[n][3] >= self.battery_charge_until_level[n]) or (agent_to_be_interrupted == True and agent_to_be_interrupted == n):
						self.agent_charging_ind[n] = False
						self.cp_state[cp_id].pop(0)
						self.cp_queue_len[cp_id-1] -= 1
						self.agent_state[n][4] = 0
						self.next_location[n] = list([self.prev_location[n][0], self.prev_location[n][1]]) # it return to original point in the block before it decided to travel to CP
						self.prev_location[n] = list(self.charger_location[cp_id-1])
						self.remaining_time_for_action[n] = self.func_travel_time(self.prev_location[n],self.next_location[n])
						self.returning_from_cp_ind[n] = True
						self.distance_travelled[n-1] += self.remaining_time_for_action[n]
						self.travel_time[n] += self.remaining_time_for_action[n]
						self.total_trip_time[n-1] = self.remaining_time_for_action[n]
						REWARD +=  0

					elif self.cp_state[cp_id][0] == n and self.agent_state[n][3] < self.battery_charge_until_level[n]: # Continue charging
						self.agent_state[n][3] = min(self.max_battery, self.agent_state[n][3] + self.recharge_rate)
						self.orders_completed[n][1] += 1
						self.orders_completed[n][2] += 1
						self.remaining_time_for_action[n] = 0
						self.charging_time[n] += 1
						REWARD += 0

					elif self.cp_state[cp_id][0] != n:
						self.orders_completed[n][1] += 1 # Wait in queue
						self.remaining_time_for_action[n] = 0
						self.wait_q[n] += 1
						REWARD += 0

			# 3. If agent is at block then
			elif self.agent_state[n][4] == 0:
			    battery_needed_to_go_to_depot = min(100,self.depletion_rate * 2 * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location))
			    next_order = self.next_order_to_pick(n)
			    if next_order == None and action in [i for i in range(1,11)]:
					self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
					cp_to_go_to = self.fixed_cp_selection_rule(n)
					self.next_location[n] = list(self.charger_location[cp_to_go_to-1])
					self.remaining_time_for_action[n] = self.func_travel_time(self.prev_location[n],self.next_location[n])
					self.battery_just_before_charging[n] = self.agent_state[n][3] - self.depletion_rate * self.remaining_time_for_action[n]
					self.battery_charge_until_level[n] = action * 10
					self.distance_travelled[n-1] += self.remaining_time_for_action[n]
					self.travel_time[n] += self.remaining_time_for_action[n]
					self.travel_cp[n] += self.remaining_time_for_action[n]
					self.total_trip_time[n-1] = self.remaining_time_for_action[n]
					REWARD += self.reward_func(n,self.reward_type,action)


			    elif next_order:
					battery_needed_to_go_to_next_order = min(100,self.depletion_rate * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),(next_order[1],next_order[2])))
					if self.agent_state[n][3] < (self.min_battery_limit + battery_needed_to_go_to_next_order) and action in [i for i in range(1,11)]:
						self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
						cp_to_go_to = self.fixed_cp_selection_rule(n)
						self.next_location[n] = list(self.charger_location[cp_to_go_to-1])
						self.remaining_time_for_action[n] = self.func_travel_time(self.prev_location[n],self.next_location[n])
						self.battery_just_before_charging[n] = self.agent_state[n][3] - self.depletion_rate * self.remaining_time_for_action[n]
						self.battery_charge_until_level[n] = action*10
						self.distance_travelled[n-1] += self.remaining_time_for_action[n]
						self.travel_time[n] += self.remaining_time_for_action[n]
						self.travel_cp[n] += self.remaining_time_for_action[n]
						REWARD += self.reward_func(n,self.reward_type,action)
						self.total_trip_time[n-1] = self.remaining_time_for_action[n]
				  
					elif self.agent_state[n][3] >= (self.min_battery_limit + battery_needed_to_go_to_next_order) and self.check_capacity_sufficiency(n, next_order[3]) == True: # Go to Order
						if action == 0:
							self.remaining_time_for_action[n] = self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),(next_order[1],next_order[2]))
							self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
							self.next_location[n] = list([next_order[1], next_order[2]])
							self.distance_travelled[n-1] += self.remaining_time_for_action[n]
							self.travel_time_pick[n] += self.remaining_time_for_action[n]
							self.travel_time[n] += self.remaining_time_for_action[n]
							self.pick_time[n] += self.remaining_time_for_action[n]
							REWARD += 0
							self.total_trip_time[n-1] = self.remaining_time_for_action[n]

						elif action in [i for i in range(1,11)]:
							self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
							cp_to_go_to = self.fixed_cp_selection_rule(n)
							self.next_location[n] = list(self.charger_location[cp_to_go_to-1])
							self.remaining_time_for_action[n] = self.func_travel_time(self.prev_location[n],self.next_location[n])
							self.battery_just_before_charging[n] = self.agent_state[n][3] - self.depletion_rate * self.remaining_time_for_action[n]
							self.battery_charge_until_level[n] = action*10
							self.distance_travelled[n-1] += self.remaining_time_for_action[n]
							self.travel_time[n] += self.remaining_time_for_action[n]
							self.travel_cp[n] += self.remaining_time_for_action[n]
							REWARD += self.reward_func(n,self.reward_type,action)
							self.total_trip_time[n-1] = self.remaining_time_for_action[n]
							  
					elif self.agent_state[n][3] >= (self.min_battery_limit + battery_needed_to_go_to_next_order) and self.check_capacity_sufficiency(n, next_order[3]) == False:
						if self.agent_state[n][3] >= self.min_battery_limit + battery_needed_to_go_to_depot:
							self.remaining_time_for_action[n] = 2 * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location)
							self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
							self.next_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
							self.travel_to_depot_ind[n] = True
							self.distance_travelled[n-1] += self.remaining_time_for_action[n]
							self.depot_counter[n][0].append(self.agent_state[n][2])
							self.depot_counter[n][1].append(self.num_of_orders_remaining(n))
							self.travel_time_depot[n] +=  2 * self.func_travel_time((self.agent_state[n][0],self.agent_state[n][1]),self.depot_location)
							self.travel_time[n] += self.remaining_time_for_action[n]
							self.travel_depot[n] += self.remaining_time_for_action[n]
							self.total_trip_time[n-1] = self.remaining_time_for_action[n]
							REWARD += 0
						
						elif self.agent_state[n][3] < (self.min_battery_limit + battery_needed_to_go_to_depot) and action in [i for i in range(1,11)]:
							self.prev_location[n] = list([self.agent_state[n][0], self.agent_state[n][1]])
							cp_to_go_to = self.fixed_cp_selection_rule(n)
							self.next_location[n] = list(self.charger_location[cp_to_go_to-1])
							self.remaining_time_for_action[n] = self.func_travel_time(self.prev_location[n],self.next_location[n])
							self.battery_just_before_charging[n] = self.agent_state[n][3] - self.depletion_rate * self.remaining_time_for_action[n]
							self.battery_charge_until_level[n] = action*10
							self.distance_travelled[n-1] += self.remaining_time_for_action[n]
							self.travel_time[n] += self.remaining_time_for_action[n]
							self.travel_cp[n] += self.remaining_time_for_action[n]
							REWARD += self.reward_func(n,self.reward_type,action)
							self.total_trip_time[n-1] = self.remaining_time_for_action[n]
							
							
		###___Change location of agents or if they are in motion, reduce their time by 1 unit___###
		for n in range(agent,agent+1):
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

				if at_cp == False:
					if self.travel_to_depot_ind[n] == False and self.returning_from_cp_ind[n] == False:
						for i, o in enumerate(self.orders):
							if (o[0] == n and o[4] == 0 and np.isclose(o[1],self.agent_state[n][0],atol=1e-3) and np.isclose(o[2],self.agent_state[n][1],atol=1e-3)):#[OB,OH,OV,OK,ind_completed,id,arrival_time,completion_time]
								o[4] = 1
								o[7] = self.global_time_step
								self.agent_state[n][2] -= o[3]
								self.orders_completed[n][0] += 1
								self.service_time_total[n] = self.service_time_total[n] + o[7] - o[6]
								self.orders.pop(i)
								break
					elif self.travel_to_depot_ind[n] == True:
						self.agent_state[n][2] = self.capacity
						self.travel_to_depot_ind[n] = False
					elif self.returning_from_cp_ind[n] == True:
						self.returning_from_cp_ind[n] = False

				else:
					for m in range(1, self.n_cp+1):
						if (np.isclose(self.agent_state[n][0],self.charger_location[m-1][0],atol=1e-3) and np.isclose(self.agent_state[n][1],self.charger_location[m-1][1],atol=1e-3)):
							if self.agent_state[n][4] != m:
								self.agent_state[n][4] = m
								self.cp_state[m].append(n)
								self.cp_queue_len[m-1] += 1

		return REWARD

	def step(self, action):
		self.action_tracker[self.curr_agent].append(action)
		reward = self.apply_step(self.curr_agent, action)
		return reward
		
###--------------------------Part 2: MAIN AGENT WRAPPER CLASS THAT CONTROLS AGENT MOVEMENTS AND STEP PROGRESSION -----------------------------###
class SingleAgentWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env

        obs_dim = 12
        action_dim = 11  # or 2 for binary
        self.decision_count = {i: 0 for i in range(1, env.n_agent + 1)}
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(action_dim)

    def action_masks(self):
        curr_agent = self._get_curr_agent()
        if curr_agent is None:
            return np.ones(self.action_space.n, dtype=bool)
        agent_mask = self.env.allowed_action(curr_agent)  # already 1D for this agent
        mask = np.array(agent_mask, dtype=bool)
        if not np.any(mask):
          print(f"WARNING: all-False mask for agent {curr_agent}")
          mask[0] = True
        return mask         

    # Decides which agent is to make decision since this is a single agent ppo
    def _get_curr_agent(self):
        for i in range(1, self.env.n_agent + 1):
            if self.env.remaining_time_for_action[i] == 0 and self.env.agent_state[i][4] == 0:
                mask = np.array(self.env.allowed_action(i), dtype=bool)
                if np.any(mask):
                    return i
        return None

    def _tick_until_decision_needed(self):
        for _ in range(10_000):
            if self.env.global_time_step >= self.env.MAX_STEPS:
                return
            # Check if any agent needs a decision
            for i in range(1, self.env.n_agent + 1):
                if self.env.remaining_time_for_action[i] == 0 and \
                  self.env.agent_state[i][4] == 0:
                    mask = np.array(self.env.allowed_action(i), dtype=bool)
                    if np.any(mask):
                        return  # stop

            # No decision needed →tick ALL agents + increment clock
            for i in range(1, self.env.n_agent + 1):
                if self.env.remaining_time_for_action[i] > 0:
                    self.env.apply_step(i, -1)
                else:
                    self.env.apply_step(i, 0)

            self.env.global_time_step += 1  # ← clock advances even when all travelling
            self.env.new_order(1)
        return False

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        self.decision_count = {i: 0 for i in range(1, self.env.n_agent + 1)}
        self._tick_until_decision_needed()
        curr_agent = self._get_curr_agent()
        self.env.curr_agent = curr_agent
        obs = self.env.observe_env()
        return np.array(obs, dtype=np.float32), {}


    def step(self, action):
        curr_agent = self._get_curr_agent()
        self.env.curr_agent = curr_agent

        if curr_agent is None:
            for i in range(1, self.env.n_agent + 1):
                if i != curr_agent:
                    default = -1 if self.env.remaining_time_for_action[i] > 0 else 0
                    self.env.apply_step(i, default)

            self.env.global_time_step += 1
            self.env.new_order(1)
            obs, reward = self.env.observe_env(), 0
            terminated = self.env.global_time_step >= self.env.MAX_STEPS
            return np.array(obs, dtype=np.float32), reward, terminated, False, {}

        reward = self.env.step(action)
        self.decision_count[curr_agent] += 1

        for i in range(1, self.env.n_agent + 1):
            if i != curr_agent:
                default = -1 if self.env.remaining_time_for_action[i] > 0 else 0
                self.env.apply_step(i, default)

        self.env.global_time_step += 1
        self.env.new_order(1)

        next_agent = self._get_curr_agent()
        if next_agent is not None:
            self.env.curr_agent = next_agent
        obs = self.env.observe_env()

        terminated = self.env.global_time_step >= self.env.MAX_STEPS
        return np.array(obs, dtype=np.float32), reward, terminated, False, {}
		
		
###-------------------------------------------------------PART3: MAIN TRAINIGN PART ---------------------------------------------------###
class TrainingMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        # Called after each PPO update — losses are logged here
        if len(self.model.logger.name_to_value) > 0:
            self.losses.append(self.model.logger.name_to_value.get('train/loss', None))
            self.policy_losses.append(self.model.logger.name_to_value.get('train/policy_gradient_loss', None))
            self.value_losses.append(self.model.logger.name_to_value.get('train/value_loss', None))
            self.entropy_losses.append(self.model.logger.name_to_value.get('train/entropy_loss', None))
			

start_time = time.time()

MAX_STEPS = 14400
total_timesteps = 4_000_000
print(f"Total timesteps: {total_timesteps}")

# Create env for training
raw_env_train = Env(
    n_row=16, n_aisle=8, n_cp=2, n_block=4,
    arrival_rate=0.6, capacity=10, max_battery=100,
    min_battery=0, min_battery_limit=15, depletion_rate=1,
    recharge_rate=2, velocity=1,
    charging_type='full', reward_type='shaped',
    interrupt_enabled=False, seed=42, MAX_STEPS = MAX_STEPS
)

wrapped_env_train = SingleAgentWrapper(raw_env_train)
masked_env = ActionMasker(wrapped_env_train, lambda env: env.action_masks())

# Create and train model
model = MaskablePPO(
    "MlpPolicy",
    masked_env,
    verbose=0,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005
)

metrics_callback = TrainingMetricsCallback()
model.learn(total_timesteps=total_timesteps, callback=metrics_callback)
model.save("maskable_ppo_agent")

end_time = time.time()
print(f"Total time taken: {end_time - start_time}")

import pickle
metrics = {
    'losses':         metrics_callback.losses,
    'policy_losses':  metrics_callback.policy_losses,
    'value_losses':   metrics_callback.value_losses,
    'entropy_losses': metrics_callback.entropy_losses,
}

with open("sb3_training_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("Model and metrics saved.")

###-------------------------------------------------------PART4: MAIN TESTING PART ---------------------------------------------------###

MAX_STEPS = 14400*2
raw_env_test = Env(
    n_row=16, n_aisle=8, n_cp=2, n_block=4,
    arrival_rate=0.6, capacity=10, max_battery=100,
    min_battery=0, min_battery_limit=15, depletion_rate=1,
    recharge_rate=2, velocity=1,
    charging_type='full', reward_type='shaped',
    interrupt_enabled=False, seed=42, MAX_STEPS = MAX_STEPS
)
wrapped_env_test = SingleAgentWrapper(raw_env_test)

model = MaskablePPO.load("maskable_ppo_agent", env=ActionMasker(wrapped_env_test, lambda env: env.action_masks()))


n_test_episodes = 10
orders_completed = []
orders_completed1 = []
orders_completed2 = []
orders_completed3 = []
orders_completed4 = []
total_orders_placed = []
total_orders_placed1 = []
total_orders_placed2 = []
total_orders_placed3 = []
total_orders_placed4 = []
travel_time_ = []
travel_time1 = []
travel_time2 = []
travel_time3 = []
travel_time4 = []
travel_depot_ = []
travel_depot1 = []
travel_depot2 = []
travel_depot3 = []
travel_depot4 = []
travel_cp_ = []
travel_cp1 = []
travel_cp2 = []
travel_cp3 = []
travel_cp4 = []
charging_time_ = []
charging_time1 = []
charging_time2 = []
charging_time3 = []
charging_time4 = []
wait_q_ = []
wait_q1 = []
wait_q2 = []
wait_q3 = []
wait_q4 = []
pick_time_ = []
pick_time1 = []
pick_time2 = []
pick_time3 = []
pick_time4 = []

results = []
actions = []
trackfree = 0
for episode in range(n_test_episodes):
    obs, _ = wrapped_env_test.reset()
    decision_count = 0

    while True:
        length = 0
        for i in range(1,wrapped_env_test.env.n_agent):
          if wrapped_env_test.env.remaining_time_for_action[i] > 0:
            length += 1
        if length > 1: trackfree += 1
        action, _ = model.predict(obs, action_masks=wrapped_env_test.action_masks(), deterministic=True)
        actions.append(action)
        obs, reward, terminated, truncated, info = wrapped_env_test.step(action)
        decision_count += 1

        if terminated or truncated:
            orders_completed.append(wrapped_env_test.env.orders_completed[1][0]+wrapped_env_test.env.orders_completed[2][0]+wrapped_env_test.env.orders_completed[3][0]+wrapped_env_test.env.orders_completed[4][0])
            orders_completed1.append(wrapped_env_test.env.orders_completed[1][0])
            orders_completed2.append(wrapped_env_test.env.orders_completed[2][0])
            orders_completed3.append(wrapped_env_test.env.orders_completed[3][0])
            orders_completed4.append(wrapped_env_test.env.orders_completed[4][0])
            total_orders_placed.append(wrapped_env_test.env.total_orders_placed[0]+wrapped_env_test.env.total_orders_placed[1]+wrapped_env_test.env.total_orders_placed[2]+wrapped_env_test.env.total_orders_placed[3])
            total_orders_placed1.append(wrapped_env_test.env.total_orders_placed[0])
            total_orders_placed2.append(wrapped_env_test.env.total_orders_placed[1])
            total_orders_placed3.append(wrapped_env_test.env.total_orders_placed[2])
            total_orders_placed4.append(wrapped_env_test.env.total_orders_placed[3])
            travel_time_.append(wrapped_env_test.env.travel_time[1]+wrapped_env_test.env.travel_time[2]+wrapped_env_test.env.travel_time[3]+wrapped_env_test.env.travel_time[4])
            travel_time1.append(wrapped_env_test.env.travel_time[1])
            travel_time2.append(wrapped_env_test.env.travel_time[2])
            travel_time3.append(wrapped_env_test.env.travel_time[3])
            travel_time4.append(wrapped_env_test.env.travel_time[4])
            travel_depot_.append(wrapped_env_test.env.travel_depot[1]+wrapped_env_test.env.travel_depot[2]+wrapped_env_test.env.travel_depot[3]+wrapped_env_test.env.travel_depot[4])
            travel_depot1.append(wrapped_env_test.env.travel_depot[1])
            travel_depot2.append(wrapped_env_test.env.travel_depot[2])
            travel_depot3.append(wrapped_env_test.env.travel_depot[3])
            travel_depot4.append(wrapped_env_test.env.travel_depot[4])
            travel_cp_.append(wrapped_env_test.env.travel_cp[1]+wrapped_env_test.env.travel_cp[2]+wrapped_env_test.env.travel_cp[3]+wrapped_env_test.env.travel_cp[4])
            travel_cp1.append(wrapped_env_test.env.travel_cp[1])
            travel_cp2.append(wrapped_env_test.env.travel_cp[2])
            travel_cp3.append(wrapped_env_test.env.travel_cp[3])
            travel_cp4.append(wrapped_env_test.env.travel_cp[4])
            charging_time_.append(wrapped_env_test.env.charging_time[1]+wrapped_env_test.env.charging_time[2]+wrapped_env_test.env.charging_time[3]+wrapped_env_test.env.charging_time[4])
            charging_time1.append(wrapped_env_test.env.charging_time[1])
            charging_time2.append(wrapped_env_test.env.charging_time[2])
            charging_time3.append(wrapped_env_test.env.charging_time[3])
            charging_time4.append(wrapped_env_test.env.charging_time[4])
            wait_q_.append(wrapped_env_test.env.wait_q[1]+wrapped_env_test.env.wait_q[2]+wrapped_env_test.env.wait_q[3]+wrapped_env_test.env.wait_q[4])
            wait_q1.append(wrapped_env_test.env.wait_q[1])
            wait_q2.append(wrapped_env_test.env.wait_q[2])
            wait_q3.append(wrapped_env_test.env.wait_q[3])
            wait_q4.append(wrapped_env_test.env.wait_q[4])
            pick_time_.append(wrapped_env_test.env.pick_time[1]+wrapped_env_test.env.pick_time[2]+wrapped_env_test.env.pick_time[3]+wrapped_env_test.env.pick_time[4])
            pick_time1.append(wrapped_env_test.env.pick_time[1])
            pick_time2.append(wrapped_env_test.env.pick_time[2])
            pick_time3.append(wrapped_env_test.env.pick_time[3])
            pick_time4.append(wrapped_env_test.env.pick_time[4])


            results.append({
                'episode':episode + 1,
                'decisions':decision_count})
            break
    # unique, counts = np.unique(actions, return_counts=True)
    # print(unique)
    # print(counts)