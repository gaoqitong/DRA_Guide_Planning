from __future__ import division
import numpy as np
from collections import defaultdict
import gym
from gym.envs.toy_text import discrete
from utils import *
from copy import deepcopy
from full_prod_DRA import *
from gym.envs.toy_text.discrete import categorical_sample


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NONE = 4

def normalize(probs):
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]

class CurrentWorld(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        delta_list = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]]
        # delta_list = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        new_position_candidates = [np.array(current[:2]) + np.array(delta)]
        new_position_candidates += [np.array(current[:2]) + np.array(i) for i in delta_list if i != delta]
        new_positions = [self._limit_coordinates(i).astype(int) for i in new_position_candidates]
        current_rabin_state = current[-1]
        next_rabin_state = [self.rabin.next_state(current, tuple(i)) for i in new_positions]
        deadlock = [False for i in range(len(new_positions))]
        
        for i in range(len(new_positions)):
            if next_rabin_state[i] in self.rabin.deadlock:
                next_rabin_state[i] = current_rabin_state
                deadlock[i] = True

        new_positions = [i.tolist() for i in new_positions]
        next_rabin_state = [int(i) for i in next_rabin_state]
        
        new_state_3d = [tuple(new_positions[i] + [next_rabin_state[i]]) for i in range(len(new_positions))]
        new_state = [np.ravel_multi_index( i, self.shape) for i in new_state_3d]
        
        # is_done = [i[-1] in self.rabin.accept for i in new_state_3d]
        is_done = [False for i in new_state_3d]

        reward_list = []

        for i in range(len(new_state)):
            if is_done[i]:
                reward_list += [100]
            elif next_rabin_state[i] in self.rabin.accept:
                reward_list += [-1]
            elif deadlock[i] == True:
                reward_list += [-100]
            elif next_rabin_state[i] in self.rabin.reject:
                reward_list += [-10]
            else:
                reward_list += [-1]



        return [(0.8, new_state[0], reward_list[0], is_done[0]),
                 (0.05, new_state[1], reward_list[1], is_done[1]), 
                 (0.05, new_state[2], reward_list[2], is_done[2]), 
                 (0.05, new_state[3], reward_list[3], is_done[3]),
                 (0.05, new_state[4], reward_list[4], is_done[4])]

    def _reset(self):
    	self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        self.last_ap_dict = deepcopy(self.ap_dict)
        # self.last_dynamic_coord_dict = deepcopy(self.dynamic_coord_dict)
        self.ap_dict = deepcopy(self.ap_dict_static)
        # self.dynamic_coord_dict = deepcopy(self.static_coord_dict)
        rand = np.random.random(self.shape[:-1])
        for k,v in self.ap_dict_static.items():
            for j in v:
                if rand[j[0]][j[1]] > self.prob_dict[k][k]:
                    self.ap_dict[k].remove(j)
                    observed_ap_candidates = deepcopy(self.prob_dict[k])
                    observed_ap_candidates.pop(k)
                    observed_ap = np.random.choice(observed_ap_candidates.keys(), p=normalize(observed_ap_candidates.values()))
                    self.ap_dict[observed_ap] += [j]
        self.update_coord_dict()
        # print "reset"

        return self.s

    def __init__(self, ltl):
        self.start_coord = (2, 2)
        self.terminal_coord = (4, 4)
        self.shape = (5, 5)

        # prob_dict = {"A": 0.8, "B": 0.8, "C": 0.05, "T": 0.8}
        prob_dict = {"A": {"A": 0.8, "B": 0.1, "C": 0.1}, "B": {"B": 0.8, "A": 0.1, "T":0.1}, "C": {"C": 0.9, "B":0.1}, "T": {"T": 0.8, "A": 0.1, "C":0.1} }
        
        ap_dict = {"A":[(4, 0)], "B":[(0, 4)], "C":[(1, 3), (4, 2)]}
        ap_dict["T"] = [self.terminal_coord]

        static_coord_dict = defaultdict(lambda x: "")

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                static_coord_dict[(i,j)] = []

        for i in ap_dict.items():
            for j in i[1]:
                static_coord_dict[j] += i[0]
        
        self.static_coord_dict = static_coord_dict
        self.ap_dict = ap_dict
        self.prob_dict = prob_dict
        self.ap_dict_static = deepcopy(ap_dict)
        self.dynamic_coord_dict = deepcopy(static_coord_dict)
        # self.last_dynamic_coord_dict = deepcopy(self.dynamic_coord_dict)
        # self.last_ap_dict = deepcopy(self.ap_dict)
                
        #ltl = ltl + " && <>[] T"
        
        self.rabin = Rabin_Automaton(ltl, self.static_coord_dict)
        self.dynamic_rabin = Rabin_Automaton(ltl, self.dynamic_coord_dict)
        self.shape = (self.shape[0], self.shape[1], self.rabin.num_of_nodes)
        
        nS = np.prod(self.shape)
        nA = 5
        
        self.start_state = tuple( list(self.start_coord) + [self.rabin.init_state] )
        self.terminal_states = [tuple(list(self.terminal_coord) + [i]) for i in self.rabin.accept]        
        
        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])
            P[s][NONE] = self._calculate_transition_prob(position, [0, 0])

#       Set start point
        isd = np.zeros(nS)
        isd[np.ravel_multi_index(self.start_state, self.shape)] = 1.0

        self.last_s = np.ravel_multi_index(self.start_state, self.shape)

        super(CurrentWorld, self).__init__(nS, nA, P, isd)

        # print "init last dicts"

        self.last_ap_dict = deepcopy(self.ap_dict)
        self.last_dynamic_coord_dict = deepcopy(self.dynamic_coord_dict)
        
    def update_rabin(self, new_ltl):
        self.dynamic_rabin.coord_dict = self.dynamic_coord_dict

    def update_coord_dict(self):
        
        # ap_dict_static_extra = set((k,tuple(v)) for k,v in self.ap_dict_static.items()) - set((k,tuple(v)) for k,v in self.ap_dict.items())
        # ap_dict_extra = set((k,tuple(v)) for k,v in self.ap_dict.items()) - set((k,tuple(v)) for k,v in self.ap_dict_static.items())

        last_ap_dict_extra = set((k,tuple(v)) for k,v in self.last_ap_dict.items()) - set((k,tuple(v)) for k,v in self.ap_dict.items())
        ap_dict_extra = set((k,tuple(v)) for k,v in self.ap_dict.items()) - set((k,tuple(v)) for k,v in self.last_ap_dict.items())

        self.last_dynamic_coord_dict = deepcopy(self.dynamic_coord_dict)
        self.dynamic_coord_dict = deepcopy(self.last_dynamic_coord_dict)

        for k,v in last_ap_dict_extra:
            for i in v:
                self.dynamic_coord_dict[i].remove(k)

        for k,v in ap_dict_extra:
            for i in v:
                self.dynamic_coord_dict[i] += [k]

        # self.dynamic_coord_dict = defaultdict(lambda x: "")

        # for i in range(self.shape[0]):
        #     for j in range(self.shape[1]):
        #         self.dynamic_coord_dict[(i,j)] = []

        # for i in self.ap_dict.items():
        #     for j in i[1]:
        #         self.dynamic_coord_dict[j] += i[0]
    
    def step(self, action):
        ns, r, d, info = super(CurrentWorld, self).step(action)

        ns_3d = np.unravel_index(ns, self.shape)
        ns_2d = ns_3d[:-1]
        last_s_3d = np.unravel_index(self.last_s, self.shape)
        rand = np.random.random(self.shape[:-1])
        self.last_ap_dict = deepcopy(self.ap_dict)
        self.ap_dict = deepcopy(self.ap_dict_static)
        for k,v in self.ap_dict_static.items():
            for j in v:
                if rand[j[0]][j[1]] > self.prob_dict[k][k]:
                    self.ap_dict[k].remove(j)
                    observed_ap_candidates = deepcopy(self.prob_dict[k])
                    observed_ap_candidates.pop(k)
                    observed_ap = np.random.choice(observed_ap_candidates.keys(), p=normalize(observed_ap_candidates.values()))
                    self.ap_dict[observed_ap] += [j]

        # if len(self.coord_dict[ns_2d]) > 0:
        #     ap = self.coord_dict[ns_2d][0]
        #     if ap in self.prob_dict.keys():
        #         prob = self.prob_dict[ap]
        #         if rand[ns_2d[0]][ns_2d[1]] > prob:
        #             ns_3d = list(ns_3d)
        #             ns_3d[-1] = last_s_3d[-1]
        #             ns = np.ravel_multi_index(ns_3d, self.shape)

        #             d = ns_3d[-1] in self.rabin.accept

        #             if d:
        #                 r = 100
        #             elif ns_3d[-1] in self.rabin.accept:
        #                 r = -1
        #             elif ns_3d[-1] in self.rabin.deadlock:
        #                 r = -100
        #             elif ns_3d[-1] in self.rabin.reject:
        #                 r = -10
        #             else:
        #                 r = -1

        #             info = "disappear"

        self.last_s = ns
        self.update_coord_dict()
        return (ns, r, d, info)
        


def limit_coordinates(coord,world):
    coord[0] = min(coord[0], np.shape(world)[0] - 1)
    coord[0] = max(coord[0], 0)
    coord[1] = min(coord[1], np.shape(world)[1] - 1)
    coord[1] = max(coord[1], 0)
    return coord

