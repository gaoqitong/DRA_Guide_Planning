import networkx as nx
from networkx import DiGraph
from networkx import dijkstra_path
from Visualize import plot_buchi
import os
import sys

def find_ampersand(input_str):
    index = []
    original_length = len(input_str)
    original_str = input_str
    while input_str.find('&')>=0:
        index += [input_str.index('&')-len(input_str)+original_length]
        input_str = original_str[index[-1]+1:]
    return index

def seperate_ap_sentence(input_str):
    if len(input_str)>1:
        index = find_ampersand(input_str)
        if len(index)>=1:
            return_str = [input_str[0:index[0]]]
        else:
            return_str = input_str
            if '!' in return_str:
                return [],[return_str.replace('!','')]
            else:
                return [return_str],[]
        for i in range(1,len(index)):
            return_str += [input_str[index[i-1]+1:index[i]]]
        return_str = return_str + [input_str[index[-1]+1:]]
        return_str = [i.replace(' ','') for i in return_str]
    elif len(input_str)==1:
        return_str = input_str
    elif len(input_str) == 0:    
        raise AttributeError('input_str has no length')
        
    without_negs = []
    negations = []
    for i in range(len(return_str)):
        if '!' in return_str[i]:
            negations += [return_str[i].replace('!','')]
        else:
            without_negs += [return_str[i]]
    return without_negs,negations

def check_negations(set1,neg1,set2,neg2):
    for i in set1:
        if i in neg2:
            return False
    for i in set2:
        if i in neg1:
            return False
    return True

class Region(object):
    def __init__(self, coord, app, name=None):
        ## ap is a LIST!!!
        self.coord = coord
        self.name = name
        self.app = app
        self.apr = [self.name]
        self.ap = self.app + self.apr

    def __repr__(self):
        return self.name

class wFTS(object):
    def __init__(self, states=set(), transition={}, initial=set()):
        # trans_relat is a matrix stores control words
        ## i.e. trans_relat[2][4] 
        # weight is a matrix stores transition cost if reachable, otherwise +infty
        ## i.e. trans_relat[4][9] represents the weight when travel from region 5 to region 10
        self.states = states
        # states stores Region objects
        self.transition = transition
        self.initial = initial
        
    def add_states(self,new):
        self.states.add(new)
        if new not in self.transition.keys():
            self.transition[new] = {}
            self.transition[new][new] = 0
        else:
            raise AttributeError('This node is already in states')
        
    def add_initial(self,new):
        self.initial.add(new)

    def replace_initial(self, new):
        while len(self.initial) > 0:
            self.initial.pop()
        self.initial.add(new)
        
    def add_transition(self,state1,state2,w=0,symmetric=True):
        if state1 not in self.transition.keys():
            self.transition[state1] = {}
        if state2 not in self.transition.keys():
            self.transition[state2] = {}
        self.transition[state1][state2] = w
        if symmetric == True:
            self.transition[state2][state1] = w
        
    def change_weight(self,state1,state2,w,symmetric=True):
        self.transition[state1][state2] = w
        if symmetric == True:
            self.transition[state2][state1] = w
        
    def L(self,region):
        return region.ap
    
    def get_weight(self, region1, region2):
        if region2 in self.transition[region1].keys():
            return self.transition[region1][region2]
        else:
            raise ValueError('Cannot transit between %s and %s' %(str(region1.name),str(region2.name)) )
    
    def post(self,region):
        if region not in self.states:
            raise AttributeError('State not valid')
        else:
            return [i for i in self.transition[region].keys()]

class Rabin_Automaton(object):
    
    def __init__(self, ltl, coord_dict):
        self.ltl = ltl
        self.coord_dict = coord_dict

        with open("my.ltl", "w") as ltlfile:
            ltlfile.write(ltl)
            
        if sys.platform == "darwin":
            result1 = os.system("ltlfilt -l -F \"my.ltl\" --output=\"my.ltl\" ")
        else:
            result1 = os.system("./ltlfilt_ubuntu -l -F \"my.ltl\" --output=\"my.ltl\" ")
        assert result1 == 0, "ltlfilt malfunctioning"
        # print result1

        if sys.platform == "darwin":
            result2 = os.system("./ltl2dstar --ltl2nba=spin:ltl2ba --stutter=no --output-format=dot my.ltl my.dot")
        else:
            result2 = os.system("./ltl2dstar_ubuntu --ltl2nba=spin:ltl2ba_ubuntu --stutter=no --output-format=dot my.ltl my.dot")
        assert result2 == 0, "ltl2dstar malfunctioning"
        # print result2

        rabin_graph = nx.nx_agraph.read_dot("my.dot")
        rabin_graph.remove_nodes_from(["comment", "type"])
        
        self.graph = rabin_graph
        self.num_of_nodes = len(self.graph.node)

        self.transition = self.graph
        
        self.accept = [int(i) for i in self.graph.node if "+0" in self.graph.node[i]["label"]]
        self.reject = [int(i) for i in self.graph.node if "-0" in self.graph.node[i]["label"]]

        self.deadlock = []
        self.states = [int(i) for i in self.graph.node]

        for i in self.reject:
            if str(i) in self.graph[str(i)].keys():
                if " true" in [ self.graph[str(i)][str(i)][j]["label"] 
                              for j in range(len(self.graph[str(i)][str(i)])) ]:
                    self.deadlock.append(i)
        
        for i in self.graph.node:
            if "fillcolor" in self.graph.node[i].keys():
                if self.graph.node[i]["fillcolor"] == "grey":
                    self.init_state = int(i)
                    break
        
    def get_graph(self):
        return self.graph
    
    def get_init_state(self):
        return self.init_state
    
    def next_state(self, current_state, next_coord):
        ap_next = self.coord_dict[tuple(next_coord)]
        next_states = self.possible_states(current_state[2])
        for i in next_states:
            next_state_aps = [self.graph[str(current_state[2])][str(i)][k]["label"] 
                              for k in range(len(self.graph[str(current_state[2])][str(i)]))]
            # May need to change later
            if " true" in next_state_aps:
                return current_state[-1]
            else:
                for j in next_state_aps:
                    if self.check_ap(ap_next, j):
                        return int(i)
                
    def possible_states(self, current_rabin_state):
        return [int(i) for i in self.graph[str(current_rabin_state)].keys()]
    
    def check_ap(self, ap_next, ap_sentence):
        pos, neg = seperate_ap_sentence(ap_sentence)
        if set(ap_next).issuperset(set(pos)) and self.check_neg(ap_next, neg):
            return True
        return False
    
    def check_neg(self, ap, negs):
        for i in ap:
            if i in negs:
                return False
        return True

    def post(self, state):
        if state not in self.states:
            raise AttributeError('State not valid')
        else:
            return [int(i) for i in self.graph[str(state)].keys()]

    def get_transition_through_ap(self, state, ap):
        if '&' in ap:
            reachable_state = set()
            seperated_ap,neg_in_ap = seperate_ap_sentence(ap)
            for i in self.transition[str(state)].keys():
                transitions_ap = self.get_ap(state,int(i))
                for j in transitions_ap:
                    seperated_j,neg_in_j = seperate_ap_sentence(j)
                    if '&' in j:
                        if len(seperated_ap) >= len(seperated_j):
                            if set(seperated_ap).issuperset(set(seperated_j)) and check_negations(seperated_j,neg_in_j,seperated_ap,neg_in_ap):
                                reachable_state.add(int(i))
                    else:
                        if '!' in j:
                            if j in ap or check_negations(seperated_j,neg_in_j,seperated_ap,neg_in_ap):
                                reachable_state.add(int(i))
                        else:
                            if j in ap or j=="true":
                                reachable_state.add(int(i))
            return reachable_state
                        
        else:
            reachable_state = set()
            seperated_ap,neg_in_ap = seperate_ap_sentence(ap)
            for i in self.transition[str(state)].keys():
                transitions_ap = self.get_ap(state,int(i))
                for j in transitions_ap:
                    seperated_j,neg_in_j = seperate_ap_sentence(j)
                    if '&' in j:
                        if len(seperated_j) <= 1:
                            if set(seperated_ap).issuperset(set(seperated_j)) and check_negations(seperated_j,neg_in_j,seperated_ap,neg_in_ap):
                                reachable_state.add(int(i))
                    else:
                        if '!' not in j:
                            if j in ap or j=="true":
                                reachable_state.add(int(i))
                        else:
                            if '!' in ap:
                                if j == ap:
                                    reachable_state.add(int(i))
                            else:
                                if check_negations(seperated_j,neg_in_j,seperated_ap,neg_in_ap):
                                    reachable_state.add(int(i))
            return reachable_state

    def get_ap(self, state1, state2):
        if state1 not in self.states or state2 not in self.states:
            raise AttributeError('State not valid')
        elif str(state2) not in self.transition[str(state1)].keys():
            raise AttributeError('State2 cannot be reached from State1')
        else:
            result = [self.transition[str(state1)][str(state2)][i]['label'][1:] for i in range(len(self.transition[str(state1)][str(state2)]))]
            return result
            # if len(result) == 1:
            #     return result
            # else:    
            #     return result[1:-1]

        

# class Buchi_Automaton(object):
#     def __init__(self, buchi):
#         self.states = [i for i in buchi.nodes()]
#         # states are str
#         self.alphabet = [i for i in buchi.graph['symbols']]
#         self.transition = buchi
#         self.initial = [i for i in buchi.graph['initial']]
#         self.accept = [i for i in buchi.graph['accept']]
#         self.buchi = buchi
        
#     def get_ap(self, state1, state2):
#         if state1 not in self.states or state2 not in self.states:
#             raise AttributeError('State not valid')
#         elif state2 not in self.transition[state1].keys():
#             raise AttributeError('State2 cannot be reached from State1')
#         else:
#             result = self.transition[state1][state2]['guard_formula']
#             if len(result) == 1:
#                 return result
#             else:    
#                 return result[1:-1]
        
#     def post(self, state):
#         if state not in self.states:
#             raise AttributeError('State not valid')
#         else:
#             return [i for i in self.transition[state].keys()]
        
#     def get_transition_through_ap(self, state, ap):
#         if '&&' in ap:
#             reachable_state = []
#             seperated_ap,neg_in_ap = self.seperate_ap_sentence(ap)
#             for i in self.transition[state].keys():
#                 j = self.get_ap(state,i)
#                 seperated_j,neg_in_j = self.seperate_ap_sentence(j)
#                 if '&&' in j:
#                     if len(seperated_ap) >= len(seperated_j):
#                         if set(seperated_ap).issuperset(set(seperated_j)) and self.check_negations(seperated_j,neg_in_j,seperated_ap,neg_in_ap):
#                             reachable_state += [i]
#                 else:
#                     if '!' in j:
#                         if j in ap or self.check_negations(seperated_j,neg_in_j,seperated_ap,neg_in_ap):
#                             reachable_state += [i]
#                     else:
#                         if j in ap or j=='1':
#                             reachable_state += [i]
#             return reachable_state
                        
#         else:
#             reachable_state = []
#             seperated_ap,neg_in_ap = self.seperate_ap_sentence(ap)
#             for i in self.transition[state].keys():
#                 j = self.get_ap(state,i)
#                 seperated_j,neg_in_j = self.seperate_ap_sentence(j)
#                 if '&&' in j:
#                     if len(seperated_j) <= 1:
#                         if set(seperated_ap).issuperset(set(seperated_j)) and self.check_negations(seperated_j,neg_in_j,seperated_ap,neg_in_ap):
#                             reachable_state += [i] 
#                 else:
#                     if '!' not in j:
#                         if j in ap or j=='1':
#                             reachable_state += [i]
#                     else:
#                         if '!' in ap:
#                             if j == ap:
#                                 reachable_state += [i]
#                         else:
#                             if self.check_negations(seperated_j,neg_in_j,seperated_ap,neg_in_ap):
#                                 reachable_state += [i]
#             return reachable_state


#     def plot(self,filename='current_buchi'):
#         plot(self.buchi,filename)
        
    
        
class FullProd(object):
    def __init__(self, wFTS, Rabin):
        self.states = set()
        self.transition = {}
        # transition is a dict where every state stores as a key and value is a dic contains every reachable
        # state from this key and the weight to transit
        self.initial = set()
        self.accept = set()
        self.wFTS = wFTS
        self.Rabin = Rabin
        
    def construct_fullproduct(self):
        alpha = 1
        for pi_i in self.wFTS.states:
            for q_m in self.Rabin.states:
                q_s = (pi_i,q_m)
                self.states.add(q_s)
                if q_s not in self.transition.keys():
                    self.transition[q_s] = {}
                if pi_i in self.wFTS.initial and q_m == self.Rabin.init_state:
                    self.initial.add(q_s)
                if q_m in self.Rabin.accept:
                    self.accept.add(q_s)
                    
                for pi_j in self.wFTS.post(pi_i):
                    for q_n in self.Rabin.post(q_m):
                        q_g = (pi_j,q_n)
                        self.states.add(q_g)
                        if q_g not in self.transition.keys():
                            self.transition[q_g] = {}
                        d = self.check_tran_b(q_m,self.wFTS.L(pi_i),q_n,self.Rabin)
                        if d >= 0:
                            self.transition[q_s][q_g] = self.wFTS.get_weight(pi_i,pi_j) + alpha*d
                            
    def get_weight(self,state1,state2):
        return self.transition[state1][state2]

                        
    def check_tran_b(self, b_state1, l, b_state2, Rabin):
        d = -1
        if len(l) == 1:
            if b_state2 in self.Rabin.get_transition_through_ap(b_state1, l[0]):
                d = 0
                return d
        if len(l) > 1:
            conjunction = [l[i]+'&' if i!=len(l)-1 else l[i] for i in range(len(l))]
            # The line above was appeared as follows, the change of "&&" into "&" needs more verification.
            # conjunction = [l[i]+'&&' if i!=len(l)-1 else l[i] for i in range(len(l))]
            conjunction = ''.join(conjunction)
            if b_state2 in self.Rabin.get_transition_through_ap(b_state1, conjunction):
                d = 0
                return d
        return d
    
    def get_state(self,index):
        return list(self.states)[index]
    
    def return_graph(self):
        index = 0
        digraph = DiGraph()
        digraph.add_nodes_from([i for i in self.states])
        for i in self.states:
            for j in self.transition[i].keys():
                if self.transition[i][j] is not None:
                    digraph.add_edge(i,j,weight=self.transition[i][j])
        return digraph
                        

def search_opt_run(FullProduct):
    candidates_pre = {}
    candidates_suf = {}
    candidates = {}
    G = FullProduct.return_graph()
    for initial_state in FullProduct.initial:
        for accept_state in FullProduct.accept:
            try:
                opt_path = nx.dijkstra_path(G,initial_state,accept_state,'weight')
                path_cost = nx.dijkstra_path_length(G,initial_state,accept_state,'weight')
                if initial_state not in candidates_pre.keys():
                    candidates_pre[initial_state] = {}
                candidates_pre[initial_state][accept_state] = (opt_path,path_cost)
            except:
                pass
#     print candidates_pre
    for accept_state in FullProduct.accept:
        successors = FullProduct.transition[accept_state].keys()
        best_path = []
        best_cost = float('inf')
        for successor in successors:
            if successor is not accept_state:
                try:
                    current_path = nx.dijkstra_path(G,accept_state,successor,'weight') + nx.dijkstra_path(G,successor,accept_state,'weight')
                    current_cost = nx.dijkstra_path_length(G,accept_state,successor,'weight') + nx.dijkstra_path_length(G,successor,accept_state,'weight')
                    if current_cost < best_cost:
                        best_path = current_path
                        best_cost = current_cost
                except:
                    pass
            else:
                current_path = [(accept_state,accept_state)]
                current_cost = FullProduct.transition[accept_state][accept_state]
                if current_cost < best_cost:
                    best_path = current_path
                    best_cost = current_cost
        if best_cost < float('inf'):
            candidates_suf[accept_state] = (best_path,best_cost)
#     print candidates_suf
    for initial_state in candidates_pre.keys():
        for accept_state in candidates_pre[initial_state].keys():
            if accept_state in candidates_suf.keys():
                candidates[(initial_state,accept_state)] = (candidates_pre[initial_state][accept_state][0]+candidates_suf[accept_state][0],
                                                           candidates_pre[initial_state][accept_state][1]+candidates_suf[accept_state][1])
    opt_run = min(candidates.items(), key=lambda x : x[1][1])
    return opt_run[1]

