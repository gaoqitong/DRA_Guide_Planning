from full_prod_DRA import *
from buchi import buchi_from_ltl
# from Visualize import plot_buchi
import numpy as np
# from env_sensing_error import CurrentWorld
import scipy
from plot_path_for_prod import *
from graphviz import Source
# from Plot_Path import *

def convert_path_to_action(path):
    current_coord = path[0]
    next_coord = path[1]
    if current_coord[0] == next_coord[0] and current_coord[1]-1 == next_coord[1]:
        return 0
    if current_coord[0]+1 == next_coord[0] and current_coord[1] == next_coord[1]:
        return 1
    if current_coord[0] == next_coord[0] and current_coord[1]+1 == next_coord[1]:
        return 2
    if current_coord[0]-1 == next_coord[0] and current_coord[1] == next_coord[1]:
        return 3
    if current_coord[0] == next_coord[0] and current_coord[1] == next_coord[1]:
        return 4


class Prod_Planning(object):
    def __init__(self, env, ltl):
        self.ltl = ltl.lower()
        self.env = env
        
        region_list = ["r"+str(i) for i in range(100)]
        for i in range(len(region_list)):
            coord = np.unravel_index(i, (env.shape[0],env.shape[1]))
            if len(env.dynamic_coord_dict[coord]) >0:
                region_list[i] = Region(coord, [ap.lower() for ap in env.dynamic_coord_dict[coord]], region_list[i])
            else:
                region_list[i] = Region(coord, [], region_list[i])
                
        wfts = wFTS(set(), {}, set())
        for i in region_list:
            wfts.add_states(i)

        for i in region_list:
            current_coord = list(i.coord)
            candidates = [np.array(current_coord), np.add(current_coord,[0,1]), np.add(current_coord,[0,-1]), 
                          np.add(current_coord,[1,0]), np.add(current_coord,[-1,0])]
            candidates = [np.ravel_multi_index(c, env.shape[:-1]) for c in candidates if c[0]>=0 and c[1]>=0 and c[0]<env.shape[0] and c[1]<env.shape[1]]
            for c in candidates:
                wfts.add_transition(i, region_list[c], 1)

        wfts.add_initial(region_list[np.ravel_multi_index(env.start_coord, env.shape[:-1])])
        
        self.region_list = region_list
        self.wfts = wfts
        self.opt_path = []

    def replace_region_list(self):
        env = self.env
        region_list = ["r"+str(i) for i in range(100)]
        for i in range(len(region_list)):
            coord = np.unravel_index(i, (env.shape[0],env.shape[1]))
            if len(env.dynamic_coord_dict[coord]) >0:
                region_list[i] = Region(coord, [ap.lower() for ap in env.dynamic_coord_dict[coord]], region_list[i])
            else:
                region_list[i] = Region(coord, [], region_list[i])
                
        wfts = wFTS(set(), {}, set())
        for i in region_list:
            wfts.add_states(i)

        for i in region_list:
            current_coord = list(i.coord)
            candidates = [np.array(current_coord), np.add(current_coord,[0,1]), np.add(current_coord,[0,-1]), 
                          np.add(current_coord,[1,0]), np.add(current_coord,[-1,0])]
            candidates = [np.ravel_multi_index(c, env.shape[:-1]) for c in candidates if c[0]>=0 and c[1]>=0 and c[0]<env.shape[0] and c[1]<env.shape[1]]
            for c in candidates:
                wfts.add_transition(i, region_list[c], 1)

        wfts.add_initial(region_list[np.ravel_multi_index(env.start_coord, env.shape[:-1])])
        self.region_list = region_list
        
    def update_wfts_ap(self):
        last_coord_dict_extra = set((k,tuple(v)) for k,v in self.env.last_dynamic_coord_dict.items()) - set((k,tuple(v)) for k,v in self.env.dynamic_coord_dict.items())
        new_coord_dict_extra = set((k,tuple(v)) for k,v in self.env.dynamic_coord_dict.items()) - set((k,tuple(v)) for k,v in self.env.last_dynamic_coord_dict.items())
#         print "last_extra: ", last_coord_dict_extra
#         print "new_extra: ", new_coord_dict_extra
        for k,v in last_coord_dict_extra:
            # print "k",k
            for i in v:
                # print "i",i
                self.region_list[np.ravel_multi_index(k, (10,10))].app.remove(i.lower())
                self.region_list[np.ravel_multi_index(k, (10,10))].ap.remove(i.lower())

        for k,v in new_coord_dict_extra:
            for i in v:
                self.region_list[np.ravel_multi_index(k, (10,10))].app += [i.lower()]
                self.region_list[np.ravel_multi_index(k, (10,10))].ap += [i.lower()]
    
    def get_global_opt(self):
        opt_path = []

#         self.region_list = self.update_wfts_ap()
        self.wfts.replace_initial(self.region_list[np.ravel_multi_index(self.env.start_coord, self.env.shape[:-1])])
        
        rabin = self.env.rabin
        
        full_prod = FullProd(self.wfts, rabin)
        self.full_prod = full_prod
        full_prod.construct_fullproduct()
        # count = 0
        # for i in full_prod.states:
        #     for j in full_prod.transition[i].keys():
        #         if full_prod.transition[i][j] is not None:
        #             count += 1 
                    
        # while len(opt_path) == 0:
        try:
            opt=search_opt_run(full_prod)
            opt_path = [tuple(list(opt[0][i][0].coord) + [opt[0][i][1]]) for i in range(len(opt[0]))]
            self.opt_path = opt_path
        except:
            pass
        
        # print 'Plan synthesized:'+str(opt_path)
#         return self.opt_path, self.region_list, self.wfts

    def get_opt_rabin(self):
        opt_rabin = []
        for i in self.opt_path:
            if len(opt_rabin) == 0:
                opt_rabin += [i[-1]]
            if i[-1] != opt_rabin[-1]:
                opt_rabin += [i[-1]]
#         return opt_rabin
        self.opt_rabin = opt_rabin
    
    def get_next_ltl(self, current_rabin):
        # input 'current_rabin' is with type int
        current_rabin = str(current_rabin)
        current_idx = np.where(np.array(self.opt_rabin) == np.int(current_rabin))[0][0]
        next_rabin = str(self.opt_rabin[current_idx+1])
        ltl_list = [i["label"][1:] for i in self.env.rabin.graph[current_rabin][next_rabin].values()]
        next_ltl = ")||<>(".join(ltl_list)
        next_ltl = "<>(" + next_ltl + ")"
        return next_ltl
    
    def get_local_opt(self, new_start_coord, new_ltl):
        opt_local_path = []

        rabin = Rabin_Automaton(new_ltl, self.env.dynamic_coord_dict)
        self.wfts.replace_initial(self.region_list[np.ravel_multi_index(new_start_coord, self.env.shape[:-1])])
        
        full_prod = FullProd(self.wfts, rabin)
        full_prod.construct_fullproduct()
        # count = 0
        # for i in full_prod.states:
        #     for j in full_prod.transition[i].keys():
        #         if full_prod.transition[i][j] is not None:
        #             count += 1 
                    
        # while len(opt_local_path) == 0:
        try:
            opt=search_opt_run(full_prod)
            opt_local_path = [tuple(list(opt[0][i][0].coord) + [opt[0][i][1]]) for i in range(len(opt[0]))]
            return opt_local_path
        except:
            return None
        # print 'Local Plan synthesized:'+str(opt_local_path)
        
    