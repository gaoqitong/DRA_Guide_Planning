from full_prod_DRA import *
import numpy as np
# from env_dynamic_ap import CurrentWorld
import scipy

def gen_dra_policy(ltl, env):

    ltl = ltl.lower()

    region_list = ["r"+str(i) for i in range(env.shape[0] * env.shape[1])]
    for i in range(len(region_list)):
        coord = np.unravel_index(i, env.shape[:-1])
        if len(env.coord_dict[coord]) >0:
            region_list[i] = Region(coord, [ap.lower() for ap in env.coord_dict[coord]], region_list[i])
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
        

    # buchi = buchi_from_ltl(ltl,None)
    # my_buchi = Buchi_Automaton(buchi)

    rabin = Rabin_Automaton(ltl, env.coord_dict)

    full_prod = FullProd(wfts, rabin)
    full_prod.construct_fullproduct()
    count = 0
    for i in full_prod.states:
        for j in full_prod.transition[i].keys():
            if full_prod.transition[i][j] is not None:
                count += 1 

    opt=search_opt_run(full_prod)
    opt_path = [tuple(list(opt[0][i][0].coord) + [opt[0][i][1]]) for i in range(len(opt[0]))]
    # print 'Plan synthesized:'+str(opt_path)
    
    # Generate Policy from Opt Path

    dra_policy = {}
    for i in opt_path:
        # blocks in down/left/up/right position
        dra_policy[tuple(np.add(list(i), [1,0,0]))] = 0
        dra_policy[tuple(np.add(list(i), [0,-1,0]))] = 1
        dra_policy[tuple(np.add(list(i), [-1,0,0]))] = 2
        dra_policy[tuple(np.add(list(i), [0,1,0]))] = 3
        # blocks in down-right/down-left/up-left/up-right position
        dra_policy[tuple(np.add(list(i), [1,1,0]))] = 0
        dra_policy[tuple(np.add(list(i), [1,-1,0]))] = 1
        dra_policy[tuple(np.add(list(i), [-1,-1,0]))] = 2
        dra_policy[tuple(np.add(list(i), [-1,1,0]))] = 3

    return dra_policy
