from __future__ import division
import time
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from itertools import count


def plot_path_for_prod(env, LTL, optimal_path):
    # env = CurrentWorld(LTL)

    state_dim = 3
    action_dim = env.nA

    state = env.reset()
    done = False

    for t in range(len(optimal_path)):
        current_state = optimal_path[t][:-1]
        render_for_prod(env, current_state)
        
        if t%20 == 0:
            plt.close("all")
        if done:
            break

def render_for_prod(env, state):
    world = np.zeros((env.shape[0], env.shape[1]))
    color_dict = {ap: color+1 for color, ap in enumerate(env.ap_dict.keys())}
    for i in env.coord_dict.keys():
        if len(env.coord_dict[i]) >=1:
            world[i] = color_dict[env.coord_dict[i][0]]
    world[state] = len(env.ap_dict) + 1
    fig, ax = plt.subplots()
    ax.clear()
    ax.imshow(world)
    for i in env.ap_dict.keys():
        for j in env.ap_dict[i]:
            ax.annotate(i, xy=(j[1] - 0.13, j[0] + 0.13), fontsize=20, color=(1,1,1))
    ax.annotate("R", xy=(state[1] - 0.13, state[0] + 0.13), fontsize=20, color=(1,0,0))
    plt.pause(0.0001)