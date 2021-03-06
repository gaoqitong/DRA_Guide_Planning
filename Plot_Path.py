from __future__ import division
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import *
from qnetwork import *
from itertools import count


def plot_path(env, LTL, saved_path, learning_rate, tau, minibatch_size, save_dir, device):
    # env = CurrentWorld(LTL)

    state_dim = 3
    action_dim = env.nA

    state = env.reset()
    done = False

    with tf.Session() as sess:
        Qnet = QNet(sess, state_dim, action_dim, learning_rate, tau, minibatch_size, save_dir, device)
        saver = tf.train.Saver()
        saver.restore(sess, saved_path)
        for t in count():
            state = np.reshape(list(np.unravel_index(state, env.shape)), (1, state_dim))
            state_for_plot = tuple(state[0][:2])
            action = Qnet.predict_a_from_save(state, saved_path)
            next_state,_,done,_ = env.step(action[0])
            render(env, state_for_plot, action[0])
            state = next_state
            if t%20 == 0:
                plt.close("all")
            if done:
                break

def render(env, state, action):
    action_dict = {0:"UP", 1:"RIGHT", 2:"DOWN", 3:"LEFT", 4: "NONE"}
    world = np.zeros((env.shape[0], env.shape[1]))
    color_dict = {ap: color+1 for color, ap in enumerate(env.ap_dict.keys())}
    for i in env.static_coord_dict.keys():
        if len(env.static_coord_dict[i]) >=1:
            world[i] = color_dict[env.static_coord_dict[i][0]]
    world[state] = len(env.ap_dict) + 1
    fig, ax = plt.subplots()
    ax.clear()
    ax.imshow(world)
    for i in env.ap_dict.keys():
        for j in env.ap_dict[i]:
            ax.annotate(i, xy=(j[1] - 0.13, j[0] + 0.13), fontsize=20, color=(1,1,1))
    ax.annotate("R", xy=(state[1] - 0.13, state[0] + 0.13), fontsize=20, color=(1,0,0))
    plt.title("ACTION = " + action_dict[action])
    plt.pause(0.0001)