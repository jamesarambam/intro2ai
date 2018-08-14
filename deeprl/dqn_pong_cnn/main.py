"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 14 Aug 2018
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import sys
import os
from pprint import pprint
import time
import pdb
import rlcompleter

import gym
from model import DQN_CNN
from utils import preprocess_observation, get_next_state
import numpy as np
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #
EPISODES = 2
MAX_STEPS = 10
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 32


# ============================================================================ #

def main():

    env = gym.make('PongDeterministic-v4')
    network_input_shape = (4, 210, 160)  # Dimension ordering: 'th' (channels first)
    actions = env.action_space.n

    agent = DQN_CNN(actions, network_input_shape, learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR)
    frame_counter = 0

    for ep in range(1, EPISODES+1):
        # Start episode
        score = 0
        # Observe reward and initialize first state
        obs = preprocess_observation(env.reset())

        # Initialize the first state with the same 4 images
        current_state = np.array([obs, obs, obs, obs])
        frame_counter += 1
        for t in range(MAX_STEPS):
            # Select an action using the DQA
            action = agent.act(np.asarray([current_state]))
            pdb.set_trace()
            # Observe reward and next state
            obs, reward, done, info = env.step(action)
            obs = preprocess_observation(obs)
            next_state = get_next_state(current_state, obs)
            frame_counter += 1
            # Store transition in replay memory
            clipped_reward = np.clip(reward, -1, 1)  # Clip the reward
            agent.add_experience(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)





# =============================================================================== #

if __name__ == '__main__':
    main()
    