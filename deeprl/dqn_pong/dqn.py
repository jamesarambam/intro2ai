"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 11 Aug 2018
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
print "# ============================ START ============================ #"
# ================================ Imports ================================ #
import sys
import os
from pprint import pprint
import time
import pdb
import rlcompleter

import argparse

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.ion()
from utils import prepro, prepro_curr, prepro_next


# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
# =============================== Variables ================================== #

MAX_EPISODES = 10000
MAX_STEPS = 500 #3000
BATCH_SIZE = 2
ep = []
score = []
action_pong = {0 : 2, 1 : 3}

# ============================================================================ #

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=BATCH_SIZE*MAX_STEPS)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def plot():

    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.plot(ep, score)
    plt.legend()
    plt.draw()
    plt.pause(1)

def log(x, y):

    ep.append(x)
    score.append(y)
    with open("./log/log.txt", 'a') as f:
        f.writelines(str(x)+","+str(y)+"\n")

def init():

    os.system("mkdir ./log/")
    os.system("rm ./log/log.txt")

def main():

    init()
    env = gym.make('Pong-v4')
    state_size = 6400 # 80 * 80 # input dimensionality: 80x80 grid
    action_size = 2 # up and down

    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    ep_reward = 0
    for e in range(1, MAX_EPISODES+1):
        observation = env.reset()
        for time in range(MAX_STEPS):

            # env.render()
            state = prepro_curr(observation, state_size)
            aid = agent.act(state)
            action = action_pong[aid]
            next_observation, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = prepro_next(next_observation, state_size)
            agent.remember(state, aid, reward, next_state, done)
            observation = next_observation
            ep_reward += reward
        #if len(agent.memory) > BATCH_SIZE:
        if e % BATCH_SIZE == 0:
            agent.replay(BATCH_SIZE)
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, MAX_EPISODES, ep_reward, agent.epsilon))
        log(e, ep_reward)
        ep_reward = 0
        # plot()


# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"