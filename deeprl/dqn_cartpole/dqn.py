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
MAX_STEPS = 100 #3000
BATCH_SIZE = 32
ep = []
score = []


# ============================================================================ #

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
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
        return np.argmax(act_values[0])  # returns action

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
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n


    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    ep_reward = 0
    for e in range(1, MAX_EPISODES+1):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(MAX_STEPS):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done else -10
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, MAX_EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)
        log(e, ep_reward)
        ep_reward = 0
        # plot()


# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"