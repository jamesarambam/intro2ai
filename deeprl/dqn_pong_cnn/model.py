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

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
import numpy as np
from random import random
import random
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #


# ============================================================================ #

class DQN_CNN:

    def __init__(self, actions, input_shape,
                 minibatch_size=32,
                 learning_rate=1e-3,
                 discount_factor=0.99,
                 load_path=None):

        # Parameters
        self.actions = actions  # Size of the network output
        self.discount_factor = discount_factor  # Discount factor of the MDP
        self.minibatch_size = minibatch_size  # Size of the training batches
        self.learning_rate = learning_rate  # Learning rate
        self.input_shape = input_shape
        self.load_path = load_path
        self.epsilon = 1,
        self.epsilon_decrease_rate = 0.99,
        self.min_epsilon = 0.1,
        self.model = self._build_model()

    def _build_model(self):

        model = Sequential()

        input_shape = self.input_shape
        # First convolutional layer
        model.add(Conv2D(32, 8, strides=(4, 4),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))

        # Second convolutional layer
        model.add(Conv2D(64, 4, strides=(2, 2),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))

        # Third convolutional layer
        model.add(Conv2D(64, 3, strides=(1, 1),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))

        # Flatten the convolution output
        model.add(Flatten())

        # First dense layer
        model.add(Dense(512, activation='relu'))

        # Output layer
        model.add(Dense(self.actions))
        if self.load_path is not None:
            self.load(self.load_path)

        model.compile(loss='mean_squared_error',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        return model


    def act(self, state):

            q_values = self.model.predict(state)
            return np.argmax(q_values)

        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.actions)
        # else:
        #     q_values = self.model.predict(state)
        #     return np.argmax(q_values)

    def add_experience(self, source, action, reward, dest, final):
        """
        Add a SARS' tuple to the experience replay.
        :param source: source state
        :param action: action index
        :param reward: reward associated to the transition
        :param dest: destination state
        :param final: whether the state is absorbing
        """
        # Remove older transitions if the replay memory is full
        if len(self.experiences) >= self.replay_memory_size:
            self.experiences.pop(0)
        # Add a tuple (source, action, reward, dest, final) to replay memory
        self.experiences.append({'source': source,
                                 'action': action,
                                 'reward': reward,
                                 'dest': dest,
                                 'final': final})




def main():
    print "Hello World"


# =============================================================================== #

if __name__ == '__main__':
    main()
    