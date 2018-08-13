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
import numpy as np

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
# =============================== Variables ================================== #
prev_x_next = None
prev_x_curr = None
# ============================================================================ #

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()


def prepro_curr(I, ipDim):

    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    global prev_x_curr
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    cur_x = I.astype(np.float32).ravel()
    x = cur_x - prev_x_curr if prev_x_curr is not None else np.zeros(ipDim, dtype=np.float32)
    prev_x_curr = cur_x
    x = x.reshape(1, ipDim)
    return x

def prepro_next(I, ipDim):

    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    global prev_x_next
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    cur_x = I.astype(np.float32).ravel()
    x = cur_x - prev_x_next if prev_x_next is not None else np.zeros(ipDim, dtype=np.float32)
    prev_x_next = cur_x
    x = x.reshape(1, ipDim)
    return x


def main():
    print "Hello World"


# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"