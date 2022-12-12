'''
Group 8: Florian Herzler, Dominik Bannwitz, Maximilian Otto
Homework 2
'''

from numpy import dtype
import numpy as np
from tqdm import tqdm
#import matplotlib.pyplot as plt
input = np.loadtxt('Input.txt')
np.random.seed(seed=int(input[0]))
nrSimulations = int(input[1])

time_max = 5.0
x_initial_state = np.array([[40.0]])

#   r1  r2  r3  r4
#X1  1  -1   1  -1
stochiometry = np.array([[1,-1,1,-1]])

def propensities(x_states):
    prop_vec = [0.15 * x_states[0] * (x_states[0] - 1),
                0.0015 * x_states[0] * (x_states[0] - 1) * (x_states[0] - 2),
                20.0,
                3.5 * x_states[0]]
    return prop_vec

def time_to_next_reaction(lamdba):
    """
    Exp distribution with mean 1/lamdba. `r` is random number between 0 and 1 and != 0.
    input: lamdba : real value positive.
    """
    r = np.random.rand()
    while r==0:
        r = np.random.rand()
    return (1.0/lamdba)*np.log(1.0/r)

def find_reaction_index(changes):
    """
    Propensitiy / reaction vector `changes`. `r` is random number between 0 and 1 and != 0.
    input: changes : Array (num_reaction,1)
    """
    r = np.random.rand()
    while r == 0:
        r = np.random.rand()
    return np.sum(np.cumsum(changes) < r*np.sum(changes))

def SSA(stochiometry, x_initial_state, time_max):
    # keep track of the states and times
    x_list = []
    timestep_list = []

    # initialize states and time
    current_time = 0.0
    x = x_initial_state
    x_list.append(x)
    timestep_list.append(current_time)

    while current_time < time_max:
        # calculate reaction propensities
        changes = propensities(x)
        # add time until next reaction
        time_skip = time_to_next_reaction(np.sum(changes))
        # termination condition
        if (current_time + time_skip > time_max) or (np.sum(changes) == 0):
            return np.array(x_list), np.array(timestep_list)
        
        current_time = current_time + time_skip
        j = find_reaction_index(changes)
        x = x + stochiometry[:,[j]]
        # keep track of states and time points
        x_list.append(x[0,0])
        timestep_list.append(current_time)

#to save trajectories
#Task 2a:
for i in tqdm(range(nrSimulations), desc='Simulating trajectories'):
    states, times = SSA(stochiometry, x_initial_state, time_max)
    Output = np.concatenate((np.array(times, ndmin=2),np.array(states, ndmin=2)), axis=0)
    np.savetxt('Task2Traj'+str(i+1)+'.txt', Output, delimiter = ',', fmt='%1.3f')
