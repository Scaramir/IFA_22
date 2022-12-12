'''
Group 8: Florian Herzler, Dominik Bannwitz, Maximilian Otto
Homework 2
'''

from numpy import dtype
import numpy as np
from tqdm import tqdm
input = np.loadtxt("Input.txt")
np.random.seed(seed=int(input[0]))
nrSimulations = int(input[1])

time_max = 10.0
lamdba = 1e-4
delta = 1e-8
beta = 5e-5
k_r = 0.3

x_initial_state = np.array([[lamdba/delta],
                            [20],
                            [0],
                            [0]])

#   r1  r2  r3  r4  r5  r6
#X1  1  -1  -1   0   0   0
#X2  0   0   1  -1  -1   0
#X3  0   0   0   0   1  -1
#X4  0   0   0   1   0   0
stochiometry_matrix = np.array([[1,-1,-1, 0, 0, 0],
                         [0, 0, 1,-1,-1, 0],
                         [0, 0, 0, 0, 1,-1],
                         [0, 0, 0, 1, 0, 0]])

def propensities(x_states,lamdba,delta,beta,k_r):
    prop_vec = [lamdba,
                x_states[0] * delta,
                x_states[0] * x_states[1] * beta,
                x_states[1] * 3e7 * delta,
                x_states[1] * k_r,
                x_states[2] * delta]
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

def SSA(stochiometry_matrix, x_initial_state, time_max, lamdba, delta, beta, k_r):
    # keep track of the states and times
    x0_list = []
    x1_list = []
    x2_list = []
    x3_list = []
    timestep_list = []

    # initialize states and time
    current_time = 0.0
    x = x_initial_state
    x0_list.append(x[0,0])
    x1_list.append(x[1,0])
    x2_list.append(x[2,0])
    x3_list.append(x[3,0])
    timestep_list.append(current_time)

    while current_time < time_max:
        # calculate reaction propensities
        changes = propensities(x,lamdba,delta,beta,k_r)
        # add time until next reaction
        time_skip = time_to_next_reaction(np.sum(changes))
        # termination condition
        if (current_time + time_skip > time_max) or (np.sum(changes) == 0):
            return np.array(x0_list), np.array(x1_list), np.array(x2_list), np.array(x3_list), np.array(timestep_list)

        current_time = current_time + time_skip
        # update model 
        j = find_reaction_index(changes)
        x = x + stochiometry_matrix[:,[j]]
        # keep track of states and time points
        x0_list.append(x[0,0])
        x1_list.append(x[1,0])
        x2_list.append(x[2,0])
        x3_list.append(x[3,0])
        timestep_list.append(current_time)

#to save trajectories
#Task 1a:
for i in tqdm(range(nrSimulations), desc='Simulations'):
    states0, states1, states2, states3, times = SSA(stochiometry_matrix, x_initial_state, time_max, lamdba, delta, beta, k_r)
    Output = np.concatenate((np.array(times, ndmin=2),np.array(states1, ndmin=2)), axis=0)
    np.savetxt('Task1Traj'+str(i+1)+'.txt', Output, delimiter = ',', fmt='%1.3f')
