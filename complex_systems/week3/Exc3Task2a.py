'''
Group 8: Florian Herzler, Dominik Bannwitz, Maximilian Otto
Homework 3
'''

import numpy as np

input = np.loadtxt("Input2.txt")
np.random.seed(seed=int(input))

# stoichiometrix matrix
# -1   0
#  1  -1
stoichiometry_matrix = np.array([[-1, 0], [1, -1]])

k_a = 0.5
k_e = 0.3

delta_t = 0.1
step_size_of_int = 1.
t_final = 24.

initial_x_states = np.array([[200], [0]])


def propensities(x_states):
    '''
    Calculates the propensities of the reactions
    :param : x_states: the current state of the system
    :return : a list of propensities
    '''
    r_0 = x_states[0] * k_a
    r_1 = x_states[1] * k_e

    return [r_0, r_1]

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

def SSA(stochiometry_matrix, x_initial_state, time_max):
    # keep track of the states and times
    x0_list = []
    x1_list = []
    timestep_list = []

    # initialize states and time
    current_time = 0.0
    x = x_initial_state
    x0_list.append(x[0,0])
    x1_list.append(x[1,0])
    timestep_list.append(current_time)

    while current_time < time_max:
        # calculate reaction propensities
        changes = propensities(x)
        # add time until next reaction
        # time_skip = 1.0 #time_to_next_reaction(np.sum(changes))
        time_skip = time_to_next_reaction(np.sum(changes))
        # termination condition
        if (current_time + time_skip > time_max) or (np.sum(changes) == 0):
            return np.array(x0_list), np.array(x1_list), np.array(timestep_list)

        current_time = current_time + time_skip
        # update model 
        j = find_reaction_index(changes)
        x = x + stochiometry_matrix[:,[j]]
        # keep track of states and time points
        x0_list.append(x[0,0])
        x1_list.append(x[1,0])
        timestep_list.append(current_time)

#Task 2a
states0, states1, times = SSA(stoichiometry_matrix.copy(), initial_x_states.copy(), t_final)
# save only the states at integer time points
#   get integer time points
time_points = np.arange(0, t_final+1, step_size_of_int)
#  get the indices of the integer time points
time_indices = np.searchsorted(times, time_points)
#  get the states at the integer time points
states1 = states1[time_indices-1]
#  save the states at the integer time points
Output = np.concatenate((np.array(time_points, ndmin=2),np.array(states1, ndmin=2)), axis=0)
np.savetxt('Task2aTraj.txt', Output, delimiter = ',', fmt='%1.2f')





















