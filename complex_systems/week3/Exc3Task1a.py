'''
Group 8: Florian Herzler, Dominik Bannwitz, Maximilian Otto
Homework 3
'''

import numpy as np
from tqdm import tqdm
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sns

# initial state of the system
inital_x_states = np.loadtxt('Input1.txt')
x_states = inital_x_states

# stoichiometrix matrix
# 1 -1 -1  0
# 0  0  1 -1
stoichiometry_matrix = np.array([[1, -1, -1, 0], [0, 0, 1, -1]])

lamdba = 0.3
k_1 = 0.01 
k_2 = 0.01
delta = 0.3

delta_t = 0.1
t_final = 10


def propensities(x_states):
    '''
    Calculates the propensities of the reactions
    :param : x_states: the current state of the system
    :return : a list of propensities
    '''
    r_1 = x_states[0] * lamdba 
    r_2 = x_states[0] * x_states[1] * k_1 
    r_3 = x_states[0] * x_states[1] * k_2 
    r_4 = x_states[1] * delta
    return [r_1, r_2, r_3, r_4] 


# right hand side for the explicit euler method
def rhs(_, x):
    return np.dot(stoichiometry_matrix, propensities(x))

# task 1a) explicit euler method
# Calculate the state of the system with the explicit euler method
def explicit_euler(x_states, delta_t, t_final):
    '''
    Calculates the state of the system with the explicit euler method
    :param : x_states: the current state of the system
    :param : delta_t: the time step
    :param : t_final: the final time
    :return : a list of the state of the system and a list of every time point
    '''
    all_time_points = []
    all_x2_states = []
    # keep the initial values
    all_time_points.append(0)
    all_x2_states.append(x_states[1])
    # compute the state of the system for every next time step
    for current_time in tqdm(np.arange(delta_t, t_final+delta_t, delta_t), desc='Simulating with explicit euler method'):
        x_states += rhs(delta_t, x_states) * delta_t
        all_time_points.append(current_time)
        all_x2_states.append(x_states[1])
    return all_x2_states, all_time_points


# save the results in a txt file
def save_results(all_x2_states, all_time_points):
    Output = np.concatenate((np.array(all_time_points, ndmin=2),np.array(all_x2_states, ndmin=2)), axis=0)
    np.savetxt('Task1aTraj.txt', Output, delimiter = ',', fmt='%1.2f')
    return


# Task 1b)
def calculate_error(all_x2_states_euler, all_x2_states_rk45):
    ''' NOTE: arrays must have the same length'''
    # error_delta_t = 1 / n (sum from i = 0 to n of |x_2_euler(i) - x_2_rk45(i)|)
    # 'cause of numpy arrays we can use the abs function and substract the arrays from each other to obtain each pairwise difference
    return np.mean(np.abs(np.array(all_x2_states_euler) - np.array(all_x2_states_rk45)))

# FIXME: wrong results, maybe the prof did a mistake!
# NOTE: for N=100 and delta_t=0.01, we want to obtain 10001 time points. states: (x_0, x_1, ..., x_10000)
def calculateErrorEulerAndRK45(x_states, delta_t, t_final=100):
    x_initial_states = x_states
    # Call Euler
    all_x2_states_euler, all_time_points_euler = explicit_euler(x_initial_states.copy(), delta_t, t_final)
    all_x2_states_rk45 = integrate.solve_ivp(rhs, (0, t_final+delta_t), x_initial_states.copy(), method='RK45', t_eval=all_time_points_euler).y[1]
    print(len(all_x2_states_euler), len(all_x2_states_rk45))
    print("Amount of time points: Euler:", len(all_x2_states_euler), " RK45:", len(all_x2_states_rk45))
    error = calculate_error(all_x2_states_euler, all_x2_states_rk45)
    return error


def exercise1b(x_states, t_final=100):
    stepsizes = [0.01, 0.02, 0.05, 0.1]
    errors = []
    initial_x_states = x_states
    for step_size in tqdm(stepsizes, desc='Calculating errors for different stepsizes'):
        x2_error = calculateErrorEulerAndRK45(initial_x_states, step_size, t_final)
        errors.append(x2_error)
    Output = np.concatenate((np.array(stepsizes, ndmin=2), np.array(errors, ndmin=2)), axis=0)
    np.savetxt('ErrorTask1b.txt', Output, delimiter = ',', fmt='%1.3f')
    return


# ------------------- main -------------------
all_x2_states, all_time_points =  explicit_euler(x_states.copy(), delta_t, t_final)
save_results(all_x2_states, all_time_points)
x_states = inital_x_states
exercise1b(x_states.copy(), t_final=100)
