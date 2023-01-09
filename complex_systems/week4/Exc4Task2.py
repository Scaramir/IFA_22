'''
Group 8: Florian Herzler, Dominik Bannwitz, Maximilian Otto
Homework 4
'''

import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import integrate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# set seed for reproducibility
np.random.seed(0)
nrSimulations = 300

# Load data
t_data, y_data = np.load('./Data.npy')
# convert the data to one-dimensional arrays
t_data = t_data.flatten()
y_data = y_data.flatten()

# ------------------ Define model and params ------------------
# params given
k0 = 100; k1 = 0.1; c2 = 0.01; c3 = 1.0; c4 = 10; c5 = 2
# our params from task one TODO: change
# k0 = 100; k1 = 0.1; c2 = 0.01; c3 = 1.0; c4 = 10; c5 = 2
initial_x_states = np.array([
    [k0/k1],
    [0],
    [1]])

# Define stoichiometry matrix
#    r0  r1  r2  r3  r4  r5
# X0  0  -1  -1   0   0   0
# X1  0   0   1  -1   0   0
# X2  0   0  -1   0   1  -1
stoichiometry_matrix = np.array([[0, -1, -1, 0, 0, 0],
                                [0, 0, 1, -1, 0, 0],
                                [0, 0, -1, 0, 1, -1]])

# Define propensities function
def propensities(x_states, c2, c3, c4, c5):
    prop_vec = [k0,
                k1 * x_states[0,0],
                c2 * x_states[0,0] * x_states[2,0],
                c3 * x_states[1,0],   
                c4 * x_states[1,0],
                c5 * x_states[2,0]]
    return prop_vec

def time_to_next_reaction(lamdba):
    """
    Exp distribution with mean 1/lamdba. `r` is random number between 0 and 1 and != 0.
    input: lamdba : real value positive.
    """
    r = np.random.rand()
    while r==0:
        r = np.random.rand()
    return (1.0/lamdba) * np.log(1.0/r)

def find_reaction_index(changes):
    """
    Propensitiy / reaction vector `changes`. `r` is random number between 0 and 1 and != 0.
    input: changes : Array (num_reaction,1)
    """
    r = np.random.rand()
    while r == 0:
        r = np.random.rand()
    return np.sum(np.cumsum(changes) < r*np.sum(changes))

def SSA(stoichiometry_matrix, x_initial_state, c2, c3, c4, c5):
    # keep track of the states and times
    x0_list = []
    x1_list = []
    x2_list = []
    timestep_list = []

    # initialize states and time
    current_time = 0.0
    x = x_initial_state
    x0_list.append(x[0,0])
    x1_list.append(x[1,0])
    x2_list.append(x[2,0])
    timestep_list.append(current_time)

    # TODO: adjust the termination condition
    # NOTE: i would suggest to use x1 (infected cells) == 0 as "viral infection has been eliminated"
    while True:
        # calculate reaction propensities
        changes = propensities(x, c2, c3, c4, c5)
        # add time until next reaction
        time_skip = time_to_next_reaction(np.sum(changes))
        # free virus exceeds 50
        if (x[2,0] >= 50):
            elimination = False
            return elimination, np.array(x0_list), np.array(x1_list), np.array(x2_list), np.array(timestep_list)
        # no infected cells and free viruses
        elif (x[1,0] == 0 and x[2,0] == 0):
            elimination = True
            return elimination, np.array(x0_list), np.array(x1_list), np.array(x2_list), np.array(timestep_list)
        
        current_time = current_time + time_skip
        # update model 
        j = find_reaction_index(changes)
        x = x + stoichiometry_matrix[:,[j]]
        # keep track of states and time points
        x0_list.append(x[0,0])
        x1_list.append(x[1,0])
        x2_list.append(x[2,0])
        timestep_list.append(current_time)

def plot_last(x2_initial, df):
    # last points before simulation end
    x0 = df[df['x2_initial'] == x2_initial]['x0_states'].apply(lambda x: x[-1]).reset_index(drop=True)
    x1 = df[df['x2_initial'] == x2_initial]['x1_states'].apply(lambda x: x[-1]).reset_index(drop=True)
    x2 = df[df['x2_initial'] == x2_initial]['x2_states'].apply(lambda x: x[-1]).reset_index(drop=True)

    # Create a figure with three subplots for x0/x1/x2 end states
    fig, axs = plt.subplots(1, 3, figsize=(12,4))

    # Create a boxplot + stripplot for each subplot
    sns.boxplot(data=x0, ax=axs[0])
    sns.stripplot(data=x0, size=5, jitter=False, linewidth=0.5, edgecolor="gray", ax=axs[0])
    axs[0].set_xticklabels(['X0 states'])
    axs[0].set_ylabel('uninfected cells')

    sns.boxplot(data=x1, ax=axs[1])
    sns.stripplot(data=x1, size=5, jitter=False, linewidth=0.5, edgecolor="gray", ax=axs[1])
    axs[1].set_xticklabels(['X1 states'])
    axs[1].set_ylabel('infected cells')

    sns.boxplot(data=x2, ax=axs[2])
    sns.stripplot(data=x2, size=5, jitter=False, linewidth=0.5, edgecolor="gray", ax=axs[2])
    axs[2].set_xticklabels(['X2 states'])
    axs[2].set_ylabel('free viruses')
    fig.suptitle('Amounts at end of simulation (X2 = {})'.format(x2_initial))
    plt.savefig('sim_ends_x2_{}.png'.format(x2_initial)) # saves white png
    plt.show()

# data structure:
# list of dictionaries w/ all trials 
simulations = []

for x2 in tqdm(range(1,6), desc='X2 Variation'):
    # reset initial state of x2 
    initial_x_states[2] = x2
    for i in tqdm(range(nrSimulations), desc='Simulations'):
        # run SSA
        elimination, x0_states, x1_states, x2_states, times = SSA(stoichiometry_matrix, initial_x_states, c2, c3, c4, c5)
        simulations.append({
            'x2_initial': x2,
            'elimination': elimination,
            'x0_states': x0_states,
            'x1_states': x1_states,
            'x2_states': x2_states,
            'time_points': times
        })

# transform into DataFrame
df = pd.concat([pd.DataFrame(simulations)])

# Calculate the probability of an infection for each value of 'x2_initial'
# basically divide infections by total number of trials (300)
probs = df[df['elimination'] == False].groupby('x2_initial')['elimination'].count() / nrSimulations

# Create a bar plot with Seaborn
ax = sns.barplot(x=probs.index, y=probs.values)

# Add percentage labels to the bars
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
    p.get_height()/2, '{:.1%}'.format(p.get_height()),
    fontsize=12, color='black', ha='center', va='center')

# Add labels and show the plot
plt.title('Probability of Infection')
plt.xlabel('initial free virus (X2)')
plt.ylabel('probability of infection')
plt.show()

# boxplot + stripplot of last x0/x1/x2 values for every initial x2
for x2 in tqdm(range(1,6), desc='Plotting last values'):
    plot_last(x2, df)
