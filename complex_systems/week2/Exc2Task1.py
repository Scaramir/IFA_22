'''
Group 8: Florian Herzler, Dominik Bannwitz, Maximilian Otto
Homework 2
'''

from numpy import dtype
import numpy as np
import matplotlib.pyplot as plt
input = np.loadtxt('Input.txt')
np.random.seed(seed=int(input[0]))
nrSimulations = int(input[1])

t_final = 10.0
lam = 1e-4
delta = 1e-8
beta = 5e-5
k_r = 0.3

X = np.array([[lam/delta],
              [20],
              [0],
              [0]])

#   r1  r2  r3  r4  r5  r6
#X1  1  -1  -1   0   0   0
#X2  0   0   1  -1  -1   0
#X3  0   0   0   0   1  -1
#X4  0   0   0   1   0   0
stochiometry = np.array([[1,-1,-1, 0, 0, 0],
                         [0, 0, 1,-1,-1, 0],
                         [0, 0, 0, 0, 1,-1],
                         [0, 0, 0, 1, 0, 0]])

def propensities(X,lam,delta,beta,k_r):
    prop_vec = [lam,
                X[0] * delta,
                X[0] * X[1] * beta,
                X[1] * 3e7 * delta,
                X[1] * k_r,
                X[2] * delta]
    return prop_vec

def Time_To_Next_Reaction(lam):
    """
    @brief The function samples from an exponential distribution
    @param lam : real value positive.
    """

    #small hack as the numpy uniform random number includes 0
    r = np.random.rand()
    while r==0:
        r = np.random.rand()
    return (1.0/lam)*np.log(1.0/r)

def Find_Reaction_Index(a):
    """
    @brief The function takes in the propensity vector and returns ...
    @param a : Array (num_reaction,1)
    """
    #small hack as the numpy random number includes 0
    r = np.random.rand()
    while r == 0:
        r = np.random.rand()
    return np.sum(np.cumsum(a) < r*np.sum(a))

def SSA(stochiometry,X_0,t_final,lam,delta,beta,k_r):
    #for storage
    X_store = []
    T_store = []
    #initialize
    t = 0.0
    x = X_0
    X_store.append(x[1,0]) 
    T_store.append(t)

    while t < t_final:
        a = propensities(x,lam,delta,beta,k_r)
        #first jump time
        tau = Time_To_Next_Reaction(np.sum(a))
        #test if we have jumped too far
        if (t + tau > t_final) or (np.sum(a) == 0):
            return np.array(X_store),np.array(T_store)
        else:
            #since we have not, we need to find the next reaction
            t = t + tau
            j = Find_Reaction_Index(a)
            x = x + stochiometry[:,[j]]
            #update our storage
            X_store.append(x[1,0])
            T_store.append(t)

#to save trajectories
#Task 1a:
for i in range(nrSimulations):
    states, times = SSA(stochiometry, X, t_final, lam, delta, beta, k_r)
    Output = np.concatenate((np.array(times,ndmin=2),np.array(states,ndmin=2)), axis=0)
    np.savetxt('Task1Traj'+str(i+1)+'.txt',Output,delimiter = ',',fmt='%1.3f')

#Task 1b:
X[1] = [2]
infected = []
for i in range(1000):
    states, times = SSA(stochiometry, X, t_final, lam, delta, beta, k_r)
    infected.append(states[-1])
#TODO: histogram of states (X[1]) @T=10 (last point)
plt.hist(infected)
#TODO: depict probability that 0,...,20 individuals infected
#TODO: histogram of X[3]