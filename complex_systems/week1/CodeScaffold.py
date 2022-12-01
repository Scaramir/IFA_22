'''
Group 8: Florian Herzler, Dominik Bannwitz, Maximilian Otto
1. Homework. Exercise 4: 
'''

# provided to you by the best bioinformatics lecturer you'll ever have (starts with M, ends with t)

import numpy as np
import filecmp

# A: -----Fixed Quantities-----
#0. initial state
X0 = np.loadtxt('Input.txt')

# ===> fill here, everywhere where a "..." is <===

#r1 = 
#r2 = 
#r3 =
#r4 = 
#r5 = 

#1. Stoichiometric matrix
S = np.array([[1,-1,-1,0,1],[0,0,1,-1,-1],[0,0,0,0,1],[0,0,-1,0,0]]) # !!check dimension of the array!!

#2. reaction parameters
k = [5,3,12,7,3]


# B: functions that depend on the state of the system X
def ReactionRates(k,X):
        R = np.zeros((5,1))
        R[0] = k[0]
        R[1] = k[1] * X[0]
        R[2] = k[2] * X[0] * X[3]
        R[3] = k[3] * X[1]
        R[4] = k[4] * X[1]
        return R
# ===>       -----------------------     <===

# compute reaction propensities/rates
R = ReactionRates(k,X0)

#compute the value of the ode with time step delta_t = 1
dX = np.dot(S,R)

##a) save stoichiometric Matrix
np.savetxt('SMatrix.txt',S,delimiter = ',',newline='\n',fmt='%1.0f')
##b) save ODE value as float with 2 digits after the comma (determined with the c-style precision argument e.g. '%3.2f')
print(dX)
np.savetxt('ODEValue.txt',dX,delimiter=',',newline='\n',fmt='%1.2f')
