import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit

# Load data
t_data, y_data = np.load('./Data.npy')
# convert the data to one-dimensional arrays
t_data = t_data.flatten()
y_data = y_data.flatten()

# Define model parameters and initial conditions
k0 = 100
k1 = 0.1
initial_x_states = np.array([
    k0/k1,
    0,
    20])

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
                k1 * x_states[0],
                c2 * x_states[0] * x_states[2],
                c3 * x_states[1],
                c4 * x_states[1],
                c5 * x_states[2]]
    return prop_vec

# Define right-hand side of ODE system
def rhs(x, t, c2, c3, c4, c5):
    return np.dot(stoichiometry_matrix, propensities(x, c2, c3, c4, c5))

# Define model prediction function
def ModelPrediction(t, c2, c3, c4, c5):
    x_states = integrate.odeint(
        rhs, initial_x_states.copy(), t, args=(c2, c3, c4, c5))
    return x_states[:, 2]  # return concentration of third species
    # return x_states  # return concentration of third species

static_p0 = [0.1, 1, 5, 1]

# Fit model to data
def easy_curve_fit(p0):
    popt, pcov = curve_fit(ModelPrediction, t_data, y_data, 
                        bounds=(0, np.inf), p0=p0)
    return popt, pcov


if __name__ == '__main__':

    popt, pcov = easy_curve_fit(static_p0)

    # Save optimized parameters to file
    np.savetxt("Params.txt", popt, fmt='%1.2f', delimiter=',')

    # Plot data and model prediction
    plt.plot(t_data, y_data, 'o', label='data')
    plt.plot(t_data, ModelPrediction(t_data, *popt), '-', label='model prediction')
    plt.xlabel("time")
    plt.ylabel("number viruses")
    plt.title("Task 1a")
    plt.legend()
    plt.savefig("Task1a.png")
    plt.show()
