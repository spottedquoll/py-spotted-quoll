import scipy.optimize as opt
import numpy as np

# Objective function
def objective(x):
    bd = np.dot(A, x) - b
    return np.dot(bd, bd)

# table values
A = np.array([[2,  3],
               [1,  2],
               [-9, 8]])

b = np.array([7,8,9])

# Set bounds for each unknown variablw
# bnds = ((0, None), (0, None))

initial_guess = [0.0, -1.0]

# Perform optimisation
result = opt.minimize(fun=objective, x0=initial_guess, options={'disp': True})
print(result)