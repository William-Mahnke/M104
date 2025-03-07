import numpy as np
import matplotlib.pyplot as plt

## Bisection Method Code
def bisection(f, a, b, tol=1e-5, max_iter=15):
    '''
    Initial data:
    * the function (f)
    * two initial points/guesses (a & b)
    * a tolerance level (tol): if the distance between the current midpoint and
    one of the endpoints is small enough, then the program will stop and return
    the current midpoint
    * the maximum iterations allowed for the algorithm (max_iter). If the maximum
    iterations is reached, then the function will return the midpoint of a and b
    '''
    midpoints = [] # creating a list for the midpoints to use in a graph
    for n in range(max_iter): # going through iterations for the process
        m = (a + b) / 2 # calculating the midpoint between
        midpoints.append(m) # adding the current midpoint to the list
        if np.abs(f(m)-f(a)) < tol: # testing the distance between an endpoint and the midpoint is small enough
            break
        if f(a) * f(m) <= 0: # testing if there's a sign change between a and m
            b = m # assigns b to the current midpoint, and the loop starts again with new a and b
        else: # if there isn't a sign change between a and m, then (assuming guaranteed convergence), there's a sign change between b and m
            a = m # assigns a to the current midpoint, and the loop starts again with new a and b
    if n == max_iter - 1:
      print('WARNING: Maximum number of iterations reached, approximation may be innaccurate')
    return (a + b) / 2, midpoints, n + 1 #returns the midpoint of the last set of a, b, the list of the midpoints, and the number of iterations

f = lambda x: x**2 - 10
a = 4
b = -5
true_root = np.sqrt(10)

# Calculating the root approximation and a print statement
bisection_root, midpoints, iterations = bisection(f, a, b)
print(f'Root from bisection method: {bisection_root}')
print(f'The actual root: {true_root}')
print(f'Error: {np.abs(bisection_root - true_root)}')
print(f'Number of iterations: {iterations}')

# Creating the plot to show the midpoints compared to the actual root
plt.figure(figsize=(10, 5))
plt.plot(midpoints, label='Midpoint Approximations')
plt.hlines(true_root, 0, len(midpoints), color = 'g', linestyles = 'dotted', label = 'True Root')
plt.hlines(bisection_root, 0, len(midpoints), colors='r', linestyles='dashed', label='Root Approximation')
plt.xlabel('Iterations')
plt.ylabel('Midpoint Approximation')
plt.title('Convergence of the Bisection Method')
plt.legend()
plt.grid(True)
plt.show()

## Newton's Method Code
def newton_method(f, df, x0, tol=1e-10, max_iter=8):
    '''
    Initial Data:
    * the function and its derivative (f and df)
    * an initial guess for the root (x0)
    * a tolerance level (tol): if the difference between the current
    guess and the previous is small enough, than the program will stop
    and return the current guess
    * a maximum number of iterations (max_iter): if the maximum number
    of iterations is reached, the program will return the most recent guess
    along with a message warning about the potential innaccuracy of the guess
    '''
    approximations = [x0] # creating a list of the guesses to later use in a plot
    for n in range(max_iter): # going through and also counting the number of iterations
        x1 = x0 - f(x0) / df(x0) # formula for the next guess using the Newton method
        approximations.append(x1) # adding the new guess to the list of guesses
        if abs(x1 - x0) < tol: # testing the difference between the current and previous guess
            break
        x0 = x1 # assigning the new guess to x0 to continue the loop and calculate the next guess
    return x1, approximations, n + 1 # returns the last guess (the root approximation), the list of approximations, and number of iterations

f = lambda x: x**(1/3)
df = lambda x: (1/3)*x**(-2/3)
x0 = 0.1

root_newton, approximations_newton, iterations = newton_method(f, df, x0)
true_root = np.sqrt(0)

print(f'Root from Newton\'s method: {root_newton}')
print(f'Actual value of the root: {true_root}')
print(f'Error: {np.abs(root_newton - true_root)}')
print(f'Number of iterations: {iterations}')

plt.figure(figsize=(10, 5))
plt.plot(approximations_newton, label='Newton Approximations')
plt.hlines(root_newton, 0, len(approximations_newton), colors='r', linestyles='dashed', label='Root')
plt.hlines(true_root, 0, len(approximations_newton), colors='g', linestyles='dotted',label='Actual Root')
plt.xlabel('Iteration')
plt.ylabel('Approximation of Root')
plt.title('Convergence of Newton\'s Method')
plt.legend()
plt.grid(True)
plt.show()
