import numpy as np
import matplotlib.pyplot as plt

def sanity_check_ode(t, x): # f(t,x) for the sanity check
    return x

def actual_ode(t, x): # f(t,x) for the ode we're approximating
    return (4 * t**2 - 2 * x) / t

def exact_solution_sanity_check(t): # actual solution for the sanity check
    return np.exp(t)

def exact_solution_actual_ode(t, x1): # actual solution for the ode we're approximaing
    #C = x1 - (4/3)
    #return (4/3) * t**3 + C / t**2
    return t**(-2) + t**2

def adam_bashforth(f, y0, t0, tf, h): # AB3 method
    N = int((tf - t0) / h) # calculating final time step
    t = np.linspace(t0, tf, N+1) # creating the time steps
    y = np.zeros(N+1) # array for approximations
    y[0:3] = y0
    for i in range(2, N): # for loop for calculating approximations using AB3
        y[i+1] = y[i] + h * (23/12*f(t[i], y[i]) - 16/12*f(t[i-1], y[i-1]) + 5/12*f(t[i-2], y[i-2]))
    return t, y

def adam_moulton(f, y0, t0, tf, h): # AM3 method
    N = int((tf - t0) / h) # calculating final time step
    t = np.linspace(t0, tf, N+1) # creating the time steps
    y = np.zeros(N+1) # array to store approximations
    y[0:3] = y0
    for i in range(2, N):
        y_pred = y[i] + h * (3/2*f(t[i], y[i]) - 1/2*f(t[i-1], y[i-1]))
        for _ in range(5):
            y_pred = y[i] + h * (9/24*f(t[i+1], y_pred) + 19/24*f(t[i], y[i]) - 5/24*f(t[i-1], y[i-1]) + 1/24*f(t[i-2], y[i-2])) # AM3 method
        y[i+1] = y_pred
    return t, y

t0 = 0
tf = 5
h = 0.1
x0_sanity = [1]
x0_actual = [2]

# Assinging arrays to plot for both the sanity check and the ode we're approximating
t_sanity_ab, y_sanity_ab = adam_bashforth(sanity_check_ode, x0_sanity, t0, tf, h)
t_sanity_am, y_sanity_am = adam_moulton(sanity_check_ode, x0_sanity, t0, tf, h)

t_actual_ab, y_actual_ab = adam_bashforth(actual_ode, x0_actual, 1, tf, h)
t_actual_am, y_actual_am = adam_moulton(actual_ode, x0_actual, 1, tf, h)

t_exact_sanity = np.linspace(t0, tf, 500)
y_exact_sanity = exact_solution_sanity_check(t_exact_sanity)

t_exact_actual = np.linspace(1, tf, 500)
y_exact_actual = exact_solution_actual_ode(t_exact_actual, x0_actual[0])

# left plot, for sanity check
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_exact_sanity, y_exact_sanity, 'k--', label='Exact Solution')
plt.plot(t_sanity_ab, y_sanity_ab, 'bo-', label='Adam-Bashforth')
plt.plot(t_sanity_am, y_sanity_am, 'rx-', label='Adam-Moulton')
plt.xlabel('Time t')
plt.ylabel('Solution x')
plt.title('Sanity Check ODE')
plt.legend()
plt.grid(True)

# right plot, for the ode we're approximating
plt.subplot(1, 2, 2)
plt.plot(t_exact_actual, y_exact_actual, 'k--', label='Exact Solution')
plt.plot(t_actual_ab, y_actual_ab, 'bo-', label='Adam-Bashforth')
plt.plot(t_actual_am, y_actual_am, 'rx-', label='Adam-Moulton')
plt.xlabel('Time t')
plt.ylabel('Solution x')
plt.title('Actual ODE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()