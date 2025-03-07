# import numpy to aid with creating functions
import numpy as np

def bisection(a, b, f, max_int = 500):
    '''
    Uses the bisection method to find a root for a function in a defined interval
    Inputs:
    a, b - interval (note a < b)
    f - function interested in finding a root for
    max_int - maximum number of iterations to perform
    Output:
    c - approximate root 
    '''
    # optional addition - add tolerance level so method stops if new value c
    # is relatively close to previous values (for some defined epsilon)
    for i in range(max_int):
        c = (a + b) / 2
        if f(a) * f(c) <= 0:
            b = c
        else:
            a = c
    return c
