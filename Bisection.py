import numpy as np

def bisection(a,b,f, max_int = 500):
    for i in range(max_int):
        c = (a+b)/2
        if f(a)*f(c) <= 0:
            b = c
        else:
            a = c
    return c

f = lambda x: np.log10(x-9)
print(bisection(2,15,f))