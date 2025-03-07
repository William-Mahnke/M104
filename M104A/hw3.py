#HW3 Coding Problems
import numpy as np
a = np.array([5*x for x in range(16)]).reshape(4,4)
#3
b = np.array([x**2 for x in range(1,5)]).reshape(4,-1)
#print(a + b) #a

c = np.array(range(4))
#print(a * c) #b

#4
ind = ([0,0,1,1,3,3],
       [0,3,0,3,0,3])
d = a[ind].reshape(3,2)
e = c[[0,-1]]
print(d - e) #a

mask = (a >= 30) & (a < 50)
#a[mask] = -a[mask]
print(a) #b

a1 = np.where((a % 2 == 0),a**2,a**3)
print(a1) #c 

#5
rando = np.random.randint(-1000000,1000000,(1000,2000))
ind = np.unravel_index(rando.argmin(),rando.shape)
print("Minimum: ", rando.min(), "Index: ", ind)
print(rando[ind] == rando.min())

#6 
X = np.random.uniform(3,3.5,(100,20))
Y = np.random.uniform(3,3.5,(100,20)) #a

error = np.abs(X-Y) #b

ind1 = np.unravel_index(error.argmax(),error.shape)
print("Maximum error: ", error.min(), "Index: ", ind1) #c

#7
w = np.random.randint(0,20,(20,)) #a - implementing part b didnt work with integeres
w = w.astype(float)
w[-3:] = np.nan #b

def removing_nan(array):
    mask = ~(np.isnan(array))
    filtered = array[mask]
    return filtered

print(removing_nan(w))