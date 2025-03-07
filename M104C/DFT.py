import numpy as np

def DFT(x):
    '''     
    Computes the Discrete Fourier Transform (DFT) of a vector 
    Input: 
    x (1D-array) - original vector
    Output:
    y (1D-array) - DFT of x
    '''
    n = x.shape[0]
    w = np.exp(-2.*np.pi*1j/n)

    k = np.arange(n) # powers for DFT matrix
    pow = k.reshape((-1,1)) * k.reshape((1,-1)) # matrix of powers

    DFT_matrix = w**pow # creating DFT matrix
    const = 1./np.sqrt(n) # constant for DFT matrix

    y = const * DFT_matrix @ x.reshape((-1,1)) # computing y

    return y