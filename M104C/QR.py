import numpy as np

# QR algorithm from Park's notes
def qr(A):
    """
    Return full QR factorization.

    Input:
        A (array): a tall rectangular or square matrix
    Output:
        Q, R (array): factors Q, R in QR factorization
    """
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)

    for i in range(n):
        k = m - i

        x = R[i:, i].reshape(-1, 1)
        w = np.zeros_like(x).reshape(-1, 1)
        w[0] = - np.sign(x[0])*np.linalg.norm(x)

        v = w - x
        v_ = ((2./(v.T @ v))*v)
        R[i:, i:] = R[i:, i:] - v_ @ (v.T @ R[i:, i:])
        
        Q[:, i:] = Q[:, i:] - (Q[:, i:] @ v_) @ v.T

    return Q, R    