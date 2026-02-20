import numpy as np
from numba import njit, prange

@njit
def get_TA(omega, phi, L, delta, epsilon0):
    """
    Layer A transfer matrix (Numba-compiled).
    """

    '''
    Conditions:
    All values must be non-negative
    epsilon0 - detla >=0
    '''
    if isinstance(epsilon0, complex):
        assert epsilon0.real - delta >= 0, "Real component of EPSILON0 must be >= DELTA"
    else:
        assert epsilon0 - delta >= 0, "EPSILON0 must be >= DELTA"

    a = omega * L
    n1 = np.sqrt(epsilon0 + delta)
    n2 = np.sqrt(epsilon0 - delta)

    u1, v1 = np.cos(n1 * a), np.sin(n1 * a)
    u2, v2 = np.cos(n2 * a), np.sin(n2 * a)

    u, v = np.cos(phi), np.sin(phi)

    # Create complex array
    T = np.zeros((4, 4), dtype=np.complex128)
    
    T[0, 0] = u*u*u1 + v*v*u2
    T[0, 1] = u*v*u1 - u*v*u2
    T[0, 2] = -1j*u*v*v1/n1 + 1j*u*v*v2/n2
    T[0, 3] = 1j*u*u*v1/n1 + 1j*v*v*v2/n2
    
    T[1, 0] = u*v*u1 - u*v*u2
    T[1, 1] = v*v*u1 + u*u*u2
    T[1, 2] = -1j*v*v*v1/n1 - 1j*u*u*v2/n2
    T[1, 3] = 1j*u*v*v1/n1 - 1j*u*v*v2/n2
    
    T[2, 0] = -1j*n1*u*v*v1 + 1j*n2*u*v*v2
    T[2, 1] = -1j*n1*v*v*v1 - 1j*n2*u*u*v2
    T[2, 2] = v*v*u1 + u*u*u2
    T[2, 3] = -u*v*u1 + u*v*u2
    
    T[3, 0] = 1j*n1*u*u*v1 + 1j*n2*v*v*v2
    T[3, 1] = 1j*n1*u*v*v1 - 1j*n2*u*v*v2
    T[3, 2] = -u*v*u1 + u*v*u2
    T[3, 3] = u*u*u1 + v*v*u2
    
    return T

@njit
def get_TF(omega, L_F, alpha, beta, epsilon0_tilde):
    """
    Layer F transfer matrix (Numba-compiled).
    """

    '''
    Conditions:
    All values must be non-negative
    EPSILON0_TILDE >= ALPHA
    0 <= BETA <= 1
    '''

    assert epsilon0_tilde - alpha >= 0, "EPSILON0_TIDLE must be >= ALPHA"
    assert 0 <= beta <= 1, "BETA MUST BE in [0,1]"

    a = omega * L_F
    mu = 1.0

    n1 = np.sqrt((epsilon0_tilde + alpha) * (mu + beta))
    n2 = np.sqrt((epsilon0_tilde - alpha) * (mu - beta))
    m1 = np.sqrt((epsilon0_tilde + alpha) / (mu + beta))
    m2 = np.sqrt((epsilon0_tilde - alpha) / (mu - beta))

    u1, v1 = np.cos(n1 * a), np.sin(n1 * a)
    u2, v2 = np.cos(n2 * a), np.sin(n2 * a)

    # Create complex array
    T = np.zeros((4, 4), dtype=np.complex128)
    
    T[0, 0] = u1 + u2
    T[0, 1] = 1j * (u1 - u2)
    T[0, 2] = v1/m1 - v2/m2
    T[0, 3] = 1j * (v1/m1 + v2/m2)
    
    T[1, 0] = -1j * (u1 - u2)
    T[1, 1] = u1 + u2
    T[1, 2] = -1j * (v1/m1 + v2/m2)
    T[1, 3] = v1/m1 - v2/m2
    
    T[2, 0] = -m1*v1 + m2*v2
    T[2, 1] = -1j * (m1*v1 + m2*v2)
    T[2, 2] = u1 + u2
    T[2, 3] = 1j * (u1 - u2)
    
    T[3, 0] = 1j * (m1*v1 + m2*v2)
    T[3, 1] = -m1*v1 + m2*v2
    T[3, 2] = -1j * (u1 - u2)
    T[3, 3] = u1 + u2

    return 0.5 * T