import numpy as np

# Initial guess
w0 = 0.1

# Search tolerance
tol = 1e-6

# Define parameter ranges
L_values = [0, 0.25, 0.5, 0.75, 1]
n_values = [np.sqrt(1.0006), np.sqrt(11.68)]  # air and silicon
k_values = [-np.pi, -(5*np.pi)/6, -(3*np.pi)/4, -(2*np.pi)/3, -np.pi/2, 
            -np.pi/3, -np.pi/4, -np.pi/6, 0, np.pi/6, np.pi/4, np.pi/3, 
            np.pi/2, (2*np.pi)/3, (3*np.pi)/4, (5*np.pi)/6, np.pi]

# Objective function to find roots (omega)
def f(x, L, k, n):
    return 2*np.cos(k*L) - 2*np.cos(x*n*L)

# Objective function derivative
def f_prime(x, L, k, n):
    return 2*n*L*np.sin(x*n*L)

def newton_raphson(f, f_prime, w0, L, k, n, tol):
    
    w = w0
    
    while np.abs(f(w, L, k, n)) > tol:
        w = w - (f(w, L, k, n))/(f_prime(w, L, k, n))
    return w

# Print header
print(f"{'L':>8} {'n':>10} {'k/π':>10} {'Root':>15}")
print("-" * 46)

# Iterate through all combinations
for L in L_values:
    for n in n_values:
        for k in k_values:
            root = newton_raphson(f, f_prime, w0, L, k, n, tol)
            
            # Format k in terms of π for cleaner output
            k_pi = k/np.pi
            
            if root is not None:
                print(f"{L:8.2f} {n:10.4f} {k_pi:10.4f} {root:15.6f}")
            else:
                print(f"{L:8.2f} {n:10.4f} {k_pi:10.4f} {'No convergence':>15}")