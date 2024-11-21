'''
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
'''

import numpy as np
import matplotlib.pyplot as plt

def equation(omega, k, epsilon1, epsilon2):
    """
    Nonlinear equation for photonic crystal bend structure
    
    Parameters:
    - omega: frequency to solve for
    - k: wave vector
    - epsilon1: permittivity of air
    - epsilon2: permittivity of silicon
    
    Returns: equation value
    """
    term1 = np.cos(2 * omega * np.sqrt(epsilon2)) * np.cos(omega * np.sqrt(epsilon1))
    term2 = -omega * np.sqrt(epsilon1) * np.sin(2 * omega * np.sqrt(epsilon2)) * np.sin(omega * np.sqrt(epsilon1))
    term3 = -omega * np.sqrt(epsilon2) * np.sin(2 * omega * np.sqrt(epsilon2)) * np.sin(omega * np.sqrt(epsilon1))
    term4 = omega**2 * np.sqrt(epsilon1) * np.sqrt(epsilon2) * np.cos(2 * omega * np.sqrt(epsilon2)) * np.cos(omega * np.sqrt(epsilon1))
    
    return term1 + term2 + term3 + term4 - 2 * np.cos(2*k)

def secant_method(f, x0, x1, tol=1e-6, max_iter=100, k=None, epsilon1=None, epsilon2=None):
    """
    Secant method for root finding
    
    Parameters:
    - f: function to solve
    - x0, x1: initial guess points
    - tol: tolerance for convergence
    - max_iter: maximum iterations
    - k: wave vector
    - epsilon1: permittivity of air
    - epsilon2: permittivity of silicon
    
    Returns: approximated root
    """
    for _ in range(max_iter):
        # Create a function with fixed parameters
        def f_fixed(omega):
            return f(omega, k, epsilon1, epsilon2)
        
        fx0 = f_fixed(x0)
        fx1 = f_fixed(x1)
        
        if abs(fx1) < tol:
            return x1
        
        # Secant method update
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        
        x0, x1 = x1, x_new
    
    return x1

def solve_photonic_crystal_dispersion(k_values, epsilon1, epsilon2):
    """
    Solve for omega values across k using secant method
    
    Parameters:
    - k_values: array of k values
    - epsilon1: permittivity of air
    - epsilon2: permittivity of silicon
    
    Returns: omega values
    """
    omega_solutions = []
    
    for k in k_values:
        def f(omega, k, epsilon1, epsilon2):
            return equation(omega, k, epsilon1, epsilon2)
        
        # Initial guesses matter for nonlinear problems
        try:
            omega_sol = secant_method(f, 1.0, 2.0, k=k, epsilon1=epsilon1, epsilon2=epsilon2)
            omega_solutions.append(omega_sol)
        except Exception as e:
            print(f"Could not solve for k={k}: {e}")
            omega_solutions.append(np.nan)
    
    return omega_solutions

# Example values (you should replace with precise experimental values)
EPSILON_AIR = 1.0  # Permittivity of air
EPSILON_SILICON = 11.68  # Approximate permittivity of silicon

# Generate k values
k_values = np.linspace(-np.pi, np.pi, 100)

# Solve for omega
omega_values = solve_photonic_crystal_dispersion(k_values, EPSILON_AIR, EPSILON_SILICON)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_values, omega_values, 'b-')
plt.title('Photonic Crystal Bend Structure')
plt.xlabel('Wave Vector k')
plt.ylabel('Frequency ω')
plt.grid(True)
plt.show()
