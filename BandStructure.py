import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import sympy as sp

# -------------------------------
# User Inputs (same as original)
# -------------------------------

'''
Figure 1 (b)
Layer A: epsilon_a = 4, mu_a = 1, d_a = 0.4
Layer B: epsilon_b = 1, mu_b = 1, d_b = 0.6
Lambda = d_a + d_b = 1

Figure 2 (b) (PC1)
Layer A: epsilon_a = 3.8, mu_a = 1, d_a = 0.42
Layer B: epislon_b = 1, mu_b = 1, d_b = 0.58
Lamba = d_a + d_b = 1

Figure 2 (c) (PC2)
Layer A: epsilon_a = 4.2, mu_a = 1, d_a = 0.38
Layer B: epsilon_b = 1, mu_b = 1, d_b = 0.62
Lambda = d_a + d_b = 1

Figure 4 (b) (PC3)
Layer A: epsilon_a = 1, mu_a = 1, d_a = 0.35
Layer B: epsilon_b = 3.5, mu_b = 1, d_b = 0.65
Lambda = d_a + d_b = 1

Figure 4 (c) (PC4)
Layer A: epsilon_a = 1, mu_a = 1, d_a = 0.6
Layer B: epsilon_b = 1, mu_b = 6, d_b = 0.4
Lambda = d_a + d_b = 1
'''

epsilons = [1.0, 1.0]
mus = [1.0, 6.0]
thicknesses = [0.6, 0.4]
Lambda = sum(thicknesses)

# -------------------------------
# Transfer Matrix for a Single Layer
# -------------------------------
def transfer_matrix_layer(omega, d, epsilon, mu):
    k = omega * np.sqrt(epsilon * mu)
    Z = np.sqrt(mu / epsilon)
    cos_term = np.cos(k * d)
    sin_term = np.sin(k * d)
    M = np.array([[cos_term, 1j * sin_term / Z],
                  [1j * Z * sin_term, cos_term]], dtype=complex)
    return M

# -------------------------------
# Compute the Unit Cell Transfer Matrix
# -------------------------------
def unit_cell_matrix(omega):
    M_total = np.eye(2, dtype=complex)
    for d, eps, mu in zip(thicknesses, epsilons, mus):
        M_total = transfer_matrix_layer(omega, d, eps, mu) @ M_total
    return M_total

# -------------------------------
# Dispersion and Its Derivative
# -------------------------------
def dispersion_func(omega, q):
    """
    f(omega; q) = 0.5 * Trace(M_unit(omega)) - cos(q * Lambda).
    We want f(omega; q) = 0 for a band solution.
    """
    M = unit_cell_matrix(omega)
    return 0.5 * np.trace(M).real - np.cos(q * Lambda)

def dispersion_func_derivative(omega, q, h=1e-6):
    """
    Finite-difference derivative df/domega at omega.
    """
    # Central finite difference for better accuracy
    f_plus  = dispersion_func(omega + h, q)
    f_minus = dispersion_func(omega - h, q)
    return (f_plus - f_minus) / (2.0 * h)

def dispersion_func_deriv_exact(omega, q):
    """
    Compute the derivative of the dispersion function with respect to omega exactly
    using symbolic differentiation (via sympy).

    The dispersion function is:
        f(omega; q) = 0.5 * Trace(M_unit(omega)) - cos(q * Lambda)
    where M_unit(omega) is the unit cell transfer matrix.
    """
    # Define symbolic variable
    omega_sym = sp.symbols('omega', real=True)

    # Build the symbolic unit cell matrix
    M_total_sym = sp.eye(2)
    for d, eps, mu in zip(thicknesses, epsilons, mus):
        k_sym = omega_sym * sp.sqrt(eps * mu)
        Z_sym = sp.sqrt(mu / eps)
        cos_term = sp.cos(k_sym * d)
        sin_term = sp.sin(k_sym * d)
        # Define the layer transfer matrix symbolically
        M_layer_sym = sp.Matrix([[cos_term, sp.I * sin_term / Z_sym],
                                 [sp.I * Z_sym * sin_term, cos_term]])
        M_total_sym = M_layer_sym * M_total_sym

    # Compute the trace of the unit cell matrix
    trace_M_sym = M_total_sym.trace()

    # Define the symbolic dispersion function
    f_sym = 0.5 * trace_M_sym - sp.cos(q * Lambda)

    # Differentiate with respect to omega
    df_domega_sym = sp.diff(f_sym, omega_sym)

    # Evaluate the derivative at the provided omega value
    df_domega_val = df_domega_sym.subs(omega_sym, omega)
    
    # Return the numerical value of the derivative
    return float(sp.N(df_domega_val))

# -------------------------------
# Root Finding via Newton's Method
# -------------------------------
q_values = np.linspace(0, np.pi, 300)
omega_max = 6 * np.pi

# We still do a coarse search for sign changes to locate possible bands,
# but then we refine each root using Newton's method.
omega_coarse = np.linspace(0.001, omega_max, 1000)

band_data = {}

for q in q_values:
    # Evaluate dispersion_func on the coarse grid
    f_vals = np.array([dispersion_func(omega, q) for omega in omega_coarse])
    
    # Identify sign changes as before
    sign_change_indices = np.where(np.diff(np.sign(f_vals)))[0]
    
    omega_solutions = []
    for idx in sign_change_indices:
        omega_left = omega_coarse[idx]
        omega_right = omega_coarse[idx + 1]
        
        # Take a midpoint as initial guess for Newton
        omega_guess = 0.5 * (omega_left + omega_right)
        
        try:
            sol = root_scalar(
                dispersion_func,
                fprime=dispersion_func_derivative, # was dispersion_func_derivative
                x0=omega_guess,
                args=(q,),
                method='newton'
            )
            if sol.converged:
                omega_solutions.append(sol.root)
        except RuntimeError:
            pass
    
    # Remove duplicates (Newton can converge to the same root from
    # neighboring intervals). A simple approach is to do a tolerance-based filter.
    omega_solutions_unique = []
    tol = 1e-3
    for w in sorted(omega_solutions):
        if (not omega_solutions_unique) or (abs(w - omega_solutions_unique[-1]) > tol):
            omega_solutions_unique.append(w)
    
    band_data[q] = omega_solutions_unique

# -------------------------------
# Collect and Mirror Data
# -------------------------------
q_list = []
omega_list = []
for q, omegas in band_data.items():
    for omega in omegas:
        q_list.append(q)
        omega_list.append(omega)

q_array = np.array(q_list)
omega_array = np.array(omega_list)

# Mirror q values for negative branch
q_array_full = np.concatenate([q_array, -q_array])
omega_array_full = np.concatenate([omega_array, omega_array])

# -------------------------------
# Normalize, Sort, and Plot
# -------------------------------
q_norm = q_array_full / (2 * np.pi)
omega_norm = omega_array_full / (2 * np.pi)

sort_idx = np.argsort(q_norm)
q_norm_sorted = q_norm[sort_idx]
omega_norm_sorted = omega_norm[sort_idx]

plt.figure(figsize=(8, 6))
plt.scatter(q_norm_sorted, omega_norm_sorted, s=10, color='blue')
plt.xlabel(r'Normalized Wave Vector $q\Lambda / 2\pi$')
plt.ylabel(r'Normalized Frequency $\omega\Lambda / 2\pi c$')
plt.title('Band Structure via Newton’s Method')
plt.xlim(-0.5, 0.5)
plt.ylim(0, 3)
plt.grid(True)
plt.show()
