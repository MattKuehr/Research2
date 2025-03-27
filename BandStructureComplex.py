import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve  # for multidimensional solving if needed
import sympy as sp

# -------------------------------
# User Inputs: PT-Symmetric Example from PhysRevB
# -------------------------------
# For a PT-symmetric crystal, we introduce balanced gain and loss:
# Here we take: 
#   layer A: εₐ = 3.8 + 0.2j, μₐ = 1, thickness dₐ = 0.42
#   layer B: ε_b = 1.0 - 0.2j, μ_b = 1, thickness d_b = 0.58
# so that the overall unit cell has Λ = dₐ + d_b = 1.
epsilons = [3.8 + 0.2j, 1.0 - 0.2j]
mus = [1.0, 1.0]
thicknesses = [0.42, 0.58]
Lambda = sum(thicknesses)  # should be 1.0

# -------------------------------
# Transfer Matrix for a Single Layer
# -------------------------------
def transfer_matrix_layer(omega, d, epsilon, mu):
    # Calculate wave number and impedance using complex arithmetic.
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
# Dispersion Relation (Complex)
# -------------------------------
def dispersion_func_complex(omega, q):
    """
    Dispersion function:
       f(omega; q) = 0.5 * Trace[M_unit(omega)] - cos(q * Lambda)
    In the PT-symmetric case this is complex.
    A valid eigenfrequency must satisfy f(omega; q) = 0 (both real and imaginary parts vanish).
    """
    M = unit_cell_matrix(omega)
    return 0.5 * np.trace(M) - np.cos(q * Lambda)

# -------------------------------
# Finite Difference Derivative for Complex Function
# -------------------------------
def dispersion_func_complex_derivative(omega, q, h=1e-6):
    f_plus  = dispersion_func_complex(omega + h, q)
    f_minus = dispersion_func_complex(omega - h, q)
    return (f_plus - f_minus) / (2.0 * h)

# -------------------------------
# Symbolic Differentiation of the Dispersion Function
# -------------------------------
def dispersion_func_deriv_exact_complex(omega_val, q_val):
    omega_sym = sp.symbols('omega', complex=True)
    M_total_sym = sp.eye(2)
    for d, eps, mu in zip(thicknesses, epsilons, mus):
        eps_sym = sp.N(eps)
        mu_sym = sp.N(mu)
        d_sym = sp.N(d)
        k_sym = omega_sym * sp.sqrt(eps_sym * mu_sym)
        Z_sym = sp.sqrt(mu_sym / eps_sym)
        cos_term = sp.cos(k_sym * d_sym)
        sin_term = sp.sin(k_sym * d_sym)
        M_layer_sym = sp.Matrix([[cos_term, sp.I * sin_term / Z_sym],
                                 [sp.I * Z_sym * sin_term, cos_term]])
        M_total_sym = M_layer_sym * M_total_sym
    trace_M_sym = M_total_sym.trace()
    f_sym = 0.5 * trace_M_sym - sp.cos(q_val * Lambda)
    df_domega_sym = sp.diff(f_sym, omega_sym)
    df_val = df_domega_sym.subs(omega_sym, omega_val)
    return complex(sp.N(df_val))

# -------------------------------
# Modified Newton's Method for Complex Roots
# -------------------------------
def newton_complex(omega0, q, tol=1e-8, maxiter=50, h=1e-6):
    omega = omega0
    for i in range(maxiter):
        f_val = dispersion_func_complex(omega, q)
        fprime = dispersion_func_complex_derivative(omega, q, h)
        if abs(fprime) < 1e-12:
            break
        omega_new = omega - f_val / fprime
        if abs(omega_new - omega) < tol:
            return omega_new, i+1
        omega = omega_new
    return omega, maxiter

# -------------------------------
# Band Structure Computation via Modified Newton's Method
# -------------------------------
q_values = np.linspace(0, np.pi, 300)
omega_max = 6 * np.pi

band_data = {}

# Use a coarse grid of initial guesses; here we span along the real axis and include a small imaginary offset.
omega_coarse = np.linspace(0.001, omega_max, 500)

for q in q_values:
    omega_solutions = []
    for omega_guess in omega_coarse:
        for offset in [0, 1e-6]:
            guess = omega_guess + offset*1j
            root, iterations = newton_complex(guess, q)
            if abs(dispersion_func_complex(root, q)) < 1e-4:
                omega_solutions.append(root)
    # Remove duplicates (clustering in the complex plane)
    omega_solutions_unique = []
    tol_cluster = 1e-3
    for sol in sorted(omega_solutions, key=lambda x: (x.real, x.imag)):
        if not omega_solutions_unique or abs(sol - omega_solutions_unique[-1]) > tol_cluster:
            omega_solutions_unique.append(sol)
    band_data[q] = omega_solutions_unique

# -------------------------------
# Collect Data for Plotting
# -------------------------------
q_list = []
omega_list = []
for q, omegas in band_data.items():
    for omega in omegas:
        q_list.append(q)
        omega_list.append(omega)

q_array = np.array(q_list)
omega_array = np.array(omega_list)

# Exploit PT symmetry to mirror q values: ω(q) = ω*(−q)
q_array_full = np.concatenate([q_array, -q_array])
omega_array_full = np.concatenate([omega_array, np.conjugate(omega_array)])

# Normalize frequencies by (2π) and q by (2π)
q_norm = q_array_full / (2 * np.pi)
omega_norm = omega_array_full / (2 * np.pi)

# Sort the data for plotting clarity.
sort_idx = np.argsort(q_norm.real)
q_norm_sorted = q_norm[sort_idx]
omega_norm_sorted = omega_norm[sort_idx]

# -------------------------------
# Plotting: Two Panels for Real and Imaginary Parts
# -------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Plot the real part of the normalized frequency.
ax1.scatter(q_norm_sorted.real, omega_norm_sorted.real, s=10, color='blue', label='Real part')
ax1.set_ylabel(r'Normalized Frequency: Re($\omega\Lambda/2\pi c$)')
ax1.set_title('Band Structure: Real Part')
ax1.grid(True)
ax1.legend()

# Plot the imaginary part of the normalized frequency.
ax2.scatter(q_norm_sorted.real, omega_norm_sorted.imag, s=10, color='red', label='Imaginary part')
ax2.set_xlabel(r'Normalized Wave Vector: $q\Lambda/2\pi$')
ax2.set_ylabel(r'Normalized Frequency: Im($\omega\Lambda/2\pi c$)')
ax2.set_title('Band Structure: Imaginary Part')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
