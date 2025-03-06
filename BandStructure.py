import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

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


# -------------------------------
# User Inputs: Example with 3 layers and repeated unit cell
# -------------------------------
n_layers = 2           # Number of unique layers in the unit cell
n_repeats = 10         # Number of unit cells repeated (for finite crystal analysis)

# Define parameters for the 3 layers:
# Layer 1: epsilon=4.0, mu=1.0, thickness=0.3
# Layer 2: epsilon=2.5, mu=1.0, thickness=0.4
# Layer 3: epsilon=1.0, mu=1.0, thickness=0.3
epsilons = [1.0, 1.0]
mus = [1.0, 6.0]
thicknesses = [0.6, 0.4]

# Total unit cell thickness Λ
Lambda = sum(thicknesses)

# -------------------------------
# Transfer Matrix for a Single Layer
# -------------------------------
def transfer_matrix_layer(omega, d, epsilon, mu):
    """
    Compute the 2x2 transfer matrix for a layer with thickness d,
    relative permittivity epsilon, and relative permeability mu at frequency omega.
    
    For normal incidence:
       M_layer = [[cos(k*d), (1j*sin(k*d))/Z],
                  [1j*Z*sin(k*d), cos(k*d)]]
    where k = omega * sqrt(epsilon*mu) and Z = sqrt(mu/epsilon).
    """
    k = omega * np.sqrt(epsilon * mu)
    Z = np.sqrt(mu / epsilon)
    cos_term = np.cos(k * d)
    sin_term = np.sin(k * d)
    M = np.array([[cos_term, 1j * sin_term / Z],
                  [1j * Z * sin_term, cos_term]])
    return M

# -------------------------------
# Compute the Unit Cell Transfer Matrix
# -------------------------------
def unit_cell_matrix(omega):
    """
    Multiply the transfer matrices of all layers in sequence
    to obtain the overall transfer matrix for one unit cell.
    """
    M_total = np.eye(2, dtype=complex)
    for d, eps, mu in zip(thicknesses, epsilons, mus):
        M_total = np.dot(transfer_matrix_layer(omega, d, eps, mu), M_total)
    return M_total

# -------------------------------
# Define the Dispersion Relation Function for a Given q
# -------------------------------
def dispersion_func(omega, q):
    """
    For a periodic photonic crystal, the dispersion relation is:
       cos(q * Λ) = 1/2 * Trace(M_unit(omega))
    Define f(omega; q) = 1/2 * Trace(M_unit(omega)) - cos(q*Λ) = 0,
    and solve f(omega, q) = 0 for ω.
    """
    M = unit_cell_matrix(omega)
    return 0.5 * np.trace(M).real - np.cos(q * Lambda)

# -------------------------------
# Root Finding: Solve for ω for Each q Value (Infinite Crystal)
# -------------------------------
# We choose q values for the positive branch from 0 to π.
q_values = np.linspace(0, np.pi, 300)
omega_max = 6 * np.pi  # Maximum frequency for the search
omega_coarse = np.linspace(0.001, omega_max, 1000)  # Coarse grid for bracketing

band_data = {}
for q in q_values:
    # Evaluate dispersion_func for each scalar omega using list comprehension.
    f_vals = np.array([dispersion_func(omega, q) for omega in omega_coarse])
    # Find intervals where the sign changes.
    sign_change_indices = np.where(np.diff(np.sign(f_vals)))[0]
    omega_solutions = []
    for idx in sign_change_indices:
        omega_left = omega_coarse[idx]
        omega_right = omega_coarse[idx + 1]
        try:
            sol = root_scalar(dispersion_func, args=(q,), bracket=[omega_left, omega_right], method='brentq')
            if sol.converged:
                omega_solutions.append(sol.root)
        except Exception:
            pass
    band_data[q] = sorted(omega_solutions)

# -------------------------------
# Collect and Mirror Data for Both q Branches
# -------------------------------
q_list = []
omega_list = []
for q, omegas in band_data.items():
    for omega in omegas:
        q_list.append(q)
        omega_list.append(omega)

q_array = np.array(q_list)
omega_array = np.array(omega_list)

# Mirror q values for the negative branch.
q_array_full = np.concatenate([q_array, -q_array])
omega_array_full = np.concatenate([omega_array, omega_array])

# -------------------------------
# Normalize and Sort Data for Plotting
# -------------------------------
# Normalization: q_norm = qΛ/(2π) and ω_norm = ω/(2π) (with c = 1)
q_norm = q_array_full / (2 * np.pi)
omega_norm = omega_array_full / (2 * np.pi)

sort_idx = np.argsort(q_norm)
q_norm_sorted = q_norm[sort_idx]
omega_norm_sorted = omega_norm[sort_idx]

# -------------------------------
# Plot the Band Structure for the Infinite Crystal
# -------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(q_norm_sorted, omega_norm_sorted, s=10, color='blue', label='Band Structure')
plt.xlabel(r'Normalized Bloch Wave Vector $\frac{q\Lambda}{2\pi}$')
plt.ylabel(r'Normalized Frequency $\frac{\omega\Lambda}{2\pi c}$')
plt.title('Band Structure of a 3-Layer Photonic Crystal\n(Infinite Crystal)')
plt.xlim(-0.5, 0.5)
plt.ylim(0, 3)
plt.grid(True)
plt.legend()
plt.show()

# -------------------------------
# Finite Crystal Analysis: Overall Transfer Matrix for n_repeats Unit Cells
# -------------------------------
def finite_crystal_matrix(omega, n_cells):
    """
    Compute the overall transfer matrix for a finite crystal
    composed of n_cells repeated unit cells.
    """
    M_uc = unit_cell_matrix(omega)
    return np.linalg.matrix_power(M_uc, n_cells)

# Example: compute and display the finite crystal transfer matrix at an example frequency
#omega_example = np.pi  # Arbitrary example frequency
#M_finite = finite_crystal_matrix(omega_example, n_repeats)
#print("Finite crystal transfer matrix at ω =", omega_example, ":\n", M_finite)
