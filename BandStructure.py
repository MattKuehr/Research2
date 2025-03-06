import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# -------------------------------
# 1. Define parameters
# -------------------------------
epsilon_a = 4.0
mu_a = 1.0
d_a = 0.4  # normalized thickness

epsilon_b = 1.0
mu_b = 1.0
d_b = 0.6

Lambda = d_a + d_b  # unit cell period (Lambda = 1)

# Refractive index and impedance (assuming c = 1)
k_a = lambda omega: omega * np.sqrt(epsilon_a * mu_a)
k_b = lambda omega: omega * np.sqrt(epsilon_b * mu_b)
z_a = np.sqrt(mu_a / epsilon_a)
z_b = np.sqrt(mu_b / epsilon_b)

# -------------------------------
# 2. Define the dispersion function F(omega)
# -------------------------------
# The dispersion relation is:
# cos(q*Lambda) = cos(k_a*d_a)*cos(k_b*d_b) - 0.5*(z_a/z_b + z_b/z_a)*sin(k_a*d_a)*sin(k_b*d_b)
F = lambda omega: (np.cos(k_a(omega) * d_a) * np.cos(k_b(omega) * d_b) -
                   0.5 * (z_a / z_b + z_b / z_a) * np.sin(k_a(omega) * d_a) * np.sin(k_b(omega) * d_b))

# -------------------------------
# 3. Define function for fixed q
# -------------------------------
# For a fixed q, we want to solve F(omega) = cos(q*Lambda),
# i.e., f(omega) = F(omega) - cos(q*Lambda) = 0.
def f_for_q(omega, q):
    return F(omega) - np.cos(q * Lambda)

# -------------------------------
# 4. Root finding for each q in [0, π]
# -------------------------------
# We define a set of q values (only positive branch) and search for ω roots.
q_values = np.linspace(0, np.pi, 300)
omega_max = 6 * np.pi
omega_coarse = np.linspace(0.001, omega_max, 1000)  # avoid omega = 0

band_data = {}
for q in q_values:
    f_vals = f_for_q(omega_coarse, q)
    # Find intervals where the function changes sign
    sign_change_indices = np.where(np.diff(np.sign(f_vals)))[0]
    omega_solutions = []
    for idx in sign_change_indices:
        omega_left = omega_coarse[idx]
        omega_right = omega_coarse[idx + 1]
        try:
            sol = root_scalar(f_for_q, args=(q,), bracket=[omega_left, omega_right], method='brentq')
            if sol.converged:
                omega_solutions.append(sol.root)
        except Exception:
            pass
    band_data[q] = sorted(omega_solutions)

# -------------------------------
# 5. Collect and mirror data
# -------------------------------
q_list = []
omega_list = []
for q, omegas in band_data.items():
    for omega in omegas:
        q_list.append(q)
        omega_list.append(omega)

# Convert to arrays for the positive branch.
q_array = np.array(q_list)
omega_array = np.array(omega_list)

# Mirror the q values to create the negative branch.
q_array_full = np.concatenate([q_array, -q_array])
omega_array_full = np.concatenate([omega_array, omega_array])

# -------------------------------
# 6. Normalize and sort data
# -------------------------------
# Normalize: q_norm = q/(2π), ω_norm = ω/(2π)
q_norm = q_array_full / (2 * np.pi)
omega_norm = omega_array_full / (2 * np.pi)

# Sort by q_norm for a cleaner plot.
sort_idx = np.argsort(q_norm)
q_norm_sorted = q_norm[sort_idx]
omega_norm_sorted = omega_norm[sort_idx]

# -------------------------------
# 7. Plot the band structure
# -------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(q_norm_sorted, omega_norm_sorted, s=10, color='blue', label='Band Structure (Root Finding)')
plt.xlabel(r'Normalized Bloch Wave Vector $q\Lambda/(2\pi)$')
plt.ylabel(r'Normalized Frequency $\omega\Lambda/(2\pi c)$')
plt.title('Band Structure of 1D Photonic Crystal\n(using Root Finding)')
plt.xlim(-0.5, 0.5)
plt.ylim(0, 3)
plt.grid(True)
plt.legend()
plt.show()
