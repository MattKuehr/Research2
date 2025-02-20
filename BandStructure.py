import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define parameters
# -------------------------------
# Layer A parameters
epsilon_a = 4.0
mu_a = 1.0
d_a = 0.4  # normalized thickness (Lambda = d_a + d_b)

# Layer B parameters
epsilon_b = 1.0
mu_b = 1.0
d_b = 0.6

Lambda = d_a + d_b  # unit cell period (Lambda = 1)

# Refractive index and impedance (assuming c = 1)
# k_i = ω * sqrt(ε_i * μ_i)
k_a = lambda omega: omega * np.sqrt(epsilon_a * mu_a)
k_b = lambda omega: omega * np.sqrt(epsilon_b * mu_b)

# Impedance: z_i = sqrt(mu_i/ε_i)
z_a = np.sqrt(mu_a / epsilon_a)  # for layer A, = 0.5
z_b = np.sqrt(mu_b / epsilon_b)  # for layer B, = 1.0

# -------------------------------
# 2. Define the dispersion function F(omega)
# -------------------------------
# The dispersion relation reads:
# cos(q * Lambda) = cos(k_a*d_a)*cos(k_b*d_b) - 0.5*(z_a/z_b + z_b/z_a)*sin(k_a*d_a)*sin(k_b*d_b)
# For our numbers, (z_a/z_b + z_b/z_a) = 0.5 + 2 = 2.5, and half of that is 1.25.
F = lambda omega: (np.cos(k_a(omega)*d_a) * np.cos(k_b(omega)*d_b)
                   - 0.5 * (z_a/z_b + z_b/z_a) * np.sin(k_a(omega)*d_a) * np.sin(k_b(omega)*d_b))

# -------------------------------
# 3. Set frequency range and compute allowed q values
# -------------------------------
# The normalized frequency is ω_norm = (ω*Lambda)/(2π). We wish to see ω_norm from 0 to 3.
# For Lambda = 1 and c = 1, this means ω from 0 to 6π ≈ 18.85.
omega_max = 6 * np.pi
omega_vals = np.linspace(0, omega_max, 20000)

# We will collect the allowed (q, ω) points.
q_plus = []   # positive branch (in radians)
q_minus = []  # negative branch (mirror of positive branch)
omega_allowed = []

# For a real Bloch wave, |F(omega)| must be <= 1.
for omega in omega_vals:
    f_val = F(omega)
    if np.abs(f_val) <= 1.0:
        # Bloch wave vector q from the eigenvalue condition:
        # 2*cos(q*Lambda) = e^(iqΛ) + e^(-iqΛ)
        # We choose q in [0, π/Lambda], then also include the negative branch.
        q_val = np.arccos(f_val)
        q_plus.append(q_val)
        q_minus.append(-q_val)
        omega_allowed.append(omega)

# Convert lists to numpy arrays
q_plus = np.array(q_plus)
q_minus = np.array(q_minus)
omega_allowed = np.array(omega_allowed)

# -------------------------------
# 4. Normalize variables for plotting
# -------------------------------
# Normalized Bloch wave vector: q_norm = (q * Lambda) / (2π)
# Since Lambda = 1, this is just q/(2π)
q_plus_norm = q_plus / (2 * np.pi)
q_minus_norm = q_minus / (2 * np.pi)

# Normalized frequency: omega_norm = (omega * Lambda)/(2π)
omega_norm = omega_allowed / (2 * np.pi)

# -------------------------------
# 5. Plot the band structure
# -------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(q_plus_norm, omega_norm, s=1, color='blue', label='q > 0')
plt.scatter(q_minus_norm, omega_norm, s=1, color='blue', label='q < 0')
plt.xlabel(r'Normalized Bloch Wave Vector $q\Lambda/(2\pi)$')
plt.ylabel(r'Normalized Frequency $\omega\Lambda/(2\pi c)$')
plt.title('Band Structure of 1D Photonic Crystal\n'
          r'($\epsilon_a=4,\ d_a=0.4\Lambda;\ \epsilon_b=1,\ d_b=0.6\Lambda$)')
plt.xlim(-0.5, 0.5)
plt.ylim(0, 3)
plt.grid(True)
plt.legend()
plt.show()

