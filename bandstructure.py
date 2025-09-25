import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters (match MATLAB exactly) ---
DELTA = 6.0
PHI1  = 0.0
PHI2  = 0.8
ALPHA = 0.5
BETA  = 0.5
EPSILON0         = 13.0          # set to 13.0+5.0j for the NH case
EPSILON0_TILDE   = 1.0

# Unit-cell lengths: L_total = 2π,  A1 = A2 = L_total/4 = π/2,  F = L_total/2 = π
L_TOTAL = 2*np.pi
L_A     = L_TOTAL/4.0            # = π/2
L_F     = L_TOTAL/2.0            # = π

# Simulation control (unchanged)
OMEGA_MAX   = 0.6
OMEGA_STEPS = 100_000
TOLERANCE   = 1e-7  # |λ|-1 tolerance for accepting a propagating mode

# --- Transfer matrices (1:1 with MATLAB) ---
def get_TA(omega, phi, L, delta, epsilon0):
    """
    Layer A transfer matrix (exact MATLAB formula).
    """
    a  = omega * L
    n1 = np.sqrt(epsilon0 + delta)
    n2 = np.sqrt(epsilon0 - delta)

    u1, v1 = np.cos(n1*a), np.sin(n1*a)
    u2, v2 = np.cos(n2*a), np.sin(n2*a)

    u, v = np.cos(phi), np.sin(phi)

    T = np.array([
        [   u*u*u1 + v*v*u2,            u*v*u1 - u*v*u2,           -1j*u*v*v1/n1 + 1j*u*v*v2/n2,   1j*u*u*v1/n1 + 1j*v*v*v2/n2 ],
        [   u*v*u1 - u*v*u2,            v*v*u1 + u*u*u2,           -1j*v*v*v1/n1 - 1j*u*u*v2/n2,   1j*u*v*v1/n1 - 1j*u*v*v2/n2 ],
        [ -1j*n1*u*v*v1 + 1j*n2*u*v*v2, -1j*n1*v*v*v1 - 1j*n2*u*u*v2,  v*v*u1 + u*u*u2,              -u*v*u1 + u*v*u2           ],
        [  1j*n1*u*u*v1 + 1j*n2*v*v*v2,  1j*n1*u*v*v1 - 1j*n2*u*v*v2, -u*v*u1 + u*v*u2,               u*u*u1 + v*v*u2           ]
    ], dtype=np.complex128)
    return T

def get_TF(omega, L_F, alpha, beta, epsilon0_tilde):
    """
    Layer F transfer matrix (exact MATLAB formula, including the 1/2 factor).
    """
    a  = omega * L_F
    mu = 1.0

    n1 = np.sqrt((epsilon0_tilde + alpha)*(mu + beta))
    n2 = np.sqrt((epsilon0_tilde - alpha)*(mu - beta))
    m1 = np.sqrt((epsilon0_tilde + alpha)/(mu + beta))
    m2 = np.sqrt((epsilon0_tilde - alpha)/(mu - beta))

    u1, v1 = np.cos(n1*a), np.sin(n1*a)
    u2, v2 = np.cos(n2*a), np.sin(n2*a)

    T = np.array([
        [  u1+u2,               1j*(u1-u2),           v1/m1 - v2/m2,        1j*(v1/m1 + v2/m2) ],
        [ -1j*(u1-u2),          u1+u2,               -1j*(v1/m1 + v2/m2),   v1/m1 - v2/m2      ],
        [ -m1*v1 + m2*v2,      -1j*(m1*v1 + m2*v2),   u1+u2,                1j*(u1-u2)        ],
        [  1j*(m1*v1 + m2*v2), -m1*v1 + m2*v2,       -1j*(u1-u2),           u1+u2             ]
    ], dtype=np.complex128)

    return 0.5 * T

# --- Original sweep-ω, solve k from eigenvalues (unchanged) ---
def calculate_band_structure():
    """
    Sweep real ω; for each ω, build M(ω)=T_A1*T_F*T_A2,
    take its eigenvalues λ = e^{ik}, keep those with |λ|≈1,
    and record (k, ω) with k = -i log λ (real part).
    """
    print("Starting calculation...")
    omega_vals = np.linspace(0, OMEGA_MAX, OMEGA_STEPS)

    k_results = []
    omega_results = []

    for omega in omega_vals:
        if omega == 0:
            continue  # unchanged

        # Correct layer thicknesses (match MATLAB)
        TA1 = get_TA(omega, PHI1, L_A, DELTA, EPSILON0)
        TFm = get_TF(omega, L_F,  ALPHA, BETA, EPSILON0_TILDE)
        TA2 = get_TA(omega, PHI2, L_A, DELTA, EPSILON0)

        # Correct multiplication order (match MATLAB: T_A1 * T_F * T_A2)
        M = TA1 @ TFm @ TA2

        eigenvalues = np.linalg.eigvals(M)
        for eig in eigenvalues:
            if np.isclose(np.abs(eig), 1.0, atol=TOLERANCE):
                k = -1j * np.log(eig)
                k_results.append(k.real)
                omega_results.append(omega)

    print("Calculation finished.")
    return k_results, omega_results

def plot_results(k_vals, omega_vals):
    plt.figure(figsize=(8, 6))
    plt.scatter(k_vals, omega_vals, s=1, c='k', alpha=0.5)
    plt.xlabel('k (Bloch Wavenumber)')
    plt.ylabel('ω (Frequency)')
    plt.title('Photonic Crystal Band Structure (Hermitian Case)')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(0, OMEGA_MAX)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    k_points, omega_points = calculate_band_structure()
    plot_results(k_points, omega_points)

