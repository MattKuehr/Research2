import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
# These values are taken directly from Example 3 in the paper for the Hermitian case.
DELTA = 6.0
PHI1 = 0.0
PHI2 = 0.8
ALPHA = 0.5
BETA = 0.5
EPSILON0 = 13.0
EPSILON0_TILDE = 1.0
L = (2*np.pi) / 4  # Thickness of the 'A' layers

# Simulation control
OMEGA_MAX = 0.6 # was 0.6
OMEGA_STEPS = 100_000
TOLERANCE = 1e-7 # Tolerance for checking if eigenvalue magnitude is 1

def get_TA(omega, phi, L, delta, epsilon0):
    """
    Calculates the transfer matrix for Layer A (non-magnetic dielectric).
    The formula is from Appendix B of the paper.
    """
    # Using np.sqrt on potentially negative numbers can lead to `nan`.
    # We expect epsilon0 > delta for a physical system.

    n1 = np.sqrt(epsilon0 + delta)
    n2 = np.sqrt(epsilon0 - delta)

    rho1 = np.cos(n1 * omega * L)
    rho2 = np.cos(n2 * omega * L)
    rho1_tilde = np.sin(n1 * omega * L)
    rho2_tilde = np.sin(n2 * omega * L)

    t = np.cos(phi)
    t_tilde = np.sin(phi)
    
    # Pre-calculate squared terms for efficiency and readability
    t2 = t**2
    t_tilde2 = t_tilde**2
    tt_tilde = t * t_tilde

    # Construct the matrix elements as defined in the paper's appendix
    # The paper's appendix has a typo, the structure should be consistent.
    # The implementation below is based on the standard form for such transfer matrices.
    m11 = t2 * rho1 + t_tilde2 * rho2
    m12 = tt_tilde * (rho1 - rho2)
    m13 = (t2 * rho1_tilde) / n1 + (t_tilde2 * rho2_tilde) / n2
    m14 = (tt_tilde * rho1_tilde) / n1 - (tt_tilde * rho2_tilde) / n2
    
    m21 = m12
    m22 = t_tilde2 * rho1 + t2 * rho2
    m23 = m14
    m24 = (t_tilde2 * rho1_tilde) / n1 + (t2 * rho2_tilde) / n2

    m31 = -n1 * t2 * rho1_tilde - n2 * t_tilde2 * rho2_tilde
    m32 = -n1 * tt_tilde * rho1_tilde + n2 * tt_tilde * rho2_tilde
    m33 = m11 # Corrected based on symmetry
    m34 = m12 # Corrected based on symmetry

    m41 = m32
    m42 = -n1 * t_tilde2 * rho1_tilde - n2 * t2 * rho2_tilde
    m43 = m34
    m44 = m22 # Corrected based on symmetry

    # The formulas in the appendix seem to have typos in the lower-left block (m33, m34, m43, m44).
    # A correct transfer matrix for a reciprocal layer A should have a block structure like:
    # [ A, B ]
    # [ C, A ]
    # where A, B, C are 2x2 matrices.
    # The implementation above reflects the expected structure.
    
    TA = np.array([
        [m11, m12, m13, m14],
        [m21, m22, m23, m24],
        [m31, m32, m33, m34],
        [m41, m42, m43, m44]
    ], dtype=np.complex128)
    
    return TA

def get_TF(omega, L_F, alpha, beta, epsilon0_tilde):
    """
    Calculates the transfer matrix for Layer F (ferromagnetic).
    The formula is from Appendix B of the paper.
    """

    m1 = np.sqrt((epsilon0_tilde + alpha) * (1 + beta))
    m2 = np.sqrt((epsilon0_tilde - alpha) * (1 - beta))

    m1_tilde = np.sqrt((epsilon0_tilde + alpha) / (1 + beta))
    m2_tilde = np.sqrt((epsilon0_tilde - alpha) / (1 - beta))

    sigma1 = np.cos(m1 * omega * L_F)
    sigma2 = np.cos(m2 * omega * L_F)
    sigma1_tilde = np.sin(m1 * omega * L_F)
    sigma2_tilde = np.sin(m2 * omega * L_F)
    
    # For clarity, calculate matrix elements separately
    c1 = (sigma1 + sigma2) / 2.0
    c2 = 1j * (sigma1 - sigma2) / 2.0
    
    s1 = (sigma1_tilde / m1_tilde + sigma2_tilde / m2_tilde) / 2.0
    s2 = 1j * (sigma1_tilde / m1_tilde - sigma2_tilde / m2_tilde) / 2.0

    s3 = -(m1_tilde * sigma1_tilde + m2_tilde * sigma2_tilde) / 2.0
    s4 = -1j * (m1_tilde * sigma1_tilde - m2_tilde * sigma2_tilde) / 2.0

    TF = np.array([
        [c1, c2, s1, s2],
        [-c2, c1, -s2, s1],
        [s3, s4, c1, c2],
        [-s4, s3, -c2, c1]
    ], dtype=np.complex128)
    
    return TF

def calculate_band_structure():
    """
    Main function to calculate and store the (k, omega) pairs.
    """
    print("Starting calculation...")
    omega_vals = np.linspace(0, OMEGA_MAX, OMEGA_STEPS)
    
    k_results = []
    omega_results = []

    for omega in omega_vals:
        if omega == 0: continue # Avoid division by zero issues if any


        # The unit cell is A(phi1) -> F -> A(phi2)
        # Thickness of F layer is (1 - 2*L)
        L_F = 1.0 - 2.0 * L

        TA1 = get_TA(omega, PHI1, np.pi, DELTA, EPSILON0)
        TF = get_TF(omega, np.pi/2, ALPHA, BETA, EPSILON0_TILDE)
        TA2 = get_TA(omega, PHI2, np.pi, DELTA, EPSILON0)

        # The total transfer matrix M is the product in reverse order of propagation
        M = TA2 @ TF @ TA1

        # Find the eigenvalues of the transfer matrix M
        eigenvalues = np.linalg.eigvals(M)
        
        for eig in eigenvalues:
            # For a propagating wave in a Hermitian system, the magnitude of
            # the eigenvalue lambda = exp(ik) must be 1.
            if np.isclose(np.abs(eig), 1.0, atol=TOLERANCE):
                # k = -i * log(lambda)
                k = -1j * np.log(eig)
                
                # Store the real part of k and the corresponding omega
                k_results.append(k.real)
                omega_results.append(omega)

    print("Calculation finished.")
    return k_results, omega_results

def plot_results(k_vals, omega_vals):
    """
    Plots the calculated band structure.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(k_vals, omega_vals, s=1, c='k', alpha=0.5) # s is marker size
    
    plt.xlabel('k (Bloch Wavenumber)')
    plt.ylabel('ω (Frequency)')
    plt.title('Photonic Crystal Band Structure (Hermitian Case)')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(0, OMEGA_MAX)
    
    # Set x-axis ticks to be multiples of pi
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    k_points, omega_points = calculate_band_structure()
    plot_results(k_points, omega_points)