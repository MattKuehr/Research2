import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.optimize import fsolve


DELTA = 6.0
PHI1  = 0.0
PHI2  = 0.8
ALPHA = 0.5
BETA  = 0.5
EPSILON0         = 13.0 + 5.0j # Hermitian case with imaginary component
EPSILON0_TILDE   = 1.0

# Unit-cell lengths: L_total = 2π,  A1 = A2 = L_total/4 = π/2,  F = L_total/2 = π
L_TOTAL = 2*np.pi
L_A     = L_TOTAL/4.0            # = π/2
L_F     = L_TOTAL/2.0            # = π

# Simulation control (unchanged)
OMEGA_MAX   = 0.6
OMEGA_STEPS = 100_000

TOLERANCE   = 1e-6  # |λ|-1 tolerance for accepting a propagating mode

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
            continue

        TA1 = get_TA(omega, PHI1, L_A, DELTA, EPSILON0)
        TFm = get_TF(omega, L_F,  ALPHA, BETA, EPSILON0_TILDE)
        TA2 = get_TA(omega, PHI2, L_A, DELTA, EPSILON0)

        M = TA1 @ TFm @ TA2

        eigenvalues = np.linalg.eigvals(M)
        for eig in eigenvalues:
            if np.isclose(np.abs(eig), 1.0, atol=TOLERANCE):
                k = -1j * np.log(eig)
                k_results.append(k.real)
                omega_results.append(omega)

    print("Calculation finished.")
    return k_results, omega_results


def dispersion_function(omega_vec, q, params):
    """
    Dispersion relation: det(M(ω) - e^(iq·L)·I) = 0
    """
    omega = omega_vec[0] + 1j * omega_vec[1]
    
    TA1 = get_TA(omega, params['PHI1'], params['L_A'], 
                 params['DELTA'], params['EPSILON0'])
    TF = get_TF(omega, params['L_F'], params['ALPHA'], 
                params['BETA'], params['EPSILON0_TILDE'])
    TA2 = get_TA(omega, params['PHI2'], params['L_A'], 
                 params['DELTA'], params['EPSILON0'])
    
    M = TA1 @ TF @ TA2
    
    # CRITICAL FIX: Include L (total unit cell length)
    L = params['L_A'] * 2 + params['L_F']  # L_A1 + L_A2 + L_F = 2*L_A + L_F
    f = np.linalg.det(M - np.exp(1j * q * L) * np.eye(4))
    
    return [f.real, f.imag]

def process_single_q(q, omega_re_guesses, omega_im_guesses, params, 
                     omega_max, tolerance):
    """
    Process a single q value - finds all omega solutions for this q.
    This function will be called in parallel.
    """
    # Pre-filter initial guesses
    good_guesses = []
    for omega_re, omega_im in zip(omega_re_guesses, omega_im_guesses):
        f_val = dispersion_function([omega_re, omega_im], q, params)
        f_mag = np.sqrt(f_val[0]**2 + f_val[1]**2)
        if f_mag < 1e4:
            good_guesses.append([omega_re, omega_im])
    
    # Solve from each good guess
    omega_solutions = []
    for omega0 in good_guesses:
        try:
            sol = fsolve(dispersion_function, omega0, args=(q, params),
                       xtol=1e-10, full_output=True)
            omega_vec, infodict, ier, msg = sol
            
            if ier == 1:  # Converged
                omega = omega_vec[0] + 1j * omega_vec[1]
                
                # Verify solution quality
                f_check = dispersion_function([omega.real, omega.imag], q, params)
                f_check_mag = np.sqrt(f_check[0]**2 + f_check[1]**2)
                
                # Accept if truly a root and in valid range
                if (f_check_mag < 1e-6 and 
                    omega.real >= -1e-10 and
                    omega.real <= omega_max and
                    omega.imag >= -0.15 and
                    omega.imag <= 0.05):
                    omega_solutions.append(omega)
        except:
            continue
    
    # Sort and remove duplicates
    if omega_solutions:
        omega_solutions = sorted(omega_solutions, key=lambda x: (x.real, x.imag))
        unique_omega = [omega_solutions[0]]
        for omega in omega_solutions[1:]:
            if np.abs(omega - unique_omega[-1]) > tolerance:
                unique_omega.append(omega)
        
        # Return list of (omega, q) tuples
        return [(omega, q) for omega in unique_omega]
    
    return []

def calculate_nonhermitian_bands():
    """
    Calculate non-Hermitian band structure by sweeping q and solving for ω.
    Uses parallel processing for speed.
    """
    print("Calculating non-Hermitian band structure...")
    
    # Parameters
    Nq = 2_000
    q_vec = np.linspace(-0.5, 0.5, Nq)
    
    # Initial guesses
    omega_max = 0.6
    Nk = 101
    omega_re_guesses = np.concatenate([
        np.linspace(0, omega_max, Nk),
        np.linspace(0, omega_max, Nk),
        np.linspace(0, omega_max, Nk),
        np.linspace(0, omega_max, Nk)
    ])
    omega_im_guesses = np.concatenate([
        0.1 * np.ones(Nk),
        -0.1 * np.ones(Nk),
        0.05 * np.ones(Nk),
        -0.05 * np.ones(Nk)
    ])
    
    params = {
        'PHI1': PHI1, 'PHI2': PHI2,
        'L_A': L_A, 'L_F': L_F,
        'DELTA': DELTA,
        'EPSILON0': EPSILON0,
        'EPSILON0_TILDE': EPSILON0_TILDE,
        'ALPHA': ALPHA, 'BETA': BETA
    }
    
    tolerance = 5e-4
    
    # Create partial function with fixed parameters
    worker = partial(process_single_q, 
                    omega_re_guesses=omega_re_guesses,
                    omega_im_guesses=omega_im_guesses,
                    params=params,
                    omega_max=omega_max,
                    tolerance=tolerance)
    
    # Parallel processing
    num_processes = cpu_count()
    print(f"Using {num_processes} CPU cores")
    
    with Pool(num_processes) as pool:
        # Map each q value to a worker process
        results = pool.map(worker, q_vec)
    
    # Flatten results
    all_omega = []
    all_q = []
    for result in results:
        for omega, q in result:
            all_omega.append(omega)
            all_q.append(q)
    
    all_omega = np.array(all_omega)
    all_q = np.array(all_q)
    
    print(f"Found {len(all_omega)} band points\n")
    return all_omega, all_q

def plot_nonhermitian_bands(omega_vals, q_vals):
    """Plot band structure in complex omega plane (reproducing Figure 1 right panel)."""
    plt.figure(figsize=(8, 6))
    
    plt.scatter(omega_vals.real, omega_vals.imag, s=1, c='k', alpha=0.5)
    plt.xlabel('Re(ω)', fontsize=12)
    plt.ylabel('Im(ω)', fontsize=12)
    plt.title('Non-Hermitian Photonic Crystal Band Structure', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(y=0, color='gray', linewidth=0.5)
    plt.axvline(x=0, color='gray', linewidth=0.5)
    
    # Set limits to match Figure 1
    plt.xlim(0, 0.6)
    plt.ylim(-0.12, 0)
    
    plt.tight_layout()
    plt.show()

# Plot results (Hermitian Case)
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
    # k_points, omega_points = calculate_band_structure()
    # plot_results(k_points, omega_points)
    omega_vals, q_vals = calculate_nonhermitian_bands()
    plot_nonhermitian_bands(omega_vals, q_vals)