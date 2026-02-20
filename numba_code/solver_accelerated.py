import numpy as np
from numba import njit, prange
from matrices_numba import get_TA, get_TF
from multiprocessing import Pool, cpu_count
from scipy.optimize import fsolve
from functools import partial

@njit(parallel=True)
def calculate_band_structure_numba(
        DELTA, PHI1, PHI2, ALPHA, BETA, EPSILON0, EPSILON0_TILDE,
        L_TOTAL, L_A, L_F, OMEGA_MAX, OMEGA_STEPS, TOLERANCE
):
    """
    Numba-accelerated band structure calculation with parallel loop.
    """
    omega_vals = np.linspace(0, OMEGA_MAX, OMEGA_STEPS)
    
    # Pre-allocate with estimated max size
    max_size = OMEGA_STEPS * 4  # Estimate: 4 eigenvalues per omega
    k_results = np.zeros(max_size)
    omega_results = np.zeros(max_size)
    counts = np.zeros(OMEGA_STEPS, dtype=np.int32)
    
    for i in prange(1, len(omega_vals)):  # Skip omega=0
        omega = omega_vals[i]
        
        TA1 = get_TA(omega, PHI1, L_A, DELTA, EPSILON0)
        TFm = get_TF(omega, L_F, ALPHA, BETA, EPSILON0_TILDE)
        TA2 = get_TA(omega, PHI2, L_A, DELTA, EPSILON0)
        
        M = TA1 @ TFm @ TA2
        eigenvalues = np.linalg.eigvals(M)
        
        local_count = 0
        for eig in eigenvalues:
            if np.abs(np.abs(eig) - 1.0) < TOLERANCE:
                local_count += 1
        
        counts[i] = local_count
    
    # Second pass to fill results (serial, but fast)
    idx = 0
    for i in range(1, len(omega_vals)):
        if counts[i] > 0:
            omega = omega_vals[i]
            
            TA1 = get_TA(omega, PHI1, L_A, DELTA, EPSILON0)
            TFm = get_TF(omega, L_F, ALPHA, BETA, EPSILON0_TILDE)
            TA2 = get_TA(omega, PHI2, L_A, DELTA, EPSILON0)
            
            M = TA1 @ TFm @ TA2
            eigenvalues = np.linalg.eigvals(M)
            
            for eig in eigenvalues:
                if np.abs(np.abs(eig) - 1.0) < TOLERANCE:
                    k = -1j * np.log(eig)
                    k_results[idx] = k.real
                    omega_results[idx] = omega
                    idx += 1
    
    return k_results[:idx], omega_results[:idx]


@njit
def compute_M_matrix(omega_real, omega_imag, PHI1, PHI2, L_A, L_F, 
                     DELTA, EPSILON0, ALPHA, BETA, EPSILON0_TILDE):
    """
    Numba-compiled function to compute M matrix for given omega.
    Returns the full M matrix.
    """
    omega = omega_real + 1j * omega_imag
    
    TA1 = get_TA(omega, PHI1, L_A, DELTA, EPSILON0)
    TF = get_TF(omega, L_F, ALPHA, BETA, EPSILON0_TILDE)
    TA2 = get_TA(omega, PHI2, L_A, DELTA, EPSILON0)
    
    M = TA1 @ TF @ TA2
    return M


@njit
def dispersion_function_numba(omega_real, omega_imag, q, L_total,
                              PHI1, PHI2, L_A, L_F, DELTA, EPSILON0,
                              ALPHA, BETA, EPSILON0_TILDE):
    """
    Numba-compiled dispersion function evaluation.
    Returns [real, imag] parts of det(M - exp(iqL)I).
    """
    M = compute_M_matrix(omega_real, omega_imag, PHI1, PHI2, L_A, L_F,
                         DELTA, EPSILON0, ALPHA, BETA, EPSILON0_TILDE)
    
    # Compute M - exp(iqL)I
    phase = np.exp(1j * q * L_total)
    M_shifted = M.copy()
    for i in range(4):
        M_shifted[i, i] -= phase
    
    # Compute determinant
    det = np.linalg.det(M_shifted)
    
    return np.array([det.real, det.imag])

def dispersion_function_wrapper(omega_vec, q, params):
    """
    Wrapper for scipy.fsolve - calls Numba function.
    """
    result = dispersion_function_numba(
        omega_vec[0], omega_vec[1], q, params['L_TOTAL'],
        params['PHI1'], params['PHI2'], params['L_A'], params['L_F'],
        params['DELTA'], params['EPSILON0'], params['ALPHA'], 
        params['BETA'], params['EPSILON0_TILDE']
    )
    return result

def process_single_q(q, omega_re_guesses, omega_im_guesses, params, 
                     omega_max, tolerance):
    """
    Process a single q value - finds all omega solutions for this q.
    Now uses Numba-accelerated dispersion function.
    """
    # Pre-filter initial guesses with Numba
    good_guesses = []
    for omega_re, omega_im in zip(omega_re_guesses, omega_im_guesses):
        f_val = dispersion_function_numba(
            omega_re, omega_im, q, params['L_TOTAL'],
            params['PHI1'], params['PHI2'], params['L_A'], params['L_F'],
            params['DELTA'], params['EPSILON0'], params['ALPHA'],
            params['BETA'], params['EPSILON0_TILDE']
        )
        f_mag = np.sqrt(f_val[0]**2 + f_val[1]**2)
        if f_mag < 1e4:
            good_guesses.append([omega_re, omega_im])
    
    # Solve from each good guess
    omega_solutions = []
    for omega0 in good_guesses:
        try:
            sol = fsolve(dispersion_function_wrapper, omega0, args=(q, params),
                       xtol=1e-10, full_output=True)
            omega_vec, infodict, ier, msg = sol
            
            if ier == 1:  # Converged
                omega = omega_vec[0] + 1j * omega_vec[1]
                
                # Verify solution quality with Numba
                f_check = dispersion_function_numba(
                    omega.real, omega.imag, q, params['L_TOTAL'],
                    params['PHI1'], params['PHI2'], params['L_A'], params['L_F'],
                    params['DELTA'], params['EPSILON0'], params['ALPHA'],
                    params['BETA'], params['EPSILON0_TILDE']
                )
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
        
        return [(omega, q) for omega in unique_omega]
    
    return []

def calculate_nonhermitian_bands_numba(
    DELTA, PHI1, PHI2, ALPHA, BETA, EPSILON0, EPSILON0_TILDE,
    L_TOTAL, L_A, L_F, OMEGA_MAX, OMEGA_STEPS, TOLERANCE
):
    """
    Calculate non-Hermitian band structure by sweeping q and solving for ω.
    Uses parallel processing + Numba acceleration.
    """
    print("Calculating non-Hermitian band structure...")
    
    # Parameters
    Nq = 1_000
    q_vec = np.linspace(-0.5, 0.5, Nq)
    
    # Initial guesses
    omega_max = 0.6
    Nk = 101 # was 101
    omega_re_samples = np.linspace(0, omega_max, Nk)
    omega_im_samples = np.array([0.1, -0.1, 0.05, -0.05])
    omega_re_grid, omega_im_grid = np.meshgrid(omega_re_samples, omega_im_samples)
    omega_re_guesses = omega_re_grid.flatten()
    omega_im_guesses = omega_im_grid.flatten()
    
    L_total = L_A * 2 + L_F
    
    params = {
        'PHI1': PHI1, 'PHI2': PHI2,
        'L_A': L_A, 'L_F': L_F,
        'L_TOTAL': L_total,
        'DELTA': DELTA,
        'EPSILON0': EPSILON0,
        'EPSILON0_TILDE': EPSILON0_TILDE,
        'ALPHA': ALPHA, 'BETA': BETA
    }
    
    tolerance = 5e-4
    
    # Warm up Numba compilation with a test call
    print("Warming up Numba JIT compilation...")
    _ = dispersion_function_numba(0.1, 0.01, 0.0, L_total,
                                  PHI1, PHI2, L_A, L_F, DELTA, EPSILON0,
                                  ALPHA, BETA, EPSILON0_TILDE)
    print("Compilation complete. Starting parallel computation...")
    
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