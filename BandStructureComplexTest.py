import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from functools import partial

# Unit‐cell dimensions
d_a, d_b = 0.42, 0.58
Λ = d_a + d_b

# Permeability μ
μ_a = μ_b = 1.0

# Bloch‐q sample
Nq = 200 
q_vals = np.linspace(-np.pi/Λ, np.pi/Λ, Nq)

# How far up in frequency (ωΛ/2πc) to look
ω_max = 3 * 2*np.pi   

# Range of imaginary magnitudes Γ to sweep
Gamma_list = [0.1, 0.5, 1.0, 2.0]

def layer_matrix(ω, d, ε, μ):
    # 2×2 propagation matrix over thickness d
    k = ω * np.sqrt(ε*μ)
    Z = np.sqrt(μ/ε)
    cd = np.cos(k*d)
    sd = np.sin(k*d)
    return np.array([[    cd, 1j*sd/Z],
                     [1j*Z*sd,    cd]], dtype=complex)

def M_unit(ω, Γ):
    # four sublayers in the two layer unit cell
    eps = [
      1 - 1j*Γ,         # B(loss)
      3.8 + 1j*Γ,       # A(gain)
      3.8 - 1j*Γ,       # A(loss)
      1 + 1j*Γ          # B(gain)
    ]
    d_half = [d_b/2, d_a/2, d_a/2, d_b/2]
    μs     = [1,1,1,1]

    M = np.eye(2, dtype=complex)
    for d, ε, μ in zip(d_half, eps, μs):
        M = layer_matrix(ω, d, ε, μ) @ M
    return M

def dispersion_vec(ω_vec, q, Γ):
    ω = ω_vec[0] + 1j*ω_vec[1]
    M = M_unit(ω, Γ)
    half_trace = 0.5 * np.trace(M)
    res = half_trace - np.cos(q*Λ)
    return [res.real, res.imag]

# frequency search grid for initial guesses (real part)
ω_re_init = np.linspace(0.01, ω_max, 60) 
# small ± imag guesses to seed PT‑broken/‑exact
ω_im_init = [+0.1, -0.1]

# store solutions: shape (len(Gamma_list), nbands, Nq)
solutions = []

for Γ in Gamma_list:
    all_bands_real = []
    all_bands_imag = []

    for q in q_vals:
        # build initial guess list of (ω_r, ω_i)
        guesses = []
        for ωr in ω_re_init:
            for ωi in ω_im_init:
                guesses.append([ωr, ωi])

        roots = []
        for guess in guesses:
            try:
                sol, info, ier, mesg = fsolve(
                    lambda w: dispersion_vec(w, q, Γ),
                    guess, full_output=True, xtol=1e-12, maxfev=200 
                )
                if ier==1:   # converged
                    ω_root = sol[0] + 1j*sol[1]
                    # keep only roots in [0,ω_max] real part
                    if 0 <= sol[0] <= ω_max:
                        roots.append(ω_root)
            except:
                pass

        # dedupe by clustering very close roots
        uniq = []
        tol = 1e-3 
        for ω in sorted(roots, key=lambda x: x.real):
            if all(abs(ω - u) > tol for u in uniq):
                uniq.append(ω)

        # sort bands by real part
        uniq.sort(key=lambda x: x.real)
        # collect real/imag lists, padding to same length
        real_parts = [u.real for u in uniq]
        imag_parts = [u.imag for u in uniq]

        all_bands_real.append(real_parts)
        all_bands_imag.append(imag_parts)

    # transpose: want bands × q arrays
    bands_real = list(map(list, zip(*all_bands_real)))
    bands_imag = list(map(list, zip(*all_bands_imag)))
    solutions.append((bands_real, bands_imag))

for idx, Γ in enumerate(Gamma_list):
    bands_real, bands_imag = solutions[idx]

    plt.figure(figsize=(6,4))
    for band in bands_real[:5]:   # plot first 5 bands real parts
        plt.plot(q_vals*Λ/(2*np.pi), np.array(band)*Λ/(2*np.pi), '-o', ms=2)
    plt.title(f'Re ωΛ/2πc, Γ={Γ}')
    plt.xlabel('qΛ/2π'); plt.ylabel('Re(ωΛ/2πc)')
    plt.grid(True)

    plt.figure(figsize=(6,4))
    for band in bands_imag[:5]:
        plt.plot(q_vals*Λ/(2*np.pi), np.array(band)*Λ/(2*np.pi), '-o', ms=2)
    plt.title(f'Im ωΛ/2πc, Γ={Γ}')
    plt.xlabel('qΛ/2π'); plt.ylabel('Im(ωΛ/2πc)')
    plt.grid(True)

plt.show()
