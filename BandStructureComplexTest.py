#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D PT-Symmetric Photonic Crystal Band Structure via Transfer Matrix Method
-----------------------------------------------------------------------------
Reference:
  [1] K. Ding, Z. Q. Zhang, and C. T. Chan, Phys. Rev. B 92, 235310 (2015) :contentReference[oaicite:1]{index=1}

This script:
  - Accepts a set of permittivities, permeabilities, and layer thicknesses.
  - Enforces that the unit cell length Λ = sum(d_i) = 1.
  - Computes the 2×2 transfer matrix for each individual layer and then the unit cell transfer matrix.
  - Defines the dispersion relation as
         f(ω; q) = ½ Tr[M(ω)] - cos(q·Λ)
    (with Λ = 1) and obtains its derivative symbolically using Sympy.
  - Uses a hybrid search:
       * A dense grid search at q = 0 to seed all branches in the ω complex plane.
       * A continuation method for q > 0 to track the branches.
  - Finds roots using Newton's method for complex functions.
  - Plots the real and imaginary parts of the band structure on dimensionless axes:
         x-axis: (q·Λ)/(2π),  y-axis: (ω·Λ)/(2πc)
    (We use c = 1 so that the results are dimensionless.)
  
Adjust the search ranges and tolerances as necessary.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

##############################################################################
# USER INPUTS and PHYSICAL CONSTANTS
##############################################################################
# Layers: permittivities (ε), permeabilities (μ), and thicknesses (d)
epsilons = [3.8 + 0.1j, 1.0 - 0.1j]    # PT-symmetric: balanced gain/loss
mus      = [1.0,       1.0      ]
ds       = [0.42,      0.58     ]

# Enforce: Unit cell length Λ = sum(ds) == 1
Lambda = sum(ds)
assert abs(Lambda - 1.0) < 1e-12, "Unit cell length (sum of ds) must equal 1."

# We adopt dimensionless units with c = 1 (so that ω is in units where ω/(2π) is the normalized frequency)
c0 = 1.0

# Wave vector q values (only need half the Brillouin zone due to symmetry)
q_min, q_max = 0.0, np.pi  # q is in [0, π]
num_q = 150               # number of q values for continuation

# Newton's method parameters
newton_tol = 1e-8
max_newton_iter = 50
root_tol = 1e-6           # tolerance for accepting a root (|f(ω,q)| < root_tol)
cluster_tol = 1e-3        # tolerance for grouping nearby roots in ω-plane

# Omega search parameters for initialization at q=0 (grid search)
omega_min = 0.05          # minimum ω (avoid ω=0)
omega_max = 10.0 * np.pi    # maximum ω (adjust as needed)
num_omega_re = 80         # number of points along real ω axis
omega_im_offsets = np.linspace(-0.1, 0.1, 5)  # span a small window in imag direction

##############################################################################
# SYMBOLIC SETUP: TRANSFER MATRICES and DISPERSION RELATION
##############################################################################
omega_sym = sp.Symbol('omega', complex=True)

def layer_transfer_matrix_symbolic(omega_val, d_val, eps_val, mu_val):
    """
    Construct the 2×2 symbolic transfer matrix for a single layer.
    Uses: k = (ω/c0)*sqrt(εμ), Z = sqrt(μ/ε)
    """
    k_sym = (omega_val/c0) * sp.sqrt(eps_val*mu_val)
    Z_sym = sp.sqrt(mu_val/eps_val)
    M = sp.Matrix([
         [ sp.cos(k_sym*d_val),  sp.I * sp.sin(k_sym*d_val) / Z_sym ],
         [ sp.I * Z_sym * sp.sin(k_sym*d_val), sp.cos(k_sym*d_val) ]
    ])
    return M

def unit_cell_transfer_matrix_symbolic(omega_val):
    """
    Multiply the layer transfer matrices in sequence.
    The multiplication order follows the physical sequence of layers.
    """
    M_total = sp.eye(2)
    for d_val, eps_val, mu_val in zip(ds, epsilons, mus):
        # Convert numeric parameters to sympy numbers (if necessary)
        M_layer = layer_transfer_matrix_symbolic(omega_val, sp.N(d_val), sp.N(eps_val), sp.N(mu_val))
        M_total = M_layer * M_total
    return M_total

# Define symbolic function f(ω) = ½ Tr(M_total(ω))
M_total_sym = unit_cell_transfer_matrix_symbolic(omega_sym)
f_sym = 0.5 * sp.trace(M_total_sym)
# The full dispersion relation is f(ω) - cos(q) = 0,
# where q is provided as a parameter (numerically) later.

# Differentiate f_sym with respect to ω (symbolically).
df_domega_sym = sp.diff(f_sym, omega_sym)

# Create callable (numpy) functions for f and its derivative.
f_func = sp.lambdify(omega_sym, f_sym, 'numpy')
df_domega_func = sp.lambdify(omega_sym, df_domega_sym, 'numpy')

##############################################################################
# NUMERICAL FUNCTIONS: NEWTON'S METHOD and DISPERSION RELATION EVALUATION
##############################################################################
def dispersion_relation(omega_val, q_val):
    """
    Evaluate the dispersion relation at a given complex ω and q:
       F(ω; q) = ½ Tr(M_total(ω)) - cos(q)
    """
    return f_func(omega_val) - np.cos(q_val)

def dispersion_relation_derivative(omega_val):
    """
    Evaluate the ω-derivative of ½ Tr(M_total(ω)).
    (The derivative of cos(q) is zero as q is treated as a parameter.)
    """
    return df_domega_func(omega_val)

def newton_complex(omega_init, q_val, tol=newton_tol, max_iter=max_newton_iter):
    """
    Solve for ω such that dispersion_relation(ω, q_val) = 0 using Newton's method.
    Uses the symbolic derivative.
    """
    omega_current = omega_init
    for i in range(max_iter):
        f_val = dispersion_relation(omega_current, q_val)
        df_val = dispersion_relation_derivative(omega_current)
        if abs(df_val) < 1e-14:
            break  # Avoid division by very small derivative.
        omega_next = omega_current - f_val / df_val
        if abs(omega_next - omega_current) < tol:
            return omega_next
        omega_current = omega_next
    return omega_current

##############################################################################
# ROOT FINDING: INITIAL GRID SEARCH at q = 0 and CONTINUATION in q
##############################################################################
q_values = np.linspace(q_min, q_max, num_q)
band_data = {}  # dictionary: key = q, value = list of ω roots

# STEP 1: At q = 0, perform a grid search in the complex ω plane.
initial_roots = []
omega_re_vals = np.linspace(omega_min, omega_max, num_omega_re)
for w_re in omega_re_vals:
    for w_im in omega_im_offsets:
        guess = w_re + 1j*w_im
        sol = newton_complex(guess, q_val=0.0)
        if abs(dispersion_relation(sol, 0.0)) < root_tol:
            initial_roots.append(sol)
# Remove near-duplicate solutions.
initial_roots = sorted(initial_roots, key=lambda x: (x.real, x.imag))
unique_roots = []
for root in initial_roots:
    if not unique_roots or abs(root - unique_roots[-1]) > cluster_tol:
        unique_roots.append(root)
band_data[0.0] = unique_roots
print(f"q = 0: Found {len(unique_roots)} solution branches.")

# STEP 2: Continuation for q > 0.
prev_q = 0.0
for q in q_values[1:]:
    roots_prev = band_data[prev_q]
    roots_current = []
    # For each branch from previous q, use it (with a tiny perturbation) as a guess.
    for root_prev in roots_prev:
        for delta in [0.0, 1e-6*1j, -1e-6*1j]:
            guess = root_prev + delta
            sol = newton_complex(guess, q_val=q)
            if abs(dispersion_relation(sol, q)) < root_tol:
                roots_current.append(sol)
    # Remove duplicates for this q value.
    roots_current = sorted(roots_current, key=lambda x: (x.real, x.imag))
    unique_current = []
    for r in roots_current:
        if not unique_current or abs(r - unique_current[-1]) > cluster_tol:
            unique_current.append(r)
    band_data[q] = unique_current
    prev_q = q
    # (Optional: print progress)
    # print(f"q = {q:.4f}: {len(unique_current)} branches.")

##############################################################################
# DATA COLLECTION and SYMMETRY: MIRRORING ABOUT q = 0
##############################################################################
# Collect (q, ω) pairs for q >= 0.
q_list, omega_list = [], []
for q_val, roots in band_data.items():
    for sol in roots:
        q_list.append(q_val)
        omega_list.append(sol)

q_array = np.array(q_list, dtype=float)
omega_array = np.array(omega_list, dtype=complex)

# By PT symmetry, for q < 0, ω(q) = ω*(−q) in the PT-exact phase.
q_array_mirror = -q_array
omega_array_mirror = np.conjugate(omega_array)

# Combine data for the full Brillouin zone.
q_full = np.concatenate([q_array, q_array_mirror])
omega_full = np.concatenate([omega_array, omega_array_mirror])

##############################################################################
# NORMALIZATION AND PLOTTING
##############################################################################
# Normalize wave vector: (q Λ)/(2π)  [with Λ=1 => q/(2π)]
k_norm = q_full / (2.0 * np.pi)
# Normalize frequency: (ω Λ)/(2π c)  [with c=1 and Λ=1 => ω/(2π)]
omega_norm = omega_full / (2.0 * np.pi)

# Sort the data for clarity in plotting.
sort_idx = np.argsort(k_norm.real)
k_norm_sorted = k_norm[sort_idx]
omega_norm_sorted = omega_norm[sort_idx]

# Separate real and imaginary parts.
omega_real = np.real(omega_norm_sorted)
omega_imag = np.imag(omega_norm_sorted)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
ax1.scatter(k_norm_sorted, omega_real, s=8, c='b', alpha=0.7, label='Re($\\omega$)')
ax1.set_ylabel(r'Re($\omega \Lambda / 2\pi c$)')
ax1.set_title("1D PT-Symmetric Photonic Crystal: Real Part of Band Structure")
ax1.grid(True)
ax1.legend()

ax2.scatter(k_norm_sorted, omega_imag, s=8, c='r', alpha=0.7, label='Im($\\omega$)')
ax2.set_xlabel(r'$q \Lambda / 2\pi$')
ax2.set_ylabel(r'Im($\omega \Lambda / 2\pi c$)')
ax2.set_title("1D PT-Symmetric Photonic Crystal: Imaginary Part of Band Structure")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

