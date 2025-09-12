# bands_4x4.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Iterable, Optional
from numpy.linalg import det, eigvals
from scipy.linalg import expm
from scipy.optimize import root, root_scalar

# ============================================================
# Config: numerical guards + defaults
# ============================================================

class ExpmGuard:
    max_power: float = 700.0   # refuse exp if ||A d|| too large
    max_norm:  float = 50.0    # substep if ||A d|| > max_norm
    substep_cap: int = 16

class Rooting:
    real_xtol: float = 1e-10
    complex_tol: float = 1e-10
    complex_maxfev: int = 2000
    complex_method: str = "hybr"  # or "df-sane"

GUARD = ExpmGuard()
ROOTS = Rooting()

# ============================================================
# Core algebra: 4x4 first-order system at normal incidence
# ============================================================

J2 = np.array([[0.0, 1.0],
               [-1.0, 0.0]], dtype=np.complex128)

def layer_generator_matrix(omega: complex,
                           eps_t: np.ndarray,
                           mu_t: np.ndarray) -> np.ndarray:
    """
    Build the 4x4 generator A(ω):
        d/dz Psi = A(ω) Psi,  Psi = [Ex, Ey, Hx, Hy]^T.
    Correct tensor order:
        dE/dz = i ω (μ_t J) H
        dH/dz = -i ω (ε_t J) E
    so  A(ω) = i ω [[0, μ_t @ J], [ -ε_t @ J, 0]].
    """
    Z = np.zeros((2, 2), dtype=np.complex128)
    # *** FIXED ORDERING: μ_t @ J,  ε_t @ J  (not J @ μ_t / J @ ε_t) ***
    A11 = Z
    A12 = mu_t @ J2
    A21 = - eps_t @ J2
    A22 = Z
    A = np.block([[A11, A12],
                  [A21, A22]])
    return 1j * omega * A

def _safe_expm(Ad: np.ndarray,
               max_norm: float = GUARD.max_norm,
               max_power: float = GUARD.max_power,
               substep_cap: int = GUARD.substep_cap) -> Optional[np.ndarray]:
    nrm = np.linalg.norm(Ad, ord=2)
    if not np.isfinite(nrm) or nrm > max_power:
        return None
    N = 1
    if nrm > max_norm:
        N = min(substep_cap, int(np.ceil(nrm / max_norm)))
    try:
        if N == 1:
            return expm(Ad)
        T_step = expm(Ad / N)
        T = np.eye(Ad.shape[0], dtype=Ad.dtype)
        for _ in range(N):
            T = T_step @ T
        return T
    except Exception:
        return None

def layer_transfer_matrix(omega: complex, d: float,
                          eps_t: np.ndarray,
                          mu_t: np.ndarray) -> np.ndarray:
    A = layer_generator_matrix(omega, eps_t, mu_t)
    T = _safe_expm(A * d)
    if T is None:
        raise FloatingPointError("exp(A d) overflow/invalid")
    return T

# ============================================================
# Tensor builders (in-plane 2x2 blocks)
# ============================================================

def rot2(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def rotate_2x2(M: np.ndarray, phi: float) -> np.ndarray:
    R = rot2(phi)
    return R @ M @ R.T.conj()

def eps_anisotropic(eps_par: complex, eps_perp: complex, phi: float = 0.0) -> np.ndarray:
    M0 = np.array([[eps_par, 0.0],
                   [0.0,     eps_perp]], dtype=np.complex128)
    return rotate_2x2(M0, phi)

def mu_anisotropic(mu_par: complex, mu_perp: complex, phi: float = 0.0) -> np.ndarray:
    M0 = np.array([[mu_par, 0.0],
                   [0.0,    mu_perp]], dtype=np.complex128)
    return rotate_2x2(M0, phi)

def eps_gyrotropic(eps_iso: complex, g_e: complex) -> np.ndarray:
    return np.array([[eps_iso, 1j*g_e],
                     [-1j*g_e, eps_iso]], dtype=np.complex128)

def mu_gyrotropic(mu_iso: complex, g_m: complex) -> np.ndarray:
    return np.array([[mu_iso, 1j*g_m],
                     [-1j*g_m, mu_iso]], dtype=np.complex128)

# ============================================================
# Layers and unit cell
# ============================================================

class Layer:
    def __init__(self, thickness: float,
                 eps_t: np.ndarray,
                 mu_t: np.ndarray,
                 name: str = ""):
        self.d = float(thickness)
        self.eps_t = np.asarray(eps_t, dtype=np.complex128)
        self.mu_t  = np.asarray(mu_t,  dtype=np.complex128)
        self.name = name

    def transfer(self, omega: complex) -> np.ndarray:
        return layer_transfer_matrix(omega, self.d, self.eps_t, self.mu_t)

class UnitCell:
    def __init__(self, layers: Iterable[Layer]):
        self.layers = list(layers)

    def M(self, omega: complex) -> np.ndarray:
        """
        Unit-cell transfer matrix from left to right.
        If layers = [L1, L2, L3], then M = T_L3 T_L2 T_L1.
        """
        M = np.eye(4, dtype=np.complex128)
        for L in self.layers:
            M = L.transfer(omega) @ M
        return M

    def floquet_eigs(self, omega: complex) -> np.ndarray:
        return eigvals(self.M(omega))

    def bloch_k_from_varpi(self, varpi: complex) -> complex:
        return -1j * np.log(varpi)

# ============================================================
# Dispersion residual & solvers
# ============================================================

def dispersion_residual(omega_complex: complex, k: float, cell: UnitCell) -> complex:
    varpi = np.exp(1j * k)
    try:
        M = cell.M(omega_complex)
        return det(M - varpi * np.eye(4, dtype=np.complex128))
    except FloatingPointError:
        w = omega_complex
        return (1e12 + 1e12j) * (1 + 0.1 * (w.real + 1j*w.imag))

def solve_omega_real_for_k(cell: UnitCell, k: float,
                           bracket: Tuple[float, float],
                           tol: float = ROOTS.real_xtol,
                           maxiter: int = 100) -> float:
    def f_re(om: float) -> float:
        return float(np.real(dispersion_residual(om + 0j, k, cell)))
    a, b = bracket
    sol = root_scalar(f_re, bracket=(a, b), xtol=tol, maxiter=maxiter, method='brentq')
    if not sol.converged:
        raise RuntimeError("Real-ω brentq failed.")
    return float(sol.root)

def solve_omega_complex_for_k(cell: UnitCell, k: float,
                              omega0: complex,
                              tol: float = ROOTS.complex_tol,
                              maxfev: int = ROOTS.complex_maxfev,
                              method: str = ROOTS.complex_method) -> complex:
    w0 = np.array([np.real(omega0), np.imag(omega0)], dtype=float)
    def F(wvec: np.ndarray) -> np.ndarray:
        w = wvec[0] + 1j*wvec[1]
        val = dispersion_residual(w, k, cell)
        return np.array([np.real(val), np.imag(val)], dtype=float)
    if method == "hybr":
        sol = root(F, w0, method="hybr", options={"xtol": tol, "maxfev": maxfev})
    else:
        sol = root(F, w0, method="df-sane", options={"fatol": tol, "maxiter": maxfev})
    if not sol.success:
        raise RuntimeError(f"Complex-ω solve failed ({method}): {sol.message}")
    return sol.x[0] + 1j*sol.x[1]

def ks_for_omega(cell: UnitCell, omega: complex) -> np.ndarray:
    varpis = cell.floquet_eigs(omega)
    return -1j * np.log(varpis)

# ============================================================
# Eq.(15) A–F–A unit cell helper
# ============================================================

def make_cell_eq15(delta: complex, phi1: float, phi2: float,
                   alpha: complex, beta: complex,
                   eps0: complex, eps0_tilde: complex,
                   L: float = 1/3, mu_iso: complex = 1.0) -> UnitCell:
    """
    A–F–A unit cell (thicknesses L, 1−2L, L), 0 < L < 1/2.
    ε_A = R(φ) diag(ε0+δ, ε0−δ) R(φ)^*
    ε_F = [[ε̃0, iα], [−iα, ε̃0]]
    μ_F = [[μ,  iβ], [−iβ, μ]]
    μ_A is taken isotropic (μ I) as in the paper’s setup.
    """
    if not (0.0 < L < 0.5):
        raise ValueError("Choose L in (0, 0.5) so that F-layer thickness 1−2L is positive.")

    # A-layers
    eps_par, eps_perp = eps0 + delta, eps0 - delta
    epsA1 = eps_anisotropic(eps_par, eps_perp, phi1)
    epsA2 = eps_anisotropic(eps_par, eps_perp, phi2)
    muA   = mu_anisotropic(mu_iso, mu_iso, 0.0)

    # F-layer
    epsF = eps_gyrotropic(eps0_tilde, alpha)
    muF  = mu_gyrotropic(mu_iso, beta)

    dA, dF = L, 1.0 - 2.0*L
    A1 = Layer(dA, epsA1, muA, name="A1")
    F  = Layer(dF, epsF,  muF, name="F")
    A2 = Layer(dA, epsA2, muA, name="A2")
    return UnitCell([A1, F, A2])

# ============================================================
# Seeding & k-sweeps (robust)
# ============================================================

def seed_real_brackets_by_scan(cell: UnitCell, k: float,
                               wmin: float, wmax: float,
                               nsamp: int = 1600,
                               min_width: float = 2e-3,
                               absf_thresh: float = 1e-3
                               ) -> List[Tuple[float, float]]:
    """
    Robustly seed brackets for real-ω roots at fixed k by:
      (i) sign changes of Re f(ω;k),
      (ii) local minima of |f(ω;k)| below absf_thresh.
    """
    w = np.linspace(wmin, wmax, nsamp)
    f = np.array([dispersion_residual(float(om), k, cell) for om in w])
    fr = np.real(f)
    mag = np.abs(f)

    brackets: List[Tuple[float, float]] = []

    # (i) Re f sign changes
    sgn = np.sign(fr)
    idx = np.where(np.diff(sgn) != 0)[0]
    for i in idx:
        a, b = float(w[i]), float(w[i+1])
        if b - a >= min_width:
            brackets.append((a, b))

    # (ii) local minima of |f|
    for i in range(1, nsamp-1):
        if mag[i] < absf_thresh and mag[i] <= mag[i-1] and mag[i] <= mag[i+1]:
            a = max(wmin, float(w[i] - 2*min_width))
            b = min(wmax, float(w[i] + 2*min_width))
            if b > a:
                brackets.append((a, b))

    # merge overlaps and enforce min width
    if not brackets:
        return []
    brackets.sort()
    merged = [brackets[0]]
    for a, b in brackets[1:]:
        a0, b0 = merged[-1]
        if a <= b0:
            merged[-1] = (a0, max(b0, b))
        else:
            merged.append((a, b))
    final = []
    for a, b in merged:
        if b - a < min_width:
            c = 0.5*(a + b)
            a, b = c - 0.5*min_width, c + 0.5*min_width
            a = max(a, wmin); b = min(b, wmax)
        if b > a:
            final.append((a, b))
    return final

def sweep_fix_k_hermitian(cell: UnitCell, k_vals: np.ndarray,
                          wmin: float, wmax: float,
                          nsamp: int = 1600, tol: float = ROOTS.real_xtol) -> List[List[float]]:
    all_roots: List[List[float]] = []
    prev: List[float] = []
    for k in k_vals:
        brackets = seed_real_brackets_by_scan(cell, k, wmin, wmax, nsamp=nsamp)
        # continuation around previous roots
        eps = 1e-3 * max(1.0, (wmax - wmin))
        cont = [(r - eps, r + eps) for r in prev if (wmin <= r <= wmax)]
        brackets = cont + brackets

        roots: List[float] = []
        for (a, b) in brackets:
            if a >= b:
                continue
            try:
                w = solve_omega_real_for_k(cell, k, (a, b), tol=tol)
                if all(abs(w - x) > 1e-6 for x in roots):
                    roots.append(w)
            except Exception:
                pass
        roots.sort()
        all_roots.append(roots)
        prev = roots[:]
    return all_roots

def sweep_fix_k_nonhermitian(cell: UnitCell, k_vals: np.ndarray,
                             hermitian_roots: List[List[float]],
                             tol: float = ROOTS.complex_tol) -> List[List[complex]]:
    all_roots: List[List[complex]] = []
    prev: List[complex] = []
    for j, k in enumerate(k_vals):
        seeds: List[complex] = []
        if j < len(hermitian_roots) and hermitian_roots[j]:
            seeds += [complex(x, 0.0) for x in hermitian_roots[j]]
        seeds += prev
        roots: List[complex] = []
        for s in seeds:
            try:
                w = solve_omega_complex_for_k(cell, k, s,
                                              tol=tol, maxfev=ROOTS.complex_maxfev,
                                              method=ROOTS.complex_method)
                if all(abs(w - x) > 1e-6 for x in roots):
                    roots.append(w)
            except Exception:
                pass
        roots.sort(key=lambda z: (np.real(z), np.imag(z)))
        all_roots.append(roots)
        prev = roots[:]
    return all_roots

# ============================================================
# Utilities: diagnostics & branch assembly
# ============================================================

def detM_error(cell: UnitCell, omega_vals: Iterable[complex]) -> float:
    errs = [abs(det(cell.M(w)) - 1.0) for w in omega_vals]
    return float(max(errs)) if errs else 0.0

def assemble_branches_real(k_vals: np.ndarray,
                           roots_per_k: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    nk = len(k_vals)
    nb = max((len(r) for r in roots_per_k), default=0)
    W = np.full((nb, nk), np.nan, dtype=float)
    if nb == 0:
        return k_vals, W
    for i, w in enumerate(roots_per_k[0]):
        W[i, 0] = w
    for j in range(1, nk):
        curr = roots_per_k[j]
        used = set()
        for i in range(nb):
            prev_w = W[i, j-1]
            if np.isnan(prev_w) or not curr:
                continue
            idx = min(((idx, abs(prev_w - curr[idx]))
                       for idx in range(len(curr)) if idx not in used),
                      key=lambda t: t[1], default=(None, None))[0]
            if idx is not None:
                W[i, j] = curr[idx]; used.add(idx)
        empties = [i for i in range(nb) if np.isnan(W[i, j])]
        leftovers = [idx for idx in range(len(curr)) if idx not in used]
        for i, idx in zip(empties, leftovers):
            W[i, j] = curr[idx]
    return k_vals, W

def assemble_branches_complex(k_vals: np.ndarray,
                              roots_per_k: List[List[complex]]) -> Tuple[np.ndarray, np.ndarray]:
    nk = len(k_vals)
    nb = max((len(r) for r in roots_per_k), default=0)
    W = np.full((nb, nk), np.nan + 1j*np.nan, dtype=np.complex128)
    if nb == 0:
        return k_vals, W
    for i, w in enumerate(roots_per_k[0]):
        W[i, 0] = w
    for j in range(1, nk):
        curr = roots_per_k[j]
        used = set()
        for i in range(nb):
            prev_w = W[i, j-1]
            if np.isnan(prev_w.real) or not curr:
                continue
            idx = min(((idx, abs(prev_w - curr[idx]))
                       for idx in range(len(curr)) if idx not in used),
                      key=lambda t: t[1], default=(None, None))[0]
            if idx is not None:
                W[i, j] = curr[idx]; used.add(idx)
        empties = [i for i in range(nb) if np.isnan(W[i, j].real)]
        leftovers = [idx for idx in range(len(curr)) if idx not in used]
        for i, idx in zip(empties, leftovers):
            W[i, j] = curr[idx]
    return k_vals, W

# ------------------------------------------------------------
# Selection: keep exactly five branches for Fig. 1
# ------------------------------------------------------------

def select_lowest_n_real_branches(k_vals: np.ndarray,
                                  roots_per_k: List[List[float]],
                                  n: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    k, W = assemble_branches_real(k_vals, roots_per_k)
    if W.size == 0:
        return k, W
    means = np.nanmean(W, axis=1)
    order = np.argsort(means)
    pick = order[:min(n, W.shape[0])]
    return k, W[pick, :]

def select_lowest_n_complex_branches(k_vals: np.ndarray,
                                     roots_per_k: List[List[complex]],
                                     n: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    k, W = assemble_branches_complex(k_vals, roots_per_k)
    if W.size == 0:
        return k, W
    means = np.nanmean(np.real(W), axis=1)
    order = np.argsort(means)
    pick = order[:min(n, W.shape[0])]
    return k, W[pick, :]

# ============================================================
# Plotting (Fig. 1 styling)
# ============================================================

def plot_fig1_hermitian(k_vals: np.ndarray,
                        roots_per_k: List[List[float]],
                        n_bands: int = 5,
                        k_lim: Tuple[float, float] = (-np.pi, np.pi),
                        w_lim: Tuple[float, float] = (0.0, 0.6)):
    k, W = select_lowest_n_real_branches(k_vals, roots_per_k, n=n_bands)
    plt.figure(figsize=(4.5, 3.2))
    for row in W:
        plt.plot(k, row, linewidth=2.0)
    plt.xlim(k_lim); plt.ylim(w_lim)
    plt.xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", r"$0$", r"$\pi$"])
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\mathcal{E}$")   # matches paper’s label
    plt.grid(False)
    plt.tight_layout()

def plot_fig1_nonhermitian_complexplane(k_vals: np.ndarray,
                                        roots_per_k: List[List[complex]],
                                        n_bands: int = 5,
                                        re_lim: Tuple[float, float] = (0.0, 0.6),
                                        im_lim: Tuple[float, float] = (-0.12, 0.0),
                                        arrow_every: int = 12,
                                        arrow_scale: float = 0.18):
    _, W = select_lowest_n_complex_branches(k_vals, roots_per_k, n=n_bands)
    plt.figure(figsize=(4.5, 3.2))
    for row in W:
        mask = ~np.isnan(row.real)
        z = row[mask]
        if len(z) < 2: continue
        plt.plot(np.real(z), np.imag(z), linewidth=2.0)
        idxs = np.arange(0, len(z) - 1, arrow_every)
        for i in idxs:
            dz = z[i+1] - z[i]
            if dz == 0: continue
            x0, y0 = np.real(z[i]), np.imag(z[i])
            x1, y1 = x0 + arrow_scale*np.real(dz), y0 + arrow_scale*np.imag(dz)
            plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="->", lw=1.2))
    plt.xlim(re_lim); plt.ylim(im_lim)
    plt.xlabel(r"$\mathrm{Re}\,\omega$")
    plt.ylabel(r"$\mathrm{Im}\,\omega$")
    plt.grid(False)
    plt.tight_layout()

# ============================================================
# Example driver (Figure 1 parameters)
# ============================================================

if __name__ == "__main__":
    # Fig. 1 caption parameters:
    delta, phi1, phi2 = 6.0, 0.0, 0.8
    alpha, beta       = 0.5, 0.5
    # IMPORTANT: 0 < L < 0.5 ; the caption doesn’t specify L.
    # Start with L = 1/3 (equal thirds). You may try 0.25 or 0.2 if needed.
    L = 1/3

    nk = 201
    k_vals = np.linspace(-np.pi, np.pi, nk)

    # Hermitian case: ε0=13, ε̃0=1
    eps0_H, eps0t_H = 13.0, 1.0
    cell_H = make_cell_eq15(delta, phi1, phi2, alpha, beta, eps0_H, eps0t_H, L=L)
    # Scan a bit wider than [0,0.6], then keep the five lowest
    wmin, wmax = 0.0, 0.75
    bands_H = sweep_fix_k_hermitian(cell_H, k_vals, wmin, wmax, nsamp=1600)
    plot_fig1_hermitian(k_vals, bands_H, n_bands=5,
                        k_lim=(-np.pi, np.pi), w_lim=(0.0, 0.6))

    # Non-Hermitian case: ε0=13+5i, ε̃0=1
    eps0_NH, eps0t_NH = 13.0 + 5.0j, 1.0
    cell_NH = make_cell_eq15(delta, phi1, phi2, alpha, beta, eps0_NH, eps0t_NH, L=L)
    bands_NH = sweep_fix_k_nonhermitian(cell_NH, k_vals, bands_H, tol=1e-10)
    plot_fig1_nonhermitian_complexplane(k_vals, bands_NH, n_bands=5,
                                        re_lim=(0.0, 0.6), im_lim=(-0.12, 0.0),
                                        arrow_every=12, arrow_scale=0.18)

    # Diagnostics
    grid = np.linspace(0.0, 0.7, 9)
    print("max |det M - 1| (Hermitian):", detM_error(cell_H, grid))
    print("max |det M - 1| (NonHerm):", detM_error(cell_NH, grid))

    plt.show()
