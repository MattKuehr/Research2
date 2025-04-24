import numpy as np
from scipy.optimize import root_scalar, fsolve
import matplotlib.pyplot as plt
import warnings

class BandStructureSolver:

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def __init__(self, epsilons, mus, thicknesses, gamma=None):
        self.epsilons     = np.array(epsilons,    dtype=complex)
        self.mus          = np.array(mus,        dtype=float)
        self.thicknesses  = np.array(thicknesses, dtype=float)
        self.gamma        = gamma          # None or a float
        self.Lambda       = self.thicknesses.sum()

    # --------------------------------------------------
    # Single-layer 2×2 transfer matrix
    # --------------------------------------------------
    def transfer_matrix_layer(self, omega, d, epsilon, mu):
        k  = omega * np.sqrt(epsilon * mu)
        Z  = np.sqrt(mu / epsilon)
        cd = np.cos(k * d)
        sd = np.sin(k * d)
        return np.array([[        cd, 1j * sd / Z],
                         [1j * Z * sd,        cd]], dtype=complex)

    # --------------------------------------------------
    # Unit-cell matrix (handles Hermitian or PT automatically)
    # --------------------------------------------------
    def unit_cell_matrix(self, omega):
        # ---------- Hermitian stack ----------
        if self.gamma is None:
            M = np.eye(2, dtype=complex)
            for d, eps, mu in zip(self.thicknesses, self.epsilons, self.mus):
                M = self.transfer_matrix_layer(omega, d, eps, mu) @ M
            return M

        # ---------- PT stack (4 half-layers) ----------
        gamma = self.gamma
        eps = [
            self.epsilons[1] - 1j*gamma,   # B  loss
            self.epsilons[0] + 1j*gamma,   # A  gain
            self.epsilons[0] - 1j*gamma,   # A  loss
            self.epsilons[1] + 1j*gamma    # B  gain
        ]
        d_half = [self.thicknesses[1]/2, self.thicknesses[0]/2,
                  self.thicknesses[0]/2, self.thicknesses[1]/2]
        mus   = [self.mus[1], self.mus[0], self.mus[0], self.mus[1]]

        M = np.eye(2, dtype=complex)
        for d, eps_i, mu_i in zip(d_half, eps, mus):
            M = self.transfer_matrix_layer(omega, d, eps_i, mu_i) @ M
        return M

    # --------------------------------------------------
    # Dispersion relation function
    # --------------------------------------------------
    def dispersion_function(self, omega_vec, q):
        """
        * Hermitian call  : omega_vec is a scalar ω  -> return float residual
        * PT-symmetric   : omega_vec = [ω_r, ω_i]   -> return [Re res, Im res]
        """
        Lambda = self.Lambda

        if self.gamma is None:
            omega = omega_vec          # just a float
            M = self.unit_cell_matrix(omega)
            res = 0.5 * np.trace(M) - np.cos(q * Lambda)
            return res.real            # purely real residual

        else:
            omega = omega_vec[0] + 1j*omega_vec[1]
            M = self.unit_cell_matrix(omega)
            res = 0.5 * np.trace(M) - np.cos(q * Lambda)
            # return real & imag parts for root-finding in ℂ
            return [res.real, res.imag]

    # --------------------------------------------------
    # Function for band calculation
    # --------------------------------------------------
    def solve_bands(self,
                    q_vals,
                    omega_max      = 6*np.pi,
                    n_omega_grid   = 200,
                    imag_seed      = (+0.5, -0.5),
                    tol_cluster    = 1e-4):
        """
        Parameters
        ----------
        q_vals        : 1-D array of Bloch wavenumbers (rad/m)
        omega_max     : upper limit for Re(ω) search
        n_omega_grid  : resolution of coarse ω grid used to bracket roots
        imag_seed     : tuple of small ±Im parts to seed fsolve (PT case)
        tol_cluster   : tolerance for merging duplicate roots

        Returns
        -------
        bands_re, bands_im : 2-D arrays (nbands x len(q_vals))
                            Im array is zeros for Hermitian stacks.
        """

        Lambda      = self.Lambda
        nq     = len(q_vals)
        bands_re, bands_im = [], []        # will expand dynamically

        # Precompute coarse grid on Re(omega)
        ω_grid = np.linspace(1e-3, omega_max, n_omega_grid)

        # ------------ Loop over k-points ------------
        for q_idx, q in enumerate(q_vals):

            # ===== Hermitian branch =====
            if self.gamma is None:
                f_vals = np.array([self.dispersion_function(ω, q)
                                for ω in ω_grid])
                sign_changes = np.where(np.diff(np.sign(f_vals)))[0]

                roots_here = []
                for idx in sign_changes:
                    wL, wR = ω_grid[idx], ω_grid[idx+1]
                    try:
                        sol = root_scalar(lambda ω: self.dispersion_function(ω, q),
                                        bracket=[wL, wR], method='brentq')
                        if sol.converged:
                            roots_here.append(sol.root)
                    except ValueError:
                        pass

            # ===== PT / complex branch =====
            else:
                roots_here = []

                # --- 1) use previously found roots as primary guesses -----------
                if q_idx > 0 and len(bands_re) > 0:
                    prev_roots = [bands_re[b][q_idx-1] + 1j*bands_im[b][q_idx-1]
                                for b in range(len(bands_re))
                                if not np.isnan(bands_re[b][q_idx-1])]
                    primary_guesses = [(r.real, r.imag) for r in prev_roots]
                else:
                    primary_guesses = []      # first k-point has none

                # --- 2) supplementary coarse grid (much smaller now) ------------
                ω_re_coarse = np.linspace(0.05, omega_max, 30)   # 30 not 60
                ω_im_seed   = (+0.1, -0.1)                       # tighter seeds
                coarse_guesses = [(wr, wi) for wr in ω_re_coarse for wi in ω_im_seed]

                # combine & de-duplicate
                guesses = primary_guesses + coarse_guesses
                guesses = list({(round(g[0],6), round(g[1],6)) for g in guesses})

                # --- 3) quick residual filter before expensive fsolve -----------
                def resid_norm(pair):
                    rr, ii = pair
                    R = self.dispersion_function([rr, ii], q)
                    return np.hypot(R[0], R[1])

                good_guesses = [g for g in guesses if resid_norm(g) < 2.0]  # <-- tune

                # silence warnings from hopeless fsolve attempts
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)

                    for g in good_guesses:
                        sol, info, ier, _ = fsolve(
                            lambda w: self.dispersion_function(w, q),
                            g,
                            full_output=True,
                            xtol=1e-10,
                            maxfev=200          # can be smaller now
                        )

                        if ier == 1:
                            ω_root = sol[0] + 1j*sol[1]
                            if 0 <= ω_root.real <= omega_max:
                                # keep only well-converged roots
                                R = self.dispersion_function([ω_root.real, ω_root.imag], q)
                                if np.hypot(R[0], R[1]) < 1e-6:
                                    roots_here.append(ω_root)

            # ---------- Cluster & sort ----------
            uniq = []
            for w in sorted(roots_here, key=lambda x: np.real(x)):
                if all(abs(w - u) > tol_cluster for u in uniq):
                    uniq.append(w)

            # ---------- Store into band arrays ----------
            nb_needed = len(uniq) - len(bands_re)
            if nb_needed > 0:               # grow band lists
                for _ in range(nb_needed):
                    bands_re.append([np.nan]*nq)
                    bands_im.append([np.nan]*nq)

            for b, w in enumerate(uniq):
                bands_re[b][q_idx] = np.real(w)
                bands_im[b][q_idx] = np.imag(w) if np.iscomplex(w) else 0.0

        # Convert results to numpy arrays
        return np.array(bands_re, dtype=float), np.array(bands_im, dtype=float)


    # ------------------------------------------------------------------
    # Plot for a purely Hermitian (real-epsilon) crystal
    # ------------------------------------------------------------------
    def plot_real_band_structure(self, q_vals, bands_re,
                                 mirror=True, color='C0'):
        """
        q_vals   : 1-D array of Bloch wave-vectors (rad/m)
        bands_re : 2-D array  (nbands × nq)  real ω roots
        """
        Lambda = self.Lambda
        q_norm = q_vals * Lambda / (2*np.pi)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,5))
        for band in bands_re:
            y = band * Lambda / (2*np.pi)
            plt.plot(q_norm, y, '-', color=color)
            if mirror:
                plt.plot(-q_norm, y, '-', color=color)

        plt.xlabel(r'$q\Lambda / 2\pi$')
        plt.ylabel(r'$\omega\Lambda / 2\pi c$')
        plt.title('Band structure (real ε)')
        plt.xlim(-0.5 if mirror else 0.0, 0.5)
        plt.ylim(0, np.nanmax(bands_re)*Lambda/(2*np.pi)*1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    # ------------------------------------------------------------------
    # Plot for a PT-symmetric / complex-epsilon crystal
    # ------------------------------------------------------------------
    def plot_complex_band_structure(self, q_vals, bands_re, bands_im, Lambda,
                                    max_bands=6, cmap='tab10'):
        """
        Plots the first `max_bands` real and imaginary branches in
        two vertically-stacked subplots sharing the x-axis.

        bands_im should contain NaNs where no root was found.
        """

        Lambda = self.Lambda
        q_norm = q_vals * Lambda / (2*np.pi)

        nbands  = min(max_bands, bands_re.shape[0])
        colors  = plt.get_cmap(cmap).colors

        fig, axes = plt.subplots(2, 1, figsize=(7,7), sharex=True,
                                gridspec_kw={'height_ratios':[2,1]})

        # --- real parts ---
        ax = axes[0]
        for i in range(nbands):
            y = bands_re[i]*Lambda/(2*np.pi)
            ax.plot(q_norm, y, '-', color=colors[i % len(colors)])
        ax.set_ylabel(r'Re$(\omega\Lambda / 2\pi c)$')
        ax.set_title('PT / complex band structure')
        ax.grid(True)

        # --- imaginary parts ---
        ax = axes[1]
        for i in range(nbands):
            y = bands_im[i]*Lambda/(2*np.pi)
            ax.plot(q_norm, y, '-', color=colors[i % len(colors)])
        ax.set_xlabel(r'$q\Lambda / 2\pi$')
        ax.set_ylabel(r'Im$(\omega\Lambda / 2\pi c)$')
        ax.grid(True)

        plt.tight_layout()
        plt.show()
