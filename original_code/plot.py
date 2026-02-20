import matplotlib.pyplot as plt
from config import get_config
import numpy as np
from typing import Optional


config = get_config()
OMEGA_MAX = config['OMEGA_MAX']


# Plot results (Hermitian Case)
def plot_results(k_vals, omega_vals, figname: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    plt.scatter(k_vals, omega_vals, s=1, c='k', alpha=0.5)
    plt.xlabel('k (Bloch Wavenumber)', fontsize=14)
    plt.ylabel('ω (Frequency)', fontsize=14)
    plt.title('Photonic Crystal Band Structure (Non-Hermitian Case)', fontsize=14)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(0, OMEGA_MAX)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', labelsize=14)

    if figname:
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()
        

def plot_nonhermitian_bands(omega_vals, q_vals, figname: Optional[str] = None):
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

    if figname:
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()


def plot_result_by_band(bands):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(bands)))

    plt.figure(figsize=(8, 6))
    for i, band in enumerate(bands):
        plt.scatter(band[:,0], band[:,1], s=1, c=colors[i], alpha=0.5, label=f"Band {i}")
    plt.xlabel('k (Bloch Wavenumber)', fontsize=14)
    plt.ylabel('ω (Frequency)', fontsize=14)
    plt.title('Photonic Crystal Band Structure (Hermitian Case)', fontsize=14)
    plt.legend(fontsize=14, loc='lower right', markerscale=10.0)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(0, OMEGA_MAX)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', labelsize=14)
    plt.show()