import numpy as np

def get_config():
    return {
        "L_TOTAL": 2*np.pi,
        "L_A": 2*np.pi / 4.0,
        "L_F": 2*np.pi / 2.0,
        "OMEGA_MAX": 0.6,
        "OMEGA_STEPS": 100_000,
        "TOLERANCE": 1e-6      
    }
