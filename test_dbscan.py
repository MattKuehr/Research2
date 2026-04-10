import numpy as np
import json
from sklearn.cluster import DBSCAN

with open('data/inputs.json', 'r') as f:
    sample_map = json.load(f)
non_hermitian_keys = [k for k, v in sample_map.items() if v.get('type') == 'non-hermitian']
sample_key = non_hermitian_keys[0]
data = np.load(f"data/arrays/{sample_key}.npz")
q_vals = data['q_values']
omega_vals = data['omega_values']

def assign_bands_dbscan(q_vals, omega_vals, eps=0.05, min_samples=3, alpha=1.0):
    points_3d = np.column_stack((omega_vals.real, omega_vals.imag, q_vals * alpha))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_3d)
    return clustering.labels_

for a in [0.01, 0.05, 0.1, 0.5, 1.0]:
    for e in [0.02, 0.05, 0.1]:
        bands = assign_bands_dbscan(q_vals, omega_vals, eps=e, alpha=a)
        unique, counts = np.unique(bands, return_counts=True)
        print(f"Alpha {a}, Eps {e} - Bands: {len(unique)}")
