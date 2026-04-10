import numpy as np
import json
from scipy.spatial import KDTree

with open('data/inputs.json', 'r') as f:
    sample_map = json.load(f)
non_hermitian_keys = [k for k, v in sample_map.items() if v.get('type') == 'non-hermitian']
sample_key = non_hermitian_keys[0]
data = np.load(f"data/arrays/{sample_key}.npz")
q_vals = data['q_values']
omega_vals = data['omega_values']

def assign_bands_kdtree_k_weighted(q_vals, omega_vals, k_neighbors=20, dist_thresh=0.05, alpha=1.0):
    points_3d = np.column_stack((omega_vals.real, omega_vals.imag, q_vals * alpha))
    tree = KDTree(points_3d)
    num_points = len(omega_vals)
    assigned_bands = np.full(num_points, -1, dtype=int)
    
    band_id = 0
    unassigned_indices = set(range(num_points))
    
    while unassigned_indices:
        start_idx = unassigned_indices.pop()
        assigned_bands[start_idx] = band_id
        
        stack = [start_idx]
        while stack:
            current_idx = stack.pop()
            current_point = points_3d[current_idx]
            dists, indices = tree.query(current_point, k=k_neighbors)
            
            for d, idx in zip(dists, indices):
                if d < dist_thresh and idx in unassigned_indices:
                    assigned_bands[idx] = band_id
                    unassigned_indices.remove(idx)
                    stack.append(idx)
                    
        band_id += 1
    return assigned_bands

for a in [0.01, 0.05, 0.1, 0.5, 1.0]:
    bands = assign_bands_kdtree_k_weighted(q_vals, omega_vals, dist_thresh=0.1, alpha=a)
    unique, counts = np.unique(bands, return_counts=True)
    print(f"Alpha {a} - Bands: {len(unique)} - Counts: {counts}")
