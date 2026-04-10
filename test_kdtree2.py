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

def assign_bands_kdtree(omega_vals, k_neighbors=50):
    points_complex = np.column_stack((omega_vals.real, omega_vals.imag))
    tree = KDTree(points_complex)
    num_points = len(omega_vals)
    assigned_bands = np.full(num_points, -1, dtype=int)
    
    band_id = 0
    unassigned_indices = set(range(num_points))
    
    while unassigned_indices:
        start_idx = unassigned_indices.pop()
        assigned_bands[start_idx] = band_id
        current_idx = start_idx
        
        while True:
            current_point = points_complex[current_idx]
            dists, indices = tree.query(current_point, k=k_neighbors)
            
            found_next = False
            for idx in indices:
                if idx in unassigned_indices:
                    assigned_bands[idx] = band_id
                    unassigned_indices.remove(idx)
                    current_idx = idx
                    found_next = True
                    break
            
            if not found_next:
                break
        band_id += 1
    return assigned_bands

bands = assign_bands_kdtree(omega_vals)
unique, counts = np.unique(bands, return_counts=True)
print(f"Bands: {len(unique)}")
print(f"Counts: {counts}")
