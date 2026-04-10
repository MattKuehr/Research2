import json
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# Load sample map to find non-Hermitian cases
with open('data/inputs.json', 'r') as f:
    sample_map = json.load(f)

non_hermitian_keys = [k for k, v in sample_map.items() if v.get('type') == 'non-hermitian']
sample_key = non_hermitian_keys[0]
data_path = f"data/arrays/{sample_key}.npz"
data = np.load(data_path)

q_vals = data['q_values']
omega_vals = data['omega_values']

def assign_bands_kdtree(omega_vals, k_neighbors=20, max_points=100_000):
    points_complex = np.column_stack((omega_vals.real, omega_vals.imag))
    tree = KDTree(points_complex)
    
    num_points = len(omega_vals)
    assigned_bands = np.full(num_points, -1, dtype=int)
    
    band_id = 0
    unassigned_indices = set(range(num_points))
    
    while unassigned_indices:
        start_idx = min(unassigned_indices)
        
        current_band_indices = [start_idx]
        assigned_bands[start_idx] = band_id
        unassigned_indices.remove(start_idx)
        
        for _ in range(max_points):
            current_point = points_complex[current_band_indices[-1]]
            
            dists, indices = tree.query(current_point, k=k_neighbors)
            
            found_next = False
            for idx in indices:
                if idx in unassigned_indices:
                    assigned_bands[idx] = band_id
                    current_band_indices.append(idx)
                    unassigned_indices.remove(idx)
                    found_next = True
                    break
            
            if not found_next:
                break
                
        band_id += 1
        
    return assigned_bands

assigned_bands = assign_bands_kdtree(omega_vals)
unique_bands = np.unique(assigned_bands)
print(f"Assigned points to {len(unique_bands)} distinct bands.")
