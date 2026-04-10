import numpy as np
import json
from scipy.spatial import KDTree
import heapq

with open('data/inputs.json', 'r') as f:
    sample_map = json.load(f)
non_hermitian_keys = [k for k, v in sample_map.items() if v.get('type') == 'non-hermitian']
sample_key = non_hermitian_keys[0]
data = np.load(f"data/arrays/{sample_key}.npz")
q_vals = data['q_values']
omega_vals = data['omega_values']

def assign_bands_kdtree_unique(k_vals, omega_vals, k_neighbors=20, dist_thresh=0.05, alpha=1.0):
    points_3d = np.column_stack((omega_vals.real, omega_vals.imag, k_vals * alpha))
    tree = KDTree(points_3d)
    
    num_points = len(omega_vals)
    assigned_bands = np.full(num_points, -1, dtype=int)
    
    band_id = 0
    unassigned_indices = set(range(num_points))
    
    while unassigned_indices:
        start_idx = unassigned_indices.pop()
        assigned_bands[start_idx] = band_id
        
        ks_in_band = {k_vals[start_idx]}
        pq = []
        
        dists, indices = tree.query(points_3d[start_idx], k=k_neighbors)
        for d, idx in zip(dists, indices):
            if d < dist_thresh and idx in unassigned_indices:
                heapq.heappush(pq, (d, idx))
                
        while pq:
            d, current_idx = heapq.heappop(pq)
            if current_idx not in unassigned_indices:
                continue
                
            current_k = k_vals[current_idx]
            if current_k in ks_in_band:
                continue
                
            assigned_bands[current_idx] = band_id
            unassigned_indices.remove(current_idx)
            ks_in_band.add(current_k)
            
            dists, indices = tree.query(points_3d[current_idx], k=k_neighbors)
            for nd, idx in zip(dists, indices):
                if nd < dist_thresh and idx in unassigned_indices:
                    heapq.heappush(pq, (nd, idx))
                    
        band_id += 1
        
    return assigned_bands

for a in [0.01, 0.05, 0.1, 0.5, 1.0]:
    bands = assign_bands_kdtree_unique(q_vals, omega_vals, dist_thresh=0.05, alpha=a)
    unique, counts = np.unique(bands, return_counts=True)
    print(f"Alpha {a} - Bands: {len(unique)}")
    print(f"Counts: {counts[:10]}...")
