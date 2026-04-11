from scipy.spatial import KDTree
import numpy as np
import heapq


def assign_bands_kdtree(k_vals, omega_vals, k_neighbors=20, dist_thresh=0.05, alpha=0.1):
    """
    Assigns each point to a band using KD trees in the 3D space (Re(omega), Im(omega), k * alpha).
    Enforces that a band can only contain ONE point for any given k value to prevent bands from merging.
    """
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
                continue  # Prevent coalescence/merging of multiple points with same k
                
            assigned_bands[current_idx] = band_id
            unassigned_indices.remove(current_idx)
            ks_in_band.add(current_k)
            
            dists, indices = tree.query(points_3d[current_idx], k=k_neighbors)
            for nd, idx in zip(dists, indices):
                if nd < dist_thresh and idx in unassigned_indices:
                    heapq.heappush(pq, (nd, idx))
                    
        band_id += 1
        
    return assigned_bands