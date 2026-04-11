from typing import List
import numpy as np
import os


def get_points_array(path: str, hermitian: bool = True):
    if hermitian:
        if os.path.exists(path):
            data = np.load(path)
        else:
            raise FileNotFoundError()
        
        k_values, omega_values = data["k_values"], data["omega_values"]
        
        indexed_k_list = list(enumerate(k_values))
        sorted_k_list = sorted(indexed_k_list, key=lambda item: item[1])
        sorted_indices = [item[0] for item in sorted_k_list]
        
        omega_values = omega_values[sorted_indices]
        k_values = np.array([item[1] for item in sorted_k_list])
        
        k_values = k_values.reshape(len(k_values), 1)
        omega_values = omega_values.reshape(len(omega_values), 1)
        points = np.hstack((k_values, omega_values))
    
    else:
        pass
    
    return points


def project_line(points: np.ndarray, x_coord: float = -np.pi / 4, n_steps: int = 10_000, verbose: bool = False):
    test_line = np.linspace(0.0, 0.6, n_steps)

    point_clusters = []

    count = 0
    new_position = 0.0
    for i in range(len(test_line)):
        test_point = np.array([x_coord, test_line[i]])
        if test_point[1] < new_position:
            if verbose:
                print("Skipped")
            continue

        diff = np.abs(points - test_point)
        indices = np.where((diff[:,0] < 1e-3) & (diff[:,1] < 1e-3))
        if len(indices[0]) > 0:
            if verbose:
                print("Found non-empty matches!")
            count += 1
            found_points, offset = perturb_up(points, indices, diff)
            point_clusters.append(found_points)
            #new_position = test_point[1] + offset
            new_position = test_point[1] + 5e-3
    
    return point_clusters


def perturb_up(points, indices: np.ndarray, diff: np.ndarray, eps: float = 1e-6, verbose: bool = False) -> np.ndarray:
    points_in_radius = diff[indices]
    original_points = points[indices]
    if verbose:
        print(points_in_radius)
    max_y_dist = np.max(np.abs(1e-3 - points_in_radius[:,1]))
    return original_points, max_y_dist + eps


def count_bands(points: np.ndarray, x_values: List[float]) -> int:
    left_count = len(project_line(points, x_values[0], n_steps=1_000))
    right_count = len(project_line(points, x_values[1], n_steps=1_000))
    if left_count == right_count:
        return left_count
    else:
        return -1