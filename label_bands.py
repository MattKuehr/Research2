import json
import numpy as np
from tqdm import tqdm
from typing import List
from utils import count_bands, get_points_array
from concurrent.futures import ProcessPoolExecutor

def process_index(idx: int, x_values: List[float]):
    try:
        points = get_points_array(f"./numpy_arrays/{idx}.npz")
        count = count_bands(points, x_values)
        return idx, count
    except Exception as e:
        return idx, -1
    
if __name__ == '__main__':
    with open("sample_map.json") as json_file:
        sample_map = json.load(json_file)
    hermitian_indices = []
    
    for key, value in sample_map.items():
        if value == 'hermitian':
            hermitian_indices.append(key)
    
    band_counts = {}
    x_values = [-np.pi / 8, np.pi / 8]

    print(f"Starting parallel processing on {len(hermitian_indices)} items...")

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(
                process_index, hermitian_indices,
                [x_values]*len(hermitian_indices)
            ),
            total_tasks = len(hermitian_indices),
            desc="Processing bands",
            unit="file"
        ))

        for idx, count in results:
            if count == -1:
                print(f"Algorithm did not converge for {idx}")
                band_counts[idx] = None
            else:
                band_counts[idx] = count
    
    print("Processing complete")
    with open("band_counts.json", "w") as json_file:
        json.dump(band_counts, json_file, indent=4)
    print(f"Wrote to band_counts.json")
