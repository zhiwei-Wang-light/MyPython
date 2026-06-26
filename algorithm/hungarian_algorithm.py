import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional


def hungarian_match(cost_matrix: np.ndarray, cost_threshold: Optional[float] = None) -> Tuple[
    List[Tuple[int, int]], List[int], List[int]]:
    cost_matrix = np.asarray(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_threshold is None or cost_matrix[r, c] <= cost_threshold:
            matches.append((r, c))
    matched_rows = set(r for r, _ in matches)
    matched_cols = set(c for _, c in matches)
    unmatched_rows = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
    unmatched_cols = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


cost = np.array([
    [3, 5, 2],
    [1, 2, 3],
    [2, 5, 3]
])

matches, un_tracks, un_dets = hungarian_match(cost, cost_threshold=10)

print("matches:", matches)
print("unmatched tracks:", un_tracks)
print("unmatched detections:", un_dets)
