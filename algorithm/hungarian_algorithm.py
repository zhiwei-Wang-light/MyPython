import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_match(cost_matrix, cost_threshold=None):
    """
    Args:
        cost_matrix: (N, M) numpy array, 越小越好
        cost_threshold: 超过该阈值的匹配会被丢弃（可选）

    Returns:
        matches: [(row, col), ...]
        unmatched_rows: [row indices]
        unmatched_cols: [col indices]
    """
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
    [0.2, 0.8, 0.6],
    [0.5, 0.3, 0.9],
    [0.7, 0.4, 0.8],
    [0.7, 0.7, 0.8]
])

matches, un_tracks, un_dets = hungarian_match(cost, cost_threshold=0.6)

print("matches:", matches)
print("unmatched tracks:", un_tracks)
print("unmatched detections:", un_dets)
