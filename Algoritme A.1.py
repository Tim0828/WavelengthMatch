import numpy as np
from collections import deque
import matplotlib.pyplot as plt

def calculate_fingerprints(array, fingerprint_size):
    """"
    Calculate the fingerprints matrix for a given array and fingerprint size.
    
    The fingerprints matrix is sensitive to clusters of noisy data, especially for small fingerprint sizes.
    
    Parameters:
    array (numpy.ndarray): The input array of data points.
    fingerprint_size (int): The size of the fingerprint window.
    
    Returns:
    numpy.ndarray: The fingerprints matrix.
    
    Calculates the relative position of each point in array
    Returns a 2D array, where each row is the fingerprint of a point

    The current fingerprint architecture is sensitive incase of a small fingerprint size 
    to clusters of noisy data
    """
    if fingerprint_size>len(array):
        fingerprint_size = len(array)
    n = len(array)
    # initialize the fingerprints matrix
    fingerprints = np.zeros((n, n))
    # calculate the relative position of each point (matrix is symmetric)
    #iterating over all elements of the matrix
    for i in range(n): 
        for j in range(n):
            # arbitrary value for the diagonal
            if i == j:
                fingerprints[i, j] = 0
            # if the distance between the points is less than half the fingerprint size
            elif abs(i-j) < fingerprint_size/2:
                fingerprints[i, j] = (array[i] - array[j])/array[i]
            # if the point is close to the beginning of the array, maintain fingerprint size by adding points to the right
            elif i < fingerprint_size / 2 and abs(i-j) < fingerprint_size/2 + abs(i-fingerprint_size / 2):
                fingerprints[i, j] = (array[i] - array[j]) / array[i]
            # if the point is close to the end of the array, maintain fingerprint size by adding points to the left
            elif i > n - fingerprint_size / 2 and abs(i-j) < fingerprint_size/2 + n - i:
                fingerprints[i, j] = (array[i] - array[j]) / array[i]
    return fingerprints

def match_fingerprints(mat_spl, mat_wldb):
    """
    Time to discover the matching fingerprints between the two matrices
    The smallest matrix, mat_spl should be matched with a unique entry in the larger matrix, mat_wldb
    These entries should be strictly increasing
    The optimal solution has the lowest sum of the differences between the matched entries
    BFS approach is used to find the optimal solution
    """
    rows_spl, _ = mat_spl.shape
    rows_wldb, __file__ = mat_wldb.shape
    if rows_spl > rows_wldb:
        raise 
    # Initialize the BFS queue
    queue = deque([(0, 0, 0, [])])  # (current row in mat_spl, current row in mat_wldb, current sum of differences, path)
    best_sum = float('inf')
    best_path = []

    while queue:
        row_spl, row_wldb, current_sum, path = queue.popleft()

        # If we have matched all rows in mat_spl
        if row_spl == rows_spl:
            if current_sum < best_sum:
                best_sum = current_sum
                best_path = path #  
            continue

        # Try to match the current row in mat_spl with each remaining row in mat_wldb
        for next_row_wldb in range(row_wldb, rows_wldb):
            wldb_row = mat_wldb[next_row_wldb][mat_wldb[next_row_wldb] != 0]
            spl_row = mat_spl[row_spl][mat_spl[row_spl] != 0]
            if len(wldb_row) != len(spl_row):
                continue
            diff = np.sum(np.abs(spl_row - wldb_row))
            new_sum = current_sum + diff
            new_path = path + [next_row_wldb]

            # Add the new state to the queue
            queue.append((row_spl + 1, next_row_wldb + 1, new_sum, new_path))

    return best_path, best_sum # best path is a list of the matched wavelengths from the database

def main():
    fingerprint_size = min(len(spectral_lines), len(NEON_WAVELN))
    # Calculate the fingerprints matrix for the spectral lines
    spectral_fingerprints = calculate_fingerprints(spectral_lines, fingerprint_size)
    wl_fingerprints = calculate_fingerprints(np.array(list(NEON_WAVELN.keys())), fingerprint_size)
    # print("Spectral fingerprints matrix:\n", spectral_fingerprints)
    # print("Wavelength fingerprints matrix:\n", wl_fingerprints)
    best_wl, best_sum = match_fingerprints(spectral_fingerprints, wl_fingerprints)
    print("Best sum of differences:", best_sum)

    # plot the matched wavelengths
    plt.figure(figsize=(10, 5))
    plt.plot(list(NEON_WAVELN.keys()), label='NEON wavelengths')
    plt.scatter(best_wl, [list(NEON_WAVELN.keys())[i] for i in best_wl], c='red', label='Matched wavelengths')
    plt.show()

spectral_lines = np.array([
    22.181795919847822,
    41.140395614525275,
    144.26273501671918,
    186.4979599536216,
    292.1601184580567,
    335.21179409990134,
    377.61005873042376,
    414.14821455025987,
    445.4639986187941,
])
NEON_WAVELN = {
    532.640 : 20,
    533.078 : 60,
    533.331 : 0,
    533.571 : 0, 
    534.109 : 100,
    534.328 : 60,
    534.920 : 0,
    535.516 : 0,
    535.542 : 0,
    535.802 : 0,
    536.001 : 0,
    536.042 : 0,
    536.223 : 0,
    537.231 : 0,
    537.498 : 0,
    538.325 : 0,
    540.056 : 200,
    541.265 : 0,
    541.856 : 0,
}

main()