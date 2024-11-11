import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import timeit
from functools import lru_cache

@dataclass
class SpectralData:
    pixels: List[float]
    wavelengths: List[float]
    
    def remove_pair(self, index: int) -> None:
        self.pixels.pop(index)
        self.wavelengths.pop(index)

class SpectralMatcher:
    def __init__(self, r2_improvement_threshold: float = 0.001, correlation_threshold: float = 0.95):
        self.r2_improvement_threshold = r2_improvement_threshold
        self.correlation_threshold = correlation_threshold
        self.best_path: List[float] = []
        self.best_correlation: float = 0
        self.model = LinearRegression()
        
    @staticmethod
    @lru_cache(maxsize=1024)
    def _calculate_correlation(wl_tuple: Tuple[float, ...], px_tuple: Tuple[float, ...]) -> float:
        """Cached correlation calculation"""
        return np.corrcoef(wl_tuple, px_tuple)[0, 1]

    def optimize_path(self, wavelengths: np.ndarray, pixels: np.ndarray) -> Tuple[List[float], float]:
        """
        Find optimal path matching wavelengths to pixels using dynamic programming
        """
        n_pixels = len(pixels)
        n_wavelengths = len(wavelengths)
        
        # Initialize DP table: [n_pixels + 1, n_wavelengths + 1]
        dp = np.zeros((n_pixels + 1, n_wavelengths + 1))
        paths = [[[] for _ in range(n_wavelengths + 1)] for _ in range(n_pixels + 1)]
        
        # Convert arrays to tuples for caching
        px_tuple = tuple(pixels)
        
        # Fill DP table
        for i in range(1, n_pixels + 1):
            for j in range(i, n_wavelengths + 1):
                # Try adding current wavelength to path
                current_path = paths[i-1][j-1] + [wavelengths[j-1]]
                
                if len(current_path) == i:
                    corr = self._calculate_correlation(tuple(current_path), px_tuple[:i])
                    
                    # Update if better correlation found
                    if corr > dp[i][j-1]:
                        dp[i][j] = corr
                        paths[i][j] = current_path
                        
                        # Early stopping if excellent correlation found
                        if corr > self.correlation_threshold:
                            self.best_path = current_path
                            self.best_correlation = corr
                            return self.best_path, self.best_correlation
                
                # Keep previous best
                if dp[i][j-1] > dp[i][j]:
                    dp[i][j] = dp[i][j-1]
                    paths[i][j] = paths[i][j-1]
        
        # Find best path
        best_idx = np.argmax(dp[n_pixels])
        self.best_path = paths[n_pixels][best_idx]
        self.best_correlation = dp[n_pixels][best_idx]
        
        return self.best_path, self.best_correlation

    def fit_regression(self, data: SpectralData) -> Tuple[float, np.ndarray]:
        X = np.array(data.pixels).reshape(-1, 1)
        y = np.array(data.wavelengths)
        if len(X > 0) and len(y > 0):
            self.model.fit(X, y)
            predictions = self.model.predict(X)
            score = r2_score(y, predictions)
            return score, predictions
        else:
            return 0, np.array([])
    
    def find_worst_residual(self, data: SpectralData, predictions: np.ndarray) -> int:
        residuals = np.abs(predictions - np.array(data.wavelengths))
        return np.argmax(residuals)
    
    def iterative_improvement(self, data: SpectralData) -> Tuple[float, np.ndarray, List[int]]:
        current_score, predictions = self.fit_regression(data)
        removed_indices = []
        
        while True:
            worst_idx = self.find_worst_residual(data, predictions)
            temp_data = SpectralData(data.pixels.copy(), data.wavelengths.copy())
            temp_data.remove_pair(worst_idx)
            
            new_score, new_predictions = self.fit_regression(temp_data)
            improvement = new_score - current_score
            
            if improvement < self.r2_improvement_threshold:
                break
                
            data.remove_pair(worst_idx)
            removed_indices.append(worst_idx)
            current_score = new_score
            predictions = new_predictions
            
        return current_score, predictions, removed_indices

class Visualizer:
    @staticmethod
    def plot_matches(data: SpectralData, predictions: np.ndarray, title: str):
        plt.figure(figsize=(10, 5))
        plt.scatter(data.wavelengths, data.pixels, c='red', label='Matched wavelengths')
        plt.plot(predictions, data.pixels, label='Predicted wavelengths')
        plt.title(title)
        plt.xlabel('Wavelength')
        plt.ylabel('Pixel')
        plt.legend()
        plt.show()

def main(wl, pix):
    matcher = SpectralMatcher(r2_improvement_threshold=0.001)
    wavelengths = np.array(wl)  #  wavelength data
    pixels = np.array(pix)      #  pixel data
    start_time = timeit.default_timer()
    best_path, correlation = matcher.optimize_path(wavelengths, pixels)
    end_time = timeit.default_timer()
    dt = end_time - start_time
    print(f"Optimization time: {dt:4f} seconds")
    print("Initial correlation:", correlation) 

    initial_data = SpectralData(
        pixels=pixels.tolist(),
        wavelengths=best_path
    )
    
    visualizer = Visualizer()
    
    # Initial fit
    initial_score, initial_predictions = matcher.fit_regression(initial_data)
    visualizer.plot_matches(initial_data, initial_predictions, "Initial Fit")
    print(f"Initial R² Score: {initial_score:.4f}")
    
    # Iterative improvement
    final_score, final_predictions, removed_indices = matcher.iterative_improvement(initial_data)
    visualizer.plot_matches(initial_data, final_predictions, "Optimized Fit")
    print(f"Final R² Score: {final_score:.4f}")
    print(f"Number of pairs removed: {len(removed_indices)}")



# Test data

NEON_WAVELN = [532.640, 533.078, 533.331, 533.571, 534.109, 534.328, 534.920, 535.516, 535.542, 535.802, 536.001, 536.042, 536.223, 537.231, 537.498, 538.325, 540.056, 541.265, 541.856]
spectral_lines = [
    22.181795919847822,
    41.140395614525275,
    144.26273501671918,
    186.4979599536216,
    292.1601184580567,
    335.21179409990134,
    377.61005873042376,
    414.14821455025987,
    445.4639986187941,
]
pix = [25.97, 44.73, 149.19, 190.28, 294.58, 319.01]
wl = [534.10938, 534.32834, 535.5164, 536.00121, 537.2311, 537.4975]

if __name__ == "__main__":
    main(NEON_WAVELN, spectral_lines)
