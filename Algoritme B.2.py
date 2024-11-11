import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import timeit

@dataclass
class SpectralData:
    pixels: List[float]
    wavelengths: List[float]
    
    def remove_pair(self, index: int) -> None:
        self.pixels.pop(index)
        self.wavelengths.pop(index)

class SpectralMatcher:
    def __init__(self, r2_improvement_threshold: float = 0.001):
        self.r2_improvement_threshold = r2_improvement_threshold
        self.best_path: List[float] = []
        self.best_correlation: float = 0
        self.model = LinearRegression()
        
    def _helper(self, 
                wldb: np.ndarray, 
                n: int, 
                start: int, 
                path: List[float], 
                spl: np.ndarray) -> None:
        """Recursive helper for path optimization"""
        if n == 0:
            current_correlation = np.corrcoef(path, spl)[0, 1]
            if current_correlation > self.best_correlation:
                self.best_path[:] = path
                self.best_correlation = current_correlation
            return
            
        for i in range(start, len(wldb)):
            if not path or wldb[i] > path[-1]:
                self._helper(wldb, n-1, i+1, path + [wldb[i]], spl)
    def optimize_path(self, 
                     wavelengths: np.ndarray, 
                     pixels: np.ndarray) -> Tuple[List[float], float]:
        """
        Find optimal path matching wavelengths to pixels
        
        Args:
            wavelengths: Array of wavelength values
            pixels: Array of pixel values
            
        Returns:
            Tuple of (best_path, correlation_score)
        """
        self.best_path = wavelengths[:len(pixels)].tolist()
        self.best_correlation = 0
        n = len(pixels)
        
        self._helper(wavelengths, n, 0, [], pixels)
        return self.best_path, self.best_correlation

    def fit_regression(self, data: SpectralData) -> Tuple[float, np.ndarray]:
        X = np.array(data.pixels).reshape(-1, 1)
        y = np.array(data.wavelengths)
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        score = r2_score(y, predictions)
        return score, predictions
    
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

def read_data(file_path: str) -> Tuple[List[float], List[float]]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        entries = [list(map(float, line.strip().split())) for line in lines]
        entries = np.array(entries)
    return entries

def main(WLfile_path, PIXfile_path):
    # Read data
    wl = read_data(WLfile_path)
    pix = read_data(PIXfile_path)

    matcher = SpectralMatcher(r2_improvement_threshold=0.001)
    
    start_time = timeit.default_timer()
    best_path, correlation = matcher.optimize_path(wl, pix)
    end_time = timeit.default_timer()
    dt = end_time - start_time
    print(f"Optimization time: {dt:4f} seconds")
    print("Initial correlation:", correlation) 

    initial_data = SpectralData(
        pixels=pix.tolist(),
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


if __name__ == "__main__":
    path_wl = 'wldb.txt'
    path_pix = 'spl.txt'	
    main(path_wl, path_pix)

