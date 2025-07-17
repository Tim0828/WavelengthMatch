# Wavelength Matching Algorithms for Spectral Calibration

## Project Overview

This repository contains various algorithms for matching detected spectral lines (in pixel coordinates) to known wavelength databases for spectral calibration purposes. The primary application is calibrating Raman spectrometers using neon reference spectra.

## Problem Statement

In spectroscopy, spectrometers need to be calibrated to convert pixel positions on a detector to actual wavelengths. This calibration process involves:

1. **Input Data**:

   - Detected spectral line positions in pixel coordinates (`spectral_lines`)
   - Known wavelength database (e.g., neon emission lines in `NEON_WAVELN`)

2. **Challenge**:

   - Match detected peaks to known wavelengths
   - Account for potential noise, missing peaks, and instrumental variations
   - Establish accurate pixel-to-wavelength conversion (dispersion relationship)

3. **Goal**:
   - Find the optimal correspondence between detected peaks and reference wavelengths
   - Calculate accurate dispersion (nm/pixel) for the spectrometer
   - Enable wavelength calibration for unknown spectra

## Data Files

- **`spl.txt`**: Spectral line positions in pixels from peak detection
- **`wldb.txt`**: Wavelength database containing known neon emission lines (nm)
- **`NEON_WAVELN`**: Dictionary of neon wavelengths with corresponding intensities

## Algorithm Approaches

### Algorithm A (Fingerprint-Based Matching)

**Files**: `Algoritme A.ipynb`, `Algoritme A.1.py`

- **Method**: Creates "fingerprints" by calculating relative positions between spectral features
- **Approach**: Uses breadth-first search to find optimal matching between fingerprint matrices
- **Strengths**: Robust to offset and scaling differences
- **Implementation**: Optimized with caching and pruning strategies

### Algorithm B (Correlation-Based Optimization)

**Files**: `Algoritme B.py`, `Algoritme B.2.py`

- **Method**: Recursively searches for wavelength combinations that maximize correlation with pixel positions
- **Approach**: Backtracking algorithm with correlation coefficient as fitness function
- **Strengths**: Direct optimization of linear relationship
- **Features**: Includes outlier removal based on residual analysis

### Algorithm C (Dynamic Programming Approach)

**File**: `Algoritme C.py`

- **Method**: Uses dynamic programming for efficient path optimization
- **Approach**: Builds optimal solutions incrementally with caching
- **Strengths**: Computationally efficient, includes iterative improvement
- **Features**: Advanced outlier detection and R² score optimization

### Algorithm Han (Physics-Based Approach)

**Files**: `Algoritme Han.ipynb`, `Algoritme Han B.ipynb`

- **Method**: Incorporates theoretical dispersion calculations based on spectrometer parameters
- **Approach**: Uses grating equation and instrument specifications to guide matching
- **Physics Parameters**:
  - Focal length: 1000 mm
  - Groove density: 1800 gr/mm
  - Blaze angle: 28.6°
  - Pixel size: 16 μm
  - Fiber optic taper: 1.48
- **Strengths**: Physics-informed matching with theoretical validation

### Julia Optimizer

**File**: `optimizer.jl`

- **Method**: High-performance correlation-based optimization in Julia
- **Purpose**: Computationally intensive optimization for large datasets
- **Features**: Recursive path finding with correlation maximization

## Methodology

### Common Workflow

1. **Data Preparation**: Load detected peaks and reference wavelengths
2. **Initial Matching**: Apply algorithm-specific matching strategy
3. **Optimization**: Refine matches using correlation or physics-based criteria
4. **Validation**: Calculate linear regression fit and remove outliers
5. **Calibration**: Determine final dispersion relationship (nm/pixel)

### Key Metrics

- **Correlation Coefficient**: Measures linear relationship quality
- **R² Score**: Coefficient of determination for regression fit
- **Dispersion**: Final calibration factor (nm/pixel)
- **Residuals**: Deviations from linear fit for outlier detection

## Instrument Configuration

The algorithms are designed for a Raman spectrometer with Littrow configuration:

- **Laser wavelength**: 532 nm
- **Grating**: 1800 gr/mm, 28.6° blaze angle
- **Detector**: 512 pixel CCD with 16 μm pixel size
- **Optics**: 1000 mm focal length, 1.48× fiber optic taper

## Usage

Each algorithm can be run independently with the provided test data:

- Spectral lines: ~9 detected peaks in pixel coordinates
- Neon database: 19 reference wavelengths (532-542 nm range)

The algorithms output:

- Matched wavelength-pixel pairs
- Dispersion calibration factor
- Quality metrics (correlation, R²)
- Visualization plots

## Dependencies

- **Python**: NumPy, Matplotlib, Scikit-learn
- **Julia**: Base Julia with linear algebra capabilities
- **Jupyter**: For interactive notebook execution

## Future Development

- Integration of multiple reference databases
- Real-time calibration capabilities
- Machine learning-based peak matching
- Uncertainty quantification in calibration
