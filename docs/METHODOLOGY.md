# Spike Triggered Moments: Detailed Methodology

## Overview

This document provides a comprehensive technical description of the spike-triggered moment decomposition algorithm, based on the MATLAB implementation from the Rieke Lab.

## Theoretical Background

### Traditional Spike-Triggered Average (STA)

The classical STA computes the average stimulus preceding spikes:

```
STA = (1/N) Σᵢ (rᵢ - r̄) * sᵢ
```

Where:
- `N` = total number of spikes
- `rᵢ` = spike count for trial i
- `r̄` = mean spike count
- `sᵢ` = stimulus for trial i

### Moment Decomposition Approach

Instead of analyzing raw stimuli, we represent each stimulus as a probability histogram and decompose responses into contributions from different statistical moments.

## Algorithm Steps

### 1. Data Preprocessing

#### Spike Detection and Counting
```python
# Extract spike times from voltage traces
spike_times = detect_spikes(voltage_trace, threshold)

# Count spikes per trial within stimulus window
spike_counts = count_spikes_per_trial(spike_times, trial_structure)
```

#### Stimulus Preparation
```python
# Reshape stimulus data to (n_trials, n_pixels)
stimuli = stimulus_data.reshape(n_trials, -1)

# Apply any necessary preprocessing
stimuli = preprocess_stimuli(stimuli)
```

### 2. Receptive Field Extraction

#### Compute Spike-Triggered Average
```python
# Mean-corrected spike counts
spike_counts_corrected = spike_counts - np.mean(spike_counts)

# Compute STA
sta = (spike_counts_corrected @ stimuli) / np.sum(spike_counts)
sta = sta.reshape(image_height, image_width)
```

#### Extract RF Mask

**Threshold-based extraction:**
```python
# Find high-weight pixels
sta_abs = np.abs(sta)
threshold = rf_threshold * np.max(sta_abs)
high_weight_mask = sta_abs >= threshold
```

**Center-of-mass calculation:**
```python
y_coords, x_coords = np.mgrid[0:height, 0:width]
x_center = np.mean(x_coords[high_weight_mask])
y_center = np.mean(y_coords[high_weight_mask])
```

**Circular mask generation:**
```python
# Calculate distances from center
distances = np.sqrt((x_coords[high_weight_mask] - x_center)**2 + 
                   (y_coords[high_weight_mask] - y_center)**2)

# Robust radius estimation (90th percentile)
rf_radius = np.percentile(distances, 90)

# Create circular mask
distance_map = np.sqrt((x_coords - x_center)**2 + (y_coords - y_center)**2)
rf_mask = distance_map <= rf_radius
```

### 3. Stimulus Histogram Generation

#### Apply RF Mask
```python
# Extract only RF pixels for each trial
n_rf_pixels = np.sum(rf_mask)
stimuli_rf = np.zeros((n_trials, n_rf_pixels))

for i in range(n_trials):
    stim_image = stimuli[i].reshape(image_shape)
    stimuli_rf[i] = stim_image[rf_mask]
```

#### Compute Histograms

**Equal-width binning (±3σ approach):**
```python
# Determine bin edges based on data distribution
all_pixels = stimuli_rf.flatten()
pixel_std = np.std(all_pixels)
bin_edges = np.linspace(-3*pixel_std, 3*pixel_std, n_bins+1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Compute histogram for each trial
stimulus_histograms = np.zeros((n_trials, n_bins))
for i in range(n_trials):
    counts, _ = np.histogram(stimuli_rf[i], bins=bin_edges, density=True)
    stimulus_histograms[i] = counts
```

**Equal-population binning (quantile-based):**
```python
# Create bins with equal number of pixels
quantile_edges = np.quantile(all_pixels, np.linspace(0, 1, n_bins+1))

# Ensure unique edges
quantile_edges = np.unique(quantile_edges)
if len(quantile_edges) < n_bins + 1:
    # Fall back to equal-width if not enough unique values
    quantile_edges = np.linspace(np.min(all_pixels), np.max(all_pixels), n_bins+1)
```

### 4. Moment Computation

For each trial, compute statistical moments from RF-masked pixels:

```python
# Raw moments (not central moments)
moment_0 = np.ones(n_trials)                           # Constant offset
moment_1 = np.mean(stimuli_rf, axis=1)                 # Mean
moment_2 = np.mean(stimuli_rf**2, axis=1)              # Second moment
moment_3 = np.mean(stimuli_rf**3, axis=1)              # Third moment

# Optional: Higher-order moments
moment_4 = np.mean(stimuli_rf**4, axis=1)              # Fourth moment
```

### 5. Ridge Regression Fitting

#### Fit Moment-Driven Models

For each moment order, fit a ridge regression model:

```python
from sklearn.linear_model import Ridge

def fit_moment_weights(histograms, moments, regularization=0):
    """Fit ridge regression for moment-driven responses."""
    model = Ridge(alpha=regularization, fit_intercept=False)
    model.fit(histograms, moments)
    
    weights = model.coef_
    predictions = model.predict(histograms)
    r2 = r2_score(moments, predictions)
    
    return weights, r2, model
```

#### Fit Real Data
```python
# Fit actual spike count data
weights_real, r2_real, model_real = fit_moment_weights(
    stimulus_histograms, spike_counts, regularization
)
```

### 6. Moment Decomposition

#### Linear System Setup

Decompose real neural weights as a linear combination of moment weights:

```python
# Stack moment weight vectors
W_moments = np.column_stack([
    weights_0,    # 0th moment (offset)
    weights_1,    # 1st moment (mean)
    weights_2,    # 2nd moment 
    weights_3     # 3rd moment
])

# Solve linear system: w_real ≈ W_moments @ coefficients
coefficients = np.linalg.lstsq(W_moments, weights_real, rcond=None)[0]
```

#### Interpret Contributions
```python
# Calculate relative contributions
abs_coefficients = np.abs(coefficients)
relative_contributions = abs_coefficients / np.sum(abs_coefficients) * 100

print(f"0th moment (offset): {relative_contributions[0]:.1f}%")
print(f"1st moment (mean): {relative_contributions[1]:.1f}%")
print(f"2nd moment: {relative_contributions[2]:.1f}%")
print(f"3rd moment: {relative_contributions[3]:.1f}%")
```

### 7. Cross-Validation

#### Train-Test Split
```python
from sklearn.model_selection import train_test_split

# Split data (70% train, 30% test)
indices = np.arange(n_trials)
train_idx, test_idx = train_test_split(
    indices, test_size=0.3, random_state=42
)

# Split all data arrays
histograms_train = stimulus_histograms[train_idx]
histograms_test = stimulus_histograms[test_idx]
spikes_train = spike_counts[train_idx]
spikes_test = spike_counts[test_idx]
stimuli_rf_train = stimuli_rf[train_idx]
stimuli_rf_test = stimuli_rf[test_idx]
```

#### Fit on Training Data
```python
# Compute moments for training set
moments_train = {
    0: np.ones(len(train_idx)),
    1: np.mean(stimuli_rf_train, axis=1),
    2: np.mean(stimuli_rf_train**2, axis=1),
    3: np.mean(stimuli_rf_train**3, axis=1)
}

# Fit moment weights on training data
moment_weights_train = {}
for order, moments in moments_train.items():
    weights, r2, model = fit_moment_weights(histograms_train, moments)
    moment_weights_train[order] = weights

# Fit real data weights
weights_real_train, _, _ = fit_moment_weights(histograms_train, spikes_train)
```

#### Decompose and Predict
```python
# Decompose training weights
W_train = np.column_stack([moment_weights_train[i] for i in range(4)])
coefficients_train = np.linalg.lstsq(W_train, weights_real_train, rcond=None)[0]

# Predict on test set using moments
moments_test = {
    0: np.ones(len(test_idx)),
    1: np.mean(stimuli_rf_test, axis=1),
    2: np.mean(stimuli_rf_test**2, axis=1),
    3: np.mean(stimuli_rf_test**3, axis=1)
}

# Combine moments using learned coefficients
spikes_predicted = sum(
    coefficients_train[i] * moments_test[i] 
    for i in range(4)
)

# Evaluate prediction
r2_holdout = r2_score(spikes_test, spikes_predicted)
```

## Synthetic Data Generation

### 2D Pink Noise

Generate spatially correlated noise for testing:

```python
def generate_pink_noise_2d(size, alpha=1.0):
    """Generate 2D pink noise with 1/f^alpha spectrum."""
    # Create frequency grid
    h, w = size
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    cx, cy = w//2, h//2
    
    # Distance from center (frequency magnitude)
    f = np.sqrt((x - cx)**2 + (y - cy)**2)
    f[cy, cx] = 1  # Avoid division by zero at DC
    
    # Generate white noise and apply 1/f filter
    white_noise = np.random.randn(h, w)
    spectrum = np.fft.fftshift(np.fft.fft2(white_noise)) / (f**alpha)
    pink_noise = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    
    # Normalize to [0, 1]
    pink_noise = (pink_noise - np.min(pink_noise)) / (np.max(pink_noise) - np.min(pink_noise))
    
    return pink_noise
```

### Response Simulation

```python
def simulate_responses(patches, response_type='mean', scale_factor=1.0):
    """Simulate neural responses to stimulus patches."""
    if response_type == 'mean':
        return np.mean(patches, axis=1)
    elif response_type == 'variance':
        return np.var(patches, axis=1)
    elif response_type == 'combo':
        mean_resp = np.mean(patches, axis=1)
        var_resp = np.var(patches, axis=1)
        return mean_resp + scale_factor * var_resp
    else:
        raise ValueError(f"Unknown response type: {response_type}")
```

## Key Implementation Details

### Memory Management
- Process stimuli in batches for large datasets
- Use sparse representations where appropriate
- Implement streaming histogram computation

### Numerical Stability
- Add small regularization to ridge regression
- Handle edge cases in RF extraction
- Validate histogram normalization

### Performance Optimization
- Vectorize operations where possible
- Use efficient numpy functions
- Consider caching intermediate results

## Validation and Quality Control

### Sanity Checks
1. **STA reconstruction**: Verify STA computed from matrices matches original
2. **Histogram normalization**: Ensure histograms sum to 1
3. **Weight reconstruction**: Check moment decomposition quality
4. **Cross-validation consistency**: Verify stable results across CV folds

### Diagnostic Metrics
- **R² scores**: Model fit quality for each component
- **Reconstruction error**: RMSE between original and reconstructed weights
- **Prediction accuracy**: Hold-out set performance
- **Contribution stability**: Consistency across different data splits

This methodology provides a robust framework for decomposing neural responses into interpretable moment contributions while maintaining statistical rigor through cross-validation.
