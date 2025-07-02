# Spike Triggered Moments

A MATLAB toolkit for decomposing neural responses into contributions from different statistical moments of stimulus distributions using histogram-based analysis.

## Overview

This package extends traditional spike-triggered average (STA) analysis by decomposing neural responses into contributions from statistical moments (mean, variance, skewness, kurtosis) of stimulus distributions. Unlike classical STA which only captures linear receptive fields, this approach reveals how neurons respond to the shape and structure of stimulus distributions within their receptive field.

The toolkit analyzes data from the **FlashedGratePlusNoise protocol** and uses **histogram-based stimulus representations** to:

1. Load neural data from experimental protocols using the Rieke Lab analysis framework
2. Extract receptive field masks from spike-triggered averages  
3. Compute stimulus histograms from RF-masked pixels
4. Fit moment-driven response models using ridge regression
5. Decompose real neural responses into moment contributions
6. Validate predictions through cross-validation

## Key Features

- **Neural data integration**: Compatible with Rieke Lab data analysis framework
- **Histogram-based analysis**: Convert stimuli to probability distributions for moment decomposition
- **Receptive field extraction**: Automatic RF mask generation from spike-triggered averages
- **Moment decomposition**: Decompose responses into 0th-3rd order moment contributions
- **Cross-validation**: Robust train/test splits with hold-out validation
- **Synthetic testing**: 2D pink noise stimulus generation for method validation
- **Ridge regression**: Regularized fitting with custom regression functions
- **Natural image processing**: Analysis of responses to natural image patches

## Requirements

- MATLAB R2018b or later
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Rieke Lab analysis framework (`edu.washington.rieke.Analysis`)
- Custom functions:
  - `manualRidgeRegressionCustom.m`
  - `SpikeDetection.Detector`
  - `getNaturalImagePatchFromLocation2`

## Installation

```matlab
% Clone the repository
git clone https://github.com/your-username/spikeTriggeredMoments.git

% Add to MATLAB path
addpath('/path/to/spikeTriggeredMoments');

% Ensure Rieke Lab framework is in path
addpath('/path/to/rieke-lab-analysis');
```

## Algorithm Overview

### Stimulus Regeneration and Processing

The toolkit processes visual stimuli using filtered Gaussian noise that is applied to natural image patches during the FlashedGratePlusNoise protocol:

1. **Noise Generation**: For each epoch, a seeded random stream is created and Gaussian white noise is generated:

   ```matlab
   noiseStream = RandStream('mt19937ar', 'Seed', noiseSeed);
   noiseFilterSD = SampleEpoch.protocolSettings.get('noiseFilterSD');
   noiseMatrix = imgaussfilt(noiseStream.randn(imageSize(1), imageSize(2)), noiseFilterSD);
   ```

2. **Spatial Filtering**: The white noise is spatially filtered using `imgaussfilt()` with the `noiseFilterSD` parameter from the protocol settings

3. **Normalization**: Filtered noise is normalized by its standard deviation to maintain consistent contrast:

   ```matlab
   noiseMatrix = noiseMatrix / std(noiseMatrix(:));
   ```

4. **Patch Sampling**: Natural image patches are sampled from specified locations using `getNaturalImagePatchFromLocation2()`

5. **Trial Structure**: Each epoch contains multiple noise repeats (`numNoiseRepeats`) with precise timing based on `preTime`, `stimTime`, and `tailTime`

### Core Methodology

#### 1. Receptive Field Extraction

```matlab
% Compute spike-triggered average
STA = (SpikeCounts_corrected * Stimuli) / totSpikes;

% Extract RF mask using adaptive thresholding
sta_abs = abs(STA);
threshold_value = 0.4 * max(sta_abs(:));
high_weight_mask = sta_abs >= threshold_value;

% Compute center of mass and robust radius
x_center = mean(X(high_weight_mask));
y_center = mean(Y(high_weight_mask));
rf_radius = prctile(distances, 90);  % 90th percentile for robustness
```

#### 2. Stimulus Histogram Generation

```matlab
% Extract RF-masked pixels for all trials
Stimuli_RF = zeros(numTrials, numPixels_rf);
for i = 1:numTrials
    stimFull = reshape(Stimuli(i, :), imageSize);
    Stimuli_RF(i, :) = stimFull(rf_mask)';
end

% Generate equal-width bins within ±3σ
pixelStd = std(Stimuli_RF(:));
histEdges = linspace(-3*pixelStd, 3*pixelStd, nbins + 1);

% Compute probability histograms for each trial
for trialIndex = 1:numTrials
    counts = histcounts(Stimuli_RF(trialIndex, :), histEdges, 'Normalization', 'probability');
    StimuliHist(trialIndex, :) = counts;
end
```

#### 3. Moment Decomposition

```matlab
% Compute moments for each trial
moment0 = ones(numTrials, 1);                 % Offset (0th moment)
moment1 = mean(Stimuli_RF, 2);                % Mean (1st moment)
moment2 = mean(Stimuli_RF.^2, 2);             % Second moment
moment3 = mean(Stimuli_RF.^3, 2);             % Third moment

% Fit ridge regression for each moment
[weights_mean, ~, ~] = manualRidgeRegressionCustom(A, moment1, binCenters, false, lambda);
[weights_second, ~, ~] = manualRidgeRegressionCustom(A, moment2, binCenters, false, lambda);
[weights_third, ~, ~] = manualRidgeRegressionCustom(A, moment3, binCenters, false, lambda);

% Decompose real weights using least squares
W_moments = [weights_mean(:), weights_second(:), weights_third(:)];
a_estimated = W_moments \ weights_real(:);
```

#### 4. Cross-Validation

```matlab
% Split data (70% training, 30% testing)
indices = randperm(numTrials);
train_size = round(0.7 * numTrials);
train_idx = indices(1:train_size);
test_idx = indices(train_size+1:end);

% Predict hold-out responses using estimated contributions
spikeCounts_predicted = a_estimated(1) * moment0_test + ...
                       a_estimated(2) * moment1_test + ...
                       a_estimated(3) * moment2_test + ...
                       a_estimated(4) * moment3_test;

% Compute R² for validation
r2_holdout = 1 - sum((actual - predicted).^2) / sum((actual - mean(actual)).^2);
```

## Quick Start

### Basic Analysis

```matlab
% 1. Set up the workspace environment
close all; clear all; clc;

% 2. Configure analysis parameters
params.FrequencyCutoff = 500;
params.Amp = 'Amp1';
params.SamplingInterval = 0.0001;
params.FrameRate = 60.1830;

% 3. Set data paths
dataFolder = '/path/to/your/data/';
exportFolder = '/path/to/your/export/';

% 4. Load experimental data using Rieke Lab framework
loader = edu.washington.rieke.Analysis.getEntityLoader();
list = loader.loadEpochList([exportFolder 'FlashedGratePlusNoise.mat'], dataFolder);

% 5. Build analysis tree and select experimental node
tree = riekesuite.analysis.buildTree(list, {...});
gui = epochTreeGUI(tree);
% Manually select node in GUI, then:
node = gui.getSelectedEpochTreeNodes;
cdt_node = node{1};

% 6. Run the complete analysis
run('spikeTriggerMoments.m');

% 7. The analysis creates these key variables:
% - STA{1}: Spike-triggered average
% - rf_mask: Receptive field mask
% - SpikeCounts: Spike counts per trial
% - moment1, moment2, moment3: Computed moments
% - a_estimated: Moment contributions [mean, second, third]
% - weights_real: Actual data weights
```

### Synthetic Data Testing

```matlab
% Synthetic testing script - stm_text.m
% Generate 2D pink noise patches
N = 512;  % Image size
[x, y] = meshgrid(1:N, 1:N);
cx = ceil(N/2); cy = ceil(N/2);
f = sqrt((x-cx).^2 + (y-cy).^2);
f(cx,cy) = 1;  % Avoid division by zero at DC

% Create pink noise spectrum
whiteNoise = randn(N);
spectrum = fftshift(fft2(whiteNoise)) ./ f;
pinkImg = real(ifft2(ifftshift(spectrum)));
pinkImg = (pinkImg - min(pinkImg(:))) / (max(pinkImg(:)) - min(pinkImg(:)));

% Sample random patches
patchSize = [16, 16];
numTrials = 100;
patches = zeros(numTrials, patchSize(1) * patchSize(2));

for i = 1:numTrials
    x = randi(N - patchSize(1) + 1);
    y = randi(N - patchSize(2) + 1);
    patch = pinkImg(x:x+patchSize(1)-1, y:y+patchSize(2)-1);
    patches(i, :) = patch(:)';
end

% Simulate responses with different moment dependencies
patchMeans = mean(patches, 2);
patchSecondMoment = mean(patches.^2, 2);
spikeCounts_combo = patchMeans + sec_scale*patchSecondMoment;
```

### Cross-Validation Analysis

```matlab
% Cross-validation script - stm_cross_valid.m or stm_clean.m
% Split data (70% training, 30% testing)
numTrials = size(Stimuli_RF, 1);
rng(1);  % For reproducibility
indices = randperm(numTrials);
train_size = round(0.7 * numTrials);
train_idx = indices(1:train_size);
test_idx = indices(train_size+1:end);

% Fit moment weights on training data
[weights_mean, ~, ~] = manualRidgeRegressionCustom(A_train, moment1_train, binCenters, false, 0);
[weights_second, ~, ~] = manualRidgeRegressionCustom(A_train, moment2_train, binCenters, false, 0);

% Decompose real weights
W_moments = [weights_zeroth(:), weights_mean(:), weights_second(:), weights_third(:)];
a_estimated = W_moments \ weights_real_train(:);

% Validate on hold-out set
spikeCounts_predicted = a_estimated(1) * moment0_test + ...
                       a_estimated(2) * moment1_test + ...
                       a_estimated(3) * moment2_test + ...
                       a_estimated(4) * moment3_test;

r2_holdout = 1 - sum((spikeCounts_test - spikeCounts_predicted).^2) / ...
             sum((spikeCounts_test - mean(spikeCounts_test)).^2);

fprintf('Hold-out R²: %.3f\n', r2_holdout);
```

## File Descriptions

### Core Analysis Scripts

- **`spikeTriggerMoments.m`**: Main analysis pipeline with full experimental data processing
- **`stm_clean.m`**: Streamlined version with cross-validation and moment decomposition  
- **`stm_cross_valid.m`**: Dedicated cross-validation analysis script
- **`stm_text.m`**: Synthetic data testing using 2D pink noise stimuli

### Key Functions Required

- **`manualRidgeRegressionCustom.m`**: Custom ridge regression implementation
- **`SpikeDetection.Detector`**: Spike detection from voltage traces
- **`getNaturalImagePatchFromLocation2`**: Natural image patch extraction
- **Rieke Lab Framework**: Data loading and experimental protocol handling

## Data Structures

### Input Data

- **Stimuli**: MATLAB matrix, size `(numTrials, totalPixels)` - flattened stimulus images
- **SpikeCounts**: MATLAB vector, size `(1, numTrials)` - spike counts per trial  
- **SpikeTimes**: Cell array containing spike times for each epoch
- **EpochData**: Voltage traces from experimental recordings

### Output Variables

After running the analysis scripts, the following variables are created in the MATLAB workspace:

```matlab
% Core Analysis Variables (spikeTriggerMoments.m):
STA                           % Cell array of spike-triggered averages
SpikeCounts                   % Original spike counts per trial (1 x numTrials)
SpikeCounts_corrected         % Mean-corrected spike counts
rf_mask                       % Boolean receptive field mask
Stimuli_RF                    % RF-masked stimulus pixels (numTrials x numPixels_rf)
StimuliHist                   % Equal-width histogram matrix (numTrials x nbins)
StimuliHistEqualPop          % Equal-population histogram matrix

% Moment Analysis Variables:
moment1, moment2, moment3     % Raw moments for each trial
weights_mean, weights_second, weights_third  % Ridge regression weights for each moment
weights_real                  % Weights fitted to real spike data
a_estimated                   % Estimated moment contributions [a1, a2, a3]
W_moments                     % Matrix of moment weights for decomposition

% Cross-Validation Variables (stm_clean.m):
a_normalized                  % Relative contributions as percentages
r2_holdout                    % Hold-out validation R²
spikeCounts_predicted_test    % Predicted spike counts on test set

% Structured Results (only in spikeTriggerMoments.m):
results.STA_original          % Main spike-triggered average
results.SpikeCounts           % Original spike counts
results.rf_mask               % Receptive field mask
results.imageSize             % Stimulus image dimensions
```

## Usage Examples

### Running Main Analysis

```matlab
% Set up paths and parameters
dataFolder = '/path/to/data/';
exportFolder = '/path/to/export/';

% Run the main analysis script
run('spikeTriggerMoments.m');

% After completion, access the variables:
fprintf('Total spikes: %d\n', totSpikes);
fprintf('RF radius: %.2f pixels\n', rf_radius);
fprintf('Mean contribution: %.4f\n', a_estimated(1));
fprintf('Second moment contribution: %.4f\n', a_estimated(2));
fprintf('Third moment contribution: %.4f\n', a_estimated(3));

% View the STA and RF mask
figure; 
subplot(1,2,1); imagesc(STA{1}); title('Spike-Triggered Average');
subplot(1,2,2); imagesc(rf_mask); title('Receptive Field Mask');
```

### Running Streamlined Analysis with Cross-Validation

```matlab
% Run the clean version with cross-validation
run('stm_clean.m');

% View moment contributions as percentages
fprintf('\n--- Moment Contributions ---\n');
fprintf('0th Moment (Offset): %.2f%%\n', a_normalized(1));
fprintf('1st Moment (Mean): %.2f%%\n', a_normalized(2));
fprintf('2nd Moment: %.2f%%\n', a_normalized(3));
fprintf('3rd Moment: %.2f%%\n', a_normalized(4));
fprintf('Hold-out R²: %.4f\n', r2_holdout);
```

### Testing with Synthetic Data

```matlab
% Generate and test synthetic responses
run('stm_text.m');

% Check if known moment contributions are recovered
fprintf('Estimated sec_scale: %.4f\n', sec_scale_estimated);
fprintf('True sec_scale: %.4f\n', sec_scale);
```

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{spike_triggered_moments_matlab,
  title={Spike Triggered Moments: MATLAB Toolkit for Neural Response Decomposition},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/spikeTriggeredMoments}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please read our contributing guidelines for details.
