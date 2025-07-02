# Spike Triggered Moments

A comprehensive toolkit for decomposing neural responses into contributions from different statistical moments of stimulus distributions using histogram-based analysis. Features both **MATLAB** and **Python** interfaces with a **universal natural image processing function**.

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

### 🧠 **Neural Data Analysis**
- **Spike-triggered moment decomposition**: Beyond traditional STA analysis
- **Histogram-based stimulus representations**: Convert stimuli to probability distributions
- **Receptive field extraction**: Automatic RF mask generation from spike-triggered averages
- **Cross-validation framework**: Robust train/test splits with hold-out validation

### 🔬 **Stimulus Processing**
- **Universal natural image patch extraction**: Single function callable from both MATLAB and Python
- **Cross-platform compatibility**: Works on Windows, macOS, and Linux
- **Automatic directory detection**: Finds natural image files across different systems
- **Robust error handling**: Comprehensive validation and informative error messages

### 🐍 **Python Integration**
- **MATLAB Engine API integration**: Seamless MATLAB function calls from Python
- **Multiple Python interfaces**: Simple functional and class-based approaches
- **NumPy compatibility**: Native numpy array support
- **Same results**: Identical output from MATLAB and Python environments

### 🔧 **Development Tools**
- **Synthetic data testing**: 2D pink noise stimulus generation for method validation
- **Ridge regression**: Regularized fitting with custom regression functions
- **Comprehensive documentation**: Examples and guides for both environments

## Requirements

### MATLAB Environment
- MATLAB R2018b or later
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Rieke Lab analysis framework (`edu.washington.rieke.Analysis`)

### Python Environment (Optional)
- Python 3.6+
- MATLAB Engine API for Python
- NumPy
- MATLAB R2018b or later (for backend processing)

### Custom Functions
- `manualRidgeRegressionCustom.m`
- `SpikeDetection.Detector`
- `getNaturalImagePatchFromLocation2_universal.m` ⭐ **Universal function**

## Installation

### MATLAB Setup
```matlab
% Clone the repository
git clone https://github.com/maxwellsdm1867/spikeTriggeredMoments.git

% Add to MATLAB path
addpath('/path/to/spikeTriggeredMoments');

% Ensure Rieke Lab framework is in path
addpath('/path/to/rieke-lab-analysis');
```

### Python Setup (Optional)
```bash
# Install MATLAB Engine API for Python
cd "matlabroot/extern/engines/python"
python setup.py install

# Install additional Python dependencies
pip install numpy
```

## 🌟 Universal Natural Image Patch Extraction

This toolkit features a **revolutionary universal function** that can be called seamlessly from both MATLAB and Python environments, eliminating the need for separate function versions.

### Key Innovation: One Function, Two Environments

**`getNaturalImagePatchFromLocation2_universal.m`** automatically detects whether it's being called from MATLAB or Python and optimizes its behavior accordingly.

#### From MATLAB:
```matlab
% Extract patches using the universal function
patches = getNaturalImagePatchFromLocation2_universal([[100,100]; [200,200]], 'image001', 'verbose', true);
fprintf('Called from: %s\n', patches.metadata.callingEnvironment);  % Output: 'MATLAB'
```

#### From Python:
```python
import matlab.engine
eng = matlab.engine.start_matlab()

# Call the SAME universal function
locations = matlab.double([[100, 100], [200, 200]])
result = eng.getNaturalImagePatchFromLocation2_universal(locations, 'image001', 'verbose', True, nargout=1)
print(f"Called from: {result['metadata']['callingEnvironment']}")  # Output: 'Python'
```

#### From Python (Using Wrappers):
```python
# Simple functional interface
from simple_patch_extractor import extract_patches
patches = extract_patches([[100, 100], [200, 200]], 'image001', verbose=True)

# Class-based interface
from natural_image_patch_extractor import NaturalImagePatchExtractor
extractor = NaturalImagePatchExtractor()
patches = extractor.extract_patches([[100, 100], [200, 200]], 'image001')
```

### Benefits of Universal Approach

- ✅ **Single source of truth** - One function, no version conflicts
- ✅ **Consistent results** - Identical output from MATLAB and Python
- ✅ **Easy maintenance** - Update one function, benefit everywhere
- ✅ **Cross-platform** - Works on Windows, macOS, and Linux
- ✅ **Auto-detection** - Finds natural image directories automatically
- ✅ **Future-proof** - Easy to extend and modify

### Quick Start Examples

For comprehensive usage examples, see:
- `universal_function_examples_matlab.m` - MATLAB examples
- `universal_function_examples_python.py` - Python examples  
- `UNIVERSAL_FUNCTION_GUIDE.md` - Complete documentation

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

4. **Patch Sampling**: Natural image patches are sampled from specified locations using the **universal function** `getNaturalImagePatchFromLocation2_universal()`, which works seamlessly from both MATLAB and Python environments

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

### Option 1: MATLAB Analysis (Traditional)

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

### Option 2: Python Interface (Modern)

```python
# Extract natural image patches using Python
from simple_patch_extractor import extract_patches

# Same universal function used in MATLAB!
patches = extract_patches(
    patch_locations=[[100, 100], [200, 200], [300, 300]],
    image_name='image001',
    patch_size=(150.0, 150.0),
    verbose=True
)

print(f"Extracted {len(patches['images'])} patches")
print(f"Valid patches: {patches['metadata']['num_valid_patches']}")
print(f"Called from: {patches['metadata']['calling_environment']}")

# Use patches in your Python analysis pipeline
import numpy as np
valid_patches = [img for img in patches['images'] if img is not None]
mean_intensities = [np.mean(patch) for patch in valid_patches]
```

### Option 3: Hybrid Workflow

```python
# Python preprocessing and patch extraction
from natural_image_patch_extractor import NaturalImagePatchExtractor

with NaturalImagePatchExtractor() as extractor:
    patches = extractor.extract_patches(locations, 'image001')
    
# Export to MATLAB format for spike analysis
import scipy.io
scipy.io.savemat('patches_for_matlab.mat', {
    'patches': patches['images'],
    'locations': patches['patch_info']
})
```

```matlab
% Load Python-extracted patches in MATLAB
load('patches_for_matlab.mat');

% Continue with spike-triggered moment analysis
run('spikeTriggerMoments.m');
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

### Core Analysis Scripts (MATLAB)

- **`spike_triggered_moments_master.m`**: 🆕 Unified master script for complete spike-triggered moment analysis
- **`spikeTriggerMoments.m`**: Main analysis pipeline with full experimental data processing
- **`stm_clean.m`**: Streamlined version with cross-validation and moment decomposition  
- **`stm_cross_valid.m`**: Dedicated cross-validation analysis script
- **`stm_text.m`**: Synthetic data testing using 2D pink noise stimuli

### Universal Natural Image Processing

- **`getNaturalImagePatchFromLocation2_universal.m`**: 🌟 **Universal function** callable from both MATLAB and Python
- **`simple_patch_extractor.py`**: Simple Python interface (calls universal function)
- **`natural_image_patch_extractor.py`**: Class-based Python interface (calls universal function)

### Documentation and Examples

- **`UNIVERSAL_FUNCTION_GUIDE.md`**: Complete guide for the universal function approach
- **`universal_function_examples_matlab.m`**: MATLAB usage examples
- **`universal_function_examples_python.py`**: Python usage examples
- **`IMPLEMENTATION_COMPLETE.md`**: Implementation summary and benefits

### Legacy Functions (Deprecated)

- **`getNaturalImagePatchFromLocation2.m`**: Original patch extraction function
- **`getNaturalImagePatchFromLocation2_improved.m`**: Enhanced version
- **`getNaturalImagePatchFromLocation2_python.m`**: Python-specific version

> **Note**: Use the universal function for all new development. Legacy functions are maintained for compatibility but will be removed in future versions.

### Required Dependencies

- **`manualRidgeRegressionCustom.m`**: Custom ridge regression implementation
- **`SpikeDetection.Detector`**: Spike detection from voltage traces
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

## Project Structure

```
spikeTriggeredMoments/
├── 📊 Core Analysis (MATLAB)
│   ├── spike_triggered_moments_master.m        # Unified master script
│   ├── spikeTriggerMoments.m                   # Main analysis pipeline
│   ├── stm_clean.m                             # Streamlined analysis + cross-validation
│   ├── stm_cross_valid.m                       # Cross-validation analysis
│   └── stm_text.m                              # Synthetic data testing
│
├── 🌟 Universal Natural Image Processing
│   ├── getNaturalImagePatchFromLocation2_universal.m  # Universal MATLAB function
│   ├── simple_patch_extractor.py              # Simple Python interface
│   └── natural_image_patch_extractor.py       # Class-based Python interface
│
├── 📚 Documentation & Examples
│   ├── README.md                               # This comprehensive guide
│   ├── UNIVERSAL_FUNCTION_GUIDE.md             # Universal function documentation
│   ├── IMPLEMENTATION_COMPLETE.md              # Implementation summary
│   ├── universal_function_examples_matlab.m    # MATLAB examples
│   └── universal_function_examples_python.py   # Python examples
│
├── 🔧 Legacy Functions (Deprecated)
│   ├── getNaturalImagePatchFromLocation2.m     # Original function
│   ├── getNaturalImagePatchFromLocation2_improved.m  # Enhanced version
│   └── getNaturalImagePatchFromLocation2_python.m    # Python-specific version
│
└── 📦 Configuration
    └── pyproject.toml                          # Python project configuration
```

## Usage Scenarios

### 🧠 Neuroscience Researchers
- Analyze spike-triggered moments in retinal ganglion cells
- Decompose neural responses to natural image stimuli
- Cross-validate moment-based models of neural computation

### 🐍 Python Developers  
- Extract natural image patches using Python workflows
- Integrate MATLAB image processing with Python analysis pipelines
- Build reproducible research pipelines with version control

### 🔬 Method Developers
- Test new stimulus decomposition approaches
- Validate algorithms with synthetic data
- Develop cross-platform analysis tools

### 👥 Collaborative Teams
- Share analysis code between MATLAB and Python users
- Ensure reproducible results across different environments
- Maintain single codebase for mixed research teams

## Getting Help

### 📖 Documentation
- **README.md** - This comprehensive guide
- **UNIVERSAL_FUNCTION_GUIDE.md** - Detailed universal function documentation
- **Function help**: `help getNaturalImagePatchFromLocation2_universal` in MATLAB

### 🔧 Troubleshooting
- Check MATLAB Engine API installation for Python usage
- Ensure natural image directories are accessible
- Verify Rieke Lab framework is properly installed
- See troubleshooting section in UNIVERSAL_FUNCTION_GUIDE.md

### 💡 Examples
- Run `universal_function_examples_matlab.m` for MATLAB examples
- Run `python universal_function_examples_python.py` for Python examples
- Check test functions in Python modules for usage patterns

## Contributing

We welcome contributions! Please:

1. **Fork the repository** on GitHub
2. **Create a feature branch** for your changes
3. **Test thoroughly** with both MATLAB and Python interfaces
4. **Update documentation** as needed
5. **Submit a pull request** with clear description

### Development Guidelines
- Maintain compatibility with both MATLAB and Python
- Add comprehensive tests for new features
- Update documentation and examples
- Follow existing code style and conventions

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{spike_triggered_moments_universal,
  title={Spike Triggered Moments: Universal Toolkit for Neural Response Decomposition},
  author={Maxwell S.D.M. and Rieke Lab},
  year={2025},
  url={https://github.com/maxwellsdm1867/spikeTriggeredMoments},
  note={Features universal MATLAB/Python natural image patch extraction}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **Rieke Lab** at University of Washington for the analysis framework
- **MATLAB** and **Python** communities for excellent cross-platform tools
- **Open source contributors** who make reproducible research possible

---

**🚀 Ready to get started?** 
- MATLAB users: Run `universal_function_examples_matlab.m`
- Python users: Run `python universal_function_examples_python.py`
- Both: Check out `UNIVERSAL_FUNCTION_GUIDE.md` for comprehensive documentation

**One function, two environments, endless possibilities! 🌟**
