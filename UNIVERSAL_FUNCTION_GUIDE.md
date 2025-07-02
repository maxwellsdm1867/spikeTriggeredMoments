# Universal Natural Image Patch Extraction Function

## Overview

This project now features a **universal MATLAB function** that can be called seamlessly from both MATLAB and Python environments, eliminating the need for separate function versions and ensuring consistency across platforms.

## The Universal Approach

### ✅ **Before: Multiple Versions**
- `getNaturalImagePatchFromLocation2.m` - Original version
- `getNaturalImagePatchFromLocation2_improved.m` - Enhanced version
- `getNaturalImagePatchFromLocation2_python.m` - Python-specific version

### 🚀 **Now: Single Universal Function**
- `getNaturalImagePatchFromLocation_universal.m` - **One function for all environments**

## Key Benefits

### 🎯 **Single Source of Truth**
- One function handles both MATLAB and Python calls
- No version synchronization issues
- Easier maintenance and updates
- Consistent behavior across platforms

### 🔍 **Automatic Environment Detection**
- Detects whether called from MATLAB or Python
- Optimizes behavior for each environment
- No manual configuration required

### 🛠️ **Enhanced Features**
- Robust error handling with descriptive messages
- Cross-platform directory auto-detection
- Comprehensive parameter validation
- Rich metadata and processing information

## Usage Examples

### From MATLAB
```matlab
% Basic usage
result = getNaturalImagePatchFromLocation_universal(patchLocations, 'image001');

% Advanced usage with parameters
result = getNaturalImagePatchFromLocation_universal(...
    patchLocations, 'image001', ...
    'patchSize', [150, 150], ...
    'verbose', true, ...
    'normalize', true);

% Check which environment was detected
fprintf('Called from: %s\n', result.metadata.callingEnvironment);
```

### From Python (Recommended - Use Python Wrapper)
```python
# Simple functional interface (RECOMMENDED)
from simple_patch_extractor import extract_patches

patches = extract_patches(
    patch_locations=[[100, 100], [200, 200]],
    image_name='image001',
    patch_size=(150.0, 150.0),
    verbose=True
)

# Class-based interface (RECOMMENDED)
from natural_image_patch_extractor import NaturalImagePatchExtractor

with NaturalImagePatchExtractor() as extractor:
    patches = extractor.extract_patches(
        [[100, 100], [200, 200]], 
        'image001'
    )
```

### From Python (Advanced - Direct MATLAB Call)
```python
import matlab.engine
import matlab

# Start MATLAB engine (handled automatically by wrappers)
eng = matlab.engine.start_matlab()

# Convert data to MATLAB format
locations = matlab.double([[100, 100], [200, 200]])

# Call the SAME universal function
result = eng.getNaturalImagePatchFromLocation_universal(
    locations, 'image001',
    'verbose', True,
    'patchSize', matlab.double([150.0, 150.0]),
    nargout=1
)
print(f"Called from: {result['metadata']['callingEnvironment']}")
```

> **📍 Important for Python Users**: While direct MATLAB calls work, we **strongly recommend using the Python wrappers** (`simple_patch_extractor.py` or `natural_image_patch_extractor.py`) as they:
> - Handle MATLAB Engine startup/cleanup automatically
> - Provide Pythonic interfaces with proper type hints
> - Convert data formats seamlessly
> - Include comprehensive error handling
> - Are easier to use and maintain

## Function Features

### 📍 **Input Parameters**
- `patchLocations` - N×2 matrix of [x,y] coordinates (pixels)
- `imageName` - Image identifier (without extension)
- `'resourcesDir'` - Image directory (auto-detects if empty)
- `'patchSize'` - Patch size in microns [height, width]
- `'verbose'` - Progress display (true/false)
- `'normalize'` - Image normalization (true/false)
- Plus many more customization options...

### 📤 **Output Structure**
```matlab
result = struct with fields:
    images: {3×1 cell}              % Extracted patches
    fullImage: [1536×1024 double]   % Full natural image
    backgroundIntensity: 0.4521     % Mean image intensity
    patchInfo: [3×1 struct]         % Per-patch information
    metadata: [1×1 struct]          % Processing metadata
```

### 🔧 **Metadata Information**
```matlab
result.metadata = struct with fields:
    imageName: 'image001'
    imageFilePath: '/path/to/image001.iml'
    numValidPatches: 3
    numClippedPatches: 0
    callingEnvironment: 'MATLAB'     % or 'Python'
    processingTime: datetime
    parameters: [1×1 struct]         % All input parameters
```

## Setup Instructions

### 1. **For MATLAB Users**
Just call the function directly - no setup required!
```matlab
result = getNaturalImagePatchFromLocation_universal(locations, 'image001');
```

### 2. **For Python Users**
Install MATLAB Engine API:
```bash
# Navigate to MATLAB installation
cd "matlabroot/extern/engines/python"
python setup.py install
```

Then use the Python interfaces (recommended):
```python
# Simple wrapper function (RECOMMENDED)
from simple_patch_extractor import extract_patches
patches = extract_patches(...)

# Class interface (RECOMMENDED)
from natural_image_patch_extractor import NaturalImagePatchExtractor
extractor = NaturalImagePatchExtractor()
patches = extractor.extract_patches(...)

# Advanced: Direct call (not recommended for most users)
# result = eng.getNaturalImagePatchFromLocation_universal(...)
```

## File Organization

```
├── getNaturalImagePatchFromLocation_universal.m     # ← THE UNIVERSAL FUNCTION
├── simple_patch_extractor.py                         # Python wrapper (calls universal)
├── natural_image_patch_extractor.py                  # Class interface (calls universal)
├── universal_function_examples_matlab.m              # MATLAB usage examples
├── universal_function_examples_python.py             # Python usage examples
└── UNIVERSAL_FUNCTION_GUIDE.md                       # This guide
```

## Migration Guide

### From Original Functions
Replace your existing calls:
```matlab
% Old way (any previous version)
result = getNaturalImagePatchFromLocation2_improved(locations, 'image001');

% New way (universal)
result = getNaturalImagePatchFromLocation_universal(locations, 'image001');
```

### From Python-Specific Version
Update your Python scripts:
```python
# Old way
result = eng.getNaturalImagePatchFromLocation2_python(...)

# New way (same function as MATLAB!)
result = eng.getNaturalImagePatchFromLocation_universal(...)
```

## Testing and Examples

### Run MATLAB Examples
```matlab
run('universal_function_examples_matlab.m')
```

### Run Python Examples
```bash
python universal_function_examples_python.py
```

## Troubleshooting

### Common Issues

1. **Function Not Found**
   - Ensure `getNaturalImagePatchFromLocation_universal.m` is in your MATLAB path
   - From Python: Check that the directory is added to MATLAB engine path

2. **Image Files Not Found**
   - Function auto-detects common image directories
   - Manually specify: `'resourcesDir', '/your/path/here'`

3. **Python Engine Issues**
   - Verify MATLAB Engine API installation
   - Check MATLAB version compatibility (R2018b+)

### Getting Help
```matlab
% View function documentation
help getNaturalImagePatchFromLocation_universal

% Run with verbose output
result = getNaturalImagePatchFromLocation_universal(..., 'verbose', true);
```

## Conclusion

The universal function approach provides:
- ✅ **Simplified maintenance** - One function to rule them all
- ✅ **Consistent behavior** - Same results from MATLAB and Python
- ✅ **Future-proof design** - Easy to extend and modify
- ✅ **Cross-platform compatibility** - Works everywhere MATLAB does

This design eliminates the complexity of maintaining multiple function versions while providing the best experience for both MATLAB and Python users.
