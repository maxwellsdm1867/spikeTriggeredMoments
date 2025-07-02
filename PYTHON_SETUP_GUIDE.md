# Python Interface for Natural Image Patch Extraction

This directory contains Python interfaces to the enhanced MATLAB natural image patch extraction function. This allows you to call the MATLAB functionality directly from Python with automatic data conversion and error handling.

## 📁 Files Overview

### Core Files
- `getNaturalImagePatchFromLocation2_python.m` - MATLAB function optimized for Python calling
- `simple_patch_extractor.py` - Simple Python interface (recommended for most users)
- `natural_image_patch_extractor.py` - Advanced Python interface with class-based approach

### Documentation
- `PYTHON_SETUP_GUIDE.md` - This file
- `python_examples.py` - Complete usage examples

## 🚀 Quick Start

### 1. Prerequisites

**MATLAB Requirements:**
- MATLAB R2018b or later
- MATLAB Engine API for Python installed

**Python Requirements:**
- Python 3.7 or later
- numpy
- MATLAB Engine API for Python

### 2. Install MATLAB Engine API for Python

```bash
# Navigate to your MATLAB installation directory
# On Windows: C:\Program Files\MATLAB\R2023a\extern\engines\python
# On macOS: /Applications/MATLAB_R2023a.app/extern/engines/python
# On Linux: /usr/local/MATLAB/R2023a/extern/engines/python

cd /path/to/matlab/extern/engines/python
python setup.py install
```

### 3. Basic Usage

```python
from simple_patch_extractor import extract_patches
import numpy as np

# Define patch locations (in pixels)
patch_locations = [
    [100, 100],   # [x, y] coordinates
    [200, 200],
    [300, 300]
]

# Extract patches
result = extract_patches(
    patch_locations=patch_locations,
    image_name='image001',  # Your image file name (without .iml extension)
    verbose=True
)

# Access results
patches = result['images']           # List of numpy arrays
full_image = result['full_image']    # Full image as numpy array
background = result['background_intensity']  # Mean intensity

print(f"Extracted {len(patches)} patches")
print(f"Valid patches: {result['metadata']['num_valid_patches']}")
```

## 📖 Detailed Usage

### Function Parameters

```python
extract_patches(
    patch_locations,      # List or numpy array: N x 2 coordinates
    image_name,           # String: image file name (without extension)
    resources_dir=None,   # String: path to images (auto-detects if None)
    stim_set='/VHsubsample_20160105/',  # String: subdirectory
    patch_size=(200.0, 200.0),         # Tuple: patch size in microns
    image_size=(1536, 1024),           # Tuple: full image size in pixels
    pixel_size=6.6,                    # Float: microns per pixel
    normalize=True,                    # Bool: normalize image intensity
    verbose=False                      # Bool: show progress messages
)
```

### Return Value Structure

```python
{
    'images': [                    # List of numpy arrays (patches)
        np.array(...),             # Patch 1 (or None if invalid)
        np.array(...),             # Patch 2
        ...
    ],
    'full_image': np.array(...),   # Full image as 2D numpy array
    'background_intensity': 0.5,   # Mean intensity (float)
    'patch_info': [                # List of patch information dicts
        {
            'location': [100, 100],      # Patch center coordinates
            'actual_size': [30, 30],     # Actual patch size in pixels
            'clipped': False,            # Whether patch was clipped
            'valid': True                # Whether patch is valid
        },
        ...
    ],
    'metadata': {                  # Processing metadata
        'image_name': 'image001',
        'image_file_path': '/path/to/image001.iml',
        'num_valid_patches': 3,
        'num_clipped_patches': 0,
        'processing_time': '2025-07-01 15:30:45'
    }
}
```

## 🔧 Setup for Different Systems

### Option 1: Current Directory Setup (Simplest)
Place all files in your working directory:
```
your_project/
├── simple_patch_extractor.py
├── getNaturalImagePatchFromLocation2_python.m
├── your_analysis_script.py
└── natural_images/
    └── VHsubsample_20160105/
        ├── imkimage001.iml
        ├── imkimage002.iml
        └── ...
```

### Option 2: Separate Directories
```
your_project/
├── python/
│   ├── simple_patch_extractor.py
│   └── your_analysis_script.py
├── matlab/
│   └── getNaturalImagePatchFromLocation2_python.m
└── data/
    └── natural_images/
        └── ...
```

Then specify paths explicitly:
```python
result = extract_patches(
    patch_locations=locations,
    image_name='image001',
    matlab_function_dir='/path/to/matlab/',
    resources_dir='/path/to/data/natural_images/'
)
```

## 🧪 Testing Your Setup

### 1. Test MATLAB Engine API
```python
try:
    import matlab.engine
    print("✓ MATLAB Engine API is available")
    
    # Test engine startup
    eng = matlab.engine.start_matlab()
    print("✓ MATLAB engine started successfully")
    eng.quit()
    print("✓ MATLAB engine stopped successfully")
    
except ImportError:
    print("✗ MATLAB Engine API not installed")
except Exception as e:
    print(f"✗ MATLAB engine error: {e}")
```

### 2. Test Function Availability
```python
from simple_patch_extractor import extract_patches

# Test with minimal parameters (will fail if images not found, but tests function loading)
try:
    result = extract_patches([[100, 100]], 'test_image', verbose=True)
except RuntimeError as e:
    if "not found" in str(e).lower():
        print("✓ Function loaded successfully (image not found is expected)")
    else:
        print(f"✗ Function error: {e}")
```

### 3. Full Integration Test
```python
# Replace 'your_actual_image' with a real image name from your dataset
test_locations = [[500, 500]]  # Image center coordinates

try:
    result = extract_patches(
        patch_locations=test_locations,
        image_name='your_actual_image',  # UPDATE THIS
        verbose=True
    )
    print("✓ Full integration test successful!")
    print(f"  Extracted {result['metadata']['num_valid_patches']} valid patches")
    
except Exception as e:
    print(f"✗ Integration test failed: {e}")
```

## 🐛 Troubleshooting

### Common Issues

1. **"MATLAB Engine API not installed"**
   - Install using `python setup.py install` in MATLAB's extern/engines/python directory
   - Ensure Python version compatibility with your MATLAB version

2. **"Could not auto-detect natural images directory"**
   - Specify `resources_dir` parameter explicitly
   - Check that image files exist and are accessible

3. **"Image file not found"**
   - Verify image file naming (with/without 'imk' prefix)
   - Check file extension (.iml)
   - Ensure correct `stim_set` subdirectory

4. **"MATLAB function not found"**
   - Ensure `getNaturalImagePatchFromLocation2_python.m` is in current directory
   - Or specify `matlab_function_dir` parameter

5. **Empty or invalid patches**
   - Check that patch locations are within image bounds
   - Verify image size parameters match your data
   - Use verbose mode to see detailed error messages

### Debug Mode
Enable verbose output for detailed information:
```python
result = extract_patches(
    patch_locations=locations,
    image_name='image001',
    verbose=True  # Shows detailed progress and error information
)
```

## 🔄 Integration with Existing Analysis

### With Spike-Triggered Average Analysis
```python
import numpy as np
from simple_patch_extractor import extract_patches

# Assume you have STA analysis results
sta_center = [peak_x, peak_y]  # From your STA analysis
patch_radius = 50  # pixels

# Generate patch locations around STA center
locations = []
for dx in [-patch_radius, 0, patch_radius]:
    for dy in [-patch_radius, 0, patch_radius]:
        locations.append([sta_center[0] + dx, sta_center[1] + dy])

# Extract patches
patches_result = extract_patches(
    patch_locations=locations,
    image_name=your_image_name,
    patch_size=(100, 100)  # microns
)

# Use patches in your moment analysis
for i, patch in enumerate(patches_result['images']):
    if patch is not None:
        # Compute moments for this patch
        mean_intensity = np.mean(patch)
        second_moment = np.mean(patch**2)
        # ... continue with your analysis
```

### Batch Processing Multiple Images
```python
image_names = ['image001', 'image002', 'image003']
patch_locations = [[100, 100], [200, 200]]  # Same locations for all images

all_results = {}
for image_name in image_names:
    try:
        result = extract_patches(
            patch_locations=patch_locations,
            image_name=image_name,
            verbose=False
        )
        all_results[image_name] = result
        print(f"✓ Processed {image_name}: {result['metadata']['num_valid_patches']} patches")
    except Exception as e:
        print(f"✗ Failed to process {image_name}: {e}")

# Analyze results across all images
all_patches = []
for image_name, result in all_results.items():
    for patch in result['images']:
        if patch is not None:
            all_patches.append(patch)

print(f"Total valid patches collected: {len(all_patches)}")
```

## 📚 Next Steps

1. **Test the basic setup** with your actual image data
2. **Integrate with your existing analysis pipeline**
3. **Optimize parameters** (patch size, locations) for your specific experiments
4. **Add error handling** appropriate for your use case
5. **Consider batch processing** for large datasets

## 💡 Tips for Optimal Performance

- **Reuse MATLAB engine**: For multiple extractions, consider keeping the engine running
- **Validate inputs**: Check patch locations are within image bounds before calling
- **Handle errors gracefully**: Use try-except blocks for robust analysis pipelines
- **Monitor memory usage**: Large images and many patches can consume significant memory
- **Use verbose mode** during development, disable for production runs

The Python interface provides the same functionality as the MATLAB version while offering the convenience of Python data structures and error handling!
