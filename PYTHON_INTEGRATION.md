# Python Integration for Natural Image Patch Extraction

This document provides comprehensive information about using the enhanced natural image patch extraction functionality from Python.

## Overview

The enhanced patch extraction toolkit now includes robust Python integration through the MATLAB Engine API, offering two complementary interfaces:

1. **Class-based interface** (`NaturalImagePatchExtractor`) - Object-oriented approach with persistent MATLAB engine
2. **Function-based interface** (`extract_patches`) - Simple function call for one-off extractions

## Key Enhancements

### MATLAB Function Improvements (`getNaturalImagePatchFromLocation2_python.m`)

- **Enhanced path expansion**: Automatic handling of `~`, `$HOME`, and other environment variables
- **Robust directory auto-detection**: Searches multiple common locations for natural image datasets
- **Python-optimized data structures**: Ensures compatibility with MATLAB Engine API
- **Comprehensive error handling**: Structured error messages with specific error codes
- **Cross-platform compatibility**: Works on Windows, macOS, and Linux

### Python Interface Features

- **Automatic MATLAB engine management**: Handles startup, execution, and cleanup
- **Numpy array integration**: Seamless conversion between Python and MATLAB data types
- **Type validation**: Comprehensive input validation with helpful error messages
- **Flexible configuration**: Support for custom directories, patch sizes, and processing parameters
- **Progress monitoring**: Optional verbose output for debugging and monitoring

## Installation

### Prerequisites

1. **MATLAB R2018b or later** with Image Processing Toolbox
2. **MATLAB Engine API for Python**
3. **Python packages**: `numpy`, `pathlib`

### Installing MATLAB Engine for Python

```bash
# Navigate to your MATLAB installation directory
cd "matlabroot/extern/engines/python"

# Install the engine (replace with your Python executable if needed)
python setup.py install

# Or install in development mode
python setup.py develop
```

### Verify Installation

```python
import matlab.engine
print("MATLAB Engine API successfully installed!")
```

## Usage Examples

### Class-Based Interface (Recommended for Multiple Extractions)

```python
from natural_image_patch_extractor import NaturalImagePatchExtractor
import numpy as np

# Initialize extractor
extractor = NaturalImagePatchExtractor()

# Define patch locations (x, y coordinates in pixels)
patch_locations = [
    [100, 100],
    [200, 200], 
    [300, 300]
]

# Extract patches
result = extractor.extract_patches(
    patch_locations=patch_locations,
    image_name='image001',
    patch_size=(150.0, 150.0),  # Size in microns
    verbose=True
)

# Access results
patches = result['images']  # List of numpy arrays
full_image = result['full_image']  # Full image as numpy array
background = result['background_intensity']  # Mean intensity

print(f"Extracted {len(patches)} patches")
print(f"Background intensity: {background:.4f}")

# Process individual patches
for i, patch in enumerate(patches):
    if patch.size > 0:  # Check if patch is valid
        print(f"Patch {i}: shape {patch.shape}, mean intensity {patch.mean():.4f}")

# Clean up
extractor.close()
```

### Function-Based Interface (Simple One-Off Extractions)

```python
from simple_patch_extractor import extract_patches
import numpy as np

# Simple extraction
result = extract_patches(
    patch_locations=[[100, 100], [200, 200]],
    image_name='image001',
    patch_size=(200.0, 200.0),
    normalize=True,
    verbose=True
)

# Access results (same structure as class-based interface)
patches = result['images']
metadata = result['metadata']

print(f"Processing time: {metadata['processing_time']}")
print(f"Valid patches: {metadata['num_valid_patches']}")
```

### Advanced Configuration

```python
# Custom directory and parameters
result = extractor.extract_patches(
    patch_locations=patch_locations,
    image_name='custom_image',
    resources_dir='/path/to/your/natural_images/',
    stim_set='/custom_dataset/',
    patch_size=(100.0, 100.0),
    image_size=(2048, 1536),  # Custom image dimensions
    pixel_size=5.0,  # Different pixel size
    normalize=False,  # Skip normalization
    verbose=True
)
```

## Error Handling

The enhanced system provides structured error handling with specific error codes:

```python
from simple_patch_extractor import extract_patches

try:
    result = extract_patches(
        patch_locations=[[100, 100]],
        image_name='nonexistent_image'
    )
except Exception as e:
    # Error messages include specific error codes and suggestions
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    
    # Common error types:
    # - NaturalImagePatch:DirectoryNotFound
    # - NaturalImagePatch:FileNotFound  
    # - NaturalImagePatch:InvalidInput
    # - NaturalImagePatch:LoadError
```

## Directory Auto-Detection

The system automatically searches for natural image datasets in common locations:

```python
# These directories are searched automatically (in order):
# 1. ~/Documents/NaturalImages
# 2. ~/Data/NaturalImages  
# 3. /data/natural_images
# 4. /home/shared/natural_images
# 5. ./natural_images
# 6. ../data/natural_images

# You can also specify a custom directory:
result = extract_patches(
    patch_locations=patch_locations,
    image_name='image001',
    resources_dir='/custom/path/to/images/'
)
```

## Batch Processing Example

```python
from natural_image_patch_extractor import NaturalImagePatchExtractor
import numpy as np

# Initialize once for multiple extractions
extractor = NaturalImagePatchExtractor()

# Process multiple images
image_names = ['image001', 'image002', 'image003']
patch_locations = np.random.randint(50, 500, (10, 2))  # Random locations

results = {}
for image_name in image_names:
    try:
        result = extractor.extract_patches(
            patch_locations=patch_locations,
            image_name=image_name,
            verbose=False  # Reduce output for batch processing
        )
        results[image_name] = result
        print(f"✓ Processed {image_name}: {len(result['images'])} patches")
    except Exception as e:
        print(f"✗ Failed to process {image_name}: {e}")

# Clean up
extractor.close()
```

## Performance Tips

1. **Use class-based interface for multiple extractions** to avoid MATLAB engine startup overhead
2. **Batch patch locations** when possible to reduce function call overhead
3. **Set `verbose=False`** for production code to improve performance
4. **Consider patch size** - larger patches take more memory but fewer boundary issues
5. **Pre-validate inputs** to catch errors early

## Troubleshooting

### Common Issues

1. **MATLAB Engine API not found**
   ```
   ImportError: No module named 'matlab.engine'
   ```
   - Install MATLAB Engine API (see installation section)

2. **Image files not found**
   ```
   NaturalImagePatch:FileNotFound: Image file not found
   ```
   - Check that `.iml` files exist in the specified directory
   - Verify the `stimSet` parameter matches your directory structure
   - Use `verbose=True` to see which paths were searched

3. **Directory auto-detection fails**
   ```
   NaturalImagePatch:DirectoryNotFound: Could not auto-detect natural images directory
   ```
   - Specify `resources_dir` parameter explicitly
   - Check file permissions
   - Ensure image files are in the expected subdirectory structure

4. **MATLAB engine startup fails**
   - Check MATLAB license availability
   - Ensure MATLAB is properly installed
   - Try starting MATLAB manually first

### Debug Mode

Enable verbose output and use try-catch blocks for debugging:

```python
try:
    result = extract_patches(
        patch_locations=[[100, 100]],
        image_name='image001',
        verbose=True  # Enable detailed output
    )
except Exception as e:
    print(f"Full error details: {e}")
    import traceback
    traceback.print_exc()
```

## Testing

Run the comprehensive test suite to verify your installation:

```bash
python test_enhanced_patch_extraction.py
```

This will test:
- MATLAB function direct calling
- Class-based Python interface  
- Function-based Python interface
- Path expansion functionality
- Error handling robustness

## Integration with Spike-Triggered Moments

The enhanced patch extraction integrates seamlessly with the main spike-triggered moments analysis:

```python
# Extract patches for specific stimulus locations
stimulus_locations = get_stimulus_locations_from_spikes(spike_data)
patches = extract_patches(
    patch_locations=stimulus_locations,
    image_name='experimental_image',
    patch_size=(200.0, 200.0)
)

# Use patches in main analysis
moments_result = calculate_spike_triggered_moments(
    patches['images'], 
    spike_times, 
    stimulus_times
)
```

## File Structure

```
spikeTriggeredMoments/
├── getNaturalImagePatchFromLocation2_python.m      # Enhanced MATLAB function
├── natural_image_patch_extractor.py                # Class-based Python interface
├── simple_patch_extractor.py                       # Function-based Python interface
├── python_examples.py                              # Usage examples
├── test_enhanced_patch_extraction.py               # Test suite
└── PYTHON_INTEGRATION.md                           # This documentation
```

For more information, see:
- `NATURAL_IMAGE_PATCH_USAGE.md` - MATLAB usage guide
- `README.md` - Main project documentation
- `python_examples.py` - Additional usage examples
