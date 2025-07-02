"""
Simple Python Interface for Natural Image Patch Extraction

This module provides a simplified Python interface to the universal MATLAB-based 
natural image patch extraction function. The same MATLAB function can be called 
directly from MATLAB or through this Python interface - no separate versions needed!

The universal function automatically detects whether it's being called from MATLAB 
or Python and optimizes its behavior accordingly.

Requirements:
    - MATLAB R2018b or later
    - MATLAB Engine API for Python
    - numpy
    - getNaturalImagePatchFromLocation2_universal.m (universal MATLAB function)

Installation of MATLAB Engine for Python:
    1. Navigate to your MATLAB installation directory
    2. cd "matlabroot/extern/engines/python"
    3. python setup.py install

Example Usage:
    from simple_patch_extractor import extract_patches
    
    # Same function used in MATLAB and Python!
    patches = extract_patches(
        patch_locations=[[100, 100], [200, 200]],
        image_name='image001',
        verbose=True
    )
"""

import numpy as np
import sys
import os
from typing import List, Tuple, Optional, Dict, Union, Any

def extract_patches(patch_locations: Union[List[List[float]], np.ndarray],
                   image_name: str,
                   resources_dir: Optional[str] = None,
                   stim_set: str = '/VHsubsample_20160105/',
                   patch_size: Tuple[float, float] = (200.0, 200.0),
                   image_size: Tuple[int, int] = (1536, 1024),
                   pixel_size: float = 6.6,
                   normalize: bool = True,
                   verbose: bool = False,
                   matlab_function_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract natural image patches using MATLAB backend.
    
    Args:
        patch_locations: N x 2 array of [x, y] coordinates for patch centers (pixels)
        image_name: Name of the natural image file (without extension)
        resources_dir: Directory containing natural image files (auto-detects if None)
        stim_set: Subdirectory within resources_dir
        patch_size: Size of patches in microns [height, width]
        image_size: Full image dimensions [height, width]
        pixel_size: Microns per pixel conversion factor
        normalize: Whether to normalize image intensity
        verbose: Display progress messages
        matlab_function_dir: Directory containing the MATLAB function (auto-detects if None)
        
    Returns:
        Dictionary with keys:
            - 'images': List of numpy arrays (patches)
            - 'full_image': Full image as numpy array
            - 'background_intensity': Mean intensity (float)
            - 'patch_info': List of dicts with patch information
            - 'metadata': Dictionary with processing metadata
            
    Raises:
        ImportError: If MATLAB Engine API is not available
        RuntimeError: If MATLAB function execution fails
        ValueError: If input parameters are invalid
    """
    
    # Check if MATLAB Engine API is available
    try:
        import matlab.engine
        import matlab
    except ImportError:
        raise ImportError(
            "MATLAB Engine API for Python is not installed. "
            "Please install it by running 'python setup.py install' "
            "in your MATLAB installation's extern/engines/python directory."
        )
    
    # Validate inputs
    patch_locations = _validate_patch_locations(patch_locations)
    
    if verbose:
        print(f"Starting MATLAB engine for patch extraction...")
        print(f"Image: {image_name}, Patches: {len(patch_locations)}")
    
    # Start MATLAB engine
    eng = None
    try:
        eng = matlab.engine.start_matlab()
        
        # Add function directory to MATLAB path
        if matlab_function_dir:
            eng.addpath(matlab_function_dir, nargout=0)
        else:
            # Try to find function in current directory or common locations
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_dirs = [
                current_dir,
                os.path.join(current_dir, '..'),
                os.path.join(current_dir, 'matlab_functions'),
            ]
            
            for dir_path in possible_dirs:
                matlab_file = os.path.join(dir_path, 'getNaturalImagePatchFromLocation2_universal.m')
                if os.path.exists(matlab_file):
                    eng.addpath(dir_path, nargout=0)
                    if verbose:
                        print(f"Added MATLAB path: {dir_path}")
                    break
            else:
                if verbose:
                    print("Warning: Could not auto-detect MATLAB function location")
        
        # Convert inputs to MATLAB format
        patch_locations_matlab = matlab.double(patch_locations.tolist())
        
        # Prepare keyword arguments
        kwargs = {
            'patchSize': matlab.double(list(patch_size)),
            'imageSize': matlab.double(list(image_size)),
            'pixelSize': float(pixel_size),
            'normalize': bool(normalize),
            'verbose': bool(verbose),
            'stimSet': stim_set
        }
        
        if resources_dir:
            kwargs['resourcesDir'] = resources_dir
        
        if verbose:
            print("Calling MATLAB function...")
        
        # Call universal MATLAB function
        result = eng.getNaturalImagePatchFromLocation2_universal(
            patch_locations_matlab, 
            image_name,
            **kwargs,
            nargout=1
        )
        
        if verbose:
            print("Converting MATLAB results to Python format...")
        
        # Convert result to Python format
        python_result = _convert_matlab_result(result)
        
        if verbose:
            print(f"Successfully extracted {len(python_result['images'])} patches")
            print(f"Valid patches: {python_result['metadata']['num_valid_patches']}")
        
        return python_result
        
    except Exception as e:
        error_msg = f"MATLAB execution failed: {str(e)}"
        if verbose:
            print(f"Error: {error_msg}")
        raise RuntimeError(error_msg)
        
    finally:
        # Clean up MATLAB engine
        if eng is not None:
            try:
                eng.quit()
            except:
                pass  # Ignore errors during cleanup


def _validate_patch_locations(patch_locations: Union[List, np.ndarray]) -> np.ndarray:
    """Validate and convert patch locations to numpy array."""
    if isinstance(patch_locations, list):
        patch_locations = np.array(patch_locations, dtype=float)
    elif isinstance(patch_locations, np.ndarray):
        patch_locations = patch_locations.astype(float)
    else:
        raise ValueError("patch_locations must be a list or numpy array")
    
    if patch_locations.ndim != 2 or patch_locations.shape[1] != 2:
        raise ValueError("patch_locations must be N x 2 array of [x, y] coordinates")
    
    if not np.all(np.isfinite(patch_locations)):
        raise ValueError("patch_locations must contain finite values")
    
    return patch_locations


def _convert_matlab_result(matlab_result) -> Dict[str, Any]:
    """Convert MATLAB result structure to Python dictionary."""
    result = {}
    
    try:
        # Convert images from cell array to list of numpy arrays
        images = []
        if hasattr(matlab_result, 'images') and matlab_result['images']:
            for i in range(len(matlab_result['images'])):
                img = matlab_result['images'][i]
                if img.size > 0:  # Check if image is not empty
                    images.append(np.array(img))
                else:
                    images.append(None)
        result['images'] = images
        
        # Convert full image
        if hasattr(matlab_result, 'fullImage'):
            result['full_image'] = np.array(matlab_result['fullImage'])
        
        # Convert scalar values
        if hasattr(matlab_result, 'backgroundIntensity'):
            result['background_intensity'] = float(matlab_result['backgroundIntensity'])
        
        # Convert patch info
        patch_info = []
        if hasattr(matlab_result, 'patchInfo'):
            patch_info_struct = matlab_result['patchInfo']
            num_patches = len(patch_info_struct['location'])
            
            for i in range(num_patches):
                info = {
                    'location': [float(x) for x in patch_info_struct['location'][i]],
                    'actual_size': [int(x) for x in patch_info_struct['actualSize'][i]],
                    'clipped': bool(patch_info_struct['clipped'][i]),
                    'valid': bool(patch_info_struct['valid'][i])
                }
                patch_info.append(info)
        result['patch_info'] = patch_info
        
        # Convert metadata
        metadata = {}
        if hasattr(matlab_result, 'metadata'):
            md = matlab_result['metadata']
            metadata = {
                'image_name': str(md['imageName']) if 'imageName' in md else '',
                'image_file_path': str(md['imageFilePath']) if 'imageFilePath' in md else '',
                'num_valid_patches': int(md['numValidPatches']) if 'numValidPatches' in md else 0,
                'num_clipped_patches': int(md['numClippedPatches']) if 'numClippedPatches' in md else 0,
                'processing_time': str(md['processingTime']) if 'processingTime' in md else ''
            }
        result['metadata'] = metadata
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Failed to convert MATLAB result: {str(e)}")


# Example and testing functions
def test_basic_extraction():
    """Test basic patch extraction functionality."""
    print("=== Testing Basic Patch Extraction ===")
    
    # Define test patch locations
    patch_locations = [
        [100.0, 100.0],
        [200.0, 200.0],
        [300.0, 300.0]
    ]
    
    try:
        result = extract_patches(
            patch_locations=patch_locations,
            image_name='image001',  # Replace with actual image name
            verbose=True
        )
        
        print(f"✓ Successfully extracted patches")
        print(f"  Total patches: {len(result['images'])}")
        print(f"  Valid patches: {result['metadata']['num_valid_patches']}")
        print(f"  Background intensity: {result['background_intensity']:.4f}")
        
        # Display patch info
        for i, info in enumerate(result['patch_info']):
            status = "Valid" if info['valid'] else "Invalid"
            clipped = " (Clipped)" if info['clipped'] else ""
            print(f"  Patch {i+1}: {status}{clipped}, Size: {info['actual_size']}")
        
        return result
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return None


def test_custom_parameters():
    """Test with custom parameters."""
    print("\n=== Testing Custom Parameters ===")
    
    patch_locations = np.array([
        [150, 150],
        [400, 400]
    ])
    
    try:
        result = extract_patches(
            patch_locations=patch_locations,
            image_name='image001',  # Replace with actual image name
            patch_size=(150.0, 150.0),
            normalize=True,
            verbose=True
        )
        
        print(f"✓ Custom parameter test successful")
        return result
        
    except Exception as e:
        print(f"✗ Custom parameter test failed: {e}")
        return None


if __name__ == "__main__":
    print("Natural Image Patch Extractor - Simple Python Interface")
    print("=" * 60)
    
    # Check if MATLAB Engine API is available
    try:
        import matlab.engine
        print("✓ MATLAB Engine API is available")
    except ImportError:
        print("✗ MATLAB Engine API is not available")
        print("Please install it following the instructions in the module docstring")
        sys.exit(1)
    
    # Run tests
    print(f"\nCurrent working directory: {os.getcwd()}")
    print("Make sure you have:")
    print("1. MATLAB function 'getNaturalImagePatchFromLocation2_universal.m' in current directory")
    print("2. Natural image files accessible")
    print("3. Valid image name for testing")
    print("4. Same function works from both MATLAB and Python!")
    print()
    
    # Uncomment to run tests (update image_name first!)
    # basic_result = test_basic_extraction()
    # custom_result = test_custom_parameters()
    
    print("\nTo use this module:")
    print("1. Update the image_name in test functions to match your data")
    print("2. Ensure natural image files are accessible")
    print("3. Run: from simple_patch_extractor import extract_patches")
