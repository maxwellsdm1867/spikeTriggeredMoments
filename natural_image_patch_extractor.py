import matlab.engine
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import warnings

class NaturalImagePatchExtractor:
    """
    Python wrapper for enhanced natural image patch extraction from MATLAB.
    
    This class provides a Python interface to the universal getNaturalImagePatchFromLocation_universal
    function, allowing seamless integration between Python and MATLAB for visual neuroscience
    experiments.
    
    Features:
    - Automatic MATLAB engine management
    - Pythonic parameter handling
    - Numpy array integration
    - Cross-platform path handling
    - Comprehensive error handling
    
    Examples:
        >>> extractor = NaturalImagePatchExtractor()
        >>> patches = extractor.extract_patches(
        ...     patch_locations=[[100, 100], [200, 200]], 
        ...     image_name='image001'
        ... )
        >>> print(f"Extracted {len(patches['images'])} patches")
    """
    
    def __init__(self, matlab_function_path: Optional[str] = None, start_matlab: bool = True):
        """
        Initialize the Natural Image Patch Extractor.
        
        Args:
            matlab_function_path: Path to the MATLAB function directory
            start_matlab: Whether to start MATLAB engine immediately
        """
        self.eng = None
        self.matlab_function_path = matlab_function_path
        
        if start_matlab:
            self.start_matlab_engine()
    
    def start_matlab_engine(self):
        """Start the MATLAB engine and add function path."""
        try:
            print("Starting MATLAB engine...")
            self.eng = matlab.engine.start_matlab()
            
            # Add function path if specified
            if self.matlab_function_path:
                self.eng.addpath(str(Path(self.matlab_function_path).resolve()))
            else:
                # Try to find the function in common locations
                current_dir = Path(__file__).parent
                possible_paths = [
                    current_dir,
                    current_dir.parent,
                    current_dir / "matlab_functions",
                    Path.home() / "Documents" / "MATLAB",
                ]
                
                for path in possible_paths:
                    if (path / "getNaturalImagePatchFromLocation_universal.m").exists():
                        self.eng.addpath(str(path.resolve()))
                        print(f"Added MATLAB path: {path}")
                        break
                else:
                    warnings.warn("Could not find MATLAB function. Please specify matlab_function_path.")
            
            print("MATLAB engine started successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to start MATLAB engine: {e}")
    
    def stop_matlab_engine(self):
        """Stop the MATLAB engine."""
        if self.eng:
            self.eng.quit()
            self.eng = None
            print("MATLAB engine stopped.")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.eng:
            self.start_matlab_engine()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_matlab_engine()
    
    def extract_patches(self, 
                       patch_locations: Union[List[List[float]], np.ndarray],
                       image_name: str,
                       resources_dir: Optional[str] = None,
                       stim_set: str = '/VHsubsample_20160105/',
                       patch_size: Tuple[float, float] = (200.0, 200.0),
                       image_size: Tuple[int, int] = (1536, 1024),
                       pixel_size: float = 6.6,
                       normalize: bool = True,
                       verbose: bool = False) -> Dict:
        """
        Extract natural image patches at specified locations.
        
        Args:
            patch_locations: N x 2 array of [x, y] coordinates for patch centers (in pixels)
            image_name: String identifier for the natural image file (without extension)
            resources_dir: Base directory containing natural image files (auto-detects if None)
            stim_set: Subdirectory within resources_dir
            patch_size: Size of extracted patches in microns [height, width]
            image_size: Full image dimensions in pixels [height, width]
            pixel_size: Microns per pixel conversion factor
            normalize: Whether to normalize image intensity
            verbose: Display progress messages
        
        Returns:
            Dictionary containing:
                - 'images': List of extracted image patches as numpy arrays
                - 'full_image': Full natural image as numpy array
                - 'background_intensity': Mean intensity of full image
                - 'patch_info': Information about extracted patches
                - 'metadata': Processing metadata and parameters
        
        Raises:
            RuntimeError: If MATLAB engine is not available or function fails
            ValueError: If input parameters are invalid
        """
        if not self.eng:
            raise RuntimeError("MATLAB engine not started. Call start_matlab_engine() first.")
        
        # Validate and convert inputs
        patch_locations = self._validate_patch_locations(patch_locations)
        
        # Convert numpy arrays to MATLAB format
        patch_locations_matlab = matlab.double(patch_locations.tolist())
        
        try:
            # Call MATLAB function
            if verbose:
                print(f"Calling MATLAB function for image: {image_name}")
            
            # Prepare arguments
            args = [patch_locations_matlab, image_name]
            kwargs = {
                'patchSize': matlab.double(list(patch_size)),
                'imageSize': matlab.double(list(image_size)),
                'pixelSize': pixel_size,
                'normalize': normalize,
                'verbose': verbose,
                'stimSet': stim_set
            }
            
            if resources_dir:
                kwargs['resourcesDir'] = resources_dir
            
            # Call the MATLAB function
            result = self.eng.getNaturalImagePatchFromLocation_universal(*args, **kwargs)
            
            # Convert MATLAB result to Python format
            return self._convert_matlab_result(result)
            
        except Exception as e:
            raise RuntimeError(f"MATLAB function call failed: {e}")
    
    def _validate_patch_locations(self, patch_locations: Union[List, np.ndarray]) -> np.ndarray:
        """Validate and convert patch locations to numpy array."""
        if isinstance(patch_locations, list):
            patch_locations = np.array(patch_locations)
        
        if not isinstance(patch_locations, np.ndarray):
            raise ValueError("patch_locations must be a list or numpy array")
        
        if patch_locations.ndim != 2 or patch_locations.shape[1] != 2:
            raise ValueError("patch_locations must be N x 2 array of [x, y] coordinates")
        
        if not np.all(np.isfinite(patch_locations)):
            raise ValueError("patch_locations must contain finite values")
        
        return patch_locations.astype(float)
    
    def _convert_matlab_result(self, matlab_result) -> Dict:
        """Convert MATLAB result structure to Python dictionary."""
        result = {}
        
        # Convert images from cell array to list of numpy arrays
        if hasattr(matlab_result, 'images') and matlab_result.images:
            images = []
            for i in range(len(matlab_result.images)):
                img = matlab_result.images[i]
                if img.size:  # Check if image is not empty
                    images.append(np.array(img))
                else:
                    images.append(None)
            result['images'] = images
        else:
            result['images'] = []
        
        # Convert full image
        if hasattr(matlab_result, 'fullImage'):
            result['full_image'] = np.array(matlab_result.fullImage)
        
        # Convert scalar values
        if hasattr(matlab_result, 'backgroundIntensity'):
            result['background_intensity'] = float(matlab_result.backgroundIntensity)
        
        # Convert patch info
        if hasattr(matlab_result, 'patchInfo'):
            patch_info = []
            for i in range(len(matlab_result.patchInfo)):
                info = matlab_result.patchInfo[i]
                patch_info.append({
                    'location': [float(info['location'][0]), float(info['location'][1])],
                    'actual_size': [int(info['actualSize'][0]), int(info['actualSize'][1])],
                    'clipped': bool(info['clipped']),
                    'valid': bool(info['valid'])
                })
            result['patch_info'] = patch_info
        
        # Convert metadata
        if hasattr(matlab_result, 'metadata'):
            metadata = matlab_result.metadata
            result['metadata'] = {
                'image_name': str(metadata['imageName']) if 'imageName' in metadata else '',
                'image_file_path': str(metadata['imageFilePath']) if 'imageFilePath' in metadata else '',
                'num_valid_patches': int(metadata['numValidPatches']) if 'numValidPatches' in metadata else 0,
                'num_clipped_patches': int(metadata['numClippedPatches']) if 'numClippedPatches' in metadata else 0,
                'processing_time': str(metadata['processingTime']) if 'processingTime' in metadata else ''
            }
        
        return result


def extract_natural_image_patches(patch_locations: Union[List[List[float]], np.ndarray],
                                 image_name: str,
                                 **kwargs) -> Dict:
    """
    Convenience function for extracting natural image patches.
    
    This function provides a simple interface that automatically manages the MATLAB engine.
    
    Args:
        patch_locations: N x 2 array of [x, y] coordinates for patch centers
        image_name: String identifier for the natural image file
        **kwargs: Additional arguments passed to extract_patches method
    
    Returns:
        Dictionary containing extracted patches and metadata
    
    Example:
        >>> patches = extract_natural_image_patches(
        ...     [[100, 100], [200, 200]], 
        ...     'image001',
        ...     verbose=True
        ... )
    """
    with NaturalImagePatchExtractor() as extractor:
        return extractor.extract_patches(patch_locations, image_name, **kwargs)


# Example usage and testing functions
def example_basic_usage():
    """Example of basic patch extraction usage."""
    print("=== Basic Natural Image Patch Extraction Example ===")
    
    # Define patch locations
    patch_locations = [
        [100, 100],   # Patch 1: [x, y] coordinates
        [200, 200],   # Patch 2
        [400, 300],   # Patch 3
    ]
    
    try:
        # Extract patches
        result = extract_natural_image_patches(
            patch_locations=patch_locations,
            image_name='image001',
            verbose=True
        )
        
        print(f"Successfully extracted {len(result['images'])} patches")
        print(f"Valid patches: {result['metadata']['num_valid_patches']}")
        print(f"Background intensity: {result['background_intensity']:.4f}")
        
        # Display patch information
        for i, info in enumerate(result['patch_info']):
            if info['valid']:
                print(f"Patch {i+1}: Size {info['actual_size']}, Clipped: {info['clipped']}")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def example_advanced_usage():
    """Example of advanced usage with custom parameters."""
    print("=== Advanced Natural Image Patch Extraction Example ===")
    
    patch_locations = np.array([
        [150, 150],
        [300, 300],
        [450, 450],
        [600, 600]
    ])
    
    try:
        # Use context manager for automatic resource cleanup
        with NaturalImagePatchExtractor() as extractor:
            result = extractor.extract_patches(
                patch_locations=patch_locations,
                image_name='image001',
                patch_size=(150, 150),  # Custom patch size
                normalize=True,
                verbose=True
            )
        
        print(f"Extracted {len(result['images'])} patches")
        
        # Analyze patches
        valid_patches = [img for img in result['images'] if img is not None]
        if valid_patches:
            mean_intensities = [np.mean(patch) for patch in valid_patches]
            print(f"Mean patch intensities: {mean_intensities}")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    print("Natural Image Patch Extractor - Python Interface")
    print("=" * 50)
    
    # Run examples
    basic_result = example_basic_usage()
    print()
    advanced_result = example_advanced_usage()
