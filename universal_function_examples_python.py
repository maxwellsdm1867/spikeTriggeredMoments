#!/usr/bin/env python3
"""
Universal Function Usage Examples - Python Side

This script demonstrates how to use the universal 
getNaturalImagePatchFromLocation2_universal function from Python.

The SAME MATLAB function is called from both Python and MATLAB - no separate versions!
"""

import numpy as np
import sys
import os

def test_universal_function_direct():
    """Test calling the universal MATLAB function directly via MATLAB Engine."""
    print("=== Universal Natural Image Patch Extraction - Python Examples ===\n")
    
    try:
        import matlab.engine
        import matlab
    except ImportError:
        print("✗ MATLAB Engine API not available")
        print("Please install it following the installation instructions")
        return False
    
    # Define patch locations
    patch_locations = [
        [100.0, 100.0],
        [200.0, 200.0], 
        [400.0, 300.0]
    ]
    
    image_name = 'image001'  # Replace with your actual image name
    
    print("--- Example 1: Direct Universal Function Call ---")
    eng = None
    try:
        # Start MATLAB engine
        print("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        
        # Add current directory to MATLAB path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        eng.addpath(current_dir, nargout=0)
        
        # Convert patch locations to MATLAB format
        locations_matlab = matlab.double(patch_locations)
        
        # Call the UNIVERSAL function - same one used in MATLAB!
        print(f"Calling universal function for image: {image_name}")
        result = eng.getNaturalImagePatchFromLocation2_universal(
            locations_matlab, 
            image_name,
            'verbose', True,
            'patchSize', matlab.double([150.0, 150.0]),
            nargout=1
        )
        
        print("✓ Universal function call successful!")
        print(f"  Environment detected: {result['metadata']['callingEnvironment']}")
        print(f"  Valid patches: {result['metadata']['numValidPatches']}")
        print(f"  Total patches: {len(result['images'])}")
        print(f"  Background intensity: {result['backgroundIntensity']:.4f}")
        
        # Show patch information
        print("\nPatch Details:")
        for i, info in enumerate(result['patchInfo']):
            if info['valid']:
                print(f"  Patch {i+1}: Location {info['location']}, Size {info['actualSize']}")
            else:
                print(f"  Patch {i+1}: Invalid")
        
        return True
        
    except Exception as e:
        print(f"✗ Direct call failed: {e}")
        return False
        
    finally:
        if eng:
            try:
                eng.quit()
            except:
                pass

def test_with_python_wrapper():
    """Test using the Python wrapper that calls the universal function."""
    print("\n--- Example 2: Using Python Wrapper ---")
    
    try:
        # Import our wrapper
        from simple_patch_extractor import extract_patches
        
        patch_locations = [
            [100.0, 100.0],
            [200.0, 200.0]
        ]
        
        # The wrapper internally calls the universal MATLAB function
        result = extract_patches(
            patch_locations=patch_locations,
            image_name='image001',  # Replace with actual image
            patch_size=(120.0, 120.0),
            verbose=True
        )
        
        print("✓ Python wrapper call successful!")
        print(f"  Environment: {result['metadata']['calling_environment']}")
        print(f"  Valid patches: {result['metadata']['num_valid_patches']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Wrapper call failed: {e}")
        return False

def test_class_interface():
    """Test using the class-based interface."""
    print("\n--- Example 3: Using Class Interface ---")
    
    try:
        from natural_image_patch_extractor import NaturalImagePatchExtractor
        
        # Create extractor instance
        extractor = NaturalImagePatchExtractor()
        
        # Extract patches using the universal function
        patches = extractor.extract_patches(
            patch_locations=[[150, 150], [250, 250]],
            image_name='image001',
            patch_size=[100.0, 100.0],
            verbose=True
        )
        
        print("✓ Class interface call successful!")
        print(f"  Patches extracted: {len(patches['images'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Class interface failed: {e}")
        return False

def compare_with_matlab():
    """Show how the same function works in both environments."""
    print("\n--- Example 4: Universal Function Benefits ---")
    
    print("The universal function provides:")
    print("✓ Single source of truth - one function for both MATLAB and Python")
    print("✓ Automatic environment detection")
    print("✓ Consistent behavior across platforms")
    print("✓ No version synchronization issues")
    print("✓ Easier maintenance and updates")
    
    print("\nFrom MATLAB:")
    print("  result = getNaturalImagePatchFromLocation2_universal(locations, 'image001', 'verbose', true);")
    
    print("\nFrom Python:")
    print("  result = eng.getNaturalImagePatchFromLocation2_universal(locations, 'image001', 'verbose', True, nargout=1)")
    
    print("\nSame function, same results, different calling conventions!")

def main():
    """Run all examples."""
    print("Universal Natural Image Patch Extraction Examples")
    print("=" * 60)
    
    print(f"Current directory: {os.getcwd()}")
    print("Required files:")
    print("• getNaturalImagePatchFromLocation2_universal.m")
    print("• Natural image files")
    print("• MATLAB Engine API for Python")
    print()
    
    # Test direct function call
    direct_success = test_universal_function_direct()
    
    # Test wrapper approaches
    if direct_success:
        test_with_python_wrapper()
        test_class_interface()
    
    # Show benefits
    compare_with_matlab()
    
    print("\n" + "=" * 60)
    print("Universal function testing complete!")
    print("The same MATLAB function works seamlessly from both environments.")

if __name__ == "__main__":
    main()
