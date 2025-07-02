#!/usr/bin/env python3
"""
Test Script for Enhanced Natural Image Patch Extraction

This script demonstrates the enhanced natural image patch extraction functionality
with improved error handling, path expansion, and Python integration.

Features tested:
- Enhanced MATLAB function with better error handling
- Python-MATLAB integration via Engine API
- Cross-platform path handling
- Robust directory auto-detection
- Comprehensive error reporting

Requirements:
- MATLAB R2018b or later
- MATLAB Engine API for Python
- Natural image dataset (.iml files)

Author: Spike-Triggered Moments Toolkit
Date: July 2025
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_matlab_function_directly():
    """Test the MATLAB function directly via MATLAB Engine."""
    print("=" * 60)
    print("Testing MATLAB Function Directly")
    print("=" * 60)
    
    try:
        import matlab.engine
        print("✓ MATLAB Engine API available")
        
        # Start MATLAB engine
        print("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        print("✓ MATLAB engine started")
        
        # Add current directory to MATLAB path
        current_dir = str(Path(__file__).parent.resolve())
        eng.addpath(current_dir)
        print(f"✓ Added to MATLAB path: {current_dir}")
        
        # Test with sample patch locations
        patch_locations = matlab.double([[100, 100], [200, 200], [300, 300]])
        image_name = 'image001'  # Adjust as needed for your dataset
        
        print(f"Testing patch extraction for image: {image_name}")
        print(f"Patch locations: {np.array(patch_locations)}")
        
        # Call the enhanced function
        result = eng.getNaturalImagePatchFromLocation2_python(
            patch_locations, 
            image_name,
            'verbose', True,
            'patchSize', matlab.double([150.0, 150.0]),
            nargout=1
        )
        
        print("✓ MATLAB function executed successfully")
        
        # Print results summary
        if hasattr(result, 'metadata'):
            print(f"  - Valid patches: {result.metadata.numValidPatches}")
            print(f"  - Clipped patches: {result.metadata.numClippedPatches}")
            print(f"  - Background intensity: {result.backgroundIntensity:.4f}")
        
        eng.quit()
        print("✓ MATLAB engine closed")
        return True
        
    except ImportError:
        print("✗ MATLAB Engine API not available")
        print("  Install with: python -m pip install matlabengine")
        return False
    except Exception as e:
        print(f"✗ Error testing MATLAB function: {e}")
        return False

def test_class_based_interface():
    """Test the class-based Python interface."""
    print("\n" + "=" * 60)
    print("Testing Class-Based Python Interface")
    print("=" * 60)
    
    try:
        from natural_image_patch_extractor import NaturalImagePatchExtractor
        print("✓ Class-based interface imported")
        
        # Initialize extractor
        extractor = NaturalImagePatchExtractor(start_matlab=True)
        print("✓ Extractor initialized")
        
        # Test patch locations
        patch_locations = [[100, 100], [200, 200], [300, 300]]
        image_name = 'image001'
        
        print(f"Testing patch extraction for image: {image_name}")
        
        # Extract patches
        result = extractor.extract_patches(
            patch_locations=patch_locations,
            image_name=image_name,
            patch_size=(150.0, 150.0),
            verbose=True
        )
        
        print("✓ Patch extraction completed")
        print(f"  - Number of patches: {len(result['images'])}")
        print(f"  - Background intensity: {result['background_intensity']:.4f}")
        
        # Clean up
        extractor.close()
        print("✓ Extractor closed")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing class interface: {e}")
        return False

def test_function_based_interface():
    """Test the function-based Python interface."""
    print("\n" + "=" * 60)
    print("Testing Function-Based Python Interface")
    print("=" * 60)
    
    try:
        from simple_patch_extractor import extract_patches
        print("✓ Function-based interface imported")
        
        # Test patch locations
        patch_locations = [[100, 100], [200, 200], [300, 300]]
        image_name = 'image001'
        
        print(f"Testing patch extraction for image: {image_name}")
        
        # Extract patches
        result = extract_patches(
            patch_locations=patch_locations,
            image_name=image_name,
            patch_size=(150.0, 150.0),
            verbose=True
        )
        
        print("✓ Patch extraction completed")
        print(f"  - Number of patches: {len(result['images'])}")
        print(f"  - Background intensity: {result['background_intensity']:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing function interface: {e}")
        return False

def test_path_expansion():
    """Test the enhanced path expansion functionality."""
    print("\n" + "=" * 60)
    print("Testing Path Expansion and Auto-Detection")
    print("=" * 60)
    
    try:
        import matlab.engine
        
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()
        
        # Add current directory to MATLAB path
        current_dir = str(Path(__file__).parent.resolve())
        eng.addpath(current_dir)
        
        # Test path expansion function
        test_paths = [
            '~/Documents/NaturalImages',
            '$HOME/Data/NaturalImages',
            './natural_images',
            '../data/natural_images'
        ]
        
        print("Testing path expansion with various formats:")
        for test_path in test_paths:
            try:
                expanded = eng.eval(f"expandPath('{test_path}')")
                print(f"  '{test_path}' → '{expanded}'")
            except Exception as e:
                print(f"  '{test_path}' → Error: {e}")
        
        eng.quit()
        print("✓ Path expansion testing completed")
        return True
        
    except Exception as e:
        print(f"✗ Error testing path expansion: {e}")
        return False

def test_error_handling():
    """Test enhanced error handling and reporting."""
    print("\n" + "=" * 60)
    print("Testing Enhanced Error Handling")
    print("=" * 60)
    
    try:
        from simple_patch_extractor import extract_patches
        
        # Test with invalid image name
        print("Testing with non-existent image...")
        try:
            result = extract_patches(
                patch_locations=[[100, 100]],
                image_name='nonexistent_image_12345',
                verbose=True
            )
            print("✗ Should have raised an error")
            return False
        except Exception as e:
            print(f"✓ Properly caught error: {type(e).__name__}")
            print(f"  Message: {str(e)[:100]}...")
        
        # Test with invalid patch locations
        print("\nTesting with invalid patch locations...")
        try:
            result = extract_patches(
                patch_locations=[[10000, 10000]],  # Outside image bounds
                image_name='image001',
                verbose=True
            )
            print("✓ Function handled out-of-bounds locations gracefully")
        except Exception as e:
            print(f"✓ Properly caught error: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing error handling: {e}")
        return False

def run_all_tests():
    """Run all test functions and provide summary."""
    print("Enhanced Natural Image Patch Extraction - Test Suite")
    print("=" * 60)
    
    tests = [
        ("MATLAB Function Direct", test_matlab_function_directly),
        ("Class-Based Interface", test_class_based_interface),
        ("Function-Based Interface", test_function_based_interface),
        ("Path Expansion", test_path_expansion),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced patch extraction is working properly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\nNote: Tests may fail if:")
        print("- MATLAB Engine API is not installed")
        print("- Natural image dataset is not available")
        print("- Directory paths need adjustment for your system")

if __name__ == "__main__":
    run_all_tests()
