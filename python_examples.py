"""
Complete Python Examples for Natural Image Patch Extraction

This file contains comprehensive examples showing how to use the Python interface
to extract natural image patches using the MATLAB backend.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import the patch extraction function
try:
    from simple_patch_extractor import extract_patches
except ImportError:
    print("Error: simple_patch_extractor.py not found in current directory")
    print("Make sure simple_patch_extractor.py is in the same directory as this script")
    sys.exit(1)


def example_1_basic_usage():
    """
    Example 1: Basic patch extraction with minimal parameters.
    """
    print("=" * 60)
    print("Example 1: Basic Patch Extraction")
    print("=" * 60)
    
    # Define patch locations around image center
    image_center = [768, 512]  # For 1536x1024 image
    patch_locations = [
        [image_center[0] - 100, image_center[1] - 100],
        [image_center[0], image_center[1]],
        [image_center[0] + 100, image_center[1] + 100]
    ]
    
    print(f"Extracting patches at locations: {patch_locations}")
    
    try:
        result = extract_patches(
            patch_locations=patch_locations,
            image_name='image001',  # Update this to your actual image name
            verbose=True
        )
        
        print(f"\\n✓ Success! Extracted {len(result['images'])} patches")
        print(f"  Valid patches: {result['metadata']['num_valid_patches']}")
        print(f"  Background intensity: {result['background_intensity']:.4f}")
        
        # Display patch information
        for i, info in enumerate(result['patch_info']):
            status = "✓ Valid" if info['valid'] else "✗ Invalid"
            clipped = " (Clipped)" if info['clipped'] else ""
            print(f"  Patch {i+1}: {status}{clipped}, Size: {info['actual_size']}")
        
        return result
        
    except Exception as e:
        print(f"\\n✗ Error: {e}")
        print("\\nTroubleshooting tips:")
        print("1. Update 'image001' to match your actual image file name")
        print("2. Ensure natural image files are accessible")
        print("3. Check MATLAB Engine API installation")
        return None


def example_2_custom_parameters():
    """
    Example 2: Using custom parameters for patch extraction.
    """
    print("\\n" + "=" * 60)
    print("Example 2: Custom Parameters")
    print("=" * 60)
    
    # Create a grid of patch locations
    grid_size = 3
    spacing = 200  # pixels
    center = [768, 512]
    
    patch_locations = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = center[0] + (i - grid_size//2) * spacing
            y = center[1] + (j - grid_size//2) * spacing
            patch_locations.append([x, y])
    
    print(f"Extracting {len(patch_locations)} patches in a {grid_size}x{grid_size} grid")
    
    try:
        result = extract_patches(
            patch_locations=patch_locations,
            image_name='image001',  # Update this
            patch_size=(150.0, 150.0),  # Smaller patches in microns
            normalize=True,
            verbose=True
        )
        
        print(f"\\n✓ Custom parameter extraction successful!")
        print(f"  Grid patches extracted: {result['metadata']['num_valid_patches']}/{len(patch_locations)}")
        
        # Analyze patch statistics
        valid_patches = [img for img in result['images'] if img is not None]
        if valid_patches:
            intensities = [np.mean(patch) for patch in valid_patches]
            print(f"  Mean patch intensities: {np.mean(intensities):.4f} ± {np.std(intensities):.4f}")
        
        return result
        
    except Exception as e:
        print(f"\\n✗ Custom parameter test failed: {e}")
        return None


def example_3_batch_processing():
    """
    Example 3: Batch processing multiple images.
    """
    print("\\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    # List of images to process (update these to your actual image names)
    image_names = ['image001', 'image002', 'image003']
    
    # Same patch locations for all images
    patch_locations = [
        [400, 300],
        [600, 400],
        [800, 500]
    ]
    
    print(f"Processing {len(image_names)} images with {len(patch_locations)} patches each")
    
    results = {}
    successful_extractions = 0
    
    for i, image_name in enumerate(image_names):
        print(f"\\nProcessing image {i+1}/{len(image_names)}: {image_name}")
        
        try:
            result = extract_patches(
                patch_locations=patch_locations,
                image_name=image_name,
                verbose=False  # Reduced verbosity for batch processing
            )
            
            results[image_name] = result
            successful_extractions += 1
            
            print(f"  ✓ Success: {result['metadata']['num_valid_patches']} valid patches")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[image_name] = None
    
    print(f"\\n📊 Batch Processing Summary:")
    print(f"  Successfully processed: {successful_extractions}/{len(image_names)} images")
    
    # Analyze results across all images
    all_patches = []
    for image_name, result in results.items():
        if result is not None:
            for patch in result['images']:
                if patch is not None:
                    all_patches.append(patch)
    
    print(f"  Total valid patches: {len(all_patches)}")
    
    if all_patches:
        # Compute statistics across all patches
        all_means = [np.mean(patch) for patch in all_patches]
        all_stds = [np.std(patch) for patch in all_patches]
        
        print(f"  Overall statistics:")
        print(f"    Mean intensity: {np.mean(all_means):.4f} ± {np.std(all_means):.4f}")
        print(f"    Mean std dev: {np.mean(all_stds):.4f} ± {np.std(all_stds):.4f}")
    
    return results


def example_4_visualization():
    """
    Example 4: Extracting patches and creating visualizations.
    """
    print("\\n" + "=" * 60)
    print("Example 4: Patch Visualization")
    print("=" * 60)
    
    # Define patch locations in a pattern
    patch_locations = [
        [400, 400],   # Center
        [300, 300],   # Top-left
        [500, 300],   # Top-right
        [300, 500],   # Bottom-left
        [500, 500]    # Bottom-right
    ]
    
    try:
        result = extract_patches(
            patch_locations=patch_locations,
            image_name='image001',  # Update this
            patch_size=(100.0, 100.0),
            verbose=True
        )
        
        if result['metadata']['num_valid_patches'] == 0:
            print("No valid patches extracted - cannot create visualization")
            return None
        
        # Create visualization
        try:
            import matplotlib.pyplot as plt
            
            # Calculate subplot layout
            n_patches = len([img for img in result['images'] if img is not None])
            cols = min(3, n_patches)
            rows = (n_patches + cols - 1) // cols
            
            fig, axes = plt.subplots(rows + 1, cols, figsize=(12, 8))
            if rows == 0:
                axes = axes.reshape(1, -1)
            
            # Show full image with patch locations
            if cols > 1:
                ax_full = plt.subplot(rows + 1, cols, (cols * rows) + 1)
            else:
                ax_full = axes[0] if rows == 0 else axes[0, 0]
                
            ax_full.imshow(result['full_image'], cmap='gray')
            ax_full.set_title('Full Image with Patch Locations')
            
            # Mark patch locations
            for i, info in enumerate(result['patch_info']):
                if info['valid']:
                    x, y = info['location']
                    ax_full.plot(y, x, 'r+', markersize=12, markeredgewidth=2)
                    ax_full.text(y + 10, x, f'{i+1}', color='red', fontweight='bold')
            
            # Show individual patches
            patch_idx = 0
            for i, (img, info) in enumerate(zip(result['images'], result['patch_info'])):
                if img is not None and info['valid']:
                    if rows > 0:
                        ax = axes[patch_idx // cols, patch_idx % cols]
                    else:
                        ax = axes[patch_idx]
                    
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f'Patch {i+1}\\nMean: {np.mean(img):.3f}')
                    ax.axis('off')
                    patch_idx += 1
            
            # Hide unused subplots
            if rows > 0:
                for idx in range(patch_idx, rows * cols):
                    axes[idx // cols, idx % cols].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\\n✓ Visualization created successfully!")
            print(f"  Displayed {n_patches} valid patches")
            
        except ImportError:
            print("\\nMatplotlib not available - skipping visualization")
            print("Install matplotlib with: pip install matplotlib")
        
        return result
        
    except Exception as e:
        print(f"\\n✗ Visualization example failed: {e}")
        return None


def example_5_integration_with_analysis():
    """
    Example 5: Integration with spike-triggered average analysis.
    """
    print("\\n" + "=" * 60)
    print("Example 5: Integration with STA Analysis")
    print("=" * 60)
    
    # Simulate STA analysis results (replace with your actual STA data)
    sta_peak_location = [600, 400]  # Peak of your STA
    receptive_field_radius = 75     # Radius in pixels
    
    print(f"STA peak at: {sta_peak_location}")
    print(f"RF radius: {receptive_field_radius} pixels")
    
    # Generate patch locations around the receptive field
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)  # 8 patches around the RF
    patch_locations = []
    
    # Center patch
    patch_locations.append(sta_peak_location)
    
    # Surrounding patches
    for angle in angles:
        x = sta_peak_location[0] + receptive_field_radius * np.cos(angle)
        y = sta_peak_location[1] + receptive_field_radius * np.sin(angle)
        patch_locations.append([int(x), int(y)])
    
    print(f"Generated {len(patch_locations)} patches: 1 center + {len(angles)} surrounding")
    
    try:
        result = extract_patches(
            patch_locations=patch_locations,
            image_name='image001',  # Update this
            patch_size=(80.0, 80.0),  # Patches smaller than RF radius
            verbose=True
        )
        
        # Analyze patches in context of receptive field
        center_patch = result['images'][0]  # First patch is center
        surrounding_patches = result['images'][1:]  # Rest are surrounding
        
        if center_patch is not None:
            center_mean = np.mean(center_patch)
            center_std = np.std(center_patch)
            
            print(f"\\n📊 Receptive Field Analysis:")
            print(f"  Center patch intensity: {center_mean:.4f} ± {center_std:.4f}")
            
            # Analyze surrounding patches
            valid_surrounding = [p for p in surrounding_patches if p is not None]
            if valid_surrounding:
                surround_means = [np.mean(p) for p in valid_surrounding]
                surround_mean = np.mean(surround_means)
                surround_std = np.std(surround_means)
                
                print(f"  Surround patches intensity: {surround_mean:.4f} ± {surround_std:.4f}")
                print(f"  Center-surround difference: {center_mean - surround_mean:.4f}")
                
                # Moment analysis example
                print(f"\\n🔬 Moment Analysis:")
                for i, patch in enumerate([center_patch] + valid_surrounding):
                    moment1 = np.mean(patch)
                    moment2 = np.mean(patch**2)
                    moment3 = np.mean(patch**3)
                    
                    patch_type = "Center" if i == 0 else f"Surround {i}"
                    print(f"  {patch_type}: M1={moment1:.4f}, M2={moment2:.4f}, M3={moment3:.4f}")
        
        return result
        
    except Exception as e:
        print(f"\\n✗ STA integration example failed: {e}")
        return None


def run_all_examples():
    """
    Run all examples in sequence.
    """
    print("🐍 Python Natural Image Patch Extraction Examples")
    print("=" * 80)
    
    # Check prerequisites
    print("Checking prerequisites...")
    
    # Check MATLAB Engine API
    try:
        import matlab.engine
        print("✓ MATLAB Engine API available")
    except ImportError:
        print("✗ MATLAB Engine API not installed")
        print("Please install it following the instructions in PYTHON_SETUP_GUIDE.md")
        return
    
    # Check numpy
    try:
        import numpy as np
        print("✓ NumPy available")
    except ImportError:
        print("✗ NumPy not available - please install: pip install numpy")
        return
    
    print(f"\\nCurrent working directory: {os.getcwd()}")
    print("\\n⚠️  IMPORTANT: Update image names in examples to match your data!")
    print("\\nRunning examples...\\n")
    
    # Run examples
    results = {}
    
    results['basic'] = example_1_basic_usage()
    results['custom'] = example_2_custom_parameters()
    results['batch'] = example_3_batch_processing()
    results['visualization'] = example_4_visualization()
    results['sta_integration'] = example_5_integration_with_analysis()
    
    # Summary
    print("\\n" + "=" * 80)
    print("📋 Examples Summary")
    print("=" * 80)
    
    successful_examples = sum(1 for result in results.values() if result is not None)
    total_examples = len(results)
    
    print(f"Successfully completed: {successful_examples}/{total_examples} examples")
    
    for name, result in results.items():
        status = "✓" if result is not None else "✗"
        print(f"  {status} {name.replace('_', ' ').title()}")
    
    if successful_examples > 0:
        print(f"\\n🎉 Great! The Python interface is working.")
        print(f"You can now integrate patch extraction into your analysis pipeline.")
    else:
        print(f"\\n🔧 No examples succeeded. Please check:")
        print(f"  1. MATLAB Engine API installation")
        print(f"  2. Image file names and locations")
        print(f"  3. MATLAB function file location")
    
    return results


if __name__ == "__main__":
    # You can run individual examples or all of them
    
    print("Choose an option:")
    print("1. Run all examples")
    print("2. Run individual example")
    print("3. Quick test")
    
    choice = input("Enter choice (1-3, or press Enter for quick test): ").strip()
    
    if choice == "1":
        run_all_examples()
    elif choice == "2":
        print("\\nAvailable examples:")
        print("1. Basic usage")
        print("2. Custom parameters")
        print("3. Batch processing")
        print("4. Visualization")
        print("5. STA integration")
        
        example_choice = input("Enter example number (1-5): ").strip()
        
        if example_choice == "1":
            example_1_basic_usage()
        elif example_choice == "2":
            example_2_custom_parameters()
        elif example_choice == "3":
            example_3_batch_processing()
        elif example_choice == "4":
            example_4_visualization()
        elif example_choice == "5":
            example_5_integration_with_analysis()
        else:
            print("Invalid choice")
    else:
        # Quick test (default)
        print("\\n🚀 Running quick test...")
        example_1_basic_usage()
