# Enhanced Natural Image Patch Extraction Function

This directory contains an improved version of the `getNaturalImagePatchFromLocation2` function for use across different laboratory computers and setups.

## Files

- `getNaturalImagePatchFromLocation2.m` - Original function (unchanged)
- `getNaturalImagePatchFromLocation2_improved.m` - Enhanced version with better error handling and documentation
- `example_usage.m` - Example script showing how to use the improved function

## Key Improvements in the Enhanced Version

### 1. **Cross-Platform Compatibility**
- Auto-detects common natural image directory structures
- Handles different naming conventions automatically
- Works on Windows, macOS, and Linux systems

### 2. **Robust Error Handling**
- Comprehensive file existence checking
- Informative error messages with suggested solutions
- Graceful handling of edge cases

### 3. **Enhanced Documentation**
- Detailed function documentation with examples
- Parameter validation with clear error messages
- Progress reporting with verbose mode

### 4. **Flexible Configuration**
- Multiple directory structure detection
- Customizable patch sizes and processing parameters
- Optional image normalization

### 5. **Better Output Structure**
- Comprehensive metadata about processing
- Patch-specific information (clipping, validity)
- Processing timestamps and parameters

## Quick Start

### Basic Usage
```matlab
% Extract patches at specified locations
patchLocations = [100, 100; 200, 200; 300, 300];  % [x, y] coordinates
imageName = 'image001';

% Use auto-detection (recommended)
result = getNaturalImagePatchFromLocation2_improved(patchLocations, imageName);

% Access extracted patches
patches = result.images;           % Cell array of image patches
fullImage = result.fullImage;      % Full natural image
background = result.backgroundIntensity;  % Mean intensity
```

### Advanced Usage
```matlab
% Specify custom directory and parameters
result = getNaturalImagePatchFromLocation2_improved(...
    patchLocations, imageName, ...
    'resourcesDir', '/path/to/your/natural/images/', ...
    'patchSize', [150, 150], ...          % Patch size in microns
    'stimSet', '/custom_image_set/', ...   % Subdirectory
    'verbose', true, ...                   % Show progress
    'normalize', true);                    % Normalize intensity

% Check patch validity
for i = 1:length(result.patchInfo)
    if result.patchInfo(i).valid
        fprintf('Patch %d: Valid, Size: [%d, %d]\n', i, result.patchInfo(i).actualSize);
    else
        fprintf('Patch %d: Invalid (outside image bounds)\n', i);
    end
end
```

## Directory Structure Setup

The function automatically searches for natural images in common locations:

### Option 1: Standard Lab Setup
```
/Users/[username]/Documents/NaturalImages/
├── VHsubsample_20160105/
│   ├── imkimage001.iml
│   ├── imkimage002.iml
│   └── ...
```

### Option 2: Shared Data Directory
```
/data/natural_images/
├── VHsubsample_20160105/
│   ├── imkimage001.iml
│   ├── imkimage002.iml
│   └── ...
```

### Option 3: Project-Specific
```
./natural_images/
├── imkimage001.iml
├── imkimage002.iml
└── ...
```

## Migration from Original Function

### Easy Migration
1. Copy `getNaturalImagePatchFromLocation2_improved.m` to your MATLAB path
2. Replace function calls:
   ```matlab
   % Old way
   patches = getNaturalImagePatchFromLocation2(locations, imageName);
   
   % New way (drop-in replacement)
   result = getNaturalImagePatchFromLocation2_improved(locations, imageName);
   patches = result.images;  % Same output format
   ```

### Recommended Updates
```matlab
% Take advantage of new features
result = getNaturalImagePatchFromLocation2_improved(...
    locations, imageName, ...
    'verbose', true, ...        % See what's happening
    'patchSize', [200, 200]);   % Explicit patch size

% Use enhanced output
patches = result.images;
metadata = result.metadata;
patchInfo = result.patchInfo;

% Check for issues
validPatches = [patchInfo.valid];
if any(~validPatches)
    fprintf('Warning: %d patches were invalid\n', sum(~validPatches));
end
```

## Troubleshooting

### Common Issues

1. **"Image file not found" Error**
   - Check that image files exist in the expected directory
   - Verify file naming convention (with/without 'imk' prefix)
   - Use `'verbose', true` to see which paths are being tried

2. **"Could not auto-detect directory" Error**
   - Manually specify the directory: `'resourcesDir', '/your/path/here'`
   - Make sure the directory contains the expected subdirectories

3. **Patches Outside Image Bounds**
   - Check your patch locations are within the image size
   - Use smaller patch sizes or adjust locations
   - The function will warn about clipped patches in verbose mode

### Getting Help
```matlab
% View detailed documentation
help getNaturalImagePatchFromLocation2_improved

% Run with verbose output to debug issues
result = getNaturalImagePatchFromLocation2_improved(...
    locations, imageName, 'verbose', true);
```

## System Requirements

- MATLAB R2018b or later
- Natural image files in binary (.iml) format
- Sufficient memory for loading full images (typically 1536×1024 pixels)

## For Colleagues Setting Up on New Computers

1. **Copy the function file** to your MATLAB path or project directory
2. **Set up image directory** in one of the standard locations (see above)
3. **Test with a simple example**:
   ```matlab
   testLocations = [500, 500];  % Single patch at image center
   result = getNaturalImagePatchFromLocation2_improved(...
       testLocations, 'image001', 'verbose', true);
   
   if result.metadata.numValidPatches > 0
       fprintf('Setup successful!\n');
   else
       fprintf('Check image files and directory structure\n');
   end
   ```

4. **Customize for your system** if needed by specifying `'resourcesDir'` parameter

This enhanced function should work across different laboratory setups with minimal configuration while providing better error messages and debugging information.
