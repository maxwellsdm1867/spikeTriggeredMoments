%% Universal Function Usage Examples - MATLAB Side
% This script demonstrates how to use the universal 
% getNaturalImagePatchFromLocation_universal function from MATLAB

%% Example 1: Basic Usage
clear; close all; clc;

fprintf('=== Universal Natural Image Patch Extraction - MATLAB Examples ===\n\n');

% Define patch locations
patchLocations = [
    100, 100;   % Patch 1: [x, y] coordinates in pixels
    200, 200;   % Patch 2
    400, 300;   % Patch 3
];

imageName = 'image001';  % Replace with your actual image name

fprintf('--- Example 1: Basic Usage ---\n');
try
    % Basic call - the universal function automatically detects it's being called from MATLAB
    result1 = getNaturalImagePatchFromLocation_universal(patchLocations, imageName, ...
        'verbose', true);
    
    fprintf('✓ Successfully extracted patches using universal function\n');
    fprintf('  Total patches: %d\n', length(result1.images));
    fprintf('  Valid patches: %d\n', result1.metadata.numValidPatches);
    fprintf('  Calling environment: %s\n', result1.metadata.callingEnvironment);
    fprintf('  Background intensity: %.4f\n\n', result1.backgroundIntensity);
    
catch ME
    fprintf('✗ Basic usage failed: %s\n\n', ME.message);
end

%% Example 2: Advanced Usage with Custom Parameters
fprintf('--- Example 2: Advanced Usage ---\n');
try
    result2 = getNaturalImagePatchFromLocation_universal(patchLocations, imageName, ...
        'patchSize', [150, 150], ...      % Custom patch size in microns
        'normalize', true, ...            % Normalize image intensity
        'verbose', true, ...              % Show progress
        'pixelSize', 6.6);               % Microns per pixel
    
    fprintf('✓ Advanced usage successful\n');
    fprintf('  Environment detected: %s\n', result2.metadata.callingEnvironment);
    fprintf('  Parameters used: patchSize=[%.1f, %.1f], normalize=%s\n', ...
        result2.metadata.parameters.patchSize(1), ...
        result2.metadata.parameters.patchSize(2), ...
        mat2str(result2.metadata.parameters.normalize));
    
catch ME
    fprintf('✗ Advanced usage failed: %s\n\n', ME.message);
end

%% Example 3: Show Patch Information
if exist('result1', 'var') && result1.metadata.numValidPatches > 0
    fprintf('--- Example 3: Patch Analysis ---\n');
    
    for i = 1:length(result1.images)
        info = result1.patchInfo(i);
        if info.valid
            patch = result1.images{i};
            fprintf('Patch %d:\n', i);
            fprintf('  Location: [%d, %d]\n', info.location(1), info.location(2));
            fprintf('  Size: [%d, %d] pixels\n', info.actualSize(1), info.actualSize(2));
            fprintf('  Mean intensity: %.4f\n', mean(patch(:)));
            fprintf('  Clipped: %s\n', mat2str(info.clipped));
        else
            fprintf('Patch %d: Invalid (outside bounds)\n', i);
        end
    end
    fprintf('\n');
end

%% Example 4: Comparison with Original Function (if available)
fprintf('--- Example 4: Compatibility Check ---\n');

% Check if original function exists
if exist('getNaturalImagePatchFromLocation2.m', 'file')
    try
        % Call original function
        originalResult = getNaturalImagePatchFromLocation2(patchLocations, imageName, ...
            'imageSize', [90, 120] * 6.6);  % Adjust for original function interface
        
        fprintf('✓ Original function available for comparison\n');
        
    catch ME
        fprintf('Note: Original function has different interface: %s\n', ME.message);
    end
else
    fprintf('Note: Original function not found - using universal function only\n');
end

%% Example 5: Demonstrate Python-Mode Detection
fprintf('--- Example 5: Force Python Mode ---\n');
try
    % Force Python mode to see the difference
    resultPythonMode = getNaturalImagePatchFromLocation_universal(patchLocations, imageName, ...
        'pythonMode', true, ...
        'verbose', true);
    
    fprintf('✓ Python mode forced successfully\n');
    fprintf('  Environment reported: %s\n', resultPythonMode.metadata.callingEnvironment);
    fprintf('  Python optimized: %s\n', mat2str(resultPythonMode.metadata.pythonOptimized));
    
catch ME
    fprintf('✗ Python mode test failed: %s\n', ME.message);
end

%% Summary
fprintf('\n=== Summary ===\n');
fprintf('The universal function getNaturalImagePatchFromLocation_universal:\n');
fprintf('• Automatically detects calling environment (MATLAB vs Python)\n');
fprintf('• Provides consistent interface for both environments\n');
fprintf('• Maintains full compatibility with enhanced features\n');
fprintf('• Eliminates need for separate function versions\n');
fprintf('• Can be forced into specific modes if needed\n\n');

fprintf('From Python, call the same function via MATLAB Engine:\n');
fprintf('  eng.getNaturalImagePatchFromLocation_universal(locations, name, nargout=1)\n\n');

fprintf('Usage completed! Check the Python examples for the other side.\n');
