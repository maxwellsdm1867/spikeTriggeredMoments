%% Example Usage of Enhanced Natural Image Patch Extraction Function
% This script demonstrates how to use the improved getNaturalImagePatchFromLocation2 function
% for extracting natural image patches in visual neuroscience experiments.
%
% AUTHOR: [Your Name] and Colleagues
% DATE: July 2025

%% Clear workspace and setup
clear; close all; clc;

fprintf('=== Natural Image Patch Extraction Example ===\n\n');

%% Define patch locations (example coordinates)
% These would typically come from your experimental protocol or STA analysis
patchLocations = [
    100, 100;   % Patch 1: [x, y] coordinates in pixels
    200, 200;   % Patch 2
    400, 300;   % Patch 3
    600, 500;   % Patch 4
];

imageName = 'image001';  % Name of the natural image (without extension)

fprintf('Extracting %d patches from image: %s\n', size(patchLocations, 1), imageName);
fprintf('Patch locations:\n');
for i = 1:size(patchLocations, 1)
    fprintf('  Patch %d: [%d, %d]\n', i, patchLocations(i, 1), patchLocations(i, 2));
end
fprintf('\n');

%% Method 1: Basic usage with auto-detection (recommended for most users)
fprintf('--- Method 1: Auto-detection ---\n');
try
    result1 = getNaturalImagePatchFromLocation2_improved(patchLocations, imageName, ...
        'verbose', true);
    
    fprintf('✓ Successfully extracted patches using auto-detection\n');
    fprintf('  Valid patches: %d/%d\n', result1.metadata.numValidPatches, length(result1.images));
    fprintf('  Background intensity: %.4f\n\n', result1.backgroundIntensity);
    
catch ME
    fprintf('✗ Auto-detection failed: %s\n', ME.message);
    fprintf('  Will try manual directory specification...\n\n');
end

%% Method 2: Manual directory specification (for custom setups)
fprintf('--- Method 2: Manual Directory ---\n');

% Customize this path for your system
customResourcesDir = '/path/to/your/natural/images/';  % Update this!

try
    result2 = getNaturalImagePatchFromLocation2_improved(patchLocations, imageName, ...
        'resourcesDir', customResourcesDir, ...
        'patchSize', [150, 150], ...      % Custom patch size in microns
        'stimSet', '/custom_images/', ... % Custom subdirectory
        'verbose', true);
    
    fprintf('✓ Successfully extracted patches with custom directory\n');
    
catch ME
    fprintf('✗ Custom directory failed: %s\n', ME.message);
    fprintf('  Please update the customResourcesDir variable above\n\n');
end

%% Method 3: Synthetic data for testing (when natural images not available)
fprintf('--- Method 3: Synthetic Testing ---\n');

% Create synthetic image data for testing
fprintf('Creating synthetic image for testing...\n');
syntheticImage = rand(1536, 1024);  % Random image
syntheticImage = imgaussfilt(syntheticImage, 2);  % Smooth it

% This would be used in a modified version that accepts image data directly
% For now, just demonstrate the patch extraction logic
fprintf('Would extract patches at specified locations from synthetic data\n\n');

%% Analysis of extracted patches (if successful)
if exist('result1', 'var') && result1.metadata.numValidPatches > 0
    result = result1;
elseif exist('result2', 'var') && result2.metadata.numValidPatches > 0
    result = result2;
else
    fprintf('No patches were successfully extracted. Check your setup.\n');
    return;
end

fprintf('--- Patch Analysis ---\n');

% Display information about each patch
for i = 1:length(result.images)
    if result.patchInfo(i).valid
        patch = result.images{i};
        patchMean = mean(patch(:));
        patchStd = std(patch(:));
        
        fprintf('Patch %d:\n', i);
        fprintf('  Location: [%d, %d]\n', result.patchInfo(i).location);
        fprintf('  Size: [%d, %d] pixels\n', result.patchInfo(i).actualSize);
        fprintf('  Mean intensity: %.4f\n', patchMean);
        fprintf('  Std intensity: %.4f\n', patchStd);
        fprintf('  Clipped: %s\n', yesno(result.patchInfo(i).clipped));
        fprintf('\n');
    else
        fprintf('Patch %d: Invalid (outside image bounds)\n\n', i);
    end
end

%% Visualization (if patches were extracted)
if result.metadata.numValidPatches > 0
    fprintf('--- Visualization ---\n');
    
    % Create figure showing full image and extracted patches
    figure('Name', 'Natural Image Patch Extraction Results', 'Position', [100, 100, 1200, 800]);
    
    % Show full image with patch locations
    subplot(2, 3, 1);
    imagesc(result.fullImage);
    colormap(gray);
    axis image;
    title('Full Natural Image');
    hold on;
    
    % Mark patch locations
    validIdx = find([result.patchInfo.valid]);
    for i = validIdx
        loc = result.patchInfo(i).location;
        plot(loc(1), loc(2), 'r+', 'MarkerSize', 12, 'LineWidth', 2);
        text(loc(1)+20, loc(2), sprintf('%d', i), 'Color', 'red', 'FontSize', 12);
    end
    
    % Show individual patches
    numToShow = min(5, result.metadata.numValidPatches);
    for i = 1:numToShow
        if result.patchInfo(i).valid
            subplot(2, 3, i+1);
            imagesc(result.images{i});
            colormap(gray);
            axis image;
            title(sprintf('Patch %d', i));
        end
    end
    
    fprintf('✓ Visualization complete\n');
end

%% Summary and next steps
fprintf('\n=== Summary ===\n');
fprintf('Example completed successfully!\n');
fprintf('\nNext steps for your research:\n');
fprintf('1. Replace example patch locations with your experimental coordinates\n');
fprintf('2. Update image names to match your dataset\n');
fprintf('3. Adjust patch sizes based on your experimental needs\n');
fprintf('4. Integrate with your spike-triggered average analysis\n');
fprintf('5. Use extracted patches for moment decomposition analysis\n');

%% Helper function
function str = yesno(logicalValue)
    if logicalValue
        str = 'Yes';
    else
        str = 'No';
    end
end
