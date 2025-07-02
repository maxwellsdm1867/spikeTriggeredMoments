function res = getNaturalImagePatchFromLocation2_improved(patchLocations, imageName, varargin)
% GETNATURALIMAGEPATCHFROMLOCATION2_IMPROVED - Enhanced natural image patch extraction function
%
% DESCRIPTION:
%   This is an improved version of the original getNaturalImagePatchFromLocation2 function
%   with better error handling, documentation, and flexibility for use across different
%   computer systems and laboratory setups. This function extracts patches from natural 
%   images at specified locations for visual neuroscience experiments.
%
% USAGE:
%   res = getNaturalImagePatchFromLocation2_improved(patchLocations, imageName)
%   res = getNaturalImagePatchFromLocation2_improved(patchLocations, imageName, 'resourcesDir', '/path/to/images/')
%   res = getNaturalImagePatchFromLocation2_improved(..., 'patchSize', [200, 200])
%
% INPUTS:
%   patchLocations - N x 2 matrix of [x, y] coordinates for patch centers (in pixels)
%   imageName      - String identifier for the natural image file (without extension)
%
% OPTIONAL PARAMETERS:
%   'resourcesDir'     - Base directory containing natural image files (auto-detects if empty)
%   'stimSet'          - Subdirectory within resourcesDir (default: '/VHsubsample_20160105/')
%   'patchSize'        - Size of extracted patches in microns [height, width] (default: [200, 200])
%   'imageSize'        - Full image dimensions in pixels [height, width] (default: [1536, 1024])
%   'pixelSize'        - Microns per pixel conversion factor (default: 6.6)
%   'normalize'        - Whether to normalize image intensity (default: true)
%   'verbose'          - Display progress messages (default: false)
%
% OUTPUTS:
%   res - Structure containing:
%     .images               - Cell array of extracted image patches
%     .fullImage           - Full natural image (normalized if requested)
%     .backgroundIntensity - Mean intensity of full image
%     .patchInfo           - Information about extracted patches
%     .metadata            - Processing metadata and parameters
%
% EXAMPLES:
%   % Basic usage with auto-detection
%   patches = getNaturalImagePatchFromLocation2_improved([[100, 100]; [200, 200]], 'image001');
%   
%   % Specify custom directory and parameters
%   patches = getNaturalImagePatchFromLocation2_improved(...
%       [[100, 100]; [200, 200]], 'image001', ...
%       'resourcesDir', '/home/user/natural_images/', ...
%       'patchSize', [150, 150], ...
%       'verbose', true);
%
% NOTES:
%   - This function is designed for compatibility across different laboratory computers
%   - Automatically handles common directory structures and naming conventions
%   - Includes robust error handling and informative error messages
%   - Compatible with both van Hateren and custom natural image datasets
%   - Supports flexible patch extraction with boundary checking
%
% DEPENDENCIES:
%   - Natural image files in binary format (.iml files)
%   - MATLAB Image Processing Toolbox (for advanced features)
%
% AUTHOR: Rieke Lab (original), Enhanced by [Your Name] and Colleagues
% DATE: July 2025
% VERSION: 2.1 (Enhanced)
% COMPATIBILITY: MATLAB R2018b and later

%% Input Validation and Parameter Parsing
    validateattributes(patchLocations, {'numeric'}, {'real', 'finite', 'size', [NaN, 2]}, mfilename, 'patchLocations', 1);
    validateattributes(imageName, {'char', 'string'}, {'scalartext'}, mfilename, 'imageName', 2);
    
    % Parse optional parameters with detailed validation
    ip = inputParser;
    addRequired(ip, 'patchLocations', @(x) validateattributes(x, {'numeric'}, {'real', 'finite', 'size', [NaN, 2]}));
    addRequired(ip, 'imageName', @(x) validateattributes(x, {'char', 'string'}, {'scalartext'}));
    
    % System and path parameters
    addParameter(ip, 'resourcesDir', '', @(x) validateattributes(x, {'char', 'string'}, {'scalartext'}));
    addParameter(ip, 'stimSet', '/VHsubsample_20160105/', @(x) validateattributes(x, {'char', 'string'}, {'scalartext'}));
    
    % Image and patch parameters
    addParameter(ip, 'patchSize', [200, 200], @(x) validateattributes(x, {'numeric'}, {'positive', 'finite', 'numel', 2}));
    addParameter(ip, 'imageSize', [1536, 1024], @(x) validateattributes(x, {'numeric'}, {'positive', 'finite', 'numel', 2}));
    addParameter(ip, 'pixelSize', 6.6, @(x) validateattributes(x, {'numeric'}, {'positive', 'finite', 'scalar'}));
    
    % Processing parameters
    addParameter(ip, 'normalize', true, @(x) validateattributes(x, {'logical'}, {'scalar'}));
    addParameter(ip, 'verbose', false, @(x) validateattributes(x, {'logical'}, {'scalar'}));
    
    parse(ip, patchLocations, imageName, varargin{:});
    
    % Extract parsed parameters
    params = ip.Results;
    
    if params.verbose
        fprintf('Starting natural image patch extraction for image: %s\n', params.imageName);
        fprintf('Extracting %d patches of size [%.1f, %.1f] microns\n', ...
            size(params.patchLocations, 1), params.patchSize(1), params.patchSize(2));
    end

%% Auto-detect Resources Directory
    if isempty(params.resourcesDir)
        % Try common directory structures
        possibleDirs = {
            '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/subunitModel_NaturalImages',
            '~/Documents/NaturalImages',
            '~/Data/NaturalImages', 
            '/data/natural_images',
            '/home/shared/natural_images',
            './natural_images',
            '../data/natural_images'
        };
        
        resourcesDir = '';
        for i = 1:length(possibleDirs)
            expandedPath = expandPath(possibleDirs{i});
            if exist(expandedPath, 'dir')
                resourcesDir = expandedPath;
                if params.verbose
                    fprintf('Auto-detected resources directory: %s\n', resourcesDir);
                end
                break;
            end
        end
        
        if isempty(resourcesDir)
            error('Could not auto-detect natural images directory. Please specify ''resourcesDir'' parameter.\nTried directories:\n%s', ...
                sprintf('  %s\n', possibleDirs{:}));
        end
    else
        resourcesDir = expandPath(params.resourcesDir);
    end

%% Construct and Validate Image File Path
    % Try multiple possible file naming conventions
    possiblePaths = {
        fullfile(resourcesDir, params.stimSet, ['imk', params.imageName, '.iml']),
        fullfile(resourcesDir, params.stimSet, [params.imageName, '.iml']),
        fullfile(resourcesDir, ['imk', params.imageName, '.iml']),
        fullfile(resourcesDir, [params.imageName, '.iml'])
    };
    
    imageFilePath = '';
    for i = 1:length(possiblePaths)
        if exist(possiblePaths{i}, 'file')
            imageFilePath = possiblePaths{i};
            if params.verbose
                fprintf('Found image file: %s\n', imageFilePath);
            end
            break;
        end
    end
    
    if isempty(imageFilePath)
        error('Image file not found. Tried the following paths:\n%s', ...
            sprintf('  %s\n', possiblePaths{:}));
    end

%% Load and Process Full Image
    if params.verbose
        fprintf('Loading image data...\n');
    end
    
    % Open file with error handling
    fileId = fopen(imageFilePath, 'rb', 'ieee-be');
    if fileId == -1
        error('Failed to open image file: %s\nCheck file permissions and format.', imageFilePath);
    end
    
    try
        % Read image data
        img = fread(fileId, params.imageSize, 'uint16');
        fclose(fileId);
        
        % Validate image data
        if numel(img) ~= prod(params.imageSize)
            warning('Image data size mismatch. Expected %d pixels, got %d pixels.', ...
                prod(params.imageSize), numel(img));
        end
        
    catch ME
        % Ensure file is closed even if error occurs
        if fileId ~= -1
            fclose(fileId);
        end
        rethrow(ME);
    end
    
    % Convert to double precision and optionally normalize
    img = double(img);
    if params.normalize
        if max(img(:)) > 0
            img = img ./ max(img(:));  % Normalize so brightest point is 1.0
        else
            warning('Image appears to be all zeros - normalization skipped');
        end
    end
    
    backgroundIntensity = mean(img(:));
    
    if params.verbose
        fprintf('Image loaded successfully. Background intensity: %.4f\n', backgroundIntensity);
    end

%% Extract Patches at Specified Locations
    % Convert patch size from microns to pixels
    patchSize_pixels = round(params.patchSize ./ params.pixelSize);
    halfPatch = round(patchSize_pixels / 2);
    
    numPatches = size(params.patchLocations, 1);
    images = cell(numPatches, 1);
    patchInfo = struct('location', {}, 'actualSize', {}, 'clipped', {}, 'valid', {});
    
    if params.verbose
        fprintf('Extracting %d patches (size: [%d, %d] pixels)...\n', ...
            numPatches, patchSize_pixels(1), patchSize_pixels(2));
    end
    
    for i = 1:numPatches
        % Get patch center coordinates (rounded to nearest pixel)
        centerX = round(params.patchLocations(i, 1));
        centerY = round(params.patchLocations(i, 2));
        
        % Calculate patch boundaries
        x_start = max(1, centerX - halfPatch(1));
        x_end = min(params.imageSize(1), centerX + halfPatch(1));
        y_start = max(1, centerY - halfPatch(2));
        y_end = min(params.imageSize(2), centerY + halfPatch(2));
        
        % Check if patch is within image bounds
        isClipped = (x_start ~= centerX - halfPatch(1)) || ...
                   (x_end ~= centerX + halfPatch(1)) || ...
                   (y_start ~= centerY - halfPatch(2)) || ...
                   (y_end ~= centerY + halfPatch(2));
        
        isValid = (x_start <= x_end) && (y_start <= y_end);
        
        if isValid
            % Extract patch
            images{i} = img(x_start:x_end, y_start:y_end);
            actualSize = size(images{i});
        else
            % Invalid patch location
            images{i} = [];
            actualSize = [0, 0];
            if params.verbose
                fprintf('Warning: Patch %d at location [%d, %d] is outside image bounds\n', ...
                    i, centerX, centerY);
            end
        end
        
        % Store patch information
        patchInfo(i).location = [centerX, centerY];
        patchInfo(i).actualSize = actualSize;
        patchInfo(i).clipped = isClipped;
        patchInfo(i).valid = isValid;
    end
    
    if params.verbose
        validPatches = sum([patchInfo.valid]);
        clippedPatches = sum([patchInfo.clipped]);
        fprintf('Extraction complete: %d valid patches (%d clipped)\n', validPatches, clippedPatches);
    end

%% Compile Results Structure
    res = struct();
    res.images = images;
    res.fullImage = img;
    res.backgroundIntensity = backgroundIntensity;
    res.patchInfo = patchInfo;
    
    % Include metadata about processing
    res.metadata = struct();
    res.metadata.imageName = params.imageName;
    res.metadata.imageFilePath = imageFilePath;
    res.metadata.parameters = params;
    res.metadata.processingTime = datetime('now');
    res.metadata.numValidPatches = sum([patchInfo.valid]);
    res.metadata.numClippedPatches = sum([patchInfo.clipped]);
    
    if params.verbose
        fprintf('Patch extraction completed successfully.\n');
    end

end

%% Helper Function: Expand Path with Tilde and Environment Variables
function expandedPath = expandPath(inputPath)
    % Expand ~ and environment variables in file paths
    expandedPath = inputPath;
    
    % Handle tilde expansion
    if startsWith(expandedPath, '~')
        if ispc
            homeDir = getenv('USERPROFILE');
        else
            homeDir = getenv('HOME');
        end
        expandedPath = fullfile(homeDir, expandedPath(2:end));
    end
    
    % Handle environment variables (basic implementation)
    if contains(expandedPath, '$')
        % This is a simplified version - could be enhanced for more complex cases
        expandedPath = strrep(expandedPath, '$HOME', getenv('HOME'));
        expandedPath = strrep(expandedPath, '$USER', getenv('USER'));
    end
end
