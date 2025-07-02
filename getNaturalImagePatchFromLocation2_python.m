function result = getNaturalImagePatchFromLocation2_python(patchLocations, imageName, varargin)
% GETNATURALIMAGEPATCHFROMLOCATION2_PYTHON - Python-optimized natural image patch extraction
%
% DESCRIPTION:
%   This is a Python-optimized version of the enhanced natural image patch extraction
%   function. It's designed to work seamlessly with the Python MATLAB Engine API,
%   providing simplified input/output handling and better error reporting for
%   Python integration.
%
% USAGE (from Python):
%   import matlab.engine
%   eng = matlab.engine.start_matlab()
%   result = eng.getNaturalImagePatchFromLocation2_python(
%       matlab.double([[100, 100], [200, 200]]), 
%       'image001',
%       'verbose', True
%   )
%
% USAGE (from MATLAB):
%   result = getNaturalImagePatchFromLocation2_python([100, 100; 200, 200], 'image001');
%
% INPUTS:
%   patchLocations - N x 2 matrix of [x, y] coordinates for patch centers (in pixels)
%   imageName      - String identifier for the natural image file (without extension)
%
% OPTIONAL PARAMETERS (Name-Value pairs):
%   'resourcesDir'     - Base directory containing natural image files (auto-detects if empty)
%   'stimSet'          - Subdirectory within resourcesDir (default: '/VHsubsample_20160105/')
%   'patchSize'        - Size of extracted patches in microns [height, width] (default: [200, 200])
%   'imageSize'        - Full image dimensions in pixels [height, width] (default: [1536, 1024])
%   'pixelSize'        - Microns per pixel conversion factor (default: 6.6)
%   'normalize'        - Whether to normalize image intensity (default: true)
%   'verbose'          - Display progress messages (default: false)
%
% OUTPUTS:
%   result - Struct containing:
%     .images               - Cell array of extracted image patches (empty cells for invalid patches)
%     .fullImage           - Full natural image (normalized if requested)
%     .backgroundIntensity - Mean intensity of full image
%     .patchInfo           - Struct array with patch information
%     .metadata            - Processing metadata and parameters
%
% DIFFERENCES FROM STANDARD VERSION:
%   - Simplified error handling for Python compatibility
%   - Consistent output structure (no empty fields)
%   - Better handling of empty/invalid patches
%   - Python-friendly data types
%
% AUTHOR: Rieke Lab (original), Enhanced for Python by [Your Name]
% DATE: July 2025
% VERSION: 2.1-Python
% COMPATIBILITY: MATLAB R2018b and later, Python MATLAB Engine API

%% Input Validation and Parameter Parsing
try
    % Basic input validation
    if nargin < 2
        error('NaturalImagePatch:InvalidInput', 'At least 2 arguments required: patchLocations and imageName');
    end
    
    % Validate patch locations
    if ~isnumeric(patchLocations) || size(patchLocations, 2) ~= 2
        error('NaturalImagePatch:InvalidInput', 'patchLocations must be N x 2 numeric matrix');
    end
    
    % Validate image name
    if ~ischar(imageName) && ~isstring(imageName)
        error('NaturalImagePatch:InvalidInput', 'imageName must be a string');
    end
    imageName = char(imageName);  % Ensure char for compatibility
    
    % Parse optional parameters
    p = inputParser;
    addParameter(p, 'resourcesDir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'stimSet', '/VHsubsample_20160105/', @(x) ischar(x) || isstring(x));
    addParameter(p, 'patchSize', [200, 200], @(x) isnumeric(x) && length(x) == 2);
    addParameter(p, 'imageSize', [1536, 1024], @(x) isnumeric(x) && length(x) == 2);
    addParameter(p, 'pixelSize', 6.6, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'normalize', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'verbose', false, @(x) islogical(x) || isnumeric(x));
    
    parse(p, varargin{:});
    params = p.Results;
    
    % Convert strings to char for compatibility
    params.resourcesDir = char(params.resourcesDir);
    params.stimSet = char(params.stimSet);
    
    % Convert numeric to logical for boolean parameters
    if isnumeric(params.normalize)
        params.normalize = logical(params.normalize);
    end
    if isnumeric(params.verbose)
        params.verbose = logical(params.verbose);
    end
    
    if params.verbose
        fprintf('Python-MATLAB: Starting natural image patch extraction for image: %s\n', imageName);
        fprintf('Python-MATLAB: Extracting %d patches of size [%.1f, %.1f] microns\n', ...
            size(patchLocations, 1), params.patchSize(1), params.patchSize(2));
    end
    
catch ME
    error('NaturalImagePatch:ParameterError', 'Parameter parsing failed: %s', ME.message);
end

%% Auto-detect Resources Directory
try
    if isempty(params.resourcesDir)
        % Try common directory structures (expanded for different systems)
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
                    fprintf('Python-MATLAB: Auto-detected resources directory: %s\n', resourcesDir);
                end
                break;
            end
        end
        
        if isempty(resourcesDir)
            error('NaturalImagePatch:DirectoryNotFound', ...
                'Could not auto-detect natural images directory. Please specify resourcesDir parameter.\nTried directories:\n%s', ...
                sprintf('  %s\n', possibleDirs{:}));
        end
    else
        resourcesDir = expandPath(params.resourcesDir);
    end
    
catch ME
    error('NaturalImagePatch:DirectoryError', 'Directory detection failed: %s', ME.message);
end

%% Construct and Validate Image File Path
try
    % Try multiple possible file naming conventions
    possiblePaths = {
        fullfile(resourcesDir, params.stimSet, ['imk', imageName, '.iml']),
        fullfile(resourcesDir, params.stimSet, [imageName, '.iml']),
        fullfile(resourcesDir, ['imk', imageName, '.iml']),
        fullfile(resourcesDir, [imageName, '.iml'])
    };
    
    imageFilePath = '';
    for i = 1:length(possiblePaths)
        if exist(possiblePaths{i}, 'file')
            imageFilePath = possiblePaths{i};
            if params.verbose
                fprintf('Python-MATLAB: Found image file: %s\n', imageFilePath);
            end
            break;
        end
    end
    
    if isempty(imageFilePath)
        error('NaturalImagePatch:FileNotFound', ...
            'Image file not found. Tried the following paths:\n%s', ...
            sprintf('  %s\n', possiblePaths{:}));
    end
    
catch ME
    error('NaturalImagePatch:FileError', 'File path construction failed: %s', ME.message);
end

%% Load and Process Full Image
try
    if params.verbose
        fprintf('Python-MATLAB: Loading image data...\n');
    end
    
    % Open file with error handling
    fileId = fopen(imageFilePath, 'rb', 'ieee-be');
    if fileId == -1
        error('NaturalImagePatch:FileOpenError', ...
            'Failed to open image file: %s\nCheck file permissions and format.', imageFilePath);
    end
    
    try
        % Read image data
        img = fread(fileId, params.imageSize, 'uint16');
        fclose(fileId);
        
        % Validate image data
        if numel(img) ~= prod(params.imageSize)
            warning('NaturalImagePatch:SizeMismatch', ...
                'Image data size mismatch. Expected %d pixels, got %d pixels.', ...
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
            warning('NaturalImagePatch:ZeroImage', 'Image appears to be all zeros - normalization skipped');
        end
    end
    
    backgroundIntensity = mean(img(:));
    
    if params.verbose
        fprintf('Python-MATLAB: Image loaded successfully. Background intensity: %.4f\n', backgroundIntensity);
    end
    
catch ME
    error('NaturalImagePatch:LoadError', 'Image loading failed: %s', ME.message);
end

%% Extract Patches at Specified Locations
try
    % Convert patch size from microns to pixels
    patchSize_pixels = round(params.patchSize ./ params.pixelSize);
    halfPatch = round(patchSize_pixels / 2);
    
    numPatches = size(patchLocations, 1);
    images = cell(numPatches, 1);  % Pre-allocate cell array
    
    % Initialize patch info structure array
    patchInfo = struct();
    patchInfo.location = cell(numPatches, 1);
    patchInfo.actualSize = cell(numPatches, 1);
    patchInfo.clipped = false(numPatches, 1);
    patchInfo.valid = false(numPatches, 1);
    
    if params.verbose
        fprintf('Python-MATLAB: Extracting %d patches (size: [%d, %d] pixels)...\n', ...
            numPatches, patchSize_pixels(1), patchSize_pixels(2));
    end
    
    for i = 1:numPatches
        % Get patch center coordinates (rounded to nearest pixel)
        centerX = round(patchLocations(i, 1));
        centerY = round(patchLocations(i, 2));
        
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
            % Invalid patch location - store empty array
            images{i} = double.empty(0, 0);  % Python-friendly empty array
            actualSize = [0, 0];
            if params.verbose
                fprintf('Python-MATLAB: Warning: Patch %d at location [%d, %d] is outside image bounds\n', ...
                    i, centerX, centerY);
            end
        end
        
        % Store patch information
        patchInfo.location{i} = [centerX, centerY];
        patchInfo.actualSize{i} = actualSize;
        patchInfo.clipped(i) = isClipped;
        patchInfo.valid(i) = isValid;
    end
    
    if params.verbose
        validPatches = sum(patchInfo.valid);
        clippedPatches = sum(patchInfo.clipped);
        fprintf('Python-MATLAB: Extraction complete: %d valid patches (%d clipped)\n', ...
            validPatches, clippedPatches);
    end
    
catch ME
    error('NaturalImagePatch:ExtractionError', 'Patch extraction failed: %s', ME.message);
end

%% Compile Results Structure (Python-friendly format)
try
    result = struct();
    result.images = images;  % Cell array of patches
    result.fullImage = img;  % Full image as matrix
    result.backgroundIntensity = backgroundIntensity;  % Scalar
    result.patchInfo = patchInfo;  % Struct with arrays
    
    % Include metadata about processing
    result.metadata = struct();
    result.metadata.imageName = imageName;
    result.metadata.imageFilePath = imageFilePath;
    result.metadata.processingTime = char(datetime('now'));  % Convert to char for Python
    result.metadata.numValidPatches = sum(patchInfo.valid);
    result.metadata.numClippedPatches = sum(patchInfo.clipped);
    result.metadata.parameters = params;
    
    if params.verbose
        fprintf('Python-MATLAB: Patch extraction completed successfully.\n');
    end
    
catch ME
    error('NaturalImagePatch:ResultError', 'Result compilation failed: %s', ME.message);
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
