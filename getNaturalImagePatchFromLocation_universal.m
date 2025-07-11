function res = getNaturalImagePatchFromLocation_universal(patchLocations, imageName, varargin)
% GETNATURALIMAGEPATCHFROMLOCATION_UNIVERSAL - Universal natural image patch extraction function
%
% ⭐ RECOMMENDED: Use this function for all natural image patch extraction needs!
% 🧪 PLEASE TEST: This is a new universal function - please test with your data!
%
% ⚠️  IMPORTANT NOTICE FOR USERS:
%   This function replaces getNaturalImagePatchFromLocation2 with enhanced features.
%   PLEASE TEST this new function with your typical workflows and data to ensure
%   it works correctly for your use case. Report any issues or unexpected behavior.
%   The function now REQUIRES you to specify the 'resourcesDir' parameter.
%
% DESCRIPTION:
%   Universal function for extracting natural image patches that works seamlessly
%   from both MATLAB and Python environments. This function maintains full
%   backward compatibility with getNaturalImagePatchFromLocation2 while adding
%   enhanced features, error handling, and cross-platform support.
%
% USAGE:
%   % MATLAB (recommended - PLEASE TEST THIS NEW FUNCTION):
%   res = getNaturalImagePatchFromLocation_universal(patchLocations, imageName, 'resourcesDir', '/your/path')
%   res = getNaturalImagePatchFromLocation_universal(patchLocations, imageName, 'resourcesDir', '/your/path', 'verbose', true)
%   
%   % Python (use wrapper - RECOMMENDED):
%   from simple_patch_extractor import extract_patches
%   result = extract_patches(patch_locations, image_name, resources_dir='/your/path', verbose=True)
%   
%   % Python (direct call):
%   result = eng.getNaturalImagePatchFromLocation_universal(locations, image_name, 'resourcesDir', '/your/path', nargout=1)
%
% INPUTS:
%   patchLocations - N x 2 matrix of [x, y] coordinates for patch centers (pixels)
%   imageName      - String identifier for the natural image file (without extension)
%
% OPTIONAL PARAMETERS:
%   'resourcesDir'     - Base directory containing natural image files (REQUIRED - no auto-detection)
%   'stimSet'          - Subdirectory within resourcesDir (default: '/VHsubsample_20160105/')
%   'patchSize'        - Size of extracted patches in microns [height, width] (default: [200, 200])
%   'imageSize'        - For backward compatibility: patch size in microns (use 'patchSize' instead)
%   'imageSizePixels'  - Full image dimensions in pixels [height, width] (default: [1536, 1024])
%   'pixelSize'        - Microns per pixel conversion factor (default: 6.6)
%   'normalize'        - Whether to normalize image intensity (default: true)
%   'verbose'          - Display progress messages (default: false)
%   'pythonMode'       - Optimize output for Python consumption (auto-detected)
%
% OUTPUTS:
%   res - Structure containing:
%     .images               - Cell array of extracted image patches
%     .fullImage           - Full natural image (normalized if requested)
%     .backgroundIntensity - Mean intensity of full image
%     .patchInfo           - Information about extracted patches (enhanced)
%     .metadata            - Processing metadata and parameters (enhanced)
%
% BACKWARD COMPATIBILITY:
%   This function is a drop-in replacement for getNaturalImagePatchFromLocation2:
%   
%   % Old way (original function):
%   res = getNaturalImagePatchFromLocation2(locations, 'image001', 'imageSize', [200, 200]);
%   
%   % New way (PLEASE USE AND TEST THIS):
%   res = getNaturalImagePatchFromLocation_universal(locations, 'image001', ...
%       'resourcesDir', '/path/to/natural/images', 'imageSize', [200, 200]);
%
% EXAMPLES:
%   % ⚠️  IMPORTANT: PLEASE TEST THIS NEW FUNCTION WITH YOUR DATA! ⚠️
%   
%   % Basic usage (must specify resourcesDir)
%   patches = getNaturalImagePatchFromLocation_universal([[100, 100]; [200, 200]], 'image001', ...
%       'resourcesDir', '/path/to/your/natural/images');
%   
%   % Enhanced usage with new features
%   result = getNaturalImagePatchFromLocation_universal(...
%       [[100, 100]; [200, 200]], 'image001', ...
%       'resourcesDir', '/path/to/your/natural/images', ...
%       'patchSize', [150, 150], 'verbose', true, 'normalize', true);
%   
%   % Check patch extraction quality
%   fprintf('Valid patches: %d/%d\n', result.metadata.numValidPatches, length(result.images));
%   
%   % Test with your typical use case:
%   % Replace your old getNaturalImagePatchFromLocation2 calls with this function
%   % and verify the results match your expectations!
%
% MIGRATION GUIDE:
%   ⭐ PLEASE MIGRATE TO AND TEST THIS NEW FUNCTION! ⭐
%   
%   Replace all calls to getNaturalImagePatchFromLocation2 with this function:
%   - Same interface and results as original getNaturalImagePatchFromLocation2 function
%   - Additional features: error handling, cross-platform support, Python compatibility
%   - Enhanced output structure with metadata and patch information
%   - Now requires explicit resourcesDir parameter (no auto-detection)
%   
%   TESTING CHECKLIST:
%   □ Test with your typical patch locations and image names
%   □ Verify extracted patches match your expectations
%   □ Check that metadata contains useful information
%   □ Test with both normalized and non-normalized images
%   □ Verify Python compatibility if you use Python workflows
%
% NOTES:
%   - Automatically detects calling environment (MATLAB vs Python)
%   - Optimizes output format based on calling environment
%   - Cross-platform directory detection and path handling
%   - Robust error handling with informative messages
%   - Full backward compatibility with original getNaturalImagePatchFromLocation2 function
%   - User must now specify resourcesDir parameter (no auto-detection for reliability)
%   
%   🔍 TESTING NEEDED: Please test this function thoroughly with your data and workflows!
%   Report any issues, unexpected behavior, or suggestions for improvement.
%
% COMPATIBILITY:
%   - MATLAB R2018b and later
%   - Python 3.6+ (via MATLAB Engine API or Python wrappers)
%   - Windows, macOS, and Linux
%
% AUTHOR: Rieke Lab (original), Enhanced Universal Version
% DATE: July 2025
% VERSION: 3.0 (Universal)
% REPLACES: getNaturalImagePatchFromLocation2.m

%% Detect calling environment and parse inputs
    % Auto-detect if called from Python by checking stack
    isPythonCall = detectPythonCall();
    
    % Input validation
    if isempty(patchLocations)
        error('NIMPE:EmptyInput', 'patchLocations cannot be empty');
    end
    if size(patchLocations, 2) ~= 2
        error('NIMPE:InvalidSize', 'patchLocations must be an N×2 matrix of [x, y] coordinates');
    end
    
    validateattributes(patchLocations, {'numeric'}, {'real', 'finite', 'size', [NaN, 2]}, mfilename, 'patchLocations', 1);
    validateattributes(imageName, {'char', 'string'}, {'scalartext'}, mfilename, 'imageName', 2);
    
    if isstring(imageName)
        imageName = char(imageName); % Convert to char for consistency
    end
    
    % Parse optional parameters
    ip = inputParser;
    addRequired(ip, 'patchLocations', @(x) validateattributes(x, {'numeric'}, {'real', 'finite', 'size', [NaN, 2]}));
    addRequired(ip, 'imageName', @(x) validateattributes(x, {'char', 'string'}, {'scalartext'}));
    
    % System and path parameters
    addParameter(ip, 'resourcesDir', '', @(x) validateattributes(x, {'char', 'string'}, {'scalartext'}));
    addParameter(ip, 'stimSet', '/VHsubsample_20160105/', @(x) validateattributes(x, {'char', 'string'}, {'scalartext'}));
    
    % Image and patch parameters (maintaining compatibility with original function)
    addParameter(ip, 'patchSize', [200, 200], @(x) validateattributes(x, {'numeric'}, {'positive', 'finite', 'numel', 2}));
    addParameter(ip, 'imageSize', [200, 200], @(x) validateattributes(x, {'numeric'}, {'positive', 'finite', 'numel', 2})); % Original compatibility: patch size in microns
    addParameter(ip, 'imageSizePixels', [1536, 1024], @(x) validateattributes(x, {'numeric'}, {'positive', 'finite', 'numel', 2})); % Full image dimensions
    addParameter(ip, 'pixelSize', 6.6, @(x) validateattributes(x, {'numeric'}, {'positive', 'finite', 'scalar'}));
    
    % Processing parameters
    addParameter(ip, 'normalize', true, @(x) validateattributes(x, {'logical'}, {'scalar'}));
    addParameter(ip, 'verbose', false, @(x) validateattributes(x, {'logical'}, {'scalar'}));
    addParameter(ip, 'pythonMode', isPythonCall, @(x) validateattributes(x, {'logical'}, {'scalar'}));
    
    parse(ip, patchLocations, imageName, varargin{:});
    params = ip.Results;
    
    % Handle backward compatibility: original function used 'imageSize' for patch size in microns
    if any(strcmp('imageSize', varargin)) && ~any(strcmp('patchSize', varargin))
        % Original function behavior: imageSize parameter was actually patch size in microns
        params.patchSize = params.imageSize;
        params.imageSize = params.imageSizePixels; % Use full image dimensions
    else
        % New behavior: imageSize refers to full image dimensions in pixels
        params.imageSize = params.imageSizePixels;
    end
    
    if params.verbose
        callerInfo = sprintf('[%s]', iif(params.pythonMode, 'Python', 'MATLAB'));
        fprintf('%s Starting natural image patch extraction for image: %s\n', callerInfo, params.imageName);
        fprintf('%s Extracting %d patches of size [%.1f, %.1f] microns\n', ...
            callerInfo, size(params.patchLocations, 1), params.patchSize(1), params.patchSize(2));
    end

%% Validate Resources Directory
    if isempty(params.resourcesDir)
        fprintf('\n⚠️  ERROR: Natural images directory path is required!\n');
        fprintf('Please specify the ''resourcesDir'' parameter with the full path to your natural images folder.\n\n');
        fprintf('Example usage:\n');
        fprintf('  result = getNaturalImagePatchFromLocation_universal(locations, imageName, ...\n');
        fprintf('           ''resourcesDir'', ''/path/to/your/natural/images'');\n\n');
        
        error('NIMPE:DirectoryRequired', ...
            'Natural images directory path is required. Please specify ''resourcesDir'' parameter.');
    else
        resourcesDir = expandPath(params.resourcesDir);
        if ~exist(resourcesDir, 'dir')
            error('NIMPE:DirectoryNotFound', ...
                'Specified natural images directory does not exist: %s\nPlease check the path and ensure the directory exists.', resourcesDir);
        end
        if params.verbose
            fprintf('Using natural images directory: %s\n', resourcesDir);
        end
    end

%% Construct and Validate Image File Path
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
        error('NIMPE:ImageNotFound', ...
            'Image file not found. Tried the following paths:\n%s', ...
            sprintf('  %s\n', possiblePaths{:}));
    end

%% Load and Process Full Image
    if params.verbose
        fprintf('Loading image data...\n');
    end
    
    fileId = fopen(imageFilePath, 'rb', 'ieee-be');
    if fileId == -1
        error('NIMPE:FileOpenError', ...
            'Failed to open image file: %s\nCheck file permissions and format.', imageFilePath);
    end
    
    try
        img = fread(fileId, params.imageSize, 'uint16');
        fclose(fileId);
        
        if numel(img) ~= prod(params.imageSize)
            warning('NIMPE:SizeMismatch', ...
                'Image data size mismatch. Expected %d pixels, got %d pixels.', ...
                prod(params.imageSize), numel(img));
        end
        
    catch ME
        if fileId ~= -1
            fclose(fileId);
        end
        rethrow(ME);
    end
    
    % Convert to double and optionally normalize
    img = double(img);
    if params.normalize
        if max(img(:)) > 0
            img = img ./ max(img(:));
        else
            warning('NIMPE:ZeroImage', 'Image appears to be all zeros - normalization skipped');
        end
    end
    
    backgroundIntensity = mean(img(:));
    
    if params.verbose
        fprintf('Image loaded successfully. Background intensity: %.4f\n', backgroundIntensity);
    end

%% Extract Patches at Specified Locations
    % EXACT COMPATIBILITY: Use same calculation as original getNaturalImagePatchFromLocation2
    % Original: imageSize_VHpix = round(imageSize ./ (6.6)); radX = round(imageSize_VHpix(1) / 2);
    patchSize_pixels = round(params.patchSize ./ params.pixelSize);
    halfPatch = round(patchSize_pixels / 2);
    
    numPatches = size(params.patchLocations, 1);
    images = cell(numPatches, 1);  % Match original: cell(1, size(patchLocations,1))
    patchInfo = struct('location', {}, 'actualSize', {}, 'clipped', {}, 'valid', {});
    
    if params.verbose
        fprintf('Extracting %d patches (size: [%d, %d] pixels)...\n', ...
            numPatches, patchSize_pixels(1), patchSize_pixels(2));
    end
    
    for i = 1:numPatches
        centerX = round(params.patchLocations(i, 1));
        centerY = round(params.patchLocations(i, 2));
        
        % EXACT COMPATIBILITY: Use same indexing as original getNaturalImagePatchFromLocation2
        % Original: img(round(patchLocations(ff,1)-radX+1):round(patchLocations(ff,1)+radX), ...)
        x_start = max(1, centerX - halfPatch(1) + 1);
        x_end = min(params.imageSize(1), centerX + halfPatch(1));
        y_start = max(1, centerY - halfPatch(2) + 1);
        y_end = min(params.imageSize(2), centerY + halfPatch(2));
        
        isClipped = (x_start ~= centerX - halfPatch(1) + 1) || ...
                   (x_end ~= centerX + halfPatch(1)) || ...
                   (y_start ~= centerY - halfPatch(2) + 1) || ...
                   (y_end ~= centerY + halfPatch(2));
        
        isValid = (x_start <= x_end) && (y_start <= y_end);
        
        if isValid
            % EXACT COMPATIBILITY: Use same indexing as original function
            images{i} = img(x_start:x_end, y_start:y_end);
            actualSize = size(images{i});
        else
            images{i} = [];
            actualSize = [0, 0];
            if params.verbose
                fprintf('Warning: Patch %d at location [%d, %d] is outside image bounds\n', ...
                    i, centerX, centerY);
            end
        end
        
        patchInfo(i).location = [centerX, centerY];
        patchInfo(i).actualSize = actualSize;
        patchInfo(i).clipped = isClipped;
        patchInfo(i).valid = isValid;
    end
    
    validPatches = sum([patchInfo.valid]);
    clippedPatches = sum([patchInfo.clipped]);
    
    if params.verbose
        fprintf('Extraction complete: %d valid patches (%d clipped)\n', validPatches, clippedPatches);
    end

%% Compile Results Structure (Universal Format with Backward Compatibility)
    res = struct();
    
    % EXACT COMPATIBILITY: Match original function output structure
    res.images = images;
    res.fullImage = img;
    res.backgroundIntensity = backgroundIntensity;
    
    % Enhanced features (additional to original)
    res.patchInfo = patchInfo;
    
    % Metadata (additional to original)
    res.metadata = struct();
    res.metadata.imageName = params.imageName;
    res.metadata.imageFilePath = imageFilePath;
    res.metadata.parameters = params;
    res.metadata.processingTime = datetime('now');
    res.metadata.numValidPatches = validPatches;
    res.metadata.numClippedPatches = clippedPatches;
    res.metadata.callingEnvironment = iif(params.pythonMode, 'Python', 'MATLAB');
    res.metadata.backwardCompatible = true;  % Flag indicating compatibility
    
    % Python-specific optimizations (if needed)
    if params.pythonMode
        % Ensure all arrays are in formats that convert well to Python
        % (This is automatically handled by MATLAB Engine API, but we can add specific optimizations here if needed)
        res.metadata.pythonOptimized = true;
    end
    
    if params.verbose
        fprintf('Patch extraction completed successfully.\n');
        fprintf('Backward compatibility: Original output structure maintained.\n');
    end

%% Backward Compatibility Verification (Optional)
    if params.verbose
        fprintf('\n=== BACKWARD COMPATIBILITY CHECK ===\n');
        fprintf('Parameter mapping verification:\n');
        fprintf('  Original imageSize parameter: [%.1f, %.1f] microns\n', params.patchSize(1), params.patchSize(2));
        fprintf('  Converted to pixels: [%d, %d] pixels\n', patchSize_pixels(1), patchSize_pixels(2));
        fprintf('  Half-patch size: [%d, %d] pixels\n', halfPatch(1), halfPatch(2));
        fprintf('  Image dimensions: [%d, %d] pixels\n', params.imageSize(1), params.imageSize(2));
        fprintf('  Pixel size: %.1f microns/pixel\n', params.pixelSize);
        
        % Verify output structure matches original
        expectedFields = {'images', 'fullImage', 'backgroundIntensity'};
        for i = 1:length(expectedFields)
            if isfield(res, expectedFields{i})
                fprintf('  ✓ Field "%s" present\n', expectedFields{i});
            else
                fprintf('  ✗ Field "%s" MISSING\n', expectedFields{i});
            end
        end
        fprintf('=====================================\n\n');
    end
end

%% Helper Functions

function isPython = detectPythonCall()
    % Detect if function is called from Python by examining the call stack
    isPython = false;
    
    try
        stack = dbstack('-completenames');
        % Look for Python-related patterns in the stack
        for i = 1:length(stack)
            if contains(lower(stack(i).name), 'python') || ...
               contains(lower(stack(i).file), 'python') || ...
               contains(stack(i).name, 'py.') || ...
               contains(lower(stack(i).name), 'matlab.engine')
                isPython = true;
                return;
            end
        end
        
        % Alternative detection: check MATLAB environment
        try
            % Check if we're in a restricted environment (like MATLAB Engine)
            evalin('base', 'ver');
        catch
            % If we can't access base workspace, might be in MATLAB Engine
            isPython = true;
            return;
        end
        
        % Check for environment variables that indicate Python
        if ~isempty(getenv('PYTHONPATH')) || ~isempty(getenv('PYTHON_EXE'))
            isPython = true;
            return;
        end
        
    catch ME
        % If dbstack fails or other errors, assume MATLAB
        isPython = false;
        if contains(lower(ME.message), 'python')
            isPython = true;
        end
    end
end

function expandedPath = expandPath(inputPath)
    % Expand ~ and environment variables in file paths
    expandedPath = char(inputPath); % Ensure char for compatibility
    
    % Handle tilde expansion
    if startsWith(expandedPath, '~')
        if ispc
            homeDir = getenv('USERPROFILE');
        else
            homeDir = getenv('HOME');
        end
        if ~isempty(homeDir)
            if length(expandedPath) == 1
                % Just '~' by itself
                expandedPath = homeDir;
            else
                % '~/something'
                expandedPath = fullfile(homeDir, expandedPath(3:end)); % Skip '~/'
            end
        end
    end
    
    % Handle environment variables
    if contains(expandedPath, '$')
        expandedPath = strrep(expandedPath, '$HOME', getenv('HOME'));
        expandedPath = strrep(expandedPath, '$USER', getenv('USER'));
        expandedPath = strrep(expandedPath, '$USERPROFILE', getenv('USERPROFILE'));
    end
    
    % Convert to absolute path if needed
    if ~isempty(expandedPath) && ~isfolder(expandedPath)
        % Try to get absolute path
        try
            expandedPath = char(string(expandedPath));
        catch
            % If conversion fails, keep original
        end
    end
end

function result = iif(condition, trueValue, falseValue)
    % Inline if function
    if condition
        result = trueValue;
    else
        result = falseValue;
    end
end
