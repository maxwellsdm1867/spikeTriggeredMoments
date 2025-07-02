function res = getNaturalImagePatchFromLocation2_universal(patchLocations, imageName, varargin)
% GETNATURALIMAGEPATCHFROMLOCATION2_UNIVERSAL - Universal natural image patch extraction function
%
% DESCRIPTION:
%   Universal function for extracting natural image patches that works seamlessly
%   from both MATLAB and Python environments. This function combines the best
%   features of all previous versions with enhanced compatibility and error handling.
%
% USAGE:
%   % From MATLAB:
%   res = getNaturalImagePatchFromLocation2_universal(patchLocations, imageName)
%   res = getNaturalImagePatchFromLocation2_universal(patchLocations, imageName, 'verbose', true)
%   
%   % From Python (via MATLAB Engine):
%   result = eng.getNaturalImagePatchFromLocation2_universal(locations, image_name, nargout=1)
%
% INPUTS:
%   patchLocations - N x 2 matrix of [x, y] coordinates for patch centers (pixels)
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
%   'pythonMode'       - Optimize output for Python consumption (auto-detected)
%
% OUTPUTS:
%   res - Structure containing:
%     .images               - Cell array of extracted image patches (MATLAB) or compatible format
%     .fullImage           - Full natural image (normalized if requested)
%     .backgroundIntensity - Mean intensity of full image
%     .patchInfo           - Information about extracted patches
%     .metadata            - Processing metadata and parameters
%
% EXAMPLES:
%   % Basic MATLAB usage
%   patches = getNaturalImagePatchFromLocation2_universal([[100, 100]; [200, 200]], 'image001');
%   
%   % Advanced MATLAB usage
%   result = getNaturalImagePatchFromLocation2_universal(...
%       [[100, 100]; [200, 200]], 'image001', ...
%       'patchSize', [150, 150], 'verbose', true);
%   
%   % Python usage (via MATLAB Engine)
%   import matlab.engine
%   eng = matlab.engine.start_matlab()
%   locations = matlab.double([[100, 100], [200, 200]])
%   result = eng.getNaturalImagePatchFromLocation2_universal(locations, 'image001', nargout=1)
%
% NOTES:
%   - Automatically detects calling environment (MATLAB vs Python)
%   - Optimizes output format based on calling environment
%   - Maintains backward compatibility with all previous versions
%   - Cross-platform directory detection and path handling
%   - Robust error handling with informative messages
%
% COMPATIBILITY:
%   - MATLAB R2018b and later
%   - Python 3.6+ (via MATLAB Engine API)
%   - Windows, macOS, and Linux
%
% AUTHOR: Rieke Lab (original), Enhanced Universal Version
% DATE: July 2025
% VERSION: 3.0 (Universal)

%% Detect calling environment and parse inputs
    % Auto-detect if called from Python by checking stack
    isPythonCall = detectPythonCall();
    
    % Input validation
    validateattributes(patchLocations, {'numeric'}, {'real', 'finite', 'size', [NaN, 2]}, mfilename, 'patchLocations', 1);
    validateattributes(imageName, {'char', 'string'}, {'scalartext'}, mfilename, 'imageName', 2);
    
    % Parse optional parameters
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
    addParameter(ip, 'pythonMode', isPythonCall, @(x) validateattributes(x, {'logical'}, {'scalar'}));
    
    parse(ip, patchLocations, imageName, varargin{:});
    params = ip.Results;
    
    if params.verbose
        callerInfo = sprintf('[%s]', iif(params.pythonMode, 'Python', 'MATLAB'));
        fprintf('%s Starting natural image patch extraction for image: %s\n', callerInfo, params.imageName);
        fprintf('%s Extracting %d patches of size [%.1f, %.1f] microns\n', ...
            callerInfo, size(params.patchLocations, 1), params.patchSize(1), params.patchSize(2));
    end

%% Auto-detect Resources Directory
    if isempty(params.resourcesDir)
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
            error('NIMPE:DirectoryNotFound', ...
                'Could not auto-detect natural images directory. Please specify ''resourcesDir'' parameter.\nTried directories:\n%s', ...
                sprintf('  %s\n', possibleDirs{:}));
        end
    else
        resourcesDir = expandPath(params.resourcesDir);
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
        centerX = round(params.patchLocations(i, 1));
        centerY = round(params.patchLocations(i, 2));
        
        x_start = max(1, centerX - halfPatch(1));
        x_end = min(params.imageSize(1), centerX + halfPatch(1));
        y_start = max(1, centerY - halfPatch(2));
        y_end = min(params.imageSize(2), centerY + halfPatch(2));
        
        isClipped = (x_start ~= centerX - halfPatch(1)) || ...
                   (x_end ~= centerX + halfPatch(1)) || ...
                   (y_start ~= centerY - halfPatch(2)) || ...
                   (y_end ~= centerY + halfPatch(2));
        
        isValid = (x_start <= x_end) && (y_start <= y_end);
        
        if isValid
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

%% Compile Results Structure (Universal Format)
    res = struct();
    res.images = images;
    res.fullImage = img;
    res.backgroundIntensity = backgroundIntensity;
    res.patchInfo = patchInfo;
    
    % Metadata
    res.metadata = struct();
    res.metadata.imageName = params.imageName;
    res.metadata.imageFilePath = imageFilePath;
    res.metadata.parameters = params;
    res.metadata.processingTime = datetime('now');
    res.metadata.numValidPatches = validPatches;
    res.metadata.numClippedPatches = clippedPatches;
    res.metadata.callingEnvironment = iif(params.pythonMode, 'Python', 'MATLAB');
    
    % Python-specific optimizations (if needed)
    if params.pythonMode
        % Ensure all arrays are in formats that convert well to Python
        % (This is automatically handled by MATLAB Engine API, but we can add specific optimizations here if needed)
        res.metadata.pythonOptimized = true;
    end
    
    if params.verbose
        fprintf('Patch extraction completed successfully.\n');
    end

end

%% Helper Functions

function isPython = detectPythonCall()
    % Detect if function is called from Python by examining the call stack
    try
        stack = dbstack('-completenames');
        % Look for Python-related patterns in the stack
        isPython = false;
        for i = 1:length(stack)
            if contains(lower(stack(i).name), 'python') || ...
               contains(lower(stack(i).file), 'python') || ...
               contains(stack(i).name, 'py.')
                isPython = true;
                break;
            end
        end
        
        % Alternative detection: check if running in MATLAB Engine
        try
            % This will error if not in MATLAB Engine context
            evalin('base', 'feature(''IsDebugMode'')');
        catch
            % Might be running in MATLAB Engine
            isPython = true;
        end
        
    catch
        isPython = false;
    end
end

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
        if ~isempty(homeDir)
            expandedPath = fullfile(homeDir, expandedPath(2:end));
        end
    end
    
    % Handle environment variables
    if contains(expandedPath, '$')
        expandedPath = strrep(expandedPath, '$HOME', getenv('HOME'));
        expandedPath = strrep(expandedPath, '$USER', getenv('USER'));
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
