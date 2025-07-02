function results = spike_triggered_moments_master(varargin)
% SPIKE_TRIGGERED_MOMENTS_MASTER - Comprehensive analysis of neural responses using moment decomposition
%
% DESCRIPTION:
%   This master function performs spike-triggered moment analysis on neural data from 
%   the FlashedGratePlusNoise protocol. It decomposes neural responses into contributions 
%   from different statistical moments (0th-3rd order) of stimulus distributions using
%   histogram-based analysis with cross-validation.
%
% USAGE:
%   results = spike_triggered_moments_master()                    % Interactive mode
%   results = spike_triggered_moments_master('synthetic', true)   % Synthetic data test
%   results = spike_triggered_moments_master('dataFolder', path)  % Specify data path
%
% OPTIONAL PARAMETERS:
%   'dataFolder'     - Path to experimental data (string)
%   'exportFolder'   - Path to export files (string) 
%   'synthetic'      - Run synthetic pink noise test (logical, default: false)
%   'nbins'          - Number of histogram bins (integer, default: 12)
%   'rfThreshold'    - RF extraction threshold (0-1, default: 0.4)
%   'lambda'         - Ridge regression regularization (default: 0)
%   'testSize'       - Cross-validation test size (0-1, default: 0.3)
%   'verbose'        - Display progress messages (logical, default: true)
%   'plotResults'    - Generate visualization plots (logical, default: true)
%
% OUTPUTS:
%   results - Structure containing:
%     .STA_original         - Spike-triggered average
%     .rf_mask              - Receptive field mask  
%     .SpikeCounts          - Original spike counts per trial
%     .moment_contributions - Estimated moment contributions [0th, 1st, 2nd, 3rd]
%     .normalized_contributions - Relative contributions as percentages
%     .r2_holdout          - Cross-validation R² score
%     .weights             - Ridge regression weights for each moment
%     .analysis_summary    - Summary statistics and parameters
%
% ALGORITHM WORKFLOW:
%   1. Load experimental data or generate synthetic stimuli
%   2. Extract spike times and compute spike-triggered average (STA)
%   3. Extract receptive field mask from STA using adaptive thresholding
%   4. Generate stimulus histograms from RF-masked pixels  
%   5. Compute moment values (0th-3rd order) for each trial
%   6. Fit ridge regression models for each moment
%   7. Decompose real neural weights into moment contributions
%   8. Perform cross-validation to assess prediction accuracy
%   9. Generate comprehensive visualizations and summary
%
% DEPENDENCIES:
%   - manualRidgeRegressionCustom.m
%   - SpikeDetection.Detector (Rieke Lab)
%   - getNaturalImagePatchFromLocation2 (Rieke Lab)
%   - edu.washington.rieke.Analysis framework
%
% AUTHOR: [Your Name]
% DATE: July 2025
% VERSION: 1.0

%% Parse Input Arguments
p = inputParser;
addParameter(p, 'dataFolder', '', @ischar);
addParameter(p, 'exportFolder', '', @ischar);
addParameter(p, 'synthetic', false, @islogical);
addParameter(p, 'nbins', 12, @(x) isnumeric(x) && x > 0);
addParameter(p, 'rfThreshold', 0.4, @(x) isnumeric(x) && x > 0 && x < 1);
addParameter(p, 'lambda', 0, @(x) isnumeric(x) && x >= 0);
addParameter(p, 'testSize', 0.3, @(x) isnumeric(x) && x > 0 && x < 1);
addParameter(p, 'verbose', true, @islogical);
addParameter(p, 'plotResults', true, @islogical);
parse(p, varargin{:});

% Extract parameters
params = p.Results;
verbose = params.verbose;

if verbose
    fprintf('\n=== SPIKE TRIGGERED MOMENTS ANALYSIS ===\n');
    fprintf('Starting comprehensive moment decomposition analysis...\n');
end

%% Initialize Environment
close all;
if verbose, fprintf('Initializing environment...\n'); end

% Set plotting defaults
set(0, 'DefaultAxesFontName', 'Helvetica');
set(0, 'DefaultAxesFontSize', 14);

% Analysis parameters
analysis_params.FrequencyCutoff = 500;
analysis_params.Amp = 'Amp1';
analysis_params.SamplingInterval = 0.0001;
analysis_params.FrameRate = 60.1830;

%% Main Analysis Branch
if params.synthetic
    if verbose, fprintf('\n--- SYNTHETIC DATA ANALYSIS ---\n'); end
    results = analyze_synthetic_data(params, verbose);
else
    if verbose, fprintf('\n--- EXPERIMENTAL DATA ANALYSIS ---\n'); end
    results = analyze_experimental_data(params, analysis_params, verbose);
end

%% Generate Final Summary
if verbose
    fprintf('\n=== ANALYSIS COMPLETE ===\n');
    print_analysis_summary(results);
end

if params.plotResults
    generate_summary_plots(results);
end

end

%% EXPERIMENTAL DATA ANALYSIS FUNCTION
function results = analyze_experimental_data(params, analysis_params, verbose)

% Initialize Rieke Lab framework
if verbose, fprintf('Loading Rieke Lab analysis framework...\n'); end
try
    loader = edu.washington.rieke.Analysis.getEntityLoader();
    import auimodel.*
    import vuidocument.*
catch ME
    error('Failed to load Rieke Lab framework: %s', ME.message);
end

% Set data paths
if isempty(params.dataFolder)
    dataFolder = uigetdir('', 'Select Data Folder');
    if dataFolder == 0
        error('No data folder selected');
    end
else
    dataFolder = params.dataFolder;
end

if isempty(params.exportFolder)
    exportFolder = dataFolder;  % Default to same as data folder
else
    exportFolder = params.exportFolder;
end

% Load experimental data
if verbose, fprintf('Loading experimental data from: %s\n', dataFolder); end
try
    list = loader.loadEpochList([exportFolder 'FlashedGratePlusNoise.mat'], dataFolder);
    
    % Build analysis tree
    dateSplit = @(list)splitOnExperimentDate(list);
    dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(list, dateSplit);
    tree = riekesuite.analysis.buildTree(list, {
        'protocolSettings(source:type)', dateSplit_java, 'cell.label', ...
        'protocolSettings(linearizeCones)', 'protocolSettings(noiseContrast)', ...
        'protocolSettings(imagePatchIndex)'});
    
    % Interactive node selection
    gui = epochTreeGUI(tree);
    if verbose
        fprintf('Please select a node in the Epoch Tree GUI, then press any key to continue...\n');
        pause;
    end
    
    node = gui.getSelectedEpochTreeNodes;
    if isempty(node)
        error('No node selected in Epoch Tree GUI');
    end
    cdt_node = node{1};
    
catch ME
    error('Failed to load experimental data: %s', ME.message);
end

% Extract experimental parameters
if verbose, fprintf('Extracting experimental parameters...\n'); end
SampleEpoch = cdt_node.epochList.elements(1);
SamplingInterval = 1000/SampleEpoch.protocolSettings.get('sampleRate');  % Convert to msec
PrePts = SampleEpoch.protocolSettings.get('preTime') / SamplingInterval;
StmPts = SampleEpoch.protocolSettings.get('stimTime') / SamplingInterval;
TailPts = SampleEpoch.protocolSettings.get('tailTime') / SamplingInterval;
numNoiseRepeats = SampleEpoch.protocolSettings.get('numNoiseRepeats');
noiseFilterSD = SampleEpoch.protocolSettings.get('noiseFilterSD');
imageName = SampleEpoch.protocolSettings.get('imageName');

% Determine image size based on rig
rig_info = SampleEpoch.protocolSettings.get('experiment:rig');
if rig_info(1) == 'B'
    imageSize = [90 120];
elseif rig_info(1) == 'G'
    imageSize = [166 268];
else
    warning('Unknown rig type: %s. Using default image size [90 120]', rig_info);
    imageSize = [90 120];
end

if verbose
    fprintf('  Image size: [%d %d]\n', imageSize(1), imageSize(2));
    fprintf('  Noise repeats: %d\n', numNoiseRepeats);
    fprintf('  Noise filter SD: %.2f\n', noiseFilterSD);
end

% Process neural data
if verbose, fprintf('Processing neural data and computing STA...\n'); end
EpochData = getSelectedData(cdt_node.epochList, analysis_params.Amp);
[SpikeTimes, ~, ~] = SpikeDetection.Detector(EpochData);

% Initialize storage
numTrials = size(EpochData, 1) * numNoiseRepeats;
SpikeCounts = zeros(1, numTrials);
Stimuli = zeros(numTrials, imageSize(1) * imageSize(2));
STA = zeros(imageSize);
trialIndex = 1;
totSpikes = 0;

% Main processing loop
if verbose, fprintf('Processing %d trials...\n', numTrials); end
for epoch = 1:size(EpochData, 1)
    noiseSeed = cdt_node.epochList.elements(epoch).protocolSettings.get('noiseSeed');
    noiseStream = RandStream('mt19937ar', 'Seed', noiseSeed);
    
    for repeat = 1:numNoiseRepeats
        % Calculate trial timing
        startTrial = PrePts * repeat + (StmPts + TailPts) * (repeat-1);
        numSpikes = length(find(SpikeTimes{epoch} > startTrial & ...
                                SpikeTimes{epoch} < (startTrial + StmPts)));
        
        % Generate noise stimulus
        noiseMatrix = imgaussfilt(noiseStream.randn(imageSize(1), imageSize(2)), noiseFilterSD);
        noiseMatrix = noiseMatrix / std(noiseMatrix(:));
        
        % Store trial data
        SpikeCounts(trialIndex) = numSpikes;
        Stimuli(trialIndex, :) = noiseMatrix(:)';
        totSpikes = totSpikes + numSpikes;
        trialIndex = trialIndex + 1;
    end
    
    if verbose && mod(epoch, 10) == 0
        fprintf('  Processed epoch %d/%d\n', epoch, size(EpochData, 1));
    end
end

% Compute spike-triggered average
meanSpikes = totSpikes / numTrials;
SpikeCounts_corrected = SpikeCounts - meanSpikes;
STA = (SpikeCounts_corrected * Stimuli) / totSpikes;
STA = reshape(STA, imageSize);

if verbose, fprintf('STA computed. Total spikes: %d, Mean spikes/trial: %.2f\n', totSpikes, meanSpikes); end

% Extract receptive field and continue with moment analysis
[rf_mask, rf_stats] = extract_receptive_field(STA, params.rfThreshold, verbose);
[moment_results] = analyze_moments(Stimuli, SpikeCounts_corrected, rf_mask, imageSize, ...
                                  params, verbose);

% Compile results
results = compile_results(STA, SpikeCounts, rf_mask, rf_stats, moment_results, ...
                         imageSize, analysis_params, params);

end

%% SYNTHETIC DATA ANALYSIS FUNCTION  
function results = analyze_synthetic_data(params, verbose)

if verbose, fprintf('Generating synthetic 2D pink noise stimuli...\n'); end

% Generate 2D pink noise
N = 512;  % Base image size
[x, y] = meshgrid(1:N, 1:N);
cx = ceil(N/2); cy = ceil(N/2);
f = sqrt((x-cx).^2 + (y-cy).^2);
f(cx,cy) = 1;  % Avoid division by zero at DC

whiteNoise = randn(N);
spectrum = fftshift(fft2(whiteNoise)) ./ f;
pinkImg = real(ifft2(ifftshift(spectrum)));
pinkImg = (pinkImg - min(pinkImg(:))) / (max(pinkImg(:)) - min(pinkImg(:)));

% Sample random patches
patchSize = [16, 16];
numTrials = 200;  % Increased for better statistics
patches = zeros(numTrials, patchSize(1) * patchSize(2));

if verbose, fprintf('Sampling %d random patches of size [%d %d]...\n', ...
    numTrials, patchSize(1), patchSize(2)); end

for i = 1:numTrials
    x_start = randi(N - patchSize(1) + 1);
    y_start = randi(N - patchSize(2) + 1);
    patch = pinkImg(x_start:x_start+patchSize(1)-1, y_start:y_start+patchSize(2)-1);
    patches(i, :) = patch(:)';
end

% Generate synthetic responses with known moment dependencies
patchMeans = mean(patches, 2);
patchSecondMoment = mean(patches.^2, 2);
patchThirdMoment = mean(patches.^3, 2);

% Define known combination weights
true_weights = [0.5, 2.0, 3.5, 1.2];  % [offset, mean, second, third]
spikeCounts = true_weights(1) + ...
              true_weights(2) * patchMeans + ...
              true_weights(3) * patchSecondMoment + ...
              true_weights(4) * patchThirdMoment;

% Add realistic noise
spikeCounts = max(0, spikeCounts + 0.3 * std(spikeCounts) * randn(size(spikeCounts)));

if verbose
    fprintf('Generated synthetic responses with known weights:\n');
    fprintf('  Offset: %.2f, Mean: %.2f, Second: %.2f, Third: %.2f\n', true_weights);
end

% Create fake RF mask (use all pixels for synthetic data)
imageSize = patchSize;
rf_mask = true(imageSize);
rf_stats.center = [imageSize(1)/2, imageSize(2)/2];
rf_stats.radius = min(imageSize)/2;
rf_stats.pixel_count = sum(rf_mask(:));

% Analyze moments
SpikeCounts_corrected = spikeCounts - mean(spikeCounts);
[moment_results] = analyze_moments(patches, SpikeCounts_corrected', rf_mask, imageSize, ...
                                  params, verbose);

% Compute synthetic STA for completeness
STA = (SpikeCounts_corrected * patches) / sum(abs(SpikeCounts_corrected));
STA = reshape(STA, imageSize);

% Store true weights for comparison
moment_results.true_weights = true_weights;
moment_results.weight_recovery_error = norm(moment_results.contributions - true_weights);

% Compile results
results = compile_results(STA, spikeCounts', rf_mask, rf_stats, moment_results, ...
                         imageSize, struct(), params);
results.synthetic = true;
results.true_weights = true_weights;

if verbose
    fprintf('Synthetic analysis complete. Weight recovery error: %.4f\n', ...
        moment_results.weight_recovery_error);
end

end

%% RECEPTIVE FIELD EXTRACTION FUNCTION
function [rf_mask, rf_stats] = extract_receptive_field(STA, threshold_factor, verbose)

if verbose, fprintf('Extracting receptive field mask...\n'); end

% Adaptive thresholding
sta_abs = abs(STA);
threshold_value = threshold_factor * max(sta_abs(:));
high_weight_mask = sta_abs >= threshold_value;

% Get spatial coordinates
[rows, cols] = size(sta_abs);
[X, Y] = meshgrid(1:cols, 1:rows);

% Compute center of mass
high_X = X(high_weight_mask);
high_Y = Y(high_weight_mask);

if isempty(high_X)
    warning('No pixels found above threshold. Using center of image.');
    x_center = cols/2;
    y_center = rows/2;
    rf_radius = min(rows, cols)/4;
else
    x_center = mean(high_X);
    y_center = mean(high_Y);
    
    % Robust radius estimation (90th percentile)
    distances = sqrt((high_X - x_center).^2 + (high_Y - y_center).^2);
    rf_radius = prctile(distances, 90);
end

% Create circular RF mask
distanceMap = sqrt((X - x_center).^2 + (Y - y_center).^2);
rf_mask = distanceMap <= rf_radius;

% Compile RF statistics
rf_stats.center = [x_center, y_center];
rf_stats.radius = rf_radius;
rf_stats.threshold = threshold_value;
rf_stats.pixel_count = sum(rf_mask(:));
rf_stats.percentage = (rf_stats.pixel_count / numel(rf_mask)) * 100;

if verbose
    fprintf('  RF center: [%.1f, %.1f]\n', x_center, y_center);
    fprintf('  RF radius: %.2f pixels\n', rf_radius);
    fprintf('  RF pixels: %d/%d (%.1f%%)\n', rf_stats.pixel_count, numel(rf_mask), rf_stats.percentage);
end

end

%% MOMENT ANALYSIS FUNCTION
function moment_results = analyze_moments(Stimuli, SpikeCounts_corrected, rf_mask, imageSize, params, verbose)

if verbose, fprintf('Performing moment decomposition analysis...\n'); end

% Extract RF-masked stimuli
numTrials = size(Stimuli, 1);
numPixels_rf = sum(rf_mask(:));
Stimuli_RF = zeros(numTrials, numPixels_rf);

for i = 1:numTrials
    stimFull = reshape(Stimuli(i, :), imageSize);
    Stimuli_RF(i, :) = stimFull(rf_mask)';
end

% Generate stimulus histograms
allPixels = Stimuli_RF(:);
pixelStd = std(allPixels);
histEdges = linspace(-3*pixelStd, 3*pixelStd, params.nbins + 1);
histCenters = (histEdges(1:end-1) + histEdges(2:end)) / 2;

StimuliHist = zeros(numTrials, params.nbins);
for i = 1:numTrials
    counts = histcounts(Stimuli_RF(i, :), histEdges, 'Normalization', 'probability');
    StimuliHist(i, :) = counts;
end

if verbose, fprintf('  Generated %d-bin histograms for %d trials\n', params.nbins, numTrials); end

% Cross-validation split
rng(42);  % For reproducibility
indices = randperm(numTrials);
train_size = round((1 - params.testSize) * numTrials);
train_idx = indices(1:train_size);
test_idx = indices(train_size+1:end);

% Split data
A_train = StimuliHist(train_idx, :);
A_test = StimuliHist(test_idx, :);
spikeCounts_train = SpikeCounts_corrected(train_idx);
spikeCounts_test = SpikeCounts_corrected(test_idx);
Stimuli_RF_train = Stimuli_RF(train_idx, :);
Stimuli_RF_test = Stimuli_RF(test_idx, :);

if verbose, fprintf('  Split: %d training, %d testing trials\n', train_size, length(test_idx)); end

% Compute moments for training set
moment0_train = ones(train_size, 1);              % Offset
moment1_train = mean(Stimuli_RF_train, 2);        % Mean
moment2_train = mean(Stimuli_RF_train.^2, 2);     % Second moment  
moment3_train = mean(Stimuli_RF_train.^3, 2);     % Third moment

% Fit ridge regression for each moment
if verbose, fprintf('  Fitting ridge regression models...\n'); end
[weights_zeroth, ~, ~] = manualRidgeRegressionCustom(A_train, moment0_train, histCenters, false, params.lambda);
[weights_mean, ~, ~] = manualRidgeRegressionCustom(A_train, moment1_train, histCenters, false, params.lambda);
[weights_second, ~, ~] = manualRidgeRegressionCustom(A_train, moment2_train, histCenters, false, params.lambda);
[weights_third, ~, ~] = manualRidgeRegressionCustom(A_train, moment3_train, histCenters, false, params.lambda);
[weights_real, ~, ~] = manualRidgeRegressionCustom(A_train, spikeCounts_train, histCenters, false, params.lambda);

% Decompose real weights into moment contributions
W_moments = [weights_zeroth(:), weights_mean(:), weights_second(:), weights_third(:)];
contributions = W_moments \ weights_real(:);

% Normalize to percentages
normalized_contributions = contributions / sum(abs(contributions)) * 100;

if verbose
    fprintf('  Moment contributions:\n');
    fprintf('    0th (Offset): %.2f%%\n', normalized_contributions(1));
    fprintf('    1st (Mean): %.2f%%\n', normalized_contributions(2));
    fprintf('    2nd Moment: %.2f%%\n', normalized_contributions(3));
    fprintf('    3rd Moment: %.2f%%\n', normalized_contributions(4));
end

% Cross-validation prediction
moment0_test = ones(length(test_idx), 1);
moment1_test = mean(Stimuli_RF_test, 2);
moment2_test = mean(Stimuli_RF_test.^2, 2);
moment3_test = mean(Stimuli_RF_test.^3, 2);

spikeCounts_predicted = contributions(1) * moment0_test + ...
                       contributions(2) * moment1_test + ...
                       contributions(3) * moment2_test + ...
                       contributions(4) * moment3_test;

% Compute R² for validation
r2_holdout = 1 - sum((spikeCounts_test - spikeCounts_predicted).^2) / ...
             sum((spikeCounts_test - mean(spikeCounts_test)).^2);

if verbose
    fprintf('  Cross-validation R²: %.4f\n', r2_holdout);
end

% Store results
moment_results.contributions = contributions;
moment_results.normalized_contributions = normalized_contributions;
moment_results.r2_holdout = r2_holdout;
moment_results.weights.zeroth = weights_zeroth;
moment_results.weights.mean = weights_mean;
moment_results.weights.second = weights_second;
moment_results.weights.third = weights_third;
moment_results.weights.real = weights_real;
moment_results.train_idx = train_idx;
moment_results.test_idx = test_idx;
moment_results.predicted_responses = spikeCounts_predicted;
moment_results.actual_responses = spikeCounts_test;

end

%% RESULTS COMPILATION FUNCTION
function results = compile_results(STA, SpikeCounts, rf_mask, rf_stats, moment_results, ...
                                  imageSize, analysis_params, params)

% Create comprehensive results structure
results.STA_original = STA;
results.rf_mask = rf_mask;
results.rf_stats = rf_stats;
results.SpikeCounts = SpikeCounts;
results.imageSize = imageSize;

% Moment analysis results
results.moment_contributions = moment_results.contributions;
results.normalized_contributions = moment_results.normalized_contributions;
results.r2_holdout = moment_results.r2_holdout;
results.weights = moment_results.weights;

% Cross-validation results
results.cv_results.train_idx = moment_results.train_idx;
results.cv_results.test_idx = moment_results.test_idx;
results.cv_results.predicted = moment_results.predicted_responses;
results.cv_results.actual = moment_results.actual_responses;

% Analysis parameters
results.analysis_summary.parameters = params;
results.analysis_summary.timestamp = datetime('now');
results.analysis_summary.total_trials = length(SpikeCounts);
results.analysis_summary.total_spikes = sum(SpikeCounts);
results.analysis_summary.mean_spikes_per_trial = mean(SpikeCounts);

% Add synthetic-specific results if applicable
if isfield(moment_results, 'true_weights')
    results.true_weights = moment_results.true_weights;
    results.weight_recovery_error = moment_results.weight_recovery_error;
end

end

%% SUMMARY PRINTING FUNCTION
function print_analysis_summary(results)

fprintf('\n--- ANALYSIS SUMMARY ---\n');
fprintf('Total trials: %d\n', results.analysis_summary.total_trials);
fprintf('Total spikes: %d\n', results.analysis_summary.total_spikes);
fprintf('Mean spikes/trial: %.2f\n', results.analysis_summary.mean_spikes_per_trial);
fprintf('RF pixels: %d (%.1f%% of image)\n', results.rf_stats.pixel_count, results.rf_stats.percentage);
fprintf('Cross-validation R²: %.4f\n', results.r2_holdout);

fprintf('\nMoment Contributions:\n');
moment_names = {'0th (Offset)', '1st (Mean)', '2nd Moment', '3rd Moment'};
for i = 1:length(moment_names)
    fprintf('  %s: %.2f%%\n', moment_names{i}, results.normalized_contributions(i));
end

if isfield(results, 'true_weights')
    fprintf('\nSynthetic Data Validation:\n');
    fprintf('  True weights: [%.2f, %.2f, %.2f, %.2f]\n', results.true_weights);
    fprintf('  Estimated weights: [%.2f, %.2f, %.2f, %.2f]\n', results.moment_contributions);
    fprintf('  Recovery error: %.4f\n', results.weight_recovery_error);
end

fprintf('Analysis completed at: %s\n', char(results.analysis_summary.timestamp));

end

%% VISUALIZATION FUNCTION
function generate_summary_plots(results)

% Create comprehensive visualization
figure('Name', 'Spike Triggered Moments Analysis', 'Position', [100, 100, 1200, 800]);

% Plot 1: Spike-triggered average with RF mask
subplot(2, 4, 1);
imagesc(results.STA_original);
colormap(gca, gray);
axis image;
title('Spike-Triggered Average');
colorbar;

subplot(2, 4, 2);
imagesc(results.STA_original);
hold on;
contour(results.rf_mask, [0.5 0.5], 'r', 'LineWidth', 2);
plot(results.rf_stats.center(1), results.rf_stats.center(2), 'b+', 'MarkerSize', 12, 'LineWidth', 2);
colormap(gca, gray);
axis image;
title('STA with RF Mask');

% Plot 3: Moment contributions
subplot(2, 4, 3);
bar(results.normalized_contributions, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTickLabel', {'0th', '1st', '2nd', '3rd'});
ylabel('Contribution (%)');
title('Moment Contributions');
grid on;

% Plot 4: Cross-validation scatter
subplot(2, 4, 4);
scatter(results.cv_results.actual, results.cv_results.predicted, 50, 'filled');
hold on;
lims = [min([results.cv_results.actual; results.cv_results.predicted]), ...
        max([results.cv_results.actual; results.cv_results.predicted])];
plot(lims, lims, 'r--', 'LineWidth', 2);
xlabel('Actual Spike Counts');
ylabel('Predicted Spike Counts');
title(sprintf('Cross-Validation (R² = %.3f)', results.r2_holdout));
grid on;
axis equal;

% Plot 5-8: Individual moment weights
moment_names = {'0th (Offset)', '1st (Mean)', '2nd Moment', '3rd Moment'};
weight_fields = {'zeroth', 'mean', 'second', 'third'};

for i = 1:4
    subplot(2, 4, 4 + i);
    plot(results.weights.(weight_fields{i}), 'o-', 'LineWidth', 1.5);
    title([moment_names{i} ' Weights']);
    xlabel('Bin Index');
    ylabel('Weight');
    grid on;
end

% Add main title
sgtitle('Spike-Triggered Moments Analysis Results', 'FontSize', 16, 'FontWeight', 'bold');

% If synthetic data, show weight comparison
if isfield(results, 'true_weights')
    figure('Name', 'Synthetic Data Validation', 'Position', [150, 150, 800, 400]);
    
    subplot(1, 2, 1);
    x = 1:4;
    bar(x - 0.2, results.true_weights, 0.4, 'FaceColor', [0.8 0.2 0.2]);
    hold on;
    bar(x + 0.2, results.moment_contributions, 0.4, 'FaceColor', [0.2 0.6 0.8]);
    set(gca, 'XTickLabel', moment_names);
    ylabel('Weight Value');
    title('True vs Estimated Weights');
    legend('True', 'Estimated', 'Location', 'best');
    grid on;
    
    subplot(1, 2, 2);
    error_vals = results.moment_contributions - results.true_weights;
    bar(error_vals, 'FaceColor', [0.6 0.6 0.6]);
    set(gca, 'XTickLabel', moment_names);
    ylabel('Error (Estimated - True)');
    title(sprintf('Weight Recovery Error (Total: %.3f)', results.weight_recovery_error));
    grid on;
end

end
