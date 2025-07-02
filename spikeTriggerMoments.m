% this is for spike triggered moments for the different cell types
% using the FlashedGratePlusNoise protocol


close all
clear all
clc


%*************************************************************************
% Initializations
%*************************************************************************

% define plot color sequence, axis fonts
PlotColors = 'bgrkymcbgrkymcbgrkymcbgrkymc';
set(0, 'DefaultAxesFontName','Helvetica')
set(0, 'DefaultAxesFontSize', 16)

colormap([0 0 0])
scrsz = get(0, 'ScreenSize');

% jauimodel stuff
loader = edu.washington.rieke.Analysis.getEntityLoader();
treeFactory = edu.washington.rieke.Analysis.getEpochTreeFactory();

%Data and export folder paths
dataFolder = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/Data/';
exportFolder = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/Data/';


import auimodel.*
import vuidocument.*

%
params.FrequencyCutoff = 500;
params.Amp = 'Amp1';
params.Verbose = 1;
params.DecimatePts = 200;
params.SamplingInterval = 0.0001;
params.FrameRate = 60.1830;
params.OvationFlag = 1;
params.SpatialFlag = 0;
params.SaveToIgor = 0;
params.SaveGraphs = 1;
params.rootDir = '~/Dropbox/LargeCells/';



%
%*************************************************************************
list = loader.loadEpochList([exportFolder 'FlashedGratePlusNoise.mat'], dataFolder);

%Tree
dateSplit = @(list)splitOnExperimentDate(list);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(list, dateSplit);
tree = riekesuite.analysis.buildTree(list, {'protocolSettings(source:type)',dateSplit_java,'cell.label','protocolSettings(linearizeCones)','protocolSettings(noiseContrast)','protocolSettings(imagePatchIndex)'});


gui = epochTreeGUI(tree);

%%
% 

contours = 8;

node = gui.getSelectedEpochTreeNodes;

cdt_node = node{1};
figure(3); clf; hold on;
curNode = 1;
SampleEpoch = cdt_node.epochList.elements(1);
SamplingInterval = SampleEpoch.protocolSettings.get('sampleRate');       % in Hz
SamplingInterval = 1000/SamplingInterval;         % now in msec
persistentGrate = SampleEpoch.protocolSettings.get('persistentGrate');
PrePts = SampleEpoch.protocolSettings.get('preTime') / SamplingInterval;
StmPts = SampleEpoch.protocolSettings.get('stimTime') / SamplingInterval;
TailPts = SampleEpoch.protocolSettings.get('tailTime') / SamplingInterval;
numNoiseRepeats = SampleEpoch.protocolSettings.get('numNoiseRepeats');
noiseFilterSD = SampleEpoch.protocolSettings.get('noiseFilterSD');
imageName = cdt_node.epochList.elements(1).protocolSettings.get('imageName');

clear STA

tempLoc = cdt_node.epochList.elements(1).protocolSettings.get('currentPatchLocation');
rig_info = cdt_node.epochList.elements(1).protocolSettings.get('experiment:rig');
if rig_info(1) == 'B'
    imageSize = [90 120];
elseif rig_info(1) == 'G'
    imageSize = [166 268];
else
    display(rig_info)
end

tempLocArray = tempLoc.toArray;
patchLocations(1, :) = tempLocArray(:);
sampledPatches = getNaturalImagePatchFromLocation2(patchLocations, imageName, 'imageSize', imageSize * 6.6);

STA{curNode} = zeros(imageSize);
EpochData = getSelectedData(cdt_node.epochList, params.Amp);
[SpikeTimes, SpikeAmplitudes, RefractoryViolations] = SpikeDetection.Detector(EpochData);
totSpikes = 0;

% New: Initialize trial-wise storage
numTrials = size(EpochData, 1) * numNoiseRepeats;
SpikeCounts = zeros(1, numTrials);  % 1 x N array
Stimuli = zeros(numTrials, imageSize(1) * imageSize(2));  % N x M matrix
trialIndex = 1;

% Compute total spikes
for epoch = 1:size(EpochData, 1)
    for repeat = 1:numNoiseRepeats
        startTrial = PrePts * repeat + (StmPts + TailPts) * (repeat-1);
        numSpikes = length(find(SpikeTimes{epoch} > startTrial & SpikeTimes{epoch} < (startTrial + StmPts)));
        totSpikes = totSpikes + numSpikes;
    end
end

if (persistentGrate)
    meanSpikes = 0;
else
    meanSpikes = totSpikes / (epoch * numNoiseRepeats);
end

% Main loop: Compute STA, store spike counts & stimuli
for epoch = 1:size(EpochData, 1)
    noiseSeed = cdt_node.epochList.elements(epoch).protocolSettings.get('noiseSeed');
    noiseStream = RandStream('mt19937ar', 'Seed', noiseSeed);

    for repeat = 1:numNoiseRepeats
        startTrial = PrePts * repeat + (StmPts + TailPts) * (repeat-1);
        numSpikes = length(find(SpikeTimes{epoch} > startTrial & SpikeTimes{epoch} < (startTrial + StmPts)));

        noiseMatrix = imgaussfilt(noiseStream.randn(imageSize(1), imageSize(2)), noiseFilterSD);
        noiseMatrix = noiseMatrix / std(noiseMatrix(:));

        % Store trial-wise data
        SpikeCounts(trialIndex) = numSpikes;
        Stimuli(trialIndex, :) = noiseMatrix(:)';  % Flatten and store as row

        % Update STA
        STA{curNode} = STA{curNode} + (numSpikes - meanSpikes) * noiseMatrix;

        trialIndex = trialIndex + 1;
    end
end

STA{curNode} = STA{curNode} / totSpikes;

% Plot STA and contours
subplot(2, length(cdt_node), curNode);
imagesc([1 imageSize(2)], [imageSize(1) 1], STA{curNode}, [-0.5 0.5]); colormap gray;
xlim([40 80]); ylim([25 65]);

subplot(2, length(cdt_node), length(cdt_node) + curNode);
contour(STA{curNode}, contours, 'k', 'LineWidth', 2); grid on;
xlim([40 80]); ylim([25 65]);

pause(1);

%*************************************************************************
% Recompute STA from SpikeCounts & Stimuli matrices with mean correction
%*************************************************************************
SpikeCounts_corrected = SpikeCounts - meanSpikes;  % Mean spike correction
STA_from_matrices = (SpikeCounts_corrected * Stimuli) / sum(SpikeCounts);  % 1 x M vector
STA_from_matrices = reshape(STA_from_matrices, imageSize);  % Reshape back to imageSize
% Compute RMSE between original STA and STA from matrices
errorRMSE = sqrt(mean((STA{curNode}(:) - STA_from_matrices(:)).^2));

% Print warning if error is too large
if errorRMSE > 1e-3
    fprintf('WARNING: High STA error for node %d: RMSE = %.5f\n', curNode, errorRMSE);
end

% Compare with STA{curNode}
figure(100 + curNode); clf;

subplot(1,3,1);
imagesc([1 imageSize(2)], [imageSize(1) 1], STA{curNode}, [-0.5 0.5]);
title(sprintf('Original STA'));
colormap gray; axis image;

subplot(1,3,2);
imagesc([1 imageSize(2)], [imageSize(1) 1], STA_from_matrices, [-0.5 0.5]);
title(sprintf('STA from Matrices | RMSE = %.5f', errorRMSE));
colormap gray; axis image;

subplot(1,3,3);
imagesc([1 imageSize(2)], [imageSize(1) 1], STA{curNode} - STA_from_matrices, [-0.1 0.1]);
title('Diff');
colormap gray; axis image; colorbar;

% Threshold-based RF region extraction
sta_abs = abs(STA{curNode});

% Set threshold relative to max value (e.g., 50% of max)
threshold_value = 0.5 * max(sta_abs(:));  % Adjustable

% Find pixels above threshold
high_weight_mask = sta_abs >= threshold_value;

% Get indices of high-weight pixels
[rows, cols] = size(sta_abs);
[X, Y] = meshgrid(1:cols, 1:rows);
high_X = X(high_weight_mask);
high_Y = Y(high_weight_mask);

% Compute center of mass of those pixels
x_center = mean(high_X);
y_center = mean(high_Y);

% Compute distances of those pixels from the center
distances = sqrt((high_X - x_center).^2 + (high_Y - y_center).^2);

% Robust radius: 90th percentile of distances (avoiding noise)
rf_radius = prctile(distances, 90);  % More robust than max

fprintf('Robust RF Radius (90th percentile): %.2f pixels\n', rf_radius);
% Calculate RF pixel count
rf_pixel_count = sum(rf_mask(:));
total_pixel_count = numel(rf_mask);
rf_pixel_percentage = (rf_pixel_count / total_pixel_count) * 100;

% Print RF pixel stats
fprintf('RF Pixels: %d / %d (%.2f%% of total pixels)\n', rf_pixel_count, total_pixel_count, rf_pixel_percentage);
% Create circular mask based on computed radius
distanceMap = sqrt((X - x_center).^2 + (Y - y_center).^2);
rf_mask = distanceMap <= rf_radius;

% Visualize STA with RF mask and center
figure; imagesc(STA{curNode}); colormap gray; hold on;
visboundaries(rf_mask, 'Color', 'r');
plot(x_center, y_center, 'b+', 'MarkerSize', 12, 'LineWidth', 2);  % Center of mass
title(sprintf('Threshold RF Mask (Threshold = %.2f * max STA)', threshold_value / max(sta_abs(:))));



%*************************************************************************
% Store results into 'results' structure with explanations
%*************************************************************************

results.STA_original = STA{curNode};
% The Spike-Triggered Average computed in real-time during the loop, with mean spike correction.

results.STA_from_matrices = STA_from_matrices;
% The STA recomputed from SpikeCounts and Stimuli matrices. Should match STA_original.

results.SpikeCounts = SpikeCounts;
% 1 x N array, where N = number of trials. Each entry is the spike count for a given trial.

results.SpikeCounts_corrected = SpikeCounts_corrected;
% SpikeCounts with meanSpikes subtracted. Used to center spike data during STA calculation.


results.meanSpikes = meanSpikes;
% Average number of spikes per trial. Used for centering spike counts.

results.totSpikes = totSpikes;
% Total number of spikes across all trials for normalization in STA computation.

results.imageSize = imageSize;
% Dimensions of the stimulus images, used to reshape flattened stimuli: [height, width].

results.patchLocations = patchLocations;
% The location in the natural image from which the stimulus patch was sampled.

results.imageName = imageName;
% The name or identifier of the natural image used in the experiment.



% Assume Stimuli (numTrials x totalPixels) already exists
% Reshape RF mask to use for all trials
numPixels_rf = sum(rf_mask(:));
Stimuli_RF = zeros(numTrials, numPixels_rf);  % Only masked pixels

% Apply mask to each trial
for i = 1:numTrials
    stimFull = reshape(Stimuli(i, :), imageSize);  % Reshape to image
    maskedStim = stimFull(rf_mask);  % Extract only RF pixels
    Stimuli_RF(i, :) = maskedStim';
end

fprintf('Extracted Stimuli_RF with %d pixels per trial inside RF mask.\n', numPixels_rf);





% Compute the mean across the entire Stimuli ensemble
Stimuli_mean = Stimuli_RF;

% Subtract mean from all stimuli points
Stimuli_centered = Stimuli_RF;

%*************************************************************************
% HISTOGRAM WITH EQUAL-WIDTH BINS (Original StimuliHist)
%*************************************************************************
allPixels = Stimuli_centered(:);
pixelStd = std(allPixels);
binRange_3sigma = 3 * pixelStd;

% 2. Define symmetric min and max based on ±3σ
minVal = -binRange_3sigma;
maxVal = binRange_3sigma;

% 3. Define equal-width bin edges within ±3σ
nbins = 12;  % Number of bins
histEdges = linspace(minVal, maxVal, nbins + 1);
histCenters = (histEdges(1:end-1) + histEdges(2:end)) / 2;

% 4. Initialize histogram matrix
StimuliHist = zeros(numTrials, nbins);

% 5. Loop over trials to compute histogram (equal-width bins within ±3σ)
for trialIndex = 1:numTrials
    centeredStimulus = Stimuli_centered(trialIndex, :);
    
    % Optional: Clip stimulus to ±3σ if you want strictly within that range
    % centeredStimulus = centeredStimulus(centeredStimulus >= minVal & centeredStimulus <= maxVal);
    
    counts = histcounts(centeredStimulus, histEdges, 'Normalization', 'probability');
    StimuliHist(trialIndex, :) = counts;
end

% 6. Visualize mean histogram
% figure;
% bar(histCenters, mean(StimuliHist, 1));
% xlabel('Pixel Value (Bin Centers within ±3σ)');
% ylabel('Probability');
% title('Mean Equal-Width Histogram Within ±3σ');
% grid on;

%*************************************************************************
% HISTOGRAM WITH EQUAL-POPULATED BINS (New StimuliHistEqualPop)
%*************************************************************************

% Compute equally populated bin edges from all centered pixels
allPixels = Stimuli_centered(:);
quantileEdges = quantile(allPixels, linspace(0, 1, nbins + 1));

% Initialize histogram matrix for equal-populated bins
StimuliHistEqualPop = zeros(numTrials, nbins);

% Loop over trials to compute histogram for each centered stimulus (equal-pop bins)
for trialIndex = 1:numTrials
    centeredStimulus = Stimuli_centered(trialIndex, :);
    counts = histcounts(centeredStimulus, quantileEdges, 'Normalization', 'probability');
    StimuliHistEqualPop(trialIndex, :) = counts;
end


results.StimuliHist = StimuliHist;  
% N x nbins matrix. Each row is a histogram of centered pixel values from the corresponding stimulus using equal-width bins.

results.histEdges = histEdges;  
% Bin edges derived from centered Stimuli ensemble (equal-width).

results.StimuliHistEqualPop = StimuliHistEqualPop;  
% N x nbins matrix. Each row is a histogram of centered pixel values from the corresponding stimulus using equal-populated bins.

results.quantileEdges = quantileEdges;  
% Bin edges derived from quantiles (equal-populated bins).

results.Stimuli_mean = Stimuli_mean;  
% The mean value subtracted from all stimuli before histogramming.
%keyboard

[weights, r2, lambda] = manualRidgeRegressionCustom(StimuliHist, SpikeCounts', histCenters, true);
% %*************************************************************************
%%
%% 1. Use StimuliHist (already computed from your RF-masked stimuli)
A = StimuliHist;  % Stimulus histogram matrix (N x nbins)
binCenters = histCenters;  % From earlier
numTrials = size(A,1);

%% 2. Compute Moments of Each Trial (Using RF-masked pixels)
% Compute raw pixel moments for each trial from Stimuli_RF
moment1 = mean(Stimuli_RF, 2);          % Mean
moment2 = mean(Stimuli_RF.^2, 2);       % Raw second moment
moment3 = mean(Stimuli_RF.^3, 2);       % Raw third moment

%% 3. Simulate Responses
spikeCounts_meanDriven = moment1;
spikeCounts_secondMomentDriven = moment2;
spikeCounts_thirdMomentDriven = moment3;

% Real observed spike counts from data
spikeCounts_real = SpikeCounts_corrected';

%% 4. Fit Weights for Each Moment-Driven Response Using Stimulus Histograms
lambda_value = 0;  % Linear regression for pure decomposition

fprintf('\n--- Mean-Driven Response ---\n');
[weights_mean, r2_mean, lambda_used_mean] = manualRidgeRegressionCustom(A, spikeCounts_meanDriven, binCenters, true, lambda_value);

fprintf('\n--- Second Moment-Driven Response ---\n');
[weights_second, r2_second, lambda_used_second] = manualRidgeRegressionCustom(A, spikeCounts_secondMomentDriven, binCenters, true, lambda_value);

fprintf('\n--- Third Moment-Driven Response ---\n');
[weights_third, r2_third, lambda_used_third] = manualRidgeRegressionCustom(A, spikeCounts_thirdMomentDriven, binCenters, true, lambda_value);

%% 5. Fit the Real Data (from actual spikeCounts)
fprintf('\n--- Real Data Fit ---\n');
[weights_real, r2_real, lambda_used_real] = manualRidgeRegressionCustom(A, spikeCounts_real, binCenters, true, lambda_value);

%% 6. Decompose Real Weights Using Moment Weights
% Stack moment-driven weights
W_moments = [weights_mean(:), weights_second(:), weights_third(:)];  % nbins x 3
w_real = weights_real(:);  % nbins x 1

% Solve w_real ≈ W_moments * [a1; a2; a3]
a_estimated = W_moments \ w_real;  % Least squares estimate

% Show estimated contributions
fprintf('\n--- Estimated Moment Contributions to Real Weights ---\n');
fprintf('1st Moment (Mean): %.4f\n', a_estimated(1));
fprintf('2nd Moment: %.4f\n', a_estimated(2));
fprintf('3rd Moment: %.4f\n', a_estimated(3));

% Reconstruct the real weights from the estimated moment contributions
weights_real_est = W_moments * a_estimated;
fit_error = norm(w_real - weights_real_est);

fprintf('Reconstruction Fit Error: %.6f\n', fit_error);

%% 7. Visual Comparison
figure;
plot(w_real, 'k-o', 'LineWidth', 1.5); hold on;
plot(weights_real_est, 'r--', 'LineWidth', 1.5);
legend('Real Data Weights', 'Reconstructed from Moments');
xlabel('Bin Index');
ylabel('Weight');
title(sprintf('Moment Decomposition | Fit Error = %.5f', fit_error));
grid on;
