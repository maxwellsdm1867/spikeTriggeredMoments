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

contours = 8;
node = gui.getSelectedEpochTreeNodes;
cdt_node = node{1};
figure(3); clf; hold on;
curNode = 1;

SampleEpoch = cdt_node.epochList.elements(1);
SamplingInterval = 1000/SampleEpoch.protocolSettings.get('sampleRate');  % in msec
persistentGrate = SampleEpoch.protocolSettings.get('persistentGrate');
PrePts = SampleEpoch.protocolSettings.get('preTime') / SamplingInterval;
StmPts = SampleEpoch.protocolSettings.get('stimTime') / SamplingInterval;
TailPts = SampleEpoch.protocolSettings.get('tailTime') / SamplingInterval;
numNoiseRepeats = SampleEpoch.protocolSettings.get('numNoiseRepeats');
noiseFilterSD = SampleEpoch.protocolSettings.get('noiseFilterSD');
imageName = cdt_node.epochList.elements(1).protocolSettings.get('imageName');

clear STA

rig_info = cdt_node.epochList.elements(1).protocolSettings.get('experiment:rig');
if rig_info(1) == 'B'
    imageSize = [90 120];
elseif rig_info(1) == 'G'
    imageSize = [166 268];
else
    display(rig_info)
end

STA{curNode} = zeros(imageSize);
EpochData = getSelectedData(cdt_node.epochList, params.Amp);
[SpikeTimes, ~, ~] = SpikeDetection.Detector(EpochData);
totSpikes = 0;
numTrials = size(EpochData, 1) * numNoiseRepeats;
SpikeCounts = zeros(1, numTrials);
Stimuli = zeros(numTrials, imageSize(1) * imageSize(2));
trialIndex = 1;

for epoch = 1:size(EpochData, 1)
    for repeat = 1:numNoiseRepeats
        startTrial = PrePts * repeat + (StmPts + TailPts) * (repeat-1);
        numSpikes = length(find(SpikeTimes{epoch} > startTrial & SpikeTimes{epoch} < (startTrial + StmPts)));
        noiseMatrix = imgaussfilt(randn(imageSize), noiseFilterSD); noiseMatrix = noiseMatrix / std(noiseMatrix(:));
        %figure; imagesc(noiseMatrix)
        SpikeCounts(trialIndex) = numSpikes;
        Stimuli(trialIndex, :) = noiseMatrix(:)';
        STA{curNode} = STA{curNode} + (numSpikes - mean(SpikeCounts)) * noiseMatrix;
        trialIndex = trialIndex + 1; totSpikes = totSpikes + numSpikes;
    end
end

STA{curNode} = STA{curNode} / totSpikes;

% RF Mask
sta_abs = abs(STA{curNode});
threshold_value = 0.4 * max(sta_abs(:));
high_weight_mask = sta_abs >= threshold_value;
[X, Y] = meshgrid(1:imageSize(2), 1:imageSize(1));
x_center = mean(X(high_weight_mask)); y_center = mean(Y(high_weight_mask));
distances = sqrt((X(high_weight_mask) - x_center).^2 + (Y(high_weight_mask) - y_center).^2);
rf_radius = prctile(distances, 90);
rf_mask = sqrt((X - x_center).^2 + (Y - y_center).^2) <= rf_radius;

% Extract RF Stimuli
numPixels_rf = sum(rf_mask(:));
Stimuli_RF = zeros(numTrials, numPixels_rf);
for i = 1:numTrials
    stimFull = reshape(Stimuli(i, :), imageSize);
    Stimuli_RF(i, :) = stimFull(rf_mask)';
end

%% ====================== HISTOGRAM BINNING ======================
allPixels = Stimuli_RF(:);
pixelStd = std(allPixels);
histEdges = linspace(-3*pixelStd, 3*pixelStd, 13); % 12 bins
histCenters = (histEdges(1:end-1) + histEdges(2:end)) / 2;

StimuliHist = zeros(numTrials, length(histCenters));
for trialIndex = 1:numTrials
    counts = histcounts(Stimuli_RF(trialIndex, :), histEdges, 'Normalization', 'probability');
    StimuliHist(trialIndex, :) = counts;
end

%% ====================== MOMENT DECOMPOSITION & VALIDATION ======================
numTrials = size(Stimuli_RF, 1); rng(1); indices = randperm(numTrials);
train_size = round(0.7 * numTrials); train_idx = indices(1:train_size); test_idx = indices(train_size+1:end);
A_train = StimuliHist(train_idx, :); A_test = StimuliHist(test_idx, :);
spikeCounts_train = SpikeCounts(train_idx)'; spikeCounts_test = SpikeCounts(test_idx)';
Stimuli_RF_train = Stimuli_RF(train_idx, :); Stimuli_RF_test = Stimuli_RF(test_idx, :);

moment0_train = ones(train_size, 1);
moment1_train = mean(Stimuli_RF_train, 2);
moment2_train = mean(Stimuli_RF_train.^2, 2);
moment3_train = mean(Stimuli_RF_train.^3, 2);

lambda_value = 0; rigd_plot = false;
[weights_zeroth, ~, ~] = manualRidgeRegressionCustom(A_train, moment0_train, histCenters, rigd_plot, lambda_value);
[weights_mean, ~, ~] = manualRidgeRegressionCustom(A_train, moment1_train, histCenters, rigd_plot, lambda_value);
[weights_second, ~, ~] = manualRidgeRegressionCustom(A_train, moment2_train, histCenters, rigd_plot, lambda_value);
[weights_third, ~, ~] = manualRidgeRegressionCustom(A_train, moment3_train, histCenters, rigd_plot, lambda_value);
[weights_real_train, ~, ~] = manualRidgeRegressionCustom(A_train, spikeCounts_train, histCenters, false, lambda_value);

% Moment Contributions
W_moments = [weights_zeroth(:), weights_mean(:), weights_second(:), weights_third(:)];
a_estimated = W_moments \ weights_real_train(:);
fprintf('\n--- Estimated Contributions (Training Set) ---\n');
fprintf('0th Moment (Offset): %.4f\n', a_estimated(1));
fprintf('1st Moment (Mean): %.4f\n', a_estimated(2));
fprintf('2nd Moment: %.4f\n', a_estimated(3));
fprintf('3rd Moment: %.4f\n', a_estimated(4));

a_normalized = a_estimated / sum(abs(a_estimated)) * 100;
fprintf('\n--- Relative Contributions (%% of total) ---\n')
fprintf('0th Moment (Offset): %.2f%%\n', a_normalized(1));
fprintf('1st Moment (Mean): %.2f%%\n', a_normalized(2));
fprintf('2nd Moment: %.2f%%\n', a_normalized(3));
fprintf('3rd Moment: %.2f%%\n', a_normalized(4));

% Predict Hold-Out
moment0_test = ones(length(test_idx), 1);
moment1_test = mean(Stimuli_RF_test, 2);
moment2_test = mean(Stimuli_RF_test.^2, 2);
moment3_test = mean(Stimuli_RF_test.^3, 2);

spikeCounts_predicted_test = a_estimated(1) * moment0_test + ...
                             a_estimated(2) * moment1_test + ...
                             a_estimated(3) * moment2_test + ...
                             a_estimated(4) * moment3_test;
r2_holdout = 1 - sum((spikeCounts_test - spikeCounts_predicted_test).^2) / sum((spikeCounts_test - mean(spikeCounts_test)).^2);
fprintf('\n--- Hold-Out Prediction R²: %.4f\n', r2_holdout);
