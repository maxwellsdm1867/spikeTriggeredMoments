%% Step 1: Generate 2D Pink Noise
N = 512;  % Image size
[x, y] = meshgrid(1:N, 1:N);
cx = ceil(N/2); cy = ceil(N/2);
f = sqrt((x-cx).^2 + (y-cy).^2);
f(cx,cy) = 1;  % Avoid division by zero at DC

whiteNoise = randn(N);
spectrum = fftshift(fft2(whiteNoise)) ./ f;
pinkImg = real(ifft2(ifftshift(spectrum)));
pinkImg = (pinkImg - min(pinkImg(:))) / (max(pinkImg(:)) - min(pinkImg(:)));  % Normalize [0,1]

figure; imagesc(pinkImg); colormap gray; axis image; title('2D Pink Noise');

%% Step 2: Sample Random Patches
patchSize = [16, 16];
numTrials = 100;
patches = zeros(numTrials, patchSize(1) * patchSize(2));

for i = 1:numTrials
    x = randi(N - patchSize(1) + 1);
    y = randi(N - patchSize(2) + 1);
    patch = pinkImg(x:x+patchSize(1)-1, y:y+patchSize(2)-1);
    patches(i, :) = patch(:)';
end

%% Step 3: Simulate Responses (Mean, Raw Second Moment, Combo)
patchMeans = mean(patches, 2);
patchSecondMoment = mean(patches.^2, 2);
sec_scale = 5;

spikeCounts_mean = patchMeans;
spikeCounts_secondMoment = patchSecondMoment;
spikeCounts_combo = patchMeans + sec_scale*patchSecondMoment;

%% Step 4: Bin Each Patch into Histograms
nbins = 10;
binEdges = linspace(0, 1, nbins + 1);
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;

stimHist = zeros(numTrials, nbins);
for i = 1:numTrials
    stimHist(i,:) = histcounts(patches(i,:), binEdges, 'Normalization', 'probability');
end

%% Step 5: Run Ridge Regression for Each Response Type
lambda_value = 0;  % Linear regression (set [] for auto ridge)

% Mean-driven response
fprintf('\n--- Mean Response ---\n');
[weights_mean, r2_mean, lambda_used_mean] = manualRidgeRegressionCustom(stimHist, spikeCounts_mean, binCenters, true, lambda_value);

% Second Moment-driven response
fprintf('\n--- Second Moment Response ---\n');
[weights_second, r2_second, lambda_used_second] = manualRidgeRegressionCustom(stimHist, spikeCounts_secondMoment, binCenters, true, lambda_value);

% Combined Mean + Second Moment response
fprintf('\n--- Combo (Mean + Second Moment) Response ---\n');
[weights_combo, r2_combo, lambda_used_combo] = manualRidgeRegressionCustom(stimHist, spikeCounts_combo, binCenters, true, lambda_value);


%%
% Estimate sec_scale using least squares fitting
% weights_combo ≈ weights_mean + sec_scale * weights_second

% Reshape vectors to ensure compatibility
w_combo = weights_combo(:);
w_mean = weights_mean(:);
w_second = weights_second(:);

% Solve for sec_scale using linear regression:
% w_combo - w_mean ≈ sec_scale * w_second

delta_w = w_combo - w_mean;

% Solve sec_scale = (w_second' * delta_w) / (w_second' * w_second)
sec_scale_estimated = (w_second' * delta_w) / (w_second' * w_second);

fprintf('Estimated sec_scale: %.4f\n', sec_scale_estimated);

% Optional: check fit quality
weights_combo_estimated = weights_mean + sec_scale_estimated * weights_second;
combo_fit_error = norm(weights_combo - weights_combo_estimated);

fprintf('Fit Error (||weights_combo - estimated||): %.6f\n', combo_fit_error);

% Visualize comparison
figure;
plot(w_combo, 'k-o', 'LineWidth', 1.5); hold on;
plot(weights_combo_estimated, 'r--', 'LineWidth', 1.5);
legend('Original Combo Weights', 'Recovered Combo Weights');
xlabel('Bin Index');
ylabel('Weight Value');
title(sprintf('Recovered sec\\_scale = %.4f', sec_scale_estimated));
grid on;