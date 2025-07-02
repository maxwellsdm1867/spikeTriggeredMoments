%% 1. Split Data into Training and Hold-Out Sets (70%-30%)
numTrials = size(Stimuli_RF, 1);
rng(1);  % For reproducibility
indices = randperm(numTrials);

train_size = round(0.7 * numTrials);
train_idx = indices(1:train_size);
test_idx = indices(train_size+1:end);

% Split everything
A_train = A(train_idx, :);  % Histograms for training
A_test = A(test_idx, :);    % Histograms for testing

spikeCounts_train = spikeCounts_real(train_idx);  % Real spike counts
spikeCounts_test = spikeCounts_real(test_idx);

Stimuli_RF_train = Stimuli_RF(train_idx, :);
Stimuli_RF_test = Stimuli_RF(test_idx, :);

%% 2. Compute Moments for Training Set (Including Zeroth Moment)
moment0_train = ones(train_size, 1);                 % Zeroth Moment (constant offset)
moment1_train = mean(Stimuli_RF_train, 2);           % Mean
moment2_train = mean(Stimuli_RF_train.^2, 2);        % Second Moment
moment3_train = mean(Stimuli_RF_train.^3, 2);        % Third Moment

%% 3. Fit Moment-Driven Weights on Training Set
lambda_value = 0;
rigd_plot = true;

fprintf('\n--- Training Set: Zeroth Moment (Offset) ---\n');
[weights_zeroth, ~, ~] = manualRidgeRegressionCustom(A_train, moment0_train, binCenters,rigd_plot , lambda_value);

fprintf('\n--- Training Set: Mean-Driven ---\n');
[weights_mean, ~, ~] = manualRidgeRegressionCustom(A_train, moment1_train, binCenters, rigd_plot, lambda_value);

fprintf('\n--- Training Set: Second Moment ---\n');
[weights_second, ~, ~] = manualRidgeRegressionCustom(A_train, moment2_train, binCenters, rigd_plot, lambda_value);

fprintf('\n--- Training Set: Third Moment ---\n');
[weights_third, ~, ~] = manualRidgeRegressionCustom(A_train, moment3_train, binCenters, rigd_plot, lambda_value);

%% 4. Fit Real Data Weights on Training Set
fprintf('\n--- Training Set: Real Spike Counts ---\n');
[weights_real_train, ~, ~] = manualRidgeRegressionCustom(A_train, spikeCounts_train, binCenters, false, lambda_value);

%% 5. Decompose Real Weights into Moment Contributions (Including Zeroth Moment)
W_moments = [weights_zeroth(:), weights_mean(:), weights_second(:), weights_third(:)];
w_real_train = weights_real_train(:);
% Use weights_real_train to predict spike counts in hold-out set
spikeCounts_predicted_holdout_trainWeights = A_test * w_real_train;

% Compute residuals and R² for this prediction
residuals_holdout_trainWeights = spikeCounts_test - spikeCounts_predicted_holdout_trainWeights;
R2_holdout_trainWeights = 1 - (sum(residuals_holdout_trainWeights.^2) / sum((spikeCounts_test - mean(spikeCounts_test)).^2));

fprintf('\n--- Hold-Out Prediction Using Training Weights ---\n');
fprintf('R² on Hold-Out Set (Training Weights): %.4f\n', R2_holdout_trainWeights);
% Visualize prediction
figure;
scatter(spikeCounts_test, spikeCounts_predicted_holdout_trainWeights, 50, 'filled');
hold on;
plot([min(spikeCounts_test) max(spikeCounts_test)], [min(spikeCounts_test) max(spikeCounts_test)], 'r--');
xlabel('Actual Spike Counts (Hold-Out)');
ylabel('Predicted Spike Counts (Hold-Out)');
title(sprintf('Hold-Out Prediction | Training Weights | R² = %.3f', R2_holdout_trainWeights));
grid on;


a_estimated = W_moments \ w_real_train;

fprintf('\n--- Estimated Contributions (Training Set, Including Zeroth) ---\n');
fprintf('0th Moment (Offset): %.4f\n', a_estimated(1));
fprintf('1st Moment (Mean): %.4f\n', a_estimated(2));
fprintf('2nd Moment: %.4f\n', a_estimated(3));
fprintf('3rd Moment: %.4f\n', a_estimated(4));

% Reconstruct real weights from moment contributions
weights_real_est = W_moments * a_estimated;

% Compute residuals and R²
residuals_weights = w_real_train - weights_real_est;
R2_weights = 1 - (sum(residuals_weights.^2) / sum((w_real_train - mean(w_real_train)).^2));

fprintf('R² for Weight Reconstruction: %.4f\n', R2_weights);

% Visualize real vs reconstructed weights
figure;
plot(w_real_train, 'k-o', 'LineWidth', 1.5); hold on;
plot(weights_real_est, 'r--', 'LineWidth', 1.5);
legend('Real Weights', 'Reconstructed Weights');
xlabel('Bin Index');
ylabel('Weight Value');
title(sprintf('Weight Reconstruction | R² = %.4f', R2_weights));
grid on;

%% 5b. Compute and Print Relative Contributions as Percentages
% Normalize contributions to percentages
a_normalized = a_estimated / sum(abs(a_estimated)) * 100;

fprintf('\n--- Relative Contributions (as %% of total) ---\n');
fprintf('0th Moment (Offset): %.2f%%\n', a_normalized(1));
fprintf('1st Moment (Mean): %.2f%%\n', a_normalized(2));
fprintf('2nd Moment: %.2f%%\n', a_normalized(3));
fprintf('3rd Moment: %.2f%%\n', a_normalized(4));

% Optional: Visualize as a bar graph
figure;
bar(a_normalized, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTickLabel', {'0th (Offset)', '1st (Mean)', '2nd Moment', '3rd Moment'});
ylabel('Relative Contribution (%)');
title('Relative Contributions of Moments to Real Weights');
grid on;

%% 6. Predict Hold-Out Set Spike Counts Using Estimated Contributions
moment0_test = ones(length(test_idx), 1);            
moment1_test = mean(Stimuli_RF_test, 2);         
moment2_test = mean(Stimuli_RF_test.^2, 2);      
moment3_test = mean(Stimuli_RF_test.^3, 2);      

% Combine using estimated contributions
spikeCounts_predicted_test = a_estimated(1) * moment0_test + ...
                             a_estimated(2) * moment1_test + ...
                             a_estimated(3) * moment2_test + ...
                             a_estimated(4) * moment3_test;

%% 7. Evaluate Prediction on Hold-Out Set
residuals_test = spikeCounts_test - spikeCounts_predicted_test;
r2_holdout = 1 - (sum(residuals_test.^2) / sum((spikeCounts_test - mean(spikeCounts_test)).^2));

fprintf('\n--- Hold-Out Set Prediction ---\n');
fprintf('R² on Hold-Out Set: %.4f\n', r2_holdout);

%% 8. Visualize Actual vs Predicted Spike Counts (Hold-Out Set)
figure;
scatter(spikeCounts_test, spikeCounts_predicted_test, 50, 'filled');
hold on;
plot([min(spikeCounts_test) max(spikeCounts_test)], [min(spikeCounts_test) max(spikeCounts_test)], 'r--');
xlabel('Actual Spike Counts (Hold-Out)');
ylabel('Predicted Spike Counts');
title(sprintf('Hold-Out Set Prediction | R² = %.3f', r2_holdout));
grid on;

%% 9. Visualize Spike Count Distributions
figure;
subplot(1,2,1);
histogram(spikeCounts_train); title('Training Set Spike Counts');
xlabel('Spike Count'); ylabel('Frequency');

subplot(1,2,2);
histogram(spikeCounts_test); title('Hold-Out Set Spike Counts');
xlabel('Spike Count'); ylabel('Frequency');
