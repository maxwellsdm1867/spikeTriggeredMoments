function [bestWeights, bestR2, bestLambda] = manualRidgeRegressionCustom(A, b, binCenters, visualize, lambda)
% manualRidgeRegressionCustom - Ridge or Linear regression with custom visuals
%
% Inputs:
%   A           - Stimulus matrix (N x M), e.g., StimuliHist
%   b           - Spike counts (N x 1)
%   binCenters  - Centers of histogram bins (1 x M), for plotting weights
%   visualize   - Boolean flag (true/false) for 4-panel plot
%   lambda      - (Optional) Set lambda manually. If empty [], automatic tuning.
%
% Outputs:
%   bestWeights - Optimal weights (M x 1)
%   bestR2      - R² value achieved
%   bestLambda  - Lambda value used (0 if linear regression)

    % Identity matrix for regularization
    I = eye(size(A,2));   

    % Check if lambda is provided
    if nargin < 5 || isempty(lambda)
        % Automatic tuning over lambdas
        lambdas = logspace(-4, 2, 50);
        R2_vals = zeros(length(lambdas), 1);  
        weights_all = zeros(size(A,2), length(lambdas));  

        for i = 1:length(lambdas)
            lambda_i = lambdas(i);
            x_temp = (A' * A + lambda_i * I) \ (A' * b);
            predicted_temp = A * x_temp;
            residuals_temp = b - predicted_temp;
            R2_vals(i) = 1 - (sum(residuals_temp.^2) / sum((b - mean(b)).^2));
            weights_all(:, i) = x_temp;
        end

        % Find best lambda
        [bestR2, bestIdx] = max(R2_vals);
        bestLambda = lambdas(bestIdx);
        bestWeights = weights_all(:, bestIdx);
        predicted_best = A * bestWeights;

    else
        % Manual lambda: can be 0 (linear regression) or >0 (ridge)
        bestLambda = lambda;

        if lambda == 0
            % Standard linear regression
            bestWeights = A \ b;
        else
            % Manual ridge regression
            bestWeights = (A' * A + lambda * I) \ (A' * b);
        end

        predicted_best = A * bestWeights;
        residuals_best = b - predicted_best;
        bestR2 = 1 - (sum(residuals_best.^2) / sum((b - mean(b)).^2));
    end

    fprintf('Lambda Used: %.4f | R²: %.4f\n', bestLambda, bestR2);

    %% Visualization (4-panel)
    if visualize
        figure;
        set(gcf, 'Position', [100, 100, 1000, 800]);  % Figure size

        % Subplot 1: Heatmap
        subplot(2,2,1);
        imagesc(binCenters, 1:size(A,1), A);
        colorbar;
        xlabel('Pixel Value (Bin Centers)');
        ylabel('Trial');
        title('Stimulus Histogram Heatmap');
        axis xy;
        axis square;

        % Subplot 2: Actual vs Predicted
        subplot(2,2,2);
        scatter(b, predicted_best, 50, 'filled');
        hold on;
        plot([min(b) max(b)], [min(b) max(b)], 'r--');
        xlabel('Actual Spike Counts');
        ylabel('Predicted Spike Counts');
        title(sprintf('Actual vs. Predicted | R² = %.3f', bestR2));
        grid on;
        axis square;

        % Subplot 3: R² vs Lambda or info
       
        axis square;
        % Subplot 4: Weights (Line with Markers)
        subplot(2,2,3);
        plot(binCenters, bestWeights, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
        xlabel('Pixel Value (Bin Centers)');
        ylabel('Weight');
        title('Best Regression Weights');
        grid on;
        axis square;

         subplot(2,2,4);
        if exist('R2_vals', 'var')
            semilogx(lambdas, R2_vals, 'b-o', 'LineWidth', 1.5);
            hold on;
            xline(bestLambda, 'r--', sprintf('\\lambda = %.4f', bestLambda));
            xlabel('Lambda (log scale)');
            ylabel('R²');
            title('R² vs. Ridge Parameter');
        else
            text(0.1, 0.5, sprintf('Fixed \\lambda = %.4f\nR² = %.3f', bestLambda, bestR2), 'FontSize', 14);
            axis off;
        end
    end
end
