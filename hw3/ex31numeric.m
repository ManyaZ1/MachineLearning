clc;clear;close all;
%r = normrnd(mu,sigma,sz1,...,szN) generates an array of normal random numbers, where sz1,...,szN indicates the size of each dimension.
n_samples=500;
W = normrnd(0,1,[1,n_samples]);  % Noise ~ N(0,1)
X = linspace(-4, 4, n_samples); %RANDOM X
Y=0.8.*X + W; % Y based on the model Y = 0.8*X + W
% EY[Y|X = X]

% Define grids for X and Y
x_values = linspace(-4, 4, n_samples); % Grid points for X
y_values = linspace(-6, 6, n_samples); % Grid points for Y

% Define conditional CDF H(y|x) (Lecture 11 Equation)
H = @(y, x) normcdf((y - 0.8 * x), 0, 1); % Conditional CDF of Y given X

% Initialize F matrix
F = zeros(n_samples, n_samples);

% Compute F matrix (Trapezoidal Rule Approximation)
for j = 1:n_samples
    % First column
    F(j, 1) = 0.5 * (H(y_values(2), x_values(j)) - H(y_values(1), x_values(j)));
    % Intermediate columns
    for i = 2:n_samples-1
        F(j, i) = 0.5 * (H(y_values(i+1), x_values(j)) - H(y_values(i-1), x_values(j)));
    end
    % Last column
    F(j, n_samples) = 0.5 * (H(y_values(n_samples), x_values(j)) - H(y_values(n_samples-1), x_values(j)));
end

% Normalize F (Row-wise normalization for probabilities)
row_sums = sum(F, 2);
F = F ./ row_sums;

% Define G(Y) functions for expectations
G1_Y = y_values'; % For E[Y|X]
G2_Y = min(1, max(-1, y_values')); % For E[min{1, max{-1, Y}}|X]

% Compute conditional expectations
V1_numeric = F * G1_Y; % Numerical approximation for E[Y|X]
V2_numeric = F * G2_Y; % Numerical approximation for E[min{1, max{-1, Y}}|X]
% Plot results
figure;
% Plot E[Y|X]
subplot(1, 2, 1);
plot(x_values, V1_numeric, 'k', 'LineWidth', 2);
title('Part (a): E[Y|X]');
xlabel('X');
ylabel('E[Y|X]');
legend('Numerical', 'Location', 'Best');
% Plot E[min{1, max{-1, Y}}|X]
subplot(1, 2, 2);
plot(x_values, V2_numeric, 'k', 'LineWidth', 2);
title('Part (b): E[min{1, max{-1, Y}}|X]');
xlabel('X');
ylabel('E[min{1, max{-1, Y}}|X]');
legend('Numerical', 'Location', 'Best');
disp('Numerical estimates computed for E[Y|X] and E[min{1, max{-1, Y}}|X].');
