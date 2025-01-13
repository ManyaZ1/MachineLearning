clc; 
close all; 
clear all; 
% Load data 
load('data21.mat'); % Generative model (A1, A2, B1, B2) 
load('data23.mat'); % Downsampled and noisy data (Xn, Xi) 
 
% Extract matrices and parameters 
A1 = A_1; % 128 x 10 
A2 = A_2; % 784 x 128 
B1 = B_1; % 128 x 1 
B2 = B_2; % 784 x 1 
Xi = X_i; % Ideal high-resolution images (784 x 4) 
Xn = X_n; % Downsampled noisy images (49 x 4) 
 
% Define T (49 x 784 matrix for downsampling) 
T = zeros(49, 784); 
for i = 1:7 
    for j = 1:7 
        row_idx = (i-1)*7 + j; 
        col_start = (i-1)*4*28 + (j-1)*4 + 1; 
        for x = 0:3 
            for y = 0:3 
                T(row_idx, col_start + x*28 + y) = 1/16; 
            end 
        end 
    end 
end 
 
% Parameters 
max_iterations = 150000; % Maximum iterations for gradient descent 
learning_rate = 0.0003; % Learning rate 
lambda = 0.0007; % Regularization parameter 
 
% Initialize storage for loss histories 
loss_histories = zeros(max_iterations, size(Xn, 2)); % Store losses for 
each image 
 
% Process each column of Xn 
for col = 1:size(Xn, 2) 
    fprintf('Processing image %d...\n', col); 
     
    % Extract noisy input 
    Xn_col = Xn(:, col); 
     
    % Initialize Z (latent variable) 
    Z = randn(10, 1); 
     
    % Gradient Descent 
    for iter = 1:max_iterations 
        % Forward pass 
        W1 = A1 * Z + B1; % Linear transformation 
        Z1 = max(W1, 0); % ReLU activation 
        W2 = A2 * Z1 + B2; % Linear transformation 
        X = 1 ./ (1 + exp(W2)); % Sigmoid activation 
         
        % Compute loss 
        residual = T * X - Xn_col; % Residual vector 
        norm_squared = norm(residual)^2; % Squared norm of the residual 
        loss = log(norm_squared) + lambda * norm(Z)^2; % Loss function 
        loss_histories(iter, col) = loss; % Store the loss 
         
        % Backpropagation 
        gradient_loss = (2 / norm_squared) * (T' * residual); % Gradient of 
the loss 
        f2_derivative = -(exp(W2)) ./ ((1 + exp(W2)).^2); % Derivative of 
sigmoid 
        v2 = gradient_loss .* f2_derivative; % Backprop through sigmoid 
        u1 = A2' * v2; % Backprop through A2 
        v1 = u1 .* (W1 > 0); % Backprop through ReLU 
        grad = A1' * v1 + 2 * lambda * Z; % Total gradient 
         
        % Update Z 
        Z = Z - learning_rate * grad; 
    end 
     
    % Reconstruct high-resolution image 
    W1 = A1 * Z + B1; 
    Z1 = max(W1, 0); 
    W2 = A2 * Z1 + B2; 
    X_reconstructed = 1 ./ (1 + exp(W2)); % Reconstructed image 
        subplot(size(Xn, 2), 3, (col - 1) * 3 + 1); % Ideal Image 
    imshow(reshape(Xi(:, col), [28, 28]), []); 
    title(['Ideal Image ', num2str(col)]); 
     
    subplot(size(Xn, 2), 3, (col - 1) * 3 + 2); % Noisy Low-Resolution 
    imshow(reshape(T' * Xn_col, [28, 28]), []); 
    title(['Noisy (Col ', num2str(col), ')']); 
     
    subplot(size(Xn, 2), 3, (col - 1) * 3 + 3); % Reconstructed Image 
    imshow(reshape(X_reconstructed, [28, 28]), []); 
    title(['Reconstructed (Col ', num2str(col), ')']); 
end 
 
% Plot all loss histories on the same graph 
figure('Name', 'Loss Histories', 'NumberTitle', 'off'); 
hold on; 
for col = 1:size(Xn, 2) 
    plot(1:max_iterations, loss_histories(:, col), 'LineWidth', 1.5); 
end 
xlabel('Iterations'); 
ylabel('Loss'); 
legend(arrayfun(@(x) ['Image ', num2str(x)], 1:size(Xn, 2), 
'UniformOutput', false)); 
title('Loss Histories for All Images'); 
grid on; 
hold off; 