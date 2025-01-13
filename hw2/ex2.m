clc; close all; clear all; 
% Load data 
load('data21.mat'); % Generative model (A1, A2, B1, B2) 
load('data22.mat'); % Matrices Xi and Xn 
A1 = A_1; % 128 x 10 
A2 = A_2; % 784 x 128 
B1 = B_1; % 128 x 1 
B2 = B_2; % 784 x 1 
Xi = X_i; % Ideal images (784 x 4) 
Xn = X_n; % Noisy images (784 x 4) 
 
% Parameters 
N_values = [500, 400, 350, 300, 250]; % Different values of N 
max_iterations = 20000; % Maximum number of iterations for gradient descent 
learning_rate = 0.001; % Learning rate for gradient descent 
lambda = 0.001; % Regularization parameter 
 
% Initialize storage for loss histories 
loss_histories = zeros(max_iterations, length(N_values), size(Xn, 2)); 
 
% Process each column of Xn 
for col = 1:size(Xn, 2) 
    fprintf('Processing image %d\n', col); 
     
    % Extract the observed noisy column of Xn 
    Xn_col = Xn(:, col); 
     
    % Create a figure to display images 
    figure('Name', ['Reconstruction Results for Image ', num2str(col)], 
'NumberTitle', 'off');   
    % Loop over different values of N 
    for idx = 1:length(N_values) 
        N = N_values(idx); 
        T = [eye(N), zeros(N, 784 - N)]; % Downsampling matrix 
        fprintf('  Using N = %d\n', N); 
        % Extract the noisy partial vector (first N values) 
        Xn_partial = Xn_col(1:N); 
         
        % Initialize Z (latent variable) 
        Z = randn(10, 1); 
        loss_history = zeros(max_iterations, 1); 
         
        % Gradient Descent to optimize Z 
        for iter = 1:max_iterations 
            % Forward pass through the generative model 
            W1 = A1 * Z + B1; 
            Z1 = max(W1, 0); % ReLU activation 
            W2 = A2 * Z1 + B2; 
            X = 1 ./ (1 + exp(W2)); % Sigmoid activation 
             
            % Compute loss 
            residual = T * X - Xn_partial; % Residual vector 
            norm_squared = norm(residual)^2; % Squared norm of the residual 
            loss = N * log(norm_squared) + lambda * norm(Z)^2; % Loss func 
            loss_history(iter) = loss; % Store the loss 
             
            % Backpropagation 
            gradient_loss = (2  / norm_squared) * (T' * residual); % ∇_x φ  
            f2_derivative = -(exp(W2)) ./ ((1 + exp(W2)).^2); %f’2 
            v2 = gradient_loss .* f2_derivative; % Backprop through sigmoid 
            u1 = A2' * v2; % Backprop through A2 
            v1 = u1 .* (W1 > 0); % Backprop through ReLU 
            grad = N *  A1' * v1 + 2 * lambda * Z; % Total gradient 
             
            % Update Z 
            Z = Z - learning_rate * grad; 
        end 
         
        % Save the loss history for this (N, col) combination 
        loss_histories(:, idx, col) = loss_history; 
         
        % Reconstruct high-resolution image 
        W1 = A1 * Z + B1; 
        Z1 = max(W1, 0); % ReLU activation 
        W2 = A2 * Z1 + B2; 
        X_reconstructed = 1 ./ (1 + exp(W2)); % Reconstructed image 
        X_reconstructed_image = reshape(X_reconstructed, [28, 28]); 
 
        % Plot the reconstructed, noisy, and ideal images 
        subplot(3, length(N_values), idx); % Row 1: Reconstructed Images 
        imshow(X_reconstructed_image, []); 
        title(['Rec(N = ', num2str(N), ')']); 
         
        subplot(3, length(N_values), idx + length(N_values)); % Row 2: 
Noisy Images 
        partial_image = Xn_col; 
        partial_image(N+1:end) = 0; % Mask the missing part 
        imshow(reshape(partial_image, [28, 28]), []); 
        title(['Noisy (N = ', num2str(N), ')']); 
         
        subplot(3, length(N_values), idx + 2*length(N_values)); % Row 3: 
Ideal Images 
        imshow(reshape(Xi(:, col), [28, 28]), []); 
        title('Ideal Image'); 
    end 
end 
% Plot smoothed loss histories for visualization 
smoothing_window = 100; % Adjust the window size for smoothing 
for col = 1:size(Xn, 2) 
    figure('Name', ['Smoothed Loss Histories for Image ', num2str(col)], 
'NumberTitle', 'off'); 
    for idx = 1:length(N_values) 
        % Smooth the loss history 
        smoothed_loss = smooth_loss(squeeze(loss_histories(:, idx, col)), 
smoothing_window); 
         
        % Plot the smoothed loss 
        subplot(ceil(length(N_values) / 3), 3, idx); % Adjust grid 
dynamically 
        plot(1:max_iterations, smoothed_loss, 'LineWidth', 1.5); 
        xlabel('Iterations'); 
        ylabel('Smoothed Loss'); 
        title(['N = ', num2str(N_values(idx))]); 
        grid on; 
    end 
end 
