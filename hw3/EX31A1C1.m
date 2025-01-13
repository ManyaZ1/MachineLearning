clc;clear;close all;

% PART 0: Data Generation
rng(0); % For reproducibility

N = 500; % Number of samples
m = 50; % Number of neurons in the hidden layer
num_iterations = 5000; % Number of gradient descent iterations
alpha = 0.01; % Learning rate
alphac=0.01; %different for c1
% Generate data: X ~ N(0,1), W ~ N(0,1), Y = 0.8X + W
X_samples = randn(1, N); % 1xN
W_samples = randn(1, N); % 1xN
Y_samples = 0.8 * X_samples + W_samples; % 1xN

% Define the grid for numerical comparison
x_min = floor(min(X_samples));
x_max = ceil(max(X_samples));
x_values = linspace(x_min, x_max, 500); % Grid for X
y_values = linspace(-5, 5, 500); % Grid for Y
H = @(y, x) normcdf((y - 0.8 * x)); % Conditional CDF H(y|x)

% Compute the F-matrix for the numerical solution
F = zeros(500, 500);
for j = 1:500
    F(j, 1) = 0.5 * (H(y_values(2), x_values(j)) - H(y_values(1), x_values(j)));
    for i = 2:499
        F(j, i) = 0.5 * (H(y_values(i+1), x_values(j)) - H(y_values(i-1), x_values(j)));
    end
    F(j, 500) = 0.5 * (H(y_values(500), x_values(j)) - H(y_values(499), x_values(j)));
end
row_sums = sum(F, 2);
F = F ./ row_sums; % Normalize rows of F

% Numerical solution for bounded expectation
G_Y = min(1, max(-1, y_values')); % G(Y) = min(1, max(-1, Y))
V = F * G_Y; % Numerical solution for E[G(Y)|X]

% Activation function (ReLU)
relu = @(z) max(0, z);
relu_derivative = @(z) (z > 0);

% Initialize weights and biases for [A1] and [C1]
W1_A1 = randn(m, 1) * 0.1; b1_A1 = zeros(m, 1); W2_A1 = randn(1, m) * 0.1; b2_A1 = 0;
W1_C1 = randn(m, 1) * 0.1; b1_C1 = zeros(m, 1); W2_C1 = randn(1, m) * 0.1; b2_C1 = 0;

% Loss values
loss_A1 = zeros(1, num_iterations);
loss_C1 = zeros(1, num_iterations);
g=@(y)min(1, max(-1, y)); 
omega = @(z) z;
%% PART 1: Train [A1]
phi_A1 = @(z) z.^2 / 2;  % Loss function for [A1]
psi_A1 = @(z) -z;

for iter = 1:num_iterations
    % Forward pass
    Z1 = W1_A1 * X_samples + b1_A1; % Input to hidden layer
    A1 = relu(Z1); % Hidden layer activations
    Z2 = W2_A1 * A1 + b2_A1; % Output layer
    u_X_A1 = Z2; % Final output

    % Compute loss
    loss_A1(iter) = mean(phi_A1(u_X_A1) + g(Y_samples) .* psi_A1(u_X_A1));

    % Backpropagation
    dZ2 = (u_X_A1 - Y_samples); % Gradient of output layer
    dW2 = (dZ2 * A1') / N; % Gradient w.r.t W2
    db2 = sum(dZ2) / N; % Gradient w.r.t b2

    dA1 = W2_A1' * dZ2; % Backprop through W2
    dZ1 = dA1 .* relu_derivative(Z1); % Backprop through ReLU
    dW1 = (dZ1 * X_samples') / N; % Gradient w.r.t W1
    db1 = sum(dZ1, 2) / N; % Gradient w.r.t b1

    % Update weights and biases
    W1_A1 = W1_A1 - alpha * dW1;
    b1_A1 = b1_A1 - alpha * db1;
    W2_A1 = W2_A1 - alpha * dW2;
    b2_A1 = b2_A1 - alpha * db2;
end

%% PART 2: Train [C1]
phi_C1 = @(z)  2./(1+exp(z))+log(1+exp(z)); %@(z) (z - 1).^2 .* (z > 1) + (z + 1).^2 .* (z < -1); % Penalize outside [-1, 1]
psi_C1 =  @(z) -log(1+exp(z));  %@(z) 2 * max(-1, min(1, z)); % Ensure bounded range
omega2 = @(z)(-1./ (1 + exp(z)))+ (exp(z) ./ (1 + exp(z))); 
W1_C1 = randn(m, 1) * 0.001; % Smaller initial weights
b1_C1 = zeros(m, 1);
W2_C1 = randn(1, m) * 0.001;
b2_C1 = 0;

for iter = 1:num_iterations
    % Forward pass
    Z1 = W1_C1 * X_samples + b1_C1; % Input to hidden layer
    A1 = relu(Z1); % Hidden layer activations
    Z2 = W2_C1 * A1 + b2_C1; % Output layer
    %u_X_C1 = Z2; % Final output
    %[a,b]=[-1,1]
    u_X_C1=-(1./(1+exp(-Z2'))).*(Y_samples-1./(1+exp(-Z2'))-(-1)./(1+exp(Z2')));
    
    % Compute loss
    %loss_C1(iter) = mean(phi_C1(u_X_C1) + g(Y_samples) .* psi_C1(u_X_C1));
    %loss_C1(iter) = mean(phi_C1(u_X_C1(:)) + g(Y_samples(:)) .* psi_C1(u_X_C1(:)));
    loss_C1(iter)=mean((2)./(1+exp(Z2'))+1*log(1+exp(Z2'))-Y_samples'.*log(1+exp(Z2')));
    % Backpropagation
    %dZ2 = (u_X_C1 - Y_samples); % Gradient of output layer
    dZ2 = -(2 * exp(Z2) ./ (1 + exp(Z2)).^2) + (exp(Z2) ./ (1 + exp(Z2))) - Y_samples .* (exp(Z2) ./ (1 + exp(Z2)));

    dW2 = (dZ2 * A1') / N; % Gradient w.r.t W2
    db2 = sum(dZ2) / N; % Gradient w.r.t b2

    dA1 = W2_C1' * dZ2; % Backprop through W2
    dZ1 = dA1 .* relu_derivative(Z1); % Backprop through ReLU
    dW1 = (dZ1 * X_samples') / N; % Gradient w.r.t W1
    db1 = sum(dZ1, 2) / N; % Gradient w.r.t b1

    % Update weights and biases
    W1_C1 = W1_C1 - alphac * dW1;
    b1_C1 = b1_C1 - alphac * db1;
    W2_C1 = W2_C1 - alphac * dW2;
    b2_C1 = b2_C1 - alphac * db2;
end

% PART 3: Compare and Plot Results
figure;
subplot(2, 1, 1);
plot(x_values, V, 'k-', 'LineWidth', 2, 'DisplayName', 'Numeric E[G(Y)|X]');
hold on;
plot(x_values, omega(W2_A1 * relu(W1_A1 * x_values + b1_A1)), 'r--', 'LineWidth', 2, 'DisplayName', '[A1]');
plot(x_values, omega2(W2_C1 * relu(W1_C1 * x_values + b1_C1)), 'b-.', 'LineWidth', 2, 'DisplayName', '[C1]');
legend('show');
xlabel('X');
ylabel('E[G(Y)|X]');
title('E[G(Y)|X], G(Y) = min(1, max(-1, Y))');

subplot(2, 1, 2);
plot(1:num_iterations, loss_A1, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Loss [A1]');
hold on;
plot(1:num_iterations, loss_C1, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Loss [C1]');
xlabel('Iterations');
ylabel('Loss');
title('Loss During Training');
legend('show');
grid on;
