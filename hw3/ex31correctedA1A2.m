clc;clear;close all;
% PART 0: Data Generation
rng(0); % For reproducibility
N = 500; % Number of samples
m = 50; % Number of neurons in the hidden layer
num_iterations = 10000; % Number of gradient descent iterations
%% Alpha
alpha = 0.001;
%alpha = 0.005; % Learning rate

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

% Numerical solutions
G_Y1 = y_values'; % For E[Y|X]
G_Y2 = min(1, max(-1, y_values')); % For E[G(Y)|X] with G(Y) bounded in [-1, 1]
V1 = F * G_Y1; % Numerical solution for E[Y|X]
V2 = F * G_Y2; % Numerical solution for E[G(Y)|X]

% Activation function (ReLU)
relu = @(z) max(0, z);
relu_derivative = @(z) (z > 0);

% Initialize weights and biases for [A1], [A2], and [C1]
W1_A1 = randn(m, 1) * 0.1; b1_A1 = zeros(m, 1); W2_A1 = randn(1, m) * 0.1; b2_A1 = 0;
W1_A2 = randn(m, 1) * 0.1; b1_A2 = zeros(m, 1); W2_A2 = randn(1, m) * 0.1; b2_A2 = 0;
W1_C1 = randn(m, 1) * 0.1; b1_C1 = zeros(m, 1); W2_C1 = randn(1, m) * 0.1; b2_C1 = 0;

% Loss values
loss_A1 = zeros(1, num_iterations);
loss_A2 = zeros(1, num_iterations);
loss_C1 = zeros(1, num_iterations);

% PART 1: Train [A1]
phi_A1 = @(z) z.^2 / 2;  % Loss function for [A1]
psi_A1 = @(z) -z;

%alpha=0.002;
for iter = 1:num_iterations
    % Forward pass
    Z1 = W1_A1 * X_samples + b1_A1; % Input to hidden layer
    A1 = relu(Z1); % Hidden layer activations
    Z2 = W2_A1 * A1 + b2_A1; % Output layer
    u_X_A1 = Z2; % Final output

    % Compute loss
    loss_A1(iter) = mean(phi_A1(u_X_A1) + Y_samples .* psi_A1(u_X_A1));

    % Backpropagation
    dZ2 = (u_X_A1 - Y_samples); % Gradient of output layer
    dW2 = (dZ2 * A1')/N ; % Gradient w.r.t W2
    db2 = sum(dZ2)/N ; % Gradient w.r.t b2

    dA1 = W2_A1' * dZ2; % Backprop through W2
    dZ1 = dA1 .* relu_derivative(Z1); % Backprop through ReLU
    dW1 = (dZ1 * X_samples')/N ; % Gradient w.r.t W1
    db1 = sum(dZ1, 2)/N ; % Gradient w.r.t b1

    % Update weights and biases
    W1_A1 = W1_A1 - alpha * dW1;
    b1_A1 = b1_A1 - alpha * db1;
    W2_A1 = W2_A1 - alpha * dW2;
    b2_A1 = b2_A1 - alpha * db2;
end

% PART 2: Train [A2]
phi_A2 = @(z) (exp(0.5*abs(z)) - 1) + (1/3).*(exp(-1.5*abs(z)) - 1);
psi_A2 = @(z) 2 * sign(z) .* (exp(-0.5*abs(z)) - 1);

for iter = 1:num_iterations
    % Forward pass
    Z1 = W1_A2 * X_samples + b1_A2; % Input to hidden layer
    A1 = relu(Z1); % Hidden layer activations
    Z2 = W2_A2 * A1 + b2_A2; % Output layer
    %u_X_A2 = Z2; % Final output
    u_X_A2=-exp(-abs(Z2)/2).*(Y_samples-sinh(Z2));
    % Compute loss
    loss_A2(iter) = mean(phi_A2(u_X_A2) + Y_samples .* psi_A2(u_X_A2));
%sign(u_X_A2).*(0.5.*(exp(0.5*abs(u_X_A2))-exp(-1.5*abs(u_X_A2)))-Y_samples.*exp(-0.5.*abs(u_X_A2)));  
    dZ2 = sign(u_X_A2) .* (0.5 .* (exp(0.5 .* abs(u_X_A2)) - exp(-1.5 .* abs(u_X_A2))) ...
       - Y_samples .* exp(-0.5 .* abs(u_X_A2)));

    % Backpropagation
    %dZ2 = (u_X_A2 - Y_samples); % Gradient of output layer
    dW2 = (dZ2 * A1')./N ; % Gradient w.r.t W2
    db2 = sum(dZ2) /N; % Gradient w.r.t b2

    dA1 = W2_A2' * dZ2; % Backprop through W2
    dZ1 = dA1 .* relu_derivative(Z1); % Backprop through ReLU
    dW1 = (dZ1 * X_samples')./N ; % Gradient w.r.t W1
    db1 = sum(dZ1, 2)/N ; % Gradient w.r.t b1

    % Update weights and biases
    W1_A2 = W1_A2 - alpha * dW1;
    b1_A2 = b1_A2 - alpha * db1;
    W2_A2 = W2_A2 - alpha * dW2;
    b2_A2 = b2_A2 - alpha * db2;
end
omega= @(z) z;
omega2= @(z) sinh(z);  
% PART 3: Compare and Plot Results
figure;
subplot(2, 1, 1);
plot(x_values, V1, 'k-', 'LineWidth', 2, 'DisplayName', 'Numeric E[Y|X]');
hold on;
plot(x_values, omega(W2_A1 * relu(W1_A1 * x_values + b1_A1)), 'r--', 'LineWidth', 2, 'DisplayName', '[A1]');
plot(x_values, omega2(W2_A2 * relu(W1_A2 * x_values + b1_A2)), 'b-.', 'LineWidth', 2, 'DisplayName', '[A2]');
legend('show');
xlabel('X');
ylabel('E[Y|X]');
title('Part (α): E[Y|X]');

subplot(2, 1, 2);
plot(1:num_iterations, loss_A1, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Loss [A1]');
hold on;
plot(1:num_iterations, loss_A2, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Loss [A2]');
xlabel('Iterations');
ylabel('Loss');
title('Loss During Training');
legend('show');
grid on;
