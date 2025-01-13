clc; close all;
rng(0);
%% Helper Definitions
reward = @(y) min(2, y.^2);

% Transition CDFs for actions
H1 = @(st1, st) normcdf(st1, 0.8*st + 1, 1);
H2 = @(st1, st) normcdf(st1, -2, 1);

gamma = 0.8;  % Discount factor
tol = 1e-6;   % Convergence tolerance
maxIter = 1000;  % Max number of iterations

%% (1) Discretize State Space
num_states = 500;
state_range = linspace(-6, 6, num_states);  % Discrete states S_t
R = reward(state_range)';  % Reward for each state (column vector)

%% (2) Build Transition Matrices F1, F2
F1 = zeros(num_states, num_states);
F2 = zeros(num_states, num_states);
mw = 0; % Mean of noise
sigma_w = 1; % Standard deviation of noise

for j = 1:num_states
    S_t = state_range(j);
    for i = 1:num_states
        S_t1 = state_range(i);
        if i == 1
            F1(j, i) = 0.5 * (normcdf(S_t1, 0.8 * S_t + 1 + 0, 1) - normcdf(S_t1 - (20 / num_states), 0.8 * S_t + 1 + 0, 1));
            F2(j, i) = 0.5 * (normcdf(S_t1,  -2 + 0, 1) - normcdf(S_t1 - (20 / num_states),   -2 , 1));
        elseif i == 2
            F1(j, i) = 0.5 * (normcdf(S_t1, 0.8 * S_t + 1 + 0, 1)  - normcdf(state_range(i-1) - (20/num_states), 0.8 * S_t + 1 , 1));
            F2(j, i) = 0.5 * (normcdf(S_t1, -2 + 0, 1)  - normcdf(state_range(i-1) - (20/num_states), + -2 , 1));
        elseif i == num_states
            F1(j, i) = 0.5 * (normcdf(S_t1, 0.8 * S_t + 1 , 1)  - normcdf(state_range(i-1), 0.8 * S_t + 1 , 1));
            F2(j, i) = 0.5 * (normcdf(S_t1,  -2 , 1)    - normcdf(state_range(i-1),   -2 , 1));
        else
            F1(j, i) = 0.5 * (normcdf(S_t1, 0.8 * S_t + 1 , 1) - normcdf(state_range(i-2), 0.8 * S_t + 1 + 0, 1));
            F2(j, i) = 0.5 * (normcdf(S_t1,   -2 , 1) ...
                - normcdf(state_range(i-2),   -2 , 1));
        end
    end
end

%% (3) Value Iteration
V1 = zeros(num_states, 1);  % Value function for action 1
V2 = zeros(num_states, 1);  % Value function for action 2

for iter = 1:maxIter
    % Compute updated value functions
    V1_new = F1 * (R + gamma * max(V1, V2));
    V2_new = F2 * (R + gamma * max(V1, V2));

    % Check convergence
    if max(abs(V1_new - V1)) < tol && max(abs(V2_new - V2)) < tol
        fprintf('Converged in %d iterations.\n', iter);
        break;
    end

    % Update value functions
    V1 = V1_new;
    V2 = V2_new;
end

%% nn
% Define reward function
reward = @(y) min(2, y.^2);

% Parameters
N = 2000;
action = randi([1, 2], 1, N);
set1 = [];
set2 = [];
state = randn;

% Generate state transitions
for i = 1:N
    W = randn;
    if action(i) == 1
        next_state = 0.8 * state + 1 + W;
        set1 = [set1; state, next_state];
    else
        next_state = -2 + W;
        set2 = [set2; state, next_state];
    end
    state = next_state;
end

% Prepare data
X1_data = set1(:, 1)';
Y1_data = set1(:, 2)';
X2_data = set2(:, 1)';
Y2_data = set2(:, 2)';

% Ensure equal lengths of Y1_data and Y2_data by padding
len_diff = abs(length(Y1_data) - length(Y2_data));
if length(Y1_data) < length(Y2_data)
    Y1_data = [Y1_data, nan(1, len_diff)];
elseif length(Y2_data) < length(Y1_data)
    Y2_data = [Y2_data, nan(1, len_diff)];
end

% Training parameters
num_iter = 10000;
lr = 0.0002;  % Reduced learning rate
m = 100;
gamma = 0.8;
loss1 = zeros(1, num_iter);
loss2 = zeros(1, num_iter);

% Neural network initialization for Network 1
w1 = 2*randn(m, 1) * sqrt(2 / m); % He initialization
b1 = zeros(m, 1);
w2 = 2/m*randn(1, m) * sqrt(2 / m);
b2 = 0;

% Neural network initialization for Network 2
w3 = 2/m*randn(m, 1) * sqrt(2 / m); % He initialization
b3 = zeros(m, 1);
w4 = 2/m*randn(1, m) * sqrt(2 / m);
b4 = 0;

% Activation and other functions
ReLU = @(z) max(0, z);
r = @(y) min(2, y.^2);
phi = @(z) 0.5 * z.^2;
psi = @(z) -z;

% Gradient clipping threshold
grad_clip = 5;

% Training loop
for iter = 1:num_iter
    % Forward pass for Network 1
    Z1 = w1 * X1_data + b1;
    A1 = ReLU(Z1);
    Z2 = w2 * A1 + b2; % Predicted output
    output1 = Z2;

    % Target calculation for Network 1
    z11 = w1 * Y1_data + b1;
    a11 = ReLU(z11);
    om11 = w2 * a11 + b2;

    z12 = w1 * Y2_data + b1;
    a12 = ReLU(z12);
    om12 = w2 * a12 + b2;

    valid_idx1 = ~isnan(Y1_data) & ~isnan(Y2_data);
    YY1 = r(Y1_data(valid_idx1)) + gamma * max(om11(valid_idx1), om12(valid_idx1));

    % Loss calculation for Network 1
    Z2_valid1 = Z2(valid_idx1);
    c1 = phi(Z2_valid1) + YY1 .* psi(Z2_valid1);
    loss1(iter) = mean(c1);

    % Backpropagation for Network 1
    dZ2_1 = Z2_valid1 - YY1;
    dW2_1 = dZ2_1 * A1(:, valid_idx1)';
    db2_1 = sum(dZ2_1);
    dA1_1 = w2' * dZ2_1;
    dZ1_1 = dA1_1 .* (Z1(:, valid_idx1) > 0);
    dW1_1 = dZ1_1 * X1_data(valid_idx1)';
    db1_1 = sum(dZ1_1, 2);

    % Gradient clipping for Network 1
    dW2_1 = max(min(dW2_1, grad_clip), -grad_clip);
    db2_1 = max(min(db2_1, grad_clip), -grad_clip);
    dW1_1 = max(min(dW1_1, grad_clip), -grad_clip);
    db1_1 = max(min(db1_1, grad_clip), -grad_clip);

    % Update Network 1
    w2 = w2 - lr * dW2_1;
    b2 = b2 - lr * db2_1;
    w1 = w1 - lr * dW1_1;
    b1 = b1 - lr * db1_1;

    % Forward pass for Network 2
    Z3 = w3 * X2_data + b3;
    A3 = ReLU(Z3);
    Z4 = w4 * A3 + b4; % Predicted output
    output2 = Z4;

    % Target calculation for Network 2
    z21 = w3 * Y2_data + b3;
    a21 = ReLU(z21);
    om21 = w4 * a21 + b4;

    z22 = w3 * Y1_data + b3;
    a22 = ReLU(z22);
    om22 = w4 * a22 + b4;

    valid_idx2 = ~isnan(Y2_data) & ~isnan(Y1_data);
    YY2 = r(Y2_data(valid_idx2)) + gamma * max(om21(valid_idx2), om22(valid_idx2));

    % Loss calculation for Network 2
    Z4_valid2 = Z4(valid_idx2);
    c2 = phi(Z4_valid2) + YY2 .* psi(Z4_valid2);
    loss2(iter) = mean(c2);

    % Backpropagation for Network 2
    dZ4_2 = Z4_valid2 - YY2;
    dW4_2 = dZ4_2 * A3(:, valid_idx2)';
    db4_2 = sum(dZ4_2);
    dA3_2 = w4' * dZ4_2;
    dZ3_2 = dA3_2 .* (Z3(:, valid_idx2) > 0);
    dW3_2 = dZ3_2 * X2_data(valid_idx2)';
    db3_2 = sum(dZ3_2, 2);

    % Gradient clipping for Network 2
    dW4_2 = max(min(dW4_2, grad_clip), -grad_clip);
    db4_2 = max(min(db4_2, grad_clip), -grad_clip);
    dW3_2 = max(min(dW3_2, grad_clip), -grad_clip);
    db3_2 = max(min(db3_2, grad_clip), -grad_clip);

    % Update Network 2
    w4 = w4 - lr * dW4_2;
    b4 = b4 - lr * db4_2;
    w3 = w3 - lr * dW3_2;
    b3 = b3 - lr * db3_2;
end

x=linspace(-6,6,1000);
Z1 = w1 * x + b1;
A1 = ReLU(Z1);
Z2 = w2 * A1 + b2; % Predicted output
outputxx1 = Z2;

Z3 = w3 * x + b3;
A3 = ReLU(Z3);
Z4 = w4 * A3 + b4; % Predicted output
outputxx2 = Z4;



% Numerical decision policy
figure; hold on;
policy = ones(num_states, 1); % Default to action 1
policy(V2 > V1) = 2; % Switch to action 2 where V2 > V1
%scatter(state_range, policy, 50, 'b', 'filled'); % Plot numerical policy
plot(state_range, policy, 'k-', 'LineWidth', 2, 'DisplayName', 'Optimal Policy numerical');
% NN-based decision policy
policyA1 = ones(length(x), 1); % Default to action 1
policyA1(outputxx2 > outputxx1) = 2; % Switch to action 2 where outputxx2 > outputxx1
%scatter(x, policyA1, 20, 'r', 'filled'); % Plot NN policy
plot(x, policyA1, 'r-', 'LineWidth', 2, 'DisplayName', 'Optimal Policy [A1]');
% Plot value functions (numerical)
plot(state_range, V1, 'k-', 'LineWidth', 1.5, 'DisplayName', 'V1 (Numerical)');
plot(state_range, V2, 'c-', 'LineWidth', 1.5, 'DisplayName', 'V2 (Numerical)');

% Plot neural network outputs
plot(x, outputxx1, 'r--', 'LineWidth', 2, 'DisplayName', 'NN V1');
plot(x, outputxx2, 'b--', 'LineWidth', 2, 'DisplayName', 'NN V2');

% Formatting
title('Decision Policy and Value Functions');
xlabel('State');
ylabel('Value / Policy');
legend('show');
grid on;
xlim([-5, 5]);
ylim([-6, 13]);
yticks([1, 2]);

% Plot 1: Output after training
figure; hold on;
plot(state_range, V1, 'k-', 'LineWidth', 1.5); hold on; grid on;
plot(state_range, V2, 'c-', 'LineWidth', 1.5); hold on;
plot(x, outputxx1, 'r--', 'LineWidth', 2);
plot(x, outputxx2, 'b--', 'LineWidth', 2);
title('Output After Training');
xlim([-5,5]);
ylim([7,12]);
xlabel('X');
ylabel('Output');

% Plot 2: Loss during training
figure;
plot(1:num_iter, loss1, 'b-', 'LineWidth', 1.5); hold on;
plot(1:num_iter, loss2, 'r-', 'LineWidth', 1.5);
title('Loss During Training');
xlabel('Iteration');
ylabel('Loss');
legend('Network 1', 'Network 2');
