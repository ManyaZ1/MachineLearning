clc;clear;close all;
clc;clear;close all;
rng(0);
%% SAMPLES for understanding
% N = 1000;
% % Generate 1000 samples, each either 1 or 2 with 50/50 probability
% action = randi([1, 2], 1, N); 
% % 2 sets
% set1 =[];
% set2=[];
% % Initial state 
% state = randn;  % Random initial state from a standard normal distribution
% % DECISIONS 
% for i = 1:N
%     W = randn; %noise
%     % Decision logic based on the action
%     if action(i) == 1
%         next_state = 0.8 * state + 1 + W;
%         set1=[set1;state,next_state];
%     elseif action(i) == 2
%         next_state = -2 + W;
%         set2=[set2;state,next_state];
%     else
%         error('Invalid input: a must be either 1 or 2.');
%     end
%     state = next_state;
% end

%% Helpers
reward = @(y) min(2, y.^2);
% normcdf(x,mu,sigma) returns the cdf of the normal distribution with mean mu and standard deviation sigma, evaluated at the values in x.
H1 = @(st1, st) normcdf(st1, 0.8 * st + 1, 1); % H(Y | X) = Φ((Y - (0.8S_t + 1)) / 1)
H2 = @(st1, st) normcdf(st1, -2, 1); % H(Y | X) = Φ((Y + 2) / 1)

%% NUMERICAL U = F * G
% Define state space
n = 3000;  % Number of discretized states
X = linspace(-4, 4, n);  % State space for S_t (X GRID)
Y = linspace(-8.2, 8.2, n);  % State space for S_{t+1} (Y GRID)

% Define reward vector R(S) = min(2, S^2)
R = reward(Y)';

% Transition matrices F1 and F2
F1 = zeros(n, n);  % For action = 1
F2 = zeros(n, n);  % For action = 2

% Compute F1 for action = 1
for j = 1:n
    F1(j, 1) = 0.5 * (H1(Y(2), X(j)) - H1(Y(1), X(j)));
    F1(j, 3:(n-1)) = 0.5 * (H1(Y(3:(n-1)), X(j)) - H1(Y(1:(n-3)), X(j)));
    F1(j, n) = 0.5 * (H1(Y(n), X(j)) - H1(Y(n-1), X(j)));
end

% Compute F2 for action = 2
for j = 1:n
    F2(j, 1) = 0.5 * (H2(Y(2), X(j)) - H2(Y(1), X(j)));
    F2(j, 3:(n-1)) = 0.5 * (H2(Y(3:(n-1)), X(j)) - H2(Y(1:(n-3)), X(j)));
    F2(j, n) = 0.5 * (H2(Y(n), X(j)) - H2(Y(n-1), X(j)));
end

% Compute rewards for each action
V1 = F1 * R;  % Reward for action = 1
V2 = F2 * R;  % Reward for action = 2
% Compute Optimal Policy
optimal_policy = zeros(1, length(X)); % Initialize optimal policy array
for i = 1:length(X)
    if V1(i) >= V2(i)
        optimal_policy(i) = 1; % Action = 1
    else
        optimal_policy(i) = 2; % Action = 2
    end
end

% Plot the results
figure;
hold on; grid on;
plot(X, V1, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Action = 1');
plot(X, V2, 'c-', 'LineWidth', 1.5, 'DisplayName', 'Action = 2');
xlabel('State S'); ylabel('Reward E[R(S_{t+1}) | S_t]');
legend('Location', 'best');
title('Reward Comparison for Action = 1 and Action = 2');

% Plot Optimal Policy
figure;
hold on; grid on;
plot(X, V1, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Action = 1');
plot(X, V2, 'c-', 'LineWidth', 1.5, 'DisplayName', 'Action = 2');
plot(X, optimal_policy, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Optimal Policy');
xlabel('State S'); 
%legend('Location', 'best');
%title('Reward Comparison for Action = 1 and Action = 2');

ylim([0.5 2.5]);
yticks([1 2]);
ylabel('Optimal Action (\alpha^*)');
title('Optimal Decision Policy Numerical');
legend('Location', 'best');
grid on;

%% DATA DRIVEN 
%% Data gen
N = 1000;
% Generate 1000 samples, each either 1 or 2 with 50/50 probability
action = randi([1, 2], 1, N); 
% 2 sets
set1 =[];
set2=[];
% Initial state 
state = randn;  % Random initial state from a standard normal distribution
% DECISIONS 
for i = 1:N
    W = randn; %noise
    % Decision logic based on the action
    if action(i) == 1
        next_state = 0.8 * state + 1 + W;
        set1=[set1;state,next_state];
    elseif action(i) == 2
        next_state = -2 + W;
        set2=[set2;state,next_state];
    else
        error('Invalid input: a must be either 1 or 2.');
    end
    state = next_state;
end

X1_data= set1(:,1)';
Y1_data= set1(:,2)';
R1_data= reward(Y1_data);

X2_data= set2(:,1)';
Y2_data= set2(:,2)';
R2_data= reward(Y2_data);

%% Neural Network
%parameters
m=100;
num_iter= 8000;
alpha= 0.001;%learning rate
%[A1]
phi_A1 = @(z) z.^2 / 2;  % Loss function for [A1]
psi_A1 = @(z) -z;
omega1= @(z) z;
%[C1]
phi_C1 = @(z)  2./(1+exp(z))+log(1+exp(z)); %@(z) (z - 1).^2 .* (z > 1) + (z + 1).^2 .* (z < -1); % Penalize outside [-1, 1]
psi_C1 =  @(z) -log(1+exp(z));  %@(z) 2 * max(-1, min(1, z)); % Ensure bounded range
omega_C1 = @(z)(-1./ (1 + exp(z)))+ (exp(z) ./ (1 + exp(z))); 
% Activation function (ReLU)
relu = @(z) max(0, z);
relu_derivative = @(z) (z > 0);

% Initialize weights and biases for [A1]v1,v2,and [C1]v2,v2
cost_v1A1= zeros(1,num_iter);
cost_v1C1= zeros(1,num_iter);
cost_v2A1= zeros(1,num_iter);
cost_v2C1= zeros(1,num_iter);

% v1(A1)
w1_v1A1= 0.01*randn(m,1);
b1_v1A1= zeros(m,1);
w2_v1A1= randn(1,m);
b2_v1A1= 0;

% v1(C1)
w1_v1C1= randn(m,1);
b1_v1C1= zeros(m,1);
w2_v1C1= randn(1,m);
b2_v1C1= 0;

% v2(A1)
w1_v2A1= 0.01*randn(m,1);
b1_v2A1= zeros(m,1);
w2_v2A1= randn(1,m);
b2_v2A1= 0;

% v2(C1)
w1_v2C1= randn(m,1);
b1_v2C1= zeros(m,1);
w2_v2C1= randn(1,m);
b2_v2C1= 0;

% Loss values
loss_v1A1 = zeros(1, num_iter);
loss_v2A1 = zeros(1, num_iter);
loss_v1C1 = zeros(1, num_iter);
loss_v2C1 = zeros(1, num_iter);


[~, ~, ~, ~, loss_v1A1,outputA1V1] = A1_training(X1_data, R1_data, w1_v1A1, b1_v1A1, w2_v1A1, b2_v1A1, alpha, num_iter, relu,loss_v1A1);
[X1_sort, idx1] = sort(X1_data);
yA1_1=outputA1V1(idx1);
[~, ~, ~, ~, loss_v1C1,outputC1V1]= C1_training(X1_data, R1_data, w1_v1A1, b1_v1A1, w2_v1A1, b2_v1A1, alpha, num_iter, relu,loss_v1C1);
yC1_1=outputC1V1(idx1);

[~, ~, ~, ~, loss_v2A1,outputA1V2] = A1_training(X2_data, R2_data, w1_v2A1, b1_v2A1, w2_v2A1, b2_v2A1, alpha, num_iter, relu,loss_v2A1);
[X2_sort, idx2] = sort(X2_data);
yA1_2=outputA1V2(idx2);
[w1, b1, w2, b2, loss_v2C1,outputC1V2] = C1_training(X2_data, R2_data, w1_v2C1, b1_v2C1, w2_v2C1, b2_v2C1, alpha, num_iter, relu,loss_v2C1);
yC1_2=outputC1V2(idx2);

figure;
plot(X1_sort, yA1_1, 'r--','LineWidth',2,'DisplayName','[A1]');
hold on
plot(X1_sort, yC1_1, 'b--','LineWidth',2,'DisplayName','[C1]');
hold on
plot(X, V1, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Action = 1');
legend('Location', 'Best');
xlim([-4,4]);
title("v1 action1 reward");
figure;
plot(X2_sort, yA1_2, 'm--','LineWidth',2,'DisplayName','[A1]');
hold on
plot(X2_sort, yC1_2, 'g--','LineWidth',2,'DisplayName','[C1]');
hold on
plot(X, V2, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Action = 2');
legend('Location', 'Best');
xlim([-4,4]);
ylim([0 4]);
title("v2 action2 reward");
% Plot all losses
figure;
plot(1:num_iter, loss_v1A1, 'r-', 'LineWidth', 1.5, 'DisplayName', 'v1 A1 Loss');
hold on;
plot(1:num_iter, loss_v1C1, 'b-', 'LineWidth', 1.5, 'DisplayName', 'v1 C1 Loss');
plot(1:num_iter, loss_v2A1, 'm-', 'LineWidth', 1.5, 'DisplayName', 'v2 A1 Loss');
plot(1:num_iter, loss_v2C1, 'g-', 'LineWidth', 1.5, 'DisplayName', 'v2 C1 Loss');
xlabel('Iteration');
ylabel('Loss');
legend('Location', 'Best');
title('Training Losses for All Networks');
grid on;

% Compute Optimal Policy
% Align lengths for A1
len_a1 = min(length(X1_sort), length(yA1_2)); % Use the shortest length
optimal_policy_nn = zeros(1, len_a1); % Initialize optimal policy array

for i = 1:len_a1
    if yA1_1(i) >= yA1_2(i)
        optimal_policy_nn(i) = 1; % Action = 1
    else
        optimal_policy_nn(i) = 2; % Action = 2
    end
end

% Plot Optimal Decision Policy for Neural Networks (A1)
figure;
plot(X1_sort(1:len_a1), yA1_1(1:len_a1), 'r--', 'LineWidth', 2, 'DisplayName', 'v_1(S) [A1]');
hold on;
plot(X1_sort(1:len_a1), yA1_2(1:len_a1), 'b--', 'LineWidth', 2, 'DisplayName', 'v_2(S) [A1]');
hold on;
plot(X1_sort(1:len_a1), optimal_policy_nn, 'k-', 'LineWidth', 2, 'DisplayName', 'Optimal Policy [A1]');
xlabel('State S');
ylabel('Optimal Action (\alpha^*)');
ylim([0.5 2.5]);
yticks([1 2]);
title('Optimal Decision Policy for Neural Networks (A1)');
legend('Location', 'best');
grid on;

% Align lengths for C1
len_c1 = min(length(X1_sort),  length(yC1_2)); % Use the shortest length
optimal_policy_nn_c1 = zeros(1, len_c1); % Initialize optimal policy array

for i = 1:len_c1
    if yC1_1(i) >= yC1_2(i)
        optimal_policy_nn_c1(i) = 1; % Action = 1
    else
        optimal_policy_nn_c1(i) = 2; % Action = 2
    end
end

% Plot Optimal Decision Policy for Neural Networks (C1)
figure;
plot(X1_sort(1:len_c1), yC1_1(1:len_c1), 'm--', 'LineWidth', 2, 'DisplayName', 'v_1(S) [C1]');
hold on;
plot(X1_sort(1:len_c1), yC1_2(1:len_c1), 'g--', 'LineWidth', 2, 'DisplayName', 'v_2(S) [C1]');
hold on;
plot(X1_sort(1:len_c1), optimal_policy_nn_c1, 'k-', 'LineWidth', 2, 'DisplayName', 'Optimal Policy [C1]');
xlabel('State S');
ylabel('Optimal Action (\alpha^*)');
ylim([0.5 2.5]);
yticks([1 2]);
title('Optimal Decision Policy for Neural Networks (C1)');
legend('Location', 'best');
grid on;


function [w1, b1, w2, b2,loss,output] = A1_training(X_data, R_data, w1, b1, w2, b2, lr, num_iter, ReLU,loss)
    % Function to train v1(A1) neural network
    % Inputs:
    %   - X_data: Input data (state values)
    %   - R_data: Reward values
    %   - w1, b1: Weights and biases for the first layer
    %   - w2, b2: Weights and biases for the second layer
    %   - lr: Learning rate
    %   - num_iter: Number of iterations
    %   - ReLU: ReLU activation function
    % Outputs:
    %   - w1, b1: Updated weights and biases for the first layer
    %   - w2, b2: Updated weights and biases for the second layer
    %   - cost: Cost at each iteration
    phi_A1 = @(z) z.^2 / 2;  % Loss function for [A1]
    psi_A1 = @(z) -z;
    omega1= @(z) z;
    % Number of samples
    N = length(X_data);
    m = size(w1, 1);  % Number of hidden units

    % Initialize cost storage
    %cost = zeros(1, num_iter);

    for iter = 1:num_iter
        % Forward pass
        Z1 = w1 * X_data + b1;  % (m x #samples)
        A1 = ReLU(Z1);          % ReLU activation
        Z2 = w2 * A1 + b2;      % (1 x #samples)
        A2 = Z2;                % Identity activation
        output=omega1(A2);
        N= length(X_data);
        % Compute cost
        %cost(iter) = mean(A2.^2 / 2 - R_data .* A2);
        loss(iter) = mean(phi_A1(A2) + R_data .* psi_A1(A2));
        % Backpropagation
        dZ2 = A2 - R_data;               % (1 x N)
        dW2 = (dZ2 * A1') / N;           % (1 x m)
        db2 = sum(dZ2, 2) / N;           % Scalar

        dA1 = w2' * dZ2;                 % (m x N)
        dZ1 = dA1 .* (Z1 > 0);           % ReLU derivative
        dW1 = (dZ1 * X_data') / N;       % (m x 1)
        db1 = sum(dZ1, 2) / N;           % (m x 1)

        % Gradient descent update
        w2 = w2 - lr * dW2;
        b2 = b2 - lr * db2;
        w1 = w1 - lr * dW1;
        b1 = b1 - lr * db1;
    end
end

function [w1, b1, w2, b2,loss,output] = C1_training(X_data, R_data, w1, b1, w2, b2, lr, num_iter, ReLU,loss)
    % Function to train v1(A1) neural network
    % Inputs:
    %   - X_data: Input data (state values)
    %   - R_data: Reward values
    %   - w1, b1: Weights and biases for the first layer
    %   - w2, b2: Weights and biases for the second layer
    %   - lr: Learning rate
    %   - num_iter: Number of iterations
    %   - ReLU: ReLU activation function
    % Outputs:
    %   - w1, b1: Updated weights and biases for the first layer
    %   - w2, b2: Updated weights and biases for the second layer
    %   - cost: Cost at each iteration
    phi_C1 = @(z)  2./(1+exp(z))+2.*log(1+exp(z)); %@(z) (z - 1).^2 .* (z > 1) + (z + 1).^2 .* (z < -1); % Penalize outside [-1, 1]
    psi_C1 =  @(z) -log(1+exp(z));  %@(z) 2 * max(-1, min(1, z)); % Ensure bounded range
    omega_C1 = @(z) 2.* exp(z)./(1+ exp(z)); 
    % Number of samples
    N = length(X_data);
    m = size(w1, 1);  % Number of hidden units

    % Initialize cost storage
    %cost = zeros(1, num_iter);

    for iter = 1:num_iter
        % Forward pass
        Z1 = w1 * X_data + b1;  % (m x #samples)
        A1 = ReLU(Z1);          % ReLU activation
        Z2 = w2 * A1 + b2;      % (1 x #samples)
        A2 = Z2;                % Identity activation
        output=omega_C1(A2);
        % Compute cost
        %cost(iter) = mean(A2.^2 / 2 - R_data .* A2);
        loss(iter) = mean(phi_C1(A2) + (R_data) .* psi_C1(A2));
        % Backpropagation
        dZ2 = -2.* exp(A2)./(1+exp(A2)).^2 + ...
                  (2 - R_data).* exp(A2)./(1+exp(A2));              % (1 x N)
        % dZ2 =-2.* exp(A2)./(1+exp(A2)).^2 + ...
        %           (2 - R_data).* exp(A2)./(1+exp(A2));
        dW2 = (dZ2 * A1') / N;           % (1 x m)
        db2 = sum(dZ2, 2) / N;           % Scalar

        dA1 = w2' * dZ2;                 % (m x N)
        dZ1 = dA1 .* (Z1 > 0);           % ReLU derivative
        dW1 = (dZ1 * X_data') / N;       % (m x 1)
        db1 = sum(dZ1, 2) / N;           % (m x 1)

        % Gradient descent update
        w2 = w2 - lr * dW2;
        b2 = b2 - lr * db2;
        w1 = w1 - lr * dW1;
        b1 = b1 - lr * db1;
    end
end
