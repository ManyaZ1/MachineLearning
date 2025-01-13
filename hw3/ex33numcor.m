
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numeric solution (Infinite Horizon)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 clc; close all;

%% (0) Helper definitions
reward = @(y) min(2, y.^2);

% Transition CDFs for actions
% H1(s_{t+1}, s_t) = Phi( s_{t+1} - [0.8*s_t + 1] )
% H2(s_{t+1}, s_t) = Phi( s_{t+1} - (-2) )
H1 = @(st1, st) normcdf(st1, 0.8*st + 1, 1);
H2 = @(st1, st) normcdf(st1, -2,         1);

gamma  = 0.8;    % discount factor
tol    = 1e-6;   % convergence tolerance
maxIter= 1000;    % max number of iterations

%% (1) Discretize State Spaces
n  = 500;  
X  = linspace(-6,  6,   n);   % possible "current" states S_t
%Y  = linspace(-3,3, n);   % possible "next" states S_{t+1}
Y=X;
Rvec = min(2, X'.^2);
%Rvec = reward(Y)';  

%% (2) Build Transition Matrices F1, F2  (each n x n)
F1 = zeros(n, n);
F2 = zeros(n, n);

for j = 1:n
   
    F1(j,1) = 0.5 * ( H1(Y(2), X(j)) - H1(Y(1), X(j)) );
    F2(j,1) = 0.5 * ( H2(Y(2), X(j)) - H2(Y(1), X(j)) );

    for k = 2:(n-1)
        F1(j,k) = 0.5 * ( H1(Y(k+1), X(j)) - H1(Y(k), X(j)) );
        F2(j,k) = 0.5 * ( H2(Y(k+1), X(j)) - H2(Y(k), X(j)) );
    end

    F1(j,n) = 0.5 * ( H1(Y(n),   X(j)) - H1(Y(n-1), X(j)) );
    F2(j,n) = 0.5 * ( H2(Y(n),   X(j)) - H2(Y(n-1), X(j)) );

end

%% (3) Value Iteration
V1 = zeros(n,1);   % n x 1
V2 = zeros(n,1);   % n x 1

for iter = 1:maxIter
    oldV1 = V1;
    oldV2 = V2;

    % bestNext is also (n x 1)
    bestNext = max(oldV1, oldV2);

    % Rvec is (n x 1), bestNext is (n x 1), so Rvec + gamma*bestNext is (n x 1).
    % F1, F2 are (n x n). => (n x n) * (n x 1) => (n x 1).
    V1 = F1 * ( Rvec + gamma * bestNext );
    V2 = F2 * ( Rvec + gamma * bestNext );
    % Check convergence
    diff1 = max(abs(V1 - oldV1));
    diff2 = max(abs(V2 - oldV2));
    if max(diff1, diff2) < tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
end

%% (4) Plot results
figure('Name','Infinite Horizon V1 vs V2');
plot(X, V1, 'k-', 'LineWidth',1.5); hold on; grid on;
plot(X, V2, 'r-', 'LineWidth',1.5);
legend('V1','V2','Location','Best');
xlabel('State S'); ylabel('Value');
xlim([-6,6]);
title('Numerical Value Functions for \gamma=0.8');

% Optional: Optimal policy
policy = ones(n,1);
policy(V2 > V1) = 2;
figure('Name','Optimal Policy');
plot(X, policy, 'b.', 'MarkerSize',8); grid on;
xlabel('State'); ylabel('Action');
title('Policy = 1 or 2');
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
num_iter = 20000;
lr = 0.0002;  % Reduced learning rate
m = 100;
gamma = 0.8;
loss = zeros(1, num_iter);

% Neural network initialization
input_size = size(X1_data, 1);
w1 = 2*randn(m, 1) * sqrt(2 / input_size); % He initialization
b1 = zeros(m, 1);
w2 = randn(1, m) * sqrt(2 / m);
b2 = 0;

% Activation and other functions
ReLU = @(z) max(0, z);
r = @(y) min(2, y.^2);
phi = @(z) 0.5 * z.^2;
psi = @(z) -z;

% Gradient clipping threshold
grad_clip = 5;

% Training loop
for iter = 1:num_iter
    % Forward pass for current X1_data
    Z1 = w1 * X1_data + b1;
    A1 = ReLU(Z1);
    Z2 = w2 * A1 + b2; % Predicted output
    output = Z2;

    % Target calculation using reward and gamma
    z11 = w1 * Y1_data + b1;
    a11 = ReLU(z11);
    om11 = w2 * a11 + b2;

    z12 = w1 * Y2_data + b1;
    a12 = ReLU(z12);
    om12 = w2 * a12 + b2;

    % Compute the target YY (skip NaNs in padding)
    valid_idx = ~isnan(Y1_data) & ~isnan(Y2_data);
    YY = r(Y1_data(valid_idx)) + gamma * max(om11(valid_idx), om12(valid_idx));

    % Loss calculation
    Z2_valid = Z2(valid_idx);
    c = phi(Z2_valid) + YY .* psi(Z2_valid);
    loss(iter) = mean(c);

    % Backpropagation
    dZ2 = Z2_valid - YY; % Error term for output layer
    dW2 = dZ2 * A1(:, valid_idx)'; % Gradient for w2
    db2 = sum(dZ2); % Gradient for b2
    dA1 = w2' * dZ2; % Error propagated to hidden layer
    dZ1 = dA1 .* (Z1(:, valid_idx) > 0); % Derivative of ReLU
    dW1 = dZ1 * X1_data(valid_idx)'; % Gradient for w1
    db1 = sum(dZ1, 2); % Gradient for b1

    % Gradient clipping
    dW2 = max(min(dW2, grad_clip), -grad_clip);
    db2 = max(min(db2, grad_clip), -grad_clip);
    dW1 = max(min(dW1, grad_clip), -grad_clip);
    db1 = max(min(db1, grad_clip), -grad_clip);

    % Gradient descent update
    w2 = w2 - lr * dW2;
    b2 = b2 - lr * db2;
    w1 = w1 - lr * dW1;
    b1 = b1 - lr * db1;
end
x=linspace(-6,6,1000);
Z1 = w1 * x + b1;
A1 = ReLU(Z1);
Z2 = w2 * A1 + b2; % Predicted output
outputxx = Z2;
figure;
plot(x,outputxx);
% Plot 1: Output after training
[X1_sort, idx1] = sort(X1_data);
yA1_1 = output(idx1);
figure; hold on;
plot(X1_sort, yA1_1, 'r--', 'LineWidth', 2);
title('Output After Training');
xlabel('Sorted X1\_data');
ylabel('Output');

% Plot 2: Loss during training
figure;
plot(1:num_iter, loss, 'b-', 'LineWidth', 1.5);
title('Loss During Training');
xlabel('Iteration');
ylabel('Loss');
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
% 
% X1_data= set1(:,1)';
% Y1_data= set1(:,2)';
% R1_data= reward(Y1_data);
% 
% X2_data= set2(:,1)';
% Y2_data= set2(:,2)';
% R2_data= reward(Y2_data);
% 
% function [w1, b1, w2, b2,loss,output] = A1_training(X_data, R_data, w1, b1, w2, b2, lr, num_iter, ReLU,loss)
%     rho=@(z) -1;
%     phi_A1 = @(z) z.^2 / 2;  % Loss function for [A1]
%     psi_A1 = @(z) -z;
%     omega1= @(z) z;
%     % Number of samples
%     N = length(X_data);
%     m = size(w1, 1);  % Number of hidden units
% 
%     % Initialize cost storage
%     %cost = zeros(1, num_iter);
% 
%     for iter = 1:num_iter
%         % Forward pass
%         Z1 = w1 * X_data + b1;  % (m x #samples)
%         A1 = ReLU(Z1);          % ReLU activation
%         Z2 = w2 * A1 + b2;      % (1 x #samples)
%         A2 = Z2;                % Identity activation
%         output=omega1(A2);
%         N= length(X_data);
%         % Compute cost
%         %cost(iter) = mean(A2.^2 / 2 - R_data .* A2);
%         loss(iter) = mean(phi_A1(A2) + R_data .* psi_A1(A2));
%         % Backpropagation
%         dZ2 = 
%         %A2 - R_data;               % (1 x N)
%         dW2 = (dZ2 * A1') / N;           % (1 x m)
%         db2 = sum(dZ2, 2) / N;           % Scalar
% 
%         dA1 = w2' * dZ2;                 % (m x N)
%         dZ1 = dA1 .* (Z1 > 0);           % ReLU derivative
%         dW1 = (dZ1 * X_data') / N;       % (m x 1)
%         db1 = sum(dZ1, 2) / N;           % (m x 1)
% 
%         % Gradient descent update
%         w2 = w2 - lr * dW2;
%         b2 = b2 - lr * db2;
%         w1 = w1 - lr * dW1;
%         b1 = b1 - lr * db1;
%     end
% end

% % moys
% times=5000;
%  X1=X1_data';
% X2=X2_data';
%  Y1=Y1_data';
%  Y2=Y2_data';
%  R1=reward(Y1);
%  R2=reward(Y2);
% rl_a1
% XX=linspace(-6,6,n);%-5+10*[0:500]'/500;
% XX=X;
% ZZ1=A01*XX+B01;
% XX1=max(ZZ1,0);
% WW1=A11'*XX1+B11;
% figure(1)
% hold on
% plot(XX,WW1./10,'b','linewidth',2)
%  ZZ2=A02*XX+B02;
%  XX2=max(ZZ2,0);
%  WW2=A12'*XX2+B12;
%  figure(1)
%  hold on
%  plot(XX,WW2,'color',[0.0 0.6 1],'linewidth',2)