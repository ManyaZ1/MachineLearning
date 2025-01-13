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
num_iter = 20000;
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


% We might have different lengths in set1 vs set2
% -> pad the shorter with NaNs
len1 = length(Y1_data);
len2 = length(Y2_data);
len_diff = abs(len1 - len2);
if len1 < len2
    % pad set1 with NaNs at the end
    X1_data = [X1_data, nan(1, len_diff)];
    Y1_data = [Y1_data, nan(1, len_diff)];
    R1_data = [R1_data, nan(1, len_diff)];
elseif len2 < len1
    % pad set2 with NaNs
    X2_data = [X2_data, nan(1, len_diff)];
    Y2_data = [Y2_data, nan(1, len_diff)];
    R2_data = [R2_data, nan(1, len_diff)];
end

%% ========================================================================
%  (3)  Neural Network Setup
% =========================================================================
times   = 20000;
mu      = 2*1e-4;   % learning rate
lambda  = 0.99;   % momentum
c       = 0.1;    % stabilization constant
m       = 100;    % # hidden neurons
a = 0; b = 10;    % final activation scaling (somewhat unusual, but we'll keep)

% Network 1 params
A01 = 2*randn(m,1)*sqrt(2/m);
B01 = zeros(m,1);
A11 = 2/m*randn(m,1)*sqrt(2/m);
B11 = -0.3;

% Keep track of momentum
PDA01 = zeros(m,1); 
PDB01 = zeros(m,1);
PDA11 = zeros(m,1);
PDB11 = 0;

% Network 2 params
A02 = 2/m*randn(m,1)*sqrt(2/m);
B02 = zeros(m,1);
A12 = 2/m*randn(m,1)*sqrt(2/m);
B12 = 0;

PDA02 = zeros(m,1);
PDB02 = zeros(m,1);
PDA12 = zeros(m,1);
PDB12 = 0;

cost_c11 = zeros(times,1);
cost_c12 = zeros(times,1);

%% ========================================================================
%  (4)  Training Loop
% =========================================================================
for iter = 1:times
    % -------------------------------------------------------------
    %  Forward for Network 1 over ALL columns (includes NaNs)
    % -------------------------------------------------------------
    Z11 = bsxfun(@plus, A01*X1_data, B01);  % size: (m x len1)
    X11 = max(Z11,0);                      % ReLU
    W11 = A11' * X11 + B11;                % (1 x len1)
    OMW11 = a./(1+exp(W11)) + b./(1+exp(-W11));  % final activation

    % Next-state evaluation for each sample => Q1(next_state) and Q2(next_state)
    z11_ns = bsxfun(@plus, A01*Y1_data, B01);%old matlab problem?? 
    x11_ns = max(z11_ns, 0);
    w11_ns = A11' * x11_ns + B11;
    omw11  = a./(1+exp(w11_ns)) + b./(1+exp(-w11_ns));  % Q1(next_state)

    z12_ns = bsxfun(@plus, A02*Y1_data, B02);
    x12_ns = max(z12_ns, 0);
    w12_ns = A12' * x12_ns + B12;
    omw12  = a./(1+exp(w12_ns)) + b./(1+exp(-w12_ns));  % Q2(next_state)

    % -------------------------------------------------------------
    %  Create a mask for valid (non-NaN) samples for network 1
    % -------------------------------------------------------------
    validMask1 = ~isnan(X1_data) & ~isnan(Y1_data) & ~isnan(R1_data);
    % Pull out the relevant columns from OMW11, omw11, omw12, ...
    OMW11_v   = OMW11(validMask1);
    omw11_v   = omw11(validMask1);
    omw12_v   = omw12(validMask1);
    R1_v      = R1_data(validMask1);

    % Build the target
    YY1 = R1_v + gamma * max(omw11_v, omw12_v);

    % We also need the corresponding hidden activations for backprop
    Z11_v = Z11(:, validMask1);      % (m x #valid)
    X11_v = X11(:, validMask1);      % ReLU
    W11_v = W11(validMask1);         % (1 x #valid)
    maskZ11_v = double(Z11_v > 0);

    % Convert to column vectors for easier vector math
    W11_v_col   = W11_v(:);
    OMW11_v_col = OMW11_v(:);
    YY1_col     = YY1(:);

    % Error derivative: d/dW = -(1/(1+exp(-W))) * (Target - Output)
    U1 = -(1./(1+exp(-W11_v_col))) .* (YY1_col - OMW11_v_col);  % (#valid x 1)

    % Grad wrt A11, B11
    DA11 = X11_v * U1;    % (m x #valid) * (#valid x 1) => (m x 1)
    DB11 = sum(U1);

    % Grad wrt A01, B01
    %    dA01 = A11 .* sum_over_samples_of( maskZ11 .* X1_data * U1 )
    % But we must replicate X1_data => Already stored in X1_data(1, idx).
    X1_data_v = X1_data(validMask1);
    X1_mat    = repmat(X1_data_v, m, 1);   % (m x #valid)
    mask_mul  = (maskZ11_v .* X1_mat);     % (m x #valid)
    DA01_vec  = mask_mul * U1;            % (m x 1)
    DA01      = A11 .* DA01_vec;

    DB01_vec  = maskZ11_v * U1;           % (m x 1)
    DB01      = A11 .* DB01_vec;

    % Momentum accumulators
    PDA11 = lambda * PDA11 + (1 - lambda)*(DA11.^2);
    PDB11 = lambda * PDB11 + (1 - lambda)*(DB11.^2);
    PDA01 = lambda * PDA01 + (1 - lambda)*(DA01.^2);
    PDB01 = lambda * PDB01 + (1 - lambda)*(DB01.^2);

    % Update
    A01 = A01 - mu*(DA01 ./ sqrt(c + PDA01));
    B01 = B01 - mu*(DB01 ./ sqrt(c + PDB01));
    A11 = A11 - mu*(DA11 ./ sqrt(c + PDA11));
    B11 = B11 - mu*(DB11 ./ sqrt(c + PDB11));

    % -------------------------------------------------------------
    %  Forward for Network 2
    % -------------------------------------------------------------
    Z22 = bsxfun(@plus, A02*X2_data, B02);  
    X22 = max(Z22,0);
    W22 = A12' * X22 + B12; 
    OMW22 = a./(1+exp(W22)) + b./(1+exp(-W22));

    % Next-state for action=2
    z22_ns = bsxfun(@plus, A02*Y2_data, B02);
    x22_ns = max(z22_ns, 0);
    w22_ns = A12' * x22_ns + B12;
    omw22  = a./(1+exp(w22_ns)) + b./(1+exp(-w22_ns));

    % Q1 of next-state as well:
    z21_ns = bsxfun(@plus, A01*Y2_data, B01);
    x21_ns = max(z21_ns, 0);
    w21_ns = A11' * x21_ns + B11;
    omw21  = a./(1+exp(w21_ns)) + b./(1+exp(-w21_ns));

    % Valid mask for net2
    validMask2 = ~isnan(X2_data) & ~isnan(Y2_data) & ~isnan(R2_data);

    % Extract columns
    OMW22_v = OMW22(validMask2);
    omw22_v = omw22(validMask2);
    omw21_v = omw21(validMask2);
    R2_v    = R2_data(validMask2);

    % Target for network2
    YY2 = R2_v + gamma * max(omw21_v, omw22_v);

    % For backprop, hidden states
    Z22_v = Z22(:, validMask2);
    X22_v = X22(:, validMask2);
    W22_v = W22(validMask2);
    maskZ22_v = double(Z22_v > 0);

    W22_v_col   = W22_v(:);
    OMW22_v_col = OMW22_v(:);
    YY2_col     = YY2(:);

    U2 = -(1./(1+exp(-W22_v_col))) .* (YY2_col - OMW22_v_col);

    DA12 = X22_v * U2;   
    DB12 = sum(U2);

    X2_data_v = X2_data(validMask2);
    X2_mat = repmat(X2_data_v, m, 1);
    mask_mul2 = (maskZ22_v .* X2_mat);
    DA02_vec  = mask_mul2 * U2;
    DA02      = A12 .* DA02_vec;

    DB02_vec = maskZ22_v * U2;
    DB02     = A12 .* DB02_vec;

    PDA12 = lambda * PDA12 + (1 - lambda)*(DA12.^2);
    PDB12 = lambda * PDB12 + (1 - lambda)*(DB12.^2);
    PDA02 = lambda * PDA02 + (1 - lambda)*(DA02.^2);
    PDB02 = lambda * PDB02 + (1 - lambda)*(DB02.^2);

    A02 = A02 - mu*(DA02 ./ sqrt(c + PDA02));
    B02 = B02 - mu*(DB02 ./ sqrt(c + PDB02));
    A12 = A12 - mu*(DA12 ./ sqrt(c + PDA12));
    B12 = B12 - mu*(DB12 ./ sqrt(c + PDB12));

    % -------------------------------------------------------------
    %  Compute a simple cost measure for logging
    % -------------------------------------------------------------
    cost_c11(iter) = 0.5 * mean((YY1_col - OMW11_v_col).^2, 'omitnan');
    cost_c12(iter) = 0.5 * mean((YY2_col - OMW22_v_col).^2, 'omitnan');
end

fprintf('Training complete. Networks updated.\n');
%%
% Network 1
Z11_eval = bsxfun(@plus, A01*y, B01);
X11_eval = max(Z11_eval,0);
W11_eval = A11'*X11_eval + B11;
OMW11_eval = a./(1+exp(W11_eval)) + b./(1+exp(-W11_eval));

% Network 2
Z22_eval = bsxfun(@plus, A02*y, B02);
X22_eval = max(Z22_eval,0);
W22_eval = A12'*X22_eval + B12;
OMW22_eval = a./(1+exp(W22_eval)) + b./(1+exp(-W22_eval));

% Plot 1: Output after training
figure; hold on;
plot(state_range, V1, 'k-', 'LineWidth', 1.5); hold on; grid on;
plot(state_range, V2, 'c-', 'LineWidth', 1.5); hold on;
plot(x, outputxx1, 'r--', 'LineWidth', 2,'DisplayName','NetworkA1_1');
plot(x, outputxx2, 'b--', 'LineWidth', 2,'DisplayName','NetworkA1_2');
hold on;
%plot(state_range, V1, 'k-','LineWidth',1.5, 'DisplayName','V1');
%plot(state_range, V2, 'k-','LineWidth',1.5, 'DisplayName','V2');
plot(y, OMW11_eval, 'g--','LineWidth',2, 'DisplayName','NetworkC1_1');
plot(y, OMW22_eval, 'g--','LineWidth',2, 'DisplayName','NetworkC1_2');
xlabel('State S'); ylabel('Network Output');
title('Final Networks Output vs. Value Iteration');
legend('Location','best'); grid on;
title('Output After Training');
xlim([-5,5]);
ylim([7,12]);
xlabel('X');
ylabel('Output');
%fprintf('All done!\n');
% Plot 2: Loss during training
figure;
plot(1:num_iter, loss1, 'r--', 'LineWidth', 1.5,'DisplayName','NetworkA1_1'); hold on;
plot(1:num_iter, loss2, 'b--', 'LineWidth', 1.5,'DisplayName','NetworkA1_2');
plot(1:times, cost_c11, 'g--','LineWidth',1.5, 'DisplayName','Network1'); hold on;
plot(1:times, cost_c12, 'r-', 'g--','LineWidth',1.5, 'DisplayName','Network2');
xlabel('Iteration');
ylabel('Loss');
legend('Network 1','Network 2');
title('Loss During Training');
grid on;

%% Optional: Optimal policy
%% Optional: Add Decision Policy
% Compute the optimal policy after value iteration
optimal_policy = zeros(num_states, 1); % Initialize policy vector

for i = 1:num_states
    if V1(i) > V2(i)
        optimal_policy(i) = 1; % Choose action 1
    else
        optimal_policy(i) = 2; % Choose action 2
    end
end
for i = 1:num_states
    if V1(i) > V2(i)
        optimal_policy(i) = 1; % Choose action 1
    else
        optimal_policy(i) = 2; % Choose action 2
    end
end
opt1=zeros(length(outputxx1), 1);
for i=1:length(outputxx1)
    if outputxx1(i)>outpuxx2(i)
        opt1(i)=1;
    else
    opt1(i)=2;
    end
end
l=min(length(OMW11_eval),length(OMW22_eval));
opt2=zeros(l, 1);
for i=1:length(outputxx1)
    if OMW11_eval(i)>OMW22_eval(i)
        opt2(i)=1;
    else
    opt2(i)=2;
    end
end
%% Plot the Decision Policy
figure;
plot(state_range, optimal_policy, 'k-', 'LineWidth', 1.5); hold on;
plot(x, opt1, 'b-', 'LineWidth', 1.5); hold on;
plot(y, opt2, 'r-', 'LineWidth', 1.5);
xlabel('State S');
ylabel('Optimal Action');
title('Optimal Decision Policy');
grid on;


