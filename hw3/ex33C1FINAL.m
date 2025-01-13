%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILE: example_value_function_training.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;
rng(0);

%% ========================================================================
%  (1)  Build/solve the MDP via Value Iteration on a discrete state space
%       (This part is just to compare with the neural nets)
% =========================================================================
reward = @(s) min(2, s.^2);

% Transition CDFs for two actions
H1 = @(st1, st) normcdf(st1, 0.8*st + 1, 1);  % Action 1
H2 = @(st1, st) normcdf(st1, -2,        1);  % Action 2

gamma = 0.8;   % Discount factor
tol   = 1e-6;  % Convergence tolerance
maxIter = 1000;

% Discretize State Space
num_states   = 500;
state_range  = linspace(-6, 6, num_states);  % S in [-6, 6]
R_discrete   = reward(state_range)';         % reward for each discrete state

% Build transition matrices F1, F2
F1 = zeros(num_states, num_states);
F2 = zeros(num_states, num_states);
for j = 1:num_states
    s_now = state_range(j);
    for i = 1:num_states
        s_next = state_range(i);

        % We'll do a simplistic approach: difference of CDFs
        % For subinterval boundaries, you can adjust carefully
        if i == 1
            F1(j,i) = H1(s_next, s_now) - H1(s_next - (12/num_states), s_now);
            F2(j,i) = H2(s_next, s_now) - H2(s_next - (12/num_states), s_now);
        else
            % Approx edges similarly. This is just a rough example.
            left_boundary  = state_range(i-1);
            p1_left  = H1(left_boundary, s_now);
            p2_left  = H2(left_boundary, s_now);
            p1_right = H1(s_next,       s_now);
            p2_right = H2(s_next,       s_now);

            F1(j,i) = p1_right - p1_left;
            F2(j,i) = p2_right - p2_left;
        end
    end
end

% Value Iteration
V1 = zeros(num_states,1);
V2 = zeros(num_states,1);
for iter = 1:maxIter
    V1_new = F1 * (R_discrete + gamma*max(V1,V2));
    V2_new = F2 * (R_discrete + gamma*max(V1,V2));

    if max(abs(V1_new - V1)) < tol && max(abs(V2_new - V2)) < tol
        fprintf('Value Iteration converged in %d steps.\n', iter);
        break;
    end
    V1 = V1_new;
    V2 = V2_new;
end

%% ========================================================================
%  (2)  Generate Random Transitions (continuous) & Extract Rewards
% =========================================================================
N = 2000;                           % Number of transitions total
action = randi([1, 2], 1, N);       % Randomly pick action 1 or 2
set1 = [];  % (state, next_state) pairs for action=1
set2 = [];  % (state, next_state) pairs for action=2

% Start from some random state
state = randn;
for i = 1:N
    noise_w = randn;   % W ~ N(0,1)
    if action(i) == 1
        next_state = 0.8*state + 1 + noise_w;
        set1 = [set1; state, next_state];
    else
        next_state = -2 + noise_w;
        set2 = [set2; state, next_state];
    end
    state = next_state;
end

% Convert to row vectors for convenience
X1_data = set1(:,1)';   % states for action=1
Y1_data = set1(:,2)';   % next states for action=1
X2_data = set2(:,1)';   % states for action=2
Y2_data = set2(:,2)';   % next states for action=2

% Compute immediate rewards for each next state
R1_data = reward(Y1_data);
R2_data = reward(Y2_data);

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
numIterations = 10000;  % training iterations
learningRate  = 2e-4;   % mu
momentum      = 0.99;   % lambda
stabConst     = 0.1;    % c
numHidden     = 100;    % # hidden neurons
a = 0; 
b = 10;   % final activation scaling

% Network 1 params
A01 =  2*randn(numHidden,1)*sqrt(2/numHidden);
B01 =  zeros(numHidden,1);
A11 = (2/numHidden)*randn(numHidden,1)*sqrt(2/numHidden);
B11 = -0.3;

% Momentum accumulators
PDA01 = zeros(numHidden,1); 
PDB01 = zeros(numHidden,1);
PDA11 = zeros(numHidden,1);
PDB11 = 0;

% Network 2 params
A02 = (2/numHidden)*randn(numHidden,1)*sqrt(2/numHidden);
B02 = zeros(numHidden,1);
A12 = (2/numHidden)*randn(numHidden,1)*sqrt(2/numHidden);
B12 = 0;

% Momentum accumulators
PDA02 = zeros(numHidden,1);
PDB02 = zeros(numHidden,1);
PDA12 = zeros(numHidden,1);
PDB12 = 0;

% For logging cost
costNet1 = zeros(numIterations,1);
costNet2 = zeros(numIterations,1);

%% ========================================================================
%  (4)  Training Loop
% =========================================================================
for iter = 1:numIterations
    
    %% ------------------- NETWORK 1: FORWARD PASS -----------------------
    % "current state" for action=1
    Z1_cur = bsxfun(@plus, A01*X1_data, B01);  % size: (numHidden x len1)
    X1_cur = max(Z1_cur, 0);                  % ReLU
    W1_cur = A11' * X1_cur + B11;             % (1 x len1)
    Q1_cur = a./(1+exp(W1_cur)) + b./(1+exp(-W1_cur));  % final activation => Q1(s)

    % "next state" evaluation => Q1(next_s) and Q2(next_s), for each sample
    Z1_nextSt = bsxfun(@plus, A01*Y1_data, B01);
    X1_nextSt = max(Z1_nextSt, 0);
    W1_nextSt = A11' * X1_nextSt + B11;
    Q1_nextSt_1 = a./(1+exp(W1_nextSt)) + b./(1+exp(-W1_nextSt));  % Q1( next_s )

    Z1_nextSt_2 = bsxfun(@plus, A02*Y1_data, B02);
    X1_nextSt_2 = max(Z1_nextSt_2, 0);
    W1_nextSt_2 = A12' * X1_nextSt_2 + B12;
    Q1_nextSt_2 = a./(1+exp(W1_nextSt_2)) + b./(1+exp(-W1_nextSt_2)); % Q2( next_s )

    % Valid samples for network 1 (ignore NaNs from padding)
    validMask1 = ~isnan(X1_data) & ~isnan(Y1_data) & ~isnan(R1_data);

    % Slice out relevant columns for training
    Q1_cur_valid     = Q1_cur(validMask1);
    Q1_ns_1_valid    = Q1_nextSt_1(validMask1);
    Q1_ns_2_valid    = Q1_nextSt_2(validMask1);
    R1_valid         = R1_data(validMask1);

    % Build the target for network 1
    Y1_target = R1_valid + gamma .* max(Q1_ns_1_valid, Q1_ns_2_valid);

    % Also gather hidden activations to backprop
    Z1_cur_valid = Z1_cur(:, validMask1);  % (numHidden x #valid)
    X1_cur_valid = X1_cur(:, validMask1);  
    W1_cur_valid = W1_cur(validMask1);
    maskZ1_cur_valid = double(Z1_cur_valid > 0);

    % Convert to column vectors for convenience
    W1_col     = W1_cur_valid(:);
    Q1_col     = Q1_cur_valid(:);
    T1_col     = Y1_target(:);

    % d/dW = -(1/(1+exp(-W))) * (Target - Output)
    deltaOutput1 = -(1./(1+exp(-W1_col))) .* (T1_col - Q1_col);  % (#valid x 1)

    % Gradient wrt A11, B11
    dA11 = X1_cur_valid * deltaOutput1;   % (numHidden x 1)
    dB11 = sum(deltaOutput1);

    % For A01, B01 we also need X1_data(valid)
    X1_data_valid = X1_data(validMask1);
    X1_data_mat   = repmat(X1_data_valid, numHidden, 1);  % (numHidden x #valid)
    mask_mul_1    = maskZ1_cur_valid .* X1_data_mat;
    dA01_vec      = mask_mul_1 * deltaOutput1;             % (numHidden x 1)
    dA01          = A11 .* dA01_vec;  % as per your custom formula

    dB01_vec = maskZ1_cur_valid * deltaOutput1;  % (numHidden x 1)
    dB01     = A11 .* dB01_vec;

    % Momentum accumulators for net 1
    PDA11 = momentum * PDA11 + (1 - momentum)*(dA11.^2);
    PDB11 = momentum * PDB11 + (1 - momentum)*(dB11.^2);
    PDA01 = momentum * PDA01 + (1 - momentum)*(dA01.^2);
    PDB01 = momentum * PDB01 + (1 - momentum)*(dB01.^2);

    % Update
    A01 = A01 - learningRate*(dA01 );
    B01 = B01 - learningRate*(dB01);
    A11 = A11 - learningRate*(dA11 );
    B11 = B11 - learningRate*(dB11 );

    %% ------------------- NETWORK 2: FORWARD PASS -----------------------
    Z2_cur = bsxfun(@plus, A02*X2_data, B02);  
    X2_cur = max(Z2_cur, 0);
    W2_cur = A12' * X2_cur + B12; 
    Q2_cur = a./(1+exp(W2_cur)) + b./(1+exp(-W2_cur));

    % Next-state
    Z2_nextSt = bsxfun(@plus, A02*Y2_data, B02);
    X2_nextSt = max(Z2_nextSt, 0);
    W2_nextSt = A12' * X2_nextSt + B12;
    Q2_nextSt_2 = a./(1+exp(W2_nextSt)) + b./(1+exp(-W2_nextSt));  % Q2(next_s)

    Z2_nextSt_1 = bsxfun(@plus, A01*Y2_data, B01);
    X2_nextSt_1 = max(Z2_nextSt_1, 0);
    W2_nextSt_1 = A11' * X2_nextSt_1 + B11;
    Q2_nextSt_1 = a./(1+exp(W2_nextSt_1)) + b./(1+exp(-W2_nextSt_1));  % Q1(next_s)

    validMask2 = ~isnan(X2_data) & ~isnan(Y2_data) & ~isnan(R2_data);

    Q2_cur_valid     = Q2_cur(validMask2);
    Q2_ns_2_valid    = Q2_nextSt_2(validMask2);
    Q2_ns_1_valid    = Q2_nextSt_1(validMask2);
    R2_valid         = R2_data(validMask2);

    % Target for network2
    Y2_target = R2_valid + gamma .* max(Q2_ns_1_valid, Q2_ns_2_valid);

    % For backprop
    Z2_cur_valid  = Z2_cur(:, validMask2);
    X2_cur_valid  = X2_cur(:, validMask2);
    W2_cur_valid  = W2_cur(validMask2);
    maskZ2_cur_valid = double(Z2_cur_valid > 0);

    W2_col  = W2_cur_valid(:);
    Q2_col  = Q2_cur_valid(:);
    T2_col  = Y2_target(:);

    deltaOutput2 = -(1./(1+exp(-W2_col))) .* (T2_col - Q2_col);

    dA12 = X2_cur_valid * deltaOutput2;   
    dB12 = sum(deltaOutput2);

    X2_data_valid = X2_data(validMask2);
    X2_mat        = repmat(X2_data_valid, numHidden, 1);
    mask_mul_2    = maskZ2_cur_valid .* X2_mat;
    dA02_vec      = mask_mul_2 * deltaOutput2;
    dA02          = A12 .* dA02_vec;

    dB02_vec = maskZ2_cur_valid * deltaOutput2;
    dB02     = A12 .* dB02_vec;

    % Momentum accumulators for net 2
    PDA12 = momentum * PDA12 + (1 - momentum)*(dA12.^2);
    PDB12 = momentum * PDB12 + (1 - momentum)*(dB12.^2);
    PDA02 = momentum * PDA02 + (1 - momentum)*(dA02.^2);
    PDB02 = momentum * PDB02 + (1 - momentum)*(dB02.^2);

    % Update
    A02 = A02 - learningRate*(dA02 ./ sqrt(stabConst + PDA02));
    B02 = B02 - learningRate*(dB02 ./ sqrt(stabConst + PDB02));
    A12 = A12 - learningRate*(dA12 ./ sqrt(stabConst + PDA12));
    B12 = B12 - learningRate*(dB12 ./ sqrt(stabConst + PDB12));

    %% -------------------------------------------------------------
    %  Compute a simple cost measure for logging
    %% -------------------------------------------------------------
    costNet1(iter) = 0.5 * mean((T1_col - Q1_col).^2, 'omitnan');
    costNet2(iter) = 0.5 * mean((T2_col - Q2_col).^2, 'omitnan');
end

fprintf('Training complete. Networks updated.\n');

%% ========================================================================
%  (5)  Plot the Loss
% =========================================================================
figure; 
plot(1:numIterations, costNet1, 'b-', 'LineWidth',1.5); hold on;
plot(1:numIterations, costNet2, 'r-', 'LineWidth',1.5);
xlabel('Iteration');
ylabel('Loss');
legend('Network 1','Network 2');
title('Loss During Training');
grid on;

%% ========================================================================
%  (6)  Evaluate final networks over a grid y in [-6,6]
% =========================================================================
y = linspace(-6,6,1000);

% Network 1
Z1_eval = bsxfun(@plus, A01*y, B01);
X1_eval = max(Z1_eval,0);
W1_eval = A11'*X1_eval + B11;
Q1_eval = a./(1+exp(W1_eval)) + b./(1+exp(-W1_eval));

% Network 2
Z2_eval = bsxfun(@plus, A02*y, B02);
X2_eval = max(Z2_eval,0);
W2_eval = A12'*X2_eval + B12;
Q2_eval = a./(1+exp(W2_eval)) + b./(1+exp(-W2_eval));

figure; hold on;
plot(state_range, V1, 'k-','LineWidth',1.5, 'DisplayName','V1 (Action1, MDP)');
plot(state_range, V2, 'k-','LineWidth',1.5, 'DisplayName','V2 (Action2, MDP)');
plot(y, Q1_eval, 'g--','LineWidth',2, 'DisplayName','NN - Q1');
plot(y, Q2_eval, 'm--','LineWidth',2, 'DisplayName','NN - Q2');
xlabel('State S'); ylabel('Network Output');
title('Final Networks Output vs. Value Iteration');
legend('Location','best'); grid on;

fprintf('All done!\n');

%% ========================================================================
%  (7)  Plot Optimal Decision Policy
% =========================================================================

% From Value Iteration (numerical)
policy_numerical = ones(num_states, 1); % Default action 1
policy_numerical(V2 > V1) = 2;         % Action 2 where V2 > V1

% From Neural Network
policy_nn = ones(size(y)); % Default action 1
policy_nn(Q2_eval > Q1_eval) = 2; % Action 2 where Q2_eval > Q1_eval

% Plotting
figure; hold on;
plot(state_range, V1, 'k-','LineWidth',1.5, 'DisplayName','V1');
plot(state_range, V2, 'k-','LineWidth',1.5, 'DisplayName','V2');
plot(y, Q1_eval, 'g--','LineWidth',2, 'DisplayName','Network1');
plot(y, Q2_eval, 'm--','LineWidth',2, 'DisplayName','Network2');
plot(state_range, policy_numerical, 'k-', 'LineWidth', 2, 'DisplayName', 'Optimal Policy numerical');
plot(y, policy_nn, 'r-', 'LineWidth', 2, 'DisplayName', 'Optimal Policy [C1]');

% Formatting
xlim([-5,5]);
ylim([-6, 13]);
yticks([1, 2]);
xlabel('State S');
ylabel('Optimal Action');
title('Optimal Decision Policy');
legend('Location', 'best');
grid on;

fprintf('Optimal Decision Policy plotted.\n');
