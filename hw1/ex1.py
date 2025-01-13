import numpy as np 
import matplotlib.pyplot as plt 
 
#####################      DATA GENERATION      ##################### 
n_samples = 10**6 
# H0: f0 ~ N(0,1) 
xH0 = np.random.normal(0, 1, (n_samples, 2)) 
# H1: f1 ~ 0.5 * N(-1,1) + 0.5 * N(1,1) 
rand_vals = np.random.rand(n_samples, 2) 
neg_indices = rand_vals < 0.5 
pos_indices = ~neg_indices 
xH1 = np.zeros((n_samples, 2)) 
xH1[neg_indices] = np.random.normal(-1, 1, np.sum(neg_indices)) 
xH1[pos_indices] = np.random.normal(1, 1, np.sum(pos_indices)) 
#####################      Bayes Rule      ##################### 
# Density functions 
def f0(x): 
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) 
def f1(x): 
    return 0.5 * (np.exp(-(x + 1)**2 / 2) + np.exp(-(x - 1)**2 / 2)) / np.sqrt(2 * np.pi) 
# Compute log-likelihood ratios for numerical stability 
def log_f0(x): 
    return -0.5 * x**2 - 0.5 * np.log(2 * np.pi) 
def log_f1(x): 
    log_f1_neg = -0.5 * (x + 1)**2 - 0.5 * np.log(2 * np.pi) 
    log_f1_pos = -0.5 * (x - 1)**2 - 0.5 * np.log(2 * np.pi) 
    # Sum the probabilities in log-space 
    max_log = np.maximum(log_f1_neg, log_f1_pos) 
    log_sum = max_log + np.log(0.5 * np.exp(log_f1_neg - max_log) + 0.5 * np.exp(log_f1_pos - max_log)) 
    return log_sum 
# Compute errors under H0 
log_q_H0 = np.sum(log_f1(xH0) - log_f0(xH0), axis=1) 
err_0 = np.sum(log_q_H0 > 0) 
# Compute errors under H1 
log_q_H1 = np.sum(log_f1(xH1) - log_f0(xH1), axis=1) 
err_1 = np.sum(log_q_H1 <= 0) 
# Total error 
errors = err_0 + err_1 
total_decisions = n_samples * 2 
error_percentage = errors / total_decisions * 100 
error_percentage_H0 = err_0 / n_samples * 100 
error_percentage_H1 = err_1 / n_samples * 100 
print(f"Bayes error percentage: {error_percentage}%") 
print(f"Bayes error percentage for H0: {error_percentage_H0}%") 
print(f"Bayes error percentage for H1: {error_percentage_H1}%") 
 
#####################      Neural Network Training     
# Training data: 200 samples from each class 
N_train = 200 
# Generate training data for H0 
xH0_train = np.random.normal(0, 1, (N_train, 2)) 
yH0_train = np.zeros(N_train) 
# Generate training data for H1 
rand_vals_train = np.random.rand(N_train, 2) 
neg_indices_train = rand_vals_train < 0.5 
pos_indices_train = ~neg_indices_train 
xH1_train = np.zeros((N_train, 2)) 
xH1_train[neg_indices_train] = np.random.normal(-1, 1, 
np.sum(neg_indices_train)) 
xH1_train[pos_indices_train] = np.random.normal(1, 1, 
np.sum(pos_indices_train)) 
yH1_train = np.ones(N_train) 
 
# Combine training data 
X_train = np.vstack((xH0_train, xH1_train)) 
y_train = np.hstack((yH0_train, yH1_train)) 
 
# Shuffle training data 
perm = np.random.permutation(len(X_train)) 
X_train = X_train[perm] 
y_train = y_train[perm] 
'''''' 
# Normalize input data 
mean = np.mean(X_train, axis=0) 
std = np.std(X_train, axis=0) 
X_train_normalized = (X_train - mean) / std 
 
# Neural network parameters 
input_size = 2 
hidden_size = 20 
output_size = 1 
 
def initialize_weights(m, n): 
    std_dev = np.sqrt(2 / (n + m)) 
    weights = np.random.normal(0, std_dev, (m, n)) 
    biases = np.zeros((m, 1)) 
    return weights, biases 
 
# Initialize weights for Cross-Entropy network 
W1_ce, b1_ce = initialize_weights(hidden_size, input_size) 
W2_ce, b2_ce = initialize_weights(output_size, hidden_size) 
 
# Initialize weights for Exponential Loss network 
W1_exp, b1_exp = initialize_weights(hidden_size, input_size) 
W2_exp, b2_exp = initialize_weights(output_size, hidden_size) 
 
# Training parameters 
learning_rate = 0.001 
beta1 = 0.9 
beta2 = 0.999 
epsilon = 1e-8 
max_epochs = 10000 
batch_size = 32 
 
# Activation functions 
def sigmoid(z): 
    return 1 / (1 + np.exp(-z)) 
 
def tanh_derivative(a): 
    return 1 - a**2 
 
################### Cross-Entropy Loss Network Training  
# Adam optimizer variables 
t_ce = 0 
m_W1_ce = np.zeros_like(W1_ce) 
v_W1_ce = np.zeros_like(W1_ce) 
m_b1_ce = np.zeros_like(b1_ce) 
v_b1_ce = np.zeros_like(b1_ce) 
m_W2_ce = np.zeros_like(W2_ce) 
v_W2_ce = np.zeros_like(W2_ce) 
m_b2_ce = np.zeros_like(b2_ce) 
v_b2_ce = np.zeros_like(b2_ce) 
 
cost_history_ce = [] 
smooth_cost_hist=[] 
#cost_history = [] 
smooth_cost_window = 20  # Using a 20-cost window for smoothing 
#convergence_threshold = 1e-6  # Define a small threshold for convergence 
for epoch in range(max_epochs): 
    # Shuffle training data at the beginning of each epoch 
    perm = np.random.permutation(len(X_train)) 
    X_train = X_train[perm] 
    y_train = y_train[perm] 
    X_train_normalized = X_train_normalized[perm] 
     
    epoch_cost = 0 
    num_batches = int(np.ceil(len(X_train) / batch_size)) 
     
    for batch in range(num_batches): 
        start = batch * batch_size 
        end = min(start + batch_size, len(X_train)) 
        X_batch = X_train_normalized[start:end].T 
        y_batch = y_train[start:end].reshape(1, -1) 
         
        # Forward pass 
        Z1 = np.dot(W1_ce, X_batch) + b1_ce 
        A1 = np.tanh(Z1) 
        Z2 = np.dot(W2_ce, A1) + b2_ce 
        A2 = sigmoid(Z2) 
                # Compute cost 
        m = y_batch.shape[1] 
        cost = -np.sum(y_batch * np.log(A2 + epsilon) + (1 - y_batch) * 
np.log(1 - A2 + epsilon)) / m 
        epoch_cost += cost 
         
        # Backward pass 
        dZ2 = A2 - y_batch 
        dW2 = np.dot(dZ2, A1.T) / m 
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m 
        dA1 = np.dot(W2_ce.T, dZ2) 
        dZ1 = dA1 * tanh_derivative(A1) 
        dW1 = np.dot(dZ1, X_batch.T) / m 
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m 
         
        # Adam optimizer update 
        t_ce += 1 
        # Update first moment estimates 
        m_W1_ce = beta1 * m_W1_ce + (1 - beta1) * dW1 
        m_b1_ce = beta1 * m_b1_ce + (1 - beta1) * db1 
        m_W2_ce = beta1 * m_W2_ce + (1 - beta1) * dW2 
        m_b2_ce = beta1 * m_b2_ce + (1 - beta1) * db2 
        # Update second moment estimates 
        v_W1_ce = beta2 * v_W1_ce + (1 - beta2) * (dW1 ** 2) 
        v_b1_ce = beta2 * v_b1_ce + (1 - beta2) * (db1 ** 2) 
        v_W2_ce = beta2 * v_W2_ce + (1 - beta2) * (dW2 ** 2) 
        v_b2_ce = beta2 * v_b2_ce + (1 - beta2) * (db2 ** 2) 
        # Compute bias-corrected moment estimates 
        m_W1_ce_hat = m_W1_ce / (1 - beta1 ** t_ce) 
        m_b1_ce_hat = m_b1_ce / (1 - beta1 ** t_ce) 
        v_W1_ce_hat = v_W1_ce / (1 - beta2 ** t_ce) 
        v_b1_ce_hat = v_b1_ce / (1 - beta2 ** t_ce) 
        m_W2_ce_hat = m_W2_ce / (1 - beta1 ** t_ce) 
        m_b2_ce_hat = m_b2_ce / (1 - beta1 ** t_ce) 
        v_W2_ce_hat = v_W2_ce / (1 - beta2 ** t_ce) 
        v_b2_ce_hat = v_b2_ce / (1 - beta2 ** t_ce) 
        # Update parameters 
        W1_ce -= learning_rate * m_W1_ce_hat / (np.sqrt(v_W1_ce_hat) + 
epsilon) 
        b1_ce -= learning_rate * m_b1_ce_hat / (np.sqrt(v_b1_ce_hat) + 
epsilon) 
        W2_ce -= learning_rate * m_W2_ce_hat / (np.sqrt(v_W2_ce_hat) + 
epsilon) 
        b2_ce -= learning_rate * m_b2_ce_hat / (np.sqrt(v_b2_ce_hat) + 
epsilon) 
     
    cost_history_ce.append(epoch_cost / num_batches) 
     # Smoothing cost calculation 
    if (len(cost_history_ce) % smooth_cost_window) == 0: 
        smoothed_cost = np.mean(cost_history_ce[-smooth_cost_window:]) 
        smooth_cost_hist.append(smoothed_cost) 
        if len(cost_history_ce) > 2*smooth_cost_window: 
            cost_dif=np.abs(smooth_cost_hist[-1]-smooth_cost_hist[-2]) 
            #if cost_dif<convergence_threshold: 
            #    print(f"cross entropy Converged at epoch {epoch}") 
            #    break 
    else: 
        smoothed_cost = np.mean(cost_history_ce) 
    # Print cost every 1000 epochs 
    if (epoch + 1) % 1000 == 0: 
        print(f"Epoch {epoch + 1}, Cross-Entropy Loss: {cost_history_ce[-1]}") 
 
################### Exponential Loss Network Training  
# Adam optimizer variables 
t_exp = 0 
m_W1_exp = np.zeros_like(W1_exp) 
v_W1_exp = np.zeros_like(W1_exp) 
m_b1_exp = np.zeros_like(b1_exp) 
v_b1_exp = np.zeros_like(b1_exp) 
m_W2_exp = np.zeros_like(W2_exp) 
v_W2_exp = np.zeros_like(W2_exp) 
m_b2_exp = np.zeros_like(b2_exp) 
v_b2_exp = np.zeros_like(b2_exp) 
 
cost_history_exp = [] 
smooth_cost_histexp=[] 
 
# Convert labels to {-1, 1} 
y_train_exp = np.where(y_train == 1, 1, -1) 
 
for epoch in range(max_epochs): 
    # Shuffle training data at the beginning of each epoch 
    perm = np.random.permutation(len(X_train)) 
    X_train = X_train[perm] 
    y_train_exp = y_train_exp[perm] 
    X_train_normalized = X_train_normalized[perm] 
     
    epoch_cost = 0 
    num_batches = int(np.ceil(len(X_train) / batch_size)) 
     
    for batch in range(num_batches): 
        start = batch * batch_size 
        end = min(start + batch_size, len(X_train)) 
        X_batch = X_train_normalized[start:end].T 
        y_batch = y_train_exp[start:end].reshape(1, -1) 
         
        # Forward pass 
        Z1 = np.dot(W1_exp, X_batch) + b1_exp 
        A1 = np.tanh(Z1) 
        Z2 = np.dot(W2_exp, A1) + b2_exp 
        A2 = Z2  # No activation in output layer 
         
        # Compute cost 
        m = y_batch.shape[1] 
        cost = np.sum(np.exp(-0.5 * y_batch * A2)) / m 
        epoch_cost += cost 
         
        # Backward pass 
        dZ2 = -0.5 * y_batch * np.exp(-0.5 * y_batch * A2) 
        dW2 = np.dot(dZ2, A1.T) / m 
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m 
        dA1 = np.dot(W2_exp.T, dZ2) 
        dZ1 = dA1 * tanh_derivative(A1) 
        dW1 = np.dot(dZ1, X_batch.T) / m 
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m 
         
        # Adam optimizer update 
        t_exp += 1 
        # Update first moment estimates 
        m_W1_exp = beta1 * m_W1_exp + (1 - beta1) * dW1 
        m_b1_exp = beta1 * m_b1_exp + (1 - beta1) * db1 
        m_W2_exp = beta1 * m_W2_exp + (1 - beta1) * dW2 
        m_b2_exp = beta1 * m_b2_exp + (1 - beta1) * db2 
        # Update second moment estimates 
        v_W1_exp = beta2 * v_W1_exp + (1 - beta2) * (dW1 ** 2) 
        v_b1_exp = beta2 * v_b1_exp + (1 - beta2) * (db1 ** 2) 
        v_W2_exp = beta2 * v_W2_exp + (1 - beta2) * (dW2 ** 2) 
        v_b2_exp = beta2 * v_b2_exp + (1 - beta2) * (db2 ** 2) 
        # Compute bias-corrected moment estimates 
        m_W1_exp_hat = m_W1_exp / (1 - beta1 ** t_exp) 
        m_b1_exp_hat = m_b1_exp / (1 - beta1 ** t_exp) 
        v_W1_exp_hat = v_W1_exp / (1 - beta2 ** t_exp) 
        v_b1_exp_hat = v_b1_exp / (1 - beta2 ** t_exp) 
        m_W2_exp_hat = m_W2_exp / (1 - beta1 ** t_exp) 
        m_b2_exp_hat = m_b2_exp / (1 - beta1 ** t_exp) 
        v_W2_exp_hat = v_W2_exp / (1 - beta2 ** t_exp) 
        v_b2_exp_hat = v_b2_exp / (1 - beta2 ** t_exp) 
        # Update parameters 
        W1_exp -= learning_rate * m_W1_exp_hat / (np.sqrt(v_W1_exp_hat) 
+ epsilon) 
        b1_exp -= learning_rate * m_b1_exp_hat / (np.sqrt(v_b1_exp_hat) 
+ epsilon) 
        W2_exp -= learning_rate * m_W2_exp_hat / (np.sqrt(v_W2_exp_hat) 
+ epsilon) 
        b2_exp -= learning_rate * m_b2_exp_hat / (np.sqrt(v_b2_exp_hat) 
+ epsilon) 
     
    cost_history_exp.append(epoch_cost / num_batches) 
     # Smoothing cost calculation 
    if (len(cost_history_exp) % smooth_cost_window) == 0: 
        smoothed_cost = np.mean(cost_history_exp[-smooth_cost_window:]) 
        smooth_cost_histexp.append(smoothed_cost) 
        if len(cost_history_exp) > 2*smooth_cost_window: 
            cost_dif=np.abs(smooth_cost_histexp[-1]-smooth_cost_histexp[-2]) 
            #if cost_dif<convergence_threshold: 
            #    print(f"exponential Converged at epoch {epoch}") 
            #    break 
    else: 
        smoothed_cost = np.mean(cost_history_exp) 
    # Print cost every 1000 epochs 
    if (epoch + 1) % 1000 == 0: 
        print(f"Epoch {epoch + 1}, Exponential Loss: {cost_history_exp[-1]}") 
##################### Testing on Original Data ##################### 
# Combine test data 
X_test = np.vstack((xH0, xH1)) 
y_test = np.hstack((np.zeros(n_samples), np.ones(n_samples))) 
# Normalize test data using training mean and std 
X_test_normalized = (X_test - mean) / std 
# Predict with Cross-Entropy network 
def predict_ce(X): 
    Z1 = np.dot(W1_ce, X.T) + b1_ce 
    A1 = np.tanh(Z1) 
    Z2 = np.dot(W2_ce, A1) + b2_ce 
    A2 = sigmoid(Z2) 
    return A2.flatten() 
predictions_ce = predict_ce(X_test_normalized) 
decisions_ce = np.where(predictions_ce >= 0.5, 1, 0) 
errors_ce = np.sum(decisions_ce != y_test) 
error_percentage_ce = errors_ce / (2 * n_samples) * 100 
error_percentage_ce_H0 = np.sum(decisions_ce[:n_samples] != 
y_test[:n_samples]) / n_samples * 100 
error_percentage_ce_H1 = np.sum(decisions_ce[n_samples:] != 
y_test[n_samples:]) / n_samples * 100 
# Predict with Exponential Loss network 
def predict_exp(X): 
    Z1 = np.dot(W1_exp, X.T) + b1_exp 
    A1 = np.tanh(Z1) 
    Z2 = np.dot(W2_exp, A1) + b2_exp 
    return Z2.flatten() 
predictions_exp = predict_exp(X_test_normalized) 
decisions_exp = np.where(predictions_exp >= 0, 1, 0) 
errors_exp = np.sum(decisions_exp != y_test) 
error_percentage_exp = errors_exp / (2 * n_samples) * 100 
error_percentage_exp_H0 = np.sum(decisions_exp[:n_samples] != 
y_test[:n_samples]) / n_samples * 100 
error_percentage_exp_H1 = np.sum(decisions_exp[n_samples:] != 
y_test[n_samples:]) / n_samples * 100 
#####################  Comparison of Error Rates ##################### 
print("\n=== Error Rates Comparison ===") 
print(f"Bayes Optimal Error Percentage: {error_percentage}%") 
print(f"    Bayes error percentage for H0: {error_percentage_H0}%") 
print(f"    Bayes error percentage for H1: {error_percentage_H1}%") 
print(f"Cross-Entropy Network Error Percentage: {error_percentage_ce}%") 
print(f"   Error Percentage for H0: {error_percentage_ce_H0}%") 
print(f"   Error Percentage for H1: {error_percentage_ce_H1}%") 
print(f"Exponential Loss Network Error Percentage: {error_percentage_exp}%") 
print(f"   Error Percentage for H0: {error_percentage_exp_H0}%") 
print(f"   Error Percentage for H1: {error_percentage_exp_H1}%") 
################### Plot Training Loss ################### 
plt.figure(figsize=(12, 5)) 
# Plot Cross-Entropy and Exponential Loss on the same plot 
plt.plot(smooth_cost_hist, color='blue', label='Cross-Entropy Loss') 
plt.plot(smooth_cost_histexp, color='red', label='Exponential Loss') 
plt.xlabel('Epochs/20') 
plt.ylabel('Loss') 
plt.title('Loss Comparison over Epochs') 
plt.legend()  # Display legend to differentiate between the two losses 
plt.grid(True) 
plt.tight_layout() 
plt.show() 