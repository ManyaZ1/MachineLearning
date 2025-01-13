import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import mnist 
import sys 
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1) 
 
def load_mnist(): 
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data() 
    return train_images, train_labels, test_images, test_labels 
 
train_images, train_labels, test_images, test_labels = load_mnist() 
 
# Filter only the numerals 0 and 8 
def filter_data(images, labels, num1=0, num2=8): 
    filter_mask = (labels == num1) | (labels == num2) 
    return images[filter_mask], labels[filter_mask] 
 
train_images, train_labels = filter_data(train_images, train_labels) 
test_images, test_labels = filter_data(test_images, test_labels) 
 
# Convert labels to binary classification (0 or 1) 
train_labels = (train_labels == 8).astype(int) 
test_labels = (test_labels == 8).astype(int) 
 
# Normalize images to the range [0, 1] 
train_images = train_images / 255.0 
test_images = test_images / 255.0 
 
# Flatten images to vectors of length 784 
train_images = train_images.reshape(train_images.shape[0], -1) 
test_images = test_images.reshape(test_images.shape[0], -1) 
 
# Initialize neural network parameters for Cross-Entropy Network with Adam Optimizer 
np.random.seed(0) 
input_size = 784 
hidden_size = 300 
output_size = 1 
 
# Weights initialization for Cross-Entropy Network 
w1_ce = np.random.normal(0, np.sqrt(1 / (input_size + hidden_size)), 
(input_size, hidden_size)) 
b1_ce = np.zeros(hidden_size) 
w2_ce = np.random.normal(0, np.sqrt(1 / (hidden_size + output_size)), 
(hidden_size, output_size)) 
b2_ce = np.zeros(output_size) 
 
# Adam optimizer variables for Cross-Entropy Network 
t_ce = 0 
m_w1_ce = np.zeros_like(w1_ce) 
v_w1_ce = np.zeros_like(w1_ce) 
m_b1_ce = np.zeros_like(b1_ce) 
v_b1_ce = np.zeros_like(b1_ce) 
m_w2_ce = np.zeros_like(w2_ce) 
v_w2_ce = np.zeros_like(w2_ce) 
m_b2_ce = np.zeros_like(b2_ce) 
v_b2_ce = np.zeros_like(b2_ce) 
 
beta1 = 0.9 
beta2 = 0.999 
epsilon = 1e-8 
 
# Activation functions 
def relu(x): 
    return np.maximum(0, x) 
 
def relu_derivative(x): 
    return (x > 0).astype(float) 
 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 
# Forward pass for Cross-Entropy Network 
def forward_pass_ce(x): 
    z1 = np.dot(x, w1_ce) + b1_ce 
    a1 = relu(z1) 
    z2 = np.dot(a1, w2_ce) + b2_ce 
    a2 = sigmoid(z2) 
    return z1, a1, z2, a2 
 
# Backward pass for Cross-Entropy Network with Adam Optimizer 
def backward_pass_ce(x, y, z1, a1, z2, a2, learning_rate): 
    global w1_ce, b1_ce, w2_ce, b2_ce, t_ce, m_w1_ce, v_w1_ce, m_b1_ce, v_b1_ce, m_w2_ce, v_w2_ce, m_b2_ce, v_b2_ce 
     
    # Output layer error 
    dz2 = a2 - y 
    dw2 = np.dot(a1.T, dz2) / x.shape[0] 
    db2 = np.sum(dz2, axis=0) / x.shape[0] 
     
    # Hidden layer error 
    da1 = np.dot(dz2, w2_ce.T) 
    dz1 = da1 * relu_derivative(a1) 
    dw1 = np.dot(x.T, dz1) / x.shape[0] 
    db1 = np.sum(dz1, axis=0) / x.shape[0] 
     
    # Adam optimizer update 
    t_ce += 1 
    # Update first moment estimates 
    m_w1_ce = beta1 * m_w1_ce + (1 - beta1) * dw1 
    m_b1_ce = beta1 * m_b1_ce + (1 - beta1) * db1 
    m_w2_ce = beta1 * m_w2_ce + (1 - beta1) * dw2 
    m_b2_ce = beta1 * m_b2_ce + (1 - beta1) * db2 
    # Update second moment estimates 
    v_w1_ce = beta2 * v_w1_ce + (1 - beta2) * (dw1 ** 2) 
    v_b1_ce = beta2 * v_b1_ce + (1 - beta2) * (db1 ** 2) 
    v_w2_ce = beta2 * v_w2_ce + (1 - beta2) * (dw2 ** 2) 
    v_b2_ce = beta2 * v_b2_ce + (1 - beta2) * (db2 ** 2) 
    # Compute bias-corrected moment estimates 
    m_w1_ce_hat = m_w1_ce / (1 - beta1 ** t_ce) 
    m_b1_ce_hat = m_b1_ce / (1 - beta1 ** t_ce) 
    v_w1_ce_hat = v_w1_ce / (1 - beta2 ** t_ce) 
    v_b1_ce_hat = v_b1_ce / (1 - beta2 ** t_ce) 
    m_w2_ce_hat = m_w2_ce / (1 - beta1 ** t_ce) 
    m_b2_ce_hat = m_b2_ce / (1 - beta1 ** t_ce) 
    v_w2_ce_hat = v_w2_ce / (1 - beta2 ** t_ce) 
    v_b2_ce_hat = v_b2_ce / (1 - beta2 ** t_ce) 
    # Update parameters 
    w1_ce -= learning_rate * m_w1_ce_hat / (np.sqrt(v_w1_ce_hat) + epsilon) 
    b1_ce -= learning_rate * m_b1_ce_hat / (np.sqrt(v_b1_ce_hat) + epsilon) 
    w2_ce -= learning_rate * m_w2_ce_hat / (np.sqrt(v_w2_ce_hat) + epsilon) 
    b2_ce -= learning_rate * m_b2_ce_hat / (np.sqrt(v_b2_ce_hat) + epsilon) 
 
# Training the Cross-Entropy Network 
epochs = 50 
learning_rate = 0.001 
batch_size = 32 
 
# Convert labels to binary classification (0 or 1) 
y_train_ce = train_labels.reshape(-1, 1) 
y_test_ce = test_labels.reshape(-1, 1) 
 
# Lists to store cost histories 
cost_history_ce = [] 
smooth_ce=[] 
smooth_exp=[] 
cost_history_exp = [] 
s_window=3 #επιλογη παραθύρου για smoothing 
 
def accuracy_score(y_true, y_pred): 
    return np.sum(y_true == y_pred) / len(y_true) 
 
for epoch in range(epochs): 
    # Cross-Entropy Network Training 
    for i in range(0, train_images.shape[0], batch_size): 
        x_batch = train_images[i:i+batch_size] 
        y_batch = y_train_ce[i:i+batch_size] 
         
        # Forward pass 
        z1, a1, z2, a2 = forward_pass_ce(x_batch) 
         
        # Compute cost 
        cost = -np.mean(y_batch * np.log(a2 + epsilon) + (1 - y_batch) * np.log(1 - a2 + epsilon)) 
        cost_history_ce.append(cost) 
         
        # Backward pass 
        backward_pass_ce(x_batch, y_batch, z1, a1, z2, a2, learning_rate) 
    if epoch% s_window==0: 
        smooth_ce.append(np.mean(cost_history_ce[-s_window:])) 
    # Compute training accuracy for Cross-Entropy Network 
    _, _, _, train_output = forward_pass_ce(train_images) 
    train_predictions = (train_output > 0.5).astype(int).flatten() 
    train_accuracy = accuracy_score(train_labels, train_predictions) 
    'print(f"Epoch {epoch + 1}/{epochs} - Cross-Entropy Training Accuracy: {train_accuracy * 100:.2f}%")' 
 
# Evaluate the Cross-Entropy Network on test data 
_, _, _, test_output = forward_pass_ce(test_images) 
test_predictions_ce = (test_output > 0.5).astype(int).flatten() 
test_accuracy_ce = accuracy_score(test_labels, test_predictions_ce) 
print(f"Cross-Entropy Network Error: {(1-test_accuracy_ce) * 100:.2f}%") 
# Error percentage for Cross-Entropy Network 
error_percentage_ce_0 = (np.sum((test_labels == 0) & 
(test_predictions_ce != test_labels)) / np.sum(test_labels == 0)) * 100 
error_percentage_ce_8 = (np.sum((test_labels == 1) & 
(test_predictions_ce != test_labels)) / np.sum(test_labels == 1)) * 100 
total_error_percentage_ce = ((np.sum(test_predictions_ce != 
test_labels)) / len(test_labels)) * 100 
print("############## CROSS-ENTROPY NETWORK ##################") 
print(f"error_percentage for 0:  {error_percentage_ce_0}%") 
print(f"error_percentage for 8:  {error_percentage_ce_8}%") 
print(f"total_error_percentage: {total_error_percentage_ce}%") 
 
# Initialize neural network parameters for Exponential Loss Network with Adam Optimizer 
np.random.seed(0) 
input_size = 784 
hidden_size = 300 
output_size = 1 
 
# Weights initialization for Exponential Loss Network 
w1_exp = np.random.normal(0, np.sqrt(1 / (input_size + hidden_size)), 
(input_size, hidden_size)) 
b1_exp = np.zeros(hidden_size) 
w2_exp = np.random.normal(0, np.sqrt(1 / (hidden_size + output_size)), 
(hidden_size, output_size)) 
b2_exp = np.zeros(output_size) 
 
# Adam optimizer variables 
t_exp = 0 
m_w1_exp = np.zeros_like(w1_exp) 
v_w1_exp = np.zeros_like(w1_exp) 
m_b1_exp = np.zeros_like(b1_exp) 
v_b1_exp = np.zeros_like(b1_exp) 
m_w2_exp = np.zeros_like(w2_exp) 
v_w2_exp = np.zeros_like(w2_exp) 
m_b2_exp = np.zeros_like(b2_exp) 
v_b2_exp = np.zeros_like(b2_exp) 
 
beta1 = 0.9 
beta2 = 0.999 
epsilon = 1e-8 
# Backward pass for Exponential Loss Network with Adam Optimizer 
def backward_pass_exp(x, y, z1, a1, z2, a2, learning_rate): 
    global w1_exp, b1_exp, w2_exp, b2_exp, t_exp, m_w1_exp, v_w1_exp, m_b1_exp, v_b1_exp, m_w2_exp, v_w2_exp, m_b2_exp, v_b2_exp 
     
    # Output layer error 
    dz2 = -0.5 * y * np.exp(-0.5 * y * a2) 
    dw2 = np.dot(a1.T, dz2) / x.shape[0] 
    db2 = np.sum(dz2, axis=0) / x.shape[0]   
    # Hidden layer error 
    da1 = np.dot(dz2, w2_exp.T) 
    dz1 = da1 * relu_derivative(a1) 
    dw1 = np.dot(x.T, dz1) / x.shape[0] 
    db1 = np.sum(dz1, axis=0) / x.shape[0]   
    # Adam optimizer update 
    t_exp += 1 
    # Update first moment estimates 
    m_w1_exp = beta1 * m_w1_exp + (1 - beta1) * dw1 
    m_b1_exp = beta1 * m_b1_exp + (1 - beta1) * db1 
    m_w2_exp = beta1 * m_w2_exp + (1 - beta1) * dw2 
    m_b2_exp = beta1 * m_b2_exp + (1 - beta1) * db2 
    # Update second moment estimates 
    v_w1_exp = beta2 * v_w1_exp + (1 - beta2) * (dw1 ** 2) 
    v_b1_exp = beta2 * v_b1_exp + (1 - beta2) * (db1 ** 2) 
    v_w2_exp = beta2 * v_w2_exp + (1 - beta2) * (dw2 ** 2) 
    v_b2_exp = beta2 * v_b2_exp + (1 - beta2) * (db2 ** 2) 
    # Compute bias-corrected moment estimates 
    m_w1_exp_hat = m_w1_exp / (1 - beta1 ** t_exp) 
    m_b1_exp_hat = m_b1_exp / (1 - beta1 ** t_exp) 
    v_w1_exp_hat = v_w1_exp / (1 - beta2 ** t_exp) 
    v_b1_exp_hat = v_b1_exp / (1 - beta2 ** t_exp) 
    m_w2_exp_hat = m_w2_exp / (1 - beta1 ** t_exp) 
    m_b2_exp_hat = m_b2_exp / (1 - beta1 ** t_exp) 
    v_w2_exp_hat = v_w2_exp / (1 - beta2 ** t_exp) 
    v_b2_exp_hat = v_b2_exp / (1 - beta2 ** t_exp) 
    # Update parameters 
    w1_exp -= learning_rate * m_w1_exp_hat / (np.sqrt(v_w1_exp_hat) + epsilon) 
    b1_exp -= learning_rate * m_b1_exp_hat / (np.sqrt(v_b1_exp_hat) + epsilon) 
    w2_exp -= learning_rate * m_w2_exp_hat / (np.sqrt(v_w2_exp_hat) + epsilon) 
    b2_exp -= learning_rate * m_b2_exp_hat / (np.sqrt(v_b2_exp_hat) + epsilon) 
# Training the Exponential Loss Network 
epochs = 50 
learning_rate = 0.001 
batch_size = 32 
# Training the Exponential Loss Network 
# Convert labels to {-1, 1} for Exponential Loss Network 
y_train_exp = np.where(train_labels == 1, 1, -1) 
def forward_pass_exp(x): 
    z1 = np.dot(x, w1_exp) + b1_exp 
    a1 = relu(z1) 
    z2 = np.dot(a1, w2_exp) + b2_exp 
    a2 = z2  # No activation in output layer 
    return z1, a1, z2, a2 
for epoch in range(epochs): 
    # Exponential Loss Network Training 
    for i in range(0, train_images.shape[0], batch_size): 
        x_batch = train_images[i:i+batch_size] 
        y_batch = y_train_exp[i:i+batch_size].reshape(-1, 1) 
        # Forward pass 
        z1, a1, z2, a2 = forward_pass_exp(x_batch)        
        # Compute cost 
        cost = np.mean(np.exp(-0.5 * y_batch * a2)) 
        cost_history_exp.append(cost)       
        # Backward pass 
        backward_pass_exp(x_batch, y_batch, z1, a1, z2, a2, learning_rate) 
    if epoch% s_window==0: 
        smooth_exp.append(np.mean(cost_history_exp[-s_window:])) 
    # Compute training accuracy for Exponential Loss Network 
    _, _, _, train_output = forward_pass_exp(train_images) 
    train_predictions = (train_output > 0).astype(int).flatten() 
    train_predictions = np.where(train_predictions == 0, -1, 1) 
    train_accuracy = accuracy_score(y_train_exp, train_predictions) 
    'print(f"Epoch {epoch + 1}/{epochs} - Exponential Loss Training Accuracy: {train_accuracy * 100:.2f}%")' 
 
# Evaluate the Exponential Loss Network on test data 
_, _, _, test_output = forward_pass_exp(test_images) 
test_predictions_exp = (test_output > 0).astype(int).flatten() 
test_predictions_exp = np.where(test_predictions_exp == 0, -1, 1) 
test_accuracy_exp = accuracy_score(np.where(test_labels == 1, 1, -1), test_predictions_exp) 
print(f"Exponential Network Error: {(1-test_accuracy_exp) * 100:.2f}%") 
#display_misclassified_images(images, true_labels, test_predictions_exp, "disapointment", num_images=10) 
# Error percentage for Exponential Loss Network 
error_percentage_exp_0 = (np.sum((test_labels == 0) & (test_predictions_exp != np.where(test_labels == 0, -1, 1))) / np.sum(test_labels == 0)) * 100 
error_percentage_exp_8 = (np.sum((test_labels == 1) & (test_predictions_exp != np.where(test_labels == 1, 1, -1))) / np.sum(test_labels == 1)) * 100 
total_error_percentage_exp = ((np.sum(test_predictions_exp != np.where(test_labels == 1, 1, -1))) / len(test_labels)) * 100 
print("################# EXPONENTIAL ###############") 
print(f"error_percentage for 0:  {error_percentage_exp_0}%") 
print(f"error_percentage for 8:  {error_percentage_exp_8}%") 
print(f"total_error_percentage:  {total_error_percentage_exp}%") 
# Plot the cost histories for both networks 
plt.plot(smooth_ce, label='Cross-Entropy Loss') 
plt.plot(smooth_exp, label='Exponential Loss') 
plt.xlabel(f'Iterations/{s_window}') 
plt.ylabel('Loss') 
plt.legend() 
plt.title('Loss Comparison Between Cross-Entropy and Exponential Loss Networks') 
plt.show()