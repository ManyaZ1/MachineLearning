% Load the data21.mat file 
clc; 
close all; 
clear all; 
load('data21.mat'); 
A1 = A_1; % 128 x 10 
A2 = A_2; % 784 x 128 
B1 = B_1; % 128 x 1 
B2 = B_2; % 784 x 1 
% Number of samples to generate 
n_samples = 100; 
% Initialize a container for generated images 
images = zeros(28, 28, n_samples); 
% Generate 100 samples of Z and corresponding images 
for i = 1:n_samples 
% Generate a random Z (10x1 vector, i.i.d. Gaussian) 
Z = randn(10, 1); 
% Compute W1 = A1 * Z + B1 
W1 = A1 * Z + B1; 
% Apply ReLU: Z1 = max(W1, 0) 
Z1 = max(W1, 0); 
% Compute W2 = A2 * Z1 + B2 
W2 = A2 * Z1 + B2; 
% Apply sigmoid activation: X = 1 / (1 + exp(-W2)) 
X = 1 ./ (1 + exp(W2)); 
% Reshape X into a 28x28 image 
images(:, :, i) = reshape(X, 28, 28); 
end 
% Create a 10x10 grid of images 
grid_size = 10; 
image_grid = zeros(28 * grid_size, 28 * grid_size); 
for row = 1:grid_size 
for col = 1:grid_size 
idx = (row - 1) * grid_size + col; 
image_grid((row - 1) * 28 + 1:row * 28, (col - 1) * 28 + 1:col * 
28) = images(:, :, idx); 
end 
end 
% Display the grid of images 
figure; 
imshow(image_grid, []); 
title('Generated Handwritten 8s');