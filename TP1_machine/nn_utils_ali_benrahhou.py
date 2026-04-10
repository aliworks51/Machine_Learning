import numpy as np

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

# ReLU
def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return np.where(x > 0, 1, 0)

# Tanh
def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x)**2

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def d_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)
