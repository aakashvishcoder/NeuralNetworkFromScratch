import numpy as np
import math
# layer architecture (layer class & base architecture)
class Dense:
    def __init__(self, neurons: int, input_size: int):
        self.neurons = neurons
        self.input_size = input_size
        self.weights = np.random.randn(self.neurons, self.input_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros(self.neurons)
    def forward(self, input):
        return np.dot(self.weights, input) + self.bias

# activation (relu, softmax, sigmoid)
class Relu:
    def forward(self, X):
        X[X < 0] = 0
        return X

class LeakyRelu:
    def __init__(self, m):
        self.m = m
    def forward(self, X):
        X[X < self.m] = self.m
        return X

class Softmax:
    def forward(self, X):
        exp_shifted = np.exp(X - np.max(X, axis=0, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)

class Sigmoid:
    def forward(self, X):
        return np.round((1) / (1 + np.exp(np.negative(X))), decimals=4)

class Tanh:
    def forward(self, X):
        return np.tanh(X)

# loss (MSE, crossentropy, binarycrossentropy, RMSE)
def MSE(y_pred, y_actual):
    return np.round((1 / len(y_pred)) * (np.sum(np.power(np.subtract(np.array(y_pred), np.array(y_actual)), 2))), decimals=4)

def RMSE(y_pred, y_actual):
    return np.round(math.sqrt((sum(np.power(np.array(y_pred) - np.array(y_actual), 2))) / (len(y_pred))), decimals=4)

def BCE(output, x):
    output = np.clip(output, 1e-10, 1 - 1e-10)
    return np.sum(x * np.log(output) + (1 - x) * np.log(1 - output)) / (-len(x))

def CE(output, x):
    output = np.clip(output, 1e-10, 1 - 1e-10)
    return np.round(-np.sum(x * np.log(output)) / len(x))

def GradientDescent(weight, loss, y_true, learning_rate=1e-5):
    loss_plus_epsilon = loss(weight + learning_rate, y_true)
    loss_minus_epsilon = loss(weight - learning_rate, y_true)
    gradient_weight = (loss_plus_epsilon - loss_minus_epsilon) / (2 * learning_rate)
    return gradient_weight

class Model:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, inp):
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def g_predict(self, inp):
        y_pred = []
        for ele in inp:
            y_pred.append(self.predict(ele))
        y_pred = np.array(y_pred)
        return y_pred

    def evaluate(self, X, y):
        acc = 0
        y_pred = self.g_predict(X)
        for idx in range(len(y_pred)):
            if (y_pred[idx] == y[idx]).all():
                acc += 1
        return acc / len(X)

    def fit(self, X, y, n_epochs: int, loss_fn, learning_rate=1e-5):
        for epoch in range(n_epochs):
            for i in range(len(X)):
                # Forward pass
                activations = [X[i]]
                for layer in self.layers:
                    activations.append(layer.forward(activations[-1]))

                # Compute loss
                loss = loss_fn(activations[-1], y[i])

                # Backward pass
                grad = activations[-1] - y[i]
                for j in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[j]
                    if isinstance(layer, Dense):
                        grad_w = np.outer(grad, activations[j])
                        grad_b = grad
                        grad = np.dot(layer.weights.T, grad)
                        layer.weights -= learning_rate * grad_w
                        layer.bias -= learning_rate * grad_b
                    elif isinstance(layer, Relu):
                        grad = grad * (activations[j] > 0)
                    elif isinstance(layer, Softmax):
                        pass  # No gradient for Softmax in this simple implementation

            # Calculate and print loss for the epoch
            y_pred = self.g_predict(X)
            calc_loss = loss_fn(y_pred, y)
            print("EPOCH: " + str(epoch + 1) + " has a loss of  " + str(calc_loss.squeeze()))


# model = Model(layers=[Dense(16, 4), Relu(), Dense(32, 16), Relu(), Dense(3, 32)])
