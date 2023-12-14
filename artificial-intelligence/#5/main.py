import numpy as np
from sklearn.model_selection import train_test_split


def relu(x):
    return np.maximum(0, x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def batch_train(x, y, activation, derivative, learning_rate=0.01, epochs=1000, hidden_size=10):
    input_size, output_size = x.shape[1], y.shape[1]
    weights_input_hidden, weights_hidden_output = np.random.randn(input_size, hidden_size), np.random.randn(hidden_size,
                                                                                                            output_size)

    for epoch in range(epochs):
        hidden_output = activation(x @ weights_input_hidden)
        final_output = activation(hidden_output @ weights_hidden_output)
        error = y - final_output

        delta_output = error * derivative(final_output)
        delta_hidden = delta_output @ weights_hidden_output.T * derivative(hidden_output)

        weights_hidden_output += hidden_output.T @ delta_output * learning_rate
        weights_input_hidden += x.T @ delta_hidden * learning_rate

    return weights_input_hidden, weights_hidden_output


def evaluate_model(x, y, weights_input_hidden, weights_hidden_output, activation):
    predictions = activation(x @ weights_input_hidden @ weights_hidden_output)
    mse = np.mean((predictions - y) ** 2)
    return mse


def main():
    data = np.loadtxt('data.txt')
    x, y = data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    weights_input_hidden_tanh, weights_hidden_output_tanh = batch_train(x_train, y_train, np.tanh, tanh_derivative)
    mse_tanh_test = evaluate_model(x_test, y_test, weights_input_hidden_tanh, weights_hidden_output_tanh, np.tanh)
    mse_tanh_train = evaluate_model(x_train, y_train, weights_input_hidden_tanh, weights_hidden_output_tanh, np.tanh)

    print(f"MSE for Tanh (test): {mse_tanh_test}")
    print(f"MSE for Tanh (train): {mse_tanh_train}")

    weights_input_hidden_relu, weights_hidden_output_relu = batch_train(x_train, y_train, relu, relu_derivative)
    mse_relu_test = evaluate_model(x_test, y_test, weights_input_hidden_relu, weights_hidden_output_relu, relu)
    mse_relu_train = evaluate_model(x_train, y_train, weights_input_hidden_relu, weights_hidden_output_relu, relu)

    print(f"MSE for ReLU (test): {mse_relu_test}")
    print(f"MSE for ReLU (train): {mse_relu_train}")


if __name__ == '__main__':
    main()
