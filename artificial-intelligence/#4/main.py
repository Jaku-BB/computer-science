from sklearn.model_selection import train_test_split
from numpy import random, tanh, power, sum, square, zeros, where, maximum, loadtxt
import matplotlib.pyplot as plot


def initialize_parameters(input_size, hidden_size, output_size):
    random.seed(0)
    weights_input_hidden = random.randn(hidden_size, input_size) * 0.01
    bias_input_hidden = zeros((hidden_size, 1))
    weights_hidden_output = random.randn(output_size, hidden_size) * 0.01
    bias_hidden_output = zeros((output_size, 1))
    return weights_input_hidden, bias_input_hidden, weights_hidden_output, bias_hidden_output


def tanh_activation(z):
    return tanh(z)


def tanh_derivative(z):
    return 1 - power(tanh(z), 2)


def relu_activation(z):
    return maximum(0, z)


def relu_derivative(z):
    return where(z > 0, 1, 0)


def forward_propagation(x, parameters, activation_fn):
    weights_input_hidden, bias_input_hidden, weights_hidden_output, bias_hidden_output = parameters
    z_input_hidden = weights_input_hidden @ x + bias_input_hidden
    a_input_hidden = activation_fn(z_input_hidden)
    z_hidden_output = weights_hidden_output @ a_input_hidden + bias_hidden_output
    a_output = z_hidden_output
    return a_output, (
        z_input_hidden, a_input_hidden, weights_input_hidden, bias_input_hidden, z_hidden_output, a_output,
        weights_hidden_output, bias_hidden_output)


def compute_loss(a_output, y):
    m = y.shape[1]
    return sum(square(a_output - y)) / m


def backward_propagation(x, y, cache, activation_derivative_fn):
    m = x.shape[1]
    (z_input_hidden, a_input_hidden, weights_input_hidden, bias_input_hidden, z_hidden_output, a_output,
     weights_hidden_output, bias_hidden_output) = cache

    dz_output = 2 * (a_output - y) / m
    dweights_hidden_output = dz_output @ a_input_hidden.T
    dbias_hidden_output = sum(dz_output, axis=1, keepdims=True)

    dz_hidden = weights_hidden_output.T @ dz_output * activation_derivative_fn(z_input_hidden)
    dweights_input_hidden = dz_hidden @ x.T
    dbias_input_hidden = sum(dz_hidden, axis=1, keepdims=True)

    return {"dweights_input_hidden": dweights_input_hidden, "dbias_input_hidden": dbias_input_hidden,
            "dweights_hidden_output": dweights_hidden_output, "dbias_hidden_output": dbias_hidden_output}


def update_parameters(parameters, gradients, learning_rate):
    weights_input_hidden, bias_input_hidden, weights_hidden_output, bias_hidden_output = parameters
    dweights_input_hidden = gradients["dweights_input_hidden"]
    dbias_input_hidden = gradients["dbias_input_hidden"]
    dweights_hidden_output = gradients["dweights_hidden_output"]
    dbias_hidden_output = gradients["dbias_hidden_output"]

    weights_input_hidden -= learning_rate * dweights_input_hidden
    bias_input_hidden -= learning_rate * dbias_input_hidden
    weights_hidden_output -= learning_rate * dweights_hidden_output
    bias_hidden_output -= learning_rate * dbias_hidden_output

    return weights_input_hidden, bias_input_hidden, weights_hidden_output, bias_hidden_output


def train_neural_network(x, y, activation_fn, activation_derivative_fn, hidden_size=10, num_epochs=1000,
                         learning_rate=0.01):
    input_size = x.shape[0]
    output_size = 1
    parameters = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(num_epochs):
        a_output, cache = forward_propagation(x, parameters, activation_fn)
        gradients = backward_propagation(x, y, cache, activation_derivative_fn)
        parameters = update_parameters(parameters, gradients, learning_rate)

    return parameters


def predict(x, parameters, activation_fn):
    a_output, _ = forward_propagation(x, parameters, activation_fn)
    return a_output


def main():
    data = loadtxt('data.txt')
    x_train, x_test, y_train, y_test = train_test_split(data[:, 0], data[:, 1], test_size=0.4, random_state=42)

    parameters_tanh = train_neural_network(x_train.reshape(1, -1), y_train.reshape(1, -1),
                                           activation_fn=tanh_activation, activation_derivative_fn=tanh_derivative,
                                        )

    parameters_relu = train_neural_network(x_train.reshape(1, -1), y_train.reshape(1, -1),
                                           activation_fn=relu_activation, activation_derivative_fn=relu_derivative)

    y_prediction_tanh = predict(x_test.reshape(1, -1), parameters_tanh, activation_fn=tanh_activation)
    y_prediction_relu = predict(x_test.reshape(1, -1), parameters_relu, activation_fn=relu_activation)

    plot.scatter(x_test, y_test, label="Actual")
    plot.scatter(x_test, y_prediction_tanh.flatten(), label="Predicted (tanh)")
    plot.scatter(x_test, y_prediction_relu.flatten(), label="Predicted (ReLU)")
    plot.legend()
    plot.show()


if __name__ == "__main__":
    main()
