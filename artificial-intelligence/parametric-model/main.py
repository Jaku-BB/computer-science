from numpy import loadtxt, linalg, hstack, ones, sum, argsort
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot


def calculate_model(x, y, degree):
    x = hstack([x ** degree1 for degree1 in range(degree, 0, -1)] + [ones(x.shape)])
    parameters = linalg.pinv(x) @ y
    error = sum((y - x @ parameters) ** 2)

    return parameters, error / error.size


def main():
    data = loadtxt('data.txt')
    x_train, x_test, y_train, y_test = train_test_split(data[:, [0]], data[:, [1]], test_size=0.2, random_state=42)

    linear_model_parameters, linear_model_error = calculate_model(x_train, y_train, 1)
    cubic_model_parameters, cubic_model_error = calculate_model(x_train, y_train, 3)

    print(f'Średni błąd kwadratowy (regresja liniowa): {linear_model_error}')
    print(f'Średni błąd kwadratowy (regresja sześcienna): {cubic_model_error}')

    plot.figure(figsize=(8, 6))
    plot.scatter(x_train, y_train, color='blue', label='Dane treningowe')

    plot.plot(x_train, linear_model_parameters[1] + linear_model_parameters[0] * x_train, color='red', linewidth=2,
              label='Regresja liniowa')

    x_train = x_train[argsort(x_train.flatten())]
    plot.plot(x_train,
              cubic_model_parameters[0] * x_train ** 3 +
              cubic_model_parameters[1] * x_train ** 2 +
              cubic_model_parameters[2] * x_train +
              cubic_model_parameters[3], color='green', linewidth=2, label='Regresja sześcienna')

    plot.title('Porównanie modeli parametrycznych')
    plot.legend()
    plot.tight_layout()
    plot.show()


if __name__ == '__main__':
    main()
