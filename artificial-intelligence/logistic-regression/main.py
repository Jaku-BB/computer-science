from numpy import random, dot, exp, clip, log, where, unique, zeros
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plot


class LogisticRegression:
    EPSILON = 1e-5

    def __init__(self, learning_rate, training_loop_count, random_state):
        self.learning_rate = learning_rate
        self.training_loop_count = training_loop_count
        self.random_state = random_state
        self.weights = None
        self.cost = None

    def fit(self, x, y):
        self.weights = random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.cost = []

        for _ in range(self.training_loop_count):
            output = self.activation(x)
            error = y - output

            self.weights[1:] += self.learning_rate * x.T.dot(error)
            self.weights[0] += self.learning_rate * error.sum()

            self.cost.append(-y.dot(log(output + self.EPSILON)) - ((1 - y).dot(log(1 - output + self.EPSILON))))

        return self

    def net_input(self, x):
        return dot(x, self.weights[1:]) + self.weights[0]

    def activation(self, x):
        return 1 / (1 + exp(-clip(self.net_input(x), -250, 250)))

    def predict(self, x):
        return where(self.net_input(x) >= 0, 1, 0)


class MultiClassLogisticRegression:
    def __init__(self, learning_rate, training_loop_count, random_state):
        self.learning_rate = learning_rate
        self.training_loop_count = training_loop_count
        self.random_state = random_state
        self.classes = None
        self.classifiers = {}

    def fit(self, x, y):
        self.classes = unique(y)

        for class_label in self.classes:
            binary_label = where(y == class_label, 1, 0)
            self.classifiers[class_label] = LogisticRegression(self.learning_rate, self.training_loop_count, self.random_state).fit(x, binary_label)

        return self

    def predict(self, x):
        predictions = zeros((x.shape[0], len(self.classes)))

        for index, class_label in enumerate(self.classes):
            predictions[:, index] = self.classifiers[class_label].activation(x)

        return self.classes[predictions.argmax(axis=1)]


LEARNING_RATE = 0.05
TRAINING_LOOP_COUNT = 1000
RANDOM_STATE = 1


def main():
    x, y = load_iris(return_X_y=True)
    x = x[:, [2, 3]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

    multi_class_perceptron = MultiClassLogisticRegression(LEARNING_RATE, TRAINING_LOOP_COUNT, RANDOM_STATE).fit(x_train,
                                                                                                                y_train)
    plot_decision_regions(x_test, y_test, clf=multi_class_perceptron)

    plot.legend()
    plot.show()


if __name__ == '__main__':
    main()
