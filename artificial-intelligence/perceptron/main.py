from numpy import zeros, dot, where, unique
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plot


class Perceptron:
    def __init__(self, learning_rate, training_loop_count):
        self.learning_rate = learning_rate
        self.training_loop_count = training_loop_count
        self.weights = None

    def fit(self, x, y):
        self.weights = zeros(1 + x.shape[1])

        for _ in range(self.training_loop_count):
            self.weights[1:] += self.learning_rate * (y - self.predict(x)).dot(x)
            self.weights[0] += self.learning_rate * (y - self.predict(x)).sum()

        return self

    def net_input(self, x):
        return dot(x, self.weights[1:]) + self.weights[0]

    def predict(self, x):
        return where(self.net_input(x) >= 0, 1, -1)


class MultiClassPerceptron:

    def __init__(self, learning_rate, training_loop_count):
        self.learning_rate = learning_rate
        self.training_loop_count = training_loop_count
        self.classes = None
        self.classifiers = {}

    def fit(self, x, y):
        self.classes = unique(y)

        for class_label in self.classes:
            binary_label = where(y == class_label, 1, -1)
            classifier = Perceptron(self.learning_rate, self.training_loop_count).fit(x, binary_label)
            self.classifiers[class_label] = classifier

        return self

    def predict(self, x):
        predictions = zeros((len(x), len(self.classes)))

        for index, class_label in enumerate(self.classes):
            predictions[:, index] = self.classifiers[class_label].net_input(x)

        return self.classes[predictions.argmax(axis=1)]


LEARNING_RATE = 0.001
TRAINING_LOOP_COUNT = 200


def main():
    x, y = load_iris(return_X_y=True)
    x = x[:, [2, 3]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

    multi_class_perceptron = MultiClassPerceptron(LEARNING_RATE, TRAINING_LOOP_COUNT).fit(x_train, y_train)
    plot_decision_regions(x_test, y_test, clf=multi_class_perceptron)

    plot.legend()
    plot.show()


if __name__ == '__main__':
    main()

