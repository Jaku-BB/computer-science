from keras import Sequential, layers
from keras.datasets import cifar10
from keras.utils import to_categorical
from numpy import isin, where


def get_convolution_classifier(layer_amount):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

    for _ in range(layer_amount - 1):
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    class_indices = [3, 5]

    train_filter = isin(y_train, class_indices)
    test_filter = isin(y_test, class_indices)

    x_train, y_train = x_train[train_filter[:, 0]], y_train[train_filter]
    x_test, y_test = x_test[test_filter[:, 0]], y_test[test_filter]

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = where(y_train == class_indices[0], 1, 0)
    y_test = where(y_test == class_indices[0], 1, 0)

    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    for layer in [1, 2, 3]:
        model = get_convolution_classifier(layer)
        model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

        accuracy = model.evaluate(x_test, y_test)[1]
        print(f'Classifier accuracy (layers: {layer}): {accuracy}')


if __name__ == '__main__':
    main()
