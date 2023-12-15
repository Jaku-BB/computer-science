from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model, load_model
from keras.datasets import mnist
from numpy import reshape, random, clip, where
import matplotlib.pyplot as plot


def get_encoder():
    input_image = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_image, decoded)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model


def add_noise(x_test, noise_type, noise_factor=0.6):
    if noise_type == 'normal':
        noise = noise_factor * random.normal(loc=0, scale=1, size=x_test.shape)
    elif noise_type == 'uniform':
        noise = noise_factor * random.uniform(low=-1, high=1, size=x_test.shape)
    elif noise_type == 'impulse':
        noise = random.choice([0, 1, 2], size=x_test.shape,
                              p=[1 - noise_factor, noise_factor / 2, noise_factor / 2])
        noise = where(noise == 1, 1, 0)
    elif noise_type == 'salt_and_pepper':
        salt_probability = noise_factor / 2.0
        pepper_probability = noise_factor / 2.0

        noisy_pixels = random.choice([0, 1, 2], size=x_test.shape,
                                     p=[1 - salt_probability - pepper_probability, salt_probability,
                                        pepper_probability])

        noise = where(noisy_pixels == 1, 1.0, 0.0)
        noise += where(noisy_pixels == 2, 0.0, -1.0)
    else:
        noise = noise_factor * random.normal(loc=0, scale=1, size=x_test.shape)

    return clip(x_test + noise, 0, 1)


def plot_result(title, x_test, x_test_noisy, decoded_images):
    plot.title(title)
    plot.axis('off')

    for index in range(10):
        plot.subplot(3, 10, index + 1)
        plot.imshow(x_test[index].reshape(28, 28))
        plot.gray()
        plot.axis('off')

        plot.subplot(3, 10, index + 1 + 10)
        plot.imshow(x_test_noisy[index].reshape(28, 28))
        plot.gray()
        plot.axis('off')

        plot.subplot(3, 10, index + 1 + 20)
        plot.imshow(decoded_images[index].reshape(28, 28))
        plot.gray()
        plot.axis('off')

    plot.show()


def main():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = reshape(x_test, (len(x_test), 28, 28, 1))

    encoder = get_encoder()
    encoder.fit(x_train, x_train, epochs=30, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

    encoder.save('encoder_model.h5')

    encoder_model = load_model('encoder_model.h5')

    for noise_type in ['normal', 'uniform', 'impulse', 'salt_and_pepper']:
        x_test_noisy = add_noise(x_test, noise_type)
        decoded_images = encoder_model.predict(x_test_noisy)

        plot_result(noise_type.capitalize() if noise_type != 'salt_and_pepper' else 'Salt and Pepper',
                    x_test, x_test_noisy, decoded_images)


if __name__ == '__main__':
    main()
