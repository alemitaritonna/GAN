import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import array_to_img, img_to_array

from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose

from keras.layers import Conv2D, Flatten, Dropout, LeakyReLU

from keras.optimizers import Adam


def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128 * 7 * 7, input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(3, (7,7), activation='sigmoid', padding='same'))
    return model 



def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=(64,64,3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return gan


def train_gan(generator, discriminator, gan, dataset, latent_dim, n_epochs=100, n_batch=128):
    #X_train = dataset.astype('float32')
    #X_train = (X_train - 127.5) / 127.5
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)
        real_images = X_train[np.random.randint(0, X_train.shape[0], half_batch)]
        X = np.concatenate([real_images, fake_images])
        y = np.zeros((n_batch, 1))
        y[:half_batch] = 1
        d_loss = discriminator.train_on_batch(X, y)
        noise = np.random.normal(0, 1, (n_batch, latent_dim))
        y = np.ones((n_batch, 1))
        g_loss = gan.train_on_batch(noise, y)
        print(f'Epoch: {i+1}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')




if __name__ == '__main__':

    # Directorio que contiene las imágenes de malezas
    data_dir = 'images/'

    # Tamaño de las imágenes que se utilizarán en el modelo
    img_size = (64, 64)

    # Lista que contendrá todas las imágenes de malezas
    weed_images = []

    # Recorrer el directorio y cargar las imágenes de malezas en la lista
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img = Image.open(os.path.join(data_dir, filename))
            img = img.resize(img_size)
            img = img_to_array(img)
            weed_images.append(img)

    # Convertir la lista de imágenes en un arreglo numpy
    X_train = np.asarray(weed_images)

    # Escalar los valores de los pixeles entre -1 y 1
    X_train = (X_train.astype('float32') - 127.5) / 127.5

    latent_dim = 100
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)

    train_gan(generator, discriminator, gan, X_train, latent_dim)
