import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import cv2

# Configuramos los parametros de entrada
LATENT_DIM = 150
IMAGE_SHAPE = (128, 128, 3)


# Cargamos las imágenes de malezas
def load_weed_images(directory):
    images = []
    i = 0
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
            images.append(img)
            i = i + 1
            print("image #", i)
    return np.array(images)


# Normalizamos las imágenes
def normalize_images(images):
    return (images - 127.5) / 127.5


# Creamos el generador
def build_generator(latent_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256 * 16 * 16, activation='relu', input_dim=latent_dim),
        tf.keras.layers.Reshape((16, 16, 256)),
        tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(3, (3,3), activation='tanh', padding='same')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

# Creamos el discriminador
def build_discriminator(img_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=img_shape),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


# Creamos el modelo GAN
def build_gan(generator, discriminator):    
    model = tf.keras.Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5) ,loss='binary_crossentropy')
    #print("COMPILO!!!")
    
    return model



# Guardamos las imágenes generadas
def save_images(generator, epoch):
    os.makedirs('images/gan', exist_ok=True)
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, LATENT_DIM))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs = gen_imgs * 255
    for i in range(r):
        for j in range(c):
            index = i * c + j
            img = gen_imgs[index].astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"images/gan/{epoch}_{index}.jpg", img)


# Entrenamos la GAN
def train_gan(gan, generator, discriminator, images, latent_dim, epochs=10, batch_size=128):
    
    batch_count = images.shape[0] // batch_size
    
    for i in range(epochs):
        for j in range(batch_count):
            # Obtenemos imágenes reales
            X_real = images[j * batch_size : (j+1) * batch_size]
            # Generamos imágenes falsas
            X_fake = generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))
            # Entrenamos el discriminador con imágenes reales y falsas
            y_real = np.ones((batch_size, 1))
            y_fake = np.zeros((batch_size, 1))
            discriminator.trainable = True
            discriminator.train_on_batch(X_real, y_real)
            discriminator.train_on_batch(X_fake, y_fake)
            discriminator.trainable = False
            
            # Entrenamos la GAN con imágenes falsas
            gan_loss = gan.train_on_batch(np.random.normal(0, 1, (batch_size, latent_dim)), y_real)
            print(f"Epoch: {i+1}/{epochs}, Batch: {j+1}/{batch_count}, GAN loss: {gan_loss}")

        # Generamos imágenes para visualizar el progreso
        save_images(generator, epoch=i+1)



if __name__ == '__main__':

    # Directorio que contiene las imágenes de malezas
    data_dir = 'images/dataset/broadleaf'

    # Cargamos las imágenes de malezas y las normalizamos
    images = load_weed_images(data_dir)

    images = normalize_images(images)

    # Construimos y compilamos la GAN
    generator = build_generator(LATENT_DIM)
    discriminator = build_discriminator(IMAGE_SHAPE)
    gan = build_gan(generator, discriminator)

    print("FIN!!!")

    # Entrenamos la GAN
    train_gan(gan, generator, discriminator, images, LATENT_DIM, epochs=100, batch_size=32)