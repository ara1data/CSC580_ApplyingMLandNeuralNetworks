# Module 8 Portfolio Project
# Option 1: Working with a Generative Adversarial Network

# Alexander Reichart-Anderson
# MS in AI and ML, Colorado State University Global
# CSC580-1: Applying Machine Learning & Neural Networks - Capstone
# Dr. Joseph Issa
# July 6, 2025

# Step 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

# Step 2: Load and preprocess the CIFAR-10 data (class 8: ship)
(X, y), (_, _) = keras.datasets.cifar10.load_data()
X = X[y.flatten() == 8]
image_shape = (32, 32, 3)
latent_dimensions = 100
X = (X / 127.5) - 1.  # Normalize to [-1, 1]

# Step 3: Build the Generator
def build_generator():
    model = Sequential()
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dimensions))
    model.add(Reshape((8, 8, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    noise = Input(shape=(latent_dimensions,))
    image = model(noise)
    return Model(noise, image)

# Step 4: Build the Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    image = Input(shape=image_shape)
    validity = model(image)
    return Model(image, validity)

# Step 5: Utility function to display generated images
def display_images(generator, latent_dimensions, epoch, title):
    r, c = 4, 4
    noise = np.random.normal(0, 1, (r * c, latent_dimensions))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]
    fig, axs = plt.subplots(r, c, figsize=(6,6))
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(generated_images[count])
            axs[i, j].axis('off')
            count += 1
    plt.suptitle(f"{title} (Epoch {epoch})")
    plt.tight_layout()
    plt.show()
    plt.close()

# Step 6: Build and compile the GAN
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
discriminator.trainable = False
generator = build_generator()
z = Input(shape=(latent_dimensions,))
img = generator(z)
valid = discriminator(img)
combined_network = Model(z, valid)
combined_network.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Step 7: Training loop
num_epochs = 15000
batch_size = 32
display_interval = 2500

# Adversarial ground truths
valid = np.ones((batch_size, 1))
valid += 0.05 * np.random.random(valid.shape)
fake = np.zeros((batch_size, 1))
fake += 0.05 * np.random.random(fake.shape)

# Store images for first and last epoch
first_epoch_images = None
last_epoch_images = None

for epoch in range(num_epochs + 1):
    # Train Discriminator
    idx = np.random.randint(0, X.shape[0], batch_size)
    imgs = X[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    gen_imgs = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # Train Generator
    g_loss = combined_network.train_on_batch(noise, valid)
    # Save images at first and last epoch
    if epoch == 0:
        first_epoch_images = generator.predict(np.random.normal(0, 1, (16, latent_dimensions)))
    if epoch == num_epochs:
        last_epoch_images = generator.predict(np.random.normal(0, 1, (16, latent_dimensions)))
    # Display progress
    if epoch % display_interval == 0:
        print(f"Epoch {epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
        display_images(generator, latent_dimensions, epoch, "Generated Images")

# Step 8: Plot images from the first epoch
def plot_epoch_images(images, title):
    images = 0.5 * images + 0.5
    fig, axs = plt.subplots(4, 4, figsize=(6,6))
    count = 0
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(images[count])
            axs[i, j].axis('off')
            count += 1
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close()

plot_epoch_images(first_epoch_images, "Generated Images - First Epoch")
plot_epoch_images(last_epoch_images, "Generated Images - Last Epoch")
