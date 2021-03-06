import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.python.compiler.mlcompute import mlcompute

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

mlcompute.set_mlc_device(device_name='gpu')
tf.config.run_functions_eagerly(False)

x_train = mnist.load_data()[0][0]
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
print(x_train.shape)

input_img = keras.Input(shape=(784,))
encoded = layers.Dense(32, activation='relu')(input_img)
decoded = layers.Dense(784, activation='sigmoid')(encoded)
# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

autoencoder.save('auto-encoder/autoencoder.h5') 