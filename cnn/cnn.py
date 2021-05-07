import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
        layers.InputLayer(input_shape = (32, 32, 3)),
        layers.Conv2D(6, 5, activation = 'relu'),
        layers.MaxPooling2D(pool_size = (2,2)),
        layers.Conv2D(16, 5, activation = 'relu'),
        layers.MaxPooling2D(pool_size = (2,2)),
        layers.Flatten(),
        layers.Dense(120, activation = 'relu'),
        layers.Dense(84, activation = 'relu'),
        layers.Dense(10, activation = 'relu'),
    ]
)
model.summary()
keras.utils.plot_model(model)