from vonenet.layers import VOneNetLayer
import tensorflow as tf
from tensorflow.keras import layers


input_shape = (28, 28, 1)
num_classes = 10

model = tf.keras.Sequential(
    [
        layers.Input(shape=input_shape),
        VOneNetLayer(shape=input_shape,
                     ksize=7,
                     stride=2, simple_channels=32, complex_channels=32),
        layers.Conv2D(64, kernel_size=(3, 3)),
        layers.BatchNormalization(),
        layers.Activation('swish'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3)),
        layers.BatchNormalization(),
        layers.Activation('swish'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

print(model.summary())
