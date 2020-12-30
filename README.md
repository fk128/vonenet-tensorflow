
## VOneBlock layer

A direct port to tensorflow of the VOneBlock layer from the VOneNet [pytorch implementation](https://github.com/dicarlolab/vonenet) of

Dapello, J., Marques, T., Schrimpf, M., Geiger, F., Cox, D.D., DiCarlo, J.J. (2020) Simulating a Primary Visual Cortex at the Front of CNNs Improves Robustness to Image Perturbations. biorxiv. doi.org/10.1101/2020.06.16.154542


### Example

```python
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
```