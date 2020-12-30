import tensorflow as tf
import numpy as np
from .utils import gabor_kernel, sample_dist, generate_gabor_param


class Identity(tf.keras.layers.Layer):
    def call(self, x):
        return x


class GFB(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=4, padding='same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = padding.upper()

        # Param instantiations
        self.kernel = np.zeros(
            (kernel_size, kernel_size, in_channels, out_channels))

    def call(self, inputs):
        return tf.nn.conv2d(inputs, self.kernel, strides=self.stride,
                            padding=self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase):
        random_channel = tf.random.uniform(
            shape=(self.out_channels,), minval=0, maxval=self.in_channels, dtype=tf.int64)
        for i in range(self.out_channels):
            self.kernel[:, :, random_channel[i], i] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                                   theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
        self.kernel = tf.constant(self.kernel, dtype=tf.float32)


class VOneBlock(tf.keras.layers.Layer):
    def __init__(self, sf, theta, sigx, sigy, phase,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=1,
                 simple_channels=128, complex_channels=128, ksize=25,
                 stride=4, input_size=224, in_channels=3, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.out_channels = simple_channels + complex_channels
        self.stride = stride
        self.input_size = input_size

        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase
        self.k_exc = k_exc

        self.set_noise_mode(noise_mode, noise_scale, noise_level)
        self.fixed_noise = None

        self.simple_conv_q0 = GFB(
            self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(
            self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2)

        self.simple = tf.nn.relu
        self.complex = Identity()
        self.gabors = Identity()
        self.noise = tf.nn.relu
        self.out = Identity()

    def call(self, x):
        # Gabor activations [Batch, H/stride, W/stride, out_channels]
        x = self.gabors_f(x)
        # Noise [Batch, H/stride, W/stride, out_channels]
        x = self.noise_f(x)
        # V1 Block output: (Batch, H/stride, W/stride, out_channels)
        x = self.out(x)
        return x

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        s_q1 = self.simple_conv_q1(x)
        c = self.complex(tf.math.sqrt(s_q0[:, :, :, self.simple_channels:] ** 2 +
                                      s_q1[:, :, :, self.simple_channels:] ** 2) / tf.math.sqrt(2.0))
        s = self.simple(s_q0[:, :, :, 0:self.simple_channels])
        return self.gabors(self.k_exc * tf.concat((s, c), axis=3))

    def noise_f(self, x):
        if self.noise_mode == 'neuronal':
            eps = 10e-5
            x *= self.noise_scale
            x += self.noise_level
            if self.fixed_noise is not None:
                x += self.fixed_noise * tf.math.sqrt(tf.nn.relu(x) + eps)
            else:
                x += tf.random.normal(tf.shape(x)) * \
                    tf.math.sqrt(tf.nn.relu(x)+eps)

            x -= self.noise_level
            x /= self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level

    def fix_noise(self, batch_size=256, seed=None):
        noise_shape = (batch_size,
                       int(self.input_size/self.stride),
                       int(self.input_size/self.stride),
                       self.out_channels)
        if self.noise_mode == 'neuronal':
            self.fixed_noise = tf.random.normal(noise_shape, seed=seed)

    def unfix_noise(self):
        self.fixed_noise = None


class VOneNetLayer(tf.keras.layers.Layer):

    def __init__(self, sf_corr=0.75, sf_max=6, sf_min=0, rand_param=False, gabor_seed=0,
                 simple_channels=256, complex_channels=256,
                 noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
                 shape=(224, 224, 3), visual_degrees=8, ksize=25, stride=4, **kwargs):
        super().__init__(**kwargs)

        out_channels = simple_channels + complex_channels

        sf, theta, phase, nx, ny = generate_gabor_param(
            out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

        self.gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels,
                             'rand_param': rand_param,
                             'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                             'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
        self.arch_params = {'k_exc': k_exc, 'ksize': ksize, 'stride': stride}

        # Conversions
        ppd = shape[0] / visual_degrees

        sf = sf / ppd
        sigx = nx / sf
        sigy = ny / sf
        theta = theta/180 * np.pi
        phase = phase / 180 * np.pi

        self.vone_block = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                                    k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                                    simple_channels=simple_channels, complex_channels=complex_channels,
                                    ksize=ksize, stride=stride, input_size=shape[0], in_channels=shape[2])

    def call(self, inputs):
        return self.vone_block(inputs)

    def get_config(self):
        config = {}
        config.update(self.gabor_params)
        config.update(self.arch_params)
        return config
