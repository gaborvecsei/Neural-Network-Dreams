from typing import Tuple, Callable

import numpy as np
from keras import backend as K
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model


class VAE:
    def __init__(self, input_image_shape: Tuple[int, int, int], encoded_output_dim: int):
        self.input_image_shape = input_image_shape
        self.encoded_output_dim = encoded_output_dim

        self.model, self.encoder, self.decoder = self._build()

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.encoded_output_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    @classmethod
    def init_project_default(cls) -> "VAE":
        obj = cls((64, 64, 3), 32)
        return obj

    def _build(self) -> Tuple[Model, Model, Model]:
        # Encoder Part

        vae_x = Input(shape=self.input_image_shape)
        vae_c1 = Conv2D(filters=32, kernel_size=4, strides=2, activation="relu")(vae_x)
        vae_c2 = Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(vae_c1)
        vae_c3 = Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(vae_c2)
        vae_c4 = Conv2D(filters=128, kernel_size=4, strides=2, activation="relu")(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(self.encoded_output_dim)(vae_z_in)
        vae_z_log_var = Dense(self.encoded_output_dim)(vae_z_in)

        vae_z = Lambda(self.sampling)([vae_z_mean, vae_z_log_var])

        # Decoder part

        vae_z_input = Input(shape=(self.encoded_output_dim,))

        vae_dense = Dense(1024)
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((1, 1, 1024))
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation="relu")
        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation="relu")
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3 = Conv2DTranspose(filters=32, kernel_size=6, strides=2, activation="relu")
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4 = Conv2DTranspose(filters=3, kernel_size=6, strides=2, activation="sigmoid")
        vae_d4_model = vae_d4(vae_d3_model)

        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)

        # Constructed Models

        vae = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        # Custom Loss Functions

        def vae_r_loss(y_true, y_pred):
            return K.sum(K.square(y_true - y_pred), axis=[1, 2, 3])

        def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.sum(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis=-1)

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)

        vae.compile(optimizer="adam", loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])

        return vae, vae_encoder, vae_decoder

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self.encoder.predict(X)

    def decode(self, X: np.ndarray, decoder_postprocessor: Callable[[np.ndarray], np.ndarray] = None) -> np.ndarray:
        decoded = self.decoder.predict(X)
        if decoder_postprocessor is not None:
            decoded = np.array([decoder_postprocessor(x) for x in decoded])
        return decoded
