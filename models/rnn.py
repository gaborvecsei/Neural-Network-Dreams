import numpy as np
from keras.layers import GRU, Dense
from keras.models import Sequential, Model

from . import mdn


class RNN:
    def __init__(self, input_dim: int, output_dim: int, time_step: int, contain_mdn_layer: bool = True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_steps = time_step
        self.contain_mdn_layer = contain_mdn_layer

        if self.contain_mdn_layer:
            self.number_of_mixtures = 10

        self.model = self._build()

    @classmethod
    def init_project_default(cls) -> "RNN":
        obj = cls(32, 32, 5, contain_mdn_layer=True)
        return obj

    def _build(self) -> Model:
        model = Sequential()
        model.add(GRU(32, input_shape=(self.time_steps, self.input_dim), return_sequences=True))
        model.add(GRU(64, return_sequences=True))
        model.add(GRU(128))
        if self.contain_mdn_layer:
            model.add(mdn.MDN(self.output_dim, self.number_of_mixtures))
            model.compile(loss=mdn.get_mixture_loss_func(self.input_dim, self.number_of_mixtures), optimizer="adam",
                          metrics=['accuracy'])
        else:
            model.add(Dense(self.output_dim, activation='softmax'))
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

        return model

    def predict(self, X: np.ndarray, temp: float = 1.0):
        preds = self.model.predict(X)

        if self.contain_mdn_layer:
            preds = np.apply_along_axis(mdn.sample_from_output,
                                        1,
                                        preds,
                                        self.output_dim,
                                        self.number_of_mixtures,
                                        temp=temp)
            preds = np.squeeze(preds, 1)
        return preds
