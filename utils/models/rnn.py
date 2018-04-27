from keras.callbacks import EarlyStopping
from keras.layers import Dense, GRU
from keras.models import Sequential

from utils.model_config import *


class RNN:
    def __init__(self, input_dim, output_dim, time_step):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_steps = time_step
        self.hidden_neurons = GRU_NB_HIDDEN_NEURONS

        self.model = self._build()

    @classmethod
    def init_default(cls):
        obj = cls(VAE_Z_DIM, VAE_Z_DIM, GRU_TIME_STEPS)
        return obj

    def _build(self):
        model = Sequential()
        model.add(GRU(self.hidden_neurons, input_shape=(self.time_steps, self.input_dim), return_sequences=True))
        model.add(GRU(self.hidden_neurons))
        model.add(Dense(self.output_dim, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
        return model

    def train(self, data_in, data_out, epochs=100, include_callbacks=True):
        callbacks_list = []

        if include_callbacks:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
            callbacks_list = [early_stopping]

        self.model.fit(data_in, data_out,
                       shuffle=True,
                       epochs=epochs,
                       batch_size=GRU_BATCH_SIZE,
                       validation_split=0.2,
                       callbacks=callbacks_list)
