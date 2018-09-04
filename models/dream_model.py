from pathlib import Path
from typing import List, Tuple, Union
import warnings
from models import vae, rnn
import numpy as np
from keras.callbacks import EarlyStopping, History
import utils


class DreamerNetwork:
    def __init__(self, image_width: int = 64, image_height: int = 64, latent_dim: int = 32, time_steps: int = 10):
        self.image_height = image_height
        self.image_width = image_width
        self.latent_dim = latent_dim
        self.time_steps = time_steps

        self.image_input_shape = (image_height, image_width, 3)

        self.model_vae = vae.VAE(self.image_input_shape, self.latent_dim)
        self.vae_train_config_dict = None

        self.model_rnn = rnn.RNN(self.latent_dim, self.latent_dim, self.time_steps, contain_mdn_layer=True)
        self.rnn_train_config_dict = None

    @staticmethod
    def init_default_vae_train_dict() -> dict:
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        return {"batch_size": 32, "epochs": 100, "validation_split": 0.2, "shuffle": True,
                "callbacks": [early_stopping]}

    @staticmethod
    def init_default_rnn_train_dict() -> dict:
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        return {"batch_size": 32, "epochs": 100, "validation_split": 0.2, "shuffle": True,
                "callbacks": [early_stopping]}

    def fit(self, image_data: np.ndarray, vae_train_config_dict: dict = None, rnn_train_config_dict: dict = None) -> \
            Tuple[History, History]:
        if vae_train_config_dict is None:
            vae_train_config_dict = self.init_default_vae_train_dict()
            warnings.warn(
                "There is no custom VAE training config dict given. Training with default config:\n{0}".format(
                    vae_train_config_dict))
        if rnn_train_config_dict is None:
            rnn_train_config_dict = self.init_default_rnn_train_dict()
            warnings.warn(
                "There is no custom RNN training config dict given. Training with default config:\n{0}".format(
                    rnn_train_config_dict))

        print("Training VAE...\n")
        hist_vae = self.model_vae.model.fit(image_data, image_data, **vae_train_config_dict)

        print("Preparing data for RNN... \n")
        encoded_images = self.model_vae.encode(image_data)
        x_rnn_data, y_rnn_data = utils.create_rnn_data(encoded_images, self.time_steps)

        print("Training RNN...\n")
        hist_rnn = self.model_rnn.model.fit(x_rnn_data, y_rnn_data, **rnn_train_config_dict)

        return hist_vae, hist_rnn

    def dream(self, n_images_to_generate: int, temperature: float = 1.0) -> np.ndarray:
        # Create the first few random frames (dream trigger)
        starter_random_frames = np.random.randint(0, 255, (self.time_steps,) + self.image_input_shape, dtype=np.uint8)
        encoded_starter_random_frames = self.model_vae.encode(starter_random_frames)

        # We will store the generated data here
        generated_encoded_frames = encoded_starter_random_frames.copy()

        for i in range(n_images_to_generate):
            next_encoded_frame = \
                self.model_rnn.predict(np.expand_dims(generated_encoded_frames[i:i + self.time_steps], axis=0),
                                       temp=temperature)[0]
            generated_encoded_frames = np.vstack((generated_encoded_frames, next_encoded_frame))

        # Getting rid of the first randomly created frames
        generated_encoded_frames = generated_encoded_frames[self.time_steps:]
        generated_decoded_frames = self.model_vae.decode(generated_encoded_frames,
                                                         decoder_postprocessor=utils.decoded_frame_postprocessor)

        return generated_decoded_frames

    def save_models(self, saved_models_folder_path: Union[Path, str], model_prefix: str = ""):
        saved_models_folder_path = Path(saved_models_folder_path)
        if not saved_models_folder_path.is_dir():
            saved_models_folder_path.mkdir(parents=True)

        vae_model_file_name = model_prefix + "vae_weights.h5"
        vae_model_file_path = saved_models_folder_path / vae_model_file_name
        self.model_vae.model.save_weights(vae_model_file_path)
        print("VAE model saved to: {0}".format(vae_model_file_path))

        rnn_model_file_name = model_prefix + "rnn_weights.h5"
        rnn_model_file_path = saved_models_folder_path / rnn_model_file_name
        self.model_rnn.model.save_weights(rnn_model_file_path)
        print("RNN model saved to: {0}".format(rnn_model_file_path))

    def load_models(self, vae_weights_file_path: Union[Path, str], rnn_weights_file_path: Union[Path, str]):
        self.model_vae.model.load_weights(vae_weights_file_path)
        self.model_rnn.model.load_weights(rnn_weights_file_path)
