import numpy as np
from utils.models.vae import VAE
from utils.models.rnn import RNN
from pathlib import Path
import cv2
from utils import utils, model_config

video_frame_folder_path = Path("../data/my_video/frames")

frame_paths = np.sort(list(video_frame_folder_path.glob("*.jpg")))
frames = np.array([utils.frame_preprocessor(cv2.imread(str(p))) for p in frame_paths])

# Train VAE
vae = VAE.init_default()
vae.train(frames, 25, include_callbacks=False)

# Create data for RNN
encoded_images = vae.encoder.predict(frames)
decoded_images = (vae.decoder.predict(encoded_images) * 255).astype(np.uint8)
x_rnn_data, y_rnn_data = utils.create_rnn_data(encoded_images, model_config.GRU_TIME_STEPS)

# Train RNN
rnn = RNN.init_default()
rnn.train(x_rnn_data, y_rnn_data, 100, include_callbacks=False)

# Generate frames
n_images_to_generate = 100
starter_frames = frames[0:model_config.GRU_TIME_STEPS]
starter_frames = np.array([utils.frame_preprocessor(x) for x in starter_frames])
encoded_frames = vae.encoder.predict(starter_frames)

for i in range(n_images_to_generate):
    next_frame = rnn.model.predict(np.expand_dims(encoded_frames[i:i + model_config.GRU_TIME_STEPS], axis=0))
    encoded_frames = np.vstack((encoded_frames, next_frame))

# Remove the manually created "starter" frames
encoded_frames = encoded_frames[model_config.GRU_TIME_STEPS:, :]
