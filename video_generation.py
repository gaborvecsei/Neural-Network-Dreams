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
generated_encoded_frames = vae.encoder.predict(starter_frames)

for i in range(n_images_to_generate):
    next_frame = rnn.model.predict(np.expand_dims(generated_encoded_frames[i:i + model_config.GRU_TIME_STEPS], axis=0))
    generated_encoded_frames = np.vstack((generated_encoded_frames, next_frame))

# Remove the manually created "starter" frames
generated_encoded_frames = generated_encoded_frames[model_config.GRU_TIME_STEPS:, :]
generated_decoded_frames = vae.decoder.predict(generated_encoded_frames)
generated_decoded_frames = (generated_decoded_frames * 255).astype(np.uint8)

# Save the generated frames
generated_frames_folder = Path("./generated_frames")
generated_frames_folder.mkdir(exist_ok=True)

for i, generated_frame in enumerate(generated_decoded_frames):
    image_name = "{0}.jpg".format(str(i).ljust(6, "0"))
    cv2.imwrite(image_name, generated_frame)
