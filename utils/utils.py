import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_rnn_data(data: np.ndarray, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    if time_steps >= len(data):
        raise ValueError("Length of data is lower then the time_steps value")

    nb_of_batches = len(data) - time_steps

    x_rnn_data = np.zeros((nb_of_batches, time_steps), dtype=data.dtype)
    y_rnn_data = np.zeros(nb_of_batches, dtype=data.dtype)

    for start_index in range(nb_of_batches):
        end_index = start_index + time_steps

        x_data = data[start_index:end_index]
        y_data = data[end_index]

        x_rnn_data[start_index] = x_data
        y_rnn_data[start_index] = y_data

    return x_rnn_data, y_rnn_data


def show_image_grid(images, n_images=10, n_rows=3, figsize=(10, 10), randomize=False):
    n_cols = int(np.ceil(n_images / n_rows))

    fig = plt.figure(figsize=figsize)

    for i in range(n_images):
        rnd = i
        if randomize:
            rnd = np.random.randint(0, len(images))
        image = images[rnd]
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(image)
        ax.set_yticks([])
        ax.set_xticks([])


def frame_preprocessor(frame):
    frame = cv2.resize(frame, (64, 64))
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = frame.astype(np.float32) / 255.0
    return frame


def get_frames_from_youtube_video(video_url, frame_preprocessor=lambda x: x):
    output_file_name = tempfile.NamedTemporaryFile().name

    download_command = ["youtube-dl", "-f", "bestvideo[height<=240]+bestaudio/best[height<=240]", "--quiet", "-o",
                        output_file_name, video_url]
    subprocess.check_call(download_command)

    # It is not always mp4 format, it can be mkv, etc... because of this we need to "search" the
    # file with the right file
    output_file_path = Path(output_file_name)
    input_file_name = str(list(output_file_path.parent.glob(output_file_path.stem + ".*"))[0])

    with tempfile.TemporaryDirectory() as tmp_folder:
        output_filename = os.path.join(tmp_folder, '%06d.jpg')
        video_to_frames_command = ['ffmpeg', '-loglevel', 'debug', '-i', input_file_name, '-vf', 'fps=24',
                                   output_filename]
        subprocess.check_call(video_to_frames_command)

        os.remove(input_file_name)

        frame_paths = np.sort(list(Path(tmp_folder).glob("*.jpg")))
        frames = [frame_preprocessor(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_RGB2BGR)) for p in frame_paths]

    return np.array(frames)
