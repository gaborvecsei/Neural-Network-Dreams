import cv2
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import subprocess
from pathlib import Path


def create_rnn_data(encoded_images, time_steps):
    x_rnn_data = []
    y_rnn_data = []

    for i in range(len(encoded_images) - 1):
        x_data = encoded_images[i:i + time_steps]
        if len(x_data) != time_steps:
            # This happens when we reach the end of the array
            break
        x_rnn_data.append(x_data)

        y_data = encoded_images[i + 1]
        y_rnn_data.append(y_data)

    x_rnn_data = np.array(x_rnn_data)
    y_rnn_data = np.array(y_rnn_data)

    return x_rnn_data, y_rnn_data


def show_image_grid(images, n_images=10, n_rows=3, figsize=(10, 10), randomize=True):
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


def get_frames_from_youtube_video(video_url, frame_preprocessor=lambda x:x, remove_tmp=True):
    output_file_name = tempfile.NamedTemporaryFile().name

    # Youtube-DL: https://rg3.github.io/youtube-dl/
    command = ["youtube-dl", "-f", "bestvideo[height<=480]+bestaudio/best[height<=480]", "--quiet", "-o",
               output_file_name, video_url]
    subprocess.check_call(command)

    # It is not always mp4 format, it can be mkv, etc... because of
    # this we need to "search" the file with the right extension
    input_file_name = str(list(Path(output_file_name).parent.glob("*"))[0])

    with tempfile.TemporaryDirectory() as tmp_folder:
        output_filename = os.path.join(tmp_folder, '%06d.jpg')
        command = ['ffmpeg', '-loglevel', 'debug', '-i', input_file_name, '-vf', 'fps=24', output_filename]
        subprocess.check_call(command)
    
        os.remove(input_file_name)
        
        frame_paths = np.sort(list(Path(tmp_folder).glob("*.jpg")))
        frames = [frame_preprocessor(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_RGB2BGR)) for p in frame_paths]

    return np.array(frames)