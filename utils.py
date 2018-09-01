import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import youtube_dl


def create_rnn_data(data: np.ndarray, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    if time_steps >= len(data):
        raise ValueError("Length of data is lower then the time_steps value")

    nb_of_batches = len(data) - time_steps

    x_rnn_data = np.zeros((nb_of_batches, time_steps, data.shape[1]), dtype=data.dtype)
    y_rnn_data = np.zeros((nb_of_batches, data.shape[1]), dtype=data.dtype)

    for start_index in range(nb_of_batches):
        end_index = start_index + time_steps

        x_data = data[start_index:end_index]
        y_data = data[end_index]

        x_rnn_data[start_index] = x_data
        y_rnn_data[start_index] = y_data

    return x_rnn_data, y_rnn_data


def show_image_grid(images, n_images=10, n_rows=3, figsize=(10, 10), randomize=False) -> None:
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


def frame_preprocessor(frame: np.ndarray) -> np.ndarray:
    frame = cv2.resize(frame, (64, 64))
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = frame.astype(np.float32) / 255.0
    return frame


def decoded_frame_postprocessor(frame: np.ndarray) -> np.ndarray:
    frame = (frame * 255).astype(np.uint8)
    return frame


def get_frames_from_youtube_video(video_url: str,
                                  frame_preprocessor: Callable[[np.ndarray], np.ndarray] = None) -> np.ndarray:
    # Downloading the video

    output_video_file_path = tempfile.NamedTemporaryFile().name

    youtube_downloader_params = {"quiet": False, "outtmpl": output_video_file_path, "format": "best[height<=240]"}
    with youtube_dl.YoutubeDL(params=youtube_downloader_params) as ydl:
        ydl.download([video_url])

    output_video_file_path = Path(output_video_file_path)

    # Frame extraction

    input_video_file_path = str(list(output_video_file_path.parent.glob(output_video_file_path.stem + "*"))[0])

    with tempfile.TemporaryDirectory() as tmp_folder:
        output_filename = os.path.join(tmp_folder, '%06d.jpg')
        video_to_frames_command = ['ffmpeg', '-loglevel', 'debug', '-i', input_video_file_path, '-vf', 'fps=24',
                                   output_filename]
        subprocess.check_call(video_to_frames_command)

        os.remove(input_video_file_path)

        frame_paths = np.sort(list(Path(tmp_folder).glob("*.jpg")))
        frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_RGB2BGR) for p in frame_paths]
        if frame_preprocessor is not None:
            frames = [frame_preprocessor(f) for f in frames]

    return np.array(frames)


def convert_video_to_gif(input_video_path, output_gif_path, fps=24):
    palette_image_path = "palette.png"
    command_palette = 'ffmpeg -y -t 0 -i {0} -vf fps={1},scale=320:-1:flags=lanczos,palettegen {2}'.format(input_video_path,
                                                                                                           fps,
                                                                                                           palette_image_path)
    command_convert = 'ffmpeg -y -t 0 -i {0} -i {1} -filter_complex "fps={2},scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse" {3}'.format(input_video_path,palette_image_path, fps, output_gif_path)
    
    try:
        subprocess.check_call(command_palette)
        subprocess.check_call(command_convert)
    except subprocess.CalledProcessError as exc:
        print(exc.output)
        raise
    finally:
        os.remove(palette_image_path)