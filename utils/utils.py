import cv2
import matplotlib.pyplot as plt
import numpy as np


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
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = frame.astype(np.float32) / 255.0
    return frame
