# Image sequence generation with VAE and RNN

This is just a quick :zap: experiment what I wanted to do.

The basic idea is that I encode frames with VAE from `(64x64x3)` to `(32,)` dim vectors and then use
RNN (GRU/LSTM) to generate new sequences.

## Train & Generate

You can find the details of the training and generation in this [Notebook](video_generation.ipynb)

## VAE encoded and then decoded frames

With the images I train the VAE and then I encode the frames to 32 dim vectors.
After this I immediately decode it and this is the result I get:

![img](art/vae_decoded_vs_original.gif)

## Generated sequence

For generation I train the VAE and then with that, I encode all the frames used in the training.
From the encoded frames, I generate data batches like `(n_data, n_time_steps, 32)` with a sliding window -> This is `X`.
`y` is with shape of `(n_data, 32)`. For every `X` there is an `y` witch is the next frame after the `n_time_steps` frames in the original sequence.

*As you can see, right now there is no much "generation" going on* :sob:

![img](art/generated_image_sequence.gif)
