# Neural Network Dreams

Image sequence generation with Variational Auto Encoder and Recurrent Neural Network

![img](art/generated_image_sequence.gif)

This is just a quick :zap: experiment what I wanted to do.

The basic idea is that I encode frames with VAE from `(64x64x3)` to `32` dim vectors and then use
RNN (GRU/LSTM) to generate new sequences.

## Setup

You can try it out easily with a Docker image (Jupyter Lab/Notebook w/ GPU access):

[Jupyter Lab Image](https://github.com/gaborvecsei/Jupyter-Lab-Docker)

## Train & Generate

You can find the details of the training and generation in this [Notebook](video_generation.ipynb)

### Generated sequence

With the first version of the RNN, which was built without a Mixture Density Layer, the generated sequence was pretty static, it looked like an average
image. Then I read the first few sentences of the paper [Mixture Density Networks (MDN)](https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf) and realized the problem.

> Minimization of a sum-of-squares or cross-entropy error function leads to network outputs
which approximate the conditional averages of the target data, conditioned on the
input vector. For classifications problems, with a suitably chosen target coding scheme,
these averages represent the posterior probabilities of class membership, and so can be
regarded as optimal. For problems involving the prediction of continuous variables, however,
the conditional averages provide only a very limited description of the properties
of the target variables.

#### Steps of training

1. Train VAE to encode & decode images
2. Encode images to `32 dim` vectors
3. Create data batches (time steps) for the RNN with a sliding window
    - Shape of `X`: `(n_data, n_time_steps, 32)`, (32 is the output dim from the VAE)
    - Shape of `y`: `(n_data, 32)`
    - Intuitively: For `n_time_steps` consecutive frames in `X`, we can find the next frame `n_time_steps + 1` in the corresponding `y`.
4. Train the RNN (currently it is GRU with MDN)
5. Generate encoded images with RNN
6. Decode it with the VAE

## Results

### VAE encoded, then decoded frames

![img](art/vae_decoded_vs_original.gif)

### Visualizing the Dream

Current Version (with MD layer)           |  First version (without MD layer)
:----------------------------------------:|:-------------------------:
![img](art/generated_image_sequence.gif)  |  ![img](art/generated_image_sequence_prev.gif)
