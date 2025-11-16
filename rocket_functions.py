# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)

import numpy as np
from numba import njit, prange

@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
def generate_kernels(input_length, num_kernels):

    candidate_lengths = np.array((7, 9, 11), dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)

    a1 = 0

    for i in range(num_kernels):

        _length = lengths[i]

        _weights = np.random.normal(0, 1, _length)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings

@njit(fastmath = True)
def apply_kernel(X, weights, length, bias, dilation, padding):

    input_length = len(X)

    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < input_length:

                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    if output_length == 0:
        output_length = 1e-7 # avoid division by zero
    return _ppv / output_length, _max

@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))", parallel = True, fastmath = True)
def apply_kernels(X, kernels):

    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype = np.float64) # 2 features per kernel

    for i in prange(num_examples):

        a1 = 0 # for weights
        a2 = 0 # for features

        for j in range(num_kernels):

            b1 = a1 + lengths[j]
            b2 = a2 + 2

            _X[i, a2:b2] = \
            apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

    return _X

from sklearn.preprocessing import StandardScaler
def window_rocket(original, window_size=0.1, num_kernels=10_000):

    def create_windows(original, window_size):
        '''
        Split windows such that:
            each window is of shape (window_size, num_features)
            and each window can be mapped back to a row of the original dataset
            num_windows = len(original) - window_size + 1
        '''
        windows = []
        for i in range(len(original) - window_size+1):
            window = original[i:i+window_size]
            if len(window.shape) < 2:
                window = window.reshape(-1,1)
            windows.append(window)
        return windows

    # Split the time series into overlapping windows
    window_size = round(window_size * len(original))  # you can adjust the window size
    windows = create_windows(original, window_size)

    num_kernels = 10_000  # you can adjust the number of kernels
    kernels = generate_kernels(window_size, num_kernels)

    new_X = []
    for window in windows:
        window = apply_kernels(window, kernels)
        # Reduce the window of size (window_size, num_kernels) to a single row of size (1, num_kernels)
        row = np.expand_dims(np.mean(window, axis=0), axis=0)
        new_X.append(row)
    new_X = np.concatenate(new_X, axis=0)

    scaler = StandardScaler()
    new_X[np.isinf(new_X)] = np.finfo(np.float32).max
    new_X[np.isneginf(new_X)] = np.finfo(np.float32).min
    generated = scaler.fit_transform(new_X)
    return generated