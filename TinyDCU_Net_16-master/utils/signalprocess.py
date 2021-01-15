import numpy as np
from numpy.fft import rfft, irfft
from scipy import signal


def analysis_window(size, overlap):
    w = np.ones(size)
    m0 = size - overlap
    m1 = size - m0
    w[:m0] = np.sqrt(0.5 * (1 - np.cos(np.pi * np.arange(1, m0 + 1) / m0)))
    w[m1 - 1:size] = np.sqrt(0.5 * (1 - np.cos(np.pi * np.arange(m0, -1, -1) / m0)))
    return w


def samples_to_stft_frames(samples, size, shift, ceil=False):
    if ceil:
        return 1 if samples <= size - shift else \
            np.ceil((samples - size + shift) / shift).astype(np.int32)
    else:
        return 1 if samples <= size else (samples - size + shift) // shift


def stft_frames_to_samples(frames, size, shift):
    return frames * shift + size - shift


# compute stft of a 1-dim time_signal
def stft_analysis(time_signal, size=512, shift=256, fading=False, ceil=False,
                  window_func=analysis_window):
    assert time_signal.ndim == 1
    time_signal = np.concatenate((np.zeros((size - shift,)), time_signal[:len(time_signal) + shift - size]))
    if fading:
        pad = [(size - shift, size - shift)]
        time_signal = np.pad(time_signal, pad, mode='constant')
    frames = samples_to_stft_frames(time_signal.shape[0], size, shift, ceil=ceil)
    samples = stft_frames_to_samples(frames, size, shift)
    if samples > time_signal.shape[0]:
        pad = [(0, samples - time_signal.shape[0])]
        time_signal = np.pad(time_signal, pad, mode='constant')
    window = window_func(size, shift)
    chunk_signal = np.zeros((frames, size))
    for i, j in enumerate(range(0, samples - size + shift, shift)):
        chunk_signal[i] = time_signal[j:j + size]

    return rfft(chunk_signal * window, axis=1)


def stft_synthesis(stft_signal, size=512, shift=256, fading=False,
                   window_func=analysis_window, signal_length=None):
    assert stft_signal.shape[1] == size // 2 + 1
    # assert (stft_signal.shape[0] * shift + size - shift) < signal_length
    window = window_func(size, shift)
    time_signal = np.zeros(signal_length + size)
    j = 0
    for i in range(0, stft_signal.shape[0]):
        time_signal[j:j + size] += window * np.real(irfft(stft_signal[i], size))
        j = j + shift

    if fading:
        sync_signal = time_signal[size - shift:size - shift + signal_length]
    else:
        sync_signal = time_signal[:signal_length]
    return sync_signal.astype(np.float32)
