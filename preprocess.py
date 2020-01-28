from scipy.signal import stft
from scipy.signal import resample
import math
import numpy as np


def process_audio(sr, audio, down_sr=16000, stft_bin=128, stft_overlap=64, debug=False):

    '''
    This function process RMS of the window and generate the Spectrogram

    https://en.wikipedia.org/wiki/Spectrogram

    :param sr: sampling rate (Hz)
    :param audio: single channel audio
    :param down_sr: down sample to (Hz)
    :param stft_bin: frequency sample. divided by 2 + 1 equals to number of possible frequency in the spectrogram.
    :param stft_overlap: window overlaps setting
    :param debug: print debug information
    :return: rms, spectrogram, frequency
    '''
    frame_size = 4
    window_size = 16

    length = audio.shape[0]
    if sr != down_sr:
        length = round(length / sr * down_sr)
        audio = resample(audio, length)
        sr = down_sr

    frame_sample_count = math.floor(sr / 1000 * frame_size)  # frame_size is millisecond
    window_sample_count = frame_sample_count * window_size
    last_window_padding = 2 * window_sample_count - (length % window_sample_count)

    if debug:
        print(
            " frame_size:", frame_size,
            " window_size:", window_size,
            " last_window_padding:", last_window_padding,
        )

    # padding
    audio = np.int32(np.pad(audio, (0, last_window_padding), "constant", constant_values=(0, 0)))

    # RMS
    audio_square = np.float64(np.square(audio))
    audio_rolling_list = np.array(
        (
            audio_square,
            np.roll(audio_square, -64 * 1),
            np.roll(audio_square, -64 * 2),
            np.roll(audio_square, -64 * 3),
            np.roll(audio_square, -64 * 4),
            np.roll(audio_square, -64 * 5),
            np.roll(audio_square, -64 * 6),
            np.roll(audio_square, -64 * 7),
            np.roll(audio_square, -64 * 8),
            np.roll(audio_square, -64 * 9),
            np.roll(audio_square, -64 * 10),
            np.roll(audio_square, -64 * 11),
            np.roll(audio_square, -64 * 12),
            np.roll(audio_square, -64 * 13),
            np.roll(audio_square, -64 * 14),
            np.roll(audio_square, -64 * 15),
        )
    ).reshape((16, -1, window_sample_count))
    rms = np.sqrt(np.mean(audio_rolling_list, axis=2)).T.reshape(-1)

    # STFT
    f, t, zxx = stft(audio.reshape(-1), nperseg=stft_bin, noverlap=stft_overlap, fs=sr)
    zxx_abs = np.absolute(zxx)

    if debug:
        print(
            " f:", f.shape,
            " t:", t.shape,
            " zxx:", zxx.shape,
        )
        print(
            " zxx_abs_min:", zxx_abs.min(),
            " zxx_abs_max:", zxx_abs.max()
        )

    zxx_log = np.log(zxx_abs + 1)

    return {
        "rms": rms,
        "zxx": zxx[1:, :-1],
        "zxx_log": zxx_log[1:, :-1]
    }