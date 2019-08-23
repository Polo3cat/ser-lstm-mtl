from librosa.core import load
from librosa.feature import mfcc, delta
import numpy as np


def normalize_gain(samples: np.ndarray) -> np.ndarray:
	min_ = samples.min()
	max_ = samples.max()
	return (samples - min_) / (max_ - min_)


def mfccs(filepath: str, frame_ms: int, sliding_ms: int, n_mfccs: int) -> np.ndarray:
	'''
		Given a filepath, computes the Mel Frequency Cepstrum Coefficients
		specified by n_mfccs using a frame width of frame_ms and a slide of
		sliding_ms
	'''
	time_series, sr = load(filepath)
	time_series = normalize_gain(time_series)
	sr_ms = sr / 1000
	return mfcc(time_series,
				sr=sr,
				n_mfcc=n_mfccs,
				n_fft=int(frame_ms*sr_ms),
				hop_length=int(sliding_ms*sr_ms))


def mfccs_deltas(mfcc: np.ndarray, N: int, order: int):
	return delta(mfcc, width=N, order=order)
