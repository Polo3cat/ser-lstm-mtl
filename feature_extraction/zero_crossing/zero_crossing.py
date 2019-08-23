from librosa.core import load
from librosa.feature import zero_crossing_rate
import numpy as np


def normalize_gain(samples: np.ndarray) -> np.ndarray:
	min_ = samples.min()
	max_ = samples.max()
	return (samples - min_) / (max_ - min_)


def zc_rate(filepath: str, frame_ms: int, sliding_ms: int, threshold: float) -> int:
	'''
		Given a filepath to an audio source (.wav format)
		returns the zero crossings rate using
		a sliding frame. Use the threshold to ignore small variations
		close to zero.
	'''
	time_series, sr = load(filepath)
	time_series = normalize_gain(time_series)
	sr_ms = sr / 1000
	return zero_crossing_rate(time_series, 
								frame_length=int(frame_ms*sr_ms), 
								hop_length=int(sliding_ms*sr_ms), 
								threshold=threshold)

