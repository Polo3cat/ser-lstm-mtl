from librosa.core import load
from librosa.util import frame
import numpy as np
from scipy.integrate import trapz


def normalize_gain(samples: np.ndarray) -> np.ndarray:
	min_ = samples.min()
	max_ = samples.max()
	return (samples - min_) / (max_ - min_)


def energy(filepath: str, frame_ms: int, sliding_ms: int) -> np.ndarray:
	'''
		Given an audio file, returns the energy (calculated as the area
		under the curve of the signal) for each frame of width
		frame_ms sliding each sliding_ms.
		This functions uses the composite trapezoidal rule to approximate
		the are, since other methods are far too expensive (like Simpson's or
		Romberg's).
	'''
	time_series, sr = load(filepath)
	sr_ms = sr / 1000
	time_series = normalize_gain(time_series)
	frames = frame(time_series, frame_length=int(sr_ms*frame_ms), hop_length=int(sr_ms*sliding_ms))
	return trapz(frames, dx=frame_ms, axis=0)
