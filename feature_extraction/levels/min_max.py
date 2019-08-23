from librosa.core import load


def min_max(filepath: str) -> tuple:
	time_series, _ = load(filepath)
	return time_series.min(), time_series.max()
