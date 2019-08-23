'''
    Credit for parabolic and freq_from_fft go to:
        endolith
        https://gist.github.com/endolith/255291
'''
from librosa.core import load
from librosa.util import frame
import numpy as np
from numpy.fft import rfft

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
   
    f is a vector and x is an index for that vector.
   
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
   
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
   
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
   
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
   
    """
    xv = 1/2 * (f[max(0,x-1)] - f[min(f.shape[0]-1, x+1)]) / (f[max(0,x-1)] - 2 * f[x] + f[min(f.shape[0]-1, x+1)]) + x
    # yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return xv


def freq_from_fft(sig, sample_rate):
    """Estimate frequency from peak of FFT
    
    Pros: Accurate, usually even more so than zero crossing counter 
    (1000.000004 Hz for 1000 Hz, for instance).  Due to parabolic interpolation 
    being a very good fit for windowed log FFT peaks?
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    Accuracy also increases with signal length
    
    Cons: Doesn't find the right value if harmonics are stronger than 
    fundamental, which is common.
    
    """
    # Compute Fourier transform of windowed signal
    f = rfft(sig)
    
    # Find the peak and interpolate to get a more accurate peak
    i = np.argmax(np.abs(f)) # Just use this for less-accurate, naive version
    true_i = parabolic(np.log(np.abs(f)), i)
    
    # Convert to equivalent frequency
    return sample_rate * true_i / len(sig)


def normalize_gain(samples: np.ndarray) -> np.ndarray:
    min_ = samples.min()
    max_ = samples.max()
    return (samples - min_) / (max_ - min_)


def fundamental(frames: np.ndarray, sr: float) -> np.ndarray:
    fundamentals = np.ndarray((frames.shape[1]))
    for i, f in enumerate(frames.T):
        fundamentals[i] = freq_from_fft(f, sr)
    return fundamentals


def _fundamental(filepath: str, frame_ms: int, sliding_ms: int) -> np.ndarray:
    '''
        Given an audio file, splits into frames and tries to
        guess the fundamental frequency of each one of them.
        The method used is the "most precise" among the easy
        ones. Here's a good explanation of them:
            https://gist.github.com/endolith/255291
        Returns an array with the F0 of each frame.
    '''
    time_series, sr = load(filepath)
    sr_ms = sr / 1000
    time_series = normalize_gain(time_series)
    frames = frame(time_series, frame_length=int(sr_ms*frame_ms), hop_length=int(sr_ms*sliding_ms))
    fundamentals = np.ndarray((frames.shape[1]))
    for i, f in enumerate(frames.T):
        fundamentals[i] = freq_from_fft(f, sr)
    return fundamentals 
