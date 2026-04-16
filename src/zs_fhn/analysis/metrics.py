import numpy as np

def compute_amplitude(u_t):
    """
    Compute oscillation amplitude from time series.
    """
    pass


def compute_frequency(u_t, t):
    """
    Compute oscillation frequency.
    """
    pass


def detect_oscillation(u_t, t, transient_ratio=0.5, tol=1e-3):
    """
    Improved oscillation detection.

    Strategy:
        1. Remove transient
        2. Check amplitude
        3. Check periodicity (zero crossings)

    Returns
    -------
    bool
    """

    n = len(u_t)
    start = int(n * transient_ratio)

    u_tail = u_t[start:]

    # ---- amplitude ----
    amp = np.max(u_tail) - np.min(u_tail)

    if amp < tol:
        return False

    # ---- zero crossings ----
    mean = np.mean(u_tail)
    centered = u_tail - mean

    crossings = np.where(np.diff(np.sign(centered)))[0]

    # 至少要有多次穿越才算振荡
    return len(crossings) > 2


def compute_wave_speed(u_xt, x, t):
    """
    Estimate wave propagation speed.
    """
    pass


def compute_pattern_wavelength(u_xt, x):
    """
    Estimate spatial wavelength of patterns.
    """
    pass