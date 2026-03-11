import numpy as np
from scipy.signal import lfilter


def _is_esh_one_channel(next_frame_ch: np.ndarray) -> bool:
    """Attack detection for one channel of next_frame (i+1), length 2048.

    Implements the procedure in the assignment:
      1) High-pass filter:
         H(z) = (0.7548 - 0.7548 z^-1) / (1 - 0.5095 z^-1)
      2) Split into 8 segments of 128 samples, compute s_l^2 (energy estimate)
      3) For l=1..7 compute ds2_l = s_l^2 / ((1/l) * sum_{m=0..l-1} s_m^2)
      4) ESH if exists l with s_l^2 > 1e-3 and ds2_l > 10
    """
    x = np.asarray(next_frame_ch, dtype=np.float64)

    b = np.array([0.7548, -0.7548], dtype=np.float64)
    a = np.array([1.0, -0.5095], dtype=np.float64)
    y = lfilter(b, a, x)

    s2 = np.zeros(8, dtype=np.float64)
    for l in range(8):
        seg = y[l * 128:(l + 1) * 128]
        s2[l] = float(np.sum(seg * seg))

    for l in range(1, 8):
        if s2[l] <= 1e-3:
            continue
        prev_sum = float(np.sum(s2[:l]))
        if prev_sum <= 0:
            continue
        prev_mean = prev_sum / l
        if prev_mean > 0 and (s2[l] / prev_mean) > 10.0:
            return True

    return False


def _combine_channel_types(t0: str, t1: str) -> str:
    """Combine per-channel decisions into a common frame_type using Table 1."""
    table = {
        ("OLS", "OLS"): "OLS",
        ("OLS", "LSS"): "LSS",
        ("OLS", "ESH"): "ESH",
        ("OLS", "LPS"): "LPS",
        ("LSS", "OLS"): "LSS",
        ("LSS", "LSS"): "LSS",
        ("LSS", "ESH"): "ESH",
        ("LSS", "LPS"): "ESH",
        ("ESH", "OLS"): "ESH",
        ("ESH", "LSS"): "ESH",
        ("ESH", "ESH"): "ESH",
        ("ESH", "LPS"): "ESH",
        ("LPS", "OLS"): "LPS",
        ("LPS", "LSS"): "ESH",
        ("LPS", "ESH"): "ESH",
        ("LPS", "LPS"): "LPS",
    }
    try:
        return table[(t0, t1)]
    except KeyError as e:
        raise ValueError(f"Invalid channel frame types: {t0}, {t1}") from e


def _transition(prev_frame_type: str, next_is_esh: bool) -> str:
    """Apply the SSC transition rules for a *single channel* (Section 2.1)."""
    if prev_frame_type == "OLS":
        return "LSS" if next_is_esh else "OLS"
    if prev_frame_type == "ESH":
        return "ESH" if next_is_esh else "LPS"
    if prev_frame_type == "LSS":
        return "ESH"
    if prev_frame_type == "LPS":
        return "OLS"
    raise ValueError(f"Bad prev_frame_type: {prev_frame_type}")


def SSC(frame_T: np.ndarray, next_frame_T: np.ndarray, prev_frame_type: str) -> str:
    """Sequence Segmentation Control (SSC).

    Parameters
    ----------
    frame_T : np.ndarray
        Current frame i in time domain, shape (2048,2).
    next_frame_T : np.ndarray
        Next frame i+1 in time domain, shape (2048,2), used for look-ahead.
    prev_frame_type : str
        Frame type used for frame i-1 ("OLS","LSS","ESH","LPS").

    Returns
    -------
    frame_type : str
        Selected frame type for frame i ("OLS","LSS","ESH","LPS").
    """
    frame_T = np.asarray(frame_T)
    next_frame_T = np.asarray(next_frame_T)
    if frame_T.shape != (2048, 2) or next_frame_T.shape != (2048, 2):
        raise ValueError("frame_T and next_frame_T must have shape (2048,2)")

    # Decide whether (i+1) is ESH per channel
    next_esh_0 = _is_esh_one_channel(next_frame_T[:, 0])
    next_esh_1 = _is_esh_one_channel(next_frame_T[:, 1])

    # Derive candidate type for frame i per channel
    t0 = _transition(prev_frame_type, next_esh_0)
    t1 = _transition(prev_frame_type, next_esh_1)

    # Combine to common type for both channels
    return _combine_channel_types(t0, t1)
