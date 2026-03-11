import numpy as np
from scipy.signal.windows import kaiser

# ------------------------------------------------------------
# Windows
# ------------------------------------------------------------

def _sin_window(N: int) -> np.ndarray:
    n = np.arange(N, dtype=np.float64)
    return np.sin(np.pi / N * (n + 0.5))


def _kbd_window(N: int, a: float) -> np.ndarray:
    """KBD window construction (assignment Section 2.2)."""
    # Kaiser window defined for n=0..N/2 (inclusive) -> length N/2+1
    w = kaiser(N // 2 + 1, beta=np.pi * a).astype(np.float64)
    c = np.cumsum(w)
    c = c / c[-1]
    left = np.sqrt(c[:-1])              
    right = left[::-1]
    return np.concatenate([left, right])   


def _base_windows(win_type: str):
    if win_type == "SIN":
        Wl = _sin_window(2048)
        Ws = _sin_window(256)
    elif win_type == "KBD":
        Wl = _kbd_window(2048, a=6)
        Ws = _kbd_window(256, a=4)
    else:
        raise ValueError("win_type must be 'KBD' or 'SIN'")
    return Wl, Ws


def _window_2048_for_frame(frame_type: str, win_type: str) -> np.ndarray:
    """Return the 2048-sample analysis/synthesis window for OLS/LSS/LPS."""
    Wl, Ws = _base_windows(win_type)

    if frame_type == "OLS":
        return Wl

    if frame_type == "LSS":
        # [ left half long (1024), ones(448), right half short (128), zeros(448) ]
        return np.concatenate([Wl[:1024], np.ones(448), Ws[128:], np.zeros(448)])

    if frame_type == "LPS":
        # [ zeros(448), left half short (128), ones(448), right half long (1024) ]
        return np.concatenate([np.zeros(448), Ws[:128], np.ones(448), Wl[1024:]])

    raise ValueError("For ESH windows are handled per-subframe (256).")


# ------------------------------------------------------------
# MDCT / IMDCT Core
# ------------------------------------------------------------

_MDCT_CACHE = {}

def _mdct_cosine_matrix(N: int) -> np.ndarray:
    """Precompute MDCT cosine basis matrix with caching."""

    if N in _MDCT_CACHE:
        return _MDCT_CACHE[N]
    
    # Precompute time shift n0 = (N/2 + 1) / 2
    n0 = (N / 2 + 1) / 2.0
    
    n = np.arange(N).reshape(N, 1)           
    k = np.arange(N // 2).reshape(1, N // 2) 
    
    # Calculate matrix C_nk = cos( (2*pi/N) * (n + n0) * (k + 1/2) )
    angle = (2.0 * np.pi / N) * (n + n0) * (k + 0.5)
    C = np.cos(angle)
    
    _MDCT_CACHE[N] = C
    return C


def mdct(x: np.ndarray) -> np.ndarray:
    """
    Forward MDCT:
    X_k = 2 * sum_n x[n] * cos(...)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    N = x.shape[0]
    C = _mdct_cosine_matrix(N)
    return 2.0 * (x @ C)


def imdct(X: np.ndarray) -> np.ndarray:
    """
    Inverse MDCT:
    x[n] = (2/N) * sum_k X[k] * cos(...)
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1)
    N2 = X.shape[0]
    N = 2 * N2
    C = _mdct_cosine_matrix(N)
    return (2.0 / N) * (C @ X)



# ------------------------------------------------------------
# Filter bank / iFilter bank
# ------------------------------------------------------------

def filter_bank(frame_T: np.ndarray, frame_type: str, win_type: str):
    """AAC filterbank (Level 1).

    Parameters
    ----------
    frame_T : (2048,2) ndarray
    frame_type : 'OLS'|'LSS'|'ESH'|'LPS'
    win_type : 'KBD'|'SIN'

    Returns
    -------
    frame_F :
      - if frame_type != 'ESH' : (1024,2)
      - if frame_type == 'ESH' : (128,8,2)  (8 short MDCTs per channel)
    """
    frame_T = np.asarray(frame_T, dtype=np.float64)
    if frame_T.shape != (2048, 2):
        raise ValueError("frame_T must have shape (2048,2)")

    Wl, Ws = _base_windows(win_type)

    if frame_type in ("OLS", "LSS", "LPS"):
        w = _window_2048_for_frame(frame_type, win_type)
        out = np.zeros((1024, 2), dtype=np.float64)
        for ch in range(2):
            out[:, ch] = mdct(frame_T[:, ch] * w)
        return out

    if frame_type == "ESH":
        # keep only the middle 1152 samples (drop 448 from each side)
        core = frame_T[448:448+1152, :]              
        out = np.zeros((128, 8, 2), dtype=np.float64)
        for s in range(8):
            start = s * 128
            sub = core[start:start+256, :]             
            for ch in range(2):
                out[:, s, ch] = mdct(sub[:, ch] * Ws)  
        return out

    raise ValueError("frame_type must be one of: OLS, LSS, ESH, LPS")


def i_filter_bank(frame_F, frame_type: str, win_type: str) -> np.ndarray:
    """AAC synthesis filterbank (inverse of filter_bank).

    Returns a *windowed* 2048-sample contribution per frame.
    The caller (i_aac_coder_1) must perform overlap-add with hop=1024.
    """
    Wl, Ws = _base_windows(win_type)

    if frame_type in ("OLS", "LSS", "LPS"):
        frame_F = np.asarray(frame_F, dtype=np.float64)
        if frame_F.shape != (1024, 2):
            raise ValueError("frame_F must have shape (1024,2) for non-ESH")
        w = _window_2048_for_frame(frame_type, win_type)
        out = np.zeros((2048, 2), dtype=np.float64)
        for ch in range(2):
            out[:, ch] = imdct(frame_F[:, ch]) * w
        return out

    if frame_type == "ESH":
        frame_F = np.asarray(frame_F, dtype=np.float64)
        if frame_F.shape != (128, 8, 2):
            raise ValueError("frame_F must have shape (128,8,2) for ESH")
        out = np.zeros((2048, 2), dtype=np.float64)
        # overlap-add the 8 short blocks into the middle region, offset 448
        for s in range(8):
            start = 448 + s * 128
            for ch in range(2):
                sub_t = imdct(frame_F[:, s, ch]) * Ws   
                out[start:start+256, ch] += sub_t
        return out

    raise ValueError("frame_type must be one of: OLS, LSS, ESH, LPS")
