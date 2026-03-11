import numpy as np
from scipy.signal import lfilter, correlate
from scipy.linalg import toeplitz, solve
from scipy.io import loadmat

def compute_tns_lpc(spectrum_norm, lpc_order=4):
    """
    Computes quantized and stability-checked LPC coefficients
    used in Temporal Noise Shaping (TNS).

    Parameters
    ----------
    spectrum_norm : ndarray
        Normalized MDCT spectrum of one frame or subframe.
    lpc_order : int
        LPC prediction order.

    Returns
    -------
    lpc_quantized : ndarray
        Stable quantized LPC coefficients.
    """

    length = len(spectrum_norm)

    # Autocorrelation sequence r(m)
    autocorr = correlate(spectrum_norm, spectrum_norm, mode='full')[length-1:length+lpc_order]

    # Yule-Walker system
    autocorr_matrix = toeplitz(autocorr[:lpc_order])
    autocorr_vector = autocorr[1:lpc_order+1]

    try:
        lpc_coeffs = solve(autocorr_matrix, autocorr_vector)
    except Exception:
        lpc_coeffs = np.zeros(lpc_order)

    # Uniform quantization with step 0.1
    lpc_quantized = np.round(lpc_coeffs / 0.1) * 0.1
    lpc_quantized = np.clip(lpc_quantized, -0.7, 0.8)

    # Stability condition: all roots strictly inside unit circle
    poly_roots = np.roots(np.concatenate(([1.0], -lpc_quantized)))

    if np.any(np.abs(poly_roots) >= 1):
        lpc_quantized = np.zeros(lpc_order)

    return lpc_quantized

def load_band_tables(mat_file_path):
    """
    Loads AAC scalefactor band definitions
    from the standard TableB219.mat file.

    Returns
    -------
    bands_long  : ndarray
        Long window scalefactor bands.
    bands_short : ndarray
        Short window scalefactor bands.
    """

    mat_contents = loadmat(mat_file_path)

    bands_long = mat_contents["B219a"]
    bands_short = mat_contents["B219b"]

    return bands_long, bands_short


def MDCT_normalize(X, bj):
    """
    Docstring for MDCT_normalize
    
    :param X: Description
    :param bj: Description
    """
    N = len(X)
    NB = bj.shape[0]  
    P = np.zeros(NB)
    Sw = np.zeros(N)
    
    for j in range(NB):
        w_low = int(bj[j, 1])
        w_high = int(bj[j, 2])
        
        P[j] = np.sum(X[w_low : w_high + 1]**2)
        Sw[w_low : w_high + 1] = np.sqrt(P[j])
        
    for k in range(N-2, -1, -1):
        Sw[k] = (Sw[k] + Sw[k+1]) / 2
    for k in range(1, N):
        Sw[k] = (Sw[k] + Sw[k-1]) / 2
        
    Xw = X / (Sw + 1e-10)
    return Xw


def tns(frame_F_in, frame_type):
    """
    Applies Temporal Noise Shaping (TNS) in the MDCT domain.

    frame_F_in contains MDCT coefficients before TNS.
    For long frames the expected shape is (1024, 1),
    for ESH frames the expected shape is (128, 8).

    The function estimates LPC coefficients per frame (or per short window),
    applies the forward prediction filter in the frequency domain,
    and returns the filtered spectrum together with the quantized LPC coefficients.
    """
    
    # Load scalefactor band definitions used for MDCT normalization
    bj_long, bj_short = load_band_tables("TableB219.mat")

    if frame_type == "ESH":
        frame_F_out = np.zeros_like(frame_F_in)
        tns_coeffs = np.zeros((4, 8))
        
        for i in range(8):
            X = frame_F_in[:, i]
            Xw = MDCT_normalize(X, bj_short)

            # Estimate LPC coefficients from normalized MDCT spectrum
            a_q = compute_tns_lpc(Xw, lpc_order=4)
            tns_coeffs[:, i] = a_q

            # Apply forward TNS filtering
            frame_F_out[:, i] = lfilter(np.concatenate(([1], -a_q)), [1.0], X)

    else:
        X = frame_F_in.flatten()
        Xw = MDCT_normalize(X, bj_long)

        # Estimate LPC coefficients from normalized MDCT spectrum
        a_q = compute_tns_lpc(Xw, lpc_order=4)
        tns_coeffs = a_q.reshape(4, 1)

        # Apply forward TNS filtering
        X_out = lfilter(np.concatenate(([1], -a_q)), [1.0], X)
        frame_F_out = X_out.reshape(frame_F_in.shape)

    return frame_F_out, tns_coeffs


def i_tns(frame_F_in, frame_type, tns_coeffs):
    """
    Applies the inverse Temporal Noise Shaping (iTNS) filter.

    frame_F_in contains MDCT coefficients after TNS.
    For long frames the expected shape is (1024, 1),
    for ESH frames the expected shape is (128, 8).

    The function reconstructs the original MDCT spectrum
    by applying the inverse prediction filter using
    the previously stored LPC coefficients.
    """

    # Ensure ESH frames are in (128, 8) layout
    if frame_F_in.ndim == 2 and frame_F_in.shape == (8, 128):
        frame_F_in = frame_F_in.T

    if frame_type == "ESH":
        frame_F_out = np.zeros_like(frame_F_in)
        
        for i in range(8):
            X = frame_F_in[:, i]
            a_q = tns_coeffs[:, i]

            # Apply inverse TNS filtering
            frame_F_out[:, i] = lfilter([1.0], np.concatenate(([1], -a_q)), X)

    else:
        X = frame_F_in.flatten()
        a_q = tns_coeffs.flatten()

        # Apply inverse TNS filtering
        X_out = lfilter([1.0], np.concatenate(([1], -a_q)), X)
        frame_F_out = X_out.reshape(frame_F_in.shape)

    return frame_F_out
