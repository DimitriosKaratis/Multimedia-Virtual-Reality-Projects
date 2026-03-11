import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

def compute_fft_spectrum(signal, N, is_esh=False):
    """Return magnitude/phase of an N-point FFT (half-spectrum), with analysis windowing."""
    window = 0.5 - 0.5 * np.cos(np.pi * (np.arange(N) + 0.5) / N)

    if is_esh:
        windowed = signal * window[np.newaxis, :]
        spectrum = np.fft.fft(windowed, axis=1)[:, :N // 2]
    else:
        windowed = signal * window
        spectrum = np.fft.fft(windowed)[:N // 2]

    return np.abs(spectrum), np.angle(spectrum)


def extract_esh_subframes(frame_T):
        """Split the ESH central region into 8 overlapped (256) blocks with hop 128."""
        central_region = frame_T[448:1600]
        return sliding_window_view(central_region, 256)[::128]


def spectral_predictability(r_curr, phi_curr, r_pred, phi_pred):
    """Compute per-bin predictability between current and predicted spectra."""
    num = np.sqrt(
        ((r_curr * np.cos(phi_curr)) - (r_pred * np.cos(phi_pred))) ** 2 +
        ((r_curr * np.sin(phi_curr)) - (r_pred * np.sin(phi_pred))) ** 2
    )
    den = r_curr + np.abs(r_pred) + 1e-10
    return num / den


def band_energy_and_weighted_predictability(r, c, bj):
    """Aggregate FFT-bin energy and energy-weighted predictability into scalefactor bands."""
    num_bands = bj.shape[0]
    leading_shape = r.shape[:-1]

    band_energy = np.zeros((*leading_shape, num_bands))
    band_cw = np.zeros((*leading_shape, num_bands))

    for b in range(num_bands):
        w_low = int(bj[b, 1])
        w_high = int(bj[b, 2])

        r_band = r[..., w_low:w_high + 1]
        c_band = c[..., w_low:w_high + 1]

        band_energy[..., b] = np.sum(r_band ** 2, axis=-1)
        band_cw[..., b] = np.sum(c_band * (r_band ** 2), axis=-1)

    return band_energy, band_cw


def apply_spreading_and_normalization(e, cb_weighted, S):
    """Spread band energies across neighbors and normalize to get cb and en per band."""
    spread_energy = np.dot(e, S)
    spread_cw = np.dot(cb_weighted, S)

    cb = spread_cw / (spread_energy + 1e-10)

    col_sums = np.sum(S, axis=0)
    en = spread_energy / (col_sums + 1e-10)

    return cb, en


def compute_energy_threshold(cb, en, TMN, NMT):
    """Convert tonality and spread energy into a masking-threshold energy per band."""
    tb = -0.299 - 0.43 * np.log(cb + 1e-10)
    tb = np.clip(tb, 0, 1)

    snr_db = tb * TMN + (1 - tb) * NMT
    bc = 10 ** (-snr_db / 10)

    nb = en * bc
    return nb


def compute_noise_partition(bj, nb, N):
    """Apply the absolute threshold in quiet (ATH) as a floor to the masking threshold."""
    epsilon = np.finfo(float).eps
    qsthr = bj[:, 5]
    qsthr_hat = epsilon * (N / 2) * 10 ** (qsthr / 10)

    if nb.ndim > 1:
        npart = np.maximum(nb, qsthr_hat[np.newaxis, :])
    else:
        npart = np.maximum(nb, qsthr_hat)

    return npart


def compute_mdct_thresholds(frame_F, smr, bj):
    """Compute per-band MDCT masking thresholds T(b) from band power and SMR."""
    if frame_F.ndim == 2 and frame_F.shape == (128, 8):
        frame_F = frame_F.T

    num_bands = bj.shape[0]
    leading_shape = frame_F.shape[:-1]
    band_power = np.zeros((*leading_shape, num_bands))

    for b in range(num_bands):
        w_low = int(bj[b, 1])
        w_high = int(bj[b, 2])

        x_band = frame_F[..., w_low:w_high + 1]
        band_power[..., b] = np.sum(x_band ** 2, axis=-1)

    smr_scalar = np.squeeze(smr)
    thresholds = band_power / (smr_scalar + 1e-15)

    return thresholds


def quantizer(X, alpha, magic_number=0.4054):
    """Quantize the input X using a non-uniform companding quantizer based on the alpha parameter."""
    S = np.sign(X) * np.floor( (np.abs(X) * 2**(-alpha/4))**0.75 + magic_number )
    return S


def i_quantizer(S, alpha):
    """Inverse quantizer to reconstruct the quantized values from S using the alpha parameter."""
    X_hat = np.sign(S) * (np.abs(S)**(4/3)) * (2**(alpha/4))
    return X_hat


def optimize_scalefactors(S_total, sfc, G, frame_F, num_bands, bj, range_s, Tb):
    """
    Optimize band-wise quantization step sizes (alpha values) so that
    quantization error energy remains below the masking threshold.

    For each band, alpha is iteratively adjusted to satisfy:
        Quantization Error Energy <= Target Masking Threshold
    """

    SFC_LIMIT = 60          # Maximum allowed scalefactor difference between adjacent bands
    S_LIMIT = 8191          # Maximum allowed quantized coefficient magnitude
    ALPHA_ABS_LIMIT = 250   # Absolute bound on alpha values
    SCALING_FACTOR = 1.0    # Increase for higher compression (lower SNR) -> We leave it at 1, so that 
                            # we follow the instructions of the project.

    for s in range(range_s):

        alpha_init = float(np.atleast_1d(G[s])[0])
        alpha_prev = alpha_init

        for b in range(num_bands):

            w_low, w_high = int(bj[b, 1]), int(bj[b, 2])
            band_coeffs = frame_F[s, w_low:w_high + 1]

            target_threshold = Tb[s, b].item() * SCALING_FACTOR
            if target_threshold <= 1e-12:
                target_threshold = 1e-12

            # Initial alpha for this band
            alpha_band = alpha_init if b == 0 else alpha_prev

            # Initial quantization
            S = quantizer(band_coeffs, alpha_band)
            X_hat = i_quantizer(S, alpha_band)
            error_energy = np.sum((band_coeffs - X_hat) ** 2)

            # Case 1: Error exceeds masking threshold → decrease alpha
            if error_energy > target_threshold:
                while error_energy > target_threshold:

                    alpha_next = alpha_band - 1

                    if b > 0 and abs(alpha_next - alpha_prev) > SFC_LIMIT:
                        break
                    if alpha_next < -ALPHA_ABS_LIMIT:
                        break

                    S_next = quantizer(band_coeffs, alpha_next)
                    X_hat_next = i_quantizer(S_next, alpha_next)
                    error_next = np.sum((band_coeffs - X_hat_next) ** 2)

                    alpha_band = alpha_next
                    error_energy = error_next
                    S = S_next

            # Case 2: Error below threshold -> try increasing alpha for more compression
            else:
                while True:

                    alpha_next = alpha_band + 1

                    if b > 0 and abs(alpha_next - alpha_prev) > SFC_LIMIT:
                        break
                    if alpha_next > ALPHA_ABS_LIMIT:
                        break

                    S_next = quantizer(band_coeffs, alpha_next)

                    # If band fully zeroed, compute error directly
                    if np.all(S_next == 0):
                        X_hat_next = np.zeros_like(band_coeffs)
                        error_next = np.sum(band_coeffs ** 2)

                        if error_next <= target_threshold:
                            alpha_band = alpha_next
                            S = S_next
                        break

                    if np.max(np.abs(S_next)) > S_LIMIT:
                        break

                    X_hat_next = i_quantizer(S_next, alpha_next)
                    error_next = np.sum((band_coeffs - X_hat_next) ** 2)

                    if error_next > target_threshold:
                        break

                    alpha_band = alpha_next
                    S = S_next
                    error_energy = error_next

            # Store quantized symbols
            S_total[s, w_low:w_high + 1] = S

            # Store global gain or scalefactor
            if b == 0:
                G[s] = alpha_band
            else:
                sfc[s, b - 1] = alpha_band - alpha_prev

            alpha_prev = alpha_band

    S_total = np.clip(S_total, -8191, 8191)

    return S_total, sfc, G


def prepare_for_huffman(data, data_type="mdct"):
    data = np.asarray(data).flatten()
    
    if data_type == "mdct":
        data = np.clip(data, -8191, 8191)
        
    elif data_type == "sfc":
        data = np.clip(data, -60, 60)
    
    data_int = np.round(data).astype(int)
    
    return data_int
