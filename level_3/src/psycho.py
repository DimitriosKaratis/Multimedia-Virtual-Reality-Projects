import numpy as np
from src.quantizer import *
from src.tns import load_band_tables


def spreading_function(i, j, bj):
    """
    Compute the spectral spreading factor from band i to band j.

    This models the psychoacoustic spreading of masking energy
    across neighboring critical bands, according to the AAC
    spreading function formulation.

    Parameters
    ----------
    i : int
        Source band index (band where masking originates).
    j : int
        Target band index (band affected by masking).
    bj : ndarray
        Band table containing band boundaries and center frequencies.

    Returns
    -------
    float
        Spreading coefficient S(i, j) in linear scale.
    """

    if i >= j:
        tmp_x = 3 * (bj[j, 4] - bj[i, 4])
    else:
        tmp_x = 1.5 * (bj[j, 4] - bj[i, 4])

    tmp_z = 8 * min((tmp_x - 0.5)**2 - 2 * (tmp_x - 0.5), 0) 

    tmp_y = 15.811389 + 7.5 * (tmp_x + 0.474) - 17.5 * (1.0 + (tmp_x + 0.474)**2)**0.5

    if tmp_y < -100:
        x = 0
    else:
        x = 10**((tmp_z + tmp_y) / 10)

    return x


def psycho(frame_T, frame_type, frame_T_prev_1, frame_T_prev_2, S_long, S_short):
    """
    Psychoacoustic analysis model.
    
    Computes the Signal-to-Mask Ratio (SMR) per scalefactor band using
    spectral predictability, energy spreading and masking threshold estimation.

    Parameters
    ----------
    frame_T : ndarray
        Current time-domain frame (2048 samples).
    frame_type : str
        Frame type ("OLS", "LSS", "ESH", "LPS").
    frame_T_prev_1 : ndarray
        Previous time-domain frame (2048 samples).
    frame_T_prev_2 : ndarray
        Frame preceding the previous one (2048 samples).
    S_long : ndarray
        Spreading matrix for long frames (69x69).
    S_short : ndarray
        Spreading matrix for short frames (42x42).

    Returns
    -------
    SMR : ndarray
        Signal-to-Mask Ratio per band.
        Shape (69,) for long frames, (8, 42) for ESH.
    """

    N_long = 2048
    N_short = 256

    NMT = 6     # Noise Masking Tone (dB)
    TMN = 18    # Tone Masking Noise (dB)
    
    bj_long, bj_short = load_band_tables("TableB219.mat")

    if frame_type == "ESH":

        # Extract 8 overlapping short windows from the central region
        subs_curr = extract_esh_subframes(frame_T)
        subs_prev_1 = extract_esh_subframes(frame_T_prev_1)

        # Build sequence for spectral prediction:
        # [prev_7, prev_8, curr_1 ... curr_8]
        combined_subs = np.vstack([subs_prev_1[6:8, :], subs_curr])

        # FFT magnitude and phase for prediction
        r_seq, phi_seq = compute_fft_spectrum(combined_subs, N_short, is_esh=True)

        # Current spectra
        r = r_seq[2:, :]
        phi = phi_seq[2:, :]

        # Linear spectral prediction
        r_pred = 2 * r_seq[1:9, :] - r_seq[0:8, :]
        phi_pred = 2 * phi_seq[1:9, :] - phi_seq[0:8, :]

        # Per-bin predictability
        c = spectral_predictability(r, phi, r_pred, phi_pred)

        # Aggregate energy and weighted predictability per band
        e, c_weighted = band_energy_and_weighted_predictability(r, c, bj_short)

        # Apply spreading function and normalize
        cb, en = apply_spreading_and_normalization(e, c_weighted, S_short)

        # Convert to masking threshold energy
        nb = compute_energy_threshold(cb, en, TMN, NMT)

        # Apply absolute threshold floor
        npart = compute_noise_partition(bj_short, nb, N_short)

        SMR = e / npart

    else:
        # Long frame analysis

        r, phi = compute_fft_spectrum(frame_T, N_long, is_esh=False)
        r_p1, phi_p1 = compute_fft_spectrum(frame_T_prev_1, N_long, is_esh=False)
        r_p2, phi_p2 = compute_fft_spectrum(frame_T_prev_2, N_long, is_esh=False)

        # Spectral prediction
        r_pred = 2 * r_p1 - r_p2
        phi_pred = 2 * phi_p1 - phi_p2

        c = spectral_predictability(r, phi, r_pred, phi_pred)

        e, c_weighted = band_energy_and_weighted_predictability(r, c, bj_long)

        cb, en = apply_spreading_and_normalization(e, c_weighted, S_long)

        nb = compute_energy_threshold(cb, en, TMN, NMT)

        npart = compute_noise_partition(bj_long, nb, N_long)

        SMR = e / npart

    return SMR