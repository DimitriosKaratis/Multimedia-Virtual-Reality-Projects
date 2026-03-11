import numpy as np
import soundfile as sf
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from src.tns import tns, i_tns
from src.ssc import SSC
from src.filterbank import filter_bank, i_filter_bank

def aac_coder_2(filename_in):
    """
    AAC Encoder Level 2: Implementation of SSC, Filterbank, and TNS.
    """
    x, fs = sf.read(filename_in, always_2d=True)
    x = x.astype(np.float64)
    hop, N = 1024, 2048
    
    # Apply padding for initial and final frame overlap-add consistency
    x_padded = np.pad(x, ((hop, N), (0, 0)), mode='constant')
    
    aac_seq_2 = []
    prev_type = "OLS"
    
    # Main encoding loop
    for start in range(0, len(x_padded) - N + 1, hop):
        frame_T = x_padded[start:start + N, :]
        next_start = start + hop
        next_frame_T = x_padded[next_start:next_start + N, :] if (next_start + N) <= len(x_padded) else frame_T
        
        # 1. Sequence Segmentation Control (SSC) & Analysis Filterbank
        f_type = SSC(frame_T, next_frame_T, prev_type)
        frame_F_mdct = filter_bank(frame_T, f_type, "KBD")
        
        # 2. TNS Processing per channel (Left/Right)
        # Left Channel (Chl)
        X_l_in = frame_F_mdct[:, :, 0] if f_type == "ESH" else frame_F_mdct[:, 0].reshape(1024, 1)
        X_l_tns, coeffs_l = tns(X_l_in, f_type)
        
        # Right Channel (Chr)
        X_r_in = frame_F_mdct[:, :, 1] if f_type == "ESH" else frame_F_mdct[:, 1].reshape(1024, 1)
        X_r_tns, coeffs_r = tns(X_r_in, f_type)
        
        # 3. Store processed frame data and TNS coefficients
        entry = {
            "frame_type": f_type,
            "win_type": "KBD",
            "chl": {"tns_coeffs": coeffs_l, "frame_F": X_l_tns},
            "chr": {"tns_coeffs": coeffs_r, "frame_F": X_r_tns}
        }
        aac_seq_2.append(entry)
        prev_type = f_type
        
    return aac_seq_2

def i_aac_coder_2(aac_seq_2, filename_out, target_len=282978):
    """
    AAC Decoder Level 2: Inverse TNS, Synthesis Filterbank, and Overlap-Add.
    """
    hop, N = 1024, 2048
    individual_outputs = []
    
    # Main decoding loop
    for entry in aac_seq_2:
        f_type = entry["frame_type"]
        w_type = entry["win_type"]
        
        # 1. Inverse TNS (iTNS) to reconstruct original MDCT coefficients
        X_l_rec = i_tns(entry["chl"]["frame_F"], f_type, entry["chl"]["tns_coeffs"])
        X_r_rec = i_tns(entry["chr"]["frame_F"], f_type, entry["chr"]["tns_coeffs"])
        
        # 2. Prepare reconstruction for Synthesis Filterbank
        if f_type == "ESH":
            frame_F_inv = np.stack([X_l_rec, X_r_rec], axis=2)
        else:
            frame_F_inv = np.column_stack([X_l_rec.flatten(), X_r_rec.flatten()])
            
        individual_outputs.append(i_filter_bank(frame_F_inv, f_type, w_type))
    
    # 3. Final signal reconstruction via Overlap-Add (OLA)
    y_out = np.zeros((target_len, 2))
    for i in range(target_len):
        hop_idx = i // hop
        sample_in_hop = i % hop
        if hop_idx + 1 < len(individual_outputs):
            y_out[i, :] = individual_outputs[hop_idx][hop + sample_in_hop, :] + \
                          individual_outputs[hop_idx + 1][sample_in_hop, :]
                      
    sf.write(filename_out, y_out, 48000)
    return y_out

def demo_aac_2(filename_in, filename_out):
    """
    Official Level 2 Demo: Evaluate TNS performance and signal integrity.
    """
    print("="*60)
    print("STARTING AAC LEVEL 2 (TNS) INTEGRITY CHECK")
    print("="*60)

    # 1. Encoding & Decoding pipeline
    aac_seq_2 = aac_coder_2(filename_in)
    
    # Read original length to ensure correct alignment
    x_orig, fs = sf.read(filename_in, always_2d=True)
    y_rec = i_aac_coder_2(aac_seq_2, filename_out, target_len=len(x_orig))
    
    # 2. Calculate Fidelity Metrics (SNR should remain high if TNS is invertible)
    N_min = min(len(x_orig), len(y_rec))
    x_ref = x_orig[:N_min, :].astype(np.float64)
    y_rec = y_rec[:N_min, :]
    
    err = x_ref - y_rec
    # SNR should still be >150dB for mathematical transparency
    snr = 10 * np.log10((np.sum(x_ref**2) + 1e-12) / (np.sum(err**2) + 1e-12))
    
    print(f"[*] Level 2 SNR: {snr:.2f} dB")
    print(f"[*] Max Abs Error: {np.max(np.abs(err)):.2e}")

    # Visualization: Inspect spectral whitening effect of TNS
    f_idx = 50
    plt.figure(figsize=(10, 4))
    plt.plot(aac_seq_2[f_idx]["chl"]["frame_F"].flatten(), color='orange', label='After TNS Filter')
    plt.title(f"MDCT Spectrum after TNS Filtering (Frame {f_idx})")
    plt.xlabel("Frequency Bin (k)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("output/mdct_after_tns.png", dpi=300)
    plt.show()
    
    return snr

if __name__ == "__main__":
    wav_in = "LicorDeCalandraca.wav"
    wav_out = "output/out_level_2.wav"

    demo_aac_2(wav_in, wav_out)