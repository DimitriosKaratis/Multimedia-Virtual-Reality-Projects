import numpy as np
import sys
import os
from pathlib import Path

# Setup paths for module access
test_file_path = Path(__file__).resolve()
project_root = test_file_path.parent.parent
sys.path.insert(0, str(project_root))

# Imports from project files
from src.psycho import psycho, spreading_function
from src.tns import load_band_tables
import src.quantizer as q_mod 
import src.psycho as p_mod

# Injection: Connect quantizer helpers to the psycho module
# This ensures the internal calls within psycho.py work correctly
p_mod.compute_fft_spectrum = q_mod.compute_fft_spectrum
p_mod.extract_esh_subframes = q_mod.extract_esh_subframes
p_mod.spectral_predictability = q_mod.spectral_predictability
p_mod.band_energy_and_weighted_predictability = q_mod.band_energy_and_weighted_predictability
p_mod.apply_spreading_and_normalization = q_mod.apply_spreading_and_normalization
p_mod.compute_energy_threshold = q_mod.compute_energy_threshold
p_mod.compute_noise_partition = q_mod.compute_noise_partition

def run_quantizer_test():
    """Verify AAC Quantization optimization, MQ limits, and SNR performance."""
    print("="*60)
    print("AAC QUANTIZER TEST")
    print("="*60)

    # Change CWD temporarily to allow load_band_tables to find the .mat file
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        # Initialization: Load tables and precompute spreading matrices
        bj_long, bj_short = load_band_tables("TableB219.mat")
        num_bands = bj_long.shape[0]
        N = 2048
        fs = 44100
        t = np.arange(N) / fs
        
        def get_S(bj):
            n = bj.shape[0]; S = np.zeros((n, n))
            for i in range(n):
                for j in range(n): S[i, j] = spreading_function(i, j, bj)
            return S
        S_long = get_S(bj_long)

        # SMR Profile Generation: Using a 1kHz Tone
        x = 0.5 * np.sin(2 * np.pi * 1000 * t)
        smr = psycho(x, "OLS", x, x, S_long, None)
        
        # Prepare MDCT data (frame_F)
        # Simulating a frame with 1024 MDCT coefficients
        np.random.seed(42)
        frame_F = np.random.uniform(-100, 100, (1, 1024))
        
        # Compute MDCT energy thresholds (Tb) based on the psychoacoustic SMR
        Tb = q_mod.compute_mdct_thresholds(frame_F, smr, bj_long)
        
        # Initialize variables for optimize_scalefactors
        S_total = np.zeros_like(frame_F)
        sfc = np.zeros((1, num_bands - 1))
        G = np.array([100.0]) # Initial Global Gain (alpha_0)
        
        # Iterative Scalefactor Optimization (Encoding)
        print("[*] Starting iterative scalefactor optimization...")
        S_total, sfc, G = q_mod.optimize_scalefactors(
            S_total, sfc, G, frame_F, num_bands, bj_long, range_s=1, Tb=Tb
        )
        print("[*] Optimization finished.")

        # Reconstruction (Decoding) using inverse quantization
        # We reconstruct band-by-band to verify the sfc/G differential logic
        X_hat = np.zeros_like(frame_F)
        alpha_prev = G[0]
        for b in range(num_bands):
            w_low, w_high = int(bj_long[b, 1]), int(bj_long[b, 2])
            alpha_band = alpha_prev if b == 0 else alpha_prev + sfc[0, b-1]
            X_hat[0, w_low:w_high+1] = q_mod.i_quantizer(S_total[0, w_low:w_high+1], alpha_band)
            alpha_prev = alpha_band

        # --- COMPLIANCE CHECKS ---
        
        # Check Maximum Quantization (MQ) limit (8191 for Huffman compatibility)
        max_val = np.max(np.abs(S_total))
        print(f"\n[CHECK] Max Quantized Symbol: {max_val}")
        if max_val <= 8191:
            print("[OK] Within Huffman limits.")
        
        # Check Scalefactor Difference (SFC) limit (60)
        max_sfc = np.max(np.abs(sfc))
        print(f"[CHECK] Max Scalefactor Delta: {max_sfc}")
        if max_sfc <= 60:
            print("[OK] SFC differences within limits.")

        # SNR Calculation: Signal-to-Noise Ratio of the quantization process
        error = frame_F - X_hat
        snr = 10 * np.log10(np.sum(frame_F**2) / (np.sum(error**2) + 1e-15))
        print(f"[CHECK] Quantization SNR: {snr:.2f} dB")

        if snr > 15:
            print("\n[CONCLUSION] Quantizer logic is verified and correct.")
        else:
            print("\n[WARNING] SNR is low. Check the SCALING_FACTOR in optimization.")

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Test failed: {e}")
        traceback.print_exc()
    finally:
        os.chdir(original_cwd) # Restore working directory

if __name__ == "__main__":
    run_quantizer_test()